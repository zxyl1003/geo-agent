"""Annotate geolocation datasets with image-text and POI attributes via Batch API.

This script is intentionally separate from ``run_batch_vlm_eval.py``: it does
not predict locations or calculate geolocation accuracy. It applies the
``image_filter.md`` schema used to build GeoExp7k and writes one annotation row
per source image for later dataset-level analysis.

Stages: ``prepare`` (offline, no key) -> ``submit`` -> ``status`` -> ``wait`` ->
``collect``. ``run`` performs all five stages.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoagent.batch.io import (
    BatchInputError,
    JsonlChunkWriter,
    build_chat_request,
    encode_jsonl_line,
    load_manifest,
    make_custom_id,
    save_manifest,
)
from geoagent.batch.provider import create_batch_provider
from geoagent.batch.settings import BatchSettings
from geoagent.core.config import load_app_config
from geoagent.core.json_utils import extract_json_payload
from geoagent.core.logging import logger, setup_logging
from geoagent.core.schemas import utc_now

_EVAL_SPEC = importlib.util.spec_from_file_location(
    "_run_dataset_eval_for_annotation", ROOT / "scripts" / "run_dataset_eval.py"
)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:
    raise RuntimeError("Cannot load run_dataset_eval.py for shared dataset discovery.")
_eval = importlib.util.module_from_spec(_EVAL_SPEC)
sys.modules[_EVAL_SPEC.name] = _eval
_EVAL_SPEC.loader.exec_module(_eval)

PROMPT_FILES = {
    "en": ROOT / "configs" / "prompts" / "image_filter.md",
    "zh": ROOT / "configs" / "prompts" / "image_filter_zh.md",
}
MANIFEST_VERSION = 1
TASK_TYPE = "image_annotation"
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

ANNOTATION_FIELDS = [
    "has_scene_text",
    "has_named_anchor",
    "anchor_types",
    "visible_text",
    "searchable_text",
    "text_legibility",
    "searchability",
    "specificity",
    "language_or_script",
    "observed_entities",
    "entity_count",
    "multi_poi",
    "generic_chain",
    "text_scene_binding",
    "image_quality",
    "quality_issues",
    "recommended_stratum",
    "filter_pass",
    "filter_reason",
    "screening_confidence",
]

FIELDNAMES = [
    "dataset",
    "subset",
    "split",
    "sample_id",
    "image_path",
    "continent",
    "country",
    "region",
    "city",
    "street",
    "poi",
    "address",
    "latitude",
    "longitude",
    "coordinate_system",
    *ANNOTATION_FIELDS,
    "status",
    "error",
    "validation_errors",
    "batch_provider",
    "batch_model",
    "batch_id",
    "custom_id",
    "request_id",
    "provider_status_code",
    "raw_model_output",
]

_ARRAY_FIELDS = {
    "anchor_types",
    "visible_text",
    "searchable_text",
    "language_or_script",
    "observed_entities",
    "quality_issues",
}
_BOOL_FIELDS = {
    "has_scene_text",
    "has_named_anchor",
    "multi_poi",
    "generic_chain",
    "filter_pass",
}
_ENUM_FIELDS = {
    "text_legibility": {"clear", "partial", "unreadable", "none"},
    "searchability": {"high", "medium", "low", "none"},
    "specificity": {"unique", "ambiguous", "generic", "none"},
    "text_scene_binding": {"clear", "ambiguous", "none"},
    "image_quality": {"good", "usable", "poor"},
    "recommended_stratum": {
        "specific_named_anchor",
        "branch_or_generic_name",
        "multi_poi",
        "partial_or_non_latin_text",
        "road_address_or_phone",
        "reject",
    },
}


def build_prompt(language: str = "en") -> str:
    return PROMPT_FILES[language].read_text(encoding="utf-8")


def _load_settings() -> BatchSettings:
    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    return BatchSettings.from_env(app_config.env)


def _iter_dataset(dataset_root: Path, name: str) -> Iterable[Any]:
    if name == "im2gps3k":
        yield from _eval.iter_standard(
            dataset_root,
            "im2gps3ktest",
            "im2gps3ktest.csv",
            "images",
        )
        return
    if name == "imageobench-dataset2":
        for item in _eval.iter_standard(
            dataset_root,
            "IMAGEOBench",
            "metadata_standard.csv",
            "IMAGEO-Bench-datasets",
        ):
            if item.dataset == "IMAGEOBench-d2":
                yield item
        return
    raise ValueError(
        f"Unsupported annotation dataset {name!r}; use im2gps3k or imageobench-dataset2."
    )


def discover_items(args: argparse.Namespace) -> list[Any]:
    selected = _eval.parse_csv_list(args.datasets)
    if not selected or "all" in selected:
        selected = ["im2gps3k", "imageobench-dataset2"]

    items: list[Any] = []
    dataset_root = Path(args.dataset_root)
    for name in selected:
        before = len(items)
        items.extend(_iter_dataset(dataset_root, name))
        logger.info(f"  {name}: {len(items) - before} items")
    if args.offset:
        items = items[args.offset :]
    if args.limit is not None:
        items = items[: args.limit]
    return items


def _item_record(item: Any, index: int, custom_id: str) -> dict[str, Any]:
    return {
        "index": index,
        "item_id": item.item_id,
        "dataset": item.dataset,
        "subset": item.subset,
        "split": item.split,
        "image_path": str(item.image_path),
        "image_rel": item.image_rel,
        "custom_id": custom_id,
        "chunk_index": None,
        "status": "prepared",
        "error": None,
        "gt_continent": item.gt_continent,
        "gt_country": item.gt_country,
        "gt_region": item.gt_region,
        "gt_city": item.gt_city,
        "gt_street": item.gt_street,
        "gt_poi": item.gt_poi,
        "gt_address": item.gt_address,
        "gt_latitude": item.gt_latitude,
        "gt_longitude": item.gt_longitude,
    }


def _save_items(run_dir: Path, items: list[dict[str, Any]]) -> Path:
    path = run_dir / "items.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    return path


def _load_items(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "items.jsonl"
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                items.append(json.loads(line))
    return items


def cmd_prepare(args: argparse.Namespace) -> None:
    settings = _load_settings()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    items = discover_items(args)
    prompt_file = PROMPT_FILES[args.prompt_language]
    prompt = build_prompt(args.prompt_language)

    writer = JsonlChunkWriter(
        run_dir / "inputs",
        max_requests=settings.max_requests_per_file,
        max_bytes=settings.max_file_bytes,
    )
    records: list[dict[str, Any]] = []
    rejected = 0
    for index, item in enumerate(items):
        custom_id = make_custom_id(index, TASK_TYPE, item.dataset, item.item_id)
        record = _item_record(item, index, custom_id)
        try:
            request = build_chat_request(
                custom_id=custom_id,
                image_path=item.image_path,
                image_relative_path=item.image_rel,
                prompt=prompt,
                settings=settings,
            )
            record["chunk_index"] = writer.add(
                encode_jsonl_line(request, settings.max_line_bytes)
            )
        except BatchInputError as exc:
            record["status"] = "input_error"
            record["error"] = str(exc)
            rejected += 1
        records.append(record)

    chunks = writer.close()
    items_path = _save_items(run_dir, records)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "task_type": TASK_TYPE,
        "created_at": utc_now().isoformat(),
        "updated_at": utc_now().isoformat(),
        "run_dir": str(run_dir),
        "datasets": args.datasets,
        "prompt_language": args.prompt_language,
        "prompt_file": str(prompt_file),
        "prompt_snapshot": prompt,
        "items_path": str(items_path),
        "requested_item_count": len(items),
        "prepared_request_count": len(records) - rejected,
        "rejected_item_count": rejected,
        "settings": settings.public_dict(),
        "chunks": chunks,
        "status": "prepared",
    }
    save_manifest(run_dir / "manifest.json", manifest)
    logger.info(
        f"Prepared {len(records) - rejected} annotation requests in {len(chunks)} chunk(s); "
        f"{rejected} input_error item(s). Manifest at {run_dir / 'manifest.json'}"
    )


def _provider() -> Any:
    return create_batch_provider(_load_settings())


def cmd_submit(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    provider = _provider()
    submitted = 0
    for chunk in manifest["chunks"]:
        if chunk.get("input_file_id") and chunk.get("batch_id"):
            continue
        input_path = Path(chunk["input_path"])
        if not input_path.exists() or input_path.stat().st_size == 0:
            chunk["status"] = "input_error"
            chunk["error"] = "Empty or missing input file."
            continue
        uploaded = provider.upload_input_file(input_path)
        chunk["input_file_id"] = uploaded.get("id")
        chunk["status"] = "uploaded"
        batch = provider.create_batch(
            chunk["input_file_id"],
            {"task_type": TASK_TYPE, "chunk": str(chunk["index"])},
        )
        chunk["batch_id"] = batch.get("id")
        chunk["status"] = "submitted"
        chunk["remote"] = batch
        submitted += 1
        save_manifest(run_dir / "manifest.json", manifest)
        logger.info(f"Submitted chunk {chunk['index']}: batch {chunk['batch_id']}")
    logger.info(f"Submitted {submitted} chunk(s).")


def _refresh_chunk(provider: Any, chunk: dict[str, Any]) -> None:
    remote = provider.retrieve_batch(chunk["batch_id"])
    chunk["remote_status"] = remote.get("status")
    chunk["output_file_id"] = remote.get("output_file_id")
    chunk["error_file_id"] = remote.get("error_file_id")
    chunk["remote"] = remote


def cmd_status(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    provider = _provider()
    for chunk in manifest["chunks"]:
        if not chunk.get("batch_id"):
            continue
        _refresh_chunk(provider, chunk)
        logger.info(
            f"chunk {chunk['index']}: {chunk['remote_status']} "
            f"(output={chunk['output_file_id']}, error={chunk['error_file_id']})"
        )
    save_manifest(run_dir / "manifest.json", manifest)


def cmd_wait(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    provider = _provider()
    settings = _load_settings()
    while True:
        pending: list[int] = []
        for chunk in manifest["chunks"]:
            if not chunk.get("batch_id"):
                continue
            if chunk.get("remote_status") not in _TERMINAL_STATUSES:
                _refresh_chunk(provider, chunk)
            if chunk.get("remote_status") not in _TERMINAL_STATUSES:
                pending.append(chunk["index"])
        save_manifest(run_dir / "manifest.json", manifest)
        if not pending:
            logger.info("All annotation batches finished.")
            return
        logger.info(f"Waiting on {len(pending)} chunk(s): {pending}")
        time.sleep(settings.poll_interval_seconds)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content or "")


def _parse_output_line(line: str) -> dict[str, Any] | None:
    if not line.strip():
        return None
    payload = json.loads(line)
    response = payload.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    content = _message_text(message)
    try:
        output = extract_json_payload(content)
    except Exception:
        output = {}
    return {
        "custom_id": payload.get("custom_id", ""),
        "request_id": body.get("id", ""),
        "status_code": response.get("status_code"),
        "raw": content,
        "output": output if isinstance(output, dict) else {},
    }


def validate_annotation(output: dict[str, Any]) -> list[str]:
    errors = [f"missing field: {field}" for field in ANNOTATION_FIELDS if field not in output]
    for field in _ARRAY_FIELDS:
        if field in output and not isinstance(output[field], list):
            errors.append(f"{field} must be an array")
    for field in _BOOL_FIELDS:
        if field in output and not isinstance(output[field], bool):
            errors.append(f"{field} must be boolean")
    for field, allowed in _ENUM_FIELDS.items():
        if field in output and output[field] not in allowed:
            errors.append(f"invalid {field}: {output[field]!r}")
    entities = output.get("observed_entities")
    count = output.get("entity_count")
    if isinstance(entities, list) and (not isinstance(count, int) or isinstance(count, bool)):
        errors.append("entity_count must be an integer")
    elif isinstance(entities, list) and count != len(entities):
        errors.append("entity_count does not match observed_entities")
    if isinstance(output.get("filter_pass"), bool):
        is_reject = output.get("recommended_stratum") == "reject"
        if output["filter_pass"] == is_reject:
            errors.append("recommended_stratum and filter_pass are inconsistent")
    return errors


def _csv_value(field: str, value: Any) -> Any:
    if field in _ARRAY_FIELDS:
        return json.dumps(value if isinstance(value, list) else [], ensure_ascii=False)
    if field in _BOOL_FIELDS:
        return "1" if value is True else "0" if value is False else ""
    if value is None:
        return ""
    return value


def _base_row(record: dict[str, Any]) -> dict[str, Any]:
    has_coordinates = record.get("gt_latitude") is not None and record.get("gt_longitude") is not None
    return {
        "dataset": record["dataset"],
        "subset": record["subset"],
        "split": record["split"],
        "sample_id": record["item_id"],
        "image_path": record["image_rel"],
        "continent": record["gt_continent"],
        "country": record["gt_country"],
        "region": record["gt_region"],
        "city": record["gt_city"],
        "street": record["gt_street"],
        "poi": record["gt_poi"],
        "address": record["gt_address"],
        "latitude": record["gt_latitude"],
        "longitude": record["gt_longitude"],
        "coordinate_system": "WGS84" if has_coordinates else "",
        "custom_id": record["custom_id"],
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def _write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    completed = [row for row in rows if row.get("status") == "completed"]
    total = len(rows)
    status_counts = Counter(str(row.get("status") or "") for row in rows)

    def true_count(field: str) -> int:
        return sum(row.get(field) == "1" for row in completed)

    summary = {
        "output_csv": str(output_path),
        "total_rows": total,
        "valid_annotations": len(completed),
        "status_counts": dict(sorted(status_counts.items())),
        "rates_among_valid": {
            field: {
                "count": true_count(field),
                "rate": true_count(field) / len(completed) if completed else 0.0,
            }
            for field in ("has_scene_text", "has_named_anchor", "multi_poi", "filter_pass")
        },
        "recommended_stratum_counts": dict(
            sorted(Counter(str(row.get("recommended_stratum") or "") for row in completed).items())
        ),
        "text_legibility_counts": dict(
            sorted(Counter(str(row.get("text_legibility") or "") for row in completed).items())
        ),
        "image_quality_counts": dict(
            sorted(Counter(str(row.get("image_quality") or "") for row in completed).items())
        ),
    }
    summary_path = output_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"Valid annotations: {len(completed)}/{total}; "
        f"named anchors: {true_count('has_named_anchor')}/{len(completed)}; "
        f"filter pass: {true_count('filter_pass')}/{len(completed)}"
    )
    logger.info(f"Summary written to {summary_path}")


def cmd_collect(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    provider = _provider()
    records = _load_items(run_dir)
    remote_dir = run_dir / "remote"
    remote_dir.mkdir(parents=True, exist_ok=True)
    parsed_by_custom_id: dict[str, dict[str, Any]] = {}
    batch_by_chunk: dict[int, str] = {}

    for chunk in manifest["chunks"]:
        batch_by_chunk[chunk["index"]] = chunk.get("batch_id", "")
        error_file_id = chunk.get("error_file_id")
        if error_file_id:
            error_path = remote_dir / f"error_{chunk['index']:04d}.jsonl"
            if not error_path.exists():
                error_path.write_bytes(provider.download_file(error_file_id))
            chunk["error_path"] = str(error_path)
        output_file_id = chunk.get("output_file_id")
        if not output_file_id:
            continue
        output_path = remote_dir / f"output_{chunk['index']:04d}.jsonl"
        if not output_path.exists():
            output_path.write_bytes(provider.download_file(output_file_id))
        chunk["output_path"] = str(output_path)
        with output_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                parsed = _parse_output_line(line)
                if parsed is not None:
                    parsed_by_custom_id[parsed["custom_id"]] = parsed
    save_manifest(run_dir / "manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item["index"]):
        row = _base_row(record)
        row["batch_provider"] = manifest["settings"].get("provider", "")
        row["batch_model"] = manifest["settings"].get("model", "")
        row["batch_id"] = batch_by_chunk.get(record.get("chunk_index"), "")
        parsed = parsed_by_custom_id.get(record["custom_id"])
        if record["status"] == "input_error":
            row["status"] = "input_error"
            row["error"] = record.get("error") or ""
        elif parsed is None:
            row["status"] = "missing_output"
            row["error"] = "No batch output was returned for this item."
        else:
            row["request_id"] = parsed["request_id"]
            row["provider_status_code"] = _eval.clean_text(parsed["status_code"])
            row["raw_model_output"] = parsed["raw"]
            output = parsed["output"]
            errors = validate_annotation(output) if output else ["empty or unparseable JSON output"]
            for field in ANNOTATION_FIELDS:
                row[field] = _csv_value(field, output.get(field))
            row["validation_errors"] = json.dumps(errors, ensure_ascii=False)
            if errors:
                row["status"] = "invalid_annotation"
                row["error"] = errors[0]
            else:
                row["status"] = "completed"
        rows.append(row)

    output_path = Path(args.output or run_dir / "annotations.csv")
    _write_csv(rows, output_path)
    _write_summary(rows, output_path)
    logger.info(f"Collected {len(rows)} annotation row(s) into {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=["prepare", "submit", "status", "wait", "collect", "run"],
        default="run",
    )
    parser.add_argument("--dataset-root", default=str(ROOT / "datasets"))
    parser.add_argument(
        "--datasets",
        default="im2gps3k,imageobench-dataset2",
        help="Comma-separated: im2gps3k,imageobench-dataset2 (or all).",
    )
    parser.add_argument("--prompt-language", choices=["en", "zh"], default="en")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="", help="Default: RUN_DIR/annotations.csv")
    args = parser.parse_args()

    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    setup_logging(app_config.log_level, colorize=True, file_enabled=False)
    if args.action == "run":
        for action in ("prepare", "submit", "wait", "collect"):
            globals()[f"cmd_{action}"](args)
    else:
        globals()[f"cmd_{args.action}"](args)


if __name__ == "__main__":
    main()
