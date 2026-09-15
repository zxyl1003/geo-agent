"""Run the Direct-VLM batch baseline evaluator.

Evaluates a vision-language model directly from the image and a fixed prompt
(no GeoAgent workflow, tools, agents, or external memory). Dataset discovery,
ground-truth fields, distance thresholds, and the main CSV columns are shared
with ``run_dataset_eval.py``; provider/model/job IDs and the raw model output are
added for auditability.

Stages: ``prepare`` (offline, no key) -> ``submit`` -> ``status`` -> ``wait`` ->
``collect``. ``run`` performs all five. Every remote file/batch ID, prompt
snapshot, and non-secret setting is persisted in ``manifest.json`` so a long
asynchronous job can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

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

# Reuse the dataset-discovery and evaluation logic from the workflow evaluator.
_EVAL_SPEC = importlib.util.spec_from_file_location(
    "_run_dataset_eval_shared", ROOT / "scripts" / "run_dataset_eval.py"
)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:
    raise RuntimeError("Cannot load run_dataset_eval.py for shared logic.")
_eval = importlib.util.module_from_spec(_EVAL_SPEC)
sys.modules[_EVAL_SPEC.name] = _eval
_EVAL_SPEC.loader.exec_module(_eval)

PROMPT_FILE = ROOT / "configs" / "prompts" / "direct_vlm_geolocation.md"
MANIFEST_VERSION = 1
TASK_TYPE = "direct_vlm_eval"

# Main CSV columns from run_dataset_eval, plus batch audit columns.
FIELDNAMES = [
    *_eval.FIELDNAMES,
    "batch_provider",
    "batch_model",
    "batch_id",
    "custom_id",
    "request_id",
    "provider_status_code",
    "raw_model_output",
]

# OpenAI-compatible batch statuses that mean "finished".
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def build_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


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
        "line_index": None,
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


def _load_items(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "items.jsonl"
    if not path.exists():
        return {}
    by_custom_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            by_custom_id[item["custom_id"]] = item
    return by_custom_id


def _load_settings(args: argparse.Namespace) -> BatchSettings:
    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    return BatchSettings.from_env(app_config.env)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    settings = _load_settings(args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    items = list(
        _eval.discover_items(
            argparse.Namespace(
                datasets=args.datasets,
                dataset_root=args.dataset_root,
                offset=args.offset,
                limit=args.limit,
            )
        )
    )
    prompt = build_prompt()

    writer = JsonlChunkWriter(
        run_dir / "inputs",
        max_requests=settings.max_requests_per_file,
        max_bytes=settings.max_file_bytes,
    )

    records: list[dict[str, Any]] = []
    rejected = 0
    for index, item in enumerate(items):
        custom_id = make_custom_id(index, item.dataset, item.subset, item.item_id)
        record = _item_record(item, index, custom_id)
        try:
            request = build_chat_request(
                custom_id=custom_id,
                image_path=item.image_path,
                image_relative_path=item.image_rel,
                prompt=prompt,
                settings=settings,
            )
            line = encode_jsonl_line(request, settings.max_line_bytes)
            chunk_index = writer.add(line)
            record["chunk_index"] = chunk_index
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
        "prompt_file": str(PROMPT_FILE),
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
        f"Prepared {len(records) - rejected} requests in {len(chunks)} chunk(s); "
        f"{rejected} input_error item(s). Manifest at {run_dir / 'manifest.json'}"
    )


# ---------------------------------------------------------------------------
# submit / status / wait / collect
# ---------------------------------------------------------------------------

def _provider_from_manifest(manifest: dict[str, Any], args: argparse.Namespace) -> Any:
    settings = _load_settings(args)
    return create_batch_provider(settings)


def cmd_submit(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    provider = _provider_from_manifest(manifest, args)

    submitted = 0
    for chunk in manifest["chunks"]:
        if chunk.get("input_file_id") and chunk.get("batch_id"):
            continue
        if chunk["status"] == "input_error":
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


def cmd_status(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    provider = _provider_from_manifest(manifest, args)

    for chunk in manifest["chunks"]:
        if not chunk.get("batch_id"):
            continue
        remote = provider.retrieve_batch(chunk["batch_id"])
        chunk["remote_status"] = remote.get("status")
        chunk["output_file_id"] = remote.get("output_file_id")
        chunk["error_file_id"] = remote.get("error_file_id")
        logger.info(
            f"chunk {chunk['index']}: {chunk['remote_status']} "
            f"(output={chunk['output_file_id']}, error={chunk['error_file_id']})"
        )
    save_manifest(run_dir / "manifest.json", manifest)


def cmd_wait(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    provider = _provider_from_manifest(manifest, args)
    settings = _load_settings(args)

    while True:
        pending = []
        for chunk in manifest["chunks"]:
            if not chunk.get("batch_id"):
                continue
            status = chunk.get("remote_status")
            if status not in _TERMINAL_STATUSES:
                remote = provider.retrieve_batch(chunk["batch_id"])
                status = remote.get("status")
                chunk["remote_status"] = status
                chunk["output_file_id"] = remote.get("output_file_id")
                chunk["error_file_id"] = remote.get("error_file_id")
            if status not in _TERMINAL_STATUSES:
                pending.append(chunk["index"])
        save_manifest(run_dir / "manifest.json", manifest)
        if not pending:
            logger.info("All batches finished.")
            break
        logger.info(f"Waiting on {len(pending)} chunk(s): {pending}")
        time.sleep(settings.poll_interval_seconds)


def _parse_output_line(line: str) -> dict[str, Any] | None:
    if not line.strip():
        return None
    payload = json.loads(line)
    custom_id = payload.get("custom_id", "")
    response = payload.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    raw_status = response.get("status_code")
    parsed: dict[str, Any] = {}
    try:
        parsed = extract_json_payload(str(content))
    except Exception:
        parsed = {}
    return {
        "custom_id": custom_id,
        "request_id": body.get("id", ""),
        "status_code": raw_status,
        "raw": str(content),
        "output": parsed if isinstance(parsed, dict) else {},
    }


def fill_prediction(row: dict[str, Any], output: dict[str, Any]) -> None:
    """Fill prediction columns from a Direct-VLM JSON output."""
    row["pred_country"] = _eval.clean_text(output.get("country"))
    row["pred_city"] = _eval.clean_text(output.get("city"))
    row["pred_latitude"] = _eval.clean_text(output.get("latitude"))
    row["pred_longitude"] = _eval.clean_text(output.get("longitude"))


def write_results_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def cmd_collect(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    provider = _provider_from_manifest(manifest, args)
    items_by_custom_id = _load_items(run_dir)
    remote_dir = run_dir / "remote"
    remote_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for chunk in manifest["chunks"]:
        if not chunk.get("batch_id"):
            continue
        output_file_id = chunk.get("output_file_id")
        if not output_file_id:
            chunk.setdefault("errors", []).append("No output file for batch.")
            continue
        output_path = remote_dir / f"output_{chunk['index']:04d}.jsonl"
        if not output_path.exists():
            output_path.write_bytes(provider.download_file(output_file_id))
        chunk["output_path"] = str(output_path)
        save_manifest(run_dir / "manifest.json", manifest)

        with output_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                parsed = _parse_output_line(line)
                if parsed is None:
                    continue
                record = items_by_custom_id.get(parsed["custom_id"])
                if record is None:
                    logger.warning(f"Unknown custom_id {parsed['custom_id']} in output.")
                    continue
                item = _eval.EvalItem(
                    dataset=record["dataset"],
                    subset=record["subset"],
                    split=record["split"],
                    item_id=record["item_id"],
                    image_path=Path(record["image_path"]),
                    image_rel=record["image_rel"],
                    gt_continent=record["gt_continent"],
                    gt_country=record["gt_country"],
                    gt_region=record["gt_region"],
                    gt_city=record["gt_city"],
                    gt_street=record["gt_street"],
                    gt_poi=record["gt_poi"],
                    gt_address=record["gt_address"],
                    gt_latitude=record["gt_latitude"],
                    gt_longitude=record["gt_longitude"],
                )
                row = _eval.base_row(item)
                if record["status"] == "input_error":
                    row["status"] = "input_error"
                    row["error"] = record.get("error") or ""
                elif parsed.get("output"):
                    row["status"] = "completed"
                    fill_prediction(row, parsed["output"])
                else:
                    row["status"] = "error"
                    row["error"] = f"Empty or unparseable model output (status={parsed['status_code']})."
                row["batch_provider"] = manifest["settings"].get("provider", "")
                row["batch_model"] = manifest["settings"].get("model", "")
                row["batch_id"] = chunk.get("batch_id", "")
                row["custom_id"] = parsed["custom_id"]
                row["request_id"] = parsed["request_id"]
                row["provider_status_code"] = _eval.clean_text(parsed["status_code"])
                row["raw_model_output"] = parsed["raw"]
                _eval.evaluate_correctness(row, item)
                rows.append(row)

    output_csv = Path(args.output or run_dir / "results.csv")
    write_results_csv(rows, output_csv, FIELDNAMES)
    logger.info(f"Collected {len(rows)} row(s) into {output_csv}")
    _eval.write_summary(output_csv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        default="all",
        help=(
            "Comma-separated list (geoexp7k-learning,geoexp7k-test,"
            "im2gps3k,imageobench-dataset2,all)."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--run-dir", required=True, help="Directory holding manifest.json and staged files.")
    parser.add_argument("--output", default="", help="Collect output CSV path (default: run-dir/results.csv).")
    args = parser.parse_args()

    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    setup_logging(
        app_config.log_level,
        colorize=True,
        file_enabled=False,
    )

    if args.action == "run":
        for action in ("prepare", "submit", "wait", "collect"):
            args.action = action
            globals()[f"cmd_{action}"](args)
    else:
        globals()[f"cmd_{args.action}"](args)


if __name__ == "__main__":
    main()
