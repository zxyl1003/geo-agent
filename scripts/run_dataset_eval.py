"""Run non-interactive dataset evaluation and write geolocation outputs to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoagent.core.config import AppConfig, load_app_config
from geoagent.core.logging import logger, setup_logging
from geoagent.eval import DISTANCE_THRESHOLDS, haversine_m
from geoagent.state.task_state import GeoLocalizationState
from geoagent.tools.registry import build_tools
from geoagent.workflows.react_workflow import ReactWorkflow


GRANULARITY_LEVELS = ("continent", "country", "region", "city", "street", "poi", "coordinates")
DEFAULT_QUERY = (
    "Geolocate this image as precisely as possible. If exact coordinates are not supported, "
    "return the most precise supported continent, country, region, city, street, or POI."
)


@dataclass(frozen=True)
class EvalItem:
    dataset: str
    subset: str
    split: str
    item_id: str
    image_path: Path
    image_rel: str = ""
    gt_continent: str = ""
    gt_country: str = ""
    gt_region: str = ""
    gt_city: str = ""
    gt_street: str = ""
    gt_poi: str = ""
    gt_address: str = ""
    gt_latitude: float | None = None
    gt_longitude: float | None = None
    source_url: str = ""


FIELDNAMES = [
    "dataset",
    "subset",
    "split",
    "item_id",
    "image_path",
    "status",
    "error",
    "pred_granularity",
    "pred_location_name",
    "pred_continent",
    "pred_country",
    "pred_region",
    "pred_city",
    "pred_street",
    "pred_poi",
    "pred_coordinates",
    "pred_longitude",
    "pred_latitude",
    "confidence",
    "uncertainty_radius_m",
    "gt_continent",
    "gt_country",
    "gt_region",
    "gt_city",
    "gt_street",
    "gt_poi",
    "gt_address",
    "gt_longitude",
    "gt_latitude",
    "source_url",
    "reasoning",
    "evidence_summary",
    "tool_trace",
    "tool_calls_json",
    "token_usage_json",
    "api_call_count_json",
    "tool_call_count_json",
    "internal_tool_call_count_json",
    "resource_events_json",
    "pred_gt_distance_m",
    "correct_10m",
    "correct_20m",
    "correct_50m",
    "correct_100m",
    "correct_1000m",
    "correct_5000m",
    "correct_25000m",
    "memory_failure_type",
    "memory_success_pattern",
    "memory_diagnosis",
    "memory_attribution_confidence",
]


def _bool_cell(value: bool | None) -> str:
    """Encode a per-image correctness verdict as a CSV cell."""

    if value is None:
        return ""
    return "1" if value else "0"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def csv_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def parse_csv_list(value: str) -> list[str]:
    items = [item.strip().lower() for item in value.split(",")]
    return [item for item in items if item]


def format_duration(seconds: float) -> str:
    """Format elapsed seconds: seconds under a minute, minutes+seconds otherwise."""

    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(round(seconds % 60))
    if secs == 60:
        minutes += 1
        secs = 0
    return f"{minutes}m {secs:02d}s"


def _safe_path_name(value: str) -> str:
    """Sanitize a string for use as a filesystem path component."""

    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value) or "item"


# ---------------------------------------------------------------------------
# Unified standard metadata loader
# ---------------------------------------------------------------------------

STANDARD_DATASET_MANIFESTS: dict[str, tuple[str, str, str, str]] = {
    "geoexp7k-learning": ("GeoExp7k", "learning_metadata.csv", "", "GeoExp7k"),
    "geoexp7k-test": ("GeoExp7k", "test_metadata.csv", "", "GeoExp7k"),
    "im2gps3k": ("im2gps3ktest", "im2gps3ktest.csv", "images", ""),
    "imageobench-dataset2": (
        "IMAGEOBench",
        "metadata_standard.csv",
        "IMAGEO-Bench-datasets",
        "",
    ),
}

ONLINE_DATASETS = {
    "geoexp7k-test",
    "im2gps3k",
    "imageobench-dataset2",
}
DEFAULT_DATASETS = ONLINE_DATASETS


def iter_standard(
    dataset_root: Path,
    dataset_dir: str,
    metadata_filename: str = "metadata_standard.csv",
    image_subdir: str = "",
    dataset_label_override: str = "",
) -> Iterable[EvalItem]:
    """Load EvalItems from a standard metadata CSV file."""
    csv_path = dataset_root / dataset_dir / metadata_filename
    if not csv_path.exists():
        return
    image_base = csv_path.parent / image_subdir
    with open(csv_path, encoding="utf-8-sig") as f:
        for index, row in enumerate(csv.DictReader(f)):
            image_rel = clean_text(row.get("image_path"))
            if not image_rel:
                continue
            image_path = image_base / image_rel
            dataset_label = dataset_label_override or clean_text(row.get("dataset")) or dataset_dir
            yield EvalItem(
                dataset=dataset_label,
                subset=dataset_dir,
                split=clean_text(row.get("experiment_split")),
                item_id=clean_text(row.get("sample_id")) or f"{index:06d}_{Path(image_rel).stem}",
                image_path=image_path,
                image_rel=(Path(dataset_dir) / image_subdir / image_rel).as_posix(),
                gt_continent=clean_text(row.get("continent")),
                gt_country=clean_text(row.get("country")),
                gt_region=clean_text(row.get("region")),
                gt_city=clean_text(row.get("city")),
                gt_street=clean_text(row.get("street")),
                gt_poi=clean_text(row.get("poi")),
                gt_address=clean_text(row.get("address")),
                gt_latitude=parse_float(row.get("latitude")),
                gt_longitude=parse_float(row.get("longitude")),
            )


def discover_items(args: argparse.Namespace) -> list[EvalItem]:
    dataset_root = Path(args.dataset_root)
    selected = set(parse_csv_list(args.datasets))
    if not selected or "all" in selected:
        selected = set(DEFAULT_DATASETS)

    unknown = selected - STANDARD_DATASET_MANIFESTS.keys()
    if unknown:
        available = sorted(STANDARD_DATASET_MANIFESTS)
        raise ValueError(
            f"Unsupported dataset(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(available)}"
        )

    items: list[EvalItem] = []
    for name in sorted(selected):
        dataset_dir, metadata_filename, image_subdir, dataset_label_override = (
            STANDARD_DATASET_MANIFESTS[name]
        )
        count_before = len(items)
        loaded = list(
            iter_standard(
                dataset_root,
                dataset_dir,
                metadata_filename,
                image_subdir,
                dataset_label_override,
            )
        )
        if name == "imageobench-dataset2":
            loaded = [item for item in loaded if item.dataset == "IMAGEOBench-d2"]
        items.extend(loaded)
        logger.info(f"  {name}: {len(items) - count_before} items")

    if args.offset:
        items = items[args.offset :]
    if args.limit is not None:
        items = items[: args.limit]
    return items


def validate_online_datasets(value: str) -> None:
    selected = set(parse_csv_list(value))
    if not selected or selected == {"all"}:
        return
    unsupported = selected - ONLINE_DATASETS
    if unsupported:
        raise ValueError(
            "The online inference entry point supports only "
            "geoexp7k-test, im2gps3k, imageobench-dataset2, or all. "
            f"Unsupported: {', '.join(sorted(unsupported))}"
        )


def configure_memory(app_config: AppConfig, args: argparse.Namespace) -> None:
    memory_mode = str(getattr(args, "memory_mode", "off") or "off").strip().lower()
    if memory_mode == "config":
        return
    memory_config = app_config.system.setdefault("memory", {})
    memory_config["enabled"] = memory_mode != "off"
    if memory_mode != "off":
        memory_dir = getattr(args, "memory_dir", "")
        if not memory_dir:
            raise ValueError("--memory-dir is required when memory is enabled.")
        memory_config["mode"] = memory_mode
        memory_config["storage_dir"] = memory_dir


def configure_logging(app_config: AppConfig, debug: bool, colorize: bool) -> None:
    logging_config = app_config.system.get("logging", {})
    file_root = Path(logging_config.get("file_root", "logs"))
    if not file_root.is_absolute():
        file_root = ROOT / file_root
    setup_logging(
        app_config.log_level,
        logging_config.get("format"),
        debug=debug,
        colorize=colorize,
        file_enabled=logging_config.get("file_enabled", True),
        file_root=file_root,
        file_name=logging_config.get("file_name", "log.txt"),
    )


def build_workflow(args: argparse.Namespace):
    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    configure_memory(app_config, args)
    if args.debug:
        app_config.log_level = "DEBUG"
    else:
        # Progress is emitted via logger.info; keep the level at INFO so it shows
        # while staying below DEBUG so model streaming and the verbose Agent/Tool/Model
        # traces (gated on _DEBUG_LOGGING in core/logging.py) stay off regardless of
        # the LOG_LEVEL set in .env.
        app_config.log_level = "INFO"
    configure_logging(app_config, debug=args.debug, colorize=not args.no_color)

    tools = build_tools(app_config)

    workflow = ReactWorkflow(app_config=app_config, tools=tools, workflow_config=app_config.workflows["react"])
    if bool((app_config.system.get("memory") or {}).get("enabled", False)):
        # Chroma bootstrap is not safe when the first workflow runs start concurrently.
        workflow.agents = workflow._default_agents()
        workflow._memory_manager()
    return workflow


def evaluate_correctness(row: dict[str, Any], item: EvalItem) -> None:
    """Compute WGS84 coordinate distance and threshold correctness."""

    pred_lat = parse_float(row.get("pred_latitude"))
    pred_lon = parse_float(row.get("pred_longitude"))
    gt_lat = item.gt_latitude
    gt_lon = item.gt_longitude
    if pred_lat is not None and pred_lon is not None and gt_lat is not None and gt_lon is not None:
        distance = haversine_m(pred_lat, pred_lon, gt_lat, gt_lon)
        row["pred_gt_distance_m"] = f"{distance:.2f}"
        for threshold in DISTANCE_THRESHOLDS:
            row[f"correct_{threshold}m"] = _bool_cell(distance <= threshold)


def base_row(item: EvalItem) -> dict[str, Any]:
    return {
        "dataset": item.dataset,
        "subset": item.subset,
        "split": item.split,
        "item_id": item.item_id,
        "image_path": str(item.image_path),
        "status": "",
        "error": "",
        "pred_granularity": "",
        "pred_location_name": "",
        "pred_continent": "",
        "pred_country": "",
        "pred_region": "",
        "pred_city": "",
        "pred_street": "",
        "pred_poi": "",
        "pred_coordinates": "",
        "pred_longitude": "",
        "pred_latitude": "",
        "confidence": "",
        "uncertainty_radius_m": "",
        "gt_continent": item.gt_continent,
        "gt_country": item.gt_country,
        "gt_region": item.gt_region,
        "gt_city": item.gt_city,
        "gt_street": item.gt_street,
        "gt_poi": item.gt_poi,
        "gt_address": item.gt_address,
        "gt_longitude": item.gt_longitude if item.gt_longitude is not None else "",
        "gt_latitude": item.gt_latitude if item.gt_latitude is not None else "",
        "source_url": item.source_url,
        "reasoning": "",
        "evidence_summary": "",
        "tool_trace": "",
        "tool_calls_json": "",
        "token_usage_json": "",
        "api_call_count_json": "",
        "tool_call_count_json": "",
        "resource_events_json": "",
        "pred_gt_distance_m": "",
        "correct_10m": "",
        "correct_20m": "",
        "correct_50m": "",
        "correct_100m": "",
        "correct_1000m": "",
        "correct_5000m": "",
        "correct_25000m": "",
    }


def task_metadata_from_eval_item(item: EvalItem) -> dict[str, Any]:
    return {
        "dataset": item.dataset,
        "subset": item.subset,
        "split": item.split,
        "item_id": item.item_id,
        "feedback_type": "ground_truth",
        "ground_truth": {
            "continent": item.gt_continent,
            "country": item.gt_country,
            "region": item.gt_region,
            "city": item.gt_city,
            "street": item.gt_street,
            "poi": item.gt_poi,
            "address": item.gt_address,
            "latitude": item.gt_latitude,
            "longitude": item.gt_longitude,
        },
    }


def fill_prediction_columns(row: dict[str, Any], state: GeoLocalizationState) -> None:
    answer = state.final_answer
    if answer is None:
        row["status"] = state.status
        row["confidence"] = state.uncertainty.confidence
        row["uncertainty_radius_m"] = (
            state.uncertainty.uncertainty_radius_m
            if state.uncertainty.uncertainty_radius_m is not None
            else ""
        )
        row["tool_calls_json"] = csv_json([call.model_dump() for call in state.tool_calls])
        fill_resource_columns(row, state)
        return

    granularity = answer.granularity or "unknown"
    location_name = answer.location_name or ""
    row["status"] = state.status
    row["pred_granularity"] = granularity
    row["pred_location_name"] = location_name
    row["pred_country"] = answer.country or ""
    row["pred_region"] = answer.region or ""
    row["pred_city"] = answer.city or ""
    row["pred_longitude"] = answer.lon if answer.lon is not None else ""
    row["pred_latitude"] = answer.lat if answer.lat is not None else ""
    row["confidence"] = answer.confidence
    row["uncertainty_radius_m"] = answer.uncertainty_radius_m if answer.uncertainty_radius_m is not None else ""
    row["reasoning"] = csv_json(answer.reasoning)
    row["evidence_summary"] = csv_json(answer.evidence_summary)
    row["tool_trace"] = csv_json(answer.tool_trace)
    row["tool_calls_json"] = csv_json([call.model_dump() for call in state.tool_calls])
    fill_resource_columns(row, state)

    column = f"pred_{granularity}"
    if column in row and granularity != "coordinates":
        row[column] = location_name
    elif granularity == "coordinates":
        row["pred_coordinates"] = location_name

    if granularity == "country" and not row["pred_country"]:
        row["pred_country"] = location_name
    if granularity == "region" and not row["pred_region"]:
        row["pred_region"] = location_name
    if granularity == "city" and not row["pred_city"]:
        row["pred_city"] = location_name
    if granularity == "continent" and not row["pred_continent"]:
        row["pred_continent"] = location_name


def fill_resource_columns(row: dict[str, Any], state: GeoLocalizationState) -> None:
    row["token_usage_json"] = csv_json(state.token_usage)
    row["api_call_count_json"] = csv_json(state.api_call_count)
    row["tool_call_count_json"] = csv_json(state.tool_call_count)
    row["internal_tool_call_count_json"] = csv_json(state.metadata.get("internal_tool_call_count") or {})
    row["resource_events_json"] = csv_json(state.resource_events)
    _fill_memory_attribution_columns(row, state)


def _fill_memory_attribution_columns(row: dict[str, Any], state: GeoLocalizationState) -> None:
    """Export the reflection attribution recorded by the memory manager."""
    attribution = state.metadata.get("memory_attribution") or {}
    row["memory_failure_type"] = clean_text(attribution.get("failure_type"))
    row["memory_success_pattern"] = clean_text(attribution.get("success_pattern"))
    row["memory_diagnosis"] = clean_text(attribution.get("rationale"))
    row["memory_attribution_confidence"] = csv_json(attribution.get("confidence"))


def write_state_trace(trace_dir: Path | None, item: EvalItem, state: GeoLocalizationState | None, error: str = "") -> None:
    if trace_dir is None:
        return
    dataset_dir = trace_dir / _safe_path_name(item.dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    image_name = _safe_path_name(item.image_path.stem)
    path = dataset_dir / f"{image_name}.json"
    payload: dict[str, Any] = {"error": error}
    if state is not None:
        payload["state"] = state.model_dump()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_existing_rows(output_path: Path) -> list[dict[str, str]]:
    """Read completed rows from a prior output CSV for --resume.

    Rows with status=error are dropped so those items are re-run (e.g.
    transient rate-limit or parse failures). Their episodes were never
    recorded by the memory system (the workflow aborts before the memory
    update), so re-running them does not double-learn.
    """
    if not output_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with output_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if clean_text(row.get("status")) == "error":
                continue
            rows.append(row)
    return rows


def existing_keys(output_path: Path) -> set[tuple[str, str, str]]:
    return {
        (clean_text(row.get("dataset")), clean_text(row.get("subset")), clean_text(row.get("item_id")))
        for row in load_existing_rows(output_path)
    }


def run_item(
    item: EvalItem,
    workflow: ReactWorkflow,
    query: str,
    trace_dir: Path | None,
    index: int,
    total: int,
) -> dict[str, Any]:
    """Run a single evaluation item and return the filled CSV row dict."""

    row = base_row(item)
    if not item.image_path.exists():
        logger.warning(f"[{index}/{total}] missing image {item.image_rel}")
        row["status"] = "missing_image"
        row["error"] = f"Image file not found: {item.image_path}"
        evaluate_correctness(row, item)
        return row

    logger.info(f"[{index}/{total}] running {item.image_rel}")
    state: GeoLocalizationState | None = None
    start = time.monotonic()
    try:
        state = workflow.run(
            input_image_path=str(item.image_path),
            user_query=query,
            task_metadata=task_metadata_from_eval_item(item),
        )
        fill_prediction_columns(row, state)
        write_state_trace(trace_dir, item, state)
    except Exception as exc:  # noqa: BLE001 - per-item failures should not abort an evaluation run.
        duration = time.monotonic() - start
        row["status"] = "error"
        row["error"] = str(exc)
        write_state_trace(trace_dir, item, state, error=str(exc))
        logger.error(f"[{index}/{total}] error {item.image_rel} ({format_duration(duration)}): {exc}")
        row["_exception"] = exc  # signal to caller for --fail-fast
        evaluate_correctness(row, item)
        return row

    duration = time.monotonic() - start
    evaluate_correctness(row, item)
    logger.info(f"[{index}/{total}] done {item.image_rel} ({format_duration(duration)})")
    return row


class ParallelEvalRunner:
    """Coordinate parallel evaluation of dataset items using a thread pool."""

    def __init__(
        self,
        workflow: ReactWorkflow,
        query: str,
        trace_dir: Path | None,
        total: int,
        fail_fast: bool = False,
        write_missing: bool = False,
    ) -> None:
        self.workflow = workflow
        self.query = query
        self.trace_dir = trace_dir
        self.total = total
        self.fail_fast = fail_fast
        self.write_missing = write_missing
        self._cancelled = False

    def run(
        self,
        items: list[EvalItem],
        workers: int,
        writer: csv.DictWriter,
        file: Any,
    ) -> None:
        # Buffered rows keyed by original sequence index.
        # Writers flush in order so CSV rows preserve item ordering.
        ordered_rows: dict[int, dict[str, Any]] = {}
        ordered_lock = threading.Lock()
        next_write = [0]  # mutable counter for ordered writes
        write_lock = threading.Lock()

        def _write_rows_in_order() -> None:
            """Flush consecutive completed rows in original item order."""
            with write_lock:
                while next_write[0] in ordered_rows:
                    row = ordered_rows.pop(next_write[0])
                    skip = row.pop("_skip_write", False)
                    row.pop("_exception", None)
                    if not skip:
                        writer.writerow(row)
                        file.flush()
                    next_write[0] += 1

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[Any, int] = {}
            for seq, item in enumerate(items):
                index = seq + 1
                future = executor.submit(
                    self._run_with_index,
                    item,
                    index,
                )
                futures[future] = seq

            for future in as_completed(futures):
                if self._cancelled:
                    break

                seq = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = base_row(items[seq])
                    row["status"] = "error"
                    row["error"] = str(exc)

                exc = row.pop("_exception", None)

                with ordered_lock:
                    ordered_rows[seq] = row

                # Flush whatever is now writable
                _write_rows_in_order()

                if exc and self.fail_fast:
                    self._cancelled = True
                    raise exc

                if self._cancelled:
                    break

        # Flush any remaining rows after all futures complete
        _write_rows_in_order()

    def _run_with_index(self, item: EvalItem, index: int) -> dict[str, Any]:
        if self._cancelled:
            row = base_row(item)
            row["status"] = "skipped"
            row["error"] = "Cancelled due to --fail-fast."
            return row

        row = run_item(item, self.workflow, self.query, self.trace_dir, index, self.total)

        # Missing-image rows are only written when --write-missing is on
        if row["status"] == "missing_image" and not self.write_missing:
            row["_skip_write"] = True

        return row


def _distance_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Compute coordinate-level distance stats and threshold accuracy."""

    distances = [parse_float(row.get("pred_gt_distance_m")) for row in rows]
    distances = [d for d in distances if d is not None]
    attempted = len(distances)
    threshold_accuracy = {}
    for threshold in DISTANCE_THRESHOLDS:
        col = f"correct_{threshold}m"
        correct = sum(1 for row in rows if clean_text(row.get(col)) == "1")
        threshold_accuracy[f"{threshold}m"] = {
            "correct": correct,
            "attempted": attempted,
            "accuracy": correct / attempted if attempted else 0.0,
        }
    stats: dict[str, Any] = {
        "attempted": attempted,
        "coverage": attempted / len(rows) if rows else 0.0,
        "threshold_accuracy": threshold_accuracy,
    }
    if distances:
        sorted_d = sorted(distances)
        stats["mean_m"] = sum(distances) / attempted
        stats["median_m"] = sorted_d[attempted // 2]
        stats["p90_m"] = sorted_d[min(int(attempted * 0.9), attempted - 1)]
        stats["max_m"] = sorted_d[-1]
    return stats


def _summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "coordinates": _distance_stats(rows),
        "granularity_calibration": _granularity_calibration(rows),
    }


def _granularity_calibration(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Cross-tabulate self-reported granularity vs actual accuracy."""
    calibration: dict[str, dict[str, int]] = {}
    for row in rows:
        granularity = clean_text(row.get("pred_granularity")) or "unknown"
        bucket = calibration.setdefault(granularity, {"count": 0, "within_1km": 0, "within_25km": 0})
        bucket["count"] += 1
        if clean_text(row.get("correct_1000m")) == "1":
            bucket["within_1km"] += 1
        if clean_text(row.get("correct_25000m")) == "1":
            bucket["within_25km"] += 1
    result: dict[str, Any] = {}
    for granularity, bucket in sorted(calibration.items()):
        n = bucket["count"]
        result[granularity] = {
            "count": n,
            "within_1km": bucket["within_1km"] / n if n else 0.0,
            "within_25km": bucket["within_25km"] / n if n else 0.0,
        }
    return result


def write_summary(output_path: Path) -> None:
    """Read the finished CSV, compute aggregate accuracy, write summary.json and print."""

    if not output_path.exists():
        return
    with output_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return

    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(clean_text(row.get("dataset")) or "unknown", []).append(row)

    summary: dict[str, Any] = {
        "output_csv": str(output_path),
        "overall": _summarize_rows(rows),
        "by_dataset": {name: _summarize_rows(group) for name, group in by_dataset.items()},
    }
    summary_path = output_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    _print_summary(summary)


def _print_summary(summary: dict[str, Any]) -> None:
    overall = summary["overall"]

    lines: list[str] = ["", "=" * 72]
    lines.append("Evaluation summary (coordinate-only)")
    lines.append("=" * 72)

    coords = overall["coordinates"]
    lines.append("-" * 72)
    lines.append(f"{'coordinates':<14}attempted={coords['attempted']}  coverage={coords['coverage'] * 100:.1f}%")
    for stat_name in ("mean_m", "median_m", "p90_m", "max_m"):
        if stat_name in coords:
            lines.append(f"{'  ' + stat_name:<14}{coords[stat_name]:.2f} m")
    lines.append("-" * 72)
    lines.append(f"{'threshold':<14}{'Accuracy':<56}")
    for threshold, stats in coords["threshold_accuracy"].items():
        acc = stats["accuracy"] * 100
        lines.append(f"{threshold:<14}{acc:6.2f}%  ({stats['correct']}/{stats['attempted']})")

    calibration = overall.get("granularity_calibration") or {}
    if calibration:
        lines.append("-" * 72)
        lines.append("Granularity calibration (self-reported granularity vs coordinate accuracy)")
        lines.append(f"{'granularity':<14}{'count':>6}  {'<=1km%':>9}  {'<=25km%':>9}")
        for granularity, bucket in calibration.items():
            lines.append(
                f"{granularity:<14}{bucket['count']:>6}"
                f"  {bucket['within_1km'] * 100:>8.1f}%"
                f"  {bucket['within_25km'] * 100:>8.1f}%"
            )

    for name, group in summary.get("by_dataset", {}).items():
        lines.append("-" * 72)
        lines.append(f"[{name}]  (n={group['total']})")
        for threshold, stats in group["coordinates"]["threshold_accuracy"].items():
            acc = stats["accuracy"] * 100
            lines.append(f"  coord@{threshold}: {acc:6.2f}%  ({stats['correct']}/{stats['attempted']})")
    lines.append("=" * 72)
    lines.append(f"Summary written to: {Path(summary['output_csv']).parent / 'summary.json'}")
    logger.info("\n".join(lines))


def run_eval(args: argparse.Namespace) -> None:
    items = discover_items(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None

    seen: set[tuple[str, str, str]] = set()
    if args.resume and output_path.exists():
        existing_rows = load_existing_rows(output_path)
        seen = {
            (clean_text(row.get("dataset")), clean_text(row.get("subset")), clean_text(row.get("item_id")))
            for row in existing_rows
        }
        # Rewrite the CSV without error rows so failed items (e.g. transient
        # rate-limit errors) are re-run and append cleanly without duplicates.
        with output_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)
        logger.info(
            f"Resume: keeping {len(existing_rows)} completed rows; "
            "previously errored items will be re-run."
        )
        mode = "a"
        write_header = False
    else:
        mode = "w"
        write_header = True

    workflow = None
    if not args.dry_run:
        workflow = build_workflow(args)
    logger.info(f"Discovered {len(items)} items. Output: {output_path}")
    if args.dry_run:
        missing = sum(1 for item in items if not item.image_path.exists())
        logger.info(f"Dry run only. Missing images: {missing}")
        return

    with output_path.open(mode, encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        workers = getattr(args, "workers", 1)
        run_start = time.monotonic()

        if workers <= 1:
            # Sequential path
            for index, item in enumerate(items, start=1):
                key = (item.dataset, item.subset, item.item_id)
                if key in seen:
                    logger.info(f"[{index}/{len(items)}] skip existing {item.image_rel}")
                    continue

                row = run_item(item, workflow, args.query, trace_dir, index, len(items))

                if row["status"] == "missing_image":
                    if args.write_missing:
                        writer.writerow(row)
                        file.flush()
                    continue

                exc = row.pop("_exception", None)
                writer.writerow(row)
                file.flush()
                if exc and args.fail_fast:
                    raise exc

        else:
            # Parallel path
            runner = ParallelEvalRunner(
                workflow=workflow,
                query=args.query,
                trace_dir=trace_dir,
                total=len(items),
                fail_fast=args.fail_fast,
                write_missing=args.write_missing,
            )

            # Pre-filter already-seen items
            runnable_items: list[EvalItem] = []
            for item in items:
                key = (item.dataset, item.subset, item.item_id)
                if key in seen:
                    logger.info(f"skip existing {item.image_rel}")
                    continue
                runnable_items.append(item)

            logger.info(f"Running {len(runnable_items)} items with {workers} workers.")

            runner.run(runnable_items, workers, writer, file)

        total_duration = time.monotonic() - run_start
        logger.info(f"Total: {format_duration(total_duration)} ({len(items)} items)")

    write_summary(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online GeoAgent inference on supported test datasets.")
    parser.add_argument("--dataset-root", default=str(ROOT / "datasets"), help="Root directory containing downloaded datasets.")
    parser.add_argument(
        "--datasets",
        default="all",
        help=(
            "Comma-separated list: geoexp7k-test,im2gps3k,"
            "imageobench-dataset2,all."
        ),
    )
    parser.add_argument("--output", default=str(ROOT / "outputs" / "results" / "dataset_eval_results.csv"), help="Output CSV path.")
    parser.add_argument("--trace-dir", default=str(ROOT / "outputs" / "results" / "traces"), help="Optional directory for full per-item state JSON traces.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="User query passed to the workflow for every image.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to run after offset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many discovered items before running.")
    parser.add_argument("--resume", action="store_true", help="Append to output CSV and skip existing dataset/subset/item_id rows.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and model streaming.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in logs.")
    parser.add_argument("--dry-run", action="store_true", help="Only discover items and check image paths; do not call models/tools.")
    parser.add_argument("--write-missing", action="store_true", help="Write missing-image metadata rows to the CSV instead of skipping them.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first per-item error.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers. Default 1 (sequential). Use 4-8 for network-bound speedup.")
    parser.add_argument(
        "--experience-mode",
        "--memory-mode",
        dest="memory_mode",
        choices=["off", "retrieve_only"],
        default="off",
        help="Experience-library mode. Use retrieve_only for frozen-library evaluation.",
    )
    parser.add_argument(
        "--experience-dir",
        "--memory-dir",
        dest="memory_dir",
        default="",
        help="Directory containing the SQLite/Chroma experience library.",
    )
    args = parser.parse_args()
    try:
        validate_online_datasets(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))
    run_eval(args)


if __name__ == "__main__":
    main()
