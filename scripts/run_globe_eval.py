"""Evaluate GLOBE text predictions through a vLLM OpenAI-compatible API.

Inference uses the official GLOBE chain-of-thought prompt and stores the raw
country/city response. A separate, resumable stage geocodes those place names
with OpenCage and reports coordinate accuracy at the configured thresholds.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoagent.batch.io import BatchInputError
from geoagent.core.config import load_app_config


_SHARED_SPEC = importlib.util.spec_from_file_location(
    "_run_geoagent_eval_shared_globe", ROOT / "scripts" / "run_geoagent_eval.py"
)
if _SHARED_SPEC is None or _SHARED_SPEC.loader is None:
    raise RuntimeError("Cannot load run_geoagent_eval.py for shared evaluation logic.")
_shared = importlib.util.module_from_spec(_SHARED_SPEC)
sys.modules[_SHARED_SPEC.name] = _shared
_SHARED_SPEC.loader.exec_module(_shared)

_eval = _shared._eval
logger = _shared.logger


OFFICIAL_SYSTEM_PROMPT = "You are a helpful assistant."
OFFICIAL_USER_PROMPT = """You are participating in a geolocation challenge. Based on the provided image:
1. Carefully analyze the image for clues about its location (architecture, signage, vegetation, terrain, etc.)
2. Think step-by-step about what country, and city this is likely to be in and why
Your final answer include these two lines somewhere in your response:
country: [country name]
city: [city name]

You MUST output the thinking process in <think> </think> and give answer in <answer> </answer> tags."""

DEFAULT_RESULTS_DIR = ROOT / "outputs" / "results"
DISTANCE_LEVELS_M = {
    "fine_500m": 500,
    "street_1km": 1_000,
    "local_2km": 2_000,
    "district_10km": 10_000,
    "city_25km": 25_000,
    "region_200km": 200_000,
    "country_750km": 750_000,
    "continent_2500km": 2_500_000,
}
DISTANCE_COLUMNS = [f"correct_{threshold}m" for threshold in DISTANCE_LEVELS_M.values()]

PREDICTION_EXTRA_FIELDS = [
    "final_answer",
    "globe_thinking",
    "parse_error",
    "eval_model_url",
    "eval_model_name",
    "response_model",
    "request_id",
    "finish_reason",
    "http_attempts",
    "latency_seconds",
    "raw_model_output",
]
PREDICTION_FIELDS = list(dict.fromkeys([*_eval.FIELDNAMES, *PREDICTION_EXTRA_FIELDS]))
RESULT_EXTRA_FIELDS = [
    *PREDICTION_EXTRA_FIELDS,
    "geocode_query",
    "opencage_status",
    "opencage_error",
    "opencage_formatted",
    "opencage_confidence",
    "opencage_components_json",
    "opencage_bounds_json",
    "opencage_rate_remaining",
    "opencage_rate_reset",
]
RESULT_FIELDS = list(dict.fromkeys([*_eval.FIELDNAMES, *DISTANCE_COLUMNS, *RESULT_EXTRA_FIELDS]))

_TAG_PATTERN = r"<{tag}\b[^>]*>(.*?)</{tag}>"
_LOCATION_PATTERNS = {
    "country": re.compile(r"(?im)^\s*(?:[-*]\s*)?\**\s*country\s*\**\s*:\s*([^\r\n<]+)"),
    "city": re.compile(r"(?im)^\s*(?:[-*]\s*)?\**\s*city\s*\**\s*:\s*([^\r\n<]+)"),
}


def _tag_content(content: str, tag: str) -> str:
    match = re.search(_TAG_PATTERN.format(tag=re.escape(tag)), content, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_globe_output(content: str) -> dict[str, str]:
    """Parse GLOBE's official country/city answer format."""

    answer = _tag_content(content, "answer") or content.strip()
    values: dict[str, str] = {}
    for name, pattern in _LOCATION_PATTERNS.items():
        match = pattern.search(answer)
        values[name] = match.group(1).strip().strip("`* ") if match else ""
    if not values["country"] and not values["city"]:
        raise ValueError("GLOBE output does not contain country or city fields.")
    return {
        "final_answer": answer,
        "country": values["country"],
        "city": values["city"],
        "thinking": _tag_content(content, "think"),
    }


def build_geocode_query(prediction: dict[str, Any]) -> str:
    """Build the city-country query used by the paper's evaluation protocol."""

    parts = [
        _eval.clean_text(prediction.get("pred_city")),
        _eval.clean_text(prediction.get("pred_country")),
    ]
    unique: list[str] = []
    for part in parts:
        if part and part.casefold() not in {value.casefold() for value in unique}:
            unique.append(part)
    return ", ".join(unique)


def build_inference_payload(item: Any, args: argparse.Namespace, model: str) -> dict[str, Any]:
    image_url = _shared.image_to_data_url(
        item.image_path,
        max_edge=args.image_max_edge,
        jpeg_quality=args.image_jpeg_quality,
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": OFFICIAL_USER_PROMPT},
                ],
            },
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
    }


def run_inference_item(
    item: Any,
    *,
    index: int,
    total: int,
    args: argparse.Namespace,
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    row = _eval.base_row(item)
    row.update({field: "" for field in PREDICTION_EXTRA_FIELDS})
    row["eval_model_url"] = base_url
    row["eval_model_name"] = model
    if not item.image_path.exists():
        row["status"] = "missing_image"
        row["error"] = f"Image file not found: {item.image_path}"
        return row

    start = time.monotonic()
    try:
        data, attempts = _shared.post_with_retry(
            url=_shared.chat_completion_url(base_url),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload=build_inference_payload(item, args, model),
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
        )
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("GLOBE response contains no choices.")
        choice = choices[0]
        content = _shared._message_text(choice.get("message") or {})
        if not content:
            raise RuntimeError("GLOBE returned empty content.")

        row["status"] = "completed"
        row["raw_model_output"] = content
        row["token_usage_json"] = _eval.csv_json(data.get("usage") or {})
        row["api_call_count_json"] = _eval.csv_json({"vlm": attempts})
        row["response_model"] = _eval.clean_text(data.get("model"))
        row["request_id"] = _eval.clean_text(data.get("id"))
        row["finish_reason"] = _eval.clean_text(choice.get("finish_reason"))
        row["http_attempts"] = attempts

        try:
            parsed = parse_globe_output(content)
            row["final_answer"] = parsed["final_answer"]
            row["pred_country"] = parsed["country"]
            row["pred_city"] = parsed["city"]
            row["pred_location_name"] = parsed["city"] or parsed["country"]
            row["pred_granularity"] = "city" if parsed["city"] else "country"
            row["globe_thinking"] = parsed["thinking"]
            row["reasoning"] = parsed["thinking"]
        except ValueError as exc:
            row["parse_error"] = str(exc)
    except (BatchInputError, OSError, RuntimeError, ValueError) as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    finally:
        row["latency_seconds"] = f"{time.monotonic() - start:.3f}"

    logger.info(f"[{index}/{total}] {row['status']} {item.image_rel}")
    return row


def run_inference(args: argparse.Namespace, output_dir: Path) -> None:
    items = _shared.discover_items(args)
    missing = sum(not item.image_path.exists() for item in items)
    logger.info(f"Discovered {len(items)} items; missing images: {missing}")
    if args.dry_run:
        return

    base_url = args.base_url or _eval.clean_text(os.getenv("EVAL_MODEL_URL"))
    model = args.model or _eval.clean_text(os.getenv("EVAL_MODEL_NAME"))
    api_key = _eval.clean_text(os.getenv("EVAL_API_KEY")) or "EMPTY"
    if not base_url:
        raise RuntimeError("--base-url or EVAL_MODEL_URL is required for inference.")
    if not model:
        raise RuntimeError("--model or EVAL_MODEL_NAME is required for inference.")

    prediction_path = output_dir / "predictions.csv"
    existing_rows = _shared._load_resumable_rows(prediction_path) if args.resume else []
    seen = {_shared._row_key(row) for row in existing_rows}
    runnable = [item for item in items if _shared._item_key(item) not in seen]

    if args.resume and prediction_path.exists():
        _shared._rewrite_rows(prediction_path, PREDICTION_FIELDS, existing_rows)
        mode = "a"
    else:
        mode = "w"

    logger.info(
        f"Running {len(runnable)} GLOBE inference request(s) with {args.workers} workers; "
        f"model={model}; endpoint={_shared.chat_completion_url(base_url)}"
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open(mode, encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_inference_item,
                    item,
                    index=index,
                    total=len(runnable),
                    args=args,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                ): item
                for index, item in enumerate(runnable, start=1)
            }
            for future in as_completed(futures):
                row = future.result()
                writer.writerow({field: row.get(field, "") for field in PREDICTION_FIELDS})
                stream.flush()
                if args.fail_fast and row["status"] == "error":
                    raise RuntimeError(row["error"])

    logger.info(f"Text predictions written to {prediction_path}")


def default_output_dir(dataset: str) -> Path:
    names = {
        "geoexp7k-learning": "geoexp7k_learning",
        "geoexp7k-test": "geoexp7k_test",
        "imageobench-dataset2": "imageobench_dataset2",
        "im2gps3k": "im2gps3k",
    }
    try:
        name = names[dataset.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc
    return DEFAULT_RESULTS_DIR / f"vllm_globe_{name}"


def _configure_shared_geocoding() -> None:
    _shared.DISTANCE_LEVELS_M = DISTANCE_LEVELS_M
    _shared.DISTANCE_COLUMNS = DISTANCE_COLUMNS
    _shared.RESULT_FIELDS = RESULT_FIELDS
    _shared.build_geocode_query = build_geocode_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["infer", "geocode"], required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["im2gps3k", "imageobench-dataset2", "geoexp7k-test", "geoexp7k-learning"],
    )
    parser.add_argument("--dataset-root", default=str(ROOT / "datasets"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-url", default="", help="Override EVAL_MODEL_URL for inference.")
    parser.add_argument("--model", default="", help="Override EVAL_MODEL_NAME for inference.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--frequency-penalty", type=float, default=0.7)
    parser.add_argument("--presence-penalty", type=float, default=0.7)
    parser.add_argument("--image-max-edge", type=int, default=0)
    parser.add_argument("--image-jpeg-quality", type=int, default=90)
    parser.add_argument("--geocode-limit", type=int, default=2400)
    parser.add_argument("--geocode-interval", type=float, default=1.1)
    parser.add_argument("--geocode-timeout", type=float, default=30.0)
    parser.add_argument("--geocode-max-retries", type=int, default=3)
    parser.add_argument(
        "--geocode-cache",
        default=str(ROOT / "outputs" / "cache" / "geocoding" / "opencage_cache.sqlite3"),
    )
    parser.add_argument("--quota-reserve", type=int, default=10)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.max_retries < 0 or args.geocode_max_retries < 0:
        parser.error("retry counts cannot be negative")
    if args.geocode_limit < 0:
        parser.error("--geocode-limit cannot be negative")
    if args.geocode_interval < 1.0:
        parser.error("--geocode-interval must be at least 1.0 for an OpenCage free account")
    if args.quota_reserve < 0:
        parser.error("--quota-reserve cannot be negative")
    if args.image_max_edge < 0:
        parser.error("--image-max-edge cannot be negative")
    if not 1 <= args.image_jpeg_quality <= 100:
        parser.error("--image-jpeg-quality must be between 1 and 100")

    load_dotenv(ROOT / ".env", override=False)
    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    _shared.setup_logging(app_config.log_level, colorize=True, file_enabled=False)
    if not args.output_dir:
        args.output_dir = str(default_output_dir(args.dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "infer":
        run_inference(args, output_dir)
    else:
        _configure_shared_geocoding()
        _shared.run_geocoding(args, output_dir)


if __name__ == "__main__":
    main()
