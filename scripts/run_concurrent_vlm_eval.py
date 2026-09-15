"""Run concurrent Direct-VLM geolocation through an OpenAI-compatible API.

This script sends one ordinary request per image. It is separate from
``run_batch_vlm_eval.py``, which uploads JSONL files to a cloud Batch API.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoagent.batch.io import BatchInputError, image_to_data_url
from geoagent.core.config import load_app_config
from geoagent.core.exceptions import ConfigError
from geoagent.core.json_utils import extract_json_payload
from geoagent.core.logging import logger, setup_logging
from geoagent.eval import haversine_m

_EVAL_SPEC = importlib.util.spec_from_file_location(
    "_run_dataset_eval_shared_concurrent", ROOT / "scripts" / "run_dataset_eval.py"
)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:
    raise RuntimeError("Cannot load run_dataset_eval.py for shared logic.")
_eval = importlib.util.module_from_spec(_EVAL_SPEC)
sys.modules[_EVAL_SPEC.name] = _eval
_EVAL_SPEC.loader.exec_module(_eval)

PROMPT_FILE = ROOT / "configs" / "prompts" / "direct_vlm_geolocation.md"
DEFAULT_OUTPUT = ROOT / "outputs" / "results" / "concurrent_glm53_flash_im2gps3k" / "results.csv"
IM2GPS_THRESHOLDS_M = {
    "street_1km": 1_000,
    "city_25km": 25_000,
    "region_200km": 200_000,
    "country_750km": 750_000,
    "continent_2500km": 2_500_000,
}
IM2GPS_COLUMNS = [f"correct_{distance}m" for distance in IM2GPS_THRESHOLDS_M.values()]
API_FIELDS = [
    "eval_model_url",
    "eval_model_name",
    "requested_provider",
    "reasoning_effort",
    "response_model",
    "provider",
    "request_id",
    "finish_reason",
    "http_attempts",
    "latency_seconds",
    "raw_model_output",
]
FIELDNAMES = list(dict.fromkeys([*_eval.FIELDNAMES, *IM2GPS_COLUMNS, *API_FIELDS]))

_thread_local = threading.local()


def build_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def chat_completion_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "geolocation_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "city": {"type": "string"},
                    "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                    "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                },
                "required": ["country", "city", "latitude", "longitude"],
                "additionalProperties": False,
            },
        },
    }


def build_payload(
    *,
    image_path: Path,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    image_max_edge: int,
    image_jpeg_quality: int,
    response_format: str,
    provider_slug: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    image_url = image_to_data_url(
        image_path,
        max_edge=image_max_edge,
        jpeg_quality=image_jpeg_quality,
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format == "json_schema":
        payload["response_format"] = _response_schema()
    elif response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    if provider_slug:
        payload["provider"] = {
            "only": [provider_slug],
            "allow_fallbacks": False,
        }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def _retry_delay(response: requests.Response | None, attempt: int, base_delay: float) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After", "").strip()
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(base_delay * (2**attempt), 60.0)


def post_with_retry(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
) -> tuple[dict[str, Any], int]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        response: requests.Response | None = None
        try:
            response = _session().post(url, headers=headers, json=payload, timeout=timeout)
            if response.ok:
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("API response must be a JSON object.")
                return data, attempt + 1
            if response.status_code != 429 and response.status_code < 500:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
        except requests.RequestException as exc:
            last_error = exc
        except ValueError as exc:
            last_error = RuntimeError(f"Invalid JSON response: {exc}")

        if attempt < max_retries:
            time.sleep(_retry_delay(response, attempt, retry_base_delay))

    raise RuntimeError(f"API request failed after {max_retries + 1} attempts: {last_error}")


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


def fill_prediction(row: dict[str, Any], output: dict[str, Any]) -> None:
    row["pred_country"] = _eval.clean_text(output.get("country"))
    row["pred_city"] = _eval.clean_text(output.get("city"))
    row["pred_latitude"] = _eval.clean_text(output.get("latitude"))
    row["pred_longitude"] = _eval.clean_text(output.get("longitude"))


def evaluate_im2gps(row: dict[str, Any], item: Any) -> None:
    gt_lat = item.gt_latitude
    gt_lon = item.gt_longitude
    if gt_lat is None or gt_lon is None:
        return

    pred_lat = _eval.parse_float(row.get("pred_latitude"))
    pred_lon = _eval.parse_float(row.get("pred_longitude"))
    valid_prediction = (
        pred_lat is not None
        and pred_lon is not None
        and -90 <= pred_lat <= 90
        and -180 <= pred_lon <= 180
    )
    distance = None
    if valid_prediction:
        distance = haversine_m(pred_lat, pred_lon, gt_lat, gt_lon)
        row["pred_gt_distance_m"] = f"{distance:.2f}"

    for threshold in set(_eval.DISTANCE_THRESHOLDS) | set(IM2GPS_THRESHOLDS_M.values()):
        row[f"correct_{threshold}m"] = "1" if distance is not None and distance <= threshold else "0"


def run_item(
    item: Any,
    *,
    index: int,
    total: int,
    prompt: str,
    args: argparse.Namespace,
    api_key: str,
    base_url: str,
    model: str,
    provider_slug: str,
) -> dict[str, Any]:
    row = _eval.base_row(item)
    row.update(
        {
            "eval_model_url": base_url,
            "eval_model_name": model,
            "requested_provider": provider_slug,
            "reasoning_effort": args.reasoning_effort,
            "response_model": "",
            "provider": "",
            "request_id": "",
            "finish_reason": "",
            "http_attempts": "",
            "latency_seconds": "",
            "raw_model_output": "",
        }
    )
    if not item.image_path.exists():
        row["status"] = "missing_image"
        row["error"] = f"Image file not found: {item.image_path}"
        evaluate_im2gps(row, item)
        return row

    start = time.monotonic()
    try:
        payload = build_payload(
            image_path=item.image_path,
            prompt=prompt,
            model=model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            image_max_edge=args.image_max_edge,
            image_jpeg_quality=args.image_jpeg_quality,
            response_format=args.response_format,
            provider_slug=provider_slug,
            reasoning_effort=args.reasoning_effort,
        )
        data, attempts = post_with_retry(
            url=chat_completion_url(base_url),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload=payload,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
        )
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        content = _message_text(choice.get("message") or {})
        output = extract_json_payload(content)

        row["status"] = "completed"
        fill_prediction(row, output)
        row["token_usage_json"] = _eval.csv_json(data.get("usage") or {})
        row["api_call_count_json"] = _eval.csv_json({"vlm": attempts})
        row["response_model"] = _eval.clean_text(data.get("model"))
        row["provider"] = _eval.clean_text(data.get("provider"))
        row["request_id"] = _eval.clean_text(data.get("id"))
        row["finish_reason"] = _eval.clean_text(choice.get("finish_reason"))
        row["http_attempts"] = attempts
        row["raw_model_output"] = content
    except (BatchInputError, OSError, RuntimeError, ValueError) as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    finally:
        row["latency_seconds"] = f"{time.monotonic() - start:.3f}"

    evaluate_im2gps(row, item)
    logger.info(f"[{index}/{total}] {row['status']} {item.image_rel}")
    return row


def _load_existing_rows(output_path: Path) -> list[dict[str, str]]:
    if not output_path.exists():
        return []
    with output_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        return [row for row in csv.DictReader(stream) if _eval.clean_text(row.get("status")) != "error"]


def _item_key(item: Any) -> tuple[str, str, str]:
    return item.dataset, item.subset, item.item_id


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _eval.clean_text(row.get("dataset")),
        _eval.clean_text(row.get("subset")),
        _eval.clean_text(row.get("item_id")),
    )


def write_summary(output_path: Path) -> None:
    with output_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        rows = list(csv.DictReader(stream))

    gt_rows = [
        row
        for row in rows
        if _eval.parse_float(row.get("gt_latitude")) is not None
        and _eval.parse_float(row.get("gt_longitude")) is not None
    ]
    coordinate_rows = [row for row in gt_rows if _eval.parse_float(row.get("pred_gt_distance_m")) is not None]
    results: dict[str, Any] = {}
    for name, threshold in IM2GPS_THRESHOLDS_M.items():
        column = f"correct_{threshold}m"
        correct = sum(_eval.clean_text(row.get(column)) == "1" for row in gt_rows)
        results[name] = {
            "threshold_m": threshold,
            "correct": correct,
            "total": len(gt_rows),
            "accuracy": correct / len(gt_rows) if gt_rows else 0.0,
        }

    summary = {
        "output_csv": str(output_path),
        "total_rows": len(rows),
        "ground_truth_rows": len(gt_rows),
        "coordinate_predictions": len(coordinate_rows),
        "coordinate_coverage": len(coordinate_rows) / len(gt_rows) if gt_rows else 0.0,
        "status_counts": {
            status: sum(_eval.clean_text(row.get("status")) == status for row in rows)
            for status in sorted({_eval.clean_text(row.get("status")) for row in rows})
        },
        "im2gps3k_accuracy": results,
    }
    summary_path = output_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        f"Coordinate coverage: {len(coordinate_rows)}/{len(gt_rows)} "
        f"({summary['coordinate_coverage'] * 100:.2f}%)"
    )
    for name, result in results.items():
        logger.info(
            f"{name}: {result['accuracy'] * 100:.2f}% "
            f"({result['correct']}/{result['total']})"
        )
    logger.info(f"Summary written to {summary_path}")


def discover_items(args: argparse.Namespace) -> list[Any]:
    return _eval.discover_items(
        argparse.Namespace(
            datasets=args.datasets,
            dataset_root=args.dataset_root,
            offset=args.offset,
            limit=args.limit,
        )
    )


def run(args: argparse.Namespace) -> None:
    items = discover_items(args)
    missing = sum(not item.image_path.exists() for item in items)
    logger.info(f"Discovered {len(items)} items; missing images: {missing}")
    if args.dry_run:
        return

    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    env = app_config.env
    if env.eval_api_key is None:
        raise ConfigError("EVAL_API_KEY is required for concurrent evaluation.")
    api_key = env.eval_api_key.get_secret_value()
    base_url = args.base_url or env.eval_model_url
    model = args.model or env.eval_model_name
    provider_slug = args.provider or env.eval_provider

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _load_existing_rows(output_path) if args.resume else []
    seen = {_row_key(row) for row in existing_rows}
    runnable = [item for item in items if _item_key(item) not in seen]
    prompt = build_prompt()

    mode = "a" if existing_rows else "w"
    if args.resume and output_path.exists():
        with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
        mode = "a"

    logger.info(
        f"Running {len(runnable)} request(s) with {args.workers} workers; "
        f"model={model}; provider={provider_slug or 'automatic'}; "
        f"reasoning_effort={args.reasoning_effort or 'default'}; "
        f"endpoint={chat_completion_url(base_url)}"
    )
    with output_path.open(mode, encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_item,
                    item,
                    index=index,
                    total=len(runnable),
                    prompt=prompt,
                    args=args,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    provider_slug=provider_slug,
                ): item
                for index, item in enumerate(runnable, start=1)
            }
            for future in as_completed(futures):
                row = future.result()
                writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
                stream.flush()
                if args.fail_fast and row["status"] == "error":
                    raise RuntimeError(row["error"])

    write_summary(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="", help="Override EVAL_MODEL_URL.")
    parser.add_argument("--model", default="", help="Override EVAL_MODEL_NAME.")
    parser.add_argument("--provider", default="", help="Override EVAL_PROVIDER.")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "high", "max"],
        default="low",
        help="Set OpenRouter reasoning effort explicitly.",
    )
    parser.add_argument("--dataset-root", default=str(ROOT / "datasets"))
    parser.add_argument(
        "--datasets",
        default="im2gps3k",
        help=(
            "Comma-separated list: geoexp7k-learning,geoexp7k-test,"
            "im2gps3k,imageobench-dataset2,all."
        ),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--image-max-edge", type=int, default=0)
    parser.add_argument("--image-jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--response-format",
        choices=["json_schema", "json_object", "none"],
        default="json_schema",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if args.image_max_edge < 0:
        parser.error("--image-max-edge cannot be negative")
    if not 1 <= args.image_jpeg_quality <= 100:
        parser.error("--image-jpeg-quality must be between 1 and 100")

    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    setup_logging(app_config.log_level, colorize=True, file_enabled=False)
    run(args)


if __name__ == "__main__":
    main()
