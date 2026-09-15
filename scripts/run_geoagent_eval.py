"""Evaluate the released GeoAgent model and geocode its textual predictions.

The script deliberately separates model inference from OpenCage forward
geocoding so the 2,500-request daily free-trial quota can be consumed over
multiple resumable runs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoagent.batch.io import BatchInputError, image_to_data_url
from geoagent.core.config import load_app_config
from geoagent.core.json_utils import extract_json_payload
from geoagent.core.logging import logger, setup_logging
from geoagent.eval import haversine_m


_EVAL_SPEC = importlib.util.spec_from_file_location(
    "_run_dataset_eval_shared_geoagent", ROOT / "scripts" / "run_dataset_eval.py"
)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:
    raise RuntimeError("Cannot load run_dataset_eval.py for shared dataset logic.")
_eval = importlib.util.module_from_spec(_EVAL_SPEC)
sys.modules[_EVAL_SPEC.name] = _eval
_EVAL_SPEC.loader.exec_module(_eval)


OFFICIAL_SYSTEM_PROMPT = """You are an expert with rich experience in the field of geolocation, skilled at accurately locating the geographic location of images through various clues in the images, such as traffic signs, architectural styles, natural landscapes, etc. At the same time, you are also a mentor in building the chain of thought, able to organize complex ideas into clear and standardized patterns.
You possess knowledge in multiple disciplines such as geography, cartography, transportation, and architecture, and are able to identify the characteristics of different countries, regions, and locations. At the same time, you have the ability to analyze logic and construct a chain of thought. Task: Output the thought chain and final answer based on the image input by the user. The thought chain includes:
        Country Identification/Regional Guess/Precise Localization.
        Possible clues include: National clues: (Example: traffic sign shape/color, language and text, driving direction, architectural style, vegetation and climate characteristics, etc.)
        Regional clues: (logo/enterprise, topography, vegetation type, regional traffic signs, dialect/spelling, license plate style, area code/postal code, infrastructure features, etc.)
        Accurate positioning: (road sign text, street name, house number, landmark building, river and lake water system, place attributes such as park/city/commercial district, shop name and storefront, etc.)
        Do not output objects that do not exist in the image.
        Output strictly in JSON format:
        {
        "ChainOfThought": {
            "CountryIdentification": {
            "Clues": [],
            "Reasoning": "",
            "Conclusion": "",
            "Uncertainty": ""
            },
            "RegionalGuess": {
            "Clues": [],
            "Reasoning": "",
            "Conclusion": "",
            "Uncertainty": ""
            },
            "PreciseLocalization": {
            "Clues": [],
            "Reasoning": "",
            "Conclusion": "",
            "Uncertainty": ""
            }
        },
        "FinalAnswer": "Country; Region; Specific Location"
        }"""

OFFICIAL_USER_PROMPT = "Based on the image, tell me the specific location and your thinking process"
OPENCAGE_URL = "https://api.opencagedata.com/geocode/v1/json"
DEFAULT_RESULTS_DIR = ROOT / "outputs" / "results"

DISTANCE_LEVELS_M = {
    "fine_10m": 10,
    "fine_20m": 20,
    "fine_50m": 50,
    "fine_100m": 100,
    "street_1km": 1_000,
    "local_5km": 5_000,
    "city_25km": 25_000,
    "region_200km": 200_000,
    "country_750km": 750_000,
    "continent_2500km": 2_500_000,
}
DISTANCE_COLUMNS = [f"correct_{threshold}m" for threshold in dict.fromkeys(DISTANCE_LEVELS_M.values())]

PREDICTION_EXTRA_FIELDS = [
    "final_answer",
    "pred_specific_location",
    "chain_of_thought_json",
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

GEOCODING_FIELDS = [
    "dataset",
    "subset",
    "item_id",
    "final_answer",
    "geocode_query",
    "status",
    "error",
    "latitude",
    "longitude",
    "formatted",
    "confidence",
    "components_json",
    "bounds_json",
    "rate_limit",
    "rate_remaining",
    "rate_reset",
    "source_item_id",
    "http_attempts",
    "latency_seconds",
]

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

_thread_local = threading.local()


class OpenCageQuotaExhausted(RuntimeError):
    """Raised when OpenCage reports that the daily hard limit is exhausted."""

    def __init__(self, reset_at: str = "") -> None:
        message = "OpenCage daily quota is exhausted."
        if reset_at:
            message += f" Reset time: {reset_at}."
        super().__init__(message)
        self.reset_at = reset_at


class RequestPacer:
    """Keep request start times at least ``interval`` seconds apart."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.last_started = 0.0

    def wait(self) -> None:
        delay = self.interval - (time.monotonic() - self.last_started)
        if delay > 0:
            time.sleep(delay)
        self.last_started = time.monotonic()


class OpenCageCache:
    """Persistent query cache shared by all GeoAgent evaluation directories."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS geocoding_cache "
                "(query_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
            )

    def get(self, query_key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as connection:
            record = connection.execute(
                "SELECT payload_json FROM geocoding_cache WHERE query_key = ?",
                (query_key,),
            ).fetchone()
        return json.loads(record[0]) if record is not None else None

    def put(self, query_key: str, row: dict[str, Any]) -> None:
        payload = json.dumps(row, ensure_ascii=False)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO geocoding_cache (query_key, payload_json) VALUES (?, ?)",
                (query_key, payload),
            )


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def chat_completion_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


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


def _shared_items(dataset_root: Path, selector: str) -> list[Any]:
    return _eval.discover_items(
        argparse.Namespace(
            datasets=selector,
            dataset_root=str(dataset_root),
            offset=0,
            limit=None,
        )
    )


def discover_items(args: argparse.Namespace) -> list[Any]:
    dataset_name = args.dataset.strip().lower()
    dataset_root = Path(args.dataset_root)

    if dataset_name == "imageobench-dataset2":
        items = _shared_items(dataset_root, "imageobench-dataset2")
    elif dataset_name == "im2gps3k":
        items = _shared_items(dataset_root, "im2gps3k")
    elif dataset_name in {"geoexp7k-learning", "geoexp7k-test"}:
        items = _shared_items(dataset_root, dataset_name)
    else:
        raise ValueError(
            "--dataset must be geoexp7k-learning, geoexp7k-test, "
            "im2gps3k, or imageobench-dataset2."
        )

    if args.offset:
        items = items[args.offset :]
    if args.limit is not None:
        items = items[: args.limit]
    return items


def _item_key(item: Any) -> tuple[str, str, str]:
    return item.dataset, item.subset, item.item_id


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _eval.clean_text(row.get("dataset")),
        _eval.clean_text(row.get("subset")),
        _eval.clean_text(row.get("item_id")),
    )


def _load_resumable_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        return [row for row in csv.DictReader(stream) if _eval.clean_text(row.get("status")) != "error"]


def _rewrite_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_geoagent_output(content: str) -> dict[str, str]:
    payload = extract_json_payload(content)
    if not isinstance(payload, dict):
        raise ValueError("GeoAgent output JSON must be an object.")

    final_answer = _eval.clean_text(payload.get("FinalAnswer"))
    if not final_answer:
        raise ValueError("GeoAgent output does not contain FinalAnswer.")

    parts = [part.strip() for part in final_answer.split(";", 2)]
    country = parts[0] if parts else ""
    region = parts[1] if len(parts) > 1 else ""
    specific_location = parts[2] if len(parts) > 2 else ""
    return {
        "final_answer": final_answer,
        "country": country,
        "region": region,
        "specific_location": specific_location,
        "chain_of_thought_json": _eval.csv_json(payload.get("ChainOfThought") or {}),
    }


def build_inference_payload(item: Any, args: argparse.Namespace, model: str) -> dict[str, Any]:
    image_url = image_to_data_url(
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
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
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
    row.update(
        {
            "final_answer": "",
            "pred_specific_location": "",
            "chain_of_thought_json": "",
            "parse_error": "",
            "eval_model_url": base_url,
            "eval_model_name": model,
            "response_model": "",
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
        return row

    start = time.monotonic()
    try:
        data, attempts = post_with_retry(
            url=chat_completion_url(base_url),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload=build_inference_payload(item, args, model),
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
        )
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("GeoAgent response contains no choices.")
        choice = choices[0]
        content = _message_text(choice.get("message") or {})
        if not content:
            raise RuntimeError("GeoAgent returned empty content.")

        row["status"] = "completed"
        row["raw_model_output"] = content
        row["token_usage_json"] = _eval.csv_json(data.get("usage") or {})
        row["api_call_count_json"] = _eval.csv_json({"vlm": attempts})
        row["response_model"] = _eval.clean_text(data.get("model"))
        row["request_id"] = _eval.clean_text(data.get("id"))
        row["finish_reason"] = _eval.clean_text(choice.get("finish_reason"))
        row["http_attempts"] = attempts

        try:
            parsed = parse_geoagent_output(content)
            row["final_answer"] = parsed["final_answer"]
            row["pred_country"] = parsed["country"]
            row["pred_region"] = parsed["region"]
            row["pred_specific_location"] = parsed["specific_location"]
            row["pred_location_name"] = parsed["specific_location"] or parsed["final_answer"]
            row["chain_of_thought_json"] = parsed["chain_of_thought_json"]
        except (TypeError, ValueError) as exc:
            row["parse_error"] = str(exc)
    except (BatchInputError, OSError, RuntimeError, ValueError) as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    finally:
        row["latency_seconds"] = f"{time.monotonic() - start:.3f}"

    logger.info(f"[{index}/{total}] {row['status']} {item.image_rel}")
    return row


def run_inference(args: argparse.Namespace, output_dir: Path) -> None:
    items = discover_items(args)
    missing = sum(not item.image_path.exists() for item in items)
    logger.info(f"Discovered {len(items)} items; missing images: {missing}")
    if args.dry_run:
        return

    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    base_url = args.base_url or app_config.env.eval_model_url
    model = args.model or app_config.env.eval_model_name
    api_key = (
        app_config.env.eval_api_key.get_secret_value()
        if app_config.env.eval_api_key is not None
        else "EMPTY"
    )

    prediction_path = output_dir / "predictions.csv"
    existing_rows = _load_resumable_rows(prediction_path) if args.resume else []
    seen = {_row_key(row) for row in existing_rows}
    runnable = [item for item in items if _item_key(item) not in seen]

    if args.resume and prediction_path.exists():
        _rewrite_rows(prediction_path, PREDICTION_FIELDS, existing_rows)
        mode = "a"
    else:
        mode = "w"

    logger.info(
        f"Running {len(runnable)} GeoAgent inference request(s) with {args.workers} workers; "
        f"model={model}; endpoint={chat_completion_url(base_url)}"
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


def build_geocode_query(prediction: dict[str, Any]) -> str:
    parts = [
        _eval.clean_text(prediction.get("pred_specific_location")),
        _eval.clean_text(prediction.get("pred_region")),
        _eval.clean_text(prediction.get("pred_country")),
    ]
    query_parts: list[str] = []
    for part in parts:
        if part and part.casefold() not in {value.casefold() for value in query_parts}:
            query_parts.append(part)
    return ", ".join(query_parts) or _eval.clean_text(prediction.get("final_answer")).replace(";", ",")


def _rate_value(data: dict[str, Any], response: requests.Response, name: str) -> str:
    rate = data.get("rate") or {}
    value = rate.get(name)
    if value is not None:
        return str(value)
    return response.headers.get(f"X-RateLimit-{name.title()}", "")


def _format_reset(value: str) -> str:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return value
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def request_opencage(
    query: str,
    *,
    api_key: str,
    timeout: float,
    max_retries: int,
    pacer: RequestPacer,
) -> tuple[dict[str, Any], requests.Response, int]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        pacer.wait()
        response: requests.Response | None = None
        try:
            response = _session().get(
                OPENCAGE_URL,
                params={
                    "q": query,
                    "key": api_key,
                    "limit": 1,
                    "no_annotations": 1,
                },
                timeout=timeout,
            )
            if response.status_code == 402:
                reset_at = _format_reset(response.headers.get("X-RateLimit-Reset", ""))
                raise OpenCageQuotaExhausted(reset_at)
            if response.ok:
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("OpenCage response must be a JSON object.")
                return data, response, attempt + 1
            if response.status_code in {401, 403}:
                raise PermissionError(f"OpenCage HTTP {response.status_code}: {response.text[:500]}")
            if response.status_code not in {408, 429, 503}:
                raise RuntimeError(f"OpenCage HTTP {response.status_code}: {response.text[:500]}")
            last_error = RuntimeError(f"OpenCage HTTP {response.status_code}: {response.text[:500]}")
        except OpenCageQuotaExhausted:
            raise
        except PermissionError:
            raise
        except requests.RequestException as exc:
            last_error = exc
        except ValueError as exc:
            last_error = RuntimeError(f"Invalid OpenCage JSON response: {exc}")

        if attempt < max_retries:
            delay = _retry_delay(response, attempt, pacer.interval)
            if delay > pacer.interval:
                time.sleep(delay - pacer.interval)

    raise RuntimeError(f"OpenCage request failed after {max_retries + 1} attempts: {last_error}")


def geocoding_row(
    prediction: dict[str, Any],
    *,
    query: str,
    data: dict[str, Any],
    response: requests.Response,
    attempts: int,
    latency_seconds: float,
) -> dict[str, Any]:
    results = data.get("results") or []
    row: dict[str, Any] = {
        "dataset": prediction.get("dataset", ""),
        "subset": prediction.get("subset", ""),
        "item_id": prediction.get("item_id", ""),
        "final_answer": prediction.get("final_answer", ""),
        "geocode_query": query,
        "status": "no_result",
        "error": "",
        "latitude": "",
        "longitude": "",
        "formatted": "",
        "confidence": "",
        "components_json": "",
        "bounds_json": "",
        "rate_limit": _rate_value(data, response, "limit"),
        "rate_remaining": _rate_value(data, response, "remaining"),
        "rate_reset": _format_reset(_rate_value(data, response, "reset")),
        "source_item_id": "",
        "http_attempts": attempts,
        "latency_seconds": f"{latency_seconds:.3f}",
    }
    if not results:
        return row

    result = results[0]
    geometry = result.get("geometry") or {}
    latitude = _eval.parse_float(geometry.get("lat"))
    longitude = _eval.parse_float(geometry.get("lng"))
    if latitude is None or longitude is None:
        row["status"] = "error"
        row["error"] = "OpenCage result does not contain a valid geometry."
        return row

    row.update(
        {
            "status": "completed",
            "latitude": latitude,
            "longitude": longitude,
            "formatted": _eval.clean_text(result.get("formatted")),
            "confidence": _eval.clean_text(result.get("confidence")),
            "components_json": _eval.csv_json(result.get("components") or {}),
            "bounds_json": _eval.csv_json(result.get("bounds") or {}),
        }
    )
    return row


def cached_geocoding_row(
    prediction: dict[str, Any], query: str, source: dict[str, Any]
) -> dict[str, Any]:
    row = {field: source.get(field, "") for field in GEOCODING_FIELDS}
    row.update(
        {
            "dataset": prediction.get("dataset", ""),
            "subset": prediction.get("subset", ""),
            "item_id": prediction.get("item_id", ""),
            "final_answer": prediction.get("final_answer", ""),
            "geocode_query": query,
            "status": "cached" if source.get("status") in {"completed", "cached"} else "no_result",
            "source_item_id": source.get("item_id", ""),
            "rate_limit": "",
            "rate_remaining": "",
            "rate_reset": "",
            "http_attempts": 0,
            "latency_seconds": "0.000",
        }
    )
    return row


def _normalized_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _error_geocoding_row(prediction: dict[str, Any], query: str, error: str) -> dict[str, Any]:
    row = {field: "" for field in GEOCODING_FIELDS}
    row.update(
        {
            "dataset": prediction.get("dataset", ""),
            "subset": prediction.get("subset", ""),
            "item_id": prediction.get("item_id", ""),
            "final_answer": prediction.get("final_answer", ""),
            "geocode_query": query,
            "status": "error",
            "error": error,
        }
    )
    return row


def run_geocoding(args: argparse.Namespace, output_dir: Path) -> None:
    prediction_path = output_dir / "predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Run the inference stage first: {prediction_path}")

    with prediction_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    candidates = [
        row
        for row in predictions
        if _eval.clean_text(row.get("status")) == "completed"
        and _eval.clean_text(row.get("final_answer"))
    ]

    geocoding_path = output_dir / "geocoding.csv"
    if geocoding_path.exists() and not args.resume:
        raise RuntimeError(f"{geocoding_path} already exists; use --resume to continue without spending duplicate quota.")
    existing_rows = _load_resumable_rows(geocoding_path) if args.resume else []
    seen = {_row_key(row) for row in existing_rows}
    pending = [row for row in candidates if _row_key(row) not in seen]

    query_cache: dict[str, dict[str, Any]] = {}
    shared_cache = OpenCageCache(Path(args.geocode_cache))
    for row in existing_rows:
        if row.get("status") in {"completed", "cached", "no_result"}:
            normalized = _normalized_query(_eval.clean_text(row.get("geocode_query")))
            query_cache[normalized] = row
            shared_cache.put(normalized, row)

    if args.resume and geocoding_path.exists():
        _rewrite_rows(geocoding_path, GEOCODING_FIELDS, existing_rows)
        mode = "a"
    else:
        mode = "w"

    api_key = _eval.clean_text(os.getenv("OPENCAGE_API_KEY"))
    if not api_key:
        raise RuntimeError("OPENCAGE_API_KEY is required in .env for the geocode stage.")

    logger.info(
        f"Geocoding {len(pending)} pending prediction(s); request cap={args.geocode_limit}; "
        f"interval={args.geocode_interval:.2f}s; quota reserve={args.quota_reserve}"
    )
    pacer = RequestPacer(args.geocode_interval)
    requests_used = 0
    requests_enabled = True
    request_limit_logged = False
    geocoding_path.parent.mkdir(parents=True, exist_ok=True)
    with geocoding_path.open(mode, encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GEOCODING_FIELDS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        for index, prediction in enumerate(pending, start=1):
            query = build_geocode_query(prediction)
            normalized = _normalized_query(query)
            cached = query_cache.get(normalized) or shared_cache.get(normalized)
            if cached is not None:
                row = cached_geocoding_row(prediction, query, cached)
            else:
                if not requests_enabled or requests_used >= args.geocode_limit:
                    if not request_limit_logged:
                        if requests_enabled:
                            reason = f"Reached --geocode-limit={args.geocode_limit}"
                        else:
                            reason = "OpenCage requests are disabled"
                        logger.info(f"{reason}; continuing with shared-cache hits only.")
                        request_limit_logged = True
                    continue
                start = time.monotonic()
                try:
                    data, response, attempts = request_opencage(
                        query,
                        api_key=api_key,
                        timeout=args.geocode_timeout,
                        max_retries=args.geocode_max_retries,
                        pacer=pacer,
                    )
                    requests_used += attempts
                    row = geocoding_row(
                        prediction,
                        query=query,
                        data=data,
                        response=response,
                        attempts=attempts,
                        latency_seconds=time.monotonic() - start,
                    )
                except OpenCageQuotaExhausted as exc:
                    logger.warning(str(exc))
                    requests_enabled = False
                    continue
                except PermissionError:
                    raise
                except RuntimeError as exc:
                    row = _error_geocoding_row(prediction, query, str(exc))

            writer.writerow({field: row.get(field, "") for field in GEOCODING_FIELDS})
            stream.flush()
            if row["status"] in {"completed", "cached", "no_result"}:
                query_cache[normalized] = row
                shared_cache.put(normalized, row)

            logger.info(
                f"[{index}/{len(pending)}] {row['status']} {prediction.get('item_id', '')}; "
                f"OpenCage requests this run: {requests_used}"
            )
            remaining = _eval.parse_float(row.get("rate_remaining"))
            if remaining is not None and remaining <= args.quota_reserve:
                reset_at = _eval.clean_text(row.get("rate_reset"))
                logger.warning(
                    f"Disabling OpenCage requests with {int(remaining)} request(s) remaining"
                    + (f" until {reset_at}" if reset_at else "")
                    + "; continuing with shared-cache hits only."
                )
                requests_enabled = False
            if args.fail_fast and row["status"] == "error":
                raise RuntimeError(row["error"])

    write_results_and_summary(output_dir)


def evaluate_coordinates(row: dict[str, Any]) -> None:
    gt_lat = _eval.parse_float(row.get("gt_latitude"))
    gt_lon = _eval.parse_float(row.get("gt_longitude"))
    pred_lat = _eval.parse_float(row.get("pred_latitude"))
    pred_lon = _eval.parse_float(row.get("pred_longitude"))
    distance: float | None = None
    if gt_lat is not None and gt_lon is not None and pred_lat is not None and pred_lon is not None:
        distance = haversine_m(pred_lat, pred_lon, gt_lat, gt_lon)
        row["pred_gt_distance_m"] = f"{distance:.2f}"
    for threshold in dict.fromkeys(DISTANCE_LEVELS_M.values()):
        if gt_lat is not None and gt_lon is not None:
            row[f"correct_{threshold}m"] = "1" if distance is not None and distance <= threshold else "0"


def write_results_and_summary(output_dir: Path) -> None:
    prediction_path = output_dir / "predictions.csv"
    geocoding_path = output_dir / "geocoding.csv"
    with prediction_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    geocoding_rows: list[dict[str, str]] = []
    if geocoding_path.exists():
        with geocoding_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            geocoding_rows = list(csv.DictReader(stream))
    geocoding_by_key = {_row_key(row): row for row in geocoding_rows}

    results: list[dict[str, Any]] = []
    for prediction in predictions:
        row: dict[str, Any] = dict(prediction)
        geocoding = geocoding_by_key.get(_row_key(prediction))
        if _eval.clean_text(prediction.get("status")) == "completed":
            if not _eval.clean_text(prediction.get("final_answer")):
                row["status"] = "geocode_unavailable"
            elif geocoding is None:
                row["status"] = "geocode_pending"
            else:
                geocode_status = _eval.clean_text(geocoding.get("status"))
                row["opencage_status"] = geocode_status
                row["geocode_query"] = geocoding.get("geocode_query", "")
                row["opencage_error"] = geocoding.get("error", "")
                row["opencage_formatted"] = geocoding.get("formatted", "")
                row["opencage_confidence"] = geocoding.get("confidence", "")
                row["opencage_components_json"] = geocoding.get("components_json", "")
                row["opencage_bounds_json"] = geocoding.get("bounds_json", "")
                row["opencage_rate_remaining"] = geocoding.get("rate_remaining", "")
                row["opencage_rate_reset"] = geocoding.get("rate_reset", "")
                if geocode_status in {"completed", "cached"}:
                    row["status"] = "completed"
                    row["pred_granularity"] = "coordinates"
                    row["pred_latitude"] = geocoding.get("latitude", "")
                    row["pred_longitude"] = geocoding.get("longitude", "")
                    row["pred_coordinates"] = (
                        f"{row['pred_latitude']}, {row['pred_longitude']}"
                    )
                elif geocode_status == "no_result":
                    row["status"] = "geocode_no_result"
                else:
                    row["status"] = "geocode_error"
        evaluate_coordinates(row)
        results.append(row)

    result_path = output_dir / "results.csv"
    _rewrite_rows(result_path, RESULT_FIELDS, results)

    gt_rows = [
        row
        for row in results
        if _eval.parse_float(row.get("gt_latitude")) is not None
        and _eval.parse_float(row.get("gt_longitude")) is not None
    ]
    coordinate_rows = [row for row in gt_rows if _eval.parse_float(row.get("pred_gt_distance_m")) is not None]
    accuracy: dict[str, Any] = {}
    for name, threshold in DISTANCE_LEVELS_M.items():
        correct = sum(_eval.clean_text(row.get(f"correct_{threshold}m")) == "1" for row in gt_rows)
        accuracy[name] = {
            "threshold_m": threshold,
            "correct": correct,
            "total": len(gt_rows),
            "accuracy": correct / len(gt_rows) if gt_rows else 0.0,
        }

    status_values = sorted({_eval.clean_text(row.get("status")) for row in results})
    summary = {
        "predictions_csv": str(prediction_path),
        "geocoding_csv": str(geocoding_path),
        "results_csv": str(result_path),
        "total_rows": len(results),
        "ground_truth_rows": len(gt_rows),
        "coordinate_predictions": len(coordinate_rows),
        "coordinate_coverage": len(coordinate_rows) / len(gt_rows) if gt_rows else 0.0,
        "status_counts": {
            status: sum(_eval.clean_text(row.get("status")) == status for row in results)
            for status in status_values
        },
        "accuracy": accuracy,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"Coordinate coverage: {len(coordinate_rows)}/{len(gt_rows)} "
        f"({summary['coordinate_coverage'] * 100:.2f}%)"
    )
    logger.info(f"Merged results written to {result_path}; summary written to {summary_path}")


def default_output_dir(dataset: str) -> Path:
    name = dataset.strip().lower()
    if name in {"geoexp7k-learning", "geoexp7k-test"}:
        name = name.replace("-", "_")
    elif name == "imageobench-dataset2":
        name = "imageobench_dataset2"
    elif name == "im2gps3k":
        name = "im2gps3k"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DEFAULT_RESULTS_DIR / f"vllm_geoagent_{name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["infer", "geocode"], required=True)
    parser.add_argument(
        "--dataset",
        default="imageobench-dataset2",
        help="geoexp7k-learning, geoexp7k-test, im2gps3k, or imageobench-dataset2.",
    )
    parser.add_argument("--dataset-root", default=str(ROOT / "datasets"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-url", default="", help="Override EVAL_MODEL_URL for inference.")
    parser.add_argument("--model", default="", help="Override EVAL_MODEL_NAME for inference.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--image-max-edge", type=int, default=0)
    parser.add_argument("--image-jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--geocode-limit",
        type=int,
        default=2400,
        help="Maximum OpenCage HTTP attempts in this run; use 0 to materialize shared-cache hits only.",
    )
    parser.add_argument(
        "--geocode-interval",
        type=float,
        default=1.1,
        help="Minimum seconds between OpenCage request starts (free accounts allow one per second).",
    )
    parser.add_argument("--geocode-timeout", type=float, default=30.0)
    parser.add_argument("--geocode-max-retries", type=int, default=3)
    parser.add_argument(
        "--geocode-cache",
        default=str(ROOT / "outputs" / "cache" / "geocoding" / "opencage_cache.sqlite3"),
        help="Persistent OpenCage query cache shared across evaluation directories.",
    )
    parser.add_argument(
        "--quota-reserve",
        type=int,
        default=10,
        help="Stop when OpenCage reports this many daily requests remaining.",
    )
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
    if not args.output_dir:
        args.output_dir = str(default_output_dir(args.dataset))

    load_dotenv(ROOT / ".env", override=False)
    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    setup_logging(app_config.log_level, colorize=True, file_enabled=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "infer":
        run_inference(args, output_dir)
    else:
        run_geocoding(args, output_dir)


if __name__ == "__main__":
    main()
