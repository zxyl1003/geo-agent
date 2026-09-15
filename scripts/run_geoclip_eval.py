"""Evaluate the official PyTorch GeoCLIP model on supported datasets."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from unittest import mock
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoagent.core.logging import logger, setup_logging
from geoagent.eval import haversine_m


_EVAL_SPEC = importlib.util.spec_from_file_location(
    "_run_dataset_eval_shared_geoclip", ROOT / "scripts" / "run_dataset_eval.py"
)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:
    raise RuntimeError("Cannot load run_dataset_eval.py for shared dataset logic.")
_eval = importlib.util.module_from_spec(_EVAL_SPEC)
sys.modules[_EVAL_SPEC.name] = _eval
_EVAL_SPEC.loader.exec_module(_eval)


DEFAULT_CACHE_DIR = ROOT / "outputs" / "geoclip_eval"
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
DISTANCE_COLUMNS = [f"correct_{value}m" for value in dict.fromkeys(DISTANCE_LEVELS_M.values())]
EXTRA_FIELDS = [
    "geoclip_version",
    "clip_model",
    "torch_version",
    "gpu_uuid",
    "gallery_size",
    "rank1_probability",
    "rank1_logit",
    "top_k_results_json",
    "latency_seconds",
]
FIELDNAMES = list(dict.fromkeys([*_eval.FIELDNAMES, *DISTANCE_COLUMNS, *EXTRA_FIELDS]))


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

    if dataset_name == "im2gps3k":
        items = _shared_items(dataset_root, "im2gps3k")
    elif dataset_name == "imageobench-dataset2":
        items = _shared_items(dataset_root, "imageobench-dataset2")
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
        return [
            row
            for row in csv.DictReader(stream)
            if _eval.clean_text(row.get("status")) != "error"
        ]


def _rewrite_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def configure_runtime(gpu_uuid: str, hf_home: str) -> None:
    value = gpu_uuid.strip()
    if not value:
        raise ValueError("--gpu-uuid must not be empty.")
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if hf_home:
        os.environ["HF_HOME"] = str(Path(hf_home).resolve())


def load_model(clip_model: str) -> tuple[Any, Any, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the selected CUDA GPU.")
    torch.cuda.set_device(0)
    try:
        from geoclip import GeoCLIP
    except ImportError as exc:
        raise RuntimeError("The PyTorch backend requires `pip install geoclip`.") from exc

    version = importlib.metadata.version("geoclip")
    logger.info(f"Loading GeoCLIP {version}; the CLIP backbone may download on first use")
    clip_path = Path(clip_model)
    if clip_path.is_dir():
        from transformers import AutoProcessor, CLIPModel

        model_loader = CLIPModel.from_pretrained
        processor_loader = AutoProcessor.from_pretrained
        with (
            mock.patch.object(
                CLIPModel,
                "from_pretrained",
                side_effect=lambda _name, *args, **kwargs: model_loader(
                    str(clip_path), *args, **kwargs
                ),
            ),
            mock.patch.object(
                AutoProcessor,
                "from_pretrained",
                side_effect=lambda _name, *args, **kwargs: processor_loader(
                    str(clip_path), *args, **kwargs
                ),
            ),
        ):
            model = GeoCLIP()
    else:
        model = GeoCLIP()
    model = model.to("cuda:0").eval()
    return torch, model, version


def _cache_matches(payload: Any, coordinates: Any, version: str) -> bool:
    if not isinstance(payload, dict):
        return False
    cached_coordinates = payload.get("coordinates")
    embeddings = payload.get("embeddings")
    return (
        payload.get("geoclip_version") == version
        and cached_coordinates is not None
        and embeddings is not None
        and cached_coordinates.shape == coordinates.shape
        and bool((cached_coordinates == coordinates).all())
        and embeddings.ndim == 2
        and embeddings.shape[0] == coordinates.shape[0]
    )


def load_or_build_gallery_embeddings(
    *,
    torch: Any,
    model: Any,
    version: str,
    cache_path: Path,
    batch_size: int,
    rebuild: bool,
) -> tuple[Any, Any]:
    coordinates = model.gps_gallery.detach().to(device="cpu", dtype=torch.float32)
    if cache_path.is_file() and not rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if _cache_matches(payload, coordinates, version):
            logger.info(f"Loading cached location embeddings from {cache_path}")
            return coordinates, payload["embeddings"].to(dtype=torch.float32)
        logger.info(f"Ignoring incompatible location embedding cache: {cache_path}")

    chunks: list[Any] = []
    total = len(coordinates)
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            location_batch = coordinates[start : start + batch_size].to("cuda:0")
            embeddings = model.location_encoder(location_batch)
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            chunks.append(embeddings.cpu())
            logger.info(f"Location gallery embeddings: {min(start + len(location_batch), total)}/{total}")
    gallery_embeddings = torch.cat(chunks, dim=0).to(dtype=torch.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(
        {
            "geoclip_version": version,
            "coordinates": coordinates,
            "embeddings": gallery_embeddings,
        },
        temporary,
    )
    temporary.replace(cache_path)
    logger.info(f"Location embeddings cached at {cache_path}")
    return coordinates, gallery_embeddings


def evaluate_coordinates(row: dict[str, Any]) -> None:
    pred_lat = _eval.parse_float(row.get("pred_latitude"))
    pred_lon = _eval.parse_float(row.get("pred_longitude"))
    gt_lat = _eval.parse_float(row.get("gt_latitude"))
    gt_lon = _eval.parse_float(row.get("gt_longitude"))
    if pred_lat is None or pred_lon is None or gt_lat is None or gt_lon is None:
        return
    distance = haversine_m(pred_lat, pred_lon, gt_lat, gt_lon)
    row["pred_gt_distance_m"] = f"{distance:.2f}"
    for threshold in dict.fromkeys(DISTANCE_LEVELS_M.values()):
        row[f"correct_{threshold}m"] = "1" if distance <= threshold else "0"


def _base_result_row(item: Any, args: argparse.Namespace, version: str, torch: Any) -> dict[str, Any]:
    row = _eval.base_row(item)
    row.update(
        {
            "geoclip_version": version,
            "clip_model": args.clip_model,
            "torch_version": torch.__version__,
            "gpu_uuid": args.gpu_uuid,
            "gallery_size": "",
            "rank1_probability": "",
            "rank1_logit": "",
            "top_k_results_json": "",
            "latency_seconds": "",
        }
    )
    return row


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_image_batch(
    *,
    items: list[Any],
    args: argparse.Namespace,
    torch: Any,
    model: Any,
    version: str,
    coordinates: Any,
    gallery_embeddings: Any,
) -> list[dict[str, Any]]:
    rows = [_base_result_row(item, args, version, torch) for item in items]
    valid_indexes: list[int] = []
    images: list[Image.Image] = []
    started: dict[int, float] = {}

    for index, (item, row) in enumerate(zip(items, rows)):
        started[index] = time.monotonic()
        row["gallery_size"] = len(coordinates)
        if not item.image_path.is_file():
            row["status"] = "missing_image"
            row["error"] = f"Image file not found: {item.image_path}"
            row["latency_seconds"] = "0.000"
            continue
        try:
            with Image.open(item.image_path) as image:
                images.append(image.convert("RGB"))
            valid_indexes.append(index)
        except (OSError, ValueError) as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            row["latency_seconds"] = f"{time.monotonic() - started[index]:.3f}"

    if not valid_indexes:
        return rows

    try:
        pixel_values = model.image_encoder.image_processor(
            images=images,
            return_tensors="pt",
        )["pixel_values"].to("cuda:0")
        with torch.inference_mode():
            image_embeddings = model.image_encoder(pixel_values)
            image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=1)
            logits = model.logit_scale.exp() * (image_embeddings @ gallery_embeddings.T)
            probabilities = logits.softmax(dim=1)
            top_k = min(args.top_k, len(coordinates))
            top_probabilities, top_indexes = torch.topk(probabilities, k=top_k, dim=1)
            top_logits = torch.gather(logits, 1, top_indexes)
        top_probabilities = top_probabilities.cpu()
        top_indexes = top_indexes.cpu()
        top_logits = top_logits.cpu()
    except (RuntimeError, TypeError, ValueError) as exc:
        for index in valid_indexes:
            rows[index]["status"] = "error"
            rows[index]["error"] = str(exc)
            rows[index]["latency_seconds"] = f"{time.monotonic() - started[index]:.3f}"
        return rows

    for batch_index, row_index in enumerate(valid_indexes):
        row = rows[row_index]
        result_items: list[dict[str, Any]] = []
        for rank, gallery_index in enumerate(top_indexes[batch_index].tolist(), start=1):
            latitude, longitude = coordinates[gallery_index].tolist()
            result_items.append(
                {
                    "rank": rank,
                    "latitude": latitude,
                    "longitude": longitude,
                    "probability": float(top_probabilities[batch_index, rank - 1]),
                    "logit": float(top_logits[batch_index, rank - 1]),
                }
            )
        best = result_items[0]
        row.update(
            {
                "status": "completed",
                "pred_granularity": "coordinates",
                "pred_latitude": best["latitude"],
                "pred_longitude": best["longitude"],
                "pred_coordinates": f"{best['latitude']}, {best['longitude']}",
                "confidence": f"{best['probability']:.10f}",
                "rank1_probability": f"{best['probability']:.10f}",
                "rank1_logit": f"{best['logit']:.6f}",
                "top_k_results_json": json.dumps(result_items, ensure_ascii=False),
                "latency_seconds": f"{time.monotonic() - started[row_index]:.3f}",
            }
        )
        evaluate_coordinates(row)
    return rows


def write_summary(output_path: Path) -> None:
    with output_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        rows = list(csv.DictReader(stream))
    gt_rows = [
        row
        for row in rows
        if _eval.parse_float(row.get("gt_latitude")) is not None
        and _eval.parse_float(row.get("gt_longitude")) is not None
    ]
    predicted_rows = [row for row in gt_rows if _eval.parse_float(row.get("pred_gt_distance_m")) is not None]
    accuracy: dict[str, Any] = {}
    for name, threshold in DISTANCE_LEVELS_M.items():
        correct = sum(_eval.clean_text(row.get(f"correct_{threshold}m")) == "1" for row in gt_rows)
        accuracy[name] = {
            "threshold_m": threshold,
            "correct": correct,
            "total": len(gt_rows),
            "accuracy": correct / len(gt_rows) if gt_rows else 0.0,
        }
    status_values = sorted({_eval.clean_text(row.get("status")) for row in rows})
    summary = {
        "output_csv": str(output_path),
        "total_rows": len(rows),
        "ground_truth_rows": len(gt_rows),
        "coordinate_predictions": len(predicted_rows),
        "coordinate_coverage": len(predicted_rows) / len(gt_rows) if gt_rows else 0.0,
        "status_counts": {
            status: sum(_eval.clean_text(row.get("status")) == status for row in rows)
            for status in status_values
        },
        "accuracy": accuracy,
    }
    summary_path = output_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"Coordinate coverage: {len(predicted_rows)}/{len(gt_rows)} "
        f"({summary['coordinate_coverage'] * 100:.2f}%)"
    )
    for name, result in accuracy.items():
        logger.info(
            f"{name}: {result['accuracy'] * 100:.2f}% "
            f"({result['correct']}/{result['total']})"
        )
    logger.info(f"Summary written to {summary_path}")


def run(args: argparse.Namespace) -> None:
    items = discover_items(args)
    missing = sum(not item.image_path.is_file() for item in items)
    logger.info(f"Discovered {len(items)} items; missing images: {missing}")
    if args.dry_run:
        return

    configure_runtime(args.gpu_uuid, args.hf_home)
    torch, model, version = load_model(args.clip_model)
    cache_path = (
        Path(args.gallery_cache).resolve()
        if args.gallery_cache
        else DEFAULT_CACHE_DIR / "gallery_embeddings_pytorch.pt"
    )
    coordinates, gallery_embeddings = load_or_build_gallery_embeddings(
        torch=torch,
        model=model,
        version=version,
        cache_path=cache_path,
        batch_size=args.location_batch_size,
        rebuild=args.rebuild_gallery_cache,
    )
    gallery_embeddings = gallery_embeddings.to("cuda:0")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _load_resumable_rows(output_path) if args.resume else []
    seen = {_row_key(row) for row in existing_rows}
    runnable = [item for item in items if _item_key(item) not in seen]
    if args.resume and output_path.exists():
        _rewrite_rows(output_path, existing_rows)
        mode = "a"
    else:
        mode = "w"

    logger.info(
        f"Running {len(runnable)} GeoCLIP image(s); batch_size={args.batch_size}; GPU={args.gpu_uuid}"
    )
    with output_path.open(mode, encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        completed = 0
        for batch in _chunks(runnable, args.batch_size):
            rows = run_image_batch(
                items=batch,
                args=args,
                torch=torch,
                model=model,
                version=version,
                coordinates=coordinates,
                gallery_embeddings=gallery_embeddings,
            )
            for item, row in zip(batch, rows):
                writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
                completed += 1
                logger.info(f"[{completed}/{len(runnable)}] {row['status']} {item.image_rel}")
                if args.fail_fast and row["status"] == "error":
                    stream.flush()
                    raise RuntimeError(row["error"])
            stream.flush()

    write_summary(output_path)
    logger.info(f"GeoCLIP results written to {output_path}")


def default_output_path(dataset: str) -> Path:
    name = dataset.strip().lower()
    if name == "imageobench-dataset2":
        name = "imageobench_dataset2"
    elif name == "im2gps3k":
        name = "im2gps3k"
    elif name not in {"geoexp7k-learning", "geoexp7k-test"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DEFAULT_RESULTS_DIR / f"geoclip_{name}" / "results.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True, help="GPU UUID used through CUDA_VISIBLE_DEVICES.")
    parser.add_argument(
        "--dataset",
        default="im2gps3k",
        help="geoexp7k-learning, geoexp7k-test, im2gps3k, or imageobench-dataset2.",
    )
    parser.add_argument("--dataset-root", default=str(ROOT / "datasets"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--location-batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-large-patch14",
        help="Hugging Face model ID or a local CLIP model directory.",
    )
    parser.add_argument("--hf-home", default="", help="Optional Hugging Face cache directory.")
    parser.add_argument("--gallery-cache", default="")
    parser.add_argument("--rebuild-gallery-cache", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.location_batch_size < 1:
        parser.error("--location-batch-size must be positive.")
    if args.top_k < 1:
        parser.error("--top-k must be positive.")
    if args.offset < 0:
        parser.error("--offset must not be negative.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if not args.output:
        args.output = str(default_output_path(args.dataset))

    setup_logging("INFO", colorize=True, file_enabled=False)
    run(args)


if __name__ == "__main__":
    main()
