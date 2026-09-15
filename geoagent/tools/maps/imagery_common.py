"""Shared helpers for map imagery verification tools."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw

from geoagent.core.config import AppConfig
from geoagent.core.json_utils import clamp_float, extract_json_payload
from geoagent.models.vlm_client import VLMClient


class MapImageryError(RuntimeError):
    """Raised when map imagery retrieval cannot complete."""


def cache_dir_for(task_id: str | None, tool_name: str) -> Path:
    safe_task_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in (task_id or "manual"))
    project_root = Path(__file__).resolve().parents[3]
    path = project_root / "outputs" / "cache" / "map_cache" / "map_imagery" / safe_task_id / tool_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_token(*parts: Any, length: int = 12) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:length]


def download_image(url: str, path: Path, timeout: int = 45, headers: dict[str, str] | None = None) -> Path:
    response = requests.get(url, timeout=timeout, headers=headers or browser_headers())
    if not response.ok:
        raise MapImageryError(f"HTTP {response.status_code}: {response.text[:200]}")
    try:
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise MapImageryError(f"Downloaded content is not an image: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def compose_tile_grid(
    tiles: list[dict[str, Any]],
    output_path: Path,
    tile_size: int = 256,
    candidate_marker: dict[str, Any] | None = None,
) -> Path:
    if not tiles:
        raise MapImageryError("No tiles available for composition.")
    rows = sorted({int(tile["grid_y"]) for tile in tiles})
    cols = sorted({int(tile["grid_x"]) for tile in tiles})
    row_index = {row: idx for idx, row in enumerate(rows)}
    col_index = {col: idx for idx, col in enumerate(cols)}
    mosaic = Image.new("RGB", (len(cols) * tile_size, len(rows) * tile_size), (238, 238, 238))
    draw = ImageDraw.Draw(mosaic)

    for tile in tiles:
        path = Path(tile["path"])
        image = Image.open(path).convert("RGB").resize((tile_size, tile_size))
        x = col_index[int(tile["grid_x"])] * tile_size
        y = row_index[int(tile["grid_y"])] * tile_size
        mosaic.paste(image, (x, y))
        label = str(tile.get("label") or "")
        if label:
            draw.rectangle((x, y, x + tile_size, y + 20), fill=(0, 0, 0))
            draw.text((x + 4, y + 4), label[:42], fill=(255, 255, 255))

    if candidate_marker:
        marker_x = int(round(float(candidate_marker["x"])))
        marker_y = int(round(float(candidate_marker["y"])))
        marker_color = (220, 0, 0)
        draw.ellipse(
            (marker_x - 11, marker_y - 11, marker_x + 11, marker_y + 11),
            outline=(255, 255, 255),
            width=5,
        )
        draw.ellipse(
            (marker_x - 9, marker_y - 9, marker_x + 9, marker_y + 9),
            outline=marker_color,
            width=3,
        )
        draw.line((marker_x - 14, marker_y, marker_x + 14, marker_y), fill=marker_color, width=3)
        draw.line((marker_x, marker_y - 14, marker_x, marker_y + 14), fill=marker_color, width=3)
        draw.ellipse((marker_x - 2, marker_y - 2, marker_x + 2, marker_y + 2), fill=marker_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(output_path)
    return output_path


def compose_contact_sheet(
    images: list[dict[str, Any]],
    output_path: Path,
    cell_size: tuple[int, int] | None = None,
    columns: int = 2,
) -> Path:
    if not images:
        raise MapImageryError("No images available for contact sheet.")
    if cell_size is None:
        cell_size = _infer_contact_sheet_cell_size(images)
    columns = max(1, min(columns, len(images)))
    rows = (len(images) + columns - 1) // columns
    label_height = 26
    sheet = Image.new("RGB", (columns * cell_size[0], rows * (cell_size[1] + label_height)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for idx, item in enumerate(images):
        col = idx % columns
        row = idx // columns
        x = col * cell_size[0]
        y = row * (cell_size[1] + label_height)
        label = str(item.get("label") or f"image_{idx}")
        draw.rectangle((x, y, x + cell_size[0], y + label_height), fill=(28, 28, 28))
        draw.text((x + 8, y + 7), label[:60], fill=(255, 255, 255))
        image = Image.open(Path(item["path"])).convert("RGB")
        image.thumbnail(cell_size)
        paste_x = x + (cell_size[0] - image.width) // 2
        paste_y = y + label_height + (cell_size[1] - image.height) // 2
        sheet.paste(image, (paste_x, paste_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _infer_contact_sheet_cell_size(images: list[dict[str, Any]]) -> tuple[int, int]:
    widths: list[int] = []
    heights: list[int] = []
    for item in images:
        with Image.open(Path(item["path"])) as image:
            widths.append(image.width)
            heights.append(image.height)
    if not widths or not heights:
        raise MapImageryError("No readable images available for contact sheet.")
    return max(widths), max(heights)


def compare_with_vlm(
    app_config: AppConfig,
    source_image_path: str,
    reference_image_path: str,
    prompt: str,
) -> dict[str, Any]:
    if not source_image_path or not Path(source_image_path).exists():
        raise MapImageryError(f"Source image is unavailable: {source_image_path}")
    if not reference_image_path or not Path(reference_image_path).exists():
        raise MapImageryError(f"Reference image is unavailable: {reference_image_path}")

    response = VLMClient(app_config=app_config).generate(
        prompt,
        image_paths=[source_image_path, reference_image_path],
        json_mode=True,
        max_tokens=1800,
    )
    model_usage = {
        "role": "tool:map_imagery_verification",
        "model": response.get("model"),
        "provider": response.get("provider"),
        "usage": response.get("usage") if isinstance(response.get("usage"), dict) else {},
    }
    try:
        payload = extract_json_payload(str(response.get("text", "")))
    except (ValueError, json.JSONDecodeError) as exc:
        raise MapImageryError(f"VLM did not return valid verification JSON: {exc}") from exc
    return normalize_verification(payload, provider=response.get("provider"), model_usage=model_usage)


def normalize_verification(
    payload: dict[str, Any],
    provider: str | None = None,
    model_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verdict = str(payload.get("verdict", "inconclusive")).strip().lower()
    if verdict not in {"supports", "contradicts", "inconclusive"}:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "confidence": clamp_float(payload.get("confidence")),
        "matching_features": _string_list(payload.get("matching_features")),
        "contradictions": _string_list(payload.get("contradictions")),
        "limitations": _string_list(payload.get("limitations")),
        "recommended_next_step": str(payload.get("recommended_next_step") or ""),
        "provider": provider or payload.get("provider") or "vlm",
        "model_usage": model_usage or {},
        "raw": payload,
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def browser_headers(referer: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    return headers
