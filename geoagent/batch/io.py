"""Input JSONL and manifest helpers for batch inference."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import Image, ImageOps

from geoagent.batch.settings import BatchSettings


class BatchInputError(ValueError):
    """A sample cannot be represented as a valid batch request."""


def make_custom_id(index: int, *identity_parts: str) -> str:
    """Create a short, stable identifier that is safe for provider limits."""

    identity = "\x1f".join(identity_parts)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"geo-{index:08d}-{digest}"


def build_chat_request(
    *,
    custom_id: str,
    image_path: Path,
    image_relative_path: str,
    prompt: str,
    settings: BatchSettings,
) -> dict[str, Any]:
    """Build one OpenAI-compatible multimodal JSONL request."""

    image_url = resolve_image_reference(
        image_path=image_path,
        image_relative_path=image_relative_path,
        settings=settings,
    )
    image_content: dict[str, Any] = {
        "type": "image_url",
        "image_url": {"url": image_url},
    }
    if settings.image_min_pixels is not None:
        image_content["min_pixels"] = settings.image_min_pixels
    if settings.image_max_pixels is not None:
        image_content["max_pixels"] = settings.image_max_pixels

    body: dict[str, Any] = dict(settings.request_extra)
    body.update(
        {
            "model": settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        }
    )
    if settings.enable_thinking is not None:
        body["enable_thinking"] = settings.enable_thinking
    if settings.thinking_budget is not None:
        body["thinking_budget"] = settings.thinking_budget
    if settings.response_format:
        body["response_format"] = {"type": settings.response_format}

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": settings.endpoint,
        "body": body,
    }


def encode_jsonl_line(request: dict[str, Any], max_line_bytes: int) -> bytes:
    line = (json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(line) > max_line_bytes:
        custom_id = request.get("custom_id", "unknown")
        raise BatchInputError(
            f"{custom_id}: encoded request is {len(line):,} bytes, exceeding "
            f"BATCH_MAX_LINE_MB ({max_line_bytes:,} bytes). Use BATCH_IMAGE_MODE=url "
            "with an HTTP(S)/OSS image host, or opt in to BATCH_IMAGE_MAX_EDGE compression."
        )
    return line


def resolve_image_reference(
    *,
    image_path: Path,
    image_relative_path: str,
    settings: BatchSettings,
) -> str:
    if not image_path.exists() or not image_path.is_file():
        raise BatchInputError(f"Image file not found: {image_path}")
    if settings.image_mode == "url":
        assert settings.image_base_url is not None
        relative = image_relative_path.replace("\\", "/").lstrip("/")
        encoded_relative = "/".join(quote(part, safe="") for part in relative.split("/"))
        return f"{settings.image_base_url.rstrip('/')}/{encoded_relative}"
    return image_to_data_url(
        image_path,
        max_edge=settings.image_max_edge,
        jpeg_quality=settings.image_jpeg_quality,
    )


def image_to_data_url(path: Path, *, max_edge: int = 0, jpeg_quality: int = 90) -> str:
    """Encode a local image, optionally resizing it before Base64 conversion."""

    data: bytes
    mime_type: str
    if max_edge > 0:
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                if max(image.size) > max_edge:
                    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                if image.mode not in {"RGB", "L"}:
                    background = Image.new("RGB", image.size, "white")
                    if "A" in image.getbands():
                        background.paste(image, mask=image.getchannel("A"))
                    else:
                        background.paste(image.convert("RGB"))
                    image = background
                elif image.mode == "L":
                    image = image.convert("RGB")
                stream = io.BytesIO()
                image.save(stream, format="JPEG", quality=jpeg_quality, optimize=True)
                data = stream.getvalue()
                mime_type = "image/jpeg"
        except (OSError, ValueError) as exc:
            raise BatchInputError(f"Cannot read image {path}: {exc}") from exc
    else:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise BatchInputError(f"Cannot read image {path}: {exc}") from exc
        mime_type = _image_mime(path)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class JsonlChunkWriter:
    """Write request lines into provider-sized JSONL chunks."""

    def __init__(
        self,
        output_dir: Path,
        *,
        max_requests: int,
        max_bytes: int,
    ) -> None:
        self.output_dir = output_dir
        self.max_requests = max_requests
        self.max_bytes = max_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stream: Any = None
        self._current: dict[str, Any] | None = None
        self.chunks: list[dict[str, Any]] = []

    def add(self, line: bytes) -> int:
        if len(line) > self.max_bytes:
            raise BatchInputError(
                f"A single JSONL line ({len(line):,} bytes) exceeds the configured "
                f"batch file limit ({self.max_bytes:,} bytes)."
            )
        if self._current is None:
            self._open_chunk()
        assert self._current is not None
        if (
            self._current["request_count"] >= self.max_requests
            or self._current["size_bytes"] + len(line) > self.max_bytes
        ):
            self._close_chunk()
            self._open_chunk()
        assert self._stream is not None and self._current is not None
        self._stream.write(line)
        self._current["request_count"] += 1
        self._current["size_bytes"] += len(line)
        return int(self._current["index"])

    def close(self) -> list[dict[str, Any]]:
        self._close_chunk()
        return list(self.chunks)

    def _open_chunk(self) -> None:
        index = len(self.chunks) + 1
        path = self.output_dir / f"input_{index:04d}.jsonl"
        self._stream = path.open("wb")
        self._current = {
            "index": index,
            "input_path": str(path),
            "request_count": 0,
            "size_bytes": 0,
            "input_file_id": None,
            "batch_id": None,
            "status": "prepared",
            "output_file_id": None,
            "error_file_id": None,
            "output_path": None,
            "error_path": None,
            "remote": {},
        }
        self.chunks.append(self._current)

    def _close_chunk(self) -> None:
        if self._stream is None:
            return
        self._stream.flush()
        self._stream.close()
        self._stream = None
        if self._current and self._current["request_count"] == 0:
            Path(self._current["input_path"]).unlink(missing_ok=True)
            self.chunks.pop()
        self._current = None


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Batch manifest not found: {path}") from exc
    if not isinstance(loaded, dict):
        raise BatchInputError(f"Batch manifest must contain a JSON object: {path}")
    return loaded


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically persist state after every remote mutation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _image_mime(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0]
    if not mime_type or not mime_type.startswith("image/"):
        raise BatchInputError(f"Unsupported or unknown image type: {path}")
    return mime_type
