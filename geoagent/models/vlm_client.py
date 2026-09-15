"""Vision client using the Brain's OpenAI-compatible endpoint."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import requests

from geoagent.core.config import AppConfig
from geoagent.core.exceptions import ConfigError
from geoagent.core.http import post_json_with_retry
from geoagent.core.logging import (
    debug_enabled,
    log_model_request,
    log_model_response,
    log_model_retry,
    log_model_stream_delta,
    log_model_stream_end,
    log_model_stream_fallback,
    log_model_stream_start,
)
from geoagent.core.text_utils import repair_latin1_mojibake, repair_text_tree
from geoagent.models.base import BaseModelClient


def brain_credentials_configured(app_config: AppConfig) -> bool:
    """Return whether the Brain has the credentials needed for real calls."""
    return bool(app_config.env.brain_api_key and app_config.env.brain_base_url)


class VLMClient(BaseModelClient):
    def __init__(self, app_config: AppConfig | None = None) -> None:
        app_config = app_config or AppConfig()
        super().__init__(config=app_config.models.get("brain"), app_config=app_config)
        self.api_key = app_config.env.brain_api_key
        self.base_url = app_config.env.brain_base_url
        self.model_name = (
            app_config.env.brain_model
            or (self.config.model_name if self.config else "")
        )

    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Call an OpenAI-compatible vision chat completion endpoint.

        Raises ConfigError when Brain credentials or a model name are missing, and
        propagates HTTP/network/JSON failures after retries.
        """

        log_model_request("VLM", self.model_name, prompt)
        self._require_credentials()
        image_paths = self._image_paths_from_kwargs(kwargs)
        if not image_paths:
            raise ValueError("VLM image input requires image_path or image_paths.")
        content = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": self.image_data_url(image_path)}})
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": kwargs.get("temperature", self.config.temperature if self.config else 0.0),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens if self.config else 1200),
        }
        if kwargs.get("json_mode", False):
            payload["response_format"] = {"type": "json_object"}
        if self._enable_thinking():
            payload["enable_thinking"] = True
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            # Qwen-class reasoning VLMs think by default; their reasoning tokens can
            # crowd out the JSON content, so disable thinking for deterministic
            # vision extraction unless explicitly enabled in config.
            payload["enable_thinking"] = False
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        data = self._post_chat_completion(payload, stream=bool(kwargs.get("stream", debug_enabled())))
        response = self._normalize_chat_response(data)
        log_model_response(
            "VLM",
            self.model_name,
            response.get("text", ""),
            {"provider": response.get("provider"), "usage": response.get("usage")},
        )
        return response

    def _enable_thinking(self) -> bool:
        """Return whether the configured Brain role opts into reasoning tokens."""
        return bool((self.config.thinking or {}).get("enabled", False)) if self.config else False

    def _require_credentials(self) -> None:
        if not (self.api_key and self.base_url):
            raise ConfigError(
                "VLMClient requires BRAIN_API_KEY and BRAIN_BASE_URL to be set in the environment or .env."
            )
        if not self.model_name:
            raise ConfigError(
                "VLMClient requires a model name. Set BRAIN_MODEL in .env or ensure configs/models.yaml defines 'brain'."
            )

    @classmethod
    def image_data_url(cls, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        raw_bytes = path.read_bytes()
        mime_type, raw_bytes = cls._normalize_image(path, raw_bytes)
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _normalize_image(path: Path, raw_bytes: bytes) -> tuple[str, bytes]:
        """Detect real image format; convert TIFF/BMP/other unsupported to JPEG."""
        # Check magic bytes for common formats
        if raw_bytes[:2] == b"\xff\xd8":
            return "image/jpeg", raw_bytes
        if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png", raw_bytes
        if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
            return "image/webp", raw_bytes
        if raw_bytes[:4] in (b"GIF8",):
            return "image/gif", raw_bytes

        # Unsupported by most VLM APIs (TIFF, BMP, etc.) — convert via Pillow
        try:
            from PIL import Image
            from io import BytesIO

            img = Image.open(BytesIO(raw_bytes))
            # Convert CMYK/LA/P etc. to RGB for JPEG compatibility
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return "image/jpeg", buf.getvalue()
        except Exception:
            # Fallback: guess from extension and hope for the best
            fallback = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            return fallback, raw_bytes

    def _image_paths_from_kwargs(self, kwargs: dict[str, Any]) -> list[str]:
        image_paths = kwargs.get("image_paths")
        if image_paths:
            return [str(path) for path in image_paths if str(path).strip()]
        image_path = kwargs.get("image_path")
        return [str(image_path)] if image_path else []

    def _post_chat_completion(self, payload: dict[str, Any], stream: bool = False) -> dict[str, Any]:
        if stream:
            try:
                return self._stream_chat_completion(payload)
            except Exception as exc:  # noqa: BLE001
                log_model_stream_fallback("VLM", str(exc))

        url = self._chat_completion_url()
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        return post_json_with_retry(
            url,
            headers,
            payload,
            timeout=240,
            log_retry=lambda attempt, error: log_model_retry("VLM", attempt, error),
        )

    def _stream_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._chat_completion_url()
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload["stream_options"] = {"include_usage": True}
        response = requests.post(url, headers=headers, json=stream_payload, timeout=120, stream=True)
        if response.status_code >= 400 and "response_format" in stream_payload:
            retry_payload = dict(stream_payload)
            retry_payload.pop("response_format", None)
            response.close()
            response = requests.post(url, headers=headers, json=retry_payload, timeout=120, stream=True)
        if not response.ok:
            raise RuntimeError(f"VLM stream HTTP {response.status_code}: {response.text[:500]}")

        content_parts: list[str] = []
        usage: dict[str, Any] = {}
        buffer = ""
        log_model_stream_start("VLM", self.model_name)
        try:
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if payload_text == "[DONE]":
                    break
                buffer += payload_text
                try:
                    chunk = json.loads(buffer)
                    buffer = ""
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage") or usage
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text = repair_latin1_mojibake(delta.get("content") or "")
                if text:
                    content_parts.append(text)
                    log_model_stream_delta(text)
        finally:
            response.close()
            log_model_stream_end("VLM")

        if not content_parts and buffer:
            raise RuntimeError("VLM stream ended with incomplete JSON chunks.")

        return {
            "id": None,
            "model": self.model_name,
            "choices": [{"message": {"content": "".join(content_parts)}}],
            "usage": usage,
        }

    def _chat_completion_url(self) -> str:
        base_url = (self.base_url or "").rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _normalize_chat_response(self, data: dict[str, Any]) -> dict[str, Any]:
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", item)) for item in content)
        content = repair_latin1_mojibake(str(content))
        usage = repair_text_tree(data.get("usage", {}))
        return {
            "model": data.get("model", self.model_name),
            "text": content,
            "usage": usage,
            "provider": "openai_compatible",
            "raw_id": data.get("id"),
        }
