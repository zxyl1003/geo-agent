"""Validated settings for asynchronous batch inference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from geoagent.core.config import EnvConfig
from geoagent.core.exceptions import ConfigError


@dataclass(frozen=True)
class BatchSettings:
    """Provider-independent settings used by the batch evaluation pipeline."""

    provider: str
    api_key: str
    base_url: str
    model: str
    endpoint: str
    files_path: str
    batches_path: str
    auth_header: str
    auth_scheme: str
    completion_window: str
    poll_interval_seconds: float
    request_timeout_seconds: float
    max_requests_per_file: int
    max_file_bytes: int
    max_line_bytes: int
    image_mode: str
    image_base_url: str | None
    image_max_edge: int
    image_jpeg_quality: int
    image_min_pixels: int | None
    image_max_pixels: int | None
    max_tokens: int
    temperature: float
    enable_thinking: bool | None
    thinking_budget: int | None
    response_format: str | None
    request_extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: EnvConfig) -> "BatchSettings":
        try:
            request_extra = json.loads(env.batch_request_extra_json or "{}")
        except json.JSONDecodeError as exc:
            raise ConfigError("BATCH_REQUEST_EXTRA_JSON must be a valid JSON object.") from exc
        if not isinstance(request_extra, dict):
            raise ConfigError("BATCH_REQUEST_EXTRA_JSON must decode to a JSON object.")

        settings = cls(
            provider=env.batch_provider.strip().lower(),
            api_key=env.batch_api_key.get_secret_value() if env.batch_api_key else "",
            base_url=env.batch_base_url.rstrip("/"),
            model=env.batch_model.strip(),
            endpoint=_leading_slash(env.batch_endpoint),
            files_path=_leading_slash(env.batch_files_path),
            batches_path=_leading_slash(env.batch_batches_path),
            auth_header=env.batch_auth_header.strip(),
            auth_scheme=env.batch_auth_scheme.strip(),
            completion_window=env.batch_completion_window.strip(),
            poll_interval_seconds=env.batch_poll_interval_seconds,
            request_timeout_seconds=env.batch_request_timeout_seconds,
            max_requests_per_file=env.batch_max_requests_per_file,
            max_file_bytes=int(env.batch_max_file_mb * 1_000_000),
            max_line_bytes=int(env.batch_max_line_mb * 1_000_000),
            image_mode=env.batch_image_mode.strip().lower(),
            image_base_url=env.batch_image_base_url,
            image_max_edge=env.batch_image_max_edge,
            image_jpeg_quality=env.batch_image_jpeg_quality,
            image_min_pixels=env.batch_image_min_pixels,
            image_max_pixels=env.batch_image_max_pixels,
            max_tokens=env.batch_max_tokens,
            temperature=env.batch_temperature,
            enable_thinking=env.batch_enable_thinking,
            thinking_budget=env.batch_thinking_budget,
            response_format=env.batch_response_format,
            request_extra=request_extra,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.provider:
            raise ConfigError("BATCH_PROVIDER cannot be empty.")
        if not self.base_url:
            raise ConfigError("BATCH_BASE_URL cannot be empty.")
        if not self.model:
            raise ConfigError("BATCH_MODEL cannot be empty.")
        if self.image_mode not in {"base64", "url"}:
            raise ConfigError("BATCH_IMAGE_MODE must be either 'base64' or 'url'.")
        if self.image_mode == "url" and not self.image_base_url:
            raise ConfigError("BATCH_IMAGE_BASE_URL is required when BATCH_IMAGE_MODE=url.")
        if self.max_requests_per_file < 1:
            raise ConfigError("BATCH_MAX_REQUESTS_PER_FILE must be positive.")
        if self.max_file_bytes < 1 or self.max_line_bytes < 1:
            raise ConfigError("BATCH_MAX_FILE_MB and BATCH_MAX_LINE_MB must be positive.")
        if self.max_line_bytes > self.max_file_bytes:
            raise ConfigError("BATCH_MAX_LINE_MB cannot exceed BATCH_MAX_FILE_MB.")
        if self.poll_interval_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ConfigError("Batch poll interval and request timeout must be positive.")
        if self.image_max_edge < 0:
            raise ConfigError("BATCH_IMAGE_MAX_EDGE cannot be negative.")
        if not 1 <= self.image_jpeg_quality <= 100:
            raise ConfigError("BATCH_IMAGE_JPEG_QUALITY must be between 1 and 100.")
        if self.max_tokens < 1:
            raise ConfigError("BATCH_MAX_TOKENS must be positive.")
        if self.thinking_budget is not None and self.thinking_budget < 1:
            raise ConfigError("BATCH_THINKING_BUDGET must be positive when configured.")
        if self.enable_thinking is False and self.thinking_budget is not None:
            raise ConfigError("BATCH_THINKING_BUDGET cannot be set when BATCH_ENABLE_THINKING=false.")
        if self.provider in {"aliyun", "dashscope"}:
            if self.max_requests_per_file > 50_000:
                raise ConfigError("Alibaba Batch permits at most 50,000 requests per input file.")
            if self.max_file_bytes > 500_000_000:
                raise ConfigError("Alibaba Batch input files cannot exceed 500 MB.")
            if self.max_line_bytes > 1_000_000:
                raise ConfigError("Alibaba Batch JSONL lines cannot exceed 1 MB.")
            window_match = re.fullmatch(r"(\d+)([hd])", self.completion_window)
            if window_match is None:
                raise ConfigError("Alibaba BATCH_COMPLETION_WINDOW must look like 24h or 14d.")
            value = int(window_match.group(1))
            hours = value * (24 if window_match.group(2) == "d" else 1)
            if not 24 <= hours <= 336:
                raise ConfigError("Alibaba BATCH_COMPLETION_WINDOW must be between 24h and 336h.")

    def require_api_key(self) -> None:
        """Fail only for network actions; offline preparation needs no key."""

        if not self.api_key:
            raise ConfigError(
                "Batch API key is missing. Set BATCH_API_KEY "
                "before submit/status/download actions."
            )

    def public_dict(self) -> dict[str, Any]:
        """Return manifest-safe configuration without credentials."""

        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "endpoint": self.endpoint,
            "files_path": self.files_path,
            "batches_path": self.batches_path,
            "completion_window": self.completion_window,
            "max_requests_per_file": self.max_requests_per_file,
            "max_file_bytes": self.max_file_bytes,
            "max_line_bytes": self.max_line_bytes,
            "image_mode": self.image_mode,
            "image_base_url": self.image_base_url,
            "image_max_edge": self.image_max_edge,
            "image_jpeg_quality": self.image_jpeg_quality,
            "image_min_pixels": self.image_min_pixels,
            "image_max_pixels": self.image_max_pixels,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
            "thinking_budget": self.thinking_budget,
            "response_format": self.response_format,
            "request_extra": self.request_extra,
        }


def _leading_slash(value: str) -> str:
    normalized = value.strip()
    return normalized if normalized.startswith("/") else f"/{normalized}"
