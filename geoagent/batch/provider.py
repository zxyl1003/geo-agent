"""Batch provider adapters.

The first adapter implements the OpenAI-compatible Files/Batches protocol used
by Alibaba Cloud Model Studio. Other providers can reuse it when their wire
protocol is compatible, or add a dedicated adapter behind the same interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import requests

from geoagent.batch.settings import BatchSettings
from geoagent.core.exceptions import ConfigError


class BatchProviderError(RuntimeError):
    """A remote batch API request failed."""

    def __init__(self, operation: str, status_code: int | None, detail: str) -> None:
        self.operation = operation
        self.status_code = status_code
        self.detail = detail
        status = f"HTTP {status_code}" if status_code is not None else "network error"
        super().__init__(f"{operation} failed ({status}): {detail}")


class BatchProvider(Protocol):
    def upload_input_file(self, path: Path) -> dict[str, Any]: ...

    def create_batch(self, input_file_id: str, metadata: dict[str, str]) -> dict[str, Any]: ...

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]: ...

    def download_file(self, file_id: str) -> bytes: ...


class OpenAICompatibleBatchProvider:
    """Files/Batches adapter for OpenAI-compatible HTTP APIs."""

    def __init__(self, settings: BatchSettings, session: requests.Session | None = None) -> None:
        settings.require_api_key()
        self.settings = settings
        self.session = session or requests.Session()

    def upload_input_file(self, path: Path) -> dict[str, Any]:
        url = self._url(self.settings.files_path)
        try:
            with path.open("rb") as stream:
                response = self.session.post(
                    url,
                    headers=self._auth_headers(),
                    data={"purpose": "batch"},
                    files={"file": (path.name, stream, "application/jsonl")},
                    timeout=self.settings.request_timeout_seconds,
                )
        except (OSError, requests.RequestException) as exc:
            raise BatchProviderError("upload input file", None, str(exc)) from exc
        return self._json_response("upload input file", response)

    def create_batch(self, input_file_id: str, metadata: dict[str, str]) -> dict[str, Any]:
        payload = {
            "input_file_id": input_file_id,
            "endpoint": self.settings.endpoint,
            "completion_window": self.settings.completion_window,
            "metadata": metadata,
        }
        response = self._request_json("POST", self.settings.batches_path, payload, "create batch")
        return response

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]:
        path = f"{self.settings.batches_path.rstrip('/')}/{quote(batch_id, safe='')}"
        return self._request_json("GET", path, None, "retrieve batch")

    def download_file(self, file_id: str) -> bytes:
        path = f"{self.settings.files_path.rstrip('/')}/{quote(file_id, safe='')}/content"
        try:
            response = self.session.get(
                self._url(path),
                headers=self._auth_headers(),
                timeout=self.settings.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise BatchProviderError("download result file", None, str(exc)) from exc
        if not response.ok:
            raise BatchProviderError(
                "download result file",
                response.status_code,
                _response_detail(response),
            )
        return response.content

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        operation: str,
    ) -> dict[str, Any]:
        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"
        try:
            response = self.session.request(
                method,
                self._url(path),
                headers=headers,
                json=payload,
                timeout=self.settings.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise BatchProviderError(operation, None, str(exc)) from exc
        return self._json_response(operation, response)

    def _json_response(self, operation: str, response: requests.Response) -> dict[str, Any]:
        if not response.ok:
            raise BatchProviderError(operation, response.status_code, _response_detail(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise BatchProviderError(operation, response.status_code, "response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise BatchProviderError(operation, response.status_code, "response JSON is not an object")
        return payload

    def _auth_headers(self) -> dict[str, str]:
        value = " ".join(part for part in (self.settings.auth_scheme, self.settings.api_key) if part).strip()
        return {self.settings.auth_header: value}

    def _url(self, path: str) -> str:
        if path.startswith(("https://", "http://")):
            return path
        return f"{self.settings.base_url}/{path.lstrip('/')}"


def create_batch_provider(
    settings: BatchSettings,
    *,
    session: requests.Session | None = None,
) -> BatchProvider:
    """Create a configured provider adapter.

    `openai_compatible` is the
    extension point for another vendor that implements the same Files/Batches
    wire protocol and Bearer-style authentication.
    """

    if settings.provider in {"aliyun", "openai_compatible"}:
        return OpenAICompatibleBatchProvider(settings, session=session)
    raise ConfigError(
        f"Unsupported BATCH_PROVIDER={settings.provider!r}. "
        "Use aliyun or openai_compatible for a compatible vendor. "
        "Providers with a different protocol require a dedicated adapter."
    )


def _response_detail(response: requests.Response) -> str:
    text = response.text.strip()
    if not text:
        return response.reason or "empty error response"
    try:
        payload = response.json()
    except ValueError:
        return text[:2000]
    return json.dumps(payload, ensure_ascii=False, default=str)[:2000]
