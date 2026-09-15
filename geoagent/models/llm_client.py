"""LLM client calling an OpenAI-compatible chat completion endpoint."""

from __future__ import annotations

from typing import Any

import json
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


# Env variables that supply credentials and model name per role.
_ROLE_ENV_VARS: dict[str, tuple[str, str, str]] = {
    "brain": ("BRAIN_API_KEY", "BRAIN_BASE_URL", "BRAIN_MODEL"),
    "memory_manager": ("MEMORY_MANAGER_API_KEY", "MEMORY_MANAGER_BASE_URL", "MEMORY_MANAGER_MODEL"),
}


class LLMClient(BaseModelClient):
    def __init__(self, app_config: AppConfig | None = None, model_role: str = "brain") -> None:
        app_config = app_config or AppConfig()
        self.model_role = model_role
        super().__init__(config=self._resolve_model_config(app_config, model_role), app_config=app_config)
        self.api_key = self._resolve_api_key(app_config, model_role)
        self.base_url = self._resolve_base_url(app_config, model_role)
        self.model_name = self._resolve_model_name(app_config, model_role)

    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Call an OpenAI-compatible chat completion endpoint.

        Raises ConfigError when the role's credentials or model name are not
        configured, and propagates HTTP/network/JSON failures after retries.
        """

        log_model_request(f"LLM:{self.model_role}", self.model_name, prompt)
        self._require_credentials()
        messages = kwargs.get("messages") or [{"role": "user", "content": prompt}]
        json_mode = bool(kwargs.get("json_mode", False))
        thinking = self._thinking_config(kwargs)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens if self.config else 1200),
        }
        if thinking:
            payload["enable_thinking"] = True
            payload["chat_template_kwargs"] = {"enable_thinking": True}
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = thinking.get("reasoning_effort", "high")
        else:
            # Qwen-class reasoning LLMs think by default; their reasoning tokens can
            # crowd out the JSON content, so disable thinking for deterministic
            # structured output unless explicitly enabled in config.
            payload["enable_thinking"] = False
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["temperature"] = kwargs.get("temperature", self.config.temperature if self.config else 0.2)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = self._post_chat_completion(payload, stream=bool(kwargs.get("stream", debug_enabled())))
        response = self._normalize_chat_response(data)
        log_model_response(
            f"LLM:{self.model_role}",
            self.model_name,
            response.get("text", ""),
            {"provider": response.get("provider"), "usage": response.get("usage")},
        )
        return response

    def _require_credentials(self) -> None:
        api_key_var, base_url_var, model_var = _ROLE_ENV_VARS.get(
            self.model_role,
            ("UNKNOWN_API_KEY", "UNKNOWN_BASE_URL", "UNKNOWN_MODEL"),
        )
        if not (self.api_key and self.base_url):
            known_roles = ", ".join(sorted(_ROLE_ENV_VARS))
            raise ConfigError(
                f"LLMClient for model_role={self.model_role!r} requires {api_key_var} and {base_url_var} "
                f"to be set in the environment or .env. Known roles: {known_roles}."
            )
        if not self.model_name:
            raise ConfigError(
                f"LLMClient for model_role={self.model_role!r} requires a model name. "
                f"Set {model_var} in .env or ensure configs/models.yaml defines role {self.model_role!r}."
            )

    def _resolve_model_config(self, app_config: AppConfig, role: str):
        if role in app_config.models:
            return app_config.models[role]
        if role == "memory_manager" and "brain" in app_config.models:
            return app_config.models["brain"]
        return app_config.models.get("brain")

    def _resolve_api_key(self, app_config: AppConfig, role: str):
        if role == "brain":
            return app_config.env.brain_api_key
        if role == "memory_manager":
            return app_config.env.memory_manager_api_key or app_config.env.brain_api_key
        return None

    def _resolve_base_url(self, app_config: AppConfig, role: str) -> str | None:
        if role == "brain":
            return app_config.env.brain_base_url
        if role == "memory_manager":
            return app_config.env.memory_manager_base_url or app_config.env.brain_base_url
        return None

    def _resolve_model_name(self, app_config: AppConfig, role: str) -> str:
        if role == "brain":
            return app_config.env.brain_model or (self.config.model_name if self.config else "")
        if role == "memory_manager":
            return (
                app_config.env.memory_manager_model
                or app_config.env.brain_model
                or (self.config.model_name if self.config else "")
            )
        return self.config.model_name if self.config is not None else ""

    def _thinking_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        override = kwargs.get("thinking")
        if override is not None:
            if isinstance(override, dict):
                enabled = bool(override.get("enabled", override.get("type") == "enabled"))
                if not enabled:
                    return {}
                effort = str(override.get("reasoning_effort") or kwargs.get("reasoning_effort") or "high")
                return {"reasoning_effort": effort}
            if not bool(override):
                return {}
            return {"reasoning_effort": str(kwargs.get("reasoning_effort") or "high")}

        configured = self.config.thinking if self.config else {}
        if not bool(configured.get("enabled", False)):
            return {}
        return {"reasoning_effort": str(configured.get("reasoning_effort") or "high")}

    def _post_chat_completion(self, payload: dict[str, Any], stream: bool = False) -> dict[str, Any]:
        if stream:
            try:
                return self._stream_chat_completion(payload)
            except Exception as exc:  # noqa: BLE001
                log_model_stream_fallback(f"LLM:{self.model_role}", str(exc))

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
            log_retry=lambda attempt, error: log_model_retry(f"LLM:{self.model_role}", attempt, error),
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
        response = requests.post(url, headers=headers, json=stream_payload, timeout=90, stream=True)
        if response.status_code >= 400 and "response_format" in stream_payload:
            retry_payload = dict(stream_payload)
            retry_payload.pop("response_format", None)
            response.close()
            response = requests.post(url, headers=headers, json=retry_payload, timeout=90, stream=True)
        if not response.ok:
            raise RuntimeError(f"LLM stream HTTP {response.status_code}: {response.text[:500]}")

        content_parts: list[str] = []
        usage: dict[str, Any] = {}
        buffer = ""
        log_model_stream_start(f"LLM:{self.model_role}", self.model_name)
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
            log_model_stream_end(f"LLM:{self.model_role}")

        if not content_parts and buffer:
            raise RuntimeError("LLM stream ended with incomplete JSON chunks.")

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
