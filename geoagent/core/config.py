"""Configuration loading for environment variables and YAML files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

from geoagent.core.exceptions import ConfigError


SENSITIVE_ENV_KEYS = (
    "EVAL_API_KEY",
    "EVAL_MODEL_URL",
    "EVAL_MODEL_NAME",
    "EVAL_PROVIDER",
    "BRAIN_API_KEY",
    "BRAIN_BASE_URL",
    "BRAIN_MODEL",
    "MEMORY_MANAGER_API_KEY",
    "MEMORY_MANAGER_BASE_URL",
    "MEMORY_MANAGER_MODEL",
    "WEB_SEARCH_PROVIDER",
    "SERPER_API_KEY",
    "SERPER_SEARCH_BASE_URL",
    "SERPER_PLACES_BASE_URL",
    "GOOGLE_MAPS_API_KEY",
    "GOOGLE_MAPS_URL_SIGNING_SECRET",
    "BAIDU_MAPS_API_KEY",
    "LOCATIONIQ_KEY",
    "FORWARD_GEOCODING_URL",
    "REVERSE_GEOCODING_URL",
    "BATCH_PROVIDER",
    "BATCH_API_KEY",
    "BATCH_BASE_URL",
    "BATCH_MODEL",
    "BATCH_ENDPOINT",
    "BATCH_FILES_PATH",
    "BATCH_BATCHES_PATH",
    "BATCH_AUTH_HEADER",
    "BATCH_AUTH_SCHEME",
    "BATCH_COMPLETION_WINDOW",
    "BATCH_POLL_INTERVAL_SECONDS",
    "BATCH_REQUEST_TIMEOUT_SECONDS",
    "BATCH_MAX_REQUESTS_PER_FILE",
    "BATCH_MAX_FILE_MB",
    "BATCH_MAX_LINE_MB",
    "BATCH_IMAGE_MODE",
    "BATCH_IMAGE_BASE_URL",
    "BATCH_IMAGE_MAX_EDGE",
    "BATCH_IMAGE_JPEG_QUALITY",
    "BATCH_IMAGE_MIN_PIXELS",
    "BATCH_IMAGE_MAX_PIXELS",
    "BATCH_MAX_TOKENS",
    "BATCH_TEMPERATURE",
    "BATCH_ENABLE_THINKING",
    "BATCH_THINKING_BUDGET",
    "BATCH_RESPONSE_FORMAT",
    "BATCH_REQUEST_EXTRA_JSON",
    "LOG_LEVEL",
)


class EnvConfig(BaseModel):
    eval_api_key: SecretStr | None = Field(default=None, repr=False)
    eval_model_url: str = "https://openrouter.ai/api/v1"
    eval_model_name: str = "z-ai/glm-5.3-flash"
    eval_provider: str = ""
    brain_api_key: SecretStr | None = Field(default=None, repr=False)
    brain_base_url: str | None = None
    brain_model: str | None = None
    memory_manager_api_key: SecretStr | None = Field(default=None, repr=False)
    memory_manager_base_url: str | None = None
    memory_manager_model: str | None = None
    web_search_provider: str = "serper"
    serper_api_key: SecretStr | None = Field(default=None, repr=False)
    serper_search_base_url: str = "https://google.serper.dev/search"
    serper_places_base_url: str = "https://google.serper.dev/places"
    google_maps_api_key: SecretStr | None = Field(default=None, repr=False)
    google_maps_url_signing_secret: SecretStr | None = Field(default=None, repr=False)
    baidu_maps_api_key: SecretStr | None = Field(default=None, repr=False)
    locationiq_api_key: SecretStr | None = Field(default=None, repr=False)
    locationiq_forward_url: str = "https://us1.locationiq.com/v1/search"
    locationiq_reverse_url: str = "https://us1.locationiq.com/v1/reverse"
    batch_provider: str = "aliyun"
    batch_api_key: SecretStr | None = Field(default=None, repr=False)
    batch_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    batch_model: str = "qwen3-vl-plus"
    batch_endpoint: str = "/v1/chat/completions"
    batch_files_path: str = "/files"
    batch_batches_path: str = "/batches"
    batch_auth_header: str = "Authorization"
    batch_auth_scheme: str = "Bearer"
    batch_completion_window: str = "24h"
    batch_poll_interval_seconds: float = 30.0
    batch_request_timeout_seconds: float = 120.0
    batch_max_requests_per_file: int = 50_000
    batch_max_file_mb: float = 500.0
    batch_max_line_mb: float = 1.0
    batch_image_mode: str = "base64"
    batch_image_base_url: str | None = None
    batch_image_max_edge: int = 2048
    batch_image_jpeg_quality: int = 90
    batch_image_min_pixels: int | None = None
    batch_image_max_pixels: int | None = None
    batch_max_tokens: int = 1000
    batch_temperature: float = 0.0
    batch_enable_thinking: bool | None = None
    batch_thinking_budget: int | None = None
    batch_response_format: str | None = None
    batch_request_extra_json: str = "{}"
    log_level: str = "INFO"


class ToolConfig(BaseModel):
    name: str
    enable: bool = True
    class_path: str
    cost_level: str = "low"
    max_calls_per_task: int = 10
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str
    provider: str = "openai-compatible"
    model_name: str = ""
    temperature: float = 0.0
    max_tokens: int = 1000
    thinking: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class WorkflowConfig(BaseModel):
    name: str
    class_path: str
    max_steps: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    project_name: str = "geo_agent_system"
    config_dir: Path = Path("configs")
    log_level: str = "INFO"
    env: EnvConfig = Field(default_factory=EnvConfig, repr=False)
    tools: dict[str, ToolConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    workflows: dict[str, WorkflowConfig] = Field(default_factory=dict)
    agents: dict[str, dict[str, Any]] = Field(default_factory=dict)
    system: dict[str, Any] = Field(default_factory=dict)

    def get_tool(self, name: str) -> ToolConfig:
        return self.tools[name]

    def get_workflow(self, name: str) -> WorkflowConfig:
        return self.workflows[name]


def _secret(value: str | None) -> SecretStr | None:
    return SecretStr(value) if value else None


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true/false when configured.")


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number.") from exc


def load_env(env_file: str | Path | None = ".env") -> EnvConfig:
    """Load sensitive values from `.env` and process environment."""

    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=False)


    return EnvConfig(
        eval_api_key=_secret(os.getenv("EVAL_API_KEY")),
        eval_model_url=os.getenv("EVAL_MODEL_URL") or "https://openrouter.ai/api/v1",
        eval_model_name=os.getenv("EVAL_MODEL_NAME") or "z-ai/glm-5.3-flash",
        eval_provider=(os.getenv("EVAL_PROVIDER") or "").strip(),
        brain_api_key=_secret(os.getenv("BRAIN_API_KEY")),
        brain_base_url=os.getenv("BRAIN_BASE_URL") or None,
        brain_model=os.getenv("BRAIN_MODEL") or None,
        memory_manager_api_key=_secret(os.getenv("MEMORY_MANAGER_API_KEY")),
        memory_manager_base_url=os.getenv("MEMORY_MANAGER_BASE_URL") or None,
        memory_manager_model=os.getenv("MEMORY_MANAGER_MODEL") or None,
        web_search_provider=(os.getenv("WEB_SEARCH_PROVIDER") or "serper").strip().lower(),
        serper_api_key=_secret(os.getenv("SERPER_API_KEY")),
        serper_search_base_url=os.getenv("SERPER_SEARCH_BASE_URL") or "https://google.serper.dev/search",
        serper_places_base_url=os.getenv("SERPER_PLACES_BASE_URL") or "https://google.serper.dev/places",
        google_maps_api_key=_secret(os.getenv("GOOGLE_MAPS_API_KEY")),
        google_maps_url_signing_secret=_secret(os.getenv("GOOGLE_MAPS_URL_SIGNING_SECRET")),
        baidu_maps_api_key=_secret(os.getenv("BAIDU_MAPS_API_KEY")),
        locationiq_api_key=_secret(os.getenv("LOCATIONIQ_KEY")),
        locationiq_forward_url=os.getenv("FORWARD_GEOCODING_URL") or "https://us1.locationiq.com/v1/search",
        locationiq_reverse_url=os.getenv("REVERSE_GEOCODING_URL") or "https://us1.locationiq.com/v1/reverse",
        batch_provider=(os.getenv("BATCH_PROVIDER") or "aliyun").strip().lower(),
        batch_api_key=_secret(os.getenv("BATCH_API_KEY")),
        batch_base_url=os.getenv("BATCH_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        batch_model=os.getenv("BATCH_MODEL") or "qwen3-vl-plus",
        batch_endpoint=os.getenv("BATCH_ENDPOINT") or "/v1/chat/completions",
        batch_files_path=os.getenv("BATCH_FILES_PATH") or "/files",
        batch_batches_path=os.getenv("BATCH_BATCHES_PATH") or "/batches",
        batch_auth_header=os.getenv("BATCH_AUTH_HEADER") or "Authorization",
        batch_auth_scheme=os.getenv("BATCH_AUTH_SCHEME", "Bearer"),
        batch_completion_window=os.getenv("BATCH_COMPLETION_WINDOW") or "24h",
        batch_poll_interval_seconds=_env_float("BATCH_POLL_INTERVAL_SECONDS", 30.0),
        batch_request_timeout_seconds=_env_float("BATCH_REQUEST_TIMEOUT_SECONDS", 120.0),
        batch_max_requests_per_file=_env_int("BATCH_MAX_REQUESTS_PER_FILE", 50_000),
        batch_max_file_mb=_env_float("BATCH_MAX_FILE_MB", 500.0),
        batch_max_line_mb=_env_float("BATCH_MAX_LINE_MB", 1.0),
        batch_image_mode=(os.getenv("BATCH_IMAGE_MODE") or "base64").strip().lower(),
        batch_image_base_url=os.getenv("BATCH_IMAGE_BASE_URL") or None,
        batch_image_max_edge=_env_int("BATCH_IMAGE_MAX_EDGE", 2048) or 0,
        batch_image_jpeg_quality=_env_int("BATCH_IMAGE_JPEG_QUALITY", 90),
        batch_image_min_pixels=_env_int("BATCH_IMAGE_MIN_PIXELS", None),
        batch_image_max_pixels=_env_int("BATCH_IMAGE_MAX_PIXELS", None),
        batch_max_tokens=_env_int("BATCH_MAX_TOKENS", 1000),
        batch_temperature=_env_float("BATCH_TEMPERATURE", 0.0),
        batch_enable_thinking=_env_bool("BATCH_ENABLE_THINKING"),
        batch_thinking_budget=_env_int("BATCH_THINKING_BUDGET", None),
        batch_response_format=os.getenv("BATCH_RESPONSE_FORMAT") or None,
        batch_request_extra_json=os.getenv("BATCH_REQUEST_EXTRA_JSON") or "{}",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict for empty files."""

    yaml_path = Path(path)
    if not yaml_path.exists():
        raise ConfigError(f"YAML config does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"YAML config must contain a mapping: {yaml_path}")
    return loaded


def _collect_extra(raw: dict[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if key not in known}


def load_app_config(
    config_dir: str | Path = "configs",
    env_file: str | Path | None = ".env",
) -> AppConfig:
    """Merge sensitive environment config with non-sensitive YAML config."""

    config_path = Path(config_dir)
    env = load_env(env_file)

    system = load_yaml_config(config_path / "system.yaml")
    models_raw = load_yaml_config(config_path / "models.yaml").get("models", {})
    tools_raw = load_yaml_config(config_path / "tools.yaml").get("tools", {})
    workflows_raw = load_yaml_config(config_path / "workflows.yaml").get("workflows", {})
    agents_raw = load_yaml_config(config_path / "agents.yaml").get("agents", {})
    tools: dict[str, ToolConfig] = {}
    for name, raw in tools_raw.items():
        normalized = dict(raw or {})
        known = {"enable", "class_path", "cost_level", "max_calls_per_task"}
        tools[name] = ToolConfig(name=name, extra=_collect_extra(raw, known), **normalized)

    models: dict[str, ModelConfig] = {}
    for name, raw in models_raw.items():
        known = {"provider", "model_name", "temperature", "max_tokens", "thinking"}
        model_name = raw.get("model_name", "")
        if name == "brain" and env.brain_model:
            model_name = env.brain_model
        if name == "memory_manager" and env.memory_manager_model:
            model_name = env.memory_manager_model
        models[name] = ModelConfig(
            name=name,
            provider=raw.get("provider", "openai-compatible"),
            model_name=model_name,
            temperature=raw.get("temperature", 0.0),
            max_tokens=raw.get("max_tokens", 1000),
            thinking=raw.get("thinking") or {},
            extra=_collect_extra(raw, known),
        )

    workflows: dict[str, WorkflowConfig] = {}
    for name, raw in workflows_raw.items():
        known = {
            "class_path",
            "max_steps",
        }
        workflows[name] = WorkflowConfig(name=name, extra=_collect_extra(raw, known), **raw)

    project = system.get("project", {})
    log_level = env.log_level or system.get("logging", {}).get("level", "INFO")

    return AppConfig(
        project_name=project.get("name", "geo_agent_system"),
        config_dir=config_path,
        log_level=log_level,
        env=env,
        tools=tools,
        models=models,
        workflows=workflows,
        agents=agents_raw,
        system=system,
    )
