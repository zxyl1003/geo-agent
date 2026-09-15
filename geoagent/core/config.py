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
    "LOG_LEVEL",
)


class EnvConfig(BaseModel):
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


def load_env(env_file: str | Path | None = ".env") -> EnvConfig:
    """Load sensitive values from `.env` and process environment."""

    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=False)


    return EnvConfig(
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
