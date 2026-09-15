"""Shared provider selection policy helpers."""

from __future__ import annotations

from typing import Any

from geoagent.core.config import AppConfig, WorkflowConfig
from geoagent.state.task_state import GeoLocalizationState


DEFAULT_PROVIDER_POLICY: dict[str, Any] = {
    "default_map_provider": "google",
    "mainland_china_map_provider": "baidu",
    "mainland_china_aliases": [
        "china",
        "prc",
        "mainland china",
        "people's republic of china",
        "cn",
        "\u4e2d\u56fd",
        "\u4e2d\u570b",
    ],
    "mainland_china_provider_prefixes": ["baidu"],
    "supported_map_providers": {
        "poi_search": ["google", "baidu"],
        "map_verification": ["google", "baidu"],
    },
    "provider_labels": {
        "google": "Google",
        "google_maps": "Google",
        "serper_places": "Google/Serper",
        "baidu": "Baidu",
        "baidu_maps": "Baidu",
    },
}


def workflow_policy(
    app_config: AppConfig,
    policy_name: str,
    workflow_config: WorkflowConfig | None = None,
    workflow_name: str = "react",
) -> dict[str, Any]:
    workflow = workflow_config or app_config.workflows.get(workflow_name)
    policy = workflow.extra.get(policy_name) if workflow else None
    return policy if isinstance(policy, dict) else {}


def provider_policy(
    app_config: AppConfig,
    workflow_config: WorkflowConfig | None = None,
    workflow_name: str = "react",
) -> dict[str, Any]:
    configured = workflow_policy(app_config, "provider_policy", workflow_config, workflow_name)
    return _deep_merge(DEFAULT_PROVIDER_POLICY, configured)


def is_mainland_china_context(
    app_config: AppConfig,
    state: GeoLocalizationState,
    args: dict[str, Any] | None = None,
    workflow_config: WorkflowConfig | None = None,
) -> bool:
    policy = provider_policy(app_config, workflow_config)
    aliases = {_normalize_country_alias(item) for item in policy.get("mainland_china_aliases") or []}
    values: list[str] = []
    if args:
        values.append(str(args.get("country") or ""))
    values.extend(str(hypothesis.country or "") for hypothesis in state.hypotheses[:3])
    normalized = {_normalize_country_alias(value) for value in values if str(value).strip()}
    return bool(normalized & aliases)


# Providers that work WITHOUT an API key for a given capability, because their
# implementation uses public web endpoints. Baidu street-view/tile verification
# (map_verification) reads no key — it must stay available in mainland China
# even when BAIDU_MAPS_API_KEY is unset. POI search/details still need keys.
KEYLESS_PROVIDERS: dict[str, set[str]] = {
    # baidu map/streetview tiles use public web endpoints — no key needed.
    "map_verification": {"baidu", "google"},
}


def _is_keyless(capability: str, provider: str) -> bool:
    return str(provider).strip().lower() in KEYLESS_PROVIDERS.get(capability, set())


def preferred_map_provider(
    app_config: AppConfig,
    state: GeoLocalizationState,
    args: dict[str, Any] | None = None,
    workflow_config: WorkflowConfig | None = None,
    capability: str | None = None,
) -> str:
    policy = provider_policy(app_config, workflow_config)
    if is_mainland_china_context(app_config, state, args, workflow_config):
        candidate = str(policy.get("mainland_china_map_provider") or "baidu").lower()
        if _provider_available(app_config, candidate, capability):
            return candidate
        # The configured China provider is unavailable; fall back to the
        # default provider.
    default = str(policy.get("default_map_provider") or "google").lower()
    return default


def supported_map_providers(
    app_config: AppConfig,
    capability: str,
    workflow_config: WorkflowConfig | None = None,
) -> set[str]:
    policy = provider_policy(app_config, workflow_config)
    supported = policy.get("supported_map_providers") or {}
    values = supported.get(capability) if isinstance(supported, dict) else None
    if not isinstance(values, list):
        values = DEFAULT_PROVIDER_POLICY["supported_map_providers"].get(capability, [])
    return {str(value).strip().lower() for value in values if str(value).strip()}


def available_map_providers(
    app_config: AppConfig,
    capability: str,
    workflow_config: WorkflowConfig | None = None,
) -> set[str]:
    """Providers that are both policy-supported AND usable given key config.

    A provider is usable if it has an API key configured, OR it is keyless for
    this capability (e.g. baidu for map_verification, which uses public web
    endpoints and reads no key). Use this (not supported_map_providers) when
    exposing provider choices to the Brain or validating tool requests.
    """

    supported = supported_map_providers(app_config, capability, workflow_config)
    return {
        provider
        for provider in supported
        if _provider_available(app_config, provider, capability)
    }


def _provider_available(app_config: AppConfig, provider: str, capability: str | None) -> bool:
    normalized = str(provider or "").strip().lower()
    if not normalized:
        return False
    if capability and _is_keyless(capability, normalized):
        return True
    return _has_provider_key(app_config, normalized, capability)


def _has_provider_key(app_config: AppConfig, provider: str, capability: str | None = None) -> bool:
    normalized = str(provider or "").strip().lower()
    env = app_config.env
    if normalized == "google":
        if capability == "poi_search":
            return bool(env.serper_api_key)
        return bool(env.google_maps_api_key)
    if normalized == "baidu":
        return bool(env.baidu_maps_api_key)
    return False


def provider_implies_mainland_china(
    app_config: AppConfig,
    provider: str,
    workflow_config: WorkflowConfig | None = None,
) -> bool:
    policy = provider_policy(app_config, workflow_config)
    prefixes = policy.get("mainland_china_provider_prefixes") or []
    normalized = str(provider or "").strip().lower()
    return any(normalized.startswith(str(prefix).strip().lower()) for prefix in prefixes if str(prefix).strip())


def provider_display_label(
    app_config: AppConfig,
    provider: str,
    workflow_config: WorkflowConfig | None = None,
) -> str:
    policy = provider_policy(app_config, workflow_config)
    labels = policy.get("provider_labels") or {}
    normalized = str(provider or "").strip().lower()
    if isinstance(labels, dict):
        for prefix, label in labels.items():
            if normalized.startswith(str(prefix).strip().lower()):
                return str(label)
    return normalized


def _normalize_country_alias(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
