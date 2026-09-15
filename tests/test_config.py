from pathlib import Path

from geoagent.core.config import load_app_config, load_yaml_config
from geoagent.core.provider_policy import (
    available_map_providers,
    preferred_map_provider,
    supported_map_providers,
)
from geoagent.core.schemas import Hypothesis
from geoagent.state.task_state import GeoLocalizationState


def test_yaml_config_loads_tools():
    raw = load_yaml_config(Path("configs") / "tools.yaml")
    assert "tools" in raw
    assert raw["tools"]["ocr"]["enable"] is True


def test_app_config_merges_env_and_yaml(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("BRAIN_MODEL", "env-brain-model")
    monkeypatch.setenv("MEMORY_MANAGER_MODEL", "env-memory-model")
    config = load_app_config(config_dir="configs", env_file=None)
    assert config.models["brain"].model_name == "env-brain-model"
    assert config.models["memory_manager"].model_name == "env-memory-model"
    assert config.models["brain"].thinking["enabled"] is False
    assert config.log_level == "DEBUG"
    assert "web_search" in config.tools
    assert "webpage_read" in config.tools
    assert "knowledge_search" not in config.tools
    assert "place_details" not in config.tools
    assert config.env.web_search_provider == "serper"
    assert config.tools["ocr"].enable is True
    assert config.system["search_cache"]["path"] == "outputs/cache/web_cache/search_cache.sqlite3"


def test_baidu_maps_env_loads(monkeypatch):
    monkeypatch.setenv("BAIDU_MAPS_API_KEY", "baidu-test-key")
    config = load_app_config(config_dir="configs", env_file=None)
    assert config.env.baidu_maps_api_key is not None
    assert config.env.baidu_maps_api_key.get_secret_value() == "baidu-test-key"


def test_google_maps_url_signing_secret_loads(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_URL_SIGNING_SECRET", "google-signing-secret")
    config = load_app_config(config_dir="configs", env_file=None)
    assert config.env.google_maps_url_signing_secret is not None
    assert config.env.google_maps_url_signing_secret.get_secret_value() == "google-signing-secret"


def test_provider_policy_is_loaded_from_workflow_config(monkeypatch):
    monkeypatch.setenv("BAIDU_MAPS_API_KEY", "baidu-test-key")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "google-test-key")
    monkeypatch.setenv("SERPER_API_KEY", "serper-test-key")
    config = load_app_config(config_dir="configs", env_file=None)
    state = GeoLocalizationState(image_path="demo.jpg")
    state.hypotheses.append(Hypothesis(name="Wuhan", country="China", region="Wuhan", score=0.5))

    assert preferred_map_provider(config, state) == "baidu"
    # supported_map_providers reports the full policy set (unchanged).
    assert supported_map_providers(config, "poi_search") == {"baidu", "google"}
    assert supported_map_providers(config, "map_verification") == {"baidu", "google"}
    # available_map_providers filters out providers without a configured key.
    assert available_map_providers(config, "poi_search") == {"baidu", "google"}


def test_preferred_map_provider_falls_back_when_china_provider_has_no_key(monkeypatch):
    # Only google key configured; baidu key absent.
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "google-test-key")
    config = load_app_config(config_dir="configs", env_file=None)
    state = GeoLocalizationState(image_path="demo.jpg")
    state.hypotheses.append(Hypothesis(name="Wuhan", country="China", region="Wuhan", score=0.5))

    # No China provider has a key -> fall back to default (google).
    assert preferred_map_provider(config, state) == "google"


def test_map_verification_keeps_baidu_without_key(monkeypatch):
    """Baidu street-view/tile verification uses public endpoints (no key), so it
    must stay available for map_verification even without BAIDU_MAPS_API_KEY."""

    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "google-test-key")
    monkeypatch.setenv("SERPER_API_KEY", "serper-test-key")
    # No baidu key.
    config = load_app_config(config_dir="configs", env_file=None)

    # map_verification: baidu is keyless -> it stays available.
    assert available_map_providers(config, "map_verification") == {"baidu", "google"}
    # poi_search: baidu needs a key -> filtered out.
    assert available_map_providers(config, "poi_search") == {"google"}

    state = GeoLocalizationState(image_path="demo.jpg")
    state.hypotheses.append(Hypothesis(name="Wuhan", country="China", region="Wuhan", score=0.5))
    # China context: map_verification prefers baidu (keyless), not google.
    assert preferred_map_provider(config, state, capability="map_verification") == "baidu"
    # POI semantics (no capability): baidu has no key -> fall back to google.
    assert preferred_map_provider(config, state) == "google"


def test_batch_env_loads(monkeypatch):
    monkeypatch.setenv("BATCH_PROVIDER", "openai_compatible")
    monkeypatch.setenv("BATCH_API_KEY", "generic-key")
    monkeypatch.setenv("BATCH_BASE_URL", "https://batch.example/v1")
    monkeypatch.setenv("BATCH_MODEL", "generic-vlm")
    monkeypatch.setenv("BATCH_ENABLE_THINKING", "false")
    config = load_app_config(config_dir="configs", env_file=None)

    assert config.env.batch_provider == "openai_compatible"
    assert config.env.batch_api_key is not None
    assert config.env.batch_api_key.get_secret_value() == "generic-key"
    assert config.env.batch_base_url == "https://batch.example/v1"
    assert config.env.batch_model == "generic-vlm"
    assert config.env.batch_enable_thinking is False


def test_concurrent_eval_env_loads(monkeypatch):
    monkeypatch.setenv("EVAL_API_KEY", "eval-key")
    monkeypatch.setenv("EVAL_MODEL_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("EVAL_MODEL_NAME", "vendor/eval-model")
    monkeypatch.setenv("EVAL_PROVIDER", "z-ai")
    config = load_app_config(config_dir="configs", env_file=None)

    assert config.env.eval_api_key is not None
    assert config.env.eval_api_key.get_secret_value() == "eval-key"
    assert config.env.eval_model_url == "https://openrouter.example/v1"
    assert config.env.eval_model_name == "vendor/eval-model"
    assert config.env.eval_provider == "z-ai"
