"""Tests for the unified geocode / reverse_geocode provider-routing facades."""

import requests
from pydantic import SecretStr

from geoagent.core.config import load_app_config
from geoagent.core.schemas import ToolResult
from geoagent.tools.registry import build_tools


class _FakeResponse:
    def __init__(self, payload, ok=True, status_code=200):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code

    def json(self):
        return self._payload

    @property
    def text(self):
        return str(self._payload)


def _config():
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.locationiq_api_key = SecretStr("locationiq-test-key")
    config.env.google_maps_api_key = SecretStr("google-test-key")
    config.env.baidu_maps_api_key = SecretStr("baidu-test-key")
    config.system["search_cache"]["enabled"] = False
    return config


def test_geocode_routes_china_address_to_baidu(monkeypatch):
    config = _config()
    tools = build_tools(config)
    seen = {}

    def fake_get(url, params, timeout):
        seen["url"] = url
        seen["params"] = params
        return _FakeResponse(
            {
                "status": 0,
                "result": {
                    "location": {"lng": 116.3975, "lat": 39.9087},
                    "precise": 1,
                    "confidence": 90,
                    "level": "tourist_attraction",
                },
            }
        )

    monkeypatch.setattr("geoagent.tools.search.baidu_maps_geocode_tool.requests.get", fake_get)
    result = tools["geocode"].run(address="天安门广场", country="China", top_k=3)

    assert result.success is True
    assert seen["url"] == "https://api.map.baidu.com/geocoding/v3/"
    assert result.data["provider"] == "baidu_maps"
    assert result.data["provider_display_name"] == "Baidu Maps"
    assert result.data["attempts"] == ["baidu_maps"]
    assert result.data["routed"] is True
    assert result.data["top_result"]["lat"] is not None


def test_geocode_routes_international_to_locationiq(monkeypatch):
    config = _config()
    tools = build_tools(config)
    seen = {}

    def fake_get(url, params, timeout):
        seen["url"] = url
        return _FakeResponse(
            [
                {
                    "lat": "35.6586",
                    "lon": "139.7454",
                    "display_name": "Tokyo Tower, 4 Chome-2-8 Shibakoen, Minato City, Tokyo, Japan",
                    "type": "tourism",
                    "class": "tourism",
                    "importance": 0.5,
                    "address": {"city": "Tokyo", "country": "Japan"},
                }
            ]
        )

    monkeypatch.setattr("geoagent.tools.search.locationiq_geocode_tool.requests.get", fake_get)
    result = tools["geocode"].run(address="Tokyo Tower", country="Japan", top_k=3)

    assert result.success is True
    assert seen["url"].endswith("/v1/search")
    assert result.data["provider"] == "locationiq"
    assert result.data["provider_display_name"] == "LocationIQ"
    assert result.data["attempts"] == ["locationiq"]
    assert result.data["top_result"]["lat"] == 35.6586
    assert result.data["top_result"]["country"] == "Japan"


def test_geocode_does_not_fall_back_when_locationiq_empty(monkeypatch):
    config = _config()
    tools = build_tools(config)
    seen = []

    def fake_locationiq_get(url, params, timeout):
        seen.append(url)
        return _FakeResponse([])

    monkeypatch.setattr("geoagent.tools.search.locationiq_geocode_tool.requests.get", fake_locationiq_get)
    result = tools["geocode"].run(address="Tokyo Tower", country="Japan", top_k=3)

    assert result.success is True
    assert seen == [config.env.locationiq_forward_url]
    assert result.data["provider"] == "locationiq"
    assert result.data["attempts"] == ["locationiq"]
    assert result.data["candidates"] == []


def test_locationiq_retries_network_errors_without_exposing_key(monkeypatch):
    config = _config()
    tools = build_tools(config)
    calls = 0

    def failing_get(url, params, timeout):
        nonlocal calls
        calls += 1
        raise requests.exceptions.SSLError(
            f"connection failed for {url}?key={params['key']}"
        )

    monkeypatch.setattr("geoagent.tools.search.locationiq_geocode_tool.requests.get", failing_get)
    result = tools["locationiq_geocode"].run(address="Tokyo Tower")

    assert result.success is False
    assert calls == 4
    assert "locationiq-test-key" not in result.error
    assert "after 4 attempts (SSLError)" in result.error


def test_reverse_geocode_routes_international_to_locationiq(monkeypatch):
    config = _config()
    tools = build_tools(config)
    seen = {}

    def fake_get(url, params, timeout):
        seen["url"] = url
        return _FakeResponse(
            {
                "place_id": "liq_1",
                "lat": "35.6586",
                "lon": "139.7454",
                "display_name": "Tokyo Tower, Minato City, Tokyo, Japan",
                "type": "tourism",
                "address": {"city": "Tokyo", "country": "Japan"},
            }
        )

    monkeypatch.setattr("geoagent.tools.search.locationiq_geocode_tool.requests.get", fake_get)
    result = tools["reverse_geocode"].run(lat=35.6586, lon=139.7454)

    assert result.success is True
    assert seen["url"].endswith("/v1/reverse")
    assert result.data["provider"] == "locationiq"
    assert result.data["attempts"] == ["locationiq"]
    assert result.data["formatted_address"] == "Tokyo Tower, Minato City, Tokyo, Japan"


def test_geocode_no_provider_configured_returns_clear_error(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["search_cache"]["enabled"] = False
    tools = build_tools(config)

    # No LOCATIONIQ/BAIDU keys are configured (env_file=None).
    result = tools["geocode"].run(address="Some Street", country="Japan")

    assert result.success is False
    assert "LOCATIONIQ" in result.error or "BAIDU" in result.error
    assert isinstance(result.data.get("attempts"), list)


def test_locationiq_tool_forward_geocodes(monkeypatch):
    config = _config()
    tools = build_tools(config)
    seen = {}

    def fake_get(url, params, timeout):
        seen["url"] = url
        seen["params"] = params
        return _FakeResponse(
            [
                {
                    "lat": "35.6586",
                    "lon": "139.7454",
                    "display_name": "Tokyo Tower, Tokyo, Japan",
                    "type": "attraction",
                    "class": "tourism",
                    "address": {"city": "Tokyo", "country": "Japan"},
                }
            ]
        )

    monkeypatch.setattr("geoagent.tools.search.locationiq_geocode_tool.requests.get", fake_get)
    result = tools["locationiq_geocode"].run(address="Tokyo Tower", top_k=3)

    assert result.success is True
    assert seen["params"]["key"] == "locationiq-test-key"
    assert result.data["provider"] == "locationiq"
    assert result.data["top_result"]["lat"] == 35.6586
    assert result.data["top_result"]["granularity"] == "street"


def test_locationiq_hidden_from_brain():
    config = _config()
    tools = build_tools(config)
    assert tools["locationiq_geocode"].hidden is True
    assert tools["baidu_maps_geocode"].hidden is True
    assert tools["google_maps_geocode"].hidden is True
    assert tools["geocode"].hidden is False
    assert tools["reverse_geocode"].hidden is False
