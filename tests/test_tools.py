import pytest
import pycountry
from pydantic import SecretStr
from PIL import Image

from geoagent.core.coordinates import (
    baidu_tile_position_from_wgs84,
    baidu_tile_from_wgs84,
    bd09_to_wgs84,
    gcj02_to_wgs84,
    google_xyz_from_wgs84,
    normalize_coordinate_fields,
    wgs84_to_bd09,
)
from geoagent.core.config import load_app_config
from geoagent.core.retrieval_cache import RetrievalCache
from geoagent.tools.registry import build_tools


class FakeResponse:
    def __init__(
        self,
        data,
        ok=True,
        status_code=200,
        text='{"status": 0}',
        content=b"",
        content_type="application/json",
    ):
        self._data = data
        self.ok = ok
        self.status_code = status_code
        self.text = text
        self.content = content
        self.headers = {"Content-Type": content_type}

    def json(self):
        return self._data


class FakeStreamResponse(FakeResponse):
    def __init__(self, body: bytes, *, content_type: str = "text/html; charset=utf-8"):
        super().__init__({}, ok=True, status_code=200, text=body.decode("utf-8"))
        self.body = body
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
        self.encoding = "utf-8"

    def iter_content(self, chunk_size=65536):
        yield self.body

    def close(self):
        return None


def test_retrieval_cache_failure_degrades_to_miss(tmp_path):
    directory_path = tmp_path / "not-a-database"
    directory_path.mkdir()
    cache = RetrievalCache(directory_path)

    assert cache.get("web_search", "serper", {"query": "x"}) is None
    cache.put("web_search", "serper", {"query": "x"}, {"results": []})


def test_disabled_tool_is_not_built():
    config = load_app_config(config_dir="configs", env_file=None)
    config.tools["ocr"].enable = False
    tools = build_tools(config)
    assert "ocr" not in tools


def test_crop_tool_is_not_available_by_default():
    config = load_app_config(config_dir="configs", env_file=None)
    tools = build_tools(config)

    assert "crop" not in tools
    assert "zoom" not in tools


def test_coordinate_helpers_roundtrip_china_point():
    lon, lat = 116.3975, 39.9087
    bd_lon, bd_lat = wgs84_to_bd09(lon, lat)
    wgs_lon, wgs_lat = bd09_to_wgs84(bd_lon, bd_lat)
    assert abs(wgs_lon - lon) < 0.0002
    assert abs(wgs_lat - lat) < 0.0002
    assert google_xyz_from_wgs84(lon, lat, 16) == google_xyz_from_wgs84(lon, lat, 16)


def test_coordinate_field_normalization_preserves_raw_provider_coordinates():
    raw_lon, raw_lat = 116.3975, 39.9087
    expected_lon, expected_lat = gcj02_to_wgs84(raw_lon, raw_lat)
    fields = normalize_coordinate_fields(raw_lat, raw_lon, "gcj02ll")

    assert fields["coordinate_system"] == "WGS84"
    assert fields["raw_coordinate_system"] == "GCJ-02"
    assert fields["raw_lat"] == raw_lat
    assert fields["raw_lon"] == raw_lon
    assert abs(fields["lat"] - expected_lat) < 0.00001
    assert abs(fields["lon"] - expected_lon) < 0.00001


def test_map_tile_verify_rejects_osm_provider():
    config = load_app_config(config_dir="configs", env_file=None)
    tools = build_tools(config)
    result = tools["map_tile_verify"].run(map_provider="osm")

    assert result.success is False
    assert result.data["supported_map_providers"] == ["baidu", "google"]


def test_baidu_roadmap_tile_url_uses_supported_vector_tile_endpoint():
    config = load_app_config(config_dir="configs", env_file=None)
    tools = build_tools(config)

    url = tools["map_tile_verify"]._tile_url("baidu", "roadmap", 92669, 26744, 19)

    assert url.startswith("http://online")
    assert "https://online" not in url
    assert "/tile/" in url
    assert "qt=vtile" in url


def test_baidu_tiles_are_arranged_north_to_south(monkeypatch, tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    tool = build_tools(config)["map_tile_verify"]
    monkeypatch.setattr(
        "geoagent.tools.maps.map_tile_verify_tool.download_image",
        lambda url, path: path,
    )
    lat = 45.70039261702186
    lon = 126.62555323141262
    center_x, center_y = baidu_tile_from_wgs84(lon, lat, 19)

    tiles = tool._retrieve_tiles("baidu", "roadmap", lon, lat, 19, 1, tmp_path)
    center_column = [tile for tile in tiles if tile["x"] == center_x]

    assert [(tile["grid_y"], tile["y"]) for tile in center_column] == [
        (0, center_y + 1),
        (1, center_y),
        (2, center_y - 1),
    ]


def test_baidu_candidate_pixel_uses_bd09mc_tile_fraction():
    tile_x, tile_y, pixel_x, pixel_y = baidu_tile_position_from_wgs84(
        126.62555323141262,
        45.70039261702186,
        18,
    )

    assert (tile_x, tile_y) == (55068, 22276)
    assert pixel_x == pytest.approx(20.3358, abs=0.01)
    assert pixel_y == pytest.approx(61.04, abs=0.01)


def test_map_tile_verify_uses_roadmap_zoom_range_without_changing_satellite(monkeypatch, tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    tool = build_tools(config)["map_tile_verify"]
    seen_zooms = []

    def fake_retrieve(provider, map_type, lon, lat, zoom, radius, tile_dir):
        seen_zooms.append((provider, map_type, zoom))
        return [{"grid_x": 0, "grid_y": 0, "path": "unused"}]

    monkeypatch.setattr(tool, "_retrieve_tiles", fake_retrieve)
    monkeypatch.setattr("geoagent.tools.maps.map_tile_verify_tool.cache_dir_for", lambda *args: tmp_path)
    monkeypatch.setattr("geoagent.tools.maps.map_tile_verify_tool.compose_tile_grid", lambda *args, **kwargs: tmp_path / "mosaic.png")
    monkeypatch.setattr(
        "geoagent.tools.maps.map_tile_verify_tool.compare_with_vlm",
        lambda **kwargs: {"verdict": "inconclusive", "confidence": 0.0},
    )

    default_roadmap = tool.run(map_provider="baidu", map_type="roadmap", lat=45.7, lon=126.6)
    baidu_roadmap = tool.run(map_provider="baidu", map_type="roadmap", lat=45.7, lon=126.6, zoom=21)
    google_roadmap = tool.run(map_provider="google", map_type="roadmap", lat=45.7, lon=126.6, zoom=21)
    satellite = tool.run(map_provider="baidu", map_type="satellite", lat=45.7, lon=126.6, zoom=21)

    assert default_roadmap.data["zoom"] == 19
    assert baidu_roadmap.data["zoom"] == 20
    assert google_roadmap.data["zoom"] == 20
    assert satellite.data["zoom"] == 20
    assert seen_zooms == [
        ("baidu", "roadmap", 19),
        ("baidu", "roadmap", 20),
        ("google", "roadmap", 20),
        ("baidu", "satellite", 20),
    ]


def test_baidu_streetview_uses_official_panorama_static_api(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.baidu_maps_api_key = SecretStr("baidu-test-key")
    tools = build_tools(config)
    seen_urls = []

    def fake_get(url, timeout):
        seen_urls.append(url)
        return FakeResponse({}, content=b"image", content_type="image/jpeg")

    monkeypatch.setattr("geoagent.tools.maps.streetview_verify_tool.requests.get", fake_get)
    lat = 39.9087
    lon = 116.3975
    metadata = tools["streetview_verify"]._fetch_baidu_metadata(lat=lat, lon=lon)

    assert metadata["available"] is True
    assert metadata["coordinate_system"] == "WGS84"
    assert metadata["provider"] == "baidu_panorama_static_api"
    assert seen_urls[0].startswith("https://api.map.baidu.com/panorama/v2?")
    assert "ak=baidu-test-key" in seen_urls[0]
    assert "coordtype=wgs84ll" in seen_urls[0]
    assert "location=116.3975%2C39.9087" in seen_urls[0]
    assert "mapsv0.bdimg.com" not in seen_urls[0]


def test_google_streetview_image_url_uses_official_static_api():
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.google_maps_api_key = SecretStr("google-test-key")
    config.env.google_maps_url_signing_secret = None
    tools = build_tools(config)

    url = tools["streetview_verify"]._streetview_image_url(
        provider="google",
        metadata={"pano_id": "test_pano_id"},
        heading=90,
        pitch=0,
        fov=100,
    )

    assert url.startswith("https://maps.googleapis.com/maps/api/streetview?")
    assert "pano=test_pano_id" in url
    assert "heading=90" in url
    assert "key=google-test-key" in url
    assert "return_error_code=true" in url


def test_google_streetview_metadata_uses_official_metadata_api(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.google_maps_api_key = SecretStr("google-test-key")
    config.env.google_maps_url_signing_secret = None
    tools = build_tools(config)
    seen = {}

    def fake_get(url, timeout):
        seen["url"] = url
        return FakeResponse(
            {
                "status": "OK",
                "pano_id": "google_panoid_123",
                "location": {"lat": 35.6586, "lng": 139.7454},
                "date": "2025-01",
            }
        )

    monkeypatch.setattr("geoagent.tools.maps.streetview_verify_tool.requests.get", fake_get)
    metadata = tools["streetview_verify"]._fetch_google_metadata(35.65858, 139.74543, 80)

    assert seen["url"].startswith("https://maps.googleapis.com/maps/api/streetview/metadata?")
    assert "location=35.65858%2C139.74543" in seen["url"]
    assert "radius=80" in seen["url"]
    assert "key=google-test-key" in seen["url"]
    assert metadata["provider"] == "google_streetview_static_api"
    assert metadata["available"] is True
    assert metadata["pano_id"] == "google_panoid_123"
    assert metadata["coordinate_system"] == "WGS84"


def test_visual_reanalysis_accepts_landmark_observed_entity():
    config = load_app_config(config_dir="configs", env_file=None)
    tools = build_tools(config)
    entities = tools["visual_reanalysis"]._observed_entities_from_analysis(
        {
            "observed_entities": [
                {
                    "id": "ent_main_landmark",
                    "entity_type": "landmark",
                    "name": "Chiang Kai-shek Memorial Hall",
                    "text_items": ["中"],
                    "region_hint": "central building facade",
                    "confidence": 0.84,
                }
            ]
        }
    )

    assert len(entities) == 1
    assert entities[0].entity_type == "landmark"
    assert entities[0].name == "Chiang Kai-shek Memorial Hall"
    assert entities[0].text_items == ["中"]


def test_baidu_maps_geocode_tool_geocodes_address(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.baidu_maps_api_key = SecretStr("baidu-test-key")
    tools = build_tools(config)
    seen = {}

    def fake_get(url, params, timeout):
        seen["url"] = url
        seen["params"] = params
        return FakeResponse(
            {
                "status": 0,
                "result": {
                    "location": {"lng": 116.3975, "lat": 39.9087},
                    "precise": 1,
                    "confidence": 90,
                    "comprehension": 90,
                    "level": "tourist_attraction",
                },
            }
        )

    monkeypatch.setattr("geoagent.tools.search.baidu_maps_geocode_tool.requests.get", fake_get)
    result = tools["baidu_maps_geocode"].run(address="Tiananmen, Dongcheng District", city="Beijing", top_k=1)

    assert result.success is True
    assert seen["url"] == "https://api.map.baidu.com/geocoding/v3/"
    assert seen["params"]["ak"] == "baidu-test-key"
    assert seen["params"]["address"] == "Tiananmen, Dongcheng District"
    assert seen["params"]["city"] == "Beijing"
    assert seen["params"]["output"] == "json"
    assert result.data["provider"] == "baidu_maps_geocoding"
    assert result.data["provider_display_name"] == "Baidu Maps"
    assert result.data["map_provider"] == "baidu"
    assert result.data["mode"] == "geocode"
    expected_lon, expected_lat = bd09_to_wgs84(116.3975, 39.9087)
    assert result.data["coordinate_system"] == "WGS84"
    assert result.data["top_result"]["coordinate_system"] == "WGS84"
    assert result.data["top_result"]["raw_coordinate_system"] == "BD-09"
    assert result.data["top_result"]["raw_lat"] == 39.9087
    assert result.data["top_result"]["raw_lon"] == 116.3975
    assert abs(result.data["top_result"]["lat"] - expected_lat) < 0.00001
    assert abs(result.data["top_result"]["lon"] - expected_lon) < 0.00001


def test_baidu_maps_geocode_tool_reverse_geocodes_lat_lng_order(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.baidu_maps_api_key = SecretStr("baidu-test-key")
    tools = build_tools(config)
    seen = {}

    def fake_get(url, params, timeout):
        seen["url"] = url
        seen["params"] = params
        return FakeResponse(
            {
                "status": 0,
                "result": {
                    "location": {"lng": 116.3975, "lat": 39.9087},
                    "formatted_address": "Dongcheng District, Beijing",
                    "formatted_address_poi": "Tiananmen, Dongcheng District, Beijing",
                    "sematic_description": "near Tiananmen",
                    "addressComponent": {"country": "China", "province": "Beijing", "city": "Beijing", "district": "Dongcheng"},
                    "pois": [
                        {
                            "uid": "baidu_poi_1",
                            "name": "Tiananmen",
                            "addr": "Dongcheng District",
                            "tag": "landmark",
                            "distance": "20",
                            "direction": "near",
                            "point": {"x": 116.3975, "y": 39.9087},
                        }
                    ],
                },
            }
        )

    monkeypatch.setattr("geoagent.tools.search.baidu_maps_geocode_tool.requests.get", fake_get)
    result = tools["baidu_maps_geocode"].run(lat=39.9087, lon=116.3975, coordtype="WGS84", top_k=1)

    assert result.success is True
    assert seen["url"] == "https://api.map.baidu.com/reverse_geocoding/v3/"
    assert seen["params"]["ak"] == "baidu-test-key"
    assert seen["params"]["location"] == "39.9087,116.3975"
    assert seen["params"]["coordtype"] == tools["baidu_maps_geocode"]._baidu_api_coordtype("WGS84")
    assert seen["params"]["extensions_poi"] == 1
    assert result.data["provider"] == "baidu_maps_geocoding"
    assert result.data["provider_display_name"] == "Baidu Maps"
    assert result.data["map_provider"] == "baidu"
    assert result.data["mode"] == "reverse_geocode"
    assert result.data["formatted_address"] == "Tiananmen, Dongcheng District, Beijing"
    expected_lon, expected_lat = bd09_to_wgs84(116.3975, 39.9087)
    assert result.data["coordinate_system"] == "WGS84"
    assert result.data["top_result"]["coordinate_system"] == "WGS84"
    assert result.data["top_result"]["raw_coordinate_system"] == "BD-09"
    assert abs(result.data["top_result"]["lat"] - expected_lat) < 0.00001
    assert abs(result.data["top_result"]["lon"] - expected_lon) < 0.00001
    assert result.data["top_result"]["pois"][0]["uid"] == "baidu_poi_1"
    assert result.data["top_result"]["pois"][0]["coordinate_system"] == "WGS84"


def test_baidu_maps_geocode_tool_reverse_defaults_to_wgs84_coordtype(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.baidu_maps_api_key = SecretStr("baidu-test-key")
    tools = build_tools(config)
    seen = {}

    def fake_get(url, params, timeout):
        seen["params"] = params
        return FakeResponse(
            {
                "status": 0,
                "result": {
                    "location": {"lng": 116.3975, "lat": 39.9087},
                    "formatted_address": "Dongcheng District, Beijing",
                    "pois": [],
                },
            }
        )

    monkeypatch.setattr("geoagent.tools.search.baidu_maps_geocode_tool.requests.get", fake_get)
    result = tools["baidu_maps_geocode"].run(lat=39.9087, lon=116.3975, top_k=1)

    assert result.success is True
    assert seen["params"]["coordtype"] == tools["baidu_maps_geocode"]._baidu_api_coordtype("WGS84")
    assert result.data["coordtype"] == "WGS84"
    assert result.data["top_result"]["raw_coordinate_system"] == "BD-09"


def test_baidu_geocode_does_not_send_wgs84_ret_coordtype(monkeypatch):
    # Baidu documents ret_coordtype as gcj02ll / bd09mc only (default bd09ll);
    # wgs84ll is not a valid return coordinate system. Requesting WGS84 output
    # must leave ret_coordtype unset and convert the default BD-09 internally.
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.baidu_maps_api_key = SecretStr("baidu-test-key")
    tools = build_tools(config)
    seen = {}

    def fake_get(url, params, timeout):
        seen["url"] = url
        seen["params"] = params
        return FakeResponse(
            {
                "status": 0,
                "result": {
                    "location": {"lng": 116.3975, "lat": 39.9087},
                    "precise": 1,
                    "confidence": 90,
                    "comprehension": 90,
                    "level": "门址",
                },
            }
        )

    monkeypatch.setattr("geoagent.tools.search.baidu_maps_geocode_tool.requests.get", fake_get)
    result = tools["baidu_maps_geocode"].run(
        address="北京市海淀区上地十街10号", city="北京", ret_coordtype="WGS84", top_k=1
    )

    assert result.success is True
    assert "ret_coordtype" not in seen["params"]
    assert result.data["ret_coordtype"] == "BD-09"
    assert result.data["top_result"]["raw_coordinate_system"] == "BD-09"
    assert result.data["top_result"]["coordinate_system"] == "WGS84"


def test_web_search_results_include_source_label():
    config = load_app_config(config_dir="configs", env_file=None)
    tools = build_tools(config)
    result = tools["web_search"]._result(
        title="Baidu listing for a restaurant",
        snippet="Full address from a web result.",
        url="https://www.baidu.com/place/example",
    )

    assert result["source_type"] == "web_search"
    assert result["source_name"] == "Baidu"
    assert result["source_label"] == "Web Search/Baidu"


def test_web_search_defaults_to_serper_and_caches_success(monkeypatch, tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.web_search_provider = "serper"
    config.env.serper_api_key = SecretStr("serper-test-key")
    config.system["search_cache"]["path"] = str(tmp_path / "search.sqlite3")
    tools = build_tools(config)
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append((url, headers, json))
        return FakeResponse(
            {
                "organic": [
                    {
                        "title": "Tokyo Tower official site",
                        "link": "https://www.tokyotower.co.jp/",
                        "snippet": "Official visitor information.",
                        "position": 1,
                    }
                ],
                "searchParameters": {"q": "Tokyo Tower official"},
                "credits": 1,
            }
        )

    monkeypatch.setattr("geoagent.tools.search.web_search_tool.requests.post", fake_post)
    first = tools["web_search"].run(query="Tokyo Tower official", top_k=3)
    config.env.serper_api_key = None
    second = tools["web_search"].run(query="Tokyo   Tower official", top_k=3)

    assert first.success is True
    assert first.data["provider"] == "serper"
    assert first.data["provider_metadata"]["credits"] == 1
    assert first.data["cache_hit"] is False
    assert second.data["cache_hit"] is True
    assert len(calls) == 1
    assert calls[0][1]["X-API-KEY"] == "serper-test-key"


def test_web_search_provider_argument_cannot_override_env_config(monkeypatch, tmp_path):
    # Regression: Brain must not be able to switch the web-search backend.
    # Provider is always serper; a provider kwarg is ignored.
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.web_search_provider = "serper"
    config.env.serper_api_key = SecretStr("serper-test-key")
    config.system["search_cache"]["path"] = str(tmp_path / "search.sqlite3")
    tools = build_tools(config)

    def fake_post(url, headers, json, timeout):
        return FakeResponse({"organic": [{"title": "R", "link": "https://e.com", "snippet": "s", "position": 1}]})

    monkeypatch.setattr("geoagent.tools.search.web_search_tool.requests.post", fake_post)
    result = tools["web_search"].run(query="landmark", provider="unknown_provider", top_k=2)

    assert result.success is True
    assert result.data["provider"] == "serper"


def test_web_search_serper_keeps_rich_result_sections(monkeypatch, tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    # Provider is selected via env config only (never a tool argument).
    config.env.web_search_provider = "serper"
    config.env.serper_api_key = SecretStr("serper-test-key")
    config.system["search_cache"]["path"] = str(tmp_path / "search.sqlite3")
    tools = build_tools(config)

    def fake_post(url, headers, json, timeout):
        return FakeResponse(
            {
                "organic": [{"title": "Result", "link": "https://example.com", "snippet": "Text", "position": 1}],
                "knowledgeGraph": {"title": "Entity", "description": "Known place"},
                "answerBox": {"answer": "Tokyo"},
                "peopleAlsoAsk": [{"question": "Where is it?", "snippet": "Tokyo"}],
                "relatedSearches": [{"query": "Tokyo landmark"}],
            }
        )

    monkeypatch.setattr("geoagent.tools.search.web_search_tool.requests.post", fake_post)
    result = tools["web_search"].run(query="landmark", top_k=2)

    assert result.success is True
    assert result.data["provider"] == "serper"
    assert result.data["knowledge_graph"]["title"] == "Entity"
    assert result.data["answer_box"]["answer"] == "Tokyo"
    assert result.data["people_also_ask"][0]["question"] == "Where is it?"
    assert result.data["related_searches"][0]["query"] == "Tokyo landmark"


def test_web_search_clamps_top_k_to_ten(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.serper_api_key = SecretStr("serper-test-key")
    config.system["search_cache"]["enabled"] = False
    tools = build_tools(config)
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen["json"] = json
        return FakeResponse(
            {
                "organic": [
                    {
                        "title": f"Result {index}",
                        "link": f"https://example.com/{index}",
                        "snippet": "text",
                        "position": index + 1,
                    }
                    for index in range(12)
                ]
            }
        )

    monkeypatch.setattr("geoagent.tools.search.web_search_tool.requests.post", fake_post)
    result = tools["web_search"].run(query="landmark", top_k=99)

    assert result.success is True
    assert seen["json"]["num"] == 10
    assert result.data["top_k"] == 10
    assert len(result.data["results"]) == 10


def test_webpage_read_extracts_static_html_and_caches(monkeypatch, tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["search_cache"]["path"] = str(tmp_path / "search.sqlite3")
    tools = build_tools(config)
    reader = tools["webpage_read"]
    calls = []
    html = b"""<html lang='en'><head><title>Place page</title><meta name='description' content='A place'></head>
    <body><nav>Menu</nav><main><h1>Tokyo Tower</h1><p>Address: Shibakoen, Tokyo.</p><script>ignore()</script></main></body></html>"""

    monkeypatch.setattr(reader, "_validate_public_url", lambda url: None)

    def fake_get(url, headers, timeout, allow_redirects, stream):
        calls.append(url)
        return FakeStreamResponse(html)

    monkeypatch.setattr("geoagent.tools.search.webpage_read_tool.requests.get", fake_get)
    first = reader.run(url="https://example.com/place", max_chars=1000)
    second = reader.run(url="https://example.com/place", max_chars=1000)

    assert first.success is True
    assert first.data["title"] == "Place page"
    assert "Tokyo Tower" in first.data["content"]
    assert "ignore" not in first.data["content"]
    assert first.data["cache_hit"] is False
    assert second.data["cache_hit"] is True
    assert calls == ["https://example.com/place"]


def test_webpage_read_blocks_local_addresses():
    config = load_app_config(config_dir="configs", env_file=None)
    tools = build_tools(config)
    result = tools["webpage_read"].run(url="http://127.0.0.1/private")

    assert result.success is False
    assert "blocked" in result.error.lower()


def test_poi_search_routes_internally_without_map_provider():
    config = load_app_config(config_dir="configs", env_file=None)
    tools = build_tools(config)
    # With a China country hint, routing picks Baidu even without map_provider.
    result = tools["poi_search"].run(query="Tiananmen", region="Beijing", country="China", top_k=2)
    assert result.success is False
    assert "BAIDU_MAPS_API_KEY" in result.error
    # International routing requires a standard country and then reaches Serper.
    result = tools["poi_search"].run(
        query="Tokyo Tower",
        region="Tokyo",
        country="Japan",
        top_k=2,
    )
    assert result.success is False
    assert "SERPER_API_KEY" in result.error


def test_poi_search_uses_baidu_when_requested(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["search_cache"]["enabled"] = False
    config.env.baidu_maps_api_key = SecretStr("baidu-test-key")
    config.env.google_maps_api_key = SecretStr("google-test-key")
    tools = build_tools(config)
    seen = {}

    def fake_get(url, params, timeout):
        seen["url"] = url
        seen["params"] = params
        return FakeResponse(
            {
                "status": 0,
                "message": "ok",
                "results": [
                    {
                        "uid": "baidu_uid_1",
                        "name": "Tiananmen",
                        "address": "Dongchang'an Street",
                        "province": "Beijing",
                        "city": "Beijing",
                        "area": "Dongcheng",
                        "location": {"lat": 39.9087, "lng": 116.3975},
                        "detail_info": {"classified_poi_tag": "landmark;attraction", "overall_rating": "4.8"},
                    }
                ]
                + [
                    {
                        "uid": f"baidu_uid_{index}",
                        "name": f"Tiananmen result {index}",
                        "address": "Dongchang'an Street",
                        "location": {"lat": 39.9087 + index / 10000, "lng": 116.3975},
                    }
                    for index in range(2, 13)
                ],
            }
        )

    monkeypatch.setattr("geoagent.tools.poi.poi_search_tool.requests.get", fake_get)
    result = tools["poi_search"].run(query="Tiananmen", region="Beijing", country="China", top_k=2)

    assert result.success is True
    assert seen["url"] == "https://api.map.baidu.com/place/v3/region"
    assert seen["params"]["ak"] == "baidu-test-key"
    assert seen["params"]["query"] == "Tiananmen"
    assert seen["params"]["region"] == "Beijing"
    assert seen["params"]["page_size"] == 2
    assert result.data["map_provider"] == "baidu"
    assert result.data["provider"] == "baidu_maps"
    assert result.data["candidates"][0]["map_provider"] == "baidu"
    expected_lon, expected_lat = bd09_to_wgs84(116.3975, 39.9087)
    assert result.data["coordinate_system"] == "WGS84"
    assert result.data["candidates"][0]["coordinate_system"] == "WGS84"
    assert result.data["candidates"][0]["raw_coordinate_system"] == "BD-09"
    assert abs(result.data["candidates"][0]["lon"] - expected_lon) < 0.00001
    assert abs(result.data["candidates"][0]["lat"] - expected_lat) < 0.00001
    assert len(result.data["candidates"]) == 2

    clamped = tools["poi_search"].run(
        query="Tiananmen", region="Beijing", country="China", top_k=99
    )
    assert clamped.success is True
    assert seen["params"]["page_size"] == 10
    assert len(clamped.data["candidates"]) == 10


def test_poi_search_uses_serper_places_for_google_provider(monkeypatch, tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.serper_api_key = SecretStr("serper-test-key")
    config.env.baidu_maps_api_key = SecretStr("baidu-test-key")
    config.system["search_cache"]["path"] = str(tmp_path / "search.sqlite3")
    tools = build_tools(config)
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        return FakeResponse(
            {
                "places": [
                    {
                        "cid": "123456",
                        "title": "Tokyo Tower",
                        "address": "4 Chome-2-8 Shibakoen, Tokyo, Japan",
                        "latitude": 35.6586,
                        "longitude": 139.7454,
                        "category": "Tourist attraction",
                        "website": "https://www.tokyotower.co.jp/",
                        "phoneNumber": "+81 3-3433-5111",
                        "rating": 4.5,
                        "ratingCount": 78000,
                    }
                ]
            }
        )

    monkeypatch.setattr("geoagent.tools.poi.poi_search_tool.requests.post", fake_post)
    result = tools["poi_search"].run(
        query="Tokyo Tower",
        region="Tokyo, Japan",
        country="Japan",
        top_k=2,
    )

    assert result.success is True
    assert seen["url"] == "https://google.serper.dev/places"
    assert seen["headers"]["X-API-KEY"] == "serper-test-key"
    assert seen["json"] == {
        "q": "Tokyo Tower",
        "location": "Tokyo, Japan",
        "gl": "jp",
        "hl": "ja",
        "num": 2,
    }
    assert result.data["map_provider"] == "google"
    assert result.data["provider"] == "serper_places"
    assert result.data["location"] == "Tokyo, Japan"
    assert result.data["gl"] == "jp"
    assert result.data["hl"] == "ja"
    assert result.data["candidates"][0]["map_provider"] == "google"
    assert result.data["candidates"][0]["coordinate_system"] == "WGS84"
    assert result.data["candidates"][0]["phone_number"] == "+81 3-3433-5111"
    assert result.data["candidates"][0]["place_id"] == "123456"
    assert result.data["candidates"][0]["category"] == "Tourist attraction"
    assert result.data["candidates"][0]["rating_count"] == 78000
    assert result.data["candidates"][0]["details_complete"] is True
    assert result.data["cache_hit"] is False

    cached = tools["poi_search"].run(
        query="Tokyo Tower",
        region="Tokyo, Japan",
        country="Japan",
        top_k=2,
    )
    assert cached.success is True
    assert cached.data["cache_hit"] is True


def test_poi_search_uses_country_to_resolve_serper_locale(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.serper_api_key = SecretStr("serper-test-key")
    config.system["search_cache"]["enabled"] = False
    tools = build_tools(config)
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen["json"] = json
        return FakeResponse({"places": []})

    monkeypatch.setattr("geoagent.tools.poi.poi_search_tool.requests.post", fake_post)
    result = tools["poi_search"].run(
        query="よみせ通りいこいの家",
        region="Tokyo",
        country="Japan",
        observed_entity_id="ent_restaurant",
        top_k=5,
    )

    assert result.success is True
    assert seen["json"] == {
        "q": "よみせ通りいこいの家",
        "location": "Tokyo",
        "gl": "jp",
        "hl": "ja",
        "num": 5,
    }
    assert result.data["gl"] == "jp"
    assert result.data["hl"] == "ja"


def test_poi_search_accepts_russia_common_name():
    config = load_app_config(config_dir="configs", env_file=None)
    tools = build_tools(config)

    location, gl, hl = tools["poi_search"]._resolve_serper_locale("", "Russia")

    assert location == "Russian Federation"
    assert gl == "ru"
    assert hl == "ru"


def test_poi_search_requires_country_for_international_search(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.serper_api_key = SecretStr("serper-test-key")
    config.system["search_cache"]["enabled"] = False
    tools = build_tools(config)

    def unexpected_post(*args, **kwargs):
        raise AssertionError("Serper must not be called for unresolved localization")

    monkeypatch.setattr("geoagent.tools.poi.poi_search_tool.requests.post", unexpected_post)
    result = tools["poi_search"].run(
        query="Visible POI",
        region="Unknown Area",
    )

    assert result.success is False
    assert "country is required for international POI search" in result.error


def test_poi_search_rejects_non_iso_country(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.env.serper_api_key = SecretStr("serper-test-key")
    config.system["search_cache"]["enabled"] = False
    tools = build_tools(config)

    def unexpected_post(*args, **kwargs):
        raise AssertionError("Serper must not be called for an invalid country")

    monkeypatch.setattr("geoagent.tools.poi.poi_search_tool.requests.post", unexpected_post)
    result = tools["poi_search"].run(
        query="Visible POI",
        region="Unknown Area",
        country="Atlantis",
    )

    assert result.success is False
    assert "country must be a standard ISO 3166-1 name or code" in result.error


def test_poi_search_converts_every_iso_country_name_and_code():
    config = load_app_config(config_dir="configs", env_file=None)
    tools = build_tools(config)
    poi_search = tools["poi_search"]

    for country in pycountry.countries:
        identifiers = [country.name, country.alpha_2, country.alpha_3]
        if hasattr(country, "official_name"):
            identifiers.append(country.official_name)
        for identifier in identifiers:
            location, gl, hl = poi_search._resolve_serper_locale("", identifier)
            assert location == country.name
            assert gl == country.alpha_2.lower()
            assert hl


@pytest.mark.parametrize("argument", ["gl", "hl", "location", "map_provider"])
def test_poi_search_rejects_provider_specific_arguments(argument):
    config = load_app_config(config_dir="configs", env_file=None)
    tools = build_tools(config)

    result = tools["poi_search"].run(
        query="Tokyo Tower",
        region="Tokyo",
        country="Japan",
        **{argument: "provider-specific"},
    )

    assert result.success is False
    assert f"Unsupported poi_search arguments: {argument}" in result.error
