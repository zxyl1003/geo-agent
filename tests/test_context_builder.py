from pydantic import SecretStr

from geoagent.core.config import load_app_config
from geoagent.core.context_builder import ContextBuilder
from geoagent.core.schemas import ToolResult
from geoagent.state.task_state import GeoLocalizationState
from geoagent.tools.registry import build_tools


def test_context_builder_includes_tool_specs():
    config = load_app_config(config_dir="configs", env_file=None)
    # Key-gated tools are exposed only when their credentials exist.
    config.env.brain_api_key = SecretStr("brain-test-key")
    config.env.brain_base_url = "https://brain.example.com"
    config.env.serper_api_key = SecretStr("serper-test-key")
    tools = build_tools(config)
    state = GeoLocalizationState(image_path="examples/images/taipei_street.jpg", user_query="Where?")

    context = ContextBuilder(config, tools).build_brain_context(state)

    assert context["task"]["user_query"] == "Where?"
    tool_names = {tool["name"] for tool in context["available_tools"]}
    assert "ocr" in tool_names
    assert "map_tile_verify" in tool_names
    assert "web_search" in tool_names


def test_context_builder_compacts_large_map_tile_results():
    config = load_app_config(config_dir="configs", env_file=None)
    result = ToolResult(
        tool_name="map_tile_verify",
        success=True,
        data={
            "tiles": [{"url": f"https://example.test/{index}", "path": f"tile_{index}.png"} for index in range(20)],
            "verification": {
                "verdict": "inconclusive",
                "matching_features": ["road"] * 10,
                "raw": {"large": "x" * 1000},
                "model_usage": {"total_tokens": 999},
            },
        },
    )

    summary = ContextBuilder(config).summarize_tool_result(result)

    assert "tiles" not in summary
    assert summary["tile_count"] == 20
    assert "raw" not in summary["verification"]
    assert "model_usage" not in summary["verification"]
    assert len(summary["verification"]["matching_features"]) == 3


def test_context_builder_keeps_only_latest_tool_result():
    config = load_app_config(config_dir="configs", env_file=None)
    state = GeoLocalizationState(image_path="examples/images/demo.jpg")
    state.add_tool_result(ToolResult(tool_name="web_search", success=True, data={"query": "older"}))
    state.add_tool_result(ToolResult(tool_name="poi_search", success=True, data={"query": "latest"}))

    context = ContextBuilder(config).build_brain_context(state)

    assert "tool_history" not in context
    assert context["latest_tool_result"]["tool_name"] == "poi_search"
    assert context["latest_tool_result"]["query"] == "latest"


def test_context_builder_preserves_requested_retrieval_result_count():
    config = load_app_config(config_dir="configs", env_file=None)
    builder = ContextBuilder(config)
    web_results = [
        {"title": f"Result {index}", "snippet": "text", "url": f"https://example.com/{index}"}
        for index in range(5)
    ]
    poi_candidates = [
        {"name": f"POI {index}", "lat": 30.0 + index, "lon": 114.0 + index}
        for index in range(5)
    ]

    web_summary = builder.summarize_tool_result(
        ToolResult(tool_name="web_search", success=True, data={"results": web_results})
    )
    poi_summary = builder.summarize_tool_result(
        ToolResult(tool_name="poi_search", success=True, data={"candidates": poi_candidates})
    )
    geocode_summary = builder.summarize_tool_result(
        ToolResult(tool_name="geocode", success=True, data={"results": poi_candidates})
    )

    assert len(web_summary["results"]) == 5
    assert len(poi_summary["candidates"]) == 5
    assert len(geocode_summary["results"]) == 5
