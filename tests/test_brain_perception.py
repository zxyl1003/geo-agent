import json

from geoagent.agents.brain_agent import BrainAgent
from geoagent.core.config import load_app_config
from geoagent.core.schemas import ObservedEntity, OCRResult, VisualCue
from geoagent.models.llm_client import LLMClient
from geoagent.state.task_state import GeoLocalizationState


def test_initial_brain_decision_reads_image_and_populates_visual_fields(tmp_path, monkeypatch):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"\xff\xd8mock-jpeg")
    analysis = {
        "scene_summary": "A commercial street with right-hand traffic.",
        "visual_cues": [
            {"cue_type": "driving_side", "text": "Traffic keeps right", "confidence": 0.9}
        ],
        "ocr_results": [{"text": "Example Market", "language": "en", "confidence": 0.8}],
        "observed_entities": [
            {
                "id": "ent_market",
                "entity_type": "poi",
                "name": "Example Market",
                "text_items": ["Example Market"],
                "phones": [],
                "region_hint": "center storefront",
                "confidence": 0.8,
            }
        ],
        "search_queries": ["Example Market location"],
        "reasoning_notes": ["The storefront name is directly visible."],
        "locatability": {
            "max_expected_granularity": "poi",
            "score": 0.8,
            "rationale": "A named storefront is visible.",
        },
    }
    captured = {}

    def fake_generate(self, prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return {
            "model": "test-multimodal-brain",
            "provider": "test",
            "text": json.dumps(
                {
                    "visual_analysis": analysis,
                    "reasoning_summary": "Search the visible storefront name.",
                    "memory_references": [],
                    "experience_request": None,
                    "action_type": "call_tool",
                    "tool_requests": [
                        {
                            "tool_name": "web_search",
                            "arguments": {"query": "Example Market"},
                            "reason": "Resolve the visible named anchor.",
                        }
                    ],
                    "hypothesis_patch": None,
                    "confidence": 0.3,
                    "uncertainty_radius_m": None,
                    "final_answer": None,
                }
            ),
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    state = GeoLocalizationState(image_path=str(image_path))
    state.metadata["phase"] = "initial_brain_decision"

    output = BrainAgent(app_config=load_app_config(config_dir="configs", env_file=None)).run(state)

    content = captured["messages"][-1]["content"]
    assert isinstance(content, list)
    assert any(item.get("type") == "image_url" for item in content)
    assert "base64," not in captured["prompt"]
    assert captured["max_tokens"] == 6000
    assert output.agent_name == "brain"
    assert set(output.state_delta) == {
        "brain_decision",
        "visual_cues",
        "observed_entities",
        "ocr_results",
    }
    assert "visual_analysis" not in output.state_delta["brain_decision"]
    assert [cue.cue_type for cue in state.visual_cues] == [
        "scene_summary",
        "driving_side",
        "search_query",
    ]
    assert state.ocr_results[0].text == "Example Market"
    assert state.observed_entities[0].id == "ent_market"
    assert state.metadata["vlm_analysis"] == analysis
    assert state.metadata["locatability"]["max_expected_granularity"] == "poi"
    assert state.api_call_count["model:brain"] == 1
    assert "model:perception" not in state.api_call_count


def test_later_brain_decision_uses_recorded_evidence_without_source_image(tmp_path, monkeypatch):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"\xff\xd8mock-jpeg")
    captured = {}

    def fake_generate(self, prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return {
            "text": json.dumps(
                {
                    "visual_analysis": None,
                    "reasoning_summary": "Submit the best supported location.",
                    "action_type": "final_answer",
                    "tool_requests": [],
                    "hypothesis_patch": None,
                    "confidence": 0.5,
                    "uncertainty_radius_m": 100000,
                    "final_answer": {
                        "location_name": "Test city",
                        "lat": 35.0,
                        "lon": 139.0,
                        "granularity": "city",
                        "confidence": 0.5,
                        "uncertainty_radius_m": 100000,
                    },
                }
            ),
        }

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    state = GeoLocalizationState(image_path=str(image_path))
    state.metadata["phase"] = "investigation"

    decision = BrainAgent(app_config=load_app_config(config_dir="configs", env_file=None)).decide(state)

    assert decision.visual_analysis is None
    assert isinstance(captured["messages"][-1]["content"], str)
    assert "source image is not attached" in captured["messages"][-1]["content"]
    assert "[image attached]" not in captured["prompt"]


def test_later_brain_decision_rejects_visual_updates_without_reanalysis(tmp_path, monkeypatch):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"\xff\xd8mock-jpeg")
    calls = 0
    prompts = []

    def fake_generate(self, prompt, **kwargs):
        nonlocal calls
        calls += 1
        prompts.append(kwargs["messages"])
        return {
            "provider": "test",
            "text": json.dumps(
                {
                    "visual_analysis": None,
                    "visual_updates": (
                        {
                            "visual_cues": [
                                {"cue_type": "road_sign", "text": "New route shield", "confidence": 0.9}
                            ],
                            "ocr_results": [],
                            "observed_entities": [],
                        }
                        if calls == 1
                        else None
                    ),
                    "reasoning_summary": "Submit the current estimate.",
                    "action_type": "final_answer",
                    "tool_requests": [],
                    "hypothesis_patch": None,
                    "confidence": 0.5,
                    "uncertainty_radius_m": 100000,
                    "final_answer": {
                        "location_name": "Test city",
                        "lat": 35.0,
                        "lon": 139.0,
                        "granularity": "city",
                        "confidence": 0.5,
                        "uncertainty_radius_m": 100000,
                    },
                }
            ),
        }

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    state = GeoLocalizationState(
        image_path=str(image_path),
        visual_cues=[VisualCue(cue_type="language_script", text="Old sign", source="vlm")],
        observed_entities=[
            ObservedEntity(
                id="ent_shop",
                entity_type="poi",
                name="Old Shop",
                text_items=["Old Shop"],
                source="vlm",
            )
        ],
        ocr_results=[OCRResult(text="OLD", source="vlm")],
    )
    state.metadata["phase"] = "investigation"

    output = BrainAgent(app_config=load_app_config(config_dir="configs", env_file=None)).run(state)

    assert calls == 2
    assert [cue.text for cue in state.visual_cues] == ["Old sign"]
    assert [result.text for result in state.ocr_results] == ["OLD"]
    assert len(state.observed_entities) == 1
    assert state.observed_entities[0].name == "Old Shop"
    assert "visual_updates" not in state.metadata
    assert "visual_updates" not in output.state_delta["brain_decision"]
    assert "request visual_reanalysis" in prompts[1][-1]["content"]


def test_later_full_visual_analysis_is_retried_as_invalid(tmp_path, monkeypatch):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"\xff\xd8mock-jpeg")
    calls = 0
    prompts = []

    def fake_generate(self, prompt, **kwargs):
        nonlocal calls
        calls += 1
        prompts.append(kwargs["messages"])
        return {
            "text": json.dumps(
                {
                    "visual_analysis": (
                        {
                            "scene_summary": "Repeated full scene.",
                            "visual_cues": [],
                            "ocr_results": [],
                            "observed_entities": [],
                            "search_queries": [],
                            "reasoning_notes": [],
                            "locatability": {
                                "max_expected_granularity": "city",
                                "score": 0.5,
                                "rationale": "Repeated analysis.",
                            },
                        }
                        if calls == 1
                        else None
                    ),
                    "visual_updates": None,
                    "reasoning_summary": "Submit the current estimate.",
                    "action_type": "final_answer",
                    "tool_requests": [],
                    "hypothesis_patch": None,
                    "confidence": 0.5,
                    "uncertainty_radius_m": 100000,
                    "final_answer": {
                        "location_name": "Test city",
                        "lat": 35.0,
                        "lon": 139.0,
                        "granularity": "city",
                        "confidence": 0.5,
                        "uncertainty_radius_m": 100000,
                    },
                }
            ),
        }

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    state = GeoLocalizationState(image_path=str(image_path))
    state.metadata["phase"] = "investigation"

    decision = BrainAgent(app_config=load_app_config(config_dir="configs", env_file=None)).decide(state)

    assert calls == 2
    assert decision.visual_analysis is None
    assert "must set visual_analysis to null" in prompts[1][-1]["content"]
