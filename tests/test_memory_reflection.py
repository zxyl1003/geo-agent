"""Tests for the memory reflection mechanism:

- ground-truth context enrichment via reverse_geocode (learning-time only)
- attribution recorded on episodes even when no memory is written
- deterministic guard blocking ground-truth place names from memory content
- new FailureAttribution model field
"""

import json

import pytest

from geoagent.core.config import load_app_config
from geoagent.core.prompt_loader import load_prompt
from geoagent.core.schemas import FinalAnswer, ToolResult
from geoagent.memory.manager import MemoryManager
from geoagent.memory.models import (
    FailureAttribution,
    MemoryAgentReview,
    MemoryCandidate,
)
from geoagent.models.llm_client import LLMClient
from geoagent.state.task_state import GeoLocalizationState


pytestmark = pytest.mark.usefixtures("fake_chroma")


def _memory_config(tmp_path, mode: str = "full") -> dict:
    return {
        "enabled": True,
        "mode": mode,
        "storage_dir": str(tmp_path),
        "chroma": {"enabled": True},
    }


def _no_write_attribution_payload() -> dict:
    """LLM review that records attribution but proposes no reusable memory."""
    return {
        "should_write": False,
        "diagnosis": "Branch POI name was treated as a location anchor.",
        "diagnosis_confidence": 0.6,
        "successful_levels": ["country"],
        "failed_levels": ["city"],
        "failure_attribution": {
            "failure_type": "branch_name_as_anchor",
            "success_pattern": "",
            "rationale": "The agent geocoded a chain-branch name and accepted its city without corroboration.",
            "confidence": 0.7,
        },
        "candidate": None,
        "rationale": "Diagnosis is clear but below the confidence bar for a reusable memory.",
    }


def _state_with_truth(image_path: str = "img.jpg") -> GeoLocalizationState:
    state = GeoLocalizationState(image_path=image_path)
    state.metadata["ground_truth"] = {
        "lat": 30.6069,
        "lon": 114.2856,
        "country": "China",
    }
    state.metadata["feedback_type"] = "ground_truth"
    state.final_answer = FinalAnswer(
        location_name="Somewhere else",
        country="China",
        region="Hubei",
        lat=29.0,
        lon=108.0,
        granularity="city",
        confidence=0.7,
    )
    return state


class _FakeReverseGeocode:
    """Stand-in for the reverse_geocode facade used at learning time."""

    def __init__(self, result: ToolResult | None = None) -> None:
        self._result = result
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return True

    def safe_run(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        if self._result is not None:
            return self._result
        return ToolResult(
            tool_name="reverse_geocode",
            success=True,
            data={
                "provider": "locationiq",
                "provider_display_name": "LocationIQ",
                "formatted_address": "Aomen Road, Jiang'an District, Wuhan, Hubei, China",
                "top_result": {
                    "lat": 30.6069,
                    "lon": 114.2856,
                    "formatted_address": "Aomen Road, Jiang'an District, Wuhan, Hubei, China",
                    "address_components": {
                        "country": "China",
                        "region": "Hubei",
                        "city": "Wuhan",
                        "district": "Jiang'an",
                        "street": "Aomen Road",
                    },
                },
            },
        )


def _make_manager(tmp_path) -> MemoryManager:
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path)
    config.env.locationiq_api_key = __import__("pydantic").SecretStr("liq-test")
    return MemoryManager.from_config(config)


def test_ground_truth_context_enriches_review_context(tmp_path, monkeypatch):
    def fake_generate(self, prompt, **kwargs):
        if self.model_role == "memory_manager":
            return {"provider": "test", "text": json.dumps(_no_write_attribution_payload()), "usage": {}}
        return {"provider": "test", "text": "{}", "usage": {}}

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    manager = _make_manager(tmp_path)
    fake_tool = _FakeReverseGeocode()
    monkeypatch.setattr(manager, "_reverse_geocode_tool", lambda: fake_tool)

    state = _state_with_truth()
    episode = manager._episode_from_state(state)
    context = manager._review_episode(state, episode)

    assert episode["ground_truth_context"] == {
        "provider": "locationiq",
        "formatted_address": "Aomen Road, Jiang'an District, Wuhan, Hubei, China",
        "country": "China",
        "region": "Hubei",
        "city": "Wuhan",
        "district": "Jiang'an",
        "street": "Aomen Road",
    }
    assert fake_tool.calls and fake_tool.calls[0]["lat"] == 30.6069
    review_context = state.metadata["memory_review_context"]
    assert review_context["ground_truth_context"]["city"] == "Wuhan"
    assert review_context["coordinate_distance_thresholds_m"] == [10, 20, 50, 100, 1000, 5000, 25000]
    assert review_context["precomputed_facts"]["evaluated_granularity"] == "city"
    assert review_context["precomputed_facts"]["coordinate_assessment_expected"] is False
    assert review_context["precomputed_facts"]["within_uncertainty_radius"] is None
    assert len(review_context["precomputed_facts"]["distance_threshold_checks"]) == 7
    assert isinstance(context, MemoryAgentReview)
    assert context.successful_levels == ["country"]
    assert context.failed_levels == ["city"]


def test_ground_truth_context_is_cached_per_coordinate(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path)
    fake_tool = _FakeReverseGeocode()
    monkeypatch.setattr(manager, "_reverse_geocode_tool", lambda: fake_tool)

    manager._ground_truth_context({"lat": 30.6069, "lon": 114.2856})
    manager._ground_truth_context({"lat": 30.6069, "lon": 114.2856})
    manager._ground_truth_context({"lat": 30.6068, "lon": 114.2855})

    assert len(fake_tool.calls) == 2  # one cache hit, one new rounded key


def test_ground_truth_context_retries_without_wrong_country_hint(tmp_path, monkeypatch):
    """A wrong country hint in the metadata (e.g. country=China but coordinates
    in the US) must not leave the GT context empty: retry without the hint so
    the international route can resolve it."""
    manager = _make_manager(tmp_path)
    calls: list[dict] = []

    class _RoutingTool:
        def is_available(self):
            return True

        def safe_run(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("country"):
                return ToolResult(tool_name="reverse_geocode", success=False, error="no result in that region")
            return ToolResult(
                tool_name="reverse_geocode",
                success=True,
                data={
                    "provider": "locationiq",
                    "formatted_address": "Shelbyville, Kentucky, USA",
                    "top_result": {
                        "address_components": {
                            "country": "United States",
                            "region": "Kentucky",
                            "city": "Shelbyville",
                            "street": "Warriors Way",
                        }
                    },
                },
            )

    monkeypatch.setattr(manager, "_reverse_geocode_tool", lambda: _RoutingTool())

    context = manager._ground_truth_context({"lat": 38.2358, "lon": -85.2389, "country": "China"})

    assert len(calls) == 2
    assert calls[0]["country"] == "China"  # first tries the hint
    assert calls[1]["country"] is None  # then retries international
    assert context is not None
    assert context["city"] == "Shelbyville"
    assert context["country"] == "United States"


def test_ground_truth_context_enrichment_failure_is_non_blocking(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path)

    class _FailingTool(_FakeReverseGeocode):
        def safe_run(self, **kwargs) -> ToolResult:
            self.calls.append(kwargs)
            raise RuntimeError("network down")

    failing = _FailingTool()
    monkeypatch.setattr(manager, "_reverse_geocode_tool", lambda: failing)

    state = _state_with_truth()
    episode = manager._episode_from_state(state)

    assert episode["ground_truth_context"] is None
    assert state.metadata["ground_truth_context"] == {}
    # Episode still records normally.
    manager.store.insert_episode(episode)
    assert manager.store.get_episode(episode["episode_id"]) is not None


def test_attribution_persisted_even_when_no_memory_written(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path)

    def fake_generate(self, prompt, **kwargs):
        if self.model_role == "memory_manager":
            return {"provider": "test", "text": json.dumps(_no_write_attribution_payload()), "usage": {}}
        return {"provider": "test", "text": "{}", "usage": {}}

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    state = _state_with_truth()
    manager.record_episode_and_update(state)

    assert state.metadata["memory_update"]["status"] == "episode_recorded"
    attribution = state.metadata.get("memory_attribution")
    assert attribution is not None
    assert attribution["failure_type"] == "branch_name_as_anchor"
    assert attribution["successful_levels"] == ["country"]
    assert attribution["failed_levels"] == ["city"]
    # Persisted on the episode row for audit.
    stored = manager.store.get_episode(state.metadata["memory_episode_id"])
    stored_attribution = json.loads(stored["attribution_json"])
    assert stored_attribution["failure_type"] == "branch_name_as_anchor"
    assert stored_attribution["successful_levels"] == ["country"]
    assert stored_attribution["failed_levels"] == ["city"]


def test_inconsistent_reflection_retries_once_and_accepts_correction(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path)
    monkeypatch.setattr(manager, "_reverse_geocode_tool", lambda: None)
    calls = []
    responses = [
        {
            **_no_write_attribution_payload(),
            "successful_levels": ["country", "coordinates"],
            "failed_levels": ["city"],
        },
        {
            **_no_write_attribution_payload(),
            "successful_levels": ["country"],
            "failed_levels": ["city", "coordinates"],
            "failure_attribution": {
                "failure_type": "single_source_precision",
                "success_pattern": "",
                "rationale": "The precise branch coordinate was not corroborated.",
                "confidence": 0.8,
            },
        },
    ]

    def fake_generate(self, prompt, **kwargs):
        if self.model_role == "memory_manager":
            calls.append(prompt)
            return {"provider": "test", "text": json.dumps(responses[len(calls) - 1]), "usage": {}}
        return {"provider": "test", "text": "{}", "usage": {}}

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    state = _state_with_truth()
    state.final_answer.granularity = "poi"
    state.final_answer.uncertainty_radius_m = 50
    episode = manager._episode_from_state(state)

    review = manager._review_episode(state, episode)

    assert len(calls) == 2
    assert review.successful_levels == ["country"]
    assert review.failed_levels == ["city", "coordinates"]
    consistency = state.metadata["memory_review_consistency"]
    assert consistency["retried"] is True
    assert consistency["accepted"] is True
    assert "exceeding the declared uncertainty radius" in " ".join(consistency["initial_issues"])
    assert state.metadata["memory_review_context"]["consistency_issues"]


def test_persistently_inconsistent_reflection_cannot_write_memory(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path)
    monkeypatch.setattr(manager, "_reverse_geocode_tool", lambda: None)
    payload = {
        "should_write": True,
        "diagnosis": "An uncorroborated precise result was accepted.",
        "diagnosis_confidence": 0.8,
        "successful_levels": ["country", "coordinates"],
        "failed_levels": ["city"],
        "failure_attribution": {
            "failure_type": "single_source_precision",
            "success_pattern": "",
            "rationale": "The coordinate came from one source.",
            "confidence": 0.8,
        },
        "candidate": {
            "memory_type": "tool_policy",
            "situation": "A precise POI result comes from one source.",
            "lesson": "One result does not establish coordinate precision.",
            "action_policy": ["Corroborate the coordinate independently."],
            "applicable_conditions": ["single_precise_source"],
            "failure_conditions": [],
            "confidence": 0.8,
            "diagnosis_confidence": 0.8,
            "metadata": {},
        },
        "merge_memory_id": None,
        "rationale": "Retain the rule.",
    }
    calls = 0

    def fake_generate(self, prompt, **kwargs):
        nonlocal calls
        if self.model_role == "memory_manager":
            calls += 1
            return {"provider": "test", "text": json.dumps(payload), "usage": {}}
        return {"provider": "test", "text": "{}", "usage": {}}

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    state = _state_with_truth()
    state.final_answer.granularity = "poi"
    state.final_answer.uncertainty_radius_m = 50
    episode = manager._episode_from_state(state)

    review = manager._review_episode(state, episode)

    assert calls == 2
    assert review.should_write is False
    assert review.candidate is None
    assert state.metadata["memory_review_consistency"]["accepted"] is False
    assert "remained inconsistent" in review.rationale


def test_policy_rejects_ground_truth_place_name_leak(tmp_path):
    manager = _make_manager(tmp_path)
    forbidden = manager._forbidden_place_names(
        {
            "country": "China",
            "region": "Hubei",
            "city": "Wuhan",
            "district": "Jiang'an",
            "street": "Aomen Road",
        }
    )
    assert "china" not in forbidden  # country too broad to leak
    assert "wuhan" in forbidden and "aomen road" in forbidden

    candidate = MemoryCandidate(
        memory_type="failure_pattern",
        situation="in Wuhan a storefront brand was used as an anchor",
        lesson="brand name is not proof of location",
        action_policy=["verify address"],
        applicable_conditions=["scene:commercial"],
        source_episode_id="e1",
        feedback_type="ground_truth",
        confidence=0.8,
        diagnosis_confidence=0.8,
    )
    decision = manager.policy.candidate_decision(candidate, forbidden)
    assert decision.accepted is False
    assert decision.reason == "ground_truth_context_leak"


def test_policy_allows_country_level_memory_mention(tmp_path):
    manager = _make_manager(tmp_path)
    forbidden = manager._forbidden_place_names(
        {"country": "China", "city": "Wuhan", "district": "Jiang'an"}
    )
    candidate = MemoryCandidate(
        memory_type="failure_pattern",
        situation="in China a storefront brand was used as an anchor",
        lesson="brand name is not proof of location",
        action_policy=["verify address"],
        applicable_conditions=["scene:commercial"],
        source_episode_id="e1",
        feedback_type="ground_truth",
        confidence=0.8,
        diagnosis_confidence=0.8,
    )
    decision = manager.policy.candidate_decision(candidate, forbidden)
    assert decision.accepted is True


def test_failure_attribution_model_parses():
    review = MemoryAgentReview.model_validate(
        {
            "should_write": False,
            "diagnosis": "x",
            "diagnosis_confidence": 0.5,
            "successful_levels": ["country", "region"],
            "failed_levels": ["city", "coordinates"],
            "failure_attribution": {
                "failure_type": "visual_cue_misread",
                "success_pattern": "",
                "rationale": "road markings misread",
                "confidence": 0.6,
            },
        }
    )
    assert isinstance(review.failure_attribution, FailureAttribution)
    assert review.failure_attribution.failure_type == "visual_cue_misread"
    assert review.successful_levels == ["country", "region"]
    assert review.failed_levels == ["city", "coordinates"]


def test_reflection_prompt_contains_protocol_and_vocabulary():
    prompt = load_prompt("configs", "memory_manager_agent.md")
    compact = " ".join(prompt.split())
    assert "Reflection Protocol" in prompt
    assert "Understand the outcome" in prompt
    assert "Attribute the trajectory" in prompt
    assert "branch_name_as_anchor" in prompt
    assert "single_source_precision" in prompt
    assert "multi_poi_proximity" in prompt
    assert "ground_truth_context" in compact
    assert "diagnosis-only" in compact
    assert "failure_attribution" in prompt
    assert "successful_levels" in prompt
    assert "failed_levels" in prompt
    assert "coordinate_distance_thresholds_m" in prompt
    assert "precomputed_facts" in prompt
    assert "consistency_issues" in prompt
