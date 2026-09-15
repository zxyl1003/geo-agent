import json
import sqlite3

import pytest

from geoagent.agents.brain_agent import BrainAgent
from geoagent.agents.memory_manager_agent import MemoryManagerAgent
from geoagent.core.config import load_app_config
from geoagent.core.context_builder import ContextBuilder
from geoagent.core.schemas import (
    AgentOutput,
    BrainDecision,
    FinalAnswer,
    Hypothesis,
    OCRResult,
    ToolRequest,
    ToolResult,
    VisualCue,
)
from geoagent.memory.manager import MemoryManager
from geoagent.memory.models import (
    ExperienceAdvice,
    ExperienceNextCheck,
    ExperienceToolCheck,
    MemoryAgentReview,
    MemoryCandidate,
)
from geoagent.models.llm_client import LLMClient
from geoagent.state.task_state import GeoLocalizationState
from geoagent.workflows.react_workflow import ReactWorkflow


pytestmark = pytest.mark.usefixtures("fake_chroma")


def _memory_config(tmp_path, mode: str = "full") -> dict:
    return {
        "enabled": True,
        "mode": mode,
        "storage_dir": str(tmp_path),
        "chroma": {"enabled": True},
    }


def _candidate(episode_id: str = "episode_test", feedback_type: str = "ground_truth") -> MemoryCandidate:
    return MemoryCandidate(
        memory_type="failure_pattern",
        situation="night urban commercial street with generic storefront OCR",
        lesson="Generic storefront OCR can cause premature convergence to the wrong city.",
        action_policy=[
            "Check whether OCR contains a unique road, institution, or administrative name.",
            "Cross-check generic text against traffic rules and road-system evidence.",
        ],
        applicable_conditions=["scene:urban_dense", "time:night", "ocr:generic_storefront"],
        failure_conditions=["ocr:unique_address", "landmark:unique"],
        source_episode_id=episode_id,
        feedback_type=feedback_type,
        confidence=0.8,
        diagnosis_confidence=0.75,
        metadata={
            "failure_type": "overtrusted_ocr",
            "cue_category": "ocr_text",
            "tool_scenario": "ocr",
        },
    )


def _memory_candidate_payload(merge_memory_id: str | None = None) -> dict:
    """A valid LLM-proposed candidate JSON the memory_manager agent would return."""
    return {
        "should_write": True,
        "diagnosis": "Generic storefront OCR caused premature convergence to the wrong city.",
        "diagnosis_confidence": 0.75,
        "merge_memory_id": merge_memory_id,
        "failure_attribution": {
            "failure_type": "overtrusted_ocr",
            "success_pattern": "",
            "rationale": "Generic storefront OCR was treated as a unique location anchor.",
            "confidence": 0.75,
        },
        "candidate": {
            "memory_type": "failure_pattern",
            "situation": "night urban commercial street with generic storefront OCR",
            "lesson": "Generic storefront OCR can cause premature convergence to the wrong city.",
            "action_policy": [
                "Check whether OCR contains a unique road, institution, or administrative name.",
                "Cross-check generic text against traffic rules and road-system evidence.",
            ],
            "applicable_conditions": ["scene:urban_dense", "time:night", "ocr:generic_storefront"],
            "failure_conditions": ["ocr:unique_address", "landmark:unique"],
            "confidence": 0.8,
            "diagnosis_confidence": 0.75,
            "metadata": {
                "failure_type": "overtrusted_ocr",
                "cue_category": "ocr_text",
                "tool_scenario": "ocr",
            },
        },
        "rationale": "LLM extracted a reusable OCR-overtrust lesson from the episode trajectory.",
    }


def _patch_memory_llm(monkeypatch) -> None:
    """Stub the LLM so the memory_manager agent returns a valid candidate."""

    def fake_generate(self, prompt, **kwargs):
        if self.model_role != "memory_manager":
            return {"provider": "test", "text": "{}", "usage": {}}
        return {"provider": "test", "text": json.dumps(_memory_candidate_payload()), "usage": {}}

    monkeypatch.setattr(LLMClient, "generate", fake_generate)


def _insert_memory(manager: MemoryManager):
    candidate = _candidate()
    decision = manager.policy.candidate_decision(candidate)
    assert decision.accepted
    item = manager.store.insert_memory(candidate)
    manager.index.upsert(item)
    return item

def _failed_ocr_state() -> GeoLocalizationState:
    state = GeoLocalizationState(image_path="night_shop_street.jpg", user_query="Where is this?")
    state.visual_cues = [
        VisualCue(cue_type="scene", text="night commercial street with many storefront signs", confidence=0.8),
        VisualCue(cue_type="urban_form", text="dense East Asian city street", confidence=0.7),
    ]
    state.ocr_results = [
        OCRResult(text="Generic Shop", confidence=0.8, source="test"),
        OCRResult(text="Coffee", confidence=0.7, source="test"),
    ]
    state.hypotheses = [
        Hypothesis(
            id="hyp_seoul",
            name="Seoul",
            country="South Korea",
            lat=37.5665,
            lon=126.978,
            score=0.82,
            rationale="OCR text and shop signs seemed to indicate Seoul.",
        )
    ]
    state.final_answer = FinalAnswer(
        location_name="Seoul",
        country="South Korea",
        lat=37.5665,
        lon=126.978,
        granularity="coordinates",
        confidence=0.82,
        reasoning=["OCR text and storefront signs were treated as the strongest evidence."],
        evidence_summary=["Multiple shop signs were visible."],
    )
    state.metadata["ground_truth"] = {"country": "Japan", "latitude": 35.6812, "longitude": 139.7671}
    state.metadata["feedback_type"] = "ground_truth"
    return state


def test_memory_manager_writes_failure_memory_and_retrieves_it(tmp_path, monkeypatch):
    _patch_memory_llm(monkeypatch)
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path)
    manager = MemoryManager.from_config(config)
    assert manager is not None

    failed_state = _failed_ocr_state()
    manager.record_episode_and_update(failed_state)

    update = failed_state.metadata["memory_update"]
    assert update["status"] == "new_memory_inserted"
    assert update["memory_type"] == "failure_pattern"
    assert failed_state.metadata["memory_agent_review"]["proposal_source"] == "llm"

    memories = manager.store.list_memories()
    assert len(memories) == 1
    memory = memories[0]
    assert memory.metadata["failure_type"] == "overtrusted_ocr"
    assert "OCR" in " ".join(memory.action_policy)

    hits = manager.retrieve(
        "Generic storefront OCR may be causing premature convergence; what should be verified next?\n"
        "reasoning_stage:evidence_weighting\n"
        'context:{"scene":"night urban commercial street","cue_types":["ocr"]}',
        top_k=3,
    )

    assert hits
    digest = hits[0].prompt_digest()
    assert digest["memory_id"] == memory.memory_id
    assert digest["note"].startswith("Experience memory")


def test_memory_manager_without_truth_records_episode_but_not_memory(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="learn_only")
    manager = MemoryManager.from_config(config)
    assert manager is not None

    state = GeoLocalizationState(image_path="unverified.jpg", user_query="Where?")
    state.visual_cues = [VisualCue(cue_type="scene", text="generic street", confidence=0.5)]
    state.final_answer = FinalAnswer(location_name="Unknown", granularity="unknown", confidence=0.2)

    manager.record_episode_and_update(state)

    assert state.metadata["memory_update"]["status"] == "episode_recorded"
    assert manager.store.list_memories() == []


def test_context_builder_includes_only_brain_recalled_experience_memories():
    config = load_app_config(config_dir="configs", env_file=None)
    state = GeoLocalizationState(image_path="demo.jpg")
    state.metadata["recalled_memories"] = [
        {
            "memory_id": "mem_test",
            "memory_type": "failure_pattern",
            "lesson": "Do not overtrust generic OCR.",
        }
    ]

    context = ContextBuilder(config).build_brain_context(state)

    assert context["memory"]["recalled_experience_memories"][0]["memory_id"] == "mem_test"


def test_memory_manager_agent_uses_llm_review(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    state = GeoLocalizationState(image_path="episode.jpg")
    state.metadata["memory_review_context"] = {
        "episode_id": "episode_test",
        "feedback_type": "ground_truth",
        "success": False,
        "outcome": {"label": "failure"},
        "visual_clues": [],
        "observed_entities": [],
        "ocr_results": [],
        "hypotheses": [],
        "tool_calls": [],
        "tool_results": [],
        "reasoning_trace": [],
        "final_answer": None,
    }

    _patch_memory_llm(monkeypatch)
    agent = MemoryManagerAgent(app_config=config)
    output = agent.run(state)

    review = output.metadata["review"]
    assert isinstance(review, MemoryAgentReview)
    assert review.proposal_source == "llm"
    assert review.candidate is not None
    assert review.candidate.metadata["failure_type"] == "overtrusted_ocr"
    assert state.api_call_count["total"] >= 1


def test_react_workflow_uses_registered_memory_agent_when_enabled(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path)
    workflow = ReactWorkflow(
        app_config=config,
        tools={},
        workflow_config=config.workflows["react"],
    )
    workflow.agents = workflow._default_agents()

    manager = workflow._memory_manager()

    assert manager is not None
    assert set(workflow.agents) == {"brain", "memory_manager"}
    assert manager.agent is workflow.agents["memory_manager"]


def test_brain_memory_references_are_persisted_as_usage(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="full")
    manager = MemoryManager.from_config(config)
    assert manager is not None
    item = _insert_memory(manager)

    state = GeoLocalizationState(image_path="query.jpg")
    state.metadata["memory_usage"] = [
        {
            "memory_id": item.memory_id,
            "retrieved_rank": 1,
            "retrieval_score": 0.9,
            "was_returned_to_brain": True,
            "was_cited_by_brain": False,
            "usage_role": "warning",
        }
    ]
    BrainAgent(app_config=config)._record_memory_references(state, [item.memory_id, "mem_not_injected"])
    manager.record_episode_and_update(state)

    episode_id = state.metadata["memory_episode_id"]
    usage = manager.store.get_usage(episode_id)[0]
    refreshed = manager.store.get_memory(item.memory_id)
    assert usage["was_cited_by_brain"] == 1
    assert refreshed is not None
    assert state.metadata["invalid_memory_references"] == ["mem_not_injected"]


def test_initial_experience_check_filters_memory_before_brain_revision(tmp_path, monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="retrieve_only")
    manager = MemoryManager.from_config(config)
    assert manager is not None
    item = _insert_memory(manager)
    workflow = ReactWorkflow(
        app_config=config,
        tools={},
        workflow_config=config.workflows["react"].model_copy(update={"max_steps": 1}),
    )
    memory_agent = MemoryManagerAgent(app_config=config)
    workflow.agents = {"brain": object(), "memory_manager": memory_agent}
    brain_calls = 0

    def fake_advise_context(state, context):
        assert len(context["candidate_memories"]) == 1
        assert context["progress"]["wake_reasons"] == ["initial_brain_decision"]
        return ExperienceAdvice(
            intervene=True,
            requires_brain_revision=True,
            selected_memory_ids=[item.memory_id],
            guidance="Verify whether the visible text is a unique location anchor.",
            applicability_reason="The proposed OCR action does not yet resolve generic-name ambiguity.",
        )

    def fake_run_agent(agent_name, state):
        nonlocal brain_calls
        brain_calls += 1
        if brain_calls == 1:
            decision = BrainDecision(
                reasoning_summary="Generic storefront OCR is uncertain; try OCR before choosing a location.",
                action_type="call_tool",
                tool_requests=[ToolRequest(tool_name="ocr", arguments={"image_path": "wrong.jpg"})],
                confidence=0.3,
            )
        else:
            assert state.metadata["pending_experience_revision"]["status"] == "pending_not_executed"
            assert state.tool_results == []
            decision = BrainDecision(
                reasoning_summary="The recalled OCR strategy supports the revised decision.",
                action_type="final_answer",
                confidence=0.8,
                final_answer=FinalAnswer(
                    location_name="Test location",
                    lat=35.0,
                    lon=139.0,
                    granularity="coordinates",
                    confidence=0.8,
                ),
            )
        return AgentOutput(
            agent_name="brain",
            state_delta={"brain_decision": decision.model_dump()},
            metadata={"decision": decision},
        )

    monkeypatch.setattr(workflow, "run_agent", fake_run_agent)
    monkeypatch.setattr(memory_agent, "advise_context", fake_advise_context)
    monkeypatch.setattr(
        "geoagent.workflows.react_workflow.validate_image_input",
        lambda path: path,
    )

    state = workflow.run("image.jpg")

    assert brain_calls == 2  # Initial reasoning followed by one memory-guided revision.
    assert state.step_count == 1
    assert state.status == "completed"
    assert state.metadata["experience_initial_check_completed"] is True
    assert state.metadata["experience_checks"][0]["wake_reasons"] == ["initial_brain_decision"]
    assert state.metadata["experience_checks"][0]["selected_memory_ids"] == [item.memory_id]
    assert state.tool_calls == []
    assert state.tool_results == []
    assert state.metadata["recalled_memories"] == []
    assert state.metadata["experience_checks"][0]["delivered_to_brain"] is True
    assert state.metadata["memory_usage"][0]["was_returned_to_brain"] is True
    episode_id = state.metadata["memory_episode_id"]
    stored_usage = manager.store.get_usage(episode_id)
    assert manager.store.get_episode(episode_id) is not None
    assert len(stored_usage) == 1
    assert stored_usage[0]["memory_id"] == item.memory_id
    assert stored_usage[0]["retrieved_rank"] == 1
    assert stored_usage[0]["retrieval_score"] == pytest.approx(
        state.metadata["memory_usage"][0]["retrieval_score"]
    )
    assert stored_usage[0]["was_returned_to_brain"] == 1
    assert stored_usage[0]["was_cited_by_brain"] == 0
    assert state.metadata["memory_update"] == {
        "episode_id": episode_id,
        "status": "retrieval_audit_recorded",
        "retrieved_count": 1,
    }
    assert len(manager.store.list_memories()) == 1


def test_experience_guidance_does_not_revise_pending_action_unless_requested(tmp_path, monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="retrieve_only")
    manager = MemoryManager.from_config(config)
    assert manager is not None
    item = _insert_memory(manager)
    workflow = ReactWorkflow(
        app_config=config,
        tools={},
        workflow_config=config.workflows["react"].model_copy(update={"max_steps": 1}),
    )
    memory_agent = MemoryManagerAgent(app_config=config)
    workflow.agents = {"brain": object(), "memory_manager": memory_agent}
    brain_calls = 0
    executed_requests = []

    def fake_run_agent(agent_name, state):
        nonlocal brain_calls
        brain_calls += 1
        decision = BrainDecision(
            reasoning_summary="Search the visible business directly.",
            action_type="call_tool",
            tool_requests=[ToolRequest(tool_name="poi_search", arguments={"query": "Visible Shop"})],
            confidence=0.4,
        )
        return AgentOutput(agent_name="brain", metadata={"decision": decision})

    def fake_advise_context(state, context):
        return ExperienceAdvice(
            intervene=True,
            requires_brain_revision=False,
            selected_memory_ids=[item.memory_id],
            guidance="If the POI result is ambiguous, add the visible phone number.",
            applicability_reason="This is a fallback refinement, not a reason to replace the pending call.",
        )

    def fake_execute_requests(state, requests):
        executed_requests.extend(requests)
        return []

    monkeypatch.setattr(workflow, "run_agent", fake_run_agent)
    monkeypatch.setattr(memory_agent, "advise_context", fake_advise_context)
    monkeypatch.setattr(workflow, "_execute_requests", fake_execute_requests)
    monkeypatch.setattr("geoagent.workflows.react_workflow.validate_image_input", lambda path: path)

    state = workflow.run("image.jpg")

    assert brain_calls == 1
    assert executed_requests[0].arguments == {"query": "Visible Shop"}
    assert state.metadata["experience_checks"][0]["brain_revision_performed"] is False
    assert state.metadata["experience_checks"][0]["executed_brain_decision"]["tool_requests"][0][
        "arguments"
    ] == {"query": "Visible Shop"}
    assert state.metadata["memory_usage"][0]["was_returned_to_brain"] is False


def test_initial_experience_check_can_be_disabled():
    config = load_app_config(config_dir="configs", env_file=None)
    workflow_config = config.workflows["react"]
    workflow = ReactWorkflow(
        app_config=config,
        tools={"placeholder": object()},
        workflow_config=workflow_config.model_copy(
            update={
                "extra": {
                    **workflow_config.extra,
                    "experience_scheduling": {
                        "initial_check": False,
                    },
                }
            }
        ),
    )

    assert workflow._experience_scheduling_config() == {
        "initial_check": False,
    }

    disabled_workflow = ReactWorkflow(
        app_config=config,
        tools={},
        workflow_config=workflow_config.model_copy(
            update={
                "extra": {
                    **workflow_config.extra,
                    "experience_scheduling": {
                        "initial_check": False,
                    },
                }
            }
        ),
    )
    state = GeoLocalizationState(image_path="demo.jpg")
    disabled_workflow._initialize_experience_scheduler(state)
    assert disabled_workflow._initial_experience_check_due(state) is False
    assert state.tool_calls == []


def test_experience_context_excludes_ground_truth_and_task_metadata():
    config = load_app_config(config_dir="configs", env_file=None)
    state = GeoLocalizationState(image_path="anonymous.jpg")
    state.metadata["ground_truth"] = {
        "city": "SECRET_GT_CITY",
        "latitude": 12.3456789,
        "longitude": 98.7654321,
    }
    state.metadata["task_metadata"] = {"private_marker": "SECRET_TASK_METADATA"}

    context = ContextBuilder(config).build_experience_context(
        state,
        wake_reasons=["initial_brain_decision"],
        proposed_decision=None,
        help_request=None,
        new_tool_results=[],
        candidate_memories=[],
    )
    serialized = json.dumps(context, ensure_ascii=False)

    assert "SECRET_GT_CITY" not in serialized
    assert "12.3456789" not in serialized
    assert "98.7654321" not in serialized
    assert "SECRET_TASK_METADATA" not in serialized


def test_followup_experience_retrieval_query_uses_progress_delta():
    config = load_app_config(config_dir="configs", env_file=None)
    state = GeoLocalizationState(
        image_path="anonymous.jpg",
        visual_cues=[VisualCue(cue_type="architecture", text="STATIC_VISUAL_DETAIL")],
        ocr_results=[OCRResult(text="STATIC_OCR_DETAIL")],
    )
    state.metadata["experience_checks"] = [
        {
            "check_id": "experience_check_1",
            "completed_tool_rounds": 0,
            "wake_reasons": ["initial_brain_decision"],
            "intervene": True,
            "selected_memory_ids": ["mem_recent"],
            "guidance": "Try the visible named anchor.",
        }
    ]
    context = ContextBuilder(config).build_experience_context(
        state,
        wake_reasons=["scheduled_round"],
        proposed_decision=None,
        help_request=None,
        new_tool_results=[ToolResult(tool_name="web_search", success=False, error="NO_RESULTS")],
        candidate_memories=[],
    )

    query = ContextBuilder(config).build_experience_retrieval_query(context)

    assert "STATIC_VISUAL_DETAIL" not in query
    assert "STATIC_OCR_DETAIL" not in query
    assert "NO_RESULTS" in query
    assert "mem_recent" in query


def test_experience_agent_retries_unknown_memory_selection(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    state = GeoLocalizationState(image_path="demo.jpg")
    responses = [
        {
            "intervene": True,
            "selected_memory_ids": ["mem_unknown"],
            "guidance": "Use the unknown memory.",
            "applicability_reason": "It seems related.",
            "next_check": {"after_rounds": None, "after_tool": None, "reason": ""},
        },
        {
            "intervene": True,
            "selected_memory_ids": ["mem_allowed"],
            "guidance": "Verify the generic text with an independent named anchor.",
            "applicability_reason": "The candidate directly addresses the unresolved ambiguity.",
            "next_check": {
                "after_rounds": None,
                "after_tool": {"tool_name": "ocr", "scope": "next_call"},
                "reason": "Review the next OCR result.",
            },
        },
    ]

    def fake_generate(self, prompt, **kwargs):
        return {
            "model": "test-memory-model",
            "provider": "test",
            "text": json.dumps(responses.pop(0)),
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    advice = MemoryManagerAgent(app_config=config).advise_context(
        state,
        {
            "candidate_memories": [{"memory_id": "mem_allowed"}],
            "available_tool_names": ["ocr"],
        },
    )

    assert advice.selected_memory_ids == ["mem_allowed"]
    assert advice.next_check.after_tool is not None
    assert advice.next_check.after_tool.tool_name == "ocr"
    assert state.api_call_count["model:experience_advisor"] == 2


def test_experience_schedule_supports_tool_or_round_trigger():
    config = load_app_config(config_dir="configs", env_file=None)

    class AvailablePOI:
        hidden = False

        def is_available(self, state=None):
            return True

    workflow = ReactWorkflow(app_config=config, tools={"poi_search": AvailablePOI()})
    state = GeoLocalizationState(image_path="demo.jpg")
    state.metadata["completed_tool_rounds"] = 2
    workflow._apply_experience_advice(
        state,
        advice=ExperienceAdvice(
            intervene=False,
            next_check=ExperienceNextCheck(
                after_rounds=4,
                after_tool=ExperienceToolCheck(tool_name="poi_search"),
                reason="Review the next POI result or wait four rounds.",
            ),
        ),
        candidate_memories=[],
        check_id="experience_check_1",
    )

    state.metadata["completed_tool_rounds"] = 3
    request = ToolRequest(tool_name="poi_search", arguments={})
    reasons = workflow._scheduled_experience_check_reasons(
        state,
        [ToolResult(tool_name="poi_search", success=False, error="No results")],
        [request],
    )

    assert reasons == ["scheduled_tool:poi_search"]
    assert state.metadata["experience_next_check"]["due_tool_round"] == 6


def test_experience_schedule_can_target_a_specific_pending_call():
    config = load_app_config(config_dir="configs", env_file=None)

    class AvailablePOI:
        hidden = False

        def is_available(self, state=None):
            return True

    workflow = ReactWorkflow(app_config=config, tools={"poi_search": AvailablePOI()})
    state = GeoLocalizationState(image_path="demo.jpg")
    target_request = ToolRequest(tool_name="poi_search", arguments={"query": "target"})
    other_request = ToolRequest(tool_name="poi_search", arguments={"query": "other"})
    workflow._apply_experience_advice(
        state,
        advice=ExperienceAdvice(
            intervene=False,
            next_check=ExperienceNextCheck(
                after_tool=ExperienceToolCheck(
                    tool_name="poi_search",
                    scope="pending_call",
                    request_id=target_request.request_id,
                ),
                reason="Review the selected pending call.",
            ),
        ),
        candidate_memories=[],
        check_id="experience_check_1",
    )

    result = ToolResult(tool_name="poi_search", success=True, data={"results": []})
    state.metadata["completed_tool_rounds"] = 1
    assert workflow._scheduled_experience_check_reasons(state, [result], [other_request]) == []
    assert workflow._scheduled_experience_check_reasons(state, [result], [target_request]) == [
        "scheduled_tool:poi_search"
    ]


def test_experience_check_with_no_candidates_can_still_schedule(tmp_path, monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="retrieve_only")
    workflow = ReactWorkflow(app_config=config, tools={})
    memory_agent = MemoryManagerAgent(app_config=config)
    workflow.agents = {"memory_manager": memory_agent}
    state = GeoLocalizationState(image_path="demo.jpg")
    workflow._initialize_experience_scheduler(state)

    def fake_advise_context(state, context):
        assert context["candidate_memories"] == []
        return ExperienceAdvice(
            intervene=False,
            applicability_reason="No candidate experience is available yet.",
            next_check=ExperienceNextCheck(
                after_rounds=2,
                reason="Check again after more evidence is collected.",
            ),
        )

    monkeypatch.setattr(memory_agent, "advise_context", fake_advise_context)

    assert workflow._initial_experience_check_due(state) is True
    advice = workflow._run_experience_check(
        state,
        wake_reasons=["initial_brain_decision"],
        proposed_decision=BrainDecision(
            reasoning_summary="Inspect the image.",
            action_type="final_answer",
            confidence=0.1,
            final_answer=FinalAnswer(
                location_name="Unknown",
                granularity="unknown",
                confidence=0.1,
            ),
        ),
        help_request=None,
    )
    assert advice is not None
    assert advice.intervene is False
    assert state.metadata["experience_checks"][0]["candidate_memories"] == []
    assert state.metadata["experience_next_check"]["due_tool_round"] == 2


def test_agent_scheduled_round_check_runs_before_next_brain_decision(tmp_path, monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="retrieve_only")
    manager = MemoryManager.from_config(config)
    assert manager is not None
    item = _insert_memory(manager)

    class AvailableOCR:
        hidden = False

        def is_available(self, state=None):
            return True

    workflow = ReactWorkflow(
        app_config=config,
        tools={"ocr": AvailableOCR()},
        workflow_config=config.workflows["react"].model_copy(update={"max_steps": 2}),
    )
    memory_agent = MemoryManagerAgent(app_config=config, tools=workflow.tools)
    workflow.agents = {"brain": object(), "memory_manager": memory_agent}
    brain_calls = 0
    advisor_calls = 0

    def fake_run_agent(agent_name, state):
        nonlocal brain_calls
        brain_calls += 1
        if brain_calls == 1:
            decision = BrainDecision(
                reasoning_summary="Read the storefront text.",
                action_type="call_tool",
                tool_requests=[ToolRequest(tool_name="ocr", arguments={})],
                confidence=0.3,
            )
        else:
            assert state.metadata["experience_guidance"]["selected_memory_ids"] == [item.memory_id]
            decision = BrainDecision(
                reasoning_summary="Use the newly selected strategy with the OCR result.",
                action_type="final_answer",
                confidence=0.7,
                final_answer=FinalAnswer(
                    location_name="Test location",
                    lat=35.0,
                    lon=139.0,
                    granularity="coordinates",
                    confidence=0.7,
                ),
            )
        return AgentOutput(agent_name="brain", metadata={"decision": decision})

    def fake_advise_context(state, context):
        nonlocal advisor_calls
        advisor_calls += 1
        if advisor_calls == 1:
            return ExperienceAdvice(
                intervene=False,
                applicability_reason="Wait for OCR evidence.",
                next_check=ExperienceNextCheck(
                    after_rounds=1,
                    reason="Review after one completed tool round.",
                ),
            )
        assert context["progress"]["wake_reasons"] == ["scheduled_round"]
        assert context["new_tool_results"][0]["tool_name"] == "ocr"
        return ExperienceAdvice(
            intervene=True,
            selected_memory_ids=[item.memory_id],
            guidance="Treat generic OCR as ambiguous and seek an independent named anchor.",
            applicability_reason="The OCR result remains generic.",
        )

    monkeypatch.setattr(workflow, "run_agent", fake_run_agent)
    monkeypatch.setattr(memory_agent, "advise_context", fake_advise_context)
    def fake_prepare_tool_requests(state, requests):
        result = ToolResult(tool_name="ocr", success=True, data={"text": ["Generic Shop"]})
        state.add_tool_result(result)
        return [], [result]

    monkeypatch.setattr(workflow, "_prepare_tool_requests", fake_prepare_tool_requests)
    monkeypatch.setattr("geoagent.workflows.react_workflow.validate_image_input", lambda path: path)

    state = workflow.run("image.jpg")

    assert brain_calls == 2
    assert advisor_calls == 2
    assert state.metadata["completed_tool_rounds"] == 1
    assert len(state.metadata["experience_checks"]) == 2
    assert state.metadata["experience_checks"][1]["wake_reasons"] == ["scheduled_round"]


def test_repeated_automatic_checks_have_a_two_round_cooldown():
    config = load_app_config(config_dir="configs", env_file=None)

    class AvailablePOI:
        hidden = False

        def is_available(self, state=None):
            return True

    workflow = ReactWorkflow(app_config=config, tools={"poi_search": AvailablePOI()})
    state = GeoLocalizationState(image_path="demo.jpg")
    state.metadata["completed_tool_rounds"] = 1
    state.metadata["experience_checks"] = [{"check_id": "experience_check_1"}]
    workflow._apply_experience_advice(
        state,
        advice=ExperienceAdvice(
            intervene=False,
            next_check=ExperienceNextCheck(
                after_rounds=1,
                after_tool=ExperienceToolCheck(tool_name="poi_search"),
                reason="Check the next result.",
            ),
        ),
        candidate_memories=[],
        check_id="experience_check_2",
    )

    request = ToolRequest(tool_name="poi_search", arguments={})
    result = ToolResult(tool_name="poi_search", success=False, error="No results")
    state.metadata["completed_tool_rounds"] = 2
    assert workflow._scheduled_experience_check_reasons(state, [result], [request]) == []
    state.metadata["completed_tool_rounds"] = 3
    assert workflow._scheduled_experience_check_reasons(state, [result], [request]) == [
        "scheduled_round",
        "scheduled_tool:poi_search",
    ]
    assert state.metadata["experience_next_check"]["requested_after_rounds"] == 1
    assert state.metadata["experience_next_check"]["after_rounds"] == 2


def test_brain_can_request_experience_before_its_action(tmp_path, monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="retrieve_only")
    manager = MemoryManager.from_config(config)
    assert manager is not None
    item = _insert_memory(manager)
    workflow_config = config.workflows["react"].model_copy(
        update={
            "max_steps": 1,
            "extra": {
                **config.workflows["react"].extra,
                "experience_scheduling": {"initial_check": False},
            },
        }
    )
    workflow = ReactWorkflow(app_config=config, tools={}, workflow_config=workflow_config)
    memory_agent = MemoryManagerAgent(app_config=config)
    workflow.agents = {"brain": object(), "memory_manager": memory_agent}
    brain_calls = 0

    def fake_run_agent(agent_name, state):
        nonlocal brain_calls
        brain_calls += 1
        request = "How should I handle a generic chain name?" if brain_calls == 1 else None
        decision = BrainDecision(
            reasoning_summary="Submit the best estimate.",
            experience_request=request,
            action_type="final_answer",
            confidence=0.5,
            final_answer=FinalAnswer(
                location_name="Test location",
                lat=35.0,
                lon=139.0,
                granularity="city",
                confidence=0.5,
            ),
        )
        return AgentOutput(agent_name="brain", metadata={"decision": decision})

    def fake_advise_context(state, context):
        assert context["progress"]["wake_reasons"] == ["brain_request"]
        assert context["brain_help_request"] == "How should I handle a generic chain name?"
        return ExperienceAdvice(
            intervene=True,
            requires_brain_revision=True,
            selected_memory_ids=[item.memory_id],
            guidance="Do not choose a branch without an independent local anchor.",
            applicability_reason="The Brain explicitly identified unresolved chain ambiguity.",
        )

    monkeypatch.setattr(workflow, "run_agent", fake_run_agent)
    monkeypatch.setattr(memory_agent, "advise_context", fake_advise_context)
    monkeypatch.setattr("geoagent.workflows.react_workflow.validate_image_input", lambda path: path)

    state = workflow.run("image.jpg")

    assert brain_calls == 2
    assert state.metadata["experience_checks"][0]["wake_reasons"] == ["brain_request"]


def test_merge_target_requires_threshold_and_matching_dimensions(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path)
    manager = MemoryManager.from_config(config)
    assert manager is not None
    original = _insert_memory(manager)
    candidate = _candidate(episode_id="episode_second")

    assert manager.settings.merge_similarity_threshold == 0.9
    assert manager._merge_target_issues(original, candidate, 0.9) == []
    assert "below threshold" in " ".join(
        manager._merge_target_issues(original, candidate, 0.899)
    )
    for key, value in {
        "failure_type": "query_design_error",
        "cue_category": "traffic_system",
        "tool_scenario": "web_search",
    }.items():
        mismatched = candidate.model_copy(
            update={"metadata": {**candidate.metadata, key: value}}
        )
        assert f"{key} does not match" in " ".join(
            manager._merge_target_issues(original, mismatched, 0.95)
        )


def test_memory_manager_rejects_disabled_chroma(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    memory_config = _memory_config(tmp_path)
    memory_config["chroma"]["enabled"] = False
    config.system["memory"] = memory_config

    with pytest.raises(ValueError, match="Chroma must be enabled"):
        MemoryManager.from_config(config)


def test_memory_manager_requires_explicit_storage_directory(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    memory_config = _memory_config(tmp_path)
    memory_config.pop("storage_dir")
    config.system["memory"] = memory_config

    with pytest.raises(ValueError, match="memory.storage_dir is required"):
        MemoryManager.from_config(config)


def test_chroma_query_failure_is_not_silently_downgraded(tmp_path, monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="retrieve_only")
    manager = MemoryManager.from_config(config)
    assert manager is not None
    _insert_memory(manager)

    def fail_query(*args, **kwargs):
        raise RuntimeError("Chroma memory index query failed: unavailable")

    monkeypatch.setattr(manager.index, "query", fail_query)

    with pytest.raises(RuntimeError, match="Chroma memory index query failed"):
        manager.retrieve("generic storefront OCR", top_k=1)


def test_missing_chroma_entries_are_rebuilt_from_sqlite(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="retrieve_only")
    manager = MemoryManager.from_config(config)
    assert manager is not None
    item = _insert_memory(manager)
    manager.index._items.clear()

    reloaded = MemoryManager.from_config(config)

    hits = reloaded.retrieve(item.searchable_text(), top_k=1)
    assert hits[0].item.memory_id == item.memory_id
    assert hits[0].retrieval_source == "chroma"


def test_memory_manager_merges_the_candidate_selected_by_review(tmp_path, monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="learn_only")
    manager = MemoryManager.from_config(config)
    assert manager is not None
    original = _insert_memory(manager)
    original_search = manager._search_memories

    def high_similarity_search(query_text, top_k):
        hits = original_search(query_text, top_k)
        for hit in hits:
            if hit.item.memory_id == original.memory_id:
                hit.score = 0.95
        return hits

    monkeypatch.setattr(manager, "_search_memories", high_similarity_search)

    def fake_generate(self, prompt, **kwargs):
        payload = _memory_candidate_payload(original.memory_id)
        payload["candidate"]["situation"] = "urban storefront OCR without a unique address"
        payload["candidate"]["lesson"] = "Generic storefront text needs independent corroboration."
        payload["candidate"]["action_policy"] = [
            "Verify a unique address or administrative anchor before finalizing."
        ]
        payload["candidate"]["applicable_conditions"] = ["ocr:generic_storefront"]
        payload["candidate"]["failure_conditions"] = ["ocr:unique_address"]
        return {
            "provider": "test",
            "text": json.dumps(payload),
            "usage": {},
        }

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    monkeypatch.setattr(manager, "_reverse_geocode_tool", lambda: None)
    state = _failed_ocr_state()

    manager.record_episode_and_update(state)

    memories = manager.store.list_memories()
    assert state.metadata["memory_update"]["status"] == "similar_memory_merged"
    assert len(memories) == 1
    assert memories[0].memory_id == original.memory_id
    assert memories[0].merge_count == 1
    assert memories[0].action_policy == [
        "Verify a unique address or administrative anchor before finalizing."
    ]
    assert memories[0].applicable_conditions == ["ocr:generic_storefront"]


def test_unverified_episode_cannot_write_memory(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path)
    manager = MemoryManager.from_config(config)
    assert manager is not None
    state = _failed_ocr_state()
    state.metadata.pop("ground_truth")
    state.metadata.pop("feedback_type")

    manager.record_episode_and_update(state)

    assert state.metadata["memory_update"]["status"] == "episode_recorded"
    assert manager.store.list_memories() == []
    stored = manager.store.get_episode(state.metadata["memory_episode_id"])
    assert stored["feedback_type"] == "none"

def test_episode_outcome_uses_declared_prediction_granularity_without_explicit_target(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path)
    manager = MemoryManager.from_config(config)
    assert manager is not None
    state = GeoLocalizationState(image_path="correct_city.jpg")
    state.final_answer = FinalAnswer(
        location_name="Tokyo",
        region="Tokyo",
        country="Japan",
        granularity="city",
        confidence=0.8,
    )

    outcome = manager._evaluate_hierarchical_outcome(
        state,
        {
            "city": "Tokyo",
            "country": "Japan",
            "latitude": 35.6812,
            "longitude": 139.7671,
        },
    )

    assert outcome.success is True
    assert outcome.label == "full_success"
    assert outcome.evaluated_granularity == "city"
    assert outcome.level_correctness["country"] is True
    assert outcome.level_correctness["city"] is True
    assert outcome.level_correctness["coordinates"] is False


def test_episode_outcome_respects_explicit_coordinate_target_and_records_partial_success(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path)
    manager = MemoryManager.from_config(config)
    assert manager is not None
    state = GeoLocalizationState(image_path="correct_city_without_coordinates.jpg")
    state.metadata["target_granularity"] = "coordinates"
    state.metadata["ground_truth"] = {
        "city": "Tokyo",
        "country": "Japan",
        "latitude": 35.6812,
        "longitude": 139.7671,
    }
    state.metadata["feedback_type"] = "ground_truth"
    state.final_answer = FinalAnswer(
        location_name="Tokyo",
        region="Tokyo",
        country="Japan",
        granularity="city",
        confidence=0.8,
    )

    outcome = manager._evaluate_hierarchical_outcome(
        state,
        state.metadata["ground_truth"],
    )

    assert outcome.success is False
    assert outcome.label == "partial_success"
    assert outcome.target_granularity == "coordinates"
    assert outcome.level_correctness["city"] is True
    assert outcome.level_correctness["coordinates"] is False


def test_episode_outcome_does_not_accept_same_country_wrong_city(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path)
    manager = MemoryManager.from_config(config)
    assert manager is not None
    state = GeoLocalizationState(image_path="wrong_city.jpg")
    state.final_answer = FinalAnswer(
        location_name="Osaka",
        region="Osaka",
        country="Japan",
        granularity="city",
        confidence=0.8,
    )

    outcome = manager._evaluate_hierarchical_outcome(
        state,
        {"city": "Tokyo", "country": "Japan"},
    )

    assert outcome.success is False
    assert outcome.label == "partial_success"
    assert outcome.level_correctness["country"] is True
    assert outcome.level_correctness["city"] is False


def test_hierarchical_episode_outcome_is_persisted(tmp_path, monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["memory"] = _memory_config(tmp_path, mode="learn_only")
    _patch_memory_llm(monkeypatch)
    manager = MemoryManager.from_config(config)
    assert manager is not None
    state = GeoLocalizationState(image_path="correct_city.jpg")
    state.metadata["ground_truth"] = {
        "city": "Tokyo",
        "country": "Japan",
        "latitude": 35.6812,
        "longitude": 139.7671,
    }
    state.metadata["feedback_type"] = "ground_truth"
    state.final_answer = FinalAnswer(
        location_name="Tokyo",
        region="Tokyo",
        country="Japan",
        granularity="city",
        confidence=0.8,
    )

    manager.record_episode_and_update(state)

    with sqlite3.connect(manager.store.db_path) as connection:
        row = connection.execute(
            "SELECT success, outcome_json FROM memory_episodes WHERE episode_id = ?",
            (state.metadata["memory_episode_id"],),
        ).fetchone()
    assert row is not None and row[0] == 1
    outcome = json.loads(row[1])
    assert outcome["evaluated_granularity"] == "city"
    assert outcome["level_correctness"]["city"] is True
    assert outcome["level_correctness"]["coordinates"] is False
