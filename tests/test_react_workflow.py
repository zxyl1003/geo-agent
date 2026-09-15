from pathlib import Path

import pytest

from geoagent.core.config import load_app_config
from geoagent.core.schemas import (
    AgentOutput,
    BrainDecision,
    FinalAnswer,
    Hypothesis,
    HypothesisPatch,
    ToolRequest,
    ToolResult,
)
from geoagent.state.task_state import GeoLocalizationState
from geoagent.workflows.react_workflow import ReactWorkflow


class _AvailableTool:
    hidden = False
    internal = False

    def is_available(self, state=None):
        return True


def _workflow(max_steps=None) -> ReactWorkflow:
    config = load_app_config(config_dir="configs", env_file=None)
    workflow_config = config.workflows["react"].model_copy(update={"max_steps": max_steps})
    tools = {
        name: _AvailableTool()
        for name in ("ocr", "visual_reanalysis", "map_tile_verify", "geocode")
    }
    return ReactWorkflow(app_config=config, tools=tools, workflow_config=workflow_config)


def test_brain_accepts_multiple_independent_tool_requests():
    decision = BrainDecision(
        reasoning_summary="Search two visible POIs independently.",
        action_type="call_tools",
        tool_requests=[
            ToolRequest(tool_name="poi_search", arguments={"query": "POI A", "region": "Wuhan"}),
            ToolRequest(tool_name="poi_search", arguments={"query": "POI B", "region": "Wuhan"}),
        ],
        confidence=0.3,
    )

    assert [request.arguments["query"] for request in decision.tool_requests] == ["POI A", "POI B"]


def test_unknown_tool_is_rejected_before_execution():
    workflow = _workflow()
    state = GeoLocalizationState(image_path="demo.jpg")

    executable, results = workflow._prepare_tool_requests(
        state,
        [ToolRequest(tool_name="imaginary_geocoder", arguments={"address": "Tokyo"})],
    )

    assert executable == []
    assert len(results) == 1
    assert results[0].success is False
    assert results[0].data["requested_tool"] == "imaginary_geocoder"
    assert state.tool_calls[0].status == "error"


def test_duplicate_tool_request_emits_one_skip_log(monkeypatch):
    workflow = _workflow()
    state = GeoLocalizationState(image_path="demo.jpg")
    request = ToolRequest(tool_name="geocode", arguments={"address": "Tokyo"})
    logs = []

    def record_log(*args):
        logs.append(args)

    monkeypatch.setattr("geoagent.workflows.base.log_skip", record_log)
    monkeypatch.setattr("geoagent.workflows.react_workflow.log_skip", record_log)

    executable, results = workflow._prepare_tool_requests(state, [request])
    duplicate_executable, duplicate_results = workflow._prepare_tool_requests(state, [request])

    assert executable == [request]
    assert results == []
    assert duplicate_executable == []
    assert duplicate_results[0].error == "Duplicate tool call with same arguments."
    assert len(logs) == 1


def test_verification_tool_is_an_ordinary_brain_step(monkeypatch):
    workflow = _workflow()
    workflow.agents = {"brain": object()}
    brain_calls = 0
    executed_tools: list[str] = []

    def fake_run_agent(agent_name, state):
        nonlocal brain_calls
        brain_calls += 1
        if brain_calls == 1:
            decision = BrainDecision(
                reasoning_summary="Verify the candidate before deciding.",
                action_type="call_tool",
                tool_requests=[
                    ToolRequest(
                        tool_name="map_tile_verify",
                        arguments={"map_provider": "google", "lat": 35.72518, "lon": 139.76413},
                    )
                ],
                confidence=0.7,
            )
        else:
            decision = BrainDecision(
                reasoning_summary="The verification is sufficient.",
                action_type="final_answer",
                confidence=0.86,
                uncertainty_radius_m=250.0,
                final_answer=FinalAnswer(
                    location_name="Bunkyo, Tokyo",
                    country="Japan",
                    region="Tokyo",
                    lat=35.72518,
                    lon=139.76413,
                    granularity="coordinates",
                    confidence=0.86,
                    uncertainty_radius_m=250.0,
                ),
            )
        return AgentOutput(
            agent_name="brain",
            state_delta={"brain_decision": decision.model_dump()},
            metadata={"decision": decision},
        )

    def fake_execute(state, requests):
        executed_tools.extend(request.tool_name for request in requests)
        return []

    monkeypatch.setattr(workflow, "run_agent", fake_run_agent)
    monkeypatch.setattr(workflow, "_execute_requests", fake_execute)
    monkeypatch.setattr("geoagent.workflows.react_workflow.validate_image_input", lambda _: Path("demo.jpg"))

    state = workflow.run("demo.jpg")

    assert brain_calls == 2
    assert state.step_count == 2
    assert executed_tools == ["map_tile_verify"]
    assert state.status == "completed"
    assert state.final_answer is not None
    assert state.final_answer.confidence == 0.86


def test_answer_without_coordinates_fails_immediately(monkeypatch):
    workflow = _workflow(max_steps=2)
    workflow.agents = {"brain": object()}

    def fake_run_agent(agent_name, state):
        decision = BrainDecision(
            reasoning_summary="No coordinate estimate is available.",
            action_type="final_answer",
            confidence=0.2,
            final_answer=FinalAnswer(location_name="Unknown", granularity="unknown", confidence=0.2),
        )
        return AgentOutput(agent_name="brain", metadata={"decision": decision})

    monkeypatch.setattr(workflow, "run_agent", fake_run_agent)
    monkeypatch.setattr("geoagent.workflows.react_workflow.validate_image_input", lambda _: Path("demo.jpg"))

    state = workflow.run("demo.jpg")

    assert state.step_count == 1
    assert state.status == "failed"
    assert state.metadata["failure_reason"] == "final_answer_without_coordinates"


def test_max_steps_creates_only_an_empty_coordinate_failure_record(monkeypatch):
    workflow = _workflow(max_steps=2)
    workflow.agents = {"brain": object()}

    def fake_run_agent(agent_name, state):
        decision = BrainDecision(
            reasoning_summary="Continue investigating.",
            action_type="call_tool",
            tool_requests=[ToolRequest(tool_name="ocr", arguments={})],
            confidence=0.2,
        )
        return AgentOutput(agent_name="brain", metadata={"decision": decision})

    monkeypatch.setattr(workflow, "run_agent", fake_run_agent)
    monkeypatch.setattr(workflow, "_execute_requests", lambda state, requests: [])
    monkeypatch.setattr("geoagent.workflows.react_workflow.validate_image_input", lambda _: Path("demo.jpg"))

    state = workflow.run("demo.jpg")

    assert state.step_count == 2
    assert state.status == "failed"
    assert state.final_answer is not None
    assert state.final_answer.lat is None
    assert state.final_answer.lon is None
    assert state.final_answer.location_name == "Unknown"
    assert state.metadata["failure_reason"] == "max_steps_exhausted"


@pytest.mark.parametrize("tool_name", ["ocr", "visual_reanalysis"])
def test_workflow_forces_current_image_path(tool_name):
    workflow = _workflow()
    state = GeoLocalizationState(image_path="datasets/images/real-image.jpg")
    request = ToolRequest(tool_name=tool_name, arguments={"image_path": "wrong.jpg"})

    workflow._normalize_tool_request(state, request)

    assert request.arguments["image_path"] == state.image_path


def test_python_does_not_infer_hypothesis_granularity():
    workflow = _workflow()
    state = GeoLocalizationState(image_path="demo.jpg")
    hypothesis = Hypothesis(name="Candidate", lat=35.0, lon=139.0, granularity="unknown", score=0.6)

    workflow._apply_hypothesis_patch(state, HypothesisPatch(add=[hypothesis]))

    assert state.hypotheses[0].granularity == "unknown"


def test_geocode_result_does_not_mutate_brain_hypothesis():
    workflow = _workflow()
    state = GeoLocalizationState(
        image_path="demo.jpg",
        hypotheses=[Hypothesis(name="Candidate", country="Japan", granularity="country", score=0.6)],
    )

    workflow.merge_tool_result(
        state,
        ToolResult(
            tool_name="geocode",
            success=True,
            data={"top_result": {"lat": 35.0, "lon": 139.0, "formatted_address": "Example"}},
        ),
    )

    assert state.hypotheses[0].lat is None
    assert state.hypotheses[0].lon is None
    assert state.hypotheses[0].granularity == "country"
