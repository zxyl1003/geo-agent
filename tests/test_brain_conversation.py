import json

import pytest

from geoagent.agents.brain_agent import BrainAgent
from geoagent.core.config import load_app_config
from geoagent.core.context_builder import ContextBuilder
from geoagent.core.schemas import ToolResult
from geoagent.models.llm_client import LLMClient
from geoagent.state.task_state import GeoLocalizationState


def test_brain_agent_sends_recent_conversation_messages(monkeypatch):
    captured = {}

    def fake_generate(self, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["messages"] = kwargs["messages"]
        return {
            "text": json.dumps(
                {
                    "reasoning_summary": "Continue from prior tool observation.",
                    "action_type": "final_answer",
                    "tool_requests": [],
                    "hypothesis_patch": None,
                    "confidence": 0.2,
                    "uncertainty_radius_m": 1000000,
                    "final_answer": {
                        "location_name": "Best estimate",
                        "lat": 0.0,
                        "lon": 0.0,
                        "granularity": "continent",
                        "confidence": 0.2,
                        "uncertainty_radius_m": 1000000,
                    },
                }
            )
        }

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    config = load_app_config(config_dir="configs", env_file=None)
    state = GeoLocalizationState(image_path="examples/images/demo.jpg", user_query="Where is this?")
    state.add_brain_message("tool", "previous web search result", name="web_search")

    agent = BrainAgent(app_config=config, tools={})
    decision = agent.decide(state)

    assert decision.action_type == "final_answer"
    assert any("previous web search result" in message["content"] for message in captured["messages"])
    assert captured["messages"][-1]["role"] == "user"
    assert "Compact State" in captured["messages"][-1]["content"]
    assert state.brain_messages[-1].role == "assistant"


def test_brain_agent_does_not_compact_before_input_budget(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["brain_conversation"] = {
        "context_window": 100000,
        "safety_margin_tokens": 100,
        "preserve_recent_rounds": 3,
        "summary_max_tokens": 200,
    }
    config.models["brain"].max_tokens = 200
    state = GeoLocalizationState(image_path="examples/images/demo.jpg")
    for index in range(11):
        state.add_brain_message("assistant", f"decision-{index}", name="brain")
        state.add_brain_message("tool", f"result-{index}", name="web_search")

    def unexpected_generate(self, prompt, **kwargs):
        raise AssertionError("conversation below the token budget must not be summarized")

    monkeypatch.setattr(LLMClient, "generate", unexpected_generate)
    agent = BrainAgent(app_config=config, tools={})
    monkeypatch.setattr(agent, "_system_prompt", lambda: "system")

    context = ContextBuilder(config).build_brain_context(state)
    messages = agent._build_messages(state, context)

    assert len(state.brain_messages) == 22
    assert state.brain_conversation_summary is None
    assert any("decision-0" in message["content"] for message in messages)


def test_brain_agent_reserves_multimodal_output_budget_within_ninety_percent_cap():
    config = load_app_config(config_dir="configs", env_file=None)
    agent = BrainAgent(app_config=config, tools={})

    assert agent._brain_input_budget() == 56464


def test_brain_agent_compacts_first_eight_of_eleven_rounds_and_deduplicates_latest_result(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["brain_conversation"] = {
        "context_window": 3000,
        "safety_margin_tokens": 100,
        "preserve_recent_rounds": 3,
        "summary_max_tokens": 200,
    }
    config.models["brain"].max_tokens = 200
    state = GeoLocalizationState(image_path="examples/images/demo.jpg")
    for index in range(11):
        state.add_brain_message("assistant", f"decision-{index} " + "x" * 300, name="brain")
        state.add_brain_message("tool", f"result-{index} " + "y" * 300, name="web_search")

    latest = "latest-result-marker"
    state.add_tool_result(
        ToolResult(
            tool_name="web_search",
            success=True,
            data={"query": latest},
        )
    )
    state.brain_messages[-1].content = json.dumps({"tool_name": "web_search", "query": latest})
    agent = BrainAgent(app_config=config, tools={})
    monkeypatch.setattr(agent, "_system_prompt", lambda: "system")

    summary_prompts = []

    def fake_generate(self, prompt, **kwargs):
        summary_prompts.append(kwargs)
        return {
            "text": json.dumps(
                {
                    "observed_evidence": ["Earlier visual evidence was recorded."],
                    "candidate_locations": [],
                    "tool_outcomes": [
                        {"tool": "web_search", "outcome": "Earlier searches were inconclusive."}
                    ],
                    "contradictions": [],
                    "unresolved_questions": ["The exact place remains unresolved."],
                    "memory_references": [],
                }
            )
        }

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    context = ContextBuilder(config).build_brain_context(state)
    messages = agent._build_messages(state, context)

    assert agent._estimate_messages_tokens(messages) <= agent._brain_input_budget()
    assert len(summary_prompts) == 1
    assert summary_prompts[0]["json_mode"] is True
    assert any("decision-0" in message["content"] for message in summary_prompts[0]["messages"])
    assert any("decision-7" in message["content"] for message in summary_prompts[0]["messages"])
    assert not any("decision-8" in message["content"] for message in summary_prompts[0]["messages"])
    assert len(state.brain_messages) == 6
    assert state.brain_messages[0].content.startswith("decision-8")
    assert json.loads(state.brain_conversation_summary)["observed_evidence"]
    assert json.loads(state.brain_conversation_summary)["tool_outcomes"][0]["tool"] == "web_search"
    assert sum(latest in message["content"] for message in messages) == 1


def test_brain_agent_rejects_invalid_structured_summary_without_dropping_history(monkeypatch):
    config = load_app_config(config_dir="configs", env_file=None)
    config.system["brain_conversation"] = {
        "context_window": 1800,
        "safety_margin_tokens": 100,
        "preserve_recent_rounds": 3,
        "summary_max_tokens": 200,
    }
    config.models["brain"].max_tokens = 200
    state = GeoLocalizationState(image_path="examples/images/demo.jpg")
    for index in range(5):
        state.add_brain_message("assistant", f"decision-{index} " + "x" * 500, name="brain")
        state.add_brain_message("tool", f"result-{index} " + "y" * 500, name="web_search")
    original_messages = list(state.brain_messages)

    summary_requests = []

    def invalid_summary(self, prompt, **kwargs):
        summary_requests.append(kwargs["messages"])
        return {"text": "not json"}

    monkeypatch.setattr(LLMClient, "generate", invalid_summary)
    agent = BrainAgent(app_config=config, tools={})
    monkeypatch.setattr(agent, "_system_prompt", lambda: "system")

    with pytest.raises(RuntimeError, match="invalid structured conversation summary"):
        agent._build_messages(state, ContextBuilder(config).build_brain_context(state))

    assert state.brain_messages == original_messages
    assert state.brain_conversation_summary is None
    assert len(summary_requests) == 3
    assert all("Return a shorter, complete JSON object" in request[-1]["content"] for request in summary_requests[1:])
