"""LLM-assisted, policy-gated manager for reusable geolocation memories."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from geoagent.agents.base import BaseAgent
from geoagent.core.json_utils import extract_json_payload
from geoagent.core.prompt_loader import load_prompt
from geoagent.core.registry import agent_registry
from geoagent.core.schemas import AgentOutput
from geoagent.memory.models import ExperienceAdvice, MemoryAgentReview
from geoagent.models.llm_client import LLMClient
from geoagent.state.task_state import GeoLocalizationState


@agent_registry.register("memory_manager")
class MemoryManagerAgent(BaseAgent):
    """Diagnose episodes and propose memories without mutating the store.

    The agent is intentionally advisory. ``MemoryManager`` validates proposals
    and owns persistence and similarity merging.
    """

    name = "memory_manager"
    description = "Diagnoses geolocation episodes and proposes reusable strategy memories."

    def run(self, state: GeoLocalizationState) -> AgentOutput:
        context = state.metadata.get("memory_review_context")
        if not isinstance(context, dict):
            review = MemoryAgentReview(rationale="No memory review context was supplied.")
        else:
            review = self.review_context(state, context)

        state.metadata["memory_agent_review"] = review.model_dump(mode="json")
        state.touch()
        return AgentOutput(
            agent_name=self.name,
            message=review.rationale or review.diagnosis,
            state_delta={"memory_agent_review": review.model_dump(mode="json")},
            metadata={"review": review},
        )

    def review_context(self, state: GeoLocalizationState, context: dict[str, Any]) -> MemoryAgentReview:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    "Review this completed geolocation episode. Return one JSON object only.\n\n"
                    + json.dumps(context, ensure_ascii=False, indent=2, default=str)
                ),
            },
        ]
        model_config = self.app_config.models.get("memory_manager") or self.app_config.models.get("brain")
        response = LLMClient(app_config=self.app_config, model_role="memory_manager").generate(
            messages[-1]["content"],
            messages=messages,
            json_mode=True,
            temperature=model_config.temperature if model_config else 0.0,
            max_tokens=model_config.max_tokens if model_config else 1800,
            thinking=model_config.thinking if model_config else {},
        )
        self.record_model_usage(state, "memory_manager", response)
        try:
            payload = extract_json_payload(str(response.get("text") or ""))
            normalized = self._normalize_payload(payload, context)
            review = MemoryAgentReview.model_validate(normalized)
            review.proposal_source = "llm"
            if review.candidate is not None:
                review.candidate.source_episode_id = str(context.get("episode_id") or "")
                review.candidate.feedback_type = str(context.get("feedback_type") or "none")
                review.candidate.diagnosis_confidence = max(
                    review.candidate.diagnosis_confidence,
                    review.diagnosis_confidence,
                )
            return review
        except (ValueError, ValidationError, json.JSONDecodeError, TypeError) as exc:
            state.metadata["memory_agent_parse_error"] = str(exc)
            return MemoryAgentReview(
                should_write=False,
                rationale=f"LLM review response could not be parsed: {exc}",
                proposal_source="none",
            )

    def advise_context(self, state: GeoLocalizationState, context: dict[str, Any]) -> ExperienceAdvice:
        """Select useful retrieved memories and schedule the next online check."""

        base_messages = [
            {"role": "system", "content": self._online_system_prompt()},
            {
                "role": "user",
                "content": (
                    "Assess the current geolocation state and the retrieved candidate memories. "
                    "Return one JSON object only.\n\n"
                    + json.dumps(context, ensure_ascii=False, indent=2, default=str)
                ),
            },
        ]
        model_config = self.app_config.models.get("memory_manager") or self.app_config.models.get("brain")
        retry_instruction = ""
        last_error: Exception | None = None
        for attempt in range(3):
            messages = list(base_messages)
            if retry_instruction:
                messages.append({"role": "user", "content": retry_instruction})
            response = LLMClient(app_config=self.app_config, model_role="memory_manager").generate(
                messages[-1]["content"],
                messages=messages,
                json_mode=True,
                temperature=model_config.temperature if model_config else 0.0,
                max_tokens=model_config.max_tokens if model_config else 1800,
                thinking=model_config.thinking if model_config else {},
            )
            self.record_model_usage(state, "experience_advisor", response)
            try:
                payload = extract_json_payload(str(response.get("text") or ""))
                advice = ExperienceAdvice.model_validate(payload)
                self._validate_online_advice(advice, context)
                return advice
            except (ValueError, ValidationError, json.JSONDecodeError, TypeError) as exc:
                last_error = exc
                if attempt < 2:
                    retry_instruction = (
                        f"The previous response was invalid: {exc}. Return a corrected JSON object that uses only "
                        "candidate memory IDs and available tool names from the supplied context."
                    )

        state.metadata.setdefault("experience_advisor_parse_errors", []).append(str(last_error))
        return ExperienceAdvice(
            applicability_reason=f"Experience advisor returned invalid output: {last_error}",
        )

    def _system_prompt(self) -> str:
        return load_prompt(self.app_config.config_dir, "memory_manager_agent.md")

    def _online_system_prompt(self) -> str:
        return load_prompt(self.app_config.config_dir, "experience_advisor_agent.md")

    def _validate_online_advice(self, advice: ExperienceAdvice, context: dict[str, Any]) -> None:
        candidate_ids = {
            str(item.get("memory_id"))
            for item in context.get("candidate_memories", [])
            if isinstance(item, dict) and item.get("memory_id")
        }
        selected_ids = [str(memory_id) for memory_id in advice.selected_memory_ids]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected_memory_ids must not contain duplicates.")
        invalid_ids = sorted(set(selected_ids) - candidate_ids)
        if invalid_ids:
            raise ValueError(f"Unknown candidate memory IDs: {', '.join(invalid_ids)}")

        proposed = context.get("proposed_brain_decision")
        if advice.requires_brain_revision and not isinstance(proposed, dict):
            raise ValueError("Brain revision requires a pending proposed_brain_decision.")

        after_tool = advice.next_check.after_tool
        available_tools = {str(name) for name in context.get("available_tool_names", [])}
        if after_tool is not None and after_tool.tool_name not in available_tools:
            raise ValueError(f"Scheduled tool is not available: {after_tool.tool_name}")
        if after_tool is not None and after_tool.scope == "pending_call":
            proposed = context.get("proposed_brain_decision")
            requests = proposed.get("tool_requests", []) if isinstance(proposed, dict) else []
            matching_request = next(
                (
                    request
                    for request in requests
                    if isinstance(request, dict) and request.get("request_id") == after_tool.request_id
                ),
                None,
            )
            if matching_request is None:
                raise ValueError(f"Unknown pending request ID: {after_tool.request_id}")
            if matching_request.get("tool_name") != after_tool.tool_name:
                raise ValueError("Scheduled pending request does not match tool_name.")

    def _normalize_payload(self, payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        candidate = normalized.get("candidate")
        if isinstance(candidate, dict):
            candidate = dict(candidate)
            candidate["source_episode_id"] = str(context.get("episode_id") or "")
            candidate["feedback_type"] = str(context.get("feedback_type") or "none")
            candidate.setdefault("diagnosis_confidence", normalized.get("diagnosis_confidence", 0.0))
            normalized["candidate"] = candidate
        normalized.setdefault("should_write", candidate is not None)
        normalized.setdefault("merge_memory_id", None)
        normalized.setdefault("proposal_source", "llm")
        return normalized
