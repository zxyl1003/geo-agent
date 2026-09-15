"""LLM-driven ReAct workflow for geolocation."""

from __future__ import annotations

import json
from time import monotonic
from typing import Any

from geoagent.agents.brain_agent import BrainAgent
from geoagent.agents.memory_manager_agent import MemoryManagerAgent
from geoagent.core.context_builder import ContextBuilder
from geoagent.core.logging import log_skip
from geoagent.core.image_validation import validate_image_input
from geoagent.core.registry import workflow_registry
from geoagent.core.schemas import BrainDecision, FinalAnswer, HypothesisPatch, ToolCall, ToolRequest
from geoagent.core.schemas import ToolResult
from geoagent.memory.manager import MemoryManager
from geoagent.memory.models import ExperienceAdvice
from geoagent.state.task_state import GeoLocalizationState
from geoagent.workflows.base import BaseWorkflow

@workflow_registry.register("react")
class ReactWorkflow(BaseWorkflow):
    name = "react"

    def _default_agents(self):
        return {
            "brain": BrainAgent(app_config=self.app_config, tools=self.tools),
            "memory_manager": MemoryManagerAgent(app_config=self.app_config, tools=self.tools),
        }

    def run(
        self,
        input_image_path: str,
        user_query: str | None = None,
        task_metadata: dict[str, Any] | None = None,
    ) -> GeoLocalizationState:
        validated_image_path = validate_image_input(input_image_path)
        if not self.agents:
            self.agents = self._default_agents()

        state = GeoLocalizationState(image_path=str(validated_image_path), user_query=user_query, status="running")
        state.metadata["workflow"] = "react"
        self._apply_task_metadata(state, task_metadata)
        state.metadata["phase"] = "initial_brain_decision"
        self._initialize_experience_scheduler(state)
        self.log_state_checkpoint(state, "initialized react workflow")

        while self.max_steps is None or state.step_count < self.max_steps:
            state.step_count += 1
            self.log_state_checkpoint(state, "react step started")
            self._mark_pending_experience_delivered(state)
            brain_output = self.run_agent("brain", state)
            decision = brain_output.metadata.get("decision")
            if not isinstance(decision, BrainDecision):
                decision = BrainDecision.model_validate(brain_output.state_delta.get("brain_decision", {}))
            self._record_brain_decision_after_experience(state, decision)
            if state.step_count == 1:
                state.metadata["phase"] = "investigation"

            wake_reasons: list[str] = []
            if state.step_count == 1 and self._initial_experience_check_due(state):
                wake_reasons.append("initial_brain_decision")
            if decision.experience_request:
                wake_reasons.append("brain_request")
            advice = None
            if wake_reasons:
                advice = self._run_experience_check(
                    state,
                    wake_reasons=wake_reasons,
                    proposed_decision=decision,
                    help_request=decision.experience_request,
                )
            if advice is not None and advice.intervene and advice.requires_brain_revision:
                state.metadata["pending_experience_revision"] = {
                    "status": "pending_not_executed",
                    "instruction": (
                        "The proposed action below has not run. Revise it using the experience guidance "
                        "without claiming or inventing any tool outcome."
                    ),
                    "proposed_decision": self._compact_brain_decision(decision),
                }
                try:
                    self._mark_pending_experience_delivered(state)
                    revised_output = self.run_agent("brain", state)
                finally:
                    state.metadata.pop("pending_experience_revision", None)
                revised_decision = revised_output.metadata.get("decision")
                if isinstance(revised_decision, BrainDecision):
                    decision = revised_decision
                else:
                    decision = BrainDecision.model_validate(
                        revised_output.state_delta.get("brain_decision", {})
                    )
                self._record_brain_decision_after_experience(
                    state,
                    decision,
                    revision_performed=True,
                )

            self._record_executed_decision_for_current_check(state, decision)

            self._apply_hypothesis_patch(state, decision.hypothesis_patch)
            self._update_uncertainty_from_decision(state, decision)

            if decision.action_type == "final_answer":
                state.final_answer = self._final_answer_from_decision(decision)
                state.metadata["experience_next_check"] = None
                break

            requests = self._requests_from_decision(decision)
            if not requests:
                continue

            self._execute_requests(state, requests)

        max_steps_exhausted = (
            state.final_answer is None
            and self.max_steps is not None
            and state.step_count >= self.max_steps
        )
        if max_steps_exhausted:
            state.final_answer = self.build_final_answer(state)
        if max_steps_exhausted:
            state.status = "failed"
            state.metadata["failure_reason"] = "max_steps_exhausted"
        elif self._has_coordinates(state.final_answer):
            state.status = "completed"
        else:
            state.status = "failed"
            state.metadata["failure_reason"] = "final_answer_without_coordinates"
        state.touch()
        self._record_external_memory_episode(state)
        self.log_state_checkpoint(state, f"{state.status} react workflow")
        return state

    def _execute_requests(self, state: GeoLocalizationState, requests: list[ToolRequest]) -> list[ToolResult]:
        """Prepare, execute, and expose a batch of tool results to Brain."""

        executable_requests, immediate_results = self._prepare_tool_requests(state, requests)
        results = list(immediate_results)
        if executable_requests:
            results.extend(self.execute_tool_requests(state, executable_requests))
        for result in results:
            self.append_brain_tool_observation(state, result)

        self._rank_hypotheses(state)
        state.metadata["completed_tool_rounds"] = int(
            state.metadata.get("completed_tool_rounds", 0)
        ) + 1
        if self.max_steps is not None and state.step_count >= self.max_steps:
            state.metadata["experience_next_check"] = None
        else:
            wake_reasons = self._scheduled_experience_check_reasons(state, results, requests)
            if wake_reasons:
                self._run_experience_check(
                    state,
                    wake_reasons=wake_reasons,
                    proposed_decision=None,
                    help_request=None,
                )
        self.log_state_checkpoint(state, "react step finished")
        return results

    def _has_coordinates(self, answer: FinalAnswer | None) -> bool:
        return bool(answer is not None and answer.lat is not None and answer.lon is not None)

    def _memory_manager(self) -> MemoryManager | None:
        if not hasattr(self, "_external_memory_manager"):
            memory_agent = self.agents.get("memory_manager")
            self._external_memory_manager = MemoryManager.from_config(
                self.app_config,
                agent=memory_agent if isinstance(memory_agent, MemoryManagerAgent) else None,
            )
        return self._external_memory_manager

    def _apply_task_metadata(self, state: GeoLocalizationState, task_metadata: dict[str, Any] | None) -> None:
        if not task_metadata:
            return
        state.metadata["task_metadata"] = task_metadata
        if "ground_truth" in task_metadata:
            state.metadata["ground_truth"] = task_metadata["ground_truth"]
        else:
            ground_truth = {
                key[3:]: value
                for key, value in task_metadata.items()
                if key.startswith("gt_") and value not in {None, ""}
            }
            if ground_truth:
                state.metadata["ground_truth"] = ground_truth
        if state.metadata.get("ground_truth"):
            state.metadata["feedback_type"] = "ground_truth"

    def _initialize_experience_scheduler(self, state: GeoLocalizationState) -> None:
        self._memory_manager()
        state.metadata["experience_scheduling"] = self._experience_scheduling_config()
        state.metadata.setdefault("completed_tool_rounds", 0)
        state.metadata.setdefault("experience_checks", [])
        state.metadata.setdefault("experience_next_check", None)
        state.metadata.setdefault("experience_last_tool_result_index", 0)
        state.metadata.setdefault("recalled_memories", [])
        state.metadata.setdefault("memory_usage", [])
        state.touch()

    def _experience_scheduling_config(self) -> dict[str, bool]:
        configured = (self.workflow_config.extra if self.workflow_config else {}).get(
            "experience_scheduling",
            {},
        )
        if not isinstance(configured, dict):
            configured = {}
        return {
            "initial_check": bool(configured.get("initial_check", False)),
        }

    def _initial_experience_check_due(self, state: GeoLocalizationState) -> bool:
        if not self._experience_scheduling_config().get("initial_check", False):
            return False
        if state.metadata.get("experience_initial_check_completed"):
            return False
        state.metadata["experience_initial_check_completed"] = True
        manager = self._memory_manager()
        return bool(manager and manager.settings.retrieve_enabled)

    def _run_experience_check(
        self,
        state: GeoLocalizationState,
        *,
        wake_reasons: list[str],
        proposed_decision: BrainDecision | None,
        help_request: str | None,
    ) -> ExperienceAdvice | None:
        manager = self._memory_manager()
        memory_agent = self.agents.get("memory_manager")
        if (
            manager is None
            or not manager.settings.retrieve_enabled
            or not isinstance(memory_agent, MemoryManagerAgent)
        ):
            return None

        result_start = int(state.metadata.get("experience_last_tool_result_index", 0))
        new_tool_results = state.tool_results[result_start:]
        context_builder = ContextBuilder(self.app_config, self.tools)
        retrieval_context = context_builder.build_experience_context(
            state,
            wake_reasons=wake_reasons,
            proposed_decision=proposed_decision,
            help_request=help_request,
            new_tool_results=new_tool_results,
            candidate_memories=[],
        )
        query_text = context_builder.build_experience_retrieval_query(retrieval_context)
        hits = manager.retrieve(
            query_text,
            top_k=manager.settings.retrieve_top_k,
        )
        candidate_memories: list[dict[str, Any]] = []
        for hit in hits:
            digest = hit.prompt_digest()
            digest["usage_role"] = manager.usage_role(hit.item)
            candidate_memories.append(digest)
        self._record_retrieved_memories(state, candidate_memories, query_text)
        advisor_context = context_builder.build_experience_context(
            state,
            wake_reasons=wake_reasons,
            proposed_decision=proposed_decision,
            help_request=help_request,
            new_tool_results=new_tool_results,
            candidate_memories=candidate_memories,
        )
        usage_before = dict(
            (state.token_usage.get("by_role") or {}).get("experience_advisor") or {}
        )
        started = monotonic()
        advice = memory_agent.advise_context(state, advisor_context)
        latency_ms = (monotonic() - started) * 1000
        usage_after = dict(
            (state.token_usage.get("by_role") or {}).get("experience_advisor") or {}
        )
        check_id = f"experience_check_{len(state.metadata.get('experience_checks') or []) + 1}"
        self._apply_experience_advice(
            state,
            advice=advice,
            candidate_memories=candidate_memories,
            check_id=check_id,
        )
        state.metadata["experience_last_tool_result_index"] = len(state.tool_results)
        state.metadata.setdefault("experience_checks", []).append(
            {
                "check_id": check_id,
                "brain_step": state.step_count,
                "completed_tool_rounds": int(state.metadata.get("completed_tool_rounds", 0)),
                "wake_reasons": list(wake_reasons),
                "retrieval_query": query_text,
                "candidate_memories": [
                    {
                        "memory_id": memory.get("memory_id"),
                        "retrieved_rank": memory.get("retrieved_rank"),
                        "retrieval_score": memory.get("retrieval_score"),
                    }
                    for memory in candidate_memories
                ],
                "intervene": advice.intervene,
                "requires_brain_revision": advice.requires_brain_revision,
                "selected_memory_ids": list(advice.selected_memory_ids),
                "guidance": advice.guidance,
                "applicability_reason": advice.applicability_reason,
                "next_check": advice.next_check.model_dump(mode="json"),
                "effective_next_check": state.metadata.get("experience_next_check"),
                "proposed_brain_decision": self._compact_brain_decision(proposed_decision),
                "brain_revision_performed": False,
                "model_usage": {
                    key: int(usage_after.get(key, 0)) - int(usage_before.get(key, 0))
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
                "latency_ms": round(latency_ms, 3),
            }
        )
        state.touch()
        self.log_state_checkpoint(state, f"experience check: {check_id}")
        return advice

    def _record_retrieved_memories(
        self,
        state: GeoLocalizationState,
        memories: list[dict[str, Any]],
        query: str,
    ) -> None:
        usages = state.metadata.setdefault("memory_usage", [])
        by_id = {
            str(item.get("memory_id")): item
            for item in usages
            if isinstance(item, dict) and item.get("memory_id")
        }
        for memory in memories:
            memory_id = str(memory["memory_id"])
            usage = by_id.get(memory_id)
            if usage is None:
                usage = {
                    "memory_id": memory_id,
                    "retrieved_rank": memory.get("retrieved_rank"),
                    "retrieval_score": memory.get("retrieval_score"),
                    "was_returned_to_brain": False,
                    "was_cited_by_brain": False,
                    "usage_role": memory.get("usage_role") or "strategy_hint",
                    "recall_step": state.step_count,
                    "recall_query": query,
                    "retrieval_count": 1,
                }
                usages.append(usage)
                by_id[memory_id] = usage
            else:
                usage["retrieved_rank"] = memory.get("retrieved_rank")
                usage["retrieval_score"] = memory.get("retrieval_score")
                usage["recall_step"] = state.step_count
                usage["recall_query"] = query
                usage["retrieval_count"] = int(usage.get("retrieval_count", 1)) + 1
        state.touch()

    def _apply_experience_advice(
        self,
        state: GeoLocalizationState,
        *,
        advice: ExperienceAdvice,
        candidate_memories: list[dict[str, Any]],
        check_id: str,
    ) -> None:
        selected_ids = set(advice.selected_memory_ids)
        selected = [
            memory for memory in candidate_memories
            if str(memory.get("memory_id")) in selected_ids
        ]
        pending_check_id = state.metadata.get("experience_pending_effect_check_id")
        if pending_check_id:
            pending_check = self._experience_check(state, str(pending_check_id))
            if pending_check is not None:
                pending_check["superseded_before_brain"] = True
            state.metadata.pop("experience_pending_effect_check_id", None)
        state.metadata["recalled_memories"] = selected if advice.intervene else []
        if advice.intervene:
            state.metadata["experience_guidance"] = {
                "check_id": check_id,
                "guidance": advice.guidance,
                "applicability_reason": advice.applicability_reason,
                "selected_memory_ids": list(advice.selected_memory_ids),
            }
            state.metadata["experience_pending_effect_check_id"] = check_id
        else:
            state.metadata.pop("experience_guidance", None)

        next_check = advice.next_check
        if next_check.after_rounds is None and next_check.after_tool is None:
            state.metadata["experience_next_check"] = None
            return
        current_round = int(state.metadata.get("completed_tool_rounds", 0))
        minimum_rounds = 1 if not state.metadata.get("experience_checks") else 2
        effective_after_rounds = (
            max(next_check.after_rounds, minimum_rounds)
            if next_check.after_rounds is not None
            else None
        )
        pending_call_check = (
            next_check.after_tool is not None
            and next_check.after_tool.scope == "pending_call"
        )
        state.metadata["experience_next_check"] = {
            "scheduled_at_tool_round": current_round,
            "requested_after_rounds": next_check.after_rounds,
            "after_rounds": effective_after_rounds,
            "due_tool_round": (
                current_round + effective_after_rounds
                if effective_after_rounds is not None
                else None
            ),
            "not_before_tool_round": current_round + (1 if pending_call_check else minimum_rounds),
            "after_tool": (
                next_check.after_tool.model_dump(mode="json")
                if next_check.after_tool is not None
                else None
            ),
            "reason": next_check.reason,
        }

    def _scheduled_experience_check_reasons(
        self,
        state: GeoLocalizationState,
        results: list[ToolResult],
        requests: list[ToolRequest],
    ) -> list[str]:
        schedule = state.metadata.get("experience_next_check")
        if not isinstance(schedule, dict):
            return []

        reasons: list[str] = []
        completed_rounds = int(state.metadata.get("completed_tool_rounds", 0))
        not_before_round = schedule.get("not_before_tool_round")
        if isinstance(not_before_round, int) and completed_rounds < not_before_round:
            return []
        due_round = schedule.get("due_tool_round")
        if isinstance(due_round, int) and completed_rounds >= due_round:
            reasons.append("scheduled_round")

        after_tool = schedule.get("after_tool")
        if isinstance(after_tool, dict):
            tool_name = str(after_tool.get("tool_name") or "")
            scope = str(after_tool.get("scope") or "next_call")
            request_id = str(after_tool.get("request_id") or "")
            next_tool_completed = tool_name and any(result.tool_name == tool_name for result in results)
            pending_tool_completed = any(
                request.tool_name == tool_name and request.request_id == request_id
                for request in requests
            )
            if (
                scope == "next_call" and next_tool_completed
                or scope == "pending_call" and pending_tool_completed
            ):
                reasons.append(f"scheduled_tool:{tool_name}")
        return reasons

    def _compact_brain_decision(self, decision: BrainDecision | None) -> dict[str, Any] | None:
        if decision is None:
            return None
        return decision.model_dump(
            mode="json",
            exclude={"visual_analysis", "visual_updates"},
        )

    def _experience_check(self, state: GeoLocalizationState, check_id: str) -> dict[str, Any] | None:
        for check in reversed(state.metadata.get("experience_checks") or []):
            if isinstance(check, dict) and check.get("check_id") == check_id:
                return check
        return None

    def _record_brain_decision_after_experience(
        self,
        state: GeoLocalizationState,
        decision: BrainDecision,
        *,
        revision_performed: bool = False,
    ) -> None:
        check_id = state.metadata.pop("experience_pending_effect_check_id", None)
        if not check_id:
            return
        check = self._experience_check(state, str(check_id))
        if check is None:
            return
        compact = self._compact_brain_decision(decision)
        check["brain_decision_after_guidance"] = compact
        check["brain_revision_performed"] = revision_performed
        if revision_performed:
            check["decision_changed"] = compact != check.get("proposed_brain_decision")
        state.metadata.pop("experience_guidance", None)
        state.metadata["recalled_memories"] = []

    def _mark_pending_experience_delivered(self, state: GeoLocalizationState) -> None:
        check_id = state.metadata.get("experience_pending_effect_check_id")
        if not check_id:
            return
        check = self._experience_check(state, str(check_id))
        if check is None:
            return
        check["delivered_to_brain"] = True
        selected_ids = {str(memory_id) for memory_id in check.get("selected_memory_ids") or []}
        for usage in state.metadata.get("memory_usage") or []:
            if isinstance(usage, dict) and str(usage.get("memory_id")) in selected_ids:
                usage["was_returned_to_brain"] = True

    def _record_executed_decision_for_current_check(
        self,
        state: GeoLocalizationState,
        decision: BrainDecision,
    ) -> None:
        checks = state.metadata.get("experience_checks") or []
        if not checks:
            return
        check = checks[-1]
        if not isinstance(check, dict) or check.get("brain_step") != state.step_count:
            return
        if check.get("proposed_brain_decision") is None:
            return
        check["executed_brain_decision"] = self._compact_brain_decision(decision)

    def _record_external_memory_episode(self, state: GeoLocalizationState) -> None:
        manager = self._memory_manager()
        if manager is not None:
            manager.record_episode_and_update(state)

    def _final_answer_from_decision(self, decision: BrainDecision) -> FinalAnswer:
        answer = decision.final_answer
        if answer is None:
            raise ValueError("A final_answer decision must include a final_answer object.")
        return answer

    def _requests_from_decision(self, decision: BrainDecision) -> list[ToolRequest]:
        return list(decision.tool_requests) if decision.action_type in {"call_tool", "call_tools"} else []

    def _prepare_tool_requests(
        self,
        state: GeoLocalizationState,
        requests: list[ToolRequest],
    ) -> tuple[list[ToolRequest], list[ToolResult]]:
        executable: list[ToolRequest] = []
        immediate_results: list[ToolResult] = []
        for request in requests:
            if not self._is_known_tool_request(request):
                immediate_results.append(self._reject_unknown_tool_request(state, request))
                continue
            tool = self.tools[request.tool_name]
            if tool.hidden:
                immediate_results.append(
                    self.skip_tool_request(
                        state,
                        request,
                        f"Tool {request.tool_name} is internal and cannot be requested by Brain.",
                    )
                )
                continue
            if not tool.is_available(state):
                immediate_results.append(
                    self.skip_tool_request(
                        state,
                        request,
                        f"Tool {request.tool_name} is not available in the current configuration or state.",
                    )
                )
                continue

            request = self._normalize_tool_request(state, request)
            if self._is_duplicate_tool_request(state, request):
                immediate_results.append(self.skip_tool_request(state, request, "Duplicate tool call with same arguments."))
                continue

            executable.append(request)
        return executable, immediate_results

    def _is_known_tool_request(self, request: ToolRequest) -> bool:
        return request.tool_name in self.tools

    def _is_internal_tool_name(self, tool_name: str) -> bool:
        tool = self.tools.get(tool_name)
        return bool(tool and tool.internal)

    def _normalize_tool_request(self, state: GeoLocalizationState, request: ToolRequest) -> ToolRequest:
        if request.tool_name in {"ocr", "visual_reanalysis"}:
            request.arguments["image_path"] = state.image_path
        if request.tool_name in {"map_tile_verify", "streetview_verify"}:
            request.arguments["image_path"] = state.image_path
            request.arguments["task_id"] = state.task_id
        return request

    def _is_duplicate_tool_request(self, state: GeoLocalizationState, request: ToolRequest) -> bool:
        signature = self._tool_signature(request)
        signatures = state.metadata.setdefault("react_tool_signatures", [])
        if signature in signatures:
            return True
        signatures.append(signature)
        return False

    def _tool_signature(self, request: ToolRequest) -> str:
        return f"{request.tool_name}:{json.dumps(request.arguments, ensure_ascii=False, sort_keys=True, default=str)}"

    def _reject_unknown_tool_request(self, state: GeoLocalizationState, request: ToolRequest) -> ToolResult:
        available_tools = sorted(
            name for name, tool in self.tools.items() if tool.is_available(state)
        )
        message = (
            f"Unknown tool '{request.tool_name}'. The Brain may only call tools listed in available_tools. "
            f"Available tools: {', '.join(available_tools)}."
        )
        call = ToolCall(
            tool_name=request.tool_name,
            arguments=request.arguments,
            step=state.step_count,
            status="error",
        )
        state.add_tool_call(call)
        result = ToolResult(
            tool_name=request.tool_name,
            success=False,
            data={
                "requested_tool": request.tool_name,
                "requested_arguments": request.arguments,
                "available_tools": available_tools,
                "message": message,
            },
            error=message,
        )
        state.add_tool_result(result)
        self._record_tool_resource(state, result, skipped=True)
        log_skip("TOOL", request.tool_name, message)
        return result

    def _apply_hypothesis_patch(self, state: GeoLocalizationState, patch: HypothesisPatch | None) -> None:
        if patch is None:
            return
        remove_ids = set(patch.remove)
        if remove_ids:
            state.hypotheses = [hyp for hyp in state.hypotheses if hyp.id not in remove_ids]
        for item in patch.add:
            state.hypotheses.append(item)
        by_id = {hyp.id: hyp for hyp in state.hypotheses}
        for update in patch.update:
            hyp = by_id.get(update.id)
            if hyp is None:
                continue
            data = update.model_dump(exclude_none=True)
            data.pop("id", None)
            metadata = data.pop("metadata", None)
            for key, value in data.items():
                setattr(hyp, key, value)
            if metadata:
                hyp.metadata.update(metadata)
        self._rank_hypotheses(state)
        state.touch()

    def _rank_hypotheses(self, state: GeoLocalizationState) -> None:
        state.hypotheses.sort(key=lambda item: item.score, reverse=True)

    def _update_uncertainty_from_decision(self, state: GeoLocalizationState, decision: BrainDecision) -> None:
        state.uncertainty.confidence = decision.confidence
        state.uncertainty.uncertainty_radius_m = decision.uncertainty_radius_m
        state.uncertainty.reasons = [decision.reasoning_summary] if decision.reasoning_summary else []
