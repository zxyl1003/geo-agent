"""Base workflow orchestration."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from geoagent.agents.base import BaseAgent
from geoagent.core.config import AppConfig, WorkflowConfig
from geoagent.core.context_builder import ContextBuilder
from geoagent.core.logging import (
    log_agent_output,
    log_agent_start,
    log_skip,
    log_tool_request,
    log_workflow,
)
from geoagent.core.schemas import (
    AgentOutput,
    FinalAnswer,
    ObservedEntity,
    OCRResult,
    ToolCall,
    ToolRequest,
    ToolResult,
    VisualCue,
)
from geoagent.state.task_state import GeoLocalizationState
from geoagent.tools.base import BaseTool
from geoagent.tools.registry import build_tools

class BaseWorkflow(ABC):
    name = "base_workflow"

    def __init__(
        self,
        app_config: AppConfig | None = None,
        tools: dict[str, BaseTool] | None = None,
        agents: dict[str, BaseAgent] | None = None,
        workflow_config: WorkflowConfig | None = None,
    ) -> None:
        self.app_config = app_config or AppConfig()
        self.tools = build_tools(self.app_config) if tools is None else tools
        self.agents = agents or {}
        self.workflow_config = workflow_config or self.app_config.workflows.get(self.name)

    @abstractmethod
    def run(
        self,
        input_image_path: str,
        user_query: str | None = None,
        task_metadata: dict[str, Any] | None = None,
    ) -> GeoLocalizationState:
        """Run the workflow."""

    @property
    def max_steps(self) -> int | None:
        return self.workflow_config.max_steps if self.workflow_config else None

    def run_agent(self, agent_name: str, state: GeoLocalizationState) -> AgentOutput:
        agent = self.agents[agent_name]
        log_agent_start(agent.name, state.step_count)
        output = agent.run(state)
        log_agent_output(
            agent.name,
            output.message,
            {
                "tool_requests": [request.model_dump() for request in output.tool_requests],
                "state_delta": output.state_delta,
            },
        )
        return output

    def execute_tool_request(self, state: GeoLocalizationState, request: ToolRequest) -> ToolResult:
        tool = self.tools.get(request.tool_name)
        log_tool_request(request.tool_name, request.arguments, request.reason)
        call = ToolCall(
            tool_name=request.tool_name,
            arguments=request.arguments,
            step=state.step_count,
            status="approved",
        )
        state.add_tool_call(call)

        if tool is None:
            call.status = "error"
            result = ToolResult(tool_name=request.tool_name, success=False, error="Tool is not enabled or registered.")
            state.add_tool_result(result)
            self._record_tool_resource(state, result, skipped=True)
            log_skip("TOOL", request.tool_name, result.error or "Tool is not enabled or registered.")
            return result

        prior_calls = [
            item
            for item in state.tool_calls
            if item.tool_name == request.tool_name and item.status in {"approved", "success", "error"}
        ]
        if len(prior_calls) > tool.max_calls_per_task:
            call.status = "skipped"
            result = ToolResult(
                tool_name=request.tool_name,
                success=False,
                error=f"Max calls exceeded for tool {request.tool_name}.",
            )
            state.add_tool_result(result)
            self._record_tool_resource(state, result, skipped=True)
            log_skip("TOOL", request.tool_name, result.error or "Max calls exceeded.")
            return result

        result = tool.safe_run(**request.arguments)
        call.status = "success" if result.success else "error"
        state.add_tool_result(result)
        self._record_tool_resource(state, result)
        self.merge_tool_result(state, result)
        return result

    def execute_tool_requests(self, state: GeoLocalizationState, requests: list[ToolRequest]) -> list[ToolResult]:
        if not requests:
            return []
        if len(requests) == 1:
            return [self.execute_tool_request(state, requests[0])]

        prepared: list[tuple[int, ToolRequest, BaseTool, ToolCall]] = []
        results_by_index: dict[int, ToolResult] = {}
        for index, request in enumerate(requests):
            tool = self.tools.get(request.tool_name)
            log_tool_request(request.tool_name, request.arguments, request.reason)
            call = ToolCall(
                tool_name=request.tool_name,
                arguments=request.arguments,
                step=state.step_count,
                status="approved",
            )
            state.add_tool_call(call)

            if tool is None:
                call.status = "error"
                result = ToolResult(tool_name=request.tool_name, success=False, error="Tool is not enabled or registered.")
                state.add_tool_result(result)
                self._record_tool_resource(state, result, skipped=True)
                log_skip("TOOL", request.tool_name, result.error or "Tool is not enabled or registered.")
                results_by_index[index] = result
                continue

            prior_calls = [
                item
                for item in state.tool_calls
                if item.tool_name == request.tool_name and item.status in {"approved", "success", "error"}
            ]
            if len(prior_calls) > tool.max_calls_per_task:
                call.status = "skipped"
                result = ToolResult(
                    tool_name=request.tool_name,
                    success=False,
                    error=f"Max calls exceeded for tool {request.tool_name}.",
                )
                state.add_tool_result(result)
                self._record_tool_resource(state, result, skipped=True)
                log_skip("TOOL", request.tool_name, result.error or "Max calls exceeded.")
                results_by_index[index] = result
                continue

            prepared.append((index, request, tool, call))

        if prepared:
            max_workers = min(4, len(prepared))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(tool.safe_run, **request.arguments): (index, call)
                    for index, request, tool, call in prepared
                }
                for future in as_completed(futures):
                    index, call = futures[future]
                    result = future.result()
                    call.status = "success" if result.success else "error"
                    results_by_index[index] = result

            for index, _request, _tool, _call in sorted(prepared, key=lambda item: item[0]):
                result = results_by_index[index]
                state.add_tool_result(result)
                self._record_tool_resource(state, result)
                self.merge_tool_result(state, result)

        return [results_by_index[index] for index in sorted(results_by_index)]

    def skip_tool_request(self, state: GeoLocalizationState, request: ToolRequest, reason: str) -> ToolResult:
        call = ToolCall(
            tool_name=request.tool_name,
            arguments=request.arguments,
            step=state.step_count,
            status="skipped",
        )
        state.add_tool_call(call)
        result = ToolResult(tool_name=request.tool_name, success=False, error=reason)
        state.add_tool_result(result)
        self._record_tool_resource(state, result, skipped=True)
        log_skip("TOOL", request.tool_name, reason)
        return result

    def _record_tool_resource(self, state: GeoLocalizationState, result: ToolResult, skipped: bool = False) -> None:
        data = result.data if isinstance(result.data, dict) else {}
        tool = self.tools.get(result.tool_name)
        if tool is not None and tool.internal:
            counts = state.metadata.setdefault("internal_tool_call_count", {"total": 0})
            counts["total"] = int(counts.get("total", 0)) + 1
            counts[result.tool_name] = int(counts.get(result.tool_name, 0)) + 1
            state.resource_events.append(
                {
                    "kind": "internal_tool",
                    "tool_name": result.tool_name,
                    "success": result.success,
                    "skipped": skipped,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                    "created_at": result.created_at,
                }
            )
            state.touch()
            return

        provider = (
            data.get("provider_display_name")
            or data.get("provider")
            or data.get("map_provider")
            or data.get("source_type")
        )
        state.add_tool_usage(
            tool_name=result.tool_name,
            success=result.success,
            latency_ms=result.latency_ms,
            provider=str(provider) if provider else None,
            skipped=skipped,
            error=result.error,
        )
        model_usage = data.get("model_usage")
        if not model_usage and isinstance(data.get("verification"), dict):
            model_usage = data["verification"].get("model_usage")
        if isinstance(model_usage, dict):
            state.add_model_usage(
                role=str(model_usage.get("role") or f"tool:{result.tool_name}"),
                model=str(model_usage.get("model") or ""),
                provider=str(model_usage.get("provider") or provider or ""),
                usage=model_usage.get("usage") if isinstance(model_usage.get("usage"), dict) else {},
            )
        if result.error:
            state.tool_failures.append(
                {
                    "tool_name": result.tool_name,
                    "error": result.error,
                    "skipped": skipped,
                    "step": state.step_count,
                }
            )

    def merge_tool_result(self, state: GeoLocalizationState, result: ToolResult) -> None:
        if not result.success:
            return

        if result.tool_name == "ocr":
            known = {(item.text, item.source) for item in state.ocr_results}
            for raw in result.data.get("ocr_results", []):
                item = OCRResult.model_validate(raw)
                if (item.text, item.source) not in known:
                    state.ocr_results.append(item)
                    known.add((item.text, item.source))

        if result.tool_name == "visual_reanalysis":
            known_cues = {(item.cue_type, item.text, item.source) for item in state.visual_cues}
            for raw in result.data.get("visual_cues", result.data.get("observations", [])):
                item = VisualCue.model_validate(raw)
                key = (item.cue_type, item.text, item.source)
                if key not in known_cues:
                    state.visual_cues.append(item)
                    known_cues.add(key)
            known_ocr = {(item.text, item.source) for item in state.ocr_results}
            for raw in result.data.get("ocr_results", []):
                item = OCRResult.model_validate(raw)
                key = (item.text, item.source)
                if key not in known_ocr:
                    state.ocr_results.append(item)
                    known_ocr.add(key)
            known_entities = {
                (
                    item.entity_type,
                    (item.name or "").strip().lower(),
                    "|".join(sorted(item.phones)) or "|".join(text.strip().lower() for text in item.text_items),
                )
                for item in state.observed_entities
            }
            for raw in result.data.get("observed_entities", []):
                item = ObservedEntity.model_validate(raw)
                key = (
                    item.entity_type,
                    (item.name or "").strip().lower(),
                    "|".join(sorted(item.phones)) or "|".join(text.strip().lower() for text in item.text_items),
                )
                if key not in known_entities:
                    state.observed_entities.append(item)
                    known_entities.add(key)

        state.touch()

    def build_final_answer(self, state: GeoLocalizationState) -> FinalAnswer:
        return FinalAnswer(
            location_name="Unknown",
            confidence=state.uncertainty.confidence,
            uncertainty_radius_m=state.uncertainty.uncertainty_radius_m,
            reasoning=["Maximum step budget exhausted before Brain submitted a final answer."],
        )

    def append_brain_tool_observation(self, state: GeoLocalizationState, result: ToolResult) -> None:
        summary = ContextBuilder(self.app_config, self.tools).summarize_tool_result(result)
        content = json.dumps(summary, ensure_ascii=False, indent=2, default=str)
        state.add_brain_message(
            "tool",
            content,
            name=result.tool_name,
            metadata={"tool_name": result.tool_name, "success": result.success},
        )

    def log_state_checkpoint(self, state: GeoLocalizationState, label: str) -> None:
        top = state.hypotheses[0] if state.hypotheses else None
        log_workflow(
            label,
            {
                "task_id": state.task_id,
                "step": state.step_count,
                "status": state.status,
                "top_hypothesis": top.model_dump() if top else None,
                "top_granularity": top.granularity if top else "unknown",
                "confidence": state.uncertainty.confidence,
                "uncertainty": state.uncertainty.model_dump(),
            },
        )
