"""Unified state for geolocation tasks."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from geoagent.core.schemas import (
    ConversationMessage,
    FinalAnswer,
    Hypothesis,
    ObservedEntity,
    OCRResult,
    ToolCall,
    ToolResult,
    UncertaintyState,
    VisualCue,
    new_id,
    utc_now,
)


class GeoLocalizationState(BaseModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    image_path: str
    user_query: str | None = None
    visual_cues: list[VisualCue] = Field(default_factory=list)
    observed_entities: list[ObservedEntity] = Field(default_factory=list)
    ocr_results: list[OCRResult] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    uncertainty: UncertaintyState = Field(default_factory=UncertaintyState)
    final_answer: FinalAnswer | None = None
    status: Literal["initialized", "running", "completed", "failed"] = "initialized"
    step_count: int = 0
    brain_messages: list[ConversationMessage] = Field(default_factory=list)
    brain_conversation_summary: str | None = None
    token_usage: dict[str, Any] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "by_role": {},
        }
    )
    api_call_count: dict[str, int] = Field(default_factory=lambda: {"total": 0})
    tool_call_count: dict[str, int] = Field(default_factory=lambda: {"total": 0})
    resource_events: list[dict[str, Any]] = Field(default_factory=list)
    tool_failures: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def add_tool_result(self, result: ToolResult) -> None:
        self.tool_results.append(result)
        self.touch()

    def add_tool_call(self, call: ToolCall) -> None:
        self.tool_calls.append(call)
        self.touch()

    def add_brain_message(
        self,
        role: str,
        content: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.brain_messages.append(
            ConversationMessage.model_validate(
                {
                    "role": role,
                    "content": content,
                    "name": name,
                    "metadata": metadata or {},
                }
            )
        )
        self.touch()

    def add_model_usage(self, role: str, model: str | None, provider: str | None, usage: dict[str, Any] | None) -> None:
        usage = usage or {}
        prompt_tokens = self._int_usage(usage.get("prompt_tokens"))
        completion_tokens = self._int_usage(usage.get("completion_tokens"))
        total_tokens = self._int_usage(usage.get("total_tokens")) or prompt_tokens + completion_tokens

        self.token_usage["prompt_tokens"] = int(self.token_usage.get("prompt_tokens", 0)) + prompt_tokens
        self.token_usage["completion_tokens"] = int(self.token_usage.get("completion_tokens", 0)) + completion_tokens
        self.token_usage["total_tokens"] = int(self.token_usage.get("total_tokens", 0)) + total_tokens

        by_role = self.token_usage.setdefault("by_role", {})
        role_usage = by_role.setdefault(role, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        role_usage["prompt_tokens"] = int(role_usage.get("prompt_tokens", 0)) + prompt_tokens
        role_usage["completion_tokens"] = int(role_usage.get("completion_tokens", 0)) + completion_tokens
        role_usage["total_tokens"] = int(role_usage.get("total_tokens", 0)) + total_tokens

        self.api_call_count["total"] = int(self.api_call_count.get("total", 0)) + 1
        model_key = f"model:{role}"
        self.api_call_count[model_key] = int(self.api_call_count.get(model_key, 0)) + 1
        self.resource_events.append(
            {
                "kind": "model",
                "role": role,
                "model": model,
                "provider": provider,
                "usage": usage,
                "created_at": utc_now(),
            }
        )
        self.touch()

    def add_tool_usage(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float | None = None,
        provider: str | None = None,
        skipped: bool = False,
        error: str | None = None,
    ) -> None:
        self.tool_call_count["total"] = int(self.tool_call_count.get("total", 0)) + 1
        self.tool_call_count[tool_name] = int(self.tool_call_count.get(tool_name, 0)) + 1
        if not skipped:
            self.api_call_count["total"] = int(self.api_call_count.get("total", 0)) + 1
            api_key = f"tool:{tool_name}"
            self.api_call_count[api_key] = int(self.api_call_count.get(api_key, 0)) + 1
        self.resource_events.append(
            {
                "kind": "tool",
                "tool_name": tool_name,
                "provider": provider,
                "success": success,
                "skipped": skipped,
                "latency_ms": latency_ms,
                "error": error,
                "created_at": utc_now(),
            }
        )
        self.touch()

    def _int_usage(self, value: Any) -> int:
        if isinstance(value, bool) or value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
