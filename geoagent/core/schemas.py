"""Shared Pydantic schemas used across tools, agents, and workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


LocationGranularity = Literal["coordinates", "poi", "street", "city", "region", "country", "continent", "unknown"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class ImageInput(BaseModel):
    image_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class VisualCue(BaseModel):
    cue_type: str
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class OCRResult(BaseModel):
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    language: str | None = None
    bbox: list[float] | None = None
    source: str = "ocr"


class ObservedEntity(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ent"))
    entity_type: Literal["poi", "landmark", "phone", "address", "sign", "vehicle", "other"] = "poi"
    name: str | None = None
    text_items: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    region_hint: str | None = None
    source: str = "unknown"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: new_id("hyp"))
    name: str
    country: str | None = None
    region: str | None = None
    granularity: LocationGranularity = "unknown"
    lat: float | None = Field(default=None, ge=-90.0, le=90.0, allow_inf_nan=False, description="WGS84 latitude.")
    lon: float | None = Field(default=None, ge=-180.0, le=180.0, allow_inf_nan=False, description="WGS84 longitude.")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: new_id("call"))
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    step: int = 0
    status: Literal["pending", "approved", "skipped", "success", "error"] = "pending"
    created_at: datetime = Field(default_factory=utc_now)


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float | None = None
    raw: Any | None = None
    created_at: datetime = Field(default_factory=utc_now)


class UncertaintyState(BaseModel):
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_radius_m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    reasons: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    location_name: str
    country: str | None = None
    region: str | None = None
    city: str | None = None
    lat: float | None = Field(default=None, ge=-90.0, le=90.0, allow_inf_nan=False, description="WGS84 latitude.")
    lon: float | None = Field(default=None, ge=-180.0, le=180.0, allow_inf_nan=False, description="WGS84 longitude.")
    granularity: LocationGranularity = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_radius_m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    reasoning: list[str] = Field(default_factory=list)
    evidence_summary: list[str] = Field(default_factory=list)
    tool_trace: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "FinalAnswer":
        if (self.lat is None) != (self.lon is None):
            raise ValueError("Final answer coordinates must contain both lat and lon, or neither.")
        if self.lat is not None and self.granularity == "unknown":
            raise ValueError("A final answer with coordinates must declare a known granularity.")
        return self


class ToolRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: new_id("request"))
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class HypothesisUpdate(BaseModel):
    id: str
    name: str | None = None
    country: str | None = None
    region: str | None = None
    granularity: LocationGranularity | None = None
    lat: float | None = Field(default=None, ge=-90.0, le=90.0, allow_inf_nan=False, description="WGS84 latitude.")
    lon: float | None = Field(default=None, ge=-180.0, le=180.0, allow_inf_nan=False, description="WGS84 longitude.")
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None
    metadata: dict[str, Any] | None = None


class HypothesisPatch(BaseModel):
    add: list[Hypothesis] = Field(default_factory=list)
    update: list[HypothesisUpdate] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)


class BrainVisualAnalysis(BaseModel):
    scene_summary: str
    visual_cues: list[dict[str, Any]]
    ocr_results: list[str | dict[str, Any]]
    observed_entities: list[dict[str, Any]]
    search_queries: list[str]
    reasoning_notes: list[str]
    locatability: dict[str, Any]


class BrainVisualUpdate(BaseModel):
    """Only newly observed or corrected visual evidence after the initial pass."""

    visual_cues: list[dict[str, Any]] = Field(default_factory=list)
    ocr_results: list[str | dict[str, Any]] = Field(default_factory=list)
    observed_entities: list[dict[str, Any]] = Field(default_factory=list)


class BrainDecision(BaseModel):
    visual_analysis: BrainVisualAnalysis | None = None
    visual_updates: BrainVisualUpdate | None = None
    reasoning_summary: str = ""
    memory_references: list[str] = Field(default_factory=list)
    experience_request: str | None = None
    action_type: Literal["call_tool", "call_tools", "final_answer"]
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    hypothesis_patch: HypothesisPatch | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_radius_m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    final_answer: FinalAnswer | None = None

    @model_validator(mode="after")
    def validate_action_contract(self) -> "BrainDecision":
        request_count = len(self.tool_requests)
        if self.action_type == "call_tool" and request_count != 1:
            raise ValueError("call_tool requires exactly one tool request.")
        if self.action_type == "call_tools" and request_count < 2:
            raise ValueError("call_tools requires at least two tool requests.")
        if self.action_type == "final_answer":
            if self.final_answer is None:
                raise ValueError("final_answer action requires a final_answer object.")
            if request_count:
                raise ValueError("final_answer action cannot include tool requests.")
            if self.final_answer.confidence != self.confidence:
                raise ValueError("Decision and final-answer confidence must match.")
            if self.final_answer.uncertainty_radius_m != self.uncertainty_radius_m:
                raise ValueError("Decision and final-answer uncertainty radius must match.")
        elif self.final_answer is not None:
            raise ValueError("Tool actions cannot include a final_answer object.")
        return self


class AgentOutput(BaseModel):
    agent_name: str
    message: str = ""
    state_delta: dict[str, Any] = Field(default_factory=dict)
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
