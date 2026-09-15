"""Data contracts for the external geolocation experience-memory system."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from geoagent.core.schemas import LocationGranularity, new_id, utc_now


MemoryType = Literal[
    "evidence_reliability",
    "failure_pattern",
    "strategy_policy",
    "tool_policy",
    "conflict_resolution",
]
MemoryProposalSource = Literal["llm", "none"]
MemoryOutcomeLabel = Literal["full_success", "partial_success", "failure", "unverifiable"]


class ExperienceToolCheck(BaseModel):
    """One-shot request to check experience after a tool next finishes."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    scope: Literal["next_call", "pending_call"] = "next_call"
    request_id: str | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> "ExperienceToolCheck":
        if self.scope == "pending_call" and not self.request_id:
            raise ValueError("pending_call requires request_id.")
        if self.scope == "next_call" and self.request_id is not None:
            raise ValueError("next_call cannot specify request_id.")
        return self


class ExperienceNextCheck(BaseModel):
    """Experience-agent schedule relative to the current completed tool round."""

    model_config = ConfigDict(extra="forbid")

    after_rounds: int | None = Field(default=None, gt=0)
    after_tool: ExperienceToolCheck | None = None
    reason: str = ""


class ExperienceAdvice(BaseModel):
    """Validated online decision about whether and when to advise Brain."""

    model_config = ConfigDict(extra="forbid")

    intervene: bool = False
    requires_brain_revision: bool = False
    selected_memory_ids: list[str] = Field(default_factory=list, max_length=3)
    guidance: str = ""
    applicability_reason: str = ""
    next_check: ExperienceNextCheck = Field(default_factory=ExperienceNextCheck)

    @model_validator(mode="after")
    def validate_intervention(self) -> "ExperienceAdvice":
        if self.intervene:
            if not self.selected_memory_ids:
                raise ValueError("An intervention must select at least one candidate memory.")
            if not self.guidance.strip():
                raise ValueError("An intervention must provide non-empty guidance.")
        elif self.selected_memory_ids or self.guidance.strip():
            raise ValueError("A silent decision cannot select memories or provide guidance.")
        if self.requires_brain_revision and not self.intervene:
            raise ValueError("A silent decision cannot require Brain revision.")
        return self


class HierarchicalMemoryOutcome(BaseModel):
    """Auditable, granularity-aware result used by memory learning."""

    model_config = ConfigDict(extra="forbid")

    target_granularity: LocationGranularity | None = None
    predicted_granularity: LocationGranularity | None = None
    evaluated_granularity: LocationGranularity | None = None
    level_correctness: dict[str, bool | None] = Field(default_factory=dict)
    success: bool | None = None
    label: MemoryOutcomeLabel = "unverifiable"
    error_distance_m: float | None = None


class MemoryCandidate(BaseModel):
    """A newly proposed abstract memory before persistence/consolidation."""

    model_config = ConfigDict(extra="forbid")

    memory_type: MemoryType
    situation: str
    lesson: str
    action_policy: list[str] = Field(default_factory=list)
    applicable_conditions: list[str] = Field(default_factory=list)
    failure_conditions: list[str] = Field(default_factory=list)
    source_episode_id: str
    feedback_type: str = "none"
    confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    diagnosis_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def searchable_text(self) -> str:
        parts = [
            self.memory_type,
            self.situation,
            self.lesson,
            *self.action_policy,
            *self.applicable_conditions,
            *self.failure_conditions,
        ]
        return "\n".join(part for part in parts if str(part).strip())


class FailureAttribution(BaseModel):
    """Structured causal attribution of an episode outcome.

    Recorded on every reviewed episode (success and failure) even when no
    reusable memory is written, so the paper can measure the distribution of
    failure modes over time. ``failure_type`` / ``success_pattern`` come from
    the controlled vocabularies in the memory manager prompt.
    """

    model_config = ConfigDict(extra="forbid")

    failure_type: str = ""
    success_pattern: str = ""
    rationale: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MemoryAgentReview(BaseModel):
    """Validated output of the memory-management agent."""

    model_config = ConfigDict(extra="forbid")

    should_write: bool = False
    diagnosis: str = ""
    diagnosis_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    successful_levels: list[LocationGranularity] = Field(default_factory=list)
    failed_levels: list[LocationGranularity] = Field(default_factory=list)
    candidate: MemoryCandidate | None = None
    merge_memory_id: str | None = None
    rationale: str = ""
    proposal_source: MemoryProposalSource = "none"
    failure_attribution: FailureAttribution | None = None


class MemoryItem(BaseModel):
    """A persisted abstract experience memory."""

    memory_id: str = Field(default_factory=lambda: new_id("mem"))
    memory_type: MemoryType
    situation: str
    lesson: str
    action_policy: list[str] = Field(default_factory=list)
    applicable_conditions: list[str] = Field(default_factory=list)
    failure_conditions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    merge_count: int = 0
    feedback_type: str = "none"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def searchable_text(self) -> str:
        parts = [
            self.memory_type,
            self.situation,
            self.lesson,
            *self.action_policy,
            *self.applicable_conditions,
            *self.failure_conditions,
        ]
        return "\n".join(part for part in parts if str(part).strip())

    def prompt_digest(self) -> dict[str, Any]:
        """Compact representation safe to inject into Brain context."""

        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "situation": self.situation,
            "lesson": self.lesson,
            "action_policy": self.action_policy,
            "applicable_conditions": self.applicable_conditions,
            "failure_conditions": self.failure_conditions,
            "confidence": round(self.confidence, 3),
            "merge_count": self.merge_count,
            "note": "Experience memory is a strategy hint or warning, not direct location evidence.",
        }

class RetrievedMemory(BaseModel):
    """A memory item returned by similarity/metadata retrieval."""

    item: MemoryItem
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    rank: int = 0
    retrieval_source: str = "sqlite"

    def prompt_digest(self) -> dict[str, Any]:
        digest = self.item.prompt_digest()
        digest["retrieval_score"] = round(self.score, 3)
        digest["retrieved_rank"] = self.rank
        digest["retrieval_source"] = self.retrieval_source
        return digest
