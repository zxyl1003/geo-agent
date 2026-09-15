"""Deterministic validation policy for memory writes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from geoagent.memory.models import MemoryCandidate


@dataclass(frozen=True)
class CandidatePolicyDecision:
    accepted: bool
    reason: str


class MemoryWritePolicy:
    """Validate whether an agent-proposed strategy is suitable for storage."""

    _COORDINATE_PATTERN = re.compile(
        r"(?<!\d)[+-]?(?:[0-8]?\d(?:\.\d+)?|90(?:\.0+)?)\s*[,/]\s*"
        r"[+-]?(?:1[0-7]\d(?:\.\d+)?|[0-9]?\d(?:\.\d+)?|180(?:\.0+)?)(?!\d)"
    )
    def candidate_decision(
        self,
        candidate: MemoryCandidate,
        forbidden_place_names: set[str] | None = None,
    ) -> CandidatePolicyDecision:
        """Validate a candidate for writing.

        ``forbidden_place_names`` are specific (sub-country) place names derived
        from the ground-truth reverse-geocoding. They must never appear in a
        memory: the reflection context is diagnosis-only, and a memory that names
        the true city/district/street is a place-fact leak, not a reusable
        strategy.
        """
        searchable = candidate.searchable_text()

        if not candidate.situation.strip() or not candidate.lesson.strip():
            return CandidatePolicyDecision(False, "missing_situation_or_lesson")
        if not any(action.strip() for action in candidate.action_policy):
            return CandidatePolicyDecision(False, "missing_action_policy")
        if candidate.metadata.get("contains_location_fact") or self._COORDINATE_PATTERN.search(searchable):
            return CandidatePolicyDecision(False, "place_specific_fact_not_allowed")
        if forbidden_place_names and self._contains_forbidden_place_name(searchable, forbidden_place_names):
            return CandidatePolicyDecision(False, "ground_truth_context_leak")
        if candidate.feedback_type != "ground_truth":
            return CandidatePolicyDecision(False, "ground_truth_required_for_reusable_memory")
        return CandidatePolicyDecision(True, "candidate_passed_write_policy")

    def _contains_forbidden_place_name(self, searchable: str, forbidden_place_names: set[str]) -> bool:
        compact = re.sub(r"[\W_]+", "", searchable.casefold(), flags=re.UNICODE)
        for name in forbidden_place_names:
            compact_name = re.sub(r"[\W_]+", "", name.casefold(), flags=re.UNICODE)
            if len(compact_name) >= 2 and compact_name in compact:
                return True
        return False
