"""External experience-memory manager for geolocation workflows."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geoagent.agents.memory_manager_agent import MemoryManagerAgent
from geoagent.core.config import AppConfig
from geoagent.core.context_builder import ContextBuilder
from geoagent.core.logging import log_workflow
from geoagent.core.schemas import new_id, utc_now
from geoagent.eval import DISTANCE_THRESHOLDS
from geoagent.memory.chroma_index import ChromaMemoryIndex
from geoagent.memory.models import (
    HierarchicalMemoryOutcome,
    MemoryAgentReview,
    MemoryCandidate,
    MemoryItem,
    RetrievedMemory,
)
from geoagent.memory.policies import MemoryWritePolicy
from geoagent.memory.sqlite_store import SQLiteMemoryStore
from geoagent.state.task_state import GeoLocalizationState

@dataclass(frozen=True)
class MemorySettings:
    storage_dir: Path
    sqlite_path: Path
    chroma_path: Path
    enabled: bool = False
    mode: str = "full"
    use_chroma: bool = True
    retrieve_top_k: int = 3
    merge_similarity_threshold: float = 0.9
    # Reflection: enrich episode review with ground-truth administrative context
    # recovered from the ground-truth coordinates at learning time.
    enrich_ground_truth_context: bool = True

    @property
    def retrieve_enabled(self) -> bool:
        return self.enabled and self.mode in {"retrieve_only", "full"}

    @property
    def write_enabled(self) -> bool:
        return self.enabled and self.mode in {"learn_only", "full"}

class MemoryManager:
    """Retrieve and update external geolocation experience memories.

    Completed verified episodes may produce one reusable strategy memory. Similar
    memories are merged; otherwise the candidate is stored directly. Chroma
    provides retrieval while SQLite remains the authoritative record store.
    """

    _MERGE_DIMENSIONS = ("failure_type", "cue_category", "tool_scenario")
    _CUE_CATEGORIES = {
        "ocr_text",
        "traffic_system",
        "vehicle_plate",
        "infrastructure",
        "architecture_urban_form",
        "vegetation_terrain_climate",
        "landmark_poi",
        "capture_metadata",
        "multi_cue",
        "none",
    }
    _TOOL_SCENARIOS = {
        "none",
        "ocr",
        "visual_reanalysis",
        "web_search",
        "webpage_read",
        "poi_search",
        "geocode",
        "reverse_geocode",
        "map_tile_verify",
        "streetview_verify",
        "multi_tool",
    }

    def __init__(
        self,
        settings: MemorySettings,
        app_config: AppConfig | None = None,
        agent: MemoryManagerAgent | None = None,
    ) -> None:
        self.settings = settings
        self.app_config = app_config or AppConfig()
        if not settings.use_chroma:
            raise ValueError("Chroma must be enabled when external memory is enabled.")
        self.store = SQLiteMemoryStore(settings.sqlite_path)
        self.index = ChromaMemoryIndex(settings.chroma_path)
        stored_memories = self.store.list_memories(limit=1_000_000)
        if self.index.count() != len(stored_memories):
            self.index.rebuild(stored_memories)
        self.policy = MemoryWritePolicy()
        self.agent = agent or MemoryManagerAgent(app_config=self.app_config)

    @classmethod
    def from_config(
        cls,
        app_config: AppConfig,
        agent: MemoryManagerAgent | None = None,
    ) -> "MemoryManager | None":
        raw = app_config.system.get("memory", {}) or {}
        if not bool(raw.get("enabled", False)):
            return None

        config_root = Path(app_config.config_dir).resolve().parent
        configured_storage_dir = raw.get("storage_dir")
        if not configured_storage_dir:
            raise ValueError("memory.storage_dir is required when external memory is enabled.")
        storage_dir = Path(str(configured_storage_dir))
        if not storage_dir.is_absolute():
            storage_dir = config_root / storage_dir
        sqlite_path = Path(raw.get("sqlite_path", storage_dir / "memory.sqlite"))
        chroma_path = Path(raw.get("chroma_path", storage_dir / "chroma"))
        if not sqlite_path.is_absolute():
            sqlite_path = config_root / sqlite_path
        if not chroma_path.is_absolute():
            chroma_path = config_root / chroma_path

        mode = str(raw.get("mode", "full")).strip().lower() or "full"
        if mode not in {"off", "retrieve_only", "learn_only", "full"}:
            raise ValueError(f"Unsupported memory mode: {mode}")
        settings = MemorySettings(
            enabled=True,
            mode=mode,
            storage_dir=storage_dir,
            sqlite_path=sqlite_path,
            chroma_path=chroma_path,
            use_chroma=bool((raw.get("chroma") or {}).get("enabled", True)),
            retrieve_top_k=int(raw.get("retrieve_top_k", 3)),
            merge_similarity_threshold=float(raw.get("merge_similarity_threshold", 0.9)),
            enrich_ground_truth_context=bool(
                (raw.get("reflection") or {}).get("enrich_ground_truth_context", True)
            ),
        )
        return cls(settings, app_config=app_config, agent=agent)

    def has_retrievable_memories(self) -> bool:
        """Return whether retrieval is enabled and the store contains a memory."""

        return bool(self.settings.retrieve_enabled and self.store.list_memories(limit=1))

    def retrieve(
        self,
        query_text: str,
        top_k: int = 3,
        memory_types: set[str] | None = None,
    ) -> list[RetrievedMemory]:
        if not self.settings.retrieve_enabled or top_k <= 0:
            return []
        requested_types = {str(item).strip() for item in (memory_types or set()) if str(item).strip()}
        search_k = max(top_k * (4 if requested_types else 2), top_k)
        hits = self._search_memories(query_text, top_k=search_k)
        selected = [
            hit for hit in hits
            if not requested_types or hit.item.memory_type in requested_types
        ][:top_k]
        for rank, hit in enumerate(selected, start=1):
            hit.rank = rank
        return selected

    def _search_memories(
        self,
        query_text: str,
        top_k: int,
    ) -> list[RetrievedMemory]:
        hits: list[RetrievedMemory] = []
        for hit in self.index.query(query_text, top_k=max(top_k * 4, 20)):
            memory_id = str(hit.get("memory_id") or "")
            item = self.store.get_memory(memory_id)
            if item is None:
                raise RuntimeError(
                    f"Chroma returned memory {memory_id!r}, but it is missing from SQLite."
                )
            hits.append(
                RetrievedMemory(
                    item=item,
                    score=float(hit.get("score") or 0.0),
                    rank=len(hits) + 1,
                    retrieval_source="chroma",
                )
            )

        hits.sort(key=lambda hit: (hit.score, hit.item.memory_id), reverse=True)
        for index, hit in enumerate(hits[:top_k], start=1):
            hit.rank = index
        return hits[:top_k]
    def record_episode_and_update(self, state: GeoLocalizationState) -> None:
        if not (self.settings.retrieve_enabled or self.settings.write_enabled):
            return

        episode = self._episode_from_state(state)
        self.store.insert_episode(episode)
        state.metadata["memory_episode_id"] = episode["episode_id"]
        usages = state.metadata.get("memory_usage") or []
        self.store.record_usage(episode["episode_id"], usages)

        if not self.settings.write_enabled:
            state.metadata["memory_update"] = {
                "episode_id": episode["episode_id"],
                "status": "retrieval_audit_recorded",
                "retrieved_count": len(usages),
            }
            state.touch()
            return

        review = self._review_episode(state, episode)
        attribution = (
            review.failure_attribution.model_dump(mode="json")
            if review.failure_attribution is not None
            else {}
        )
        attribution["successful_levels"] = list(review.successful_levels)
        attribution["failed_levels"] = list(review.failed_levels)
        # The attribution is audit data: persist it on the episode even when no
        # reusable memory is written.
        self.store.update_episode_attribution(episode["episode_id"], attribution)
        state.metadata["memory_attribution"] = attribution

        forbidden_place_names = self._forbidden_place_names(episode.get("ground_truth_context"))
        candidate = review.candidate if review.should_write else None
        if candidate is None:
            state.metadata["memory_update"] = {
                "episode_id": episode["episode_id"],
                "status": "episode_recorded",
                "reason": review.rationale or "No abstract memory candidate met write-policy conditions.",
            }
            state.touch()
            return

        candidate_decision = self.policy.candidate_decision(candidate, forbidden_place_names)
        if not candidate_decision.accepted:
            state.metadata["memory_update"] = {
                "episode_id": episode["episode_id"],
                "status": "candidate_rejected",
                "reason": candidate_decision.reason,
            }
            state.touch()
            return

        if review.merge_memory_id:
            offered_memories = {
                str(memory.get("memory_id") or ""): memory
                for memory in state.metadata.get("memory_review_context", {}).get(
                    "existing_memory_candidates", []
                )
            }
            offered_memory = offered_memories.get(review.merge_memory_id)
            merge_target = self.store.get_memory(review.merge_memory_id)
            if (
                offered_memory is None
                or merge_target is None
                or merge_target.memory_type != candidate.memory_type
            ):
                state.metadata["memory_update"] = {
                    "episode_id": episode["episode_id"],
                    "status": "candidate_rejected",
                    "reason": "invalid_merge_memory_id",
                }
                state.touch()
                return
            merge_issues = self._merge_target_issues(
                merge_target,
                candidate,
                self._candidate_similarity_to_memory(candidate, merge_target.memory_id),
            )
            if merge_issues:
                state.metadata["memory_update"] = {
                    "episode_id": episode["episode_id"],
                    "status": "candidate_rejected",
                    "reason": "; ".join(merge_issues),
                }
                state.touch()
                return
            item, update_status = self._merge_memory(merge_target, candidate)
        else:
            item = self.store.insert_memory(candidate)
            self.index.upsert(item)
            update_status = "new_memory_inserted"
        state.metadata["memory_update"] = {
            "episode_id": episode["episode_id"],
            "status": update_status,
            "memory_id": item.memory_id,
            "memory_type": item.memory_type,
            "confidence": item.confidence,
            "merge_count": item.merge_count,
            "proposal_source": review.proposal_source,
        }
        state.touch()

    def _review_episode(
        self,
        state: GeoLocalizationState,
        episode: dict[str, Any],
    ) -> MemoryAgentReview:
        if episode.get("success") is None and str(episode.get("feedback_type") or "none") == "none":
            return MemoryAgentReview(
                should_write=False,
                rationale="Unverified episode retained without reusable-memory abstraction.",
                proposal_source="none",
            )
        review_context = {
            "episode_id": episode["episode_id"],
            "feedback_type": episode.get("feedback_type") or "none",
            "success": episode.get("success"),
            "error_distance_m": episode.get("error_distance_m"),
            "coordinate_distance_thresholds_m": list(DISTANCE_THRESHOLDS),
            "precomputed_facts": self._precomputed_reflection_facts(episode),
            "ground_truth_context": episode.get("ground_truth_context"),
            "outcome": episode.get("outcome") or {},
            "scene_type": episode.get("scene_type"),
            "visual_conditions": episode.get("visual_conditions") or {},
            "visual_clues": (episode.get("visual_clues") or [])[:12],
            "observed_entities": (episode.get("observed_entities") or [])[:8],
            "ocr_results": (episode.get("ocr_results") or [])[:8],
            "hypotheses": (episode.get("hypotheses") or [])[:5],
            "final_answer": episode.get("final_answer"),
            "tool_calls": (episode.get("tool_calls") or [])[-12:],
            "tool_results": (episode.get("tool_results") or [])[-12:],
            "reasoning_trace": (episode.get("reasoning_trace") or [])[-12:],
            "retrieved_memory_usage": state.metadata.get("memory_usage") or [],
        }
        existing_query = "\n".join(
            [
                str(review_context.get("scene_type") or ""),
                str(review_context.get("outcome") or ""),
                str(review_context.get("visual_clues") or ""),
                str(review_context.get("ocr_results") or ""),
                str(review_context.get("tool_calls") or ""),
                str(review_context.get("reasoning_trace") or ""),
            ]
        )
        existing_memory_candidates = []
        for hit in self._search_memories(existing_query, top_k=8):
            digest = hit.prompt_digest()
            digest["merge_dimensions"] = {
                key: hit.item.metadata[key]
                for key in self._MERGE_DIMENSIONS
                if key in hit.item.metadata
            }
            existing_memory_candidates.append(digest)
        review_context["existing_memory_candidates"] = existing_memory_candidates
        review_context["merge_requirements"] = {
            "minimum_similarity": self.settings.merge_similarity_threshold,
            "exact_match_dimensions": ["failure_type", "cue_category", "tool_scenario"],
            "candidate_must_be_complete_resynthesis": True,
        }
        state.metadata["memory_review_context"] = review_context
        output = self.agent.run(state)
        review = output.metadata.get("review")
        if not isinstance(review, MemoryAgentReview):
            review = MemoryAgentReview.model_validate(output.state_delta.get("memory_agent_review") or {})

        self._set_proposed_merge_similarity(review, review_context)
        initial_issues = self._review_consistency_issues(review, review_context)
        final_issues = initial_issues
        if initial_issues:
            review_context["consistency_issues"] = initial_issues
            review_context["consistency_retry_instruction"] = (
                "Correct every listed inconsistency and return the complete JSON review again."
            )
            state.metadata["memory_review_context"] = review_context
            output = self.agent.run(state)
            review = output.metadata.get("review")
            if not isinstance(review, MemoryAgentReview):
                review = MemoryAgentReview.model_validate(
                    output.state_delta.get("memory_agent_review") or {}
                )
            self._set_proposed_merge_similarity(review, review_context)
            final_issues = self._review_consistency_issues(review, review_context)

        state.metadata["memory_review_consistency"] = {
            "initial_issues": initial_issues,
            "retried": bool(initial_issues),
            "final_issues": final_issues,
            "accepted": not final_issues,
        }
        if final_issues:
            reason = "Memory review remained inconsistent after one retry: " + "; ".join(final_issues)
            review = review.model_copy(
                update={
                    "should_write": False,
                    "candidate": None,
                    "merge_memory_id": None,
                    "rationale": reason,
                }
            )
            state.metadata["memory_agent_review"] = review.model_dump(mode="json")
        state.touch()
        return review

    def _precomputed_reflection_facts(self, episode: dict[str, Any]) -> dict[str, Any]:
        outcome = episode.get("outcome") or {}
        evaluated_granularity = self._normalize_granularity(outcome.get("evaluated_granularity"))
        coordinate_assessment_expected = evaluated_granularity in {"street", "poi", "coordinates"}
        error_distance_m = self._float_or_none(episode.get("error_distance_m"))
        final_answer = episode.get("final_answer") or {}
        uncertainty_radius_m = self._float_or_none(final_answer.get("uncertainty_radius_m"))
        within_uncertainty_radius = None
        if error_distance_m is not None and uncertainty_radius_m is not None:
            within_uncertainty_radius = error_distance_m <= uncertainty_radius_m
        return {
            "evaluated_granularity": evaluated_granularity,
            "coordinate_assessment_expected": coordinate_assessment_expected,
            "known_level_correctness": outcome.get("level_correctness") or {},
            "error_distance_m": error_distance_m,
            "uncertainty_radius_m": uncertainty_radius_m,
            "within_uncertainty_radius": within_uncertainty_radius,
            "distance_threshold_checks": [
                {
                    "threshold_m": threshold,
                    "within_threshold": None
                    if error_distance_m is None
                    else error_distance_m <= threshold,
                }
                for threshold in DISTANCE_THRESHOLDS
            ],
        }

    @classmethod
    def _review_consistency_issues(
        cls,
        review: MemoryAgentReview,
        review_context: dict[str, Any],
    ) -> list[str]:
        issues: list[str] = []
        successful = list(review.successful_levels)
        failed = list(review.failed_levels)
        successful_set = set(successful)
        failed_set = set(failed)
        allowed_levels = {"continent", "country", "region", "city", "street", "poi", "coordinates"}

        if len(successful) != len(successful_set):
            issues.append("successful_levels contains duplicates")
        if len(failed) != len(failed_set):
            issues.append("failed_levels contains duplicates")
        overlap = sorted(successful_set & failed_set)
        if overlap:
            issues.append(f"levels appear in both successful_levels and failed_levels: {overlap}")
        unsupported = sorted((successful_set | failed_set) - allowed_levels)
        if unsupported:
            issues.append(f"unsupported assessed levels: {unsupported}")

        facts = review_context.get("precomputed_facts") or {}
        for level, is_correct in (facts.get("known_level_correctness") or {}).items():
            if is_correct is True and level in failed_set:
                issues.append(f"{level} is precomputed correct but listed as failed")
            elif is_correct is False and level in successful_set:
                issues.append(f"{level} is precomputed incorrect but listed as successful")

        coordinates_assessed = "coordinates" in successful_set or "coordinates" in failed_set
        coordinate_assessment_expected = bool(facts.get("coordinate_assessment_expected"))
        within_uncertainty_radius = facts.get("within_uncertainty_radius")
        if not coordinate_assessment_expected and coordinates_assessed:
            issues.append("coordinates is assessed although the evaluated granularity is coarser")
        elif coordinate_assessment_expected and within_uncertainty_radius is not None:
            if not coordinates_assessed:
                issues.append("coordinates is missing for a checkable fine-grained claim")
            elif within_uncertainty_radius is True and "coordinates" in failed_set:
                issues.append("coordinates is failed despite being within the declared uncertainty radius")
            elif within_uncertainty_radius is False and "coordinates" in successful_set:
                issues.append("coordinates is successful despite exceeding the declared uncertainty radius")

        attribution = review.failure_attribution
        if attribution is not None and attribution.failure_type and attribution.success_pattern:
            issues.append("failure_type and success_pattern cannot both be populated")
        if review.should_write and review.candidate is None:
            issues.append("should_write is true but candidate is null")
        if not review.should_write and review.candidate is not None:
            issues.append("should_write is false but candidate is populated")
        if review.candidate is not None:
            metadata = review.candidate.metadata
            for key in cls._MERGE_DIMENSIONS:
                if key not in metadata:
                    issues.append(f"candidate merge dimension {key} is missing")
            cue_category = str(metadata.get("cue_category") or "").strip().casefold()
            tool_scenario = str(metadata.get("tool_scenario") or "").strip().casefold()
            if cue_category not in cls._CUE_CATEGORIES:
                issues.append(f"unsupported cue_category: {cue_category or '<empty>'}")
            if tool_scenario not in cls._TOOL_SCENARIOS:
                issues.append(f"unsupported tool_scenario: {tool_scenario or '<empty>'}")
            if review.failure_attribution is not None:
                attributed_failure_type = review.failure_attribution.failure_type.strip().casefold()
                candidate_failure_type = str(metadata.get("failure_type") or "").strip().casefold()
                if candidate_failure_type != attributed_failure_type:
                    issues.append(
                        "candidate failure_type does not match failure_attribution.failure_type"
                    )
        if review.merge_memory_id and (not review.should_write or review.candidate is None):
            issues.append("merge_memory_id requires a retained candidate")
        elif review.merge_memory_id and review.candidate is not None:
            offered = {
                str(memory.get("memory_id") or ""): memory
                for memory in review_context.get("existing_memory_candidates", [])
            }.get(review.merge_memory_id)
            if offered is None:
                issues.append("merge_memory_id is not an offered memory candidate")
            else:
                threshold = float(
                    (review_context.get("merge_requirements") or {}).get(
                        "minimum_similarity", 0.9
                    )
                )
                proposed_merge = review_context.get("proposed_merge") or {}
                similarity = (
                    float(proposed_merge.get("similarity") or 0.0)
                    if proposed_merge.get("memory_id") == review.merge_memory_id
                    else 0.0
                )
                if similarity < threshold:
                    issues.append(
                        f"merge similarity {similarity:.3f} is below threshold {threshold:.3f}"
                    )
                if offered.get("memory_type") != review.candidate.memory_type:
                    issues.append("merge memory_type does not match the candidate")
                issues.extend(
                    cls._merge_dimension_issues(
                        offered.get("merge_dimensions") or {},
                        review.candidate.metadata,
                    )
                )
        return issues

    def _set_proposed_merge_similarity(
        self,
        review: MemoryAgentReview,
        review_context: dict[str, Any],
    ) -> None:
        review_context.pop("proposed_merge", None)
        if review.merge_memory_id and review.candidate is not None:
            review_context["proposed_merge"] = {
                "memory_id": review.merge_memory_id,
                "similarity": self._candidate_similarity_to_memory(
                    review.candidate,
                    review.merge_memory_id,
                ),
            }

    def _candidate_similarity_to_memory(
        self,
        candidate: MemoryCandidate,
        memory_id: str,
    ) -> float:
        for hit in self._search_memories(candidate.searchable_text(), top_k=20):
            if hit.item.memory_id == memory_id:
                return hit.score
        return 0.0

    @staticmethod
    def _merge_dimension_issues(
        existing_metadata: dict[str, Any],
        candidate_metadata: dict[str, Any],
    ) -> list[str]:
        issues = []
        for key in MemoryManager._MERGE_DIMENSIONS:
            if key not in existing_metadata or key not in candidate_metadata:
                issues.append(f"merge dimension {key} is missing")
                continue
            existing_value = str(existing_metadata[key] or "").strip().casefold()
            candidate_value = str(candidate_metadata[key] or "").strip().casefold()
            if key != "failure_type" and (not existing_value or not candidate_value):
                issues.append(f"merge dimension {key} is empty")
            elif existing_value != candidate_value:
                issues.append(f"merge dimension {key} does not match")
        return issues

    def _merge_target_issues(
        self,
        item: MemoryItem,
        candidate: MemoryCandidate,
        similarity: float,
    ) -> list[str]:
        issues = []
        if similarity < self.settings.merge_similarity_threshold:
            issues.append(
                "merge similarity "
                f"{similarity:.3f} is below threshold "
                f"{self.settings.merge_similarity_threshold:.3f}"
            )
        issues.extend(self._merge_dimension_issues(item.metadata, candidate.metadata))
        return issues

    def _merge_memory(
        self,
        item: MemoryItem,
        candidate: MemoryCandidate,
    ) -> tuple[MemoryItem, str]:
        item.situation = candidate.situation
        item.lesson = candidate.lesson
        item.action_policy = list(candidate.action_policy)
        item.applicable_conditions = list(candidate.applicable_conditions)
        item.failure_conditions = list(candidate.failure_conditions)
        item.confidence = max(item.confidence, candidate.confidence)
        item.merge_count += 1
        item.feedback_type = candidate.feedback_type
        item.metadata = dict(candidate.metadata)
        self.store.update_memory(item)
        self.store.link_memory_episode(
            item.memory_id,
            candidate.source_episode_id,
            "merge",
            diagnosis_confidence=candidate.diagnosis_confidence,
            note="similar strategy memory merged",
        )
        self.index.upsert(item)
        return item, "similar_memory_merged"

    def rebuild_index(self) -> None:
        self.index.rebuild(self.store.list_memories(limit=1_000_000))
    def _episode_from_state(self, state: GeoLocalizationState) -> dict[str, Any]:
        episode_id = str(state.metadata.get("memory_episode_id") or new_id("episode"))
        ground_truth = state.metadata.get("ground_truth") or {}
        feedback_type = "ground_truth" if ground_truth else "none"
        outcome = self._evaluate_hierarchical_outcome(state, ground_truth)
        success = outcome.success
        error_distance_m = outcome.error_distance_m
        state.metadata["memory_outcome"] = outcome.model_dump(mode="json")
        context_builder = ContextBuilder(self.app_config)
        ground_truth_context = (
            self._ground_truth_context(ground_truth)
            if self.settings.write_enabled
            else None
        )
        state.metadata["ground_truth_context"] = ground_truth_context or {}
        return {
            "episode_id": episode_id,
            "task_id": state.task_id,
            "image_path": state.image_path,
            "user_query": state.user_query,
            "scene_type": state.metadata.get("scene_type"),
            "visual_conditions": {},
            "visual_clues": [cue.model_dump() for cue in state.visual_cues],
            "observed_entities": [entity.model_dump() for entity in state.observed_entities],
            "ocr_results": [ocr.model_dump() for ocr in state.ocr_results],
            "hypotheses": [hypothesis.model_dump() for hypothesis in state.hypotheses],
            "final_answer": state.final_answer.model_dump() if state.final_answer else None,
            "tool_calls": [call.model_dump() for call in state.tool_calls],
            "tool_results": [context_builder.summarize_tool_result(result) for result in state.tool_results],
            "feedback_type": feedback_type,
            "ground_truth": ground_truth or None,
            "ground_truth_context": ground_truth_context or None,
            "success": success,
            "error_distance_m": error_distance_m,
            "outcome": outcome.model_dump(mode="json"),
            "reasoning_trace": [message.model_dump() for message in state.brain_messages[-12:]],
            "created_at": utc_now(),
        }

    def _ground_truth_context(self, ground_truth: dict[str, Any]) -> dict[str, Any] | None:
        """Reverse-geocode the ground-truth coordinates at learning time.

        This recovers the *true* administrative context (country/region/city/
        road) that the reviewer needs to attribute failure modes such as
        "branch POI name used as an anchor" when the ground truth carries only
        coordinates. The context is diagnosis-only input to the reflection LLM;
        the deterministic policy blocks it from leaking into memory content.
        """

        if not self.settings.enrich_ground_truth_context:
            return None
        lat = self._float_or_none(self._first_present(ground_truth, "lat", "latitude"))
        lon = self._float_or_none(self._first_present(ground_truth, "lon", "lng", "longitude"))
        if lat is None or lon is None:
            return None
        country = str(self._first_present(ground_truth, "country", "nation") or "").strip() or None

        cache_key = (round(lat, 4), round(lon, 4))
        if not hasattr(self, "_gt_context_cache"):
            self._gt_context_cache: dict[tuple[float, float], dict[str, Any]] = {}
        if cache_key in self._gt_context_cache:
            return self._gt_context_cache[cache_key]

        tool = self._reverse_geocode_tool()
        if tool is None:
            return None
        result = None
        for attempt_country in (country, None):
            try:
                result = tool.safe_run(lat=lat, lon=lon, country=attempt_country)
            except Exception as exc:  # noqa: BLE001 - enrichment must never break recording
                log_workflow("Ground-truth context enrichment failed", {"error": str(exc)})
                result = None
                continue
            if result is not None and result.success and isinstance(result.data, dict):
                break
            # The country hint can be wrong in the metadata (e.g. country=China
            # but coordinates in the US), routing to a provider that cannot
            # resolve the location. Retry without the hint to use the
            # international route.
            if attempt_country is not None:
                log_workflow(
                    "Ground-truth context enrichment retried without country hint",
                    {"country": attempt_country, "lat": lat, "lon": lon},
                )
        if result is None or not result.success or not isinstance(result.data, dict):
            return None

        top_result = result.data.get("top_result") or {}
        components = top_result.get("address_components") or result.data.get("address_components") or {}
        context = {
            "provider": result.data.get("provider") or result.data.get("provider_display_name"),
            "formatted_address": result.data.get("formatted_address") or top_result.get("formatted_address"),
            "country": self._first_present(components, "country"),
            "region": self._first_present(components, "region", "province", "state"),
            "city": self._first_present(components, "city"),
            "district": self._first_present(components, "district", "county"),
            "town": self._first_present(components, "town", "suburb"),
            "street": self._first_present(components, "street", "road"),
            "street_number": self._first_present(components, "street_number", "house_number", "door"),
        }
        context = {key: value for key, value in context.items() if value is not None and value != ""}
        if not context:
            return None
        self._gt_context_cache[cache_key] = context
        return context

    def _forbidden_place_names(self, ground_truth_context: dict[str, Any] | None) -> set[str]:
        """Specific (sub-country) place names that must never appear in memory
        content. Country and the full formatted address are excluded: a country
        is too broad to leak a specific location fact, and the formatted address
        string contains the country name which may legitimately appear in a
        strategy memory."""
        if not ground_truth_context:
            return set()
        forbidden = set()
        for key in ("region", "city", "district", "town", "street", "street_number"):
            value = str(ground_truth_context.get(key) or "").strip()
            if len(value) >= 2:
                forbidden.add(value.casefold())
        return forbidden

    def _reverse_geocode_tool(self):
        """Lazily instantiate the reverse_geocode facade for learning-time use."""
        from geoagent.core.registry import tool_registry

        if not hasattr(self, "_reverse_geocode_tool_instance"):
            candidate = None
            tool_config = self.app_config.tools.get("reverse_geocode")
            if tool_config is not None and tool_config.enable:
                cls = tool_registry.maybe_get("reverse_geocode")
                if cls is None:
                    cls = tool_registry.register_from_class_path("reverse_geocode", tool_config.class_path)
                candidate = cls(config=tool_config, app_config=self.app_config)
            if candidate is not None and not candidate.is_available():
                candidate = None
            self._reverse_geocode_tool_instance = candidate
        return self._reverse_geocode_tool_instance

    def _evaluate_hierarchical_outcome(
        self,
        state: GeoLocalizationState,
        ground_truth: dict[str, Any],
    ) -> HierarchicalMemoryOutcome:
        """Evaluate geographic levels independently for memory learning.

        ``target_granularity`` is the task contract when explicitly supplied.
        Otherwise the model's declared prediction granularity is evaluated, so
        coordinates present in the ground truth do not automatically turn a
        correct city- or country-level answer into a failure.
        """

        target = self._normalize_granularity(
            state.metadata.get("target_granularity") or ground_truth.get("target_granularity")
        )
        if not ground_truth:
            return HierarchicalMemoryOutcome(target_granularity=target)
        if state.final_answer is None:
            return HierarchicalMemoryOutcome(
                target_granularity=target,
                success=False,
                label="failure",
            )

        answer = state.final_answer
        predicted = self._normalize_granularity(answer.granularity)
        levels = ("continent", "country", "region", "city", "street", "poi", "coordinates")
        correctness: dict[str, bool | None] = {level: None for level in levels}
        text_fields: dict[str, tuple[Any, Any]] = {
            "continent": (self._first_present(ground_truth, "continent"), None),
            "country": (self._first_present(ground_truth, "country"), answer.country),
            "region": (
                self._first_present(ground_truth, "region", "state", "province"),
                answer.region if predicted != "city" else None,
            ),
            "city": (
                self._first_present(ground_truth, "city"),
                answer.location_name if predicted == "city" else answer.region,
            ),
            "street": (
                self._first_present(ground_truth, "street", "road"),
                answer.location_name if predicted == "street" else None,
            ),
            "poi": (
                self._first_present(ground_truth, "poi", "place_name", "name"),
                answer.location_name if predicted == "poi" else None,
            ),
        }
        if predicted in text_fields and not text_fields[predicted][1]:
            gt_value, _ = text_fields[predicted]
            text_fields[predicted] = (gt_value, answer.location_name)
        for level, (gt_value, pred_value) in text_fields.items():
            normalized_gt = self._normalize_name(gt_value)
            if not normalized_gt:
                continue
            normalized_pred = self._normalize_name(pred_value)
            correctness[level] = bool(
                normalized_pred
                and (normalized_gt in normalized_pred or normalized_pred in normalized_gt)
            )

        gt_lat = self._float_or_none(self._first_present(ground_truth, "lat", "latitude"))
        gt_lon = self._float_or_none(self._first_present(ground_truth, "lon", "lng", "longitude"))
        pred_lat = answer.lat
        pred_lon = answer.lon
        error_distance_m: float | None = None
        if gt_lat is not None and gt_lon is not None:
            if pred_lat is None or pred_lon is None:
                correctness["coordinates"] = False
            else:
                error_distance_m = self._haversine_m(pred_lat, pred_lon, gt_lat, gt_lon)

        evaluated = target or predicted
        success = self._success_at_granularity(correctness, evaluated)
        any_correct = any(value is True for value in correctness.values())
        if success is True:
            label = "full_success"
        elif any_correct:
            label = "partial_success"
        elif success is False:
            label = "failure"
        else:
            label = "unverifiable"
        return HierarchicalMemoryOutcome(
            target_granularity=target,
            predicted_granularity=predicted,
            evaluated_granularity=evaluated,
            level_correctness=correctness,
            success=success,
            label=label,
            error_distance_m=error_distance_m,
        )

    @staticmethod
    def _normalize_granularity(value: Any) -> str | None:
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "coordinate": "coordinates",
            "lat_lon": "coordinates",
            "latlng": "coordinates",
            "point": "coordinates",
            "place": "poi",
            "landmark": "poi",
            "address": "street",
            "road": "street",
            "state": "region",
            "province": "region",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in {"continent", "country", "region", "city", "street", "poi", "coordinates"}:
            return normalized
        return None

    @staticmethod
    def _success_at_granularity(correctness: dict[str, bool | None], granularity: str | None) -> bool | None:
        if granularity is None:
            return None
        levels = ("continent", "country", "region", "city", "street", "poi", "coordinates")
        start = levels.index(granularity)
        comparable = [correctness[level] for level in levels[start:] if correctness[level] is not None]
        if any(value is True for value in comparable):
            return True
        if comparable:
            return False
        return None

    def usage_role(self, item: MemoryItem) -> str:
        if item.memory_type == "failure_pattern":
            return "failure_avoidance"
        if item.memory_type == "tool_policy":
            return "tool_policy"
        if item.memory_type == "evidence_reliability":
            return "warning"
        return "strategy_hint"

    def _normalize_name(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _float_or_none(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _first_present(self, mapping: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = mapping.get(key)
            if value is not None and value != "":
                return value
        return None

    def _haversine_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
