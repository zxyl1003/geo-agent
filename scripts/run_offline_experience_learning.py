"""Build an experience library from three GeoExp7K learning-set runs.

The script first runs the complete ``learning`` split three times without
experience retrieval or online writes. It then groups the three independent
trajectories for every sample, asks the experience model for at most one reusable
lesson, and persists accepted lessons to the SQLite/Chroma experience store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dataset_eval as dataset_eval  # noqa: E402
from geoagent.core.config import load_app_config  # noqa: E402
from geoagent.core.json_utils import extract_json_payload  # noqa: E402
from geoagent.core.prompt_loader import load_prompt  # noqa: E402
from geoagent.memory.manager import MemoryManager  # noqa: E402
from geoagent.memory.models import MemoryAgentReview  # noqa: E402
from geoagent.models.llm_client import LLMClient  # noqa: E402
from geoagent.state.task_state import GeoLocalizationState  # noqa: E402


DEFAULT_TRAJECTORY_COUNT = 3
DEFAULT_CONTEXT_CHAR_BUDGET = 48_000
DEFAULT_SELECTION_RADIUS_M = 25_000.0
CONSISTENCY_RETRY_RESERVE_CHARS = 2_000


@dataclass(frozen=True)
class TraceTrajectory:
    run_label: str
    trace_path: Path
    trace_error: str
    state: GeoLocalizationState | None


def _item_id(state: GeoLocalizationState) -> str | None:
    task_metadata = state.metadata.get("task_metadata") or {}
    value = str(task_metadata.get("item_id") or "").strip()
    return value or None


def load_trace_run(run_dir: Path) -> dict[str, TraceTrajectory]:
    trace_root = run_dir / "traces" if (run_dir / "traces").is_dir() else run_dir
    if not trace_root.is_dir():
        raise FileNotFoundError(f"Trace directory does not exist: {trace_root}")

    run_label = run_dir.resolve().name
    trajectories: dict[str, TraceTrajectory] = {}
    for trace_path in sorted(trace_root.rglob("*.json")):
        if trace_path.name == "summary.json":
            continue
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        state_payload = payload.get("state")
        state = (
            GeoLocalizationState.model_validate(state_payload)
            if isinstance(state_payload, dict)
            else None
        )
        trace_key = trace_path.relative_to(trace_root).with_suffix("").as_posix()
        if trace_key in trajectories:
            raise ValueError(f"Duplicate trace key {trace_key!r} in {trace_root}")
        trajectories[trace_key] = TraceTrajectory(
            run_label=run_label,
            trace_path=trace_path,
            trace_error=str(payload.get("error") or ""),
            state=state,
        )
    if not trajectories:
        raise ValueError(f"No trace JSON files found under {trace_root}")
    return trajectories


def load_trajectory_groups(run_dirs: list[Path]) -> dict[str, list[TraceTrajectory]]:
    runs = [load_trace_run(run_dir) for run_dir in run_dirs]
    reference_ids = set(runs[0])
    mismatches: list[str] = []
    for run_dir, run in zip(run_dirs[1:], runs[1:]):
        missing = sorted(reference_ids - set(run))
        extra = sorted(set(run) - reference_ids)
        if missing or extra:
            mismatches.append(
                f"{run_dir}: missing={len(missing)} {missing[:3]}, "
                f"extra={len(extra)} {extra[:3]}"
            )
    if mismatches:
        raise ValueError("Run trace sets do not match:\n" + "\n".join(mismatches))
    groups: dict[str, list[TraceTrajectory]] = {}
    for trace_key in sorted(reference_ids):
        trajectories = [run[trace_key] for run in runs]
        item_ids = {
            item_id
            for trajectory in trajectories
            if trajectory.state is not None
            for item_id in [_item_id(trajectory.state)]
            if item_id
        }
        if len(item_ids) > 1:
            raise ValueError(
                f"Trace {trace_key!r} has inconsistent item_ids across runs: {sorted(item_ids)}"
            )
        item_id = next(iter(item_ids), trace_key)
        if item_id in groups:
            raise ValueError(f"Duplicate grouped item_id {item_id!r}")
        groups[item_id] = trajectories
    return groups


def _shorten(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _compact_value(
    value: Any,
    *,
    string_limit: int,
    list_limit: int,
    depth: int = 0,
) -> Any:
    if depth >= 5:
        return _shorten(value, string_limit)
    if isinstance(value, dict):
        return {
            str(key): _compact_value(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
                depth=depth + 1,
            )
            for key, item in value.items()
            if key not in {"raw", "image_base64"}
        }
    if isinstance(value, list):
        return [
            _compact_value(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
                depth=depth + 1,
            )
            for item in value[:list_limit]
        ]
    if isinstance(value, str):
        return _shorten(value, string_limit)
    return value


def _selected_indices(total: int, limit: int) -> list[int]:
    if total <= limit:
        return list(range(total))
    first_count = (limit + 1) // 2
    last_count = limit - first_count
    return list(range(first_count)) + list(range(total - last_count, total))


def _trajectory_summary(
    trajectory: TraceTrajectory,
    episode: dict[str, Any],
    *,
    tool_limit: int,
    evidence_limit: int,
    hypothesis_limit: int,
    string_limit: int,
) -> dict[str, Any]:
    calls = episode.get("tool_calls") or []
    results = episode.get("tool_results") or []
    interactions = []
    for index in _selected_indices(max(len(calls), len(results)), tool_limit):
        call = calls[index] if index < len(calls) else None
        result = results[index] if index < len(results) else None
        interactions.append(
            {
                "sequence_index": index,
                "call": _compact_value(
                    call,
                    string_limit=string_limit,
                    list_limit=evidence_limit,
                ),
                "result": _compact_value(
                    result,
                    string_limit=string_limit,
                    list_limit=evidence_limit,
                ),
            }
        )

    state = trajectory.state
    if state is None:
        raise ValueError(f"Trace has no recoverable state: {trajectory.trace_path}")
    final_decision = state.metadata.get("last_brain_decision") or {}
    hypotheses = sorted(
        episode.get("hypotheses") or [],
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )[:hypothesis_limit]
    return {
        "run_label": trajectory.run_label,
        "trace_error": trajectory.trace_error or None,
        "episode_id": episode["episode_id"],
        "outcome": episode.get("outcome") or {},
        "final_answer": _compact_value(
            episode.get("final_answer"),
            string_limit=string_limit,
            list_limit=evidence_limit,
        ),
        "perception": {
            "scene_summary": _shorten(
                (state.metadata.get("vlm_analysis") or {}).get("scene_summary"),
                string_limit,
            ),
            "visual_clues": _compact_value(
                (episode.get("visual_clues") or [])[:evidence_limit],
                string_limit=string_limit,
                list_limit=evidence_limit,
            ),
            "ocr_results": _compact_value(
                (episode.get("ocr_results") or [])[:evidence_limit],
                string_limit=string_limit,
                list_limit=evidence_limit,
            ),
            "observed_entities": _compact_value(
                (episode.get("observed_entities") or [])[:evidence_limit],
                string_limit=string_limit,
                list_limit=evidence_limit,
            ),
        },
        "top_hypotheses": _compact_value(
            hypotheses,
            string_limit=string_limit,
            list_limit=evidence_limit,
        ),
        "reasoning_summary": {
            "conversation_summary": _shorten(state.brain_conversation_summary, string_limit),
            "final_decision": _shorten(final_decision.get("reasoning_summary"), string_limit),
        },
        "tool_interactions": interactions,
        "resource_usage": {
            "model_calls": state.api_call_count.get("total", 0),
            "tool_calls": state.tool_call_count.get("total", len(calls)),
            "tokens": state.token_usage.get("total_tokens", 0),
        },
    }


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _ground_truth(state: GeoLocalizationState) -> dict[str, Any]:
    return dict(state.metadata.get("ground_truth") or {})


def _coordinates(mapping: dict[str, Any]) -> tuple[float, float] | None:
    lat = _first_present(mapping, "lat", "latitude")
    lon = _first_present(mapping, "lon", "lng", "longitude")
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def _haversine_m(first: tuple[float, float], second: tuple[float, float]) -> float:
    radius = 6_371_000.0
    lat1, lon1 = first
    lat2, lon2 = second
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def classify_radius_outcome(
    trajectories: list[TraceTrajectory],
    selection_radius_m: float,
) -> dict[str, Any]:
    radius_checks: list[bool | None] = []
    predicted_coordinates: list[tuple[str, tuple[float, float]]] = []
    for trajectory in trajectories:
        state = trajectory.state
        if state is None:
            radius_checks.append(None)
            continue
        truth_coordinates = _coordinates(_ground_truth(state))
        answer = state.final_answer
        prediction_coordinates = (
            (answer.lat, answer.lon)
            if answer is not None and answer.lat is not None and answer.lon is not None
            else None
        )
        if truth_coordinates is None or prediction_coordinates is None:
            radius_checks.append(None)
        else:
            radius_checks.append(
                _haversine_m(prediction_coordinates, truth_coordinates) <= selection_radius_m
            )
            predicted_coordinates.append((trajectory.run_label, prediction_coordinates))

    pairwise = []
    for first_index, (first_label, first_coord) in enumerate(predicted_coordinates):
        for second_label, second_coord in predicted_coordinates[first_index + 1 :]:
            pairwise.append(
                {
                    "runs": [first_label, second_label],
                    "prediction_distance_m": round(_haversine_m(first_coord, second_coord), 2),
                }
            )
    max_pairwise = max(
        (item["prediction_distance_m"] for item in pairwise),
        default=None,
    )
    true_count = sum(value is True for value in radius_checks)
    false_count = sum(value is False for value in radius_checks)
    if true_count == len(radius_checks):
        outcome_pattern = "all_within_selection_radius"
    elif true_count and false_count:
        outcome_pattern = "mixed_radius_outcome"
    elif false_count == len(radius_checks):
        suffix = (
            "divergent"
            if max_pairwise is not None and max_pairwise > selection_radius_m
            else "convergent"
        )
        outcome_pattern = f"all_outside_selection_radius_{suffix}"
    else:
        outcome_pattern = "missing_coordinate_outcome"
    return {
        "outcome_pattern": outcome_pattern,
        "within_selection_radius_count": true_count,
        "outside_selection_radius_count": false_count,
        "pairwise_prediction_distances": pairwise,
        "max_pairwise_prediction_distance_m": max_pairwise,
    }


def _name_match(first: Any, second: Any) -> bool | None:
    first_text = " ".join(str(first or "").casefold().split())
    second_text = " ".join(str(second or "").casefold().split())
    if not first_text or not second_text:
        return None
    return first_text in second_text or second_text in first_text


def _administrative_comparison(
    state: GeoLocalizationState,
    ground_truth_context: dict[str, Any],
) -> dict[str, Any]:
    answer = state.final_answer
    if answer is None:
        return {}
    ground_truth = _ground_truth(state)
    predicted_city = answer.city
    if not predicted_city and str(answer.granularity) == "city":
        predicted_city = answer.location_name
    pairs = {
        "country": (
            answer.country,
            _first_present(ground_truth_context, "country")
            or _first_present(ground_truth, "country"),
        ),
        "region": (
            answer.region,
            _first_present(ground_truth_context, "region")
            or _first_present(ground_truth, "region", "state", "province"),
        ),
        "city": (
            predicted_city,
            _first_present(ground_truth_context, "city", "town")
            or _first_present(ground_truth, "city"),
        ),
    }
    return {
        level: {
            "predicted": predicted,
            "ground_truth": truth,
            "strict_name_match": _name_match(predicted, truth),
        }
        for level, (predicted, truth) in pairs.items()
        if predicted or truth
    }


def _aggregate_facts(
    manager: MemoryManager,
    trajectories: list[TraceTrajectory],
    episodes: list[dict[str, Any]],
    selection_radius_m: float,
) -> dict[str, Any]:
    per_trajectory = []
    ground_truth_context = dict(episodes[0].get("ground_truth_context") or {})
    for trajectory, episode in zip(trajectories, episodes):
        facts = manager._precomputed_reflection_facts(episode)
        distance = episode.get("error_distance_m")
        within_radius = None if distance is None else float(distance) <= selection_radius_m
        if trajectory.state is None:
            raise ValueError(f"Trace has no recoverable state: {trajectory.trace_path}")
        administrative_comparison = _administrative_comparison(
            trajectory.state,
            ground_truth_context,
        )
        known_level_correctness = dict(facts.get("known_level_correctness") or {})
        for level in ("country", "region", "city"):
            strict_match = (administrative_comparison.get(level) or {}).get(
                "strict_name_match"
            )
            if strict_match is not None:
                known_level_correctness[level] = strict_match
        facts["known_level_correctness"] = known_level_correctness
        per_trajectory.append(
            {
                "run_label": trajectory.run_label,
                **facts,
                "within_selection_radius": within_radius,
                "administrative_text_comparison": administrative_comparison,
            }
        )

    radius_outcome = classify_radius_outcome(trajectories, selection_radius_m)

    levels = ("continent", "country", "region", "city", "street", "poi", "coordinates")
    aggregate_correctness: dict[str, bool | None] = {}
    for level in levels:
        values = [
            facts.get("known_level_correctness", {}).get(level)
            for facts in per_trajectory
        ]
        known = [value for value in values if value is not None]
        if known and all(value is True for value in known):
            aggregate_correctness[level] = True
        elif known and all(value is False for value in known):
            aggregate_correctness[level] = False
        else:
            aggregate_correctness[level] = None

    uncertainty_values = [
        item.get("within_uncertainty_radius")
        for item in per_trajectory
        if item.get("coordinate_assessment_expected")
    ]
    known_uncertainty = [value for value in uncertainty_values if value is not None]
    aggregate_uncertainty = None
    if known_uncertainty and all(value is True for value in known_uncertainty):
        aggregate_uncertainty = True
    elif known_uncertainty and all(value is False for value in known_uncertainty):
        aggregate_uncertainty = False

    distance_checks = []
    for threshold in manager._precomputed_reflection_facts(episodes[0])[
        "distance_threshold_checks"
    ]:
        threshold_m = threshold["threshold_m"]
        checks = [
            item
            for episode in episodes
            for item in manager._precomputed_reflection_facts(episode)[
                "distance_threshold_checks"
            ]
            if item["threshold_m"] == threshold_m
        ]
        distance_checks.append(
            {
                "threshold_m": threshold_m,
                "within_count": sum(item["within_threshold"] is True for item in checks),
                "outside_count": sum(item["within_threshold"] is False for item in checks),
                "missing_count": sum(item["within_threshold"] is None for item in checks),
            }
        )

    return {
        "evaluated_granularity": "multi_trajectory",
        "coordinate_assessment_expected": any(
            item.get("coordinate_assessment_expected") for item in per_trajectory
        ),
        "known_level_correctness": aggregate_correctness,
        "within_uncertainty_radius": aggregate_uncertainty,
        "distance_threshold_checks": distance_checks,
        "selection_radius_m": selection_radius_m,
        "selection_radius_is_for_routing_not_success_definition": True,
        **radius_outcome,
        "per_trajectory": per_trajectory,
    }


def _existing_memories(
    manager: MemoryManager,
    comparison_facts: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    query_parts = [str(comparison_facts.get("outcome_pattern") or "")]
    for summary in summaries:
        query_parts.extend(
            [
                str(summary.get("final_answer") or ""),
                str(summary.get("top_hypotheses") or ""),
                str(summary.get("reasoning_summary") or ""),
                str(summary.get("tool_interactions") or ""),
            ]
        )
    hits = manager._search_memories(_shorten("\n".join(query_parts), 12_000), top_k=8)
    candidates = []
    for hit in hits:
        digest = hit.prompt_digest()
        digest["merge_dimensions"] = {
            key: hit.item.metadata[key]
            for key in manager._MERGE_DIMENSIONS
            if key in hit.item.metadata
        }
        candidates.append(digest)
    return candidates


def build_review_context(
    manager: MemoryManager,
    item_id: str,
    trajectories: list[TraceTrajectory],
    episodes: list[dict[str, Any]],
    *,
    selection_radius_m: float,
    max_context_chars: int,
) -> dict[str, Any]:
    comparison_facts = _aggregate_facts(
        manager,
        trajectories,
        episodes,
        selection_radius_m,
    )
    profiles = [
        (8, 8, 4, 700),
        (6, 6, 3, 500),
        (4, 4, 2, 350),
        (2, 3, 0, 240),
        (1, 2, 0, 160),
    ]
    last_size = 0
    for compression_level, (tool_limit, evidence_limit, hypothesis_limit, string_limit) in enumerate(profiles):
        summaries = [
            _trajectory_summary(
                trajectory,
                episode,
                tool_limit=tool_limit,
                evidence_limit=evidence_limit,
                hypothesis_limit=hypothesis_limit,
                string_limit=string_limit,
            )
            for trajectory, episode in zip(trajectories, episodes)
        ]
        context = {
            "review_mode": "offline_multi_trajectory_contrastive",
            "episode_id": episodes[0]["episode_id"],
            "source_episode_ids": [episode["episode_id"] for episode in episodes],
            "sample_id": item_id,
            "feedback_type": "ground_truth",
            "trajectory_count": len(trajectories),
            "ground_truth_context": episodes[0].get("ground_truth_context"),
            "precomputed_facts": comparison_facts,
            "trajectory_summaries": summaries,
            "existing_memory_candidates": _compact_value(
                _existing_memories(manager, comparison_facts, summaries),
                string_limit=500,
                list_limit=8,
            ),
            "merge_requirements": {
                "minimum_similarity": manager.settings.merge_similarity_threshold,
                "exact_match_dimensions": ["failure_type", "cue_category", "tool_scenario"],
                "candidate_must_be_complete_resynthesis": True,
            },
            "context_budget": {
                "maximum_serialized_characters": max_context_chars,
                "compression_level": compression_level,
            },
        }
        serialized = json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
        last_size = len(serialized)
        if last_size <= max_context_chars - CONSISTENCY_RETRY_RESERVE_CHARS:
            context["context_budget"]["actual_serialized_characters"] = last_size
            actual_size = len(
                json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
            )
            context["context_budget"]["actual_serialized_characters"] = actual_size
            if actual_size <= max_context_chars - CONSISTENCY_RETRY_RESERVE_CHARS:
                return context
    raise ValueError(
        f"Compressed review context is still too large ({last_size} > {max_context_chars} chars)."
    )


def _normalize_review_payload(payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    candidate = normalized.get("candidate")
    if isinstance(candidate, dict):
        candidate = dict(candidate)
        candidate["source_episode_id"] = str(context["episode_id"])
        candidate["feedback_type"] = "ground_truth"
        candidate.setdefault("diagnosis_confidence", normalized.get("diagnosis_confidence", 0.0))
        normalized["candidate"] = candidate
    normalized.setdefault("should_write", candidate is not None)
    normalized.setdefault("merge_memory_id", None)
    normalized.setdefault("proposal_source", "llm")
    return normalized


def _call_reviewer(
    manager: MemoryManager,
    state: GeoLocalizationState,
    context: dict[str, Any],
) -> MemoryAgentReview:
    system_prompt = "\n\n".join(
        [
            load_prompt(manager.app_config.config_dir, "memory_manager_agent.md"),
            load_prompt(manager.app_config.config_dir, "offline_memory_reflection_agent.md"),
        ]
    )
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    user_prompt = (
        "Compare these completed independent geolocation trajectories and return one JSON "
        "object only.\n\n" + context_json
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    model_config = manager.app_config.models.get("memory_manager") or manager.app_config.models.get("brain")
    response = LLMClient(
        app_config=manager.app_config,
        model_role="memory_manager",
    ).generate(
        user_prompt,
        messages=messages,
        json_mode=True,
        temperature=model_config.temperature if model_config else 0.0,
        max_tokens=model_config.max_tokens if model_config else 1800,
        thinking=model_config.thinking if model_config else {},
    )
    state.add_model_usage(
        "memory_manager",
        model_config.model_name if model_config else None,
        str(response.get("provider") or "") or None,
        response.get("usage") or {},
    )
    payload = extract_json_payload(str(response.get("text") or ""))
    review = MemoryAgentReview.model_validate(_normalize_review_payload(payload, context))
    review.proposal_source = "llm"
    if review.candidate is not None:
        review.candidate.diagnosis_confidence = max(
            review.candidate.diagnosis_confidence,
            review.diagnosis_confidence,
        )
    return review


def review_with_consistency_retry(
    manager: MemoryManager,
    state: GeoLocalizationState,
    context: dict[str, Any],
) -> MemoryAgentReview:
    review = _call_reviewer(manager, state, context)
    manager._set_proposed_merge_similarity(review, context)
    initial_issues = manager._review_consistency_issues(review, context)
    final_issues = initial_issues
    if initial_issues:
        context["consistency_issues"] = [
            _shorten(issue, 500) for issue in initial_issues[:12]
        ]
        context["consistency_retry_instruction"] = (
            "Correct every listed inconsistency and return the complete JSON review again."
        )
        retry_size = len(
            json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
        )
        if retry_size > int(context["context_budget"]["maximum_serialized_characters"]):
            raise ValueError(
                "Consistency-retry context exceeds its configured character budget."
            )
        review = _call_reviewer(manager, state, context)
        manager._set_proposed_merge_similarity(review, context)
        final_issues = manager._review_consistency_issues(review, context)

    state.metadata["memory_review_consistency"] = {
        "initial_issues": initial_issues,
        "retried": bool(initial_issues),
        "final_issues": final_issues,
        "accepted": not final_issues,
    }
    if final_issues:
        review = review.model_copy(
            update={
                "should_write": False,
                "candidate": None,
                "merge_memory_id": None,
                "rationale": (
                    "Offline memory review remained inconsistent after one retry: "
                    + "; ".join(final_issues)
                ),
            }
        )
    state.metadata["memory_agent_review"] = review.model_dump(mode="json")
    return review


def persist_group_review(
    manager: MemoryManager,
    state: GeoLocalizationState,
    episodes: list[dict[str, Any]],
    context: dict[str, Any],
    review: MemoryAgentReview,
) -> dict[str, Any]:
    attribution = (
        review.failure_attribution.model_dump(mode="json")
        if review.failure_attribution is not None
        else {}
    )
    attribution.update(
        {
            "successful_levels": list(review.successful_levels),
            "failed_levels": list(review.failed_levels),
            "review_mode": context["review_mode"],
            "trajectory_count": context["trajectory_count"],
            "outcome_pattern": context["precomputed_facts"]["outcome_pattern"],
        }
    )
    for episode in episodes:
        manager.store.update_episode_attribution(episode["episode_id"], attribution)

    candidate = review.candidate if review.should_write else None
    if candidate is None:
        return {
            "status": "episodes_recorded",
            "reason": review.rationale or "No reusable contrastive lesson was retained.",
        }

    candidate.metadata.update(
        {
            "review_mode": context["review_mode"],
            "trajectory_count": context["trajectory_count"],
            "outcome_pattern": context["precomputed_facts"]["outcome_pattern"],
            "source_episode_ids": context["source_episode_ids"],
        }
    )
    forbidden_names = manager._forbidden_place_names(context.get("ground_truth_context"))
    decision = manager.policy.candidate_decision(candidate, forbidden_names)
    if not decision.accepted:
        return {"status": "candidate_rejected", "reason": decision.reason}

    if review.merge_memory_id:
        offered_ids = {
            str(memory.get("memory_id") or "")
            for memory in context.get("existing_memory_candidates", [])
        }
        target = manager.store.get_memory(review.merge_memory_id)
        if review.merge_memory_id not in offered_ids or target is None:
            return {"status": "candidate_rejected", "reason": "invalid_merge_memory_id"}
        issues = manager._merge_target_issues(
            target,
            candidate,
            manager._candidate_similarity_to_memory(candidate, target.memory_id),
        )
        if issues:
            return {"status": "candidate_rejected", "reason": "; ".join(issues)}
        item, status = manager._merge_memory(target, candidate)
    else:
        item = manager.store.insert_memory(candidate)
        manager.index.upsert(item)
        status = "new_memory_inserted"

    for episode_id in context["source_episode_ids"][1:]:
        manager.store.link_memory_episode(
            item.memory_id,
            episode_id,
            "contrastive_source",
            diagnosis_confidence=candidate.diagnosis_confidence,
            note="independent trajectory used by offline contrastive reflection",
        )
    return {
        "status": status,
        "memory_id": item.memory_id,
        "memory_type": item.memory_type,
        "confidence": item.confidence,
        "merge_count": item.merge_count,
    }


def _offline_episode_id(item_id: str, run_label: str) -> str:
    digest = hashlib.sha256(f"{item_id}\n{run_label}".encode("utf-8")).hexdigest()[:24]
    return f"episode_offline_{digest}"


def process_group(
    manager: MemoryManager,
    item_id: str,
    trajectories: list[TraceTrajectory],
    *,
    selection_radius_m: float,
    max_context_chars: int,
    dry_run: bool,
) -> dict[str, Any]:
    missing_runs = [
        trajectory.run_label for trajectory in trajectories if trajectory.state is None
    ]
    if missing_runs:
        return {
            "sample_id": item_id,
            "trajectory_count": len(trajectories),
            "status": "incomplete_trajectory_group",
            "missing_state_runs": missing_runs,
            "trace_errors": {
                trajectory.run_label: trajectory.trace_error
                for trajectory in trajectories
                if trajectory.state is None
            },
        }
    states = [
        trajectory.state.model_copy(deep=True)
        for trajectory in trajectories
        if trajectory.state is not None
    ]
    prepared = [
        TraceTrajectory(
            run_label=trajectory.run_label,
            trace_path=trajectory.trace_path,
            trace_error=trajectory.trace_error,
            state=state,
        )
        for trajectory, state in zip(trajectories, states)
    ]
    radius_outcome = classify_radius_outcome(prepared, selection_radius_m)
    if radius_outcome["outcome_pattern"] != "mixed_radius_outcome":
        return {
            "sample_id": item_id,
            "trajectory_count": len(prepared),
            "run_labels": [trajectory.run_label for trajectory in prepared],
            "outcome_pattern": radius_outcome["outcome_pattern"],
            "within_selection_radius_count": radius_outcome[
                "within_selection_radius_count"
            ],
            "outside_selection_radius_count": radius_outcome[
                "outside_selection_radius_count"
            ],
            "status": "skipped_non_mixed_outcome",
        }
    episodes = []
    for trajectory in prepared:
        trajectory.state.metadata["memory_episode_id"] = _offline_episode_id(
            item_id,
            trajectory.run_label,
        )
        episode = manager._episode_from_state(trajectory.state)
        episodes.append(episode)
        if not dry_run:
            manager.store.insert_episode(episode)
            manager.store.record_usage(
                episode["episode_id"],
                trajectory.state.metadata.get("memory_usage") or [],
            )

    context = build_review_context(
        manager,
        item_id,
        prepared,
        episodes,
        selection_radius_m=selection_radius_m,
        max_context_chars=max_context_chars,
    )
    result = {
        "sample_id": item_id,
        "trajectory_count": len(prepared),
        "run_labels": [trajectory.run_label for trajectory in prepared],
        "outcome_pattern": context["precomputed_facts"]["outcome_pattern"],
        "within_selection_radius_count": context["precomputed_facts"][
            "within_selection_radius_count"
        ],
        "context_characters": context["context_budget"]["actual_serialized_characters"],
        "compression_level": context["context_budget"]["compression_level"],
    }
    if dry_run:
        result["status"] = "dry_run_context_ready"
        return result

    primary_state = prepared[0].state
    primary_state.metadata["memory_review_context"] = context
    review = review_with_consistency_retry(manager, primary_state, context)
    update = persist_group_review(manager, primary_state, episodes, context, review)
    result.update(
        {
            "status": update["status"],
            "memory_update": update,
            "review": review.model_dump(mode="json"),
            "consistency": primary_state.metadata.get("memory_review_consistency") or {},
        }
    )
    return result


def _read_completed(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("sample_id") and row.get("status") not in {
            "error",
            "incomplete_trajectory_group",
            "dry_run_context_ready",
        }:
            completed.add(str(row["sample_id"]))
    return completed


def _write_summary(output_path: Path, rows: list[dict[str, Any]]) -> Path:
    statuses: dict[str, int] = {}
    patterns: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        pattern = str(row.get("outcome_pattern") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        patterns[pattern] = patterns.get(pattern, 0) + 1
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "output": str(output_path),
                "processed": len(rows),
                "status_counts": statuses,
                "outcome_pattern_counts": patterns,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def _evaluation_args(args: argparse.Namespace, run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_root=args.dataset_root,
        datasets="geoexp7k-learning",
        output=str(run_dir / "results.csv"),
        trace_dir=str(run_dir / "traces"),
        query=args.query,
        limit=args.limit,
        offset=args.offset,
        resume=args.resume,
        debug=args.debug,
        no_color=args.no_color,
        dry_run=args.dry_run,
        write_missing=args.write_missing,
        fail_fast=args.fail_fast,
        workers=args.workers,
        memory_mode="off",
        memory_dir="",
    )


def run_learning_trajectories(args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir)
    run_dirs = [output_dir / f"run_{index}" for index in range(1, DEFAULT_TRAJECTORY_COUNT + 1)]
    selected_runs = run_dirs[:1] if args.dry_run else run_dirs
    for index, run_dir in enumerate(selected_runs, start=1):
        print(
            f"=== GeoExp7K learning trajectory {index}/{DEFAULT_TRAJECTORY_COUNT} ===",
            flush=True,
        )
        dataset_eval.run_eval(_evaluation_args(args, run_dir))
    return run_dirs


def run_reflection(args: argparse.Namespace, run_dirs: list[Path]) -> None:
    resolved_run_dirs = [run_dir.resolve() for run_dir in run_dirs]
    groups = load_trajectory_groups(resolved_run_dirs)
    item_ids = list(groups)

    output_path = (
        Path(args.reviews_output)
        if args.reviews_output
        else Path(args.output_dir) / "reviews.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_completed(output_path) if args.resume else set()
    file_mode = "a" if args.resume else "w"

    app_config = load_app_config(config_dir=ROOT / "configs", env_file=ROOT / ".env")
    memory_config = app_config.system.setdefault("memory", {})
    memory_config.update(
        {
            "enabled": True,
            "mode": "learn_only",
            "storage_dir": str(Path(args.experience_dir).resolve()),
        }
    )
    manager = MemoryManager.from_config(app_config)
    if manager is None:
        raise RuntimeError("Could not initialize the offline experience manager.")

    rows: list[dict[str, Any]] = []
    with output_path.open(file_mode, encoding="utf-8") as output_file:
        for index, item_id in enumerate(item_ids, start=1):
            if item_id in completed:
                continue
            try:
                row = process_group(
                    manager,
                    item_id,
                    groups[item_id],
                    selection_radius_m=args.selection_radius_m,
                    max_context_chars=args.max_context_chars,
                    dry_run=False,
                )
            except Exception as exc:  # noqa: BLE001 - isolate per-sample external-model failures
                row = {
                    "sample_id": item_id,
                    "status": "error",
                    "error": str(exc),
                }
                if args.fail_fast:
                    raise
            output_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            output_file.flush()
            rows.append(row)
            print(f"[{index}/{len(item_ids)}] {item_id}: {row['status']}", flush=True)

    if args.resume:
        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    summary_path = _write_summary(output_path, rows)
    print(f"Experience library: {Path(args.experience_dir).resolve()}")
    print(f"Reviews: {output_path}")
    print(f"Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GeoExp7K learning three times, compare each sample's trajectories, "
            "and build an offline experience library."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(ROOT / "datasets"),
        help="Root directory containing datasets/GeoExp7k.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "experience_learning" / "geoexp7k"),
        help="Directory for the three trajectory runs and review log.",
    )
    parser.add_argument(
        "--experience-dir",
        "--memory-dir",
        dest="experience_dir",
        default=str(ROOT / "outputs" / "experience" / "geoexp7k"),
        help="Output directory for the SQLite/Chroma experience library.",
    )
    parser.add_argument(
        "--reviews-output",
        default="",
        help="Per-sample review JSONL. Default: OUTPUT_DIR/reviews.jsonl.",
    )
    parser.add_argument("--query", default=dataset_eval.DEFAULT_QUERY)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--write-missing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate GeoExp7K learning discovery without model calls or experience writes.",
    )
    parser.add_argument(
        "--selection-radius-m",
        type=float,
        default=DEFAULT_SELECTION_RADIUS_M,
        help="Radius used only to classify trajectory disagreement, not to define success.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_CONTEXT_CHAR_BUDGET,
        help="Hard serialized-character limit for each sample's comparison context.",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    if args.max_context_chars < 8_000:
        raise ValueError("--max-context-chars must be at least 8000.")
    if args.selection_radius_m <= 0:
        raise ValueError("--selection-radius-m must be positive.")
    if not args.resume and not args.dry_run:
        for path in (Path(args.output_dir), Path(args.experience_dir)):
            if path.is_dir() and any(path.iterdir()):
                raise FileExistsError(
                    f"Output directory is not empty: {path}. "
                    "Use --resume or choose a new directory."
                )
    run_dirs = run_learning_trajectories(args)
    if args.dry_run:
        print("Dry run complete. No model calls or experience writes were made.")
        return
    run_reflection(args, run_dirs)


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
