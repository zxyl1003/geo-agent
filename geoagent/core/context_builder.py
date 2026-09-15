"""Build compact, tool-aware context for LLM-driven agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geoagent.core.config import AppConfig
from geoagent.core.provider_policy import available_map_providers
from geoagent.state.task_state import GeoLocalizationState
from geoagent.tools.base import BaseTool


class ContextBuilder:
    def __init__(self, app_config: AppConfig, tools: dict[str, BaseTool] | None = None) -> None:
        self.app_config = app_config
        self.tools = tools or {}

    def build_brain_context(self, state: GeoLocalizationState) -> dict[str, Any]:
        return {
            "task": {
                "task_id": state.task_id,
                "image_path": Path(state.image_path).name,
                "user_query": state.user_query,
                "step_count": state.step_count,
                "status": state.status,
                "phase": state.metadata.get("phase"),
                "locatability": state.metadata.get("locatability"),  # Soft hint: max expected granularity
            },
            "visual_cues": [
                cue.model_dump()
                for cue in sorted(state.visual_cues, key=lambda item: item.confidence or 0.0, reverse=True)[:8]
            ],
            "observed_entities": [
                entity.model_dump()
                for entity in sorted(state.observed_entities, key=lambda item: item.confidence or 0.0, reverse=True)[:8]
            ],
            "ocr_results": [
                result.model_dump()
                for result in sorted(state.ocr_results, key=lambda item: item.confidence or 0.0, reverse=True)[:6]
            ],
            "hypotheses": [self._summarize_hypothesis(hyp) for hyp in state.hypotheses[:5]],
            "latest_tool_result": self.summarize_tool_result(state.tool_results[-1]) if state.tool_results else None,
            "memory": self._structured_memory(state),
            "resource_usage": self._resource_usage(state),
            "available_tools": self.available_tool_specs(state),
            "uncertainty": state.uncertainty.model_dump(),
            "notes": {
                "tool_outputs_are_untrusted": True,
                "avoid_duplicate_tool_calls": True,
                "duplicate_tool_result_policy": (
                    "After a duplicate result, change the arguments meaningfully, choose a different available tool, "
                    "or submit the best supported final answer. Never submit the same request again."
                ),
                "coordinate_contract": "All public lat/lon fields in state and tool results are WGS84. Provider-native coordinates, when present, are stored as raw_lat/raw_lon/raw_coordinate_system.",
                "pending_experience_revision": state.metadata.get("pending_experience_revision"),
            },
        }

    def build_experience_context(
        self,
        state: GeoLocalizationState,
        *,
        wake_reasons: list[str],
        proposed_decision: Any | None,
        help_request: str | None,
        new_tool_results: list[Any],
        candidate_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a whitelisted online context that cannot expose evaluation truth."""

        if proposed_decision is not None:
            decision = proposed_decision.model_dump(
                mode="json",
                exclude={"visual_analysis", "visual_updates"},
            )
        else:
            raw_decision = state.metadata.get("last_brain_decision")
            decision = (
                self._compact_dict(
                    raw_decision,
                    (
                        "reasoning_summary",
                        "memory_references",
                        "experience_request",
                        "action_type",
                        "tool_requests",
                        "hypothesis_patch",
                        "confidence",
                        "uncertainty_radius_m",
                    ),
                )
                if isinstance(raw_decision, dict)
                else None
            )
        return {
            "progress": {
                "brain_step": state.step_count,
                "completed_tool_rounds": int(state.metadata.get("completed_tool_rounds", 0)),
                "wake_reasons": wake_reasons,
            },
            "visual_cues": [
                {
                    "cue_type": cue.cue_type,
                    "text": cue.text,
                    "confidence": cue.confidence,
                    "source": cue.source,
                }
                for cue in sorted(state.visual_cues, key=lambda item: item.confidence or 0.0, reverse=True)[:8]
            ],
            "observed_entities": [
                {
                    "id": entity.id,
                    "entity_type": entity.entity_type,
                    "name": entity.name,
                    "text_items": entity.text_items[:5],
                    "phones": entity.phones[:3],
                    "region_hint": entity.region_hint,
                    "confidence": entity.confidence,
                    "source": entity.source,
                }
                for entity in sorted(state.observed_entities, key=lambda item: item.confidence or 0.0, reverse=True)[:8]
            ],
            "ocr_results": [
                {
                    "text": result.text,
                    "confidence": result.confidence,
                    "language": result.language,
                    "source": result.source,
                }
                for result in sorted(state.ocr_results, key=lambda item: item.confidence or 0.0, reverse=True)[:6]
            ],
            "hypotheses": [self._summarize_hypothesis(hypothesis) for hypothesis in state.hypotheses[:5]],
            "proposed_brain_decision": decision,
            "proposed_decision_status": "pending_not_executed" if proposed_decision is not None else None,
            "new_tool_results": [self.summarize_tool_result(result) for result in new_tool_results],
            "brain_help_request": help_request,
            "previous_experience_guidance": state.metadata.get("experience_guidance"),
            "current_schedule": state.metadata.get("experience_next_check"),
            "recent_experience_checks": [
                {
                    "check_id": check.get("check_id"),
                    "completed_tool_rounds": check.get("completed_tool_rounds"),
                    "wake_reasons": check.get("wake_reasons"),
                    "intervene": check.get("intervene"),
                    "selected_memory_ids": check.get("selected_memory_ids"),
                    "guidance": self._shorten(check.get("guidance"), 300),
                    "effective_next_check": check.get("effective_next_check"),
                }
                for check in (state.metadata.get("experience_checks") or [])[-3:]
                if isinstance(check, dict)
            ],
            "candidate_memories": candidate_memories,
            "available_tool_names": sorted(
                name
                for name, tool in self.tools.items()
                if not tool.hidden and tool.is_available(state)
            ),
            "notes": {
                "candidate_memories_are_strategy_only": True,
                "ground_truth_is_not_available": True,
            },
        }

    def build_experience_retrieval_query(self, context: dict[str, Any]) -> str:
        """Construct the retrieval query directly from the whitelisted task state."""

        keys = [
            "progress",
            "hypotheses",
            "proposed_brain_decision",
            "proposed_decision_status",
            "new_tool_results",
            "brain_help_request",
            "recent_experience_checks",
        ]
        wake_reasons = (context.get("progress") or {}).get("wake_reasons") or []
        if "initial_brain_decision" in wake_reasons:
            keys[1:1] = ["visual_cues", "observed_entities", "ocr_results"]
        query_context = {key: context.get(key) for key in keys}
        if "initial_brain_decision" not in wake_reasons:
            query_context["visual_evidence_digest"] = {
                "cue_types": sorted(
                    {
                        str(cue.get("cue_type"))
                        for cue in context.get("visual_cues") or []
                        if isinstance(cue, dict) and cue.get("cue_type")
                    }
                ),
                "named_entities": [
                    {
                        "entity_type": entity.get("entity_type"),
                        "name": entity.get("name"),
                        "phones": entity.get("phones"),
                    }
                    for entity in context.get("observed_entities") or []
                    if isinstance(entity, dict) and entity.get("name")
                ][:5],
            }
        return json.dumps(query_context, ensure_ascii=False, sort_keys=True, default=str)

    def available_tool_specs(self, state: GeoLocalizationState) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for name, tool in sorted(self.tools.items()):
            if tool.hidden:
                continue
            if not tool.is_available(state):
                continue
            calls = [call for call in state.tool_calls if call.tool_name == name]
            remaining = max(0, tool.max_calls_per_task - len(calls))
            specs.append(
                {
                    "name": name,
                    "description": tool.description,
                    "cost_level": tool.cost_level,
                    "max_calls_per_task": tool.max_calls_per_task,
                    "remaining_calls": remaining,
                    "argument_guidance": self._argument_guidance(name),
                }
            )
        return specs

    def _compact_dict(self, value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {key: value.get(key) for key in keys if key in value and value.get(key) is not None}

    def _summarize_hypothesis(self, hypothesis: Any) -> dict[str, Any]:
        metadata = self._compact_dict(
            getattr(hypothesis, "metadata", {}) or {},
            (
                "address",
                "address_hint",
                "geocoded_address",
                "place_id",
                "poi_place_id",
                "resource_name",
                "geocoding_provider",
                "poi_provider",
                "poi_map_provider",
                "precise_poi_coordinate",
                "poi_direct_finalizable",
                "coordinate_note",
                "reference_coordinate",
                "rejected",
            ),
        )
        rationale = getattr(hypothesis, "rationale", None)
        return {
            "id": hypothesis.id,
            "name": hypothesis.name,
            "country": hypothesis.country,
            "region": hypothesis.region,
            "granularity": hypothesis.granularity,
            "lat": hypothesis.lat,
            "lon": hypothesis.lon,
            "score": hypothesis.score,
            "rationale": rationale[:500] if isinstance(rationale, str) else rationale,
            "metadata": metadata,
        }

    def _shorten(self, value: Any, limit: int = 300) -> Any:
        if isinstance(value, str):
            return value[:limit]
        return value

    def _compact_search_result(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return self._shorten(value, 240)
        compact = self._compact_dict(
            value,
            (
                "title",
                "name",
                "snippet",
                "description",
                "url",
                "source",
                "source_label",
                "position",
                "engine",
                "address",
                "formatted_address",
                "lat",
                "lon",
            ),
        )
        for key in ("snippet", "description"):
            if key in compact:
                compact[key] = self._shorten(compact[key], 240)
        return compact

    def _compact_candidate(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return self._shorten(value, 240)
        compact = self._compact_dict(
            value,
            (
                "place_id",
                "resource_name",
                "name",
                "address",
                "formatted_address",
                "category",
                "types",
                "lat",
                "lon",
                "map_provider",
                "provider",
                "observed_entity_id",
                "telephone",
                "phone",
                "phone_number",
                "website",
                "description",
                "rating",
                "rating_count",
                "cid",
                "position",
                "details_complete",
            ),
        )
        for key in ("types",):
            if isinstance(compact.get(key), list):
                compact[key] = compact[key][:5]
        return compact

    def _compact_details(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return self._shorten(value, 300)
        return self._compact_dict(
            value,
            (
                "place_id",
                "resource_name",
                "name",
                "address",
                "formatted_address",
                "category",
                "types",
                "lat",
                "lon",
                "map_provider",
                "provider",
                "telephone",
                "phone",
                "shop_hours",
                "observed_entity_id",
            ),
        )

    def _compact_verification(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return self._shorten(value, 300)
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"raw", "model_usage"}:
                continue
            if isinstance(item, list):
                compact[key] = item[:3]
            elif isinstance(item, dict):
                compact[key] = self._compact_dict(
                    item,
                    (
                        "verdict",
                        "confidence",
                        "summary",
                        "message",
                        "matched",
                        "contradicted",
                        "has_coverage",
                        "streetview_available",
                    ),
                )
            elif isinstance(item, str):
                compact[key] = self._shorten(item, 400)
            else:
                compact[key] = item
        return compact

    def summarize_tool_result(self, result: Any) -> dict[str, Any]:
        if result is None:
            return {}
        data = result.data if hasattr(result, "data") else {}
        tool_name = getattr(result, "tool_name", None)
        summary = {
            "tool_name": tool_name,
            "success": getattr(result, "success", None),
            "error": getattr(result, "error", None),
        }
        if not isinstance(data, dict):
            summary["data"] = self._shorten(data, 500)
            return summary

        for key in (
            "provider",
            "provider_display_name",
            "map_provider",
            "source_type",
            "map_type",
            "query",
            "region",
            "observed_entity_id",
            "reconsidering_rejected_candidate",
            "rejected_anchor_policy",
            "has_coverage",
            "streetview_available",
            "verdict",
            "confidence",
            "message",
        ):
            if key in data:
                summary[key] = self._shorten(data[key], 400)
        if tool_name in {"geocode", "reverse_geocode"}:
            for key in (
                "mode",
                "status",
                "address",
                "city",
                "latlng",
                "coordtype",
                "ret_coordtype",
                "country_code",
                "reconsidering_rejected_candidate",
                "rejected_anchor_policy",
                "rejected_candidate",
                "lat",
                "lon",
                "formatted_address",
                "top_result",
                "results",
                "attempts",
            ):
                if key in data:
                    value = data[key]
                    if key == "top_result" and isinstance(value, dict):
                        value = self._compact_candidate(value)
                    elif key == "results" and isinstance(value, list):
                        value = [self._compact_candidate(item) for item in value]
                    elif key == "attempts" and isinstance(value, list):
                        value = value[:5]
                    summary[key] = value
        if "text" in data:
            summary["text"] = data["text"][:6] if isinstance(data["text"], list) else self._shorten(data["text"], 500)
        if "results" in data and "results" not in summary:
            summary["results"] = [self._compact_search_result(item) for item in data["results"]]
        if tool_name == "web_search":
            for key in ("knowledge_graph", "answer_box", "people_also_ask", "related_searches", "provider_metadata", "cache_hit"):
                if key in data:
                    value = data[key]
                    if isinstance(value, list):
                        value = value[:5]
                    summary[key] = self._shorten(value, 800)
        if tool_name == "webpage_read":
            for key in (
                "url",
                "final_url",
                "canonical_url",
                "title",
                "description",
                "language",
                "content_type",
                "content",
                "char_count",
                "original_char_count",
                "truncated",
                "cache_hit",
            ):
                if key in data:
                    summary[key] = self._shorten(data[key], 5000 if key == "content" else 500)
        if "candidates" in data:
            summary["candidates"] = [self._compact_candidate(item) for item in data["candidates"]]
        if "details" in data:
            summary["details"] = self._compact_details(data["details"])
        if "coverage" in data:
            summary["coverage"] = data["coverage"]
        if "tiles" in data:
            summary["tile_count"] = len(data["tiles"]) if isinstance(data["tiles"], list) else None
        if tool_name == "map_tile_verify":
            for key in (
                "candidate_name",
                "candidate_id",
                "lat",
                "lon",
                "zoom",
                "tile_radius",
                "mosaic_path",
                "verification",
            ):
                if key in data:
                    summary[key] = self._compact_verification(data[key]) if key == "verification" else data[key]
            if "tiles" in data:
                summary["tile_count"] = len(data["tiles"]) if isinstance(data["tiles"], list) else None
        if tool_name == "streetview_verify":
            for key in (
                "candidate_name",
                "candidate_id",
                "lat",
                "lon",
                "radius_m",
                "headings",
                "contact_sheet_path",
                "verification",
            ):
                if key in data:
                    summary[key] = self._compact_verification(data[key]) if key == "verification" else data[key]
            if "metadata" in data and isinstance(data["metadata"], dict):
                summary["metadata"] = {
                    key: value
                    for key, value in data["metadata"].items()
                    if key not in {"raw"}
                }
            if "images" in data:
                summary["image_count"] = len(data["images"]) if isinstance(data["images"], list) else None
                summary["images"] = [self._compact_dict(item, ("heading", "path", "status")) for item in data["images"][:3] if isinstance(item, dict)]
        if "ocr_results" in data:
            summary["ocr_results"] = data["ocr_results"][:6]
        if tool_name == "visual_reanalysis":
            for key in (
                "question",
                "region_hint",
                "target",
                "analysis_type",
                "answer",
                "observations",
                "visual_cues",
                "uncertainty",
                "suggested_next_questions",
            ):
                if key in data:
                    value = data[key]
                    if isinstance(value, list):
                        value = value[:5]
                    summary[key] = self._shorten(value, 500)
        return summary

    def _structured_memory(self, state: GeoLocalizationState) -> dict[str, Any]:
        return {
            "tool_failures": state.tool_failures[-8:],
            "recalled_experience_memories": (state.metadata.get("recalled_memories") or [])[:3],
            "experience_guidance": state.metadata.get("experience_guidance"),
        }

    def _resource_usage(self, state: GeoLocalizationState) -> dict[str, Any]:
        return {
            "token_usage": state.token_usage,
            "api_call_count": state.api_call_count,
            "tool_call_count": state.tool_call_count,
            "internal_tool_call_count": state.metadata.get("internal_tool_call_count") or {},
        }

    def _argument_guidance(self, name: str) -> dict[str, Any]:
        return {
            "ocr": {"image_path": "current image path"},
            "visual_reanalysis": {
                "image_path": "current image path",
                "question": "specific visual question Brain wants answered",
                "region_hint": "optional natural language region such as upper-left, far background, storefront",
                "target": "optional object/sign/landmark to inspect",
                "analysis_type": "text | signage | landmark | architecture | road | vehicle | vegetation | infrastructure | object | scene | custom",
            },
            "web_search": {
                "query": "search query",
                "top_k": "integer 1-10; Brain chooses the desired result count (default 5)",
            },
            "webpage_read": {
                "url": "public http(s) URL returned by web_search",
                "max_chars": "optional output text limit, default 12000",
                "output": "static HTML text only; no JavaScript execution",
            },
            "geocode": {
                "address": "address string for geocoding to coordinates",
                "country": "country name when known; helps the tool route (mainland China -> Baidu, international -> LocationIQ)",
                "region": "city/province hint when known",
                "language": "optional language code, default en",
                "output": "provider routing is internal; tool returns public lat/lon as WGS84 plus the final provider in the result",
                "top_k": "integer 1-10; Brain chooses the desired result count (default 3)",
            },
            "reverse_geocode": {
                "lat": "latitude for reverse geocoding",
                "lon": "longitude for reverse geocoding",
                "latlng": "optional 'lat,lon' string for reverse geocoding",
                "country": "country name when known; helps the tool route to the right provider",
                "language": "optional language code, default en",
                "output": "provider routing is internal; tool returns formatted address plus the final provider in the result",
            },
            "poi_search": {
                "query": "POI text query — search ONE POI name per call. When multiple POIs are visible, search each separately then compare their candidate lat/lon for proximity (within ~500m = mutual corroboration)",
                "region": "candidate city or area",
                "country": "required for international search: ISO 3166-1 English name, alpha-2, or alpha-3 code (for example Japan, JP, or JPN); provider-specific gl/hl/location/map_provider arguments are not accepted",
                "observed_entity_id": "optional id from observed_entities when this query is tied to one visible storefront/sign/landmark",
                "output": "provider routing is internal; candidate lat/lon are WGS84",
                "top_k": "integer 1-10; Brain chooses the desired result count (default 5)",
            },
            "map_tile_verify": {
                "map_provider": self._provider_options("map_verification"),
                "map_type": "satellite | roadmap",
                "lat": "candidate latitude",
                "lon": "candidate longitude",
                "coordinate_system": "input coordinate system, default wgs84",
                "zoom": "optional zoom; roadmap defaults to 19 and should normally use 19-20 "
                        "for POI labels; satellite defaults to 19 and remains limited to 20. "
                        "You can adaptively adjust the zoom value.",
                "tile_radius": "0 for one tile, 1 for 3x3, 2 for 5x5",
                "candidate_name": "candidate name being verified",
                "candidate_id": "optional hypothesis id",
                "reason": "why map tiles can test this candidate",
            },
            "streetview_verify": {
                "map_provider": self._provider_options("map_verification"),
                "lat": "candidate latitude",
                "lon": "candidate longitude",
                "coordinate_system": "input coordinate system, default wgs84",
                "radius_m": "street-view metadata search radius",
                "headings": "optional list, default [0, 90, 180, 270]",
                "pitch": "optional pitch, default 0",
                "fov": "optional field of view, default 90",
                "candidate_name": "candidate name being verified",
                "candidate_id": "optional hypothesis id",
                "reason": "why street view can test this candidate",
            },
        }.get(name, {})

    def _provider_options(self, capability: str) -> str:
        # Only offer providers whose API key is configured, so the Brain never
        # selects a provider that cannot run.
        providers = sorted(available_map_providers(self.app_config, capability))
        return " | ".join(providers)
