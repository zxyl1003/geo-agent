"""LLM-driven central brain agent for ReAct geolocation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from geoagent.agents.base import BaseAgent
from geoagent.core.context_builder import ContextBuilder
from geoagent.core.json_utils import clamp_float, extract_json_payload
from geoagent.core.prompt_loader import load_prompt
from geoagent.core.registry import agent_registry
from geoagent.core.schemas import (
    AgentOutput,
    BrainDecision,
    BrainVisualAnalysis,
    BrainVisualUpdate,
    ObservedEntity,
    OCRResult,
    ToolRequest,
    VisualCue,
)
from geoagent.models.llm_client import LLMClient
from geoagent.models.vlm_client import VLMClient
from geoagent.state.task_state import GeoLocalizationState


SummaryItem = str | dict[str, Any]


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_evidence: list[SummaryItem] = Field(default_factory=list)
    candidate_locations: list[SummaryItem] = Field(default_factory=list)
    tool_outcomes: list[SummaryItem] = Field(default_factory=list)
    contradictions: list[SummaryItem] = Field(default_factory=list)
    unresolved_questions: list[SummaryItem] = Field(default_factory=list)
    memory_references: list[SummaryItem] = Field(default_factory=list)


@agent_registry.register("brain")
class BrainAgent(BaseAgent):
    name = "brain"
    description = "Perceives the image, chooses tools, and finalizes the geolocation answer."

    def run(self, state: GeoLocalizationState) -> AgentOutput:
        decision = self.decide(state)
        self._record_memory_references(state, decision.memory_references)
        decision_payload = decision.model_dump(exclude={"visual_analysis", "visual_updates"})
        state.metadata["last_brain_decision"] = decision_payload
        state.touch()

        tool_requests = self._tool_requests_from_decision(decision)
        state_delta: dict[str, Any] = {"brain_decision": decision_payload}
        if decision.visual_analysis is not None or decision.visual_updates is not None:
            state_delta.update(
                {
                    "visual_cues": [cue.model_dump() for cue in state.visual_cues],
                    "observed_entities": [entity.model_dump() for entity in state.observed_entities],
                    "ocr_results": [ocr.model_dump() for ocr in state.ocr_results],
                }
            )

        return AgentOutput(
            agent_name=self.name,
            message=decision.reasoning_summary,
            tool_requests=tool_requests,
            state_delta=state_delta,
            metadata={"decision": decision},
        )

    def _apply_visual_analysis(
        self,
        state: GeoLocalizationState,
        visual_analysis: BrainVisualAnalysis,
        response: dict[str, Any],
    ) -> None:
        analysis = visual_analysis.model_dump(mode="json")
        self._merge_visual_evidence(state, analysis)
        state.metadata["vlm_analysis"] = analysis
        state.metadata["vlm_provider"] = response.get("provider")
        locatability = analysis.get("locatability")
        if locatability and isinstance(locatability, dict):
            state.metadata["locatability"] = {
                "max_expected_granularity": str(locatability.get("max_expected_granularity", "country")),
                "score": clamp_float(locatability.get("score")),
                "rationale": str(locatability.get("rationale", ""))[:300],
            }
        state.touch()

    def _apply_visual_updates(
        self,
        state: GeoLocalizationState,
        visual_updates: BrainVisualUpdate,
        response: dict[str, Any],
    ) -> None:
        updates = visual_updates.model_dump(mode="json")
        before = (len(state.visual_cues), len(state.observed_entities), len(state.ocr_results))
        self._merge_visual_evidence(state, updates)
        after = (len(state.visual_cues), len(state.observed_entities), len(state.ocr_results))
        state.metadata.setdefault("visual_updates", []).append(
            {
                **updates,
                "added_counts": {
                    "visual_cues": after[0] - before[0],
                    "observed_entities": after[1] - before[1],
                    "ocr_results": after[2] - before[2],
                },
            }
        )
        state.metadata["vlm_provider"] = response.get("provider")
        state.touch()

    def _merge_visual_evidence(self, state: GeoLocalizationState, analysis: dict[str, Any]) -> None:
        cues: list[VisualCue] = list(state.visual_cues)
        observed_entities: list[ObservedEntity] = list(state.observed_entities)
        ocr_results: list[OCRResult] = list(state.ocr_results)
        cues.extend(self._visual_cues_from_analysis(analysis))
        new_entities = self._observed_entities_from_analysis(analysis)
        entity_indices = {entity.id: index for index, entity in enumerate(observed_entities)}
        for entity in new_entities:
            if entity.id in entity_indices:
                observed_entities[entity_indices[entity.id]] = entity
            else:
                observed_entities.append(entity)
        ocr_results.extend(self._ocr_results_from_analysis(analysis))
        observed_entities.extend(self._unassigned_phone_entities(cues, ocr_results))
        state.visual_cues = self._dedupe_cues(cues)
        state.observed_entities = self._dedupe_entities(observed_entities)
        state.ocr_results = self._dedupe_ocr(ocr_results)

    def _visual_cues_from_analysis(self, analysis: dict[str, Any]) -> list[VisualCue]:
        cues: list[VisualCue] = []
        summary = analysis.get("scene_summary")
        if summary:
            cues.append(VisualCue(cue_type="scene_summary", text=str(summary), source="vlm"))
        for raw in analysis.get("visual_cues", []) or []:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            cues.append(
                VisualCue(
                    cue_type=str(raw.get("cue_type", "other")),
                    text=text,
                    confidence=clamp_float(raw.get("confidence")),
                    source="vlm",
                    metadata={
                        key: value
                        for key, value in raw.items()
                        if key not in {"cue_type", "text", "confidence"}
                    },
                )
            )
        for query in analysis.get("search_queries", []) or []:
            if str(query).strip():
                cues.append(VisualCue(cue_type="search_query", text=str(query).strip(), source="vlm"))
        return cues

    def _ocr_results_from_analysis(self, analysis: dict[str, Any]) -> list[OCRResult]:
        results: list[OCRResult] = []
        for raw in analysis.get("ocr_results", []) or []:
            if isinstance(raw, str):
                text = raw.strip()
                language = None
                confidence = None
            elif isinstance(raw, dict):
                text = str(raw.get("text", "")).strip()
                language = raw.get("language")
                confidence = clamp_float(raw.get("confidence"))
            else:
                continue
            if text:
                results.append(OCRResult(text=text, language=language, confidence=confidence, source="vlm"))
        return results

    def _observed_entities_from_analysis(self, analysis: dict[str, Any]) -> list[ObservedEntity]:
        entities: list[ObservedEntity] = []
        for raw in analysis.get("observed_entities", []) or []:
            if not isinstance(raw, dict):
                continue
            text_items = self._string_list(raw.get("text_items"))
            phones = self._string_list(raw.get("phones"))
            name = str(raw.get("name") or "").strip() or None
            if not any((name, text_items, phones)):
                continue
            payload = {
                "entity_type": self._entity_type(raw.get("entity_type")),
                "name": name,
                "text_items": text_items,
                "phones": phones,
                "region_hint": str(raw.get("region_hint") or "").strip() or None,
                "source": "vlm",
                "confidence": clamp_float(raw.get("confidence")),
                "metadata": {
                    key: value
                    for key, value in raw.items()
                    if key not in {
                        "id",
                        "entity_type",
                        "name",
                        "text_items",
                        "phones",
                        "region_hint",
                        "confidence",
                    }
                },
            }
            raw_id = str(raw.get("id") or "").strip()
            if raw_id:
                payload["id"] = raw_id
            entities.append(ObservedEntity(**payload))
        return entities

    def _unassigned_phone_entities(
        self,
        cues: list[VisualCue],
        ocr_results: list[OCRResult],
    ) -> list[ObservedEntity]:
        entities: list[ObservedEntity] = []
        for text, source, confidence in [
            *((cue.text, cue.source, cue.confidence) for cue in cues),
            *((result.text, result.source, result.confidence) for result in ocr_results),
        ]:
            for phone in self._phones_from_text(text):
                entities.append(
                    ObservedEntity(
                        entity_type="phone",
                        name=None,
                        text_items=[text],
                        phones=[phone],
                        region_hint="unassigned visible phone/text",
                        source=source,
                        confidence=confidence,
                        metadata={"attribution": "unassigned_phone"},
                    )
                )
        return entities

    def _phones_from_text(self, text: str) -> list[str]:
        phones: list[str] = []
        for match in re.finditer(r"(?<!\d)(?:\+?\d[\d\s().-]{5,}\d)(?!\d)", text):
            digits = re.sub(r"\D", "", match.group(0))
            if 6 <= len(digits) <= 16 and digits not in phones:
                phones.append(digits)
        return phones

    def _dedupe_cues(self, cues: list[VisualCue]) -> list[VisualCue]:
        deduped: list[VisualCue] = []
        seen: set[tuple[str, str]] = set()
        for cue in cues:
            key = (cue.cue_type.lower(), cue.text.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cue)
        return deduped

    def _dedupe_ocr(self, results: list[OCRResult]) -> list[OCRResult]:
        deduped: list[OCRResult] = []
        seen: set[str] = set()
        for result in results:
            key = result.text.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    def _dedupe_entities(self, entities: list[ObservedEntity]) -> list[ObservedEntity]:
        deduped: list[ObservedEntity] = []
        seen: set[tuple[str, str, str]] = set()
        for entity in entities:
            key = (
                entity.entity_type,
                (entity.name or "").strip().lower(),
                "|".join(sorted(entity.phones))
                or "|".join(item.strip().lower() for item in entity.text_items),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list | tuple | set):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]

    def _entity_type(self, value: Any) -> str:
        text = str(value or "poi").strip().lower()
        return text if text in {
            "poi",
            "landmark",
            "phone",
            "address",
            "sign",
            "vehicle",
            "other",
        } else "other"

    def decide(self, state: GeoLocalizationState) -> BrainDecision:
        max_retries = 3
        last_error = None
        retry_instruction = ""
        require_visual_analysis = (
            state.metadata.get("phase") == "initial_brain_decision"
            and not state.metadata.get("vlm_analysis")
        )

        for attempt in range(max_retries):
            context = ContextBuilder(self.app_config, self.tools).build_brain_context(state)
            messages = self._build_messages(state, context)
            if retry_instruction:
                messages.append({"role": "user", "content": retry_instruction})
            prompt_preview = self._messages_preview(messages)
            response = LLMClient(app_config=self.app_config, model_role="brain").generate(
                prompt_preview,
                messages=messages,
                json_mode=True,
                temperature=self.app_config.models.get("brain").temperature
                if self.app_config.models
                else 0.15,
                max_tokens=self.app_config.models.get("brain").max_tokens
                if self.app_config.models
                else 1800,
            )
            raw_text = str(response.get("text", ""))
            self.record_model_usage(state, "brain", response)
            try:
                payload = extract_json_payload(raw_text)
                decision = BrainDecision.model_validate(payload)
                if require_visual_analysis and decision.visual_analysis is None:
                    raise ValueError("The initial Brain decision must include visual_analysis.")
                if require_visual_analysis and decision.visual_updates is not None:
                    raise ValueError("The initial Brain decision cannot include visual_updates.")
                if not require_visual_analysis and decision.visual_analysis is not None:
                    raise ValueError("Later Brain decisions must set visual_analysis to null.")
                if not require_visual_analysis and decision.visual_updates is not None:
                    raise ValueError(
                        "Later Brain decisions must set visual_updates to null and request visual_reanalysis "
                        "when targeted source-image inspection is necessary."
                    )
                if decision.visual_analysis is not None:
                    self._apply_visual_analysis(state, decision.visual_analysis, response)
                elif decision.visual_updates is not None:
                    self._apply_visual_updates(state, decision.visual_updates, response)
                self._append_decision_message(state, decision, raw_text=raw_text)
                return decision
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    state.metadata[f"brain_parse_error_attempt_{attempt + 1}"] = str(exc)
                    retry_instruction = (
                        f"The previous response violated the BrainDecision contract: {exc}. "
                        "Return a corrected JSON object. On later decisions visual_analysis and visual_updates "
                        "must both be null; request visual_reanalysis if source-image inspection is necessary."
                    )
                else:
                    state.metadata["brain_parse_error"] = str(exc)
                    raise RuntimeError(f"Brain model did not return a valid decision JSON after {max_retries} attempts: {exc}") from exc

    def _system_prompt(self) -> str:
        visual_instructions = load_prompt(
            self.app_config.config_dir,
            "brain_perception.md",
        )
        decision_instructions = load_prompt(
            self.app_config.config_dir,
            "brain_agent.md",
            shared_names=(
                "coordinate_contract.md",
                "provider_attribution.md",
                "granularity_levels.md",
            ),
        )
        return f"{visual_instructions}\n\n{decision_instructions}"

    def _build_messages(self, state: GeoLocalizationState, context: dict[str, Any]) -> list[dict[str, Any]]:
        messages = self._assemble_messages(state, context)
        input_budget = self._brain_input_budget()
        estimated_tokens = self._estimate_messages_tokens(messages)
        if estimated_tokens <= input_budget:
            return messages

        old_message_count = self._message_count_before_recent_rounds(state)
        if old_message_count:
            self._summarize_old_messages(state, old_message_count)
            messages = self._assemble_messages(state, context)
            estimated_tokens = self._estimate_messages_tokens(messages)

        if estimated_tokens > input_budget:
            raise RuntimeError(
                "Brain system prompt, current state, structured summary, and the preserved recent rounds exceed "
                f"the configured input token budget ({estimated_tokens} estimated tokens > {input_budget})."
            )
        return messages

    def _assemble_messages(
        self,
        state: GeoLocalizationState,
        context: dict[str, Any],
        *,
        include_summary: bool = True,
    ) -> list[dict[str, Any]]:
        # Some OpenAI-compatible hosts (e.g. SiliconFlow Qwen) reject a payload
        # with more than one system message ("System message must be at the
        # beginning"). Fold the compaction summary into the single system prompt
        # instead of emitting a second system message.
        system_content = self._system_prompt()
        if include_summary and state.brain_conversation_summary:
            system_content += (
                "\n\nStructured conversation summary from earlier rounds:\n"
                + state.brain_conversation_summary
            )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]

        conversation_messages = state.brain_messages
        if conversation_messages and self._latest_tool_observation(state) is conversation_messages[-1]:
            # The newest tool observation is already present in latest_tool_result.
            conversation_messages = conversation_messages[:-1]
        for message in conversation_messages:
            messages.append(self._to_chat_message(message.model_dump()))

        current_state = json.dumps(context, ensure_ascii=False, indent=2, default=str)
        initial_visual_analysis = (
            state.metadata.get("phase") == "initial_brain_decision"
            and not state.metadata.get("vlm_analysis")
        )
        if initial_visual_analysis:
            current_request = (
                "Current compact state is below. Inspect the attached source image directly, begin the "
                "geolocation investigation, and return the next BrainDecision JSON.\n\n"
                f"## Compact State\n{current_state}"
            )
        else:
            current_request = (
                "Current compact state is below. Continue the geolocation investigation from the recorded visual "
                "evidence and tool results; the source image is not attached on this turn. Request "
                "visual_reanalysis only if a targeted source-image inspection is necessary, and return the next "
                "BrainDecision JSON.\n\n"
                f"## Compact State\n{current_state}"
            )
        content: str | list[dict[str, Any]] = current_request
        if initial_visual_analysis and Path(state.image_path).exists():
            content = [
                {"type": "text", "text": current_request},
                {
                    "type": "image_url",
                    "image_url": {"url": VLMClient.image_data_url(state.image_path)},
                },
            ]
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )
        return messages

    def _latest_tool_observation(self, state: GeoLocalizationState) -> Any | None:
        if not state.tool_results or not state.brain_messages:
            return None
        message = state.brain_messages[-1]
        if message.role == "tool" and message.name == state.tool_results[-1].tool_name:
            return message
        return None

    def _brain_input_budget(self) -> int:
        config = self.app_config.system.get("brain_conversation", {})
        context_window = int(config.get("context_window", 65536))
        max_input_ratio = float(config.get("max_input_ratio", 0.90))
        if not 0 < max_input_ratio <= 1:
            raise ValueError("Brain max_input_ratio must be greater than zero and at most one.")
        safety_margin = int(config.get("safety_margin_tokens", 3072))
        model_config = self.app_config.models.get("brain")
        output_tokens = model_config.max_tokens if model_config else 2048
        ratio_budget = math.floor(context_window * max_input_ratio)
        capacity_budget = context_window - output_tokens - safety_margin
        input_budget = min(ratio_budget, capacity_budget)
        if input_budget <= 0:
            raise ValueError("Brain context window must exceed max_tokens plus safety_margin_tokens.")
        return input_budget

    def _estimate_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        # UTF-8 bytes / 3 is conservative for mixed JSON, English, and CJK text
        # without coupling context management to one provider's tokenizer.
        return sum(
            4 + math.ceil(len(self._text_message_content(message.get("content", "")).encode("utf-8")) / 3)
            for message in messages
        )

    def _summary_max_tokens(self) -> int:
        config = self.app_config.system.get("brain_conversation", {})
        return int(config.get("summary_max_tokens", 2048))

    def _to_chat_message(self, message: dict[str, Any]) -> dict[str, Any]:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        name = message.get("name")
        if role == "tool":
            role = "user"
            content = f"Tool result from {name or 'tool'}:\n{content}"
        return {"role": role, "content": content}

    def _messages_preview(self, messages: list[dict[str, Any]]) -> str:
        preview = []
        for message in messages:
            raw_content = message.get("content", "")
            content = self._text_message_content(raw_content)
            if isinstance(raw_content, list) and any(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in raw_content
            ):
                content += "\n[image attached]"
            preview.append(f"{message.get('role')}: {content[:800]}")
        return "\n\n".join(preview)

    def _text_message_content(self, content: Any) -> str:
        if not isinstance(content, list):
            return str(content)
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    def _append_decision_message(
        self,
        state: GeoLocalizationState,
        decision: BrainDecision,
        raw_text: str | None,
    ) -> None:
        content = raw_text or json.dumps(decision.model_dump(), ensure_ascii=False, default=str)
        metadata: dict[str, Any] = {
            "action_type": decision.action_type,
            "confidence": decision.confidence,
            "memory_references": decision.memory_references,
        }
        if decision.tool_requests:
            metadata["tool_requests"] = [request.model_dump() for request in decision.tool_requests]
        state.add_brain_message("assistant", content, name="brain", metadata=metadata)

    def _record_memory_references(self, state: GeoLocalizationState, references: list[str]) -> None:
        """Attribute only explicitly cited memories that were returned to Brain."""

        usages = state.metadata.get("memory_usage")
        if not isinstance(usages, list):
            return
        returned_ids = {
            str(usage.get("memory_id"))
            for usage in usages
            if isinstance(usage, dict) and usage.get("was_returned_to_brain") and usage.get("memory_id")
        }
        cited_ids = {str(memory_id) for memory_id in references if str(memory_id) in returned_ids}
        invalid_ids = [str(memory_id) for memory_id in references if str(memory_id) not in returned_ids]
        for usage in usages:
            if not isinstance(usage, dict):
                continue
            memory_id = str(usage.get("memory_id") or "")
            if memory_id in cited_ids:
                usage["was_cited_by_brain"] = True
        if invalid_ids:
            state.metadata["invalid_memory_references"] = sorted(set(invalid_ids))
        state.metadata["cited_memory_ids"] = sorted(
            {
                str(usage.get("memory_id"))
                for usage in usages
                if isinstance(usage, dict) and usage.get("was_cited_by_brain")
            }
        )

    def _message_count_before_recent_rounds(self, state: GeoLocalizationState) -> int:
        config = self.app_config.system.get("brain_conversation", {})
        preserve_recent_rounds = int(config.get("preserve_recent_rounds", 3))
        if preserve_recent_rounds <= 0:
            raise ValueError("preserve_recent_rounds must be greater than zero.")
        round_starts = [
            index
            for index, message in enumerate(state.brain_messages)
            if message.role == "assistant"
        ]
        if len(round_starts) <= preserve_recent_rounds:
            return 0
        return round_starts[-preserve_recent_rounds]

    def _summarize_old_messages(self, state: GeoLocalizationState, count: int) -> None:
        latest_tool_observation = self._latest_tool_observation(state)
        old_messages = state.brain_messages[:count]
        messages_to_summarize = [message for message in old_messages if message is not latest_tool_observation]
        if messages_to_summarize:
            state.brain_conversation_summary = self._generate_conversation_summary(
                state=state,
                existing_summary=state.brain_conversation_summary,
                old_messages=messages_to_summarize,
                max_tokens=self._summary_max_tokens(),
            )
        state.brain_messages = state.brain_messages[count:]
        state.touch()

    def _generate_conversation_summary(
        self,
        state: GeoLocalizationState,
        existing_summary: str | None,
        old_messages: list[Any],
        max_tokens: int,
    ) -> str:
        transcript = []
        for message in old_messages:
            label = message.role + (f":{message.name}" if message.name else "")
            transcript.append(f"[{label}]\n{message.content}")
        transcript_text = "\n\n".join(transcript) if transcript else "(none; compress the existing summary further)"
        summary_messages = [
            {
                "role": "system",
                "content": (
                    "You are the geolocation Brain compressing your own earlier conversation for later continuation. "
                    "Produce a concise factual handoff, not a new decision. Preserve observed evidence and provenance, "
                    "candidate locations and coordinates, tool calls and outcomes, contradictions, unresolved questions, "
                    "and cited memory IDs. Do not invent evidence, reassess the case, recommend a next action, or emit "
                    "BrainDecision JSON. Return only one JSON object with exactly these array fields: "
                    "observed_evidence, candidate_locations, tool_outcomes, contradictions, unresolved_questions, "
                    "and memory_references. Each array item may be either a concise string or a JSON object. "
                    "Use an empty array when a category has no facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Existing summary:\n"
                    f"{existing_summary or '(none)'}\n\n"
                    "Earlier messages to absorb:\n"
                    f"{transcript_text}"
                ),
            },
        ]
        summary = None
        retry_instruction = None
        for attempt in range(3):
            request_messages = list(summary_messages)
            if retry_instruction:
                request_messages.append({"role": "user", "content": retry_instruction})
            response = LLMClient(app_config=self.app_config, model_role="brain").generate(
                self._messages_preview(request_messages),
                messages=request_messages,
                json_mode=True,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            self.record_model_usage(state, "brain_summary", response)
            raw_text = str(response.get("text", "")).strip()
            try:
                payload = extract_json_payload(raw_text)
                summary = ConversationSummary.model_validate(payload)
                break
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Brain returned an invalid structured conversation summary after 3 attempts: {exc}"
                    ) from exc
                retry_instruction = (
                    f"The previous summary was invalid: {exc}. Regenerate it from the original material. "
                    "Return a shorter, complete JSON object with every brace, bracket, and string closed. "
                    "Use only the six required array fields and no Markdown."
                )

        if summary is None:
            raise AssertionError("Conversation summary generation completed without a summary.")
        if not any(summary.model_dump().values()):
            raise RuntimeError("Brain returned an empty structured conversation summary.")

        summary_text = json.dumps(summary.model_dump(), ensure_ascii=False, indent=2)
        source_text = "\n".join(part for part in (existing_summary, transcript_text) if part)
        if self._estimate_messages_tokens([{"content": summary_text}]) >= self._estimate_messages_tokens(
            [{"content": source_text}]
        ):
            raise RuntimeError("Brain structured conversation summary did not reduce the prior context size.")
        return summary_text

    def _tool_requests_from_decision(self, decision: BrainDecision) -> list[ToolRequest]:
        return list(decision.tool_requests) if decision.action_type in {"call_tool", "call_tools"} else []
