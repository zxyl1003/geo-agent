"""Targeted Brain visual reanalysis tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geoagent.core.json_utils import clamp_float, extract_json_payload
from geoagent.core.registry import tool_registry
from geoagent.core.schemas import ObservedEntity, OCRResult, VisualCue
from geoagent.models.vlm_client import VLMClient, brain_credentials_configured
from geoagent.tools.base import BaseTool


@tool_registry.register("visual_reanalysis")
class VisualReanalysisTool(BaseTool):
    name = "visual_reanalysis"
    description = (
        "Ask the Brain to re-examine the source image for one concrete unresolved region, object, text, "
        "or visual question that could materially affect the location decision."
    )

    def is_available(self, state: Any = None) -> bool:
        return brain_credentials_configured(self.app_config)

    def run(self, **kwargs: Any):
        image_path = str(kwargs.get("image_path", ""))
        question = str(kwargs.get("question") or "Re-examine the image for geolocation clues.")
        region_hint = str(kwargs.get("region_hint") or "")
        target = str(kwargs.get("target") or "")
        analysis_type = str(kwargs.get("analysis_type") or "custom")

        analysis, response = self._run_vlm_reanalysis(
            image_path=image_path,
            question=question,
            region_hint=region_hint,
            target=target,
            analysis_type=analysis_type,
        )
        model_usage: dict[str, Any] = {
            "role": "tool:visual_reanalysis",
            "model": response.get("model"),
            "provider": response.get("provider"),
            "usage": response.get("usage") if isinstance(response.get("usage"), dict) else {},
        }

        observations = self._visual_cues_from_analysis(analysis)
        ocr_results = self._ocr_results_from_analysis(analysis)
        observed_entities = self._observed_entities_from_analysis(analysis)
        return self.result(
            data={
                "image_path": image_path,
                "provider": response.get("provider"),
                "question": question,
                "region_hint": region_hint or None,
                "target": target or None,
                "analysis_type": analysis_type,
                "answer": analysis.get("answer", ""),
                "observations": [item.model_dump() for item in observations],
                "visual_cues": [item.model_dump() for item in observations],
                "ocr_results": [item.model_dump() for item in ocr_results],
                "observed_entities": [item.model_dump() for item in observed_entities],
                "uncertainty": analysis.get("uncertainty"),
                "suggested_next_questions": analysis.get("suggested_next_questions", []),
                "model_usage": model_usage,
                "raw_analysis": analysis,
            }
        )

    def _run_vlm_reanalysis(
        self,
        image_path: str,
        question: str,
        region_hint: str,
        target: str,
        analysis_type: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        prompt = f"""
You are a targeted visual reanalysis tool for geolocation.

Inspect the same street-view image again, but focus on the requested region/object/question.
Do not guess beyond visual evidence. If text is uncertain, include the uncertainty.
If the focused object is a globally or nationally famous landmark/building and you can directly recognize it from visual form, visible text fragments, symbols, or facade features, add it to observed_entities with entity_type="landmark". Distinguish direct recognition from mere resemblance in the answer and uncertainty fields.

Request:
- question: {question}
- region_hint: {region_hint or "not specified"}
- target: {target or "not specified"}
- analysis_type: {analysis_type}

Return only JSON in this exact shape:
{{
  "answer": "direct answer to the question",
  "observations": [
    {{
      "cue_type": "text|signage|landmark|architecture|road|vehicle|vegetation|infrastructure|object|scene|other",
      "text": "observable clue",
      "confidence": 0.0,
      "region": "where in the image this was observed",
      "rationale": "why this clue matters or why it is uncertain"
    }}
  ],
  "ocr_results": [
    {{"text": "exact visible text if any", "language": "language or null", "confidence": 0.0}}
  ],
  "observed_entities": [
    {{
      "id": "stable id such as ent_left_store",
      "entity_type": "poi|landmark|phone|address|sign|vehicle|other",
      "name": "visible POI/business/sign/landmark name or directly recognized landmark name, or null",
      "text_items": ["all text visibly tied to this same object/storefront"],
      "phones": ["phone numbers only if visually tied to this same object/storefront"],
      "region_hint": "where this entity appears in the image",
      "confidence": 0.0
    }}
  ],
  "uncertainty": "what remains visually unclear",
  "suggested_next_questions": ["optional next visual question"]
}}
""".strip()
        response = VLMClient(app_config=self.app_config).generate(
            prompt,
            image_path=image_path,
            json_mode=True,
            max_tokens=1600,
        )
        try:
            payload = extract_json_payload(str(response.get("text", "")))
        except Exception as exc:  # noqa: BLE001 - re-raise as a clear tool error.
            raise RuntimeError(f"VLM reanalysis response could not be parsed: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("VLM reanalysis response did not contain a JSON object.")
        return payload, response

    def _visual_cues_from_analysis(self, analysis: dict[str, Any]) -> list[VisualCue]:
        cues: list[VisualCue] = []
        answer = str(analysis.get("answer") or "").strip()
        if answer:
            cues.append(VisualCue(cue_type="visual_reanalysis_answer", text=answer, source=self.name))
        for raw in analysis.get("observations", []) or []:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            metadata = {
                key: value
                for key, value in raw.items()
                if key not in {"cue_type", "text", "confidence"}
            }
            cues.append(
                VisualCue(
                    cue_type=str(raw.get("cue_type", "other")),
                    text=text,
                    confidence=clamp_float(raw.get("confidence")),
                    source=self.name,
                    metadata=metadata,
                )
            )
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
                results.append(OCRResult(text=text, language=language, confidence=confidence, source=self.name))
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
                "source": self.name,
                "confidence": clamp_float(raw.get("confidence")),
                "metadata": {
                    key: value
                    for key, value in raw.items()
                    if key not in {"id", "entity_type", "name", "text_items", "phones", "region_hint", "confidence"}
                },
            }
            raw_id = str(raw.get("id") or "").strip()
            if raw_id:
                payload["id"] = raw_id
            entities.append(ObservedEntity(**payload))
        return entities

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list | tuple | set):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]

    def _entity_type(self, value: Any) -> str:
        text = str(value or "other").strip().lower()
        return text if text in {"poi", "landmark", "phone", "address", "sign", "vehicle", "other"} else "other"
