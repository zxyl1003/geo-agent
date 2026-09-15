
"""OCR tool using a VLM for text extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geoagent.core.json_utils import clamp_float, extract_json_payload
from geoagent.core.registry import tool_registry
from geoagent.core.schemas import OCRResult
from geoagent.models.vlm_client import VLMClient, brain_credentials_configured
from geoagent.tools.base import BaseTool


@tool_registry.register("ocr")
class OCRTool(BaseTool):
    name = "ocr"
    description = "Extract visible text from an image."

    def is_available(self, state: Any = None) -> bool:
        return brain_credentials_configured(self.app_config)

    def run(self, **kwargs: Any):
        image_path = str(kwargs.get("image_path", ""))
        results, response = self._real_ocr(image_path)
        model_usage: dict[str, Any] = {
            "role": "tool:ocr",
            "model": response.get("model"),
            "provider": response.get("provider"),
            "usage": response.get("usage") if isinstance(response.get("usage"), dict) else {},
        }

        return self.result(
            data={
                "image_path": image_path,
                "provider": response.get("provider"),
                "error": None,
                "ocr_results": [item.model_dump() for item in results],
                "text": [item.text for item in results],
                "model_usage": model_usage,
            }
        )

    # NOTE: 这里的 ocr 是使用 VLM 完成的, 后续或者根据需要可以修改为专用的 OCR 模型, 例如 PaddleOCR
    def _real_ocr(self, image_path: str) -> tuple[list[OCRResult], dict[str, Any]]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        prompt = """
Extract ALL visible text from this street-view image. Be thorough — read every sign, label, sticker, poster, licence plate, building number, shop name, and piece of visible writing.

Return only JSON:
{
  "ocr_results": [
    {"text": "exact visible text copied character-by-character", "language": "detected language or null", "confidence": 0.0}
  ]
}

Rules:
- Copy text EXACTLY as written, preserving original script (Japanese, Arabic, Chinese, Cyrillic, Thai, Hindi, etc.)
- Do NOT transliterate — if the sign says "止まれ", output "止まれ", not "tomare"
- Include the TYPE of text source when possible: sign, plate, shop, billboard, bus, etc.
- Include text from: street signs, direction signs, shop names, billboards, bus destinations, licence plates, building numbers, construction signs, speed limits, warning signs, graffiti, vehicle text
- Skip text that is too blurry or obscured to read reliably (confidence < 0.3)
- For licence plates: note the colour scheme (e.g. "white plate with blue strip", "yellow plate with black text")
""".strip()
        response = VLMClient(app_config=self.app_config).generate(
            prompt,
            image_path=image_path,
            json_mode=True,
            max_tokens=2500,
        )
        try:
            payload = extract_json_payload(str(response.get("text", "")))
        except Exception as exc:  # noqa: BLE001 - re-raise as a clear tool error.
            raise RuntimeError(f"OCR VLM response could not be parsed: {exc}") from exc
        results: list[OCRResult] = []
        for raw in payload.get("ocr_results", []) or []:
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
                results.append(OCRResult(text=text, confidence=confidence, language=language, source=self.name))
        return results, response
