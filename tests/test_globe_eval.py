"""Tests for the GLOBE text-and-geocoding evaluator."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_globe_eval", _ROOT / "scripts" / "run_globe_eval.py"
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def test_parse_official_globe_output_and_build_geocode_query():
    content = """<think>French language and the Eiffel Tower are visible.</think>
<answer>
country: France
city: Paris
</answer>"""

    parsed = _SCRIPT.parse_globe_output(content)
    query = _SCRIPT.build_geocode_query(
        {"pred_country": parsed["country"], "pred_city": parsed["city"]}
    )

    assert parsed["country"] == "France"
    assert parsed["city"] == "Paris"
    assert parsed["thinking"] == "French language and the Eiffel Tower are visible."
    assert query == "Paris, France"


def test_parse_globe_output_accepts_answer_without_tags():
    parsed = _SCRIPT.parse_globe_output("**Country:** Japan\n**City:** Kyoto")

    assert parsed["country"] == "Japan"
    assert parsed["city"] == "Kyoto"


def test_inference_payload_matches_official_generation_defaults(monkeypatch):
    monkeypatch.setattr(_SCRIPT._shared, "image_to_data_url", lambda *_args, **_kwargs: "data:image/jpeg;base64,x")
    item = argparse.Namespace(image_path=Path("image.jpg"))
    args = argparse.Namespace(
        image_max_edge=0,
        image_jpeg_quality=90,
        temperature=0.1,
        max_tokens=512,
        frequency_penalty=0.7,
        presence_penalty=0.7,
    )

    payload = _SCRIPT.build_inference_payload(item, args, "qwen2.5-vl")

    assert payload["model"] == "qwen2.5-vl"
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 512
    assert payload["frequency_penalty"] == 0.7
    assert payload["presence_penalty"] == 0.7
    assert "<answer>" in payload["messages"][1]["content"][1]["text"]


def test_coordinate_thresholds_include_project_and_paper_levels():
    assert list(_SCRIPT.DISTANCE_LEVELS_M.values()) == [
        500,
        1_000,
        2_000,
        10_000,
        25_000,
        200_000,
        750_000,
        2_500_000,
    ]
