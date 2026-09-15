"""Tests for the concurrent Direct-VLM evaluator."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_concurrent_vlm_eval", _ROOT / "scripts" / "run_concurrent_vlm_eval.py"
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def test_chat_completion_url_accepts_base_or_full_endpoint():
    assert (
        _SCRIPT.chat_completion_url("https://openrouter.ai/api/v1")
        == "https://openrouter.ai/api/v1/chat/completions"
    )
    endpoint = "https://example.test/v1/chat/completions"
    assert _SCRIPT.chat_completion_url(endpoint) == endpoint


def test_direct_prompt_contains_task_without_placeholder():
    prompt = _SCRIPT.build_prompt()
    properties = _SCRIPT._response_schema()["json_schema"]["schema"]["properties"]

    assert "Always provide one best WGS84 coordinate estimate" in prompt
    assert "{query}" not in prompt
    assert set(properties) == {"country", "city", "latitude", "longitude"}


def test_run_item_parses_response_and_scores_im2gps(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (2, 2), "white").save(image_path)
    item = _SCRIPT._eval.EvalItem(
        dataset="im2gps3ktest",
        subset="im2gps3ktest",
        split="test",
        item_id="sample",
        image_path=image_path,
        image_rel="sample.jpg",
        gt_latitude=10.0,
        gt_longitude=20.0,
    )
    output = {
        "country": "Exampleland",
        "city": "Example City",
        "latitude": 10.0,
        "longitude": 20.0,
    }
    captured = {}

    def fake_post_with_retry(**kwargs):
        captured["payload"] = kwargs["payload"]
        return (
            {
                "id": "generation-1",
                "model": "z-ai/glm-5.3-flash",
                "provider": "Z.AI",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(output)}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.001},
            },
            1,
        )

    monkeypatch.setattr(
        _SCRIPT,
        "post_with_retry",
        fake_post_with_retry,
    )
    args = argparse.Namespace(
        max_tokens=1200,
        temperature=0.0,
        image_max_edge=0,
        image_jpeg_quality=90,
        response_format="json_schema",
        timeout=240.0,
        max_retries=4,
        retry_base_delay=2.0,
        reasoning_effort="low",
    )

    row = _SCRIPT.run_item(
        item,
        index=1,
        total=1,
        prompt="Geolocate this image.",
        args=args,
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="z-ai/glm-5.3-flash",
        provider_slug="z-ai",
    )

    assert row["status"] == "completed"
    assert row["request_id"] == "generation-1"
    assert row["requested_provider"] == "z-ai"
    assert row["provider"] == "Z.AI"
    assert captured["payload"]["provider"] == {
        "only": ["z-ai"],
        "allow_fallbacks": False,
    }
    assert captured["payload"]["reasoning"] == {"effort": "low"}
    assert row["pred_country"] == "Exampleland"
    assert row["pred_city"] == "Example City"
    assert row["pred_granularity"] == ""
    assert row["correct_1000m"] == "1"
    assert row["correct_2500000m"] == "1"


def test_summary_counts_missing_coordinates_as_incorrect(tmp_path):
    output_path = tmp_path / "results.csv"
    rows = [
        {
            "item_id": "a",
            "status": "completed",
            "gt_latitude": "10",
            "gt_longitude": "20",
            "pred_gt_distance_m": "0",
            **{column: "1" for column in _SCRIPT.IM2GPS_COLUMNS},
        },
        {
            "item_id": "b",
            "status": "completed",
            "gt_latitude": "10",
            "gt_longitude": "20",
            "pred_gt_distance_m": "",
            **{column: "0" for column in _SCRIPT.IM2GPS_COLUMNS},
        },
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_SCRIPT.FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    _SCRIPT.write_summary(output_path)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["coordinate_coverage"] == 0.5
    assert summary["im2gps3k_accuracy"]["street_1km"]["accuracy"] == 0.5
