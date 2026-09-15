"""Tests for the released GeoAgent text-and-geocoding evaluator."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_geoagent_eval", _ROOT / "scripts" / "run_geoagent_eval.py"
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def test_parse_official_geoagent_output_and_build_geocode_query():
    content = json.dumps(
        {
            "ChainOfThought": {"CountryIdentification": {"Conclusion": "France"}},
            "FinalAnswer": "France; Île-de-France; Eiffel Tower, Paris",
        }
    )

    parsed = _SCRIPT.parse_geoagent_output(content)
    query = _SCRIPT.build_geocode_query(
        {
            "pred_country": parsed["country"],
            "pred_region": parsed["region"],
            "pred_specific_location": parsed["specific_location"],
            "final_answer": parsed["final_answer"],
        }
    )

    assert parsed["country"] == "France"
    assert parsed["region"] == "Île-de-France"
    assert parsed["specific_location"] == "Eiffel Tower, Paris"
    assert query == "Eiffel Tower, Paris, Île-de-France, France"


def test_imageobench_dataset2_discovery_uses_canonical_selector(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    d2 = _SCRIPT._eval.EvalItem(
        dataset="IMAGEOBench-d2",
        subset="IMAGEOBench",
        split="",
        item_id="d2",
        image_path=dataset_root / "IMAGEOBench" / "IMAGEO-Bench-datasets" / "dataset2" / "d2.jpg",
    )
    selectors = []

    def shared(_root, selector):
        selectors.append(selector)
        return [d2]

    monkeypatch.setattr(_SCRIPT, "_shared_items", shared)
    args = argparse.Namespace(
        dataset="imageobench-dataset2",
        dataset_root=str(dataset_root),
        offset=0,
        limit=None,
    )

    items = _SCRIPT.discover_items(args)

    assert selectors == ["imageobench-dataset2"]
    assert [item.item_id for item in items] == ["d2"]


def test_default_output_directories_follow_results_layout():
    assert _SCRIPT.default_output_dir("geoexp7k-learning") == (
        _ROOT / "outputs" / "results" / "vllm_geoagent_geoexp7k_learning"
    )
    assert _SCRIPT.default_output_dir("imageobench-dataset2") == (
        _ROOT / "outputs" / "results" / "vllm_geoagent_imageobench_dataset2"
    )
    assert _SCRIPT.default_output_dir("im2gps3k") == (
        _ROOT / "outputs" / "results" / "vllm_geoagent_im2gps3k"
    )


def test_resume_drops_only_error_rows(tmp_path):
    path = tmp_path / "predictions.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "status"])
        writer.writeheader()
        writer.writerows(
            [
                {"item_id": "done", "status": "completed"},
                {"item_id": "retry", "status": "error"},
                {"item_id": "missing", "status": "missing_image"},
            ]
        )

    rows = _SCRIPT._load_resumable_rows(path)

    assert [row["item_id"] for row in rows] == ["done", "missing"]


def test_shared_opencage_cache_round_trip(tmp_path):
    cache = _SCRIPT.OpenCageCache(tmp_path / "opencage.sqlite3")
    row = {"status": "completed", "latitude": 48.8584, "longitude": 2.2945}

    cache.put("eiffel tower, france", row)

    assert cache.get("eiffel tower, france") == row
    assert cache.get("missing") is None


def test_cached_geocoding_row_does_not_reuse_stale_quota():
    source = {
        "item_id": "source",
        "status": "completed",
        "latitude": "48.8584",
        "longitude": "2.2945",
        "rate_limit": "2500",
        "rate_remaining": "10",
        "rate_reset": "tomorrow",
    }

    row = _SCRIPT.cached_geocoding_row(
        {"dataset": "test", "item_id": "target", "final_answer": "Eiffel Tower"},
        "Eiffel Tower, France",
        source,
    )

    assert row["status"] == "cached"
    assert row["latitude"] == "48.8584"
    assert row["rate_remaining"] == ""
    assert row["http_attempts"] == 0


def test_coordinate_evaluation_includes_public_benchmark_thresholds():
    row = {
        "gt_latitude": "0",
        "gt_longitude": "0",
        "pred_latitude": "0",
        "pred_longitude": "0",
    }

    _SCRIPT.evaluate_coordinates(row)

    assert row["correct_10m"] == "1"
    assert row["correct_25000m"] == "1"
    assert row["correct_2500000m"] == "1"
