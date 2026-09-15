"""Tests for the GeoCLIP PyTorch dataset evaluator."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_geoclip_eval", _ROOT / "scripts" / "run_geoclip_eval.py"
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def _args(tmp_path: Path, dataset: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset,
        dataset_root=str(tmp_path / "datasets"),
        offset=0,
        limit=None,
    )


def test_dataset_discovery_uses_supported_canonical_names(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    imageo_base = dataset_root / "IMAGEOBench"
    d2 = _SCRIPT._eval.EvalItem(
        dataset="IMAGEOBench-d2",
        subset="IMAGEOBench",
        split="",
        item_id="d2",
        image_path=imageo_base / "IMAGEO-Bench-datasets" / "dataset2" / "d2.jpg",
    )
    im2gps = _SCRIPT._eval.EvalItem(
        dataset="im2gps3k",
        subset="im2gps3ktest",
        split="",
        item_id="im2gps",
        image_path=dataset_root / "im2gps3ktest" / "images" / "one.jpg",
    )
    learning = _SCRIPT._eval.EvalItem(
        dataset="GeoExp7k",
        subset="GeoExp7k",
        split="learning",
        item_id="learning",
        image_path=dataset_root / "GeoExp7k" / "learning.jpg",
    )

    def shared(_root, selector):
        return {
            "imageobench-dataset2": [d2],
            "im2gps3k": [im2gps],
            "geoexp7k-learning": [learning],
        }[selector]

    monkeypatch.setattr(_SCRIPT, "_shared_items", shared)

    assert _SCRIPT.discover_items(_args(tmp_path, "im2gps3k")) == [im2gps]
    assert _SCRIPT.discover_items(_args(tmp_path, "geoexp7k-learning")) == [learning]
    imageo_items = _SCRIPT.discover_items(_args(tmp_path, "imageobench-dataset2"))
    assert [item.item_id for item in imageo_items] == ["d2"]


def test_default_output_paths_follow_results_layout():
    assert _SCRIPT.default_output_path("geoexp7k-learning") == (
        _ROOT / "outputs" / "results" / "geoclip_geoexp7k-learning" / "results.csv"
    )
    assert _SCRIPT.default_output_path("imageobench-dataset2") == (
        _ROOT / "outputs" / "results" / "geoclip_imageobench_dataset2" / "results.csv"
    )


def test_resume_drops_only_error_rows(tmp_path):
    path = tmp_path / "results.csv"
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


def test_location_embedding_cache_is_reused(tmp_path):
    class LocationEncoder:
        def __init__(self):
            self.calls = 0

        def __call__(self, values):
            self.calls += 1
            return torch.column_stack((values, torch.ones(len(values), device=values.device)))

    class Model:
        def __init__(self):
            self.gps_gallery = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
            self.location_encoder = LocationEncoder()

    cache_path = tmp_path / "gallery.pt"
    first = Model()

    coordinates, embeddings = _SCRIPT.load_or_build_gallery_embeddings(
        torch=torch,
        model=first,
        version="test",
        cache_path=cache_path,
        batch_size=1,
        rebuild=False,
    )
    second = Model()
    cached_coordinates, cached = _SCRIPT.load_or_build_gallery_embeddings(
        torch=torch,
        model=second,
        version="test",
        cache_path=cache_path,
        batch_size=1,
        rebuild=False,
    )

    assert first.location_encoder.calls == 2
    assert second.location_encoder.calls == 0
    torch.testing.assert_close(cached_coordinates, coordinates)
    torch.testing.assert_close(cached, embeddings)
