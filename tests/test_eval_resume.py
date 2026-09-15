"""Tests for run_dataset_eval resume behavior: errored items must be re-run."""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("run_dataset_eval", _ROOT / "scripts" / "run_dataset_eval.py")
_EVAL = importlib.util.module_from_spec(_SPEC)
sys.modules["run_dataset_eval"] = _EVAL
_SPEC.loader.exec_module(_EVAL)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_load_existing_rows_drops_error_rows(tmp_path):
    """Rate-limit / parse failures wrote status=error rows; on resume those
    items must count as unfinished so they are re-run."""
    out = tmp_path / "eval.csv"
    _write_csv(
        out,
        [
            {"dataset": "D", "subset": "S", "item_id": "a", "status": "completed", "error": ""},
            {"dataset": "D", "subset": "S", "item_id": "b", "status": "error", "error": "HTTP 429: TPM limit reached"},
            {"dataset": "D", "subset": "S", "item_id": "c", "status": "completed", "error": ""},
        ],
    )

    kept = _EVAL.load_existing_rows(out)

    assert [row["item_id"] for row in kept] == ["a", "c"]
    keys = _EVAL.existing_keys(out)
    assert keys == {("D", "S", "a"), ("D", "S", "c")}


def test_load_existing_rows_keeps_missing_and_skipped(tmp_path):
    """missing_image and skipped are terminal outcomes - only error re-runs."""
    out = tmp_path / "eval.csv"
    _write_csv(
        out,
        [
            {"dataset": "D", "subset": "S", "item_id": "a", "status": "missing_image", "error": ""},
            {"dataset": "D", "subset": "S", "item_id": "b", "status": "skipped", "error": ""},
        ],
    )

    keys = _EVAL.existing_keys(out)

    assert keys == {("D", "S", "a"), ("D", "S", "b")}


def test_load_existing_rows_missing_file(tmp_path):
    assert _EVAL.load_existing_rows(tmp_path / "nope.csv") == []
    assert _EVAL.existing_keys(tmp_path / "nope.csv") == set()


def test_dataset_registry_excludes_unsupported_datasets(tmp_path):
    assert set(_EVAL.STANDARD_DATASET_MANIFESTS) == {
        "geoexp7k-experience-effect-206",
        "geoexp7k-learning",
        "geoexp7k-test",
        "im2gps3k",
        "imageobench-dataset2",
    }

    for unsupported in ("mapbench", "geoexp7k-train"):
        args = argparse.Namespace(
            datasets=unsupported,
            dataset_root=str(tmp_path),
            offset=0,
            limit=None,
        )
        with pytest.raises(ValueError, match="Unsupported dataset"):
            _EVAL.discover_items(args)


def test_imageobench_loader_keeps_only_dataset2(tmp_path):
    metadata_dir = tmp_path / "IMAGEOBench"
    metadata_dir.mkdir()
    _write_csv(
        metadata_dir / "metadata_standard.csv",
        [
            {"image_path": "dataset1/one.jpg", "dataset": "IMAGEOBench-d1"},
            {"image_path": "dataset2/two.jpg", "dataset": "IMAGEOBench-d2"},
            {"image_path": "dataset3/three.jpg", "dataset": "IMAGEOBench-d3"},
        ],
    )
    args = argparse.Namespace(
        datasets="imageobench-dataset2",
        dataset_root=str(tmp_path),
        offset=0,
        limit=None,
    )

    items = _EVAL.discover_items(args)

    assert [item.item_id for item in items] == ["000001_two"]
    assert items[0].image_path == (
        metadata_dir / "IMAGEO-Bench-datasets" / "dataset2" / "two.jpg"
    )


def test_evaluation_schema_and_summary_are_coordinate_only(tmp_path):
    item = _EVAL.EvalItem(
        dataset="D",
        subset="S",
        split="test",
        item_id="a",
        image_path=tmp_path / "a.jpg",
        gt_latitude=0.0,
        gt_longitude=0.0,
    )
    row = _EVAL.base_row(item)
    row["status"] = "completed"
    row["pred_granularity"] = "coordinates"
    row["pred_latitude"] = "0.0005"
    row["pred_longitude"] = "0"

    _EVAL.evaluate_correctness(row, item)

    assert float(row["pred_gt_distance_m"]) > 50
    assert row["correct_50m"] == "0"
    assert row["correct_100m"] == "1"
    assert "correct_country" not in _EVAL.FIELDNAMES
    assert "match_notes" not in _EVAL.FIELDNAMES

    output = tmp_path / "eval.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_EVAL.FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    _EVAL.write_summary(output)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["overall"]) == {"total", "coordinates", "granularity_calibration"}
