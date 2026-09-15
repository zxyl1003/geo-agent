"""Tests for the standalone Batch image-annotation script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_batch_image_annotation",
    _ROOT / "scripts" / "run_batch_image_annotation.py",
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def _valid_annotation() -> dict:
    return {
        "has_scene_text": True,
        "has_named_anchor": True,
        "anchor_types": ["poi_business"],
        "visible_text": ["Example Cafe"],
        "searchable_text": ["Example Cafe"],
        "text_legibility": "clear",
        "searchability": "high",
        "specificity": "unique",
        "language_or_script": ["Latin"],
        "observed_entities": [
            {
                "id": "ent_1",
                "entity_type": "poi_business",
                "name": "Example Cafe",
                "text_items": ["Example Cafe"],
                "phones": [],
                "panel_hint": "center",
                "confidence": 0.9,
            }
        ],
        "entity_count": 1,
        "multi_poi": False,
        "generic_chain": False,
        "text_scene_binding": "clear",
        "image_quality": "good",
        "quality_issues": [],
        "recommended_stratum": "specific_named_anchor",
        "filter_pass": True,
        "filter_reason": "A readable named storefront is visible.",
        "screening_confidence": 0.9,
    }


def test_annotation_schema_accepts_consistent_output_and_rejects_conflicts():
    output = _valid_annotation()
    assert _SCRIPT.validate_annotation(output) == []

    output["entity_count"] = 2
    output["recommended_stratum"] = "reject"
    errors = _SCRIPT.validate_annotation(output)
    assert "entity_count does not match observed_entities" in errors
    assert "recommended_stratum and filter_pass are inconsistent" in errors


def test_dataset_loaders_select_im2gps3k_and_only_imageo_dataset2(tmp_path):
    im2gps_dir = tmp_path / "im2gps3ktest"
    im2gps_dir.mkdir()
    (im2gps_dir / "im2gps3ktest.csv").write_text(
        "image_path,latitude,longitude\nphoto.jpg,1.5,2.5\n",
        encoding="utf-8",
    )

    imageo_dir = tmp_path / "IMAGEOBench"
    imageo_dir.mkdir()
    (imageo_dir / "metadata_standard.csv").write_text(
        "image_path,latitude,longitude,dataset\n"
        "dataset1/one.jpg,3,4,IMAGEOBench-d1\n"
        "dataset2/two.jpg,5,6,IMAGEOBench-d2\n",
        encoding="utf-8",
    )

    im2gps = list(_SCRIPT._iter_dataset(tmp_path, "im2gps3k"))
    imageo = list(_SCRIPT._iter_dataset(tmp_path, "imageobench-dataset2"))

    assert len(im2gps) == 1
    assert im2gps[0].image_rel == "im2gps3ktest/images/photo.jpg"
    assert len(imageo) == 1
    assert imageo[0].dataset == "IMAGEOBench-d2"
    assert imageo[0].image_rel == "IMAGEOBench/IMAGEO-Bench-datasets/dataset2/two.jpg"
