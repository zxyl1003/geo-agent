"""Tests for the three-run offline experience-learning entry point."""

import argparse
import importlib.util
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_offline_experience_learning",
    _ROOT / "scripts" / "run_offline_experience_learning.py",
)
_LEARNING = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEARNING
_SPEC.loader.exec_module(_LEARNING)


def _args(tmp_path: Path, *, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_root=str(tmp_path / "datasets"),
        output_dir=str(tmp_path / "learning"),
        experience_dir=str(tmp_path / "experience"),
        reviews_output="",
        query="Where is this image?",
        workers=4,
        offset=0,
        limit=None,
        resume=True,
        debug=False,
        no_color=True,
        write_missing=False,
        fail_fast=False,
        dry_run=dry_run,
        selection_radius_m=25_000.0,
        max_context_chars=48_000,
    )


def test_learning_entry_point_runs_three_independent_learning_passes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(_LEARNING.dataset_eval, "run_eval", calls.append)

    run_dirs = _LEARNING.run_learning_trajectories(_args(tmp_path))

    assert [path.name for path in run_dirs] == ["run_1", "run_2", "run_3"]
    assert len(calls) == 3
    assert {call.datasets for call in calls} == {"geoexp7k-learning"}
    assert {call.memory_mode for call in calls} == {"off"}
    assert [Path(call.trace_dir).parent.name for call in calls] == [
        "run_1",
        "run_2",
        "run_3",
    ]


def test_dry_run_validates_only_one_pass(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(_LEARNING.dataset_eval, "run_eval", calls.append)

    _LEARNING.run_learning_trajectories(_args(tmp_path, dry_run=True))

    assert len(calls) == 1
    assert calls[0].dry_run is True


def test_new_learning_run_rejects_nonempty_output_directory(tmp_path):
    args = _args(tmp_path)
    args.resume = False
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "existing.txt").write_text("existing", encoding="utf-8")

    try:
        _LEARNING.run_pipeline(args)
    except FileExistsError as exc:
        assert "Use --resume" in str(exc)
    else:
        raise AssertionError("Expected a nonempty output directory to be rejected")


def test_repository_exposes_only_three_scripts():
    assert {path.name for path in (_ROOT / "scripts").glob("*.py")} == {
        "run_dataset_eval.py",
        "run_offline_experience_learning.py",
        "run_single_image.py",
    }
