from pathlib import Path

import pytest
from PIL import Image

from geoagent.core.config import load_app_config
from geoagent.core.exceptions import ImageInputError
from geoagent.core.image_validation import validate_image_input
from geoagent.workflows.react_workflow import ReactWorkflow
from scripts import run_single_image


def test_validate_image_input_accepts_decodable_image(tmp_path):
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (8, 6), color="red").save(image_path)

    assert validate_image_input(image_path) == image_path


@pytest.mark.parametrize("value", ["", "   "])
def test_validate_image_input_rejects_empty_path(value):
    with pytest.raises(ImageInputError, match="path is empty"):
        validate_image_input(value)


def test_validate_image_input_rejects_missing_path(tmp_path):
    missing = tmp_path / "missing.jpg"

    with pytest.raises(ImageInputError, match="does not exist"):
        validate_image_input(missing)


def test_validate_image_input_rejects_directory(tmp_path):
    with pytest.raises(ImageInputError, match="is not a file"):
        validate_image_input(tmp_path)


def test_validate_image_input_rejects_corrupt_image(tmp_path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_text("not an image", encoding="utf-8")

    with pytest.raises(ImageInputError, match="cannot be decoded"):
        validate_image_input(corrupt)


def test_workflow_rejects_missing_image_before_agents_are_created(tmp_path):
    config = load_app_config(config_dir="configs", env_file=None)
    workflow = ReactWorkflow(app_config=config, workflow_config=config.workflows["react"])

    with pytest.raises(ImageInputError, match="correct image path"):
        workflow.run(str(tmp_path / "missing.jpg"), "Where is this?")

    assert workflow.agents == {}


def test_single_image_cli_reports_invalid_path(monkeypatch, capsys, tmp_path):
    missing = tmp_path / "missing.jpg"
    monkeypatch.setattr(
        "sys.argv",
        ["run_single_image.py", "--image-path", str(missing)],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_single_image.main()

    assert exc_info.value.code == 2
    error_output = capsys.readouterr().err
    assert "does not exist" in error_output
    assert "correct image path" in error_output
