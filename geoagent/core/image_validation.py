"""Validation for image inputs at public workflow boundaries."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from geoagent.core.exceptions import ImageInputError


def validate_image_input(image_path: str | Path) -> Path:
    """Return a validated path or raise a user-facing input error."""

    raw_path = str(image_path or "").strip()
    if not raw_path:
        raise ImageInputError("Input image path is empty. Please provide a valid readable image file.")

    path = Path(raw_path).expanduser()
    if not path.exists():
        raise ImageInputError(
            f"Input image does not exist: '{raw_path}'. Please provide the correct image path."
        )
    if not path.is_file():
        raise ImageInputError(
            f"Input image path is not a file: '{raw_path}'. Please provide a valid readable image file."
        )

    try:
        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError("image dimensions must be positive")
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ImageInputError(
            f"Input image cannot be decoded: '{raw_path}'. "
            "Please provide a valid, readable image file."
        ) from exc
    return path
