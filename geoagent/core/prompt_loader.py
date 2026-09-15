"""Prompt loading helpers."""

from __future__ import annotations

from pathlib import Path


def load_prompt(
    config_dir: str | Path,
    prompt_name: str,
    shared_names: tuple[str, ...] = (),
) -> str:
    """Load a prompt and optional shared prompt fragments from configs/prompts."""

    prompts_dir = Path(config_dir) / "prompts"
    sections: list[str] = []
    for name in shared_names:
        sections.append(_read_prompt(prompts_dir / "shared" / name))
    sections.append(_read_prompt(prompts_dir / prompt_name))
    return "\n\n".join(section.strip() for section in sections if section.strip())


def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    return path.read_text(encoding="utf-8")
