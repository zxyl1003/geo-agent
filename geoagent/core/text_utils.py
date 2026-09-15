"""Text normalization helpers for API and terminal output."""

from __future__ import annotations

import re
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def repair_latin1_mojibake(text: str) -> str:
    """Repair text that was UTF-8 bytes decoded as Latin-1.

    This handles strings such as ``ãã¿`` and C1-control variants copied
    from terminal logs. It is conservative: if a round-trip cannot improve
    the text, the original string is returned.
    """

    if not _looks_like_latin1_mojibake(text):
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return _LATIN1_RUN_RE.sub(_repair_latin1_run, text)
    return repaired if _repair_score(repaired) >= _repair_score(text) else text


def repair_text_tree(value: Any) -> Any:
    if isinstance(value, str):
        return repair_latin1_mojibake(value)
    if isinstance(value, list):
        return [repair_text_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: repair_text_tree(item) for key, item in value.items()}
    return value


def _looks_like_latin1_mojibake(text: str) -> bool:
    if not text:
        return False
    suspicious = sum(1 for ch in text if 0x80 <= ord(ch) <= 0xFF)
    return suspicious >= 2 and any(ch in text for ch in ("ã", "æ", "è", "é", "å", "\x81", "\x82", "\x83"))


_LATIN1_RUN_RE = re.compile(r"[\x80-\xff]{2,}")


def _repair_latin1_run(match: re.Match[str]) -> str:
    run = match.group(0)
    try:
        repaired = run.encode("latin1").decode("utf-8")
    except UnicodeError:
        return run
    return repaired if _repair_score(repaired) >= _repair_score(run) else run


def _repair_score(text: str) -> int:
    japanese = sum(1 for ch in text if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")
    controls = sum(1 for ch in text if 0x80 <= ord(ch) <= 0x9F)
    latin1_markers = sum(text.count(ch) for ch in ("ã", "æ", "è", "é", "å"))
    return japanese * 3 - controls * 3 - latin1_markers
