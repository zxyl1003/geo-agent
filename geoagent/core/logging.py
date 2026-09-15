"""Centralized loguru configuration and colored debug trace helpers."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import SecretStr

from geoagent.core.text_utils import repair_latin1_mojibake


_DEBUG_LOGGING = False
_STREAM_OPEN = False
_COLORIZE = True
_LOG_FILE_PATH: Path | None = None

_ANSI = {
    "reset": "\033[0m",
    "agent": "\033[96m",
    "tool": "\033[93m",
    "model": "\033[95m",
    "workflow": "\033[94m",
    "success": "\033[92m",
    "error": "\033[91m",
    "muted": "\033[90m",
}


def setup_logging(
    level: str = "INFO",
    fmt: str | None = None,
    debug: bool | None = None,
    colorize: bool = True,
    file_enabled: bool = True,
    file_root: str | Path = "logs",
    file_name: str = "log.txt",
    file_date: str | None = None,
) -> None:
    """Configure loguru without exposing sensitive configuration values."""

    global _DEBUG_LOGGING, _COLORIZE, _LOG_FILE_PATH
    configure_console_encoding()
    effective_level = "DEBUG" if debug else level.upper()
    _DEBUG_LOGGING = bool(debug) or effective_level == "DEBUG"
    _COLORIZE = colorize
    log_format = fmt or "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"

    logger.remove()
    logger.add(
        sys.stderr,
        level=effective_level,
        colorize=colorize,
        format=log_format,
    )
    _LOG_FILE_PATH = None
    if file_enabled:
        date_part = file_date or datetime.now().strftime("%Y-%m-%d")
        file_path = Path(file_root) / date_part / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE_PATH = file_path
        logger.add(
            file_path,
            level=effective_level,
            colorize=False,
            encoding="utf-8",
            format=fmt or "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )


def debug_enabled() -> bool:
    return _DEBUG_LOGGING


def current_log_file_path() -> Path | None:
    return _LOG_FILE_PATH


def configure_console_encoding() -> None:
    """Use UTF-8 tolerant console streams on Windows and other narrow terminals."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def safe_log_config(config: Any) -> dict[str, Any]:
    """Return a redacted summary suitable for debug logs."""

    return {
        "project_name": getattr(config, "project_name", "geo_agent_system"),
        "log_level": getattr(config, "log_level", "INFO"),
        "tools": sorted(getattr(config, "tools", {}).keys()),
        "models": sorted(getattr(config, "models", {}).keys()),
        "workflows": sorted(getattr(config, "workflows", {}).keys()),
    }


def log_workflow(message: str, payload: Any | None = None) -> None:
    _debug("workflow", "WORKFLOW", message, payload)


def log_agent_start(agent_name: str, step: int | None = None) -> None:
    suffix = f" step={step}" if step is not None else ""
    _debug("agent", f"AGENT:{agent_name}", f"start{suffix}")


def log_agent_output(agent_name: str, message: str, payload: Any | None = None) -> None:
    _debug("agent", f"AGENT:{agent_name}", message, payload)


def log_tool_request(tool_name: str, arguments: dict[str, Any], reason: str | None = None) -> None:
    payload = {"reason": reason, "arguments": arguments}
    _debug("tool", f"TOOL:{tool_name}", "request", payload)


def log_tool_result(tool_name: str, success: bool, payload: Any | None = None, error: str | None = None) -> None:
    status = "success" if success else "error"
    color_key = "success" if success else "error"
    _debug(color_key, f"TOOL:{tool_name}", status, {"error": error, "result": payload})


def log_model_request(kind: str, model: str, prompt_preview: str | None = None) -> None:
    payload = {"model": model}
    if prompt_preview:
        payload["prompt_preview"] = _truncate(prompt_preview, 500)
    _debug("model", f"MODEL:{kind}", "request", payload)


def log_model_response(kind: str, model: str, text: str, metadata: dict[str, Any] | None = None) -> None:
    payload = {"model": model, "text": text}
    if metadata:
        payload["metadata"] = metadata
    _debug("model", f"MODEL:{kind}", "response", payload)


def log_model_stream_start(kind: str, model: str) -> None:
    if not _DEBUG_LOGGING:
        return
    global _STREAM_OPEN
    _STREAM_OPEN = True
    prefix = _ANSI["model"] if _COLORIZE else ""
    message = f"[MODEL:{kind}] streaming {model}\n"
    sys.stderr.write(f"{prefix}{message}")
    sys.stderr.flush()
    _append_stream_file(f"{_timestamp()} | DEBUG | {message}")


def log_model_stream_delta(text: str) -> None:
    if not _DEBUG_LOGGING or not text:
        return
    repaired = repair_latin1_mojibake(text)
    sys.stderr.write(repaired)
    sys.stderr.flush()
    _append_stream_file(repaired)


def log_model_stream_end(kind: str) -> None:
    if not _DEBUG_LOGGING:
        return
    global _STREAM_OPEN
    if _STREAM_OPEN:
        reset = _ANSI["reset"] if _COLORIZE else ""
        muted = _ANSI["muted"] if _COLORIZE else ""
        message = f"[MODEL:{kind}] stream end"
        sys.stderr.write(f"{reset}\n{muted}{message}{reset}\n")
        sys.stderr.flush()
        _append_stream_file(f"\n{_timestamp()} | DEBUG | {message}\n")
        _STREAM_OPEN = False


def log_model_stream_fallback(kind: str, error: str) -> None:
    _debug("error", f"MODEL:{kind}", "stream unavailable; retrying non-stream", {"error": error})


def log_model_retry(kind: str, attempt: int, error: Exception) -> None:
    _debug("error", f"MODEL:{kind}", f"request failed; retry #{attempt}", {"error": str(error)})


def log_skip(component: str, name: str, reason: str) -> None:
    _debug("muted", f"{component}:{name}", "skipped", {"reason": reason})


def _debug(color_key: str, label: str, message: str, payload: Any | None = None) -> None:
    if not _DEBUG_LOGGING:
        return
    safe_payload = _redact(payload)
    color = _loguru_color(color_key)
    if safe_payload is None:
        logger.opt(colors=True).debug(f"<{color}>[{{}}]</{color}> {{}}", label, message)
        return
    payload_text = json.dumps(safe_payload, ensure_ascii=False, default=str)
    logger.opt(colors=True).debug(
        f"<{color}>[{{}}]</{color}> {{}} | {{}}",
        label,
        message,
        payload_text,
    )


def _append_stream_file(text: str) -> None:
    if _LOG_FILE_PATH is None:
        return
    try:
        with _LOG_FILE_PATH.open("a", encoding="utf-8", errors="replace") as file:
            file.write(text)
    except Exception:
        pass


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _redact(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return "***"
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lower = str(key).lower()
            if any(token in lower for token in ("api_key", "apikey", "authorization", "secret", "password")):
                redacted[key] = "***"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        max_items = 8
        items = [_redact(item) for item in value[:max_items]]
        if len(value) > max_items:
            items.append(f"... {len(value) - max_items} more")
        return items
    if hasattr(value, "model_dump"):
        return _redact(value.model_dump())
    if isinstance(value, str):
        return _truncate(value, 2000)
    return value


def _truncate(text: str, limit: int) -> str:
    compact = text.strip()
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _loguru_color(color_key: str) -> str:
    return {
        "agent": "cyan",
        "tool": "yellow",
        "model": "magenta",
        "workflow": "blue",
        "success": "green",
        "error": "red",
        "muted": "black",
    }.get(color_key, "white")


__all__ = [
    "debug_enabled",
    "configure_console_encoding",
    "current_log_file_path",
    "logger",
    "log_agent_output",
    "log_agent_start",
    "log_model_request",
    "log_model_response",
    "log_model_stream_delta",
    "log_model_stream_end",
    "log_model_stream_fallback",
    "log_model_stream_start",
    "log_model_retry",
    "log_skip",
    "log_tool_request",
    "log_tool_result",
    "log_workflow",
    "safe_log_config",
    "setup_logging",
]
