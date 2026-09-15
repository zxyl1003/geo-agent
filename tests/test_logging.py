import shutil
from pathlib import Path

from geoagent.core.logging import (
    current_log_file_path,
    logger,
    log_model_stream_delta,
    log_model_stream_end,
    log_model_stream_start,
    log_tool_result,
    log_workflow,
    setup_logging,
)


def test_setup_logging_writes_daily_file():
    log_root = Path("logs") / "_test_logging"
    if log_root.exists():
        shutil.rmtree(log_root)

    try:
        setup_logging("INFO", debug=True, colorize=False, file_root=log_root, file_date="2026-05-20")

        log_workflow("daily file test", {"api_key": "secret-value", "status": "ok"})
        log_model_stream_start("LLM:brain", "test-model")
        log_model_stream_delta("streamed text")
        log_model_stream_end("LLM:brain")

        log_path = log_root / "2026-05-20" / "log.txt"
        assert current_log_file_path() == log_path
        assert log_path.exists()

        text = log_path.read_text(encoding="utf-8")
        assert "daily file test" in text
        assert "streamed text" in text
        assert "secret-value" not in text
        assert '"api_key": "***"' in text
    finally:
        logger.remove()
        if log_root.exists():
            shutil.rmtree(log_root)


def test_tool_result_logging_escapes_html_like_payload():
    log_root = Path("logs") / "_test_logging_html"
    if log_root.exists():
        shutil.rmtree(log_root)

    try:
        setup_logging("INFO", debug=True, colorize=True, file_root=log_root, file_date="2026-05-20")

        log_tool_result(
            "streetview_verify",
            False,
            {"error_body": "<html><body>upstream error</body></html>"},
            "HTTP response was HTML",
        )

        log_path = log_root / "2026-05-20" / "log.txt"
        text = log_path.read_text(encoding="utf-8")
        assert "<html><body>upstream error</body></html>" in text
    finally:
        logger.remove()
        if log_root.exists():
            shutil.rmtree(log_root)
