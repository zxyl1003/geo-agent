"""Tests for post_json_with_retry, especially TPM/RPM rate-limit handling."""

from unittest.mock import MagicMock

import pytest

from geoagent.core.http import (
    RATE_LIMIT_RETRY_DELAY_SECONDS,
    _backoff_delay,
    _is_rate_limited,
    post_json_with_retry,
)


def _response(status_code: int, text: str = ""):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.ok = status_code < 400
    response.content = text.encode("utf-8")
    return response


class _SleepRecorder:
    def __init__(self):
        self.sleeps: list[float] = []

    def __call__(self, seconds: float):
        self.sleeps.append(seconds)


def test_is_rate_limited_detection():
    assert _is_rate_limited(_response(429)) is True
    assert _is_rate_limited(_response(400, '{"code":50602,"message":"TPM limit reached"}')) is True
    assert _is_rate_limited(_response(400, "Request was rejected due to rate limiting")) is True
    assert _is_rate_limited(_response(400, "bad request")) is False
    assert _is_rate_limited(_response(200)) is False


def test_rate_limit_waits_full_window_then_succeeds(monkeypatch):
    """A 429 TPM error must wait the full rate-limit window (10 minutes), not
    the usual exponential backoff, and then retry."""
    sleep = _SleepRecorder()
    monkeypatch.setattr("geoagent.core.http.time.sleep", sleep)
    responses = [
        _response(429, '{"code":50602,"message":"Request was rejected due to rate limiting. Details: TPM limit reached."}'),
        _response(200, "{}"),
    ]
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"payload": json})
        return responses[len(calls) - 1]

    monkeypatch.setattr("geoagent.core.http.requests.post", fake_post)
    retries = []
    data = post_json_with_retry(
        "https://api.test/v1/chat/completions",
        {},
        {"model": "m", "messages": []},
        log_retry=lambda attempt, error: retries.append((attempt, str(error))),
    )

    assert data == {}
    assert len(calls) == 2
    assert sleep.sleeps == [RATE_LIMIT_RETRY_DELAY_SECONDS]
    assert "rate limited" in retries[0][1]
    assert "120s" in retries[0][1]


def test_rate_limit_does_not_drop_response_format(monkeypatch):
    """A 429 is a quota issue; the retry helper must not waste quota by
    resending without response_format first."""
    sleep = _SleepRecorder()
    monkeypatch.setattr("geoagent.core.http.time.sleep", sleep)
    responses = [
        _response(429, '{"code":50602,"message":"TPM limit reached"}'),
        _response(200, "{}"),
    ]
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return responses[len(calls) - 1]

    monkeypatch.setattr("geoagent.core.http.requests.post", fake_post)
    payload = {"model": "m", "messages": [], "response_format": {"type": "json_object"}}

    data = post_json_with_retry("https://api.test/v1/chat/completions", {}, payload)

    assert data == {}
    # Both requests kept response_format: no drop-resend happened.
    assert len(calls) == 2
    assert all(call.get("response_format") == {"type": "json_object"} for call in calls)


def test_other_client_errors_still_use_exponential_backoff(monkeypatch):
    sleep = _SleepRecorder()
    monkeypatch.setattr("geoagent.core.http.time.sleep", sleep)
    responses = [_response(500, "server error"), _response(200, "{}")]

    def fake_post(url, headers=None, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr("geoagent.core.http.requests.post", fake_post)
    data = post_json_with_retry("https://api.test/v1/chat/completions", {}, {"model": "m"})

    assert data == {}
    assert sleep.sleeps == [_backoff_delay(0, 1.0)]


def test_persistent_rate_limit_raises_after_retries(monkeypatch):
    sleep = _SleepRecorder()
    monkeypatch.setattr("geoagent.core.http.time.sleep", sleep)

    def fake_post(url, headers=None, json=None, timeout=None):
        return _response(429, '{"code":50602,"message":"TPM limit reached"}')

    monkeypatch.setattr("geoagent.core.http.requests.post", fake_post)
    with pytest.raises(RuntimeError, match="rate limited"):
        post_json_with_retry("https://api.test/v1/chat/completions", {}, {"model": "m"})
    # max_retries=3 -> 3 rate-limit waits, then give up.
    assert sleep.sleeps == [RATE_LIMIT_RETRY_DELAY_SECONDS] * 3
