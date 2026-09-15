"""Small HTTP helpers used by real API adapters."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import requests


DEFAULT_TIMEOUT = 45
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0
# TPM/RPM rate limits (e.g. SiliconFlow 50602 "TPM limit reached") recover on a
# minutes-scale window, so the usual exponential backoff is useless: wait long
# enough for the quota window to reset before retrying.
RATE_LIMIT_RETRY_DELAY_SECONDS = 120.0


def raise_for_api_error(response: requests.Response) -> None:
    """Raise a compact error without exposing request headers or credentials."""

    if response.ok:
        return
    body = response.text[:500]
    raise RuntimeError(f"HTTP {response.status_code}: {body}")


def request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    timeout = kwargs.pop("timeout", DEFAULT_TIMEOUT)
    response = requests.request(method=method, url=url, timeout=timeout, **kwargs)
    raise_for_api_error(response)
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Expected JSON object response.")
    return data


class _RetryableRequestError(RuntimeError):
    """Internal marker: the request should be retried."""


class _RateLimitRequestError(RuntimeError):
    """Internal marker: the request hit a TPM/RPM quota; wait out the window."""


def _is_rate_limited(response: requests.Response) -> bool:
    if response.status_code == 429:
        return True
    if response.status_code >= 400:
        text = (response.text or "")[:500].casefold()
        return any(marker in text for marker in ("rate limit", "ratelimit", "tpm", "rpm", "50602"))
    return False


def _backoff_delay(attempt: int, base_delay: float, max_delay: float = 8.0) -> float:
    return min(base_delay * (2**attempt), max_delay)


def post_json_with_retry(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    log_retry: Callable[[int, Exception], None] | None = None,
) -> dict[str, Any]:
    """POST JSON to an OpenAI-compatible endpoint, retrying transient failures.

    Retries on network errors, rate limits (429 and TPM/RPM bodies), 5xx server
    errors, and responses whose body is not valid JSON. 4xx client errors are
    deterministic (bad payload, auth, oversized context), so they fail
    immediately instead of burning retries and backoff on an identical request.
    ``json_mode`` retries are handled by dropping ``response_format``
    once inside an attempt, mirroring the previous client behaviour. Exponentially
    backs off between attempts and raises ``RuntimeError`` once all attempts fail.
    """

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if (
                response.status_code >= 400
                and response.status_code != 429
                and "response_format" in payload
            ):
                # Some OpenAI-compatible endpoints reject response_format; drop it and
                # resend within the same attempt before giving up on this request.
                # (A 429 is a quota issue, not a payload issue - resending would
                # only burn more quota; fall through to the rate-limit wait.)
                retry_payload = dict(payload)
                retry_payload.pop("response_format", None)
                response = requests.post(url, headers=headers, json=retry_payload, timeout=timeout)
            if not response.ok:
                if _is_rate_limited(response):
                    raise _RateLimitRequestError(
                        f"HTTP {response.status_code} rate limited; waiting "
                        f"{RATE_LIMIT_RETRY_DELAY_SECONDS:.0f}s before retry: {response.text[:300]}"
                    )
                if 400 <= response.status_code < 500:
                    # Client errors are deterministic (bad payload, auth, oversized
                    # context); resending the identical request cannot succeed, so
                    # fail immediately. _is_rate_limited above already captured the
                    # providers that report TPM/RPM quota errors as 400.
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                raise _RetryableRequestError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
            data = json.loads(response.content.decode("utf-8", errors="replace"))
            if not isinstance(data, dict):
                raise _RetryableRequestError("Response body is not a JSON object.")
            return data
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
        except _RateLimitRequestError as exc:
            last_error = exc
        except _RetryableRequestError as exc:
            last_error = exc
        if attempt < max_retries:
            if log_retry is not None:
                log_retry(attempt + 1, last_error)
            if isinstance(last_error, _RateLimitRequestError):
                time.sleep(RATE_LIMIT_RETRY_DELAY_SECONDS)
            else:
                time.sleep(_backoff_delay(attempt, base_delay))
    raise RuntimeError(
        f"Model API request failed after {max_retries + 1} attempts: {last_error}"
    )
