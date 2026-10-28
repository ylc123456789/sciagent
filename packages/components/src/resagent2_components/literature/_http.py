"""Small, process-local HTTP policy shared by the literature backends."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import Lock
import httpx
from resagent2_runtime.budget import DeadlineExceededError, current_budget, remaining_timeout
from resagent2_runtime.cancellation import check_cancelled, current_cancellation
from resagent2_runtime.http import ResponseTooLargeError


USER_AGENT = "ResAgent2/0.1 (+https://github.com/ylc123456789/sciagent)"
# An engineering protection for metadata, independent of PDF/body policies.
METADATA_RESPONSE_MAX_BYTES = 4 * 1024 * 1024
COOLDOWN_SECONDS = 60.0
_QUOTA_HEADERS = (
    "X-RateLimit-Limit", "X-RateLimit-Remaining",
    "X-RateLimit-Credits-Used", "X-RateLimit-Reset",
)
_LOGGER = logging.getLogger(__name__)


class LiteratureSearchError(RuntimeError):
    """A search did not produce a valid, normalized result."""

    def __init__(self, message, *, error_type='invalid_response', source_attempts=None,
                 retry_after=None):
        super().__init__(message)
        self.error_type = error_type
        self.source_attempts = list(source_attempts or [])
        self.retry_after = retry_after


class LiteratureUnavailableError(LiteratureSearchError):
    """This source cannot serve the request; another source may be tried."""

    def __init__(self, message, *, error_type='unavailable', source_attempts=None,
                 retry_after=None):
        super().__init__(message, error_type=error_type, source_attempts=source_attempts,
                         retry_after=retry_after)


def _retry_after(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return float(max(0, int(value)))
    except ValueError:
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            return max(0.0, (deadline - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0.0


def _response_diagnostics(
    source: str, response: httpx.Response, *, attempt: int,
    max_attempts: int, retry_after: float, cooldown_seconds: float,
) -> dict:
    """Keep only status, timing and numeric quota facts, never response text."""
    quota = {}
    for name in _QUOTA_HEADERS:
        raw = response.headers.get(name)
        if raw is None or len(raw) > 64:
            continue
        try:
            number = float(raw)
        except ValueError:
            continue
        if math.isfinite(number) and number >= 0:
            quota[name] = number
    return {
        "source": source,
        "status": response.status_code,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "authentication_configured": bool(response.request.headers.get("Authorization")),
        "retry_after_present": "Retry-After" in response.headers,
        "retry_after_seconds": retry_after,
        "cooldown_seconds": cooldown_seconds,
        "quota": quota,
    }


def _log_response(source: str, response: httpx.Response, **timing) -> None:
    level = logging.INFO if response.is_success else logging.WARNING
    _LOGGER.log(
        level, "Literature HTTP response %s",
        json.dumps(_response_diagnostics(source, response, **timing), sort_keys=True),
    )


class LiteratureHTTP:
    """Serialize requests to one source, including across backend instances.

    Each provider owns one module-level instance. This is not a cross-process
    limiter: deployments must not run parallel clients against the same quota.
    Cooldown fails promptly instead of sleeping for minutes inside an Agent.
    """

    def __init__(self, source: str, *, interval_seconds: float) -> None:
        self.source = source
        self.interval_seconds = interval_seconds
        self._lock = Lock()
        self._next_request_at = 0.0
        self._cooldown_until = 0.0

    def fetch(
        self, request: Callable[[], httpx.Response], *, max_attempts: int,
        quota_cooldown: Callable[[httpx.Response], float] | None = None,
    ) -> bytes:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        budget = current_budget()
        cancellation = current_cancellation()
        check_cancelled()
        if cancellation is None:
            if not self._lock.acquire(timeout=budget.remaining_timeout() if budget else -1):
                raise DeadlineExceededError("execution deadline exceeded waiting for literature access")
        else:
            # The shared source lock is a local wait, so it must be interruptible
            # before an HTTP request starts, including without a Run deadline.
            while True:
                check_cancelled()
                timeout = min(0.05, budget.remaining_timeout()) if budget else 0.05
                if self._lock.acquire(timeout=timeout):
                    break
        try:
            remaining = self._cooldown_until - time.monotonic()
            if remaining > 0:
                raise LiteratureUnavailableError(
                    f"{self.source} cooling down; retry in {remaining:.0f}s",
                    error_type="cooldown", retry_after=remaining
                )
            reason = ""
            error_type = "unavailable"
            for attempt in range(max_attempts):
                wait = self._next_request_at - time.monotonic()
                if wait > 0:
                    if cancellation is not None:
                        cancellation.wait(remaining_timeout(wait))
                    else:
                        time.sleep(remaining_timeout(wait))
                check_cancelled()
                if current_budget() is not None:
                    current_budget().remaining_timeout()
                retry_after = 0.0
                try:
                    response = request()
                    _log_response(
                        self.source, response, attempt=attempt + 1,
                        max_attempts=max_attempts,
                        retry_after=_retry_after(response.headers.get("Retry-After")),
                        cooldown_seconds=0.0,
                    )
                    return response.content
                except httpx.HTTPStatusError as error:
                    status = error.response.status_code
                    retry_after = _retry_after(
                        error.response.headers.get("Retry-After")
                    )
                    cooldown_seconds = 0.0
                    transient = status == 408 or 500 <= status <= 599
                    error_type = "timeout" if status == 408 else "service_error"
                    if status == 429 or (transient and retry_after > 0):
                        cooldown_seconds = max(COOLDOWN_SECONDS, retry_after)
                    elif transient and attempt == max_attempts - 1:
                        cooldown_seconds = COOLDOWN_SECONDS
                    if status == 429 and quota_cooldown is not None:
                        cooldown_seconds = max(
                            cooldown_seconds, quota_cooldown(error.response),
                        )
                    _log_response(
                        self.source, error.response, attempt=attempt + 1,
                        max_attempts=max_attempts, retry_after=retry_after,
                        cooldown_seconds=cooldown_seconds,
                    )
                    # Do not include response bodies, query URLs or auth in errors.
                    reason = f"HTTP {status}"
                    if status == 406:
                        # Do not repeat a rejected request against this source;
                        # a peer source may still serve the same query.
                        raise LiteratureUnavailableError(
                            f"{self.source} HTTP 406; source cannot serve this request",
                            error_type="request_rejected"
                        ) from None
                    if status == 429:
                        self._cooldown_until = time.monotonic() + cooldown_seconds
                        raise LiteratureUnavailableError(
                            f"{self.source} HTTP 429; rate limited, cooldown active",
                            error_type="rate_limited", retry_after=cooldown_seconds
                        ) from None
                    if status != 408 and not 500 <= status <= 599:
                        raise LiteratureSearchError(
                            f"{self.source} HTTP {status}; request rejected",
                            error_type="request_rejected"
                        ) from None
                    if retry_after > 0:
                        self._cooldown_until = time.monotonic() + max(
                            COOLDOWN_SECONDS, retry_after
                        )
                        raise LiteratureUnavailableError(
                            f"{self.source} {reason}; Retry-After cooldown active",
                            error_type=error_type, retry_after=max(COOLDOWN_SECONDS, retry_after)
                        ) from None
                except ResponseTooLargeError:
                    raise LiteratureSearchError(
                        f"{self.source} response exceeded the metadata byte limit",
                        error_type="response_too_large",
                    ) from None
                except DeadlineExceededError:
                    raise
                except httpx.DecodingError:
                    raise LiteratureSearchError(
                        f"{self.source} response body could not be decoded",
                        error_type="invalid_response",
                    ) from None
                except (httpx.TransportError, TimeoutError, ConnectionError) as error:
                    reason = type(error).__name__
                    error_type = ("timeout" if isinstance(error, (httpx.TimeoutException, TimeoutError))
                                  else "network_error")
                finally:
                    self._next_request_at = time.monotonic() + self.interval_seconds
                if attempt < max_attempts - 1:
                    self._next_request_at = time.monotonic() + max(
                        self.interval_seconds, 3.0 * 2**attempt
                    )
            self._cooldown_until = time.monotonic() + COOLDOWN_SECONDS
            raise LiteratureUnavailableError(
                f"{self.source} request failed after {max_attempts} attempts: {reason}",
                error_type=error_type, retry_after=COOLDOWN_SECONDS
            )
        finally:
            self._lock.release()
