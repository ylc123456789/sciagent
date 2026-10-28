"""One-shot JSON service requests sharing time and cancellation, not LLM usage."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import httpx

from ._trace import write_service_trace_record
from .budget import DeadlineExceededError, current_budget, remaining_timeout
from .cancellation import OperationCancelled, check_cancelled
from .http import ResponseTooLargeError, send_request


def _retry_after(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value.isascii() and value.isdigit() and len(value) <= 10:
        return value
    try:
        datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT")
        return value
    except (TypeError, ValueError, OverflowError):
        return None


def _redact(value, api_key: str):
    if isinstance(value, str):
        for secret in (api_key, json.dumps(api_key, ensure_ascii=False)[1:-1]):
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {_redact(key, api_key): _redact(item, api_key) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, api_key) for item in value]
    return value


class ServiceRequestError(RuntimeError):
    """A safe external-service failure without provider text or credentials."""

    def __init__(self, message: str, *, error_type: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class ServiceRequestHTTPError(ServiceRequestError):
    """HTTP status and sanitized Retry-After, without a raw HTTP response."""

    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(f"service request returned HTTP {status_code}", error_type="http_error")
        self.status_code = status_code
        self.retry_after = retry_after


class ServiceRequestClient:
    """POST caller-owned JSON once; preserve the enclosing Run time and cancellation.

    Service calls do not consume the model wallet. Their optional diagnostics are
    separate from LLM accounting and are not a durable spending ledger or quota.
    The caller validates its provider's business response.
    """

    def __init__(self, *, endpoint: str, api_key_env: str, timeout_seconds: float = 60,
                 max_response_bytes: int = 2_000_000, trace_dir: Path | None = None,
                 trace_level: str = "off") -> None:
        url = httpx.URL(endpoint)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
            raise ValueError("service endpoint must be an HTTP(S) URL without credentials")
        if (not isinstance(api_key_env, str) or not api_key_env.strip()
                or "=" in api_key_env or "\x00" in api_key_env):
            raise ValueError("api_key_env must be a nonempty environment variable name")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if type(max_response_bytes) is not int or max_response_bytes < 1:
            raise ValueError("max_response_bytes must be a positive integer")
        if trace_level not in {"off", "metadata", "full"}:
            raise ValueError("trace_level must be off, metadata or full")
        if trace_level != "off" and trace_dir is None:
            raise ValueError("trace_dir is required when trace_level is enabled")
        self.endpoint, self.api_key_env = str(url), api_key_env
        self.timeout_seconds, self.max_response_bytes = timeout_seconds, max_response_bytes
        self.trace_dir = Path(trace_dir).expanduser().resolve() if trace_dir else None
        self.trace_level = trace_level
        self.last_attempts = 0
        self._trace_context: dict = {}
        self._trace_seq = 0

    def set_trace_context(self, **kwargs) -> None:
        """Attach Run/Session/tool correlation to subsequent service diagnostics."""
        self._trace_context = dict(kwargs)

    def request(self, body: dict) -> dict:
        """Return one JSON object, without retrying or charging model usage."""
        self.last_attempts = 0
        check_cancelled()
        if not isinstance(body, dict):
            raise TypeError("service request body must be an object")
        request_text = json.dumps(body, ensure_ascii=False, allow_nan=False)
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ServiceRequestError("missing service API key", error_type="missing_credentials")
        timeout = remaining_timeout(self.timeout_seconds)
        request = httpx.Request("POST", self.endpoint, content=request_text.encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        call_id, created_at = uuid.uuid4().hex, datetime.now(UTC).isoformat()
        started = time.monotonic()
        response_bytes = raw_response_text = None
        error_type = validation_error = status_code = retry_after = None
        outcome = "unknown"
        self.last_attempts = 1
        try:
            response = send_request(request, timeout=timeout, max_response_bytes=self.max_response_bytes)
            status_code = response.status_code
            outcome = "failed"
            response_bytes = response.read()
            raw_response_text = response_bytes.decode("utf-8", errors="replace")
            try:
                payload = json.loads(response_bytes)
                if not isinstance(payload, dict):
                    raise TypeError("response must be an object")
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                raise ServiceRequestError("service response must be a valid JSON object",
                    error_type="invalid_response") from None
            outcome = "succeeded"
            return payload
        except OperationCancelled:
            error_type, validation_error = "interrupted", "local service request interrupted"
            raise
        except httpx.HTTPStatusError as error:
            outcome, status_code = "failed", error.response.status_code
            retry_after = _retry_after(error.response.headers.get("Retry-After"))
            safe_error = ServiceRequestHTTPError(status_code, retry_after)
            error_type, validation_error = safe_error.error_type, str(safe_error)
            raise safe_error from None
        except ResponseTooLargeError:
            outcome, error_type = "failed", "response_too_large"
            validation_error = "service response exceeded the byte limit"
            raise ServiceRequestError(validation_error, error_type=error_type) from None
        except DeadlineExceededError:
            error_type, validation_error = "deadline_exceeded", "service request execution limit exceeded"
            raise
        except (httpx.TimeoutException, TimeoutError):
            error_type, validation_error = "timeout", "service request timed out"
            parent = current_budget()
            if parent is not None:
                try:
                    parent.remaining_timeout()
                except DeadlineExceededError as error:
                    error_type, validation_error = "deadline_exceeded", "service request execution limit exceeded"
                    raise error from None
            raise ServiceRequestError(validation_error, error_type=error_type) from None
        except httpx.DecodingError:
            outcome, error_type = "failed", "invalid_response"
            validation_error = "service response body could not be decoded"
            raise ServiceRequestError(validation_error, error_type=error_type) from None
        except httpx.TransportError:
            error_type, validation_error = "network_error", "service request transport failed"
            raise ServiceRequestError(validation_error, error_type=error_type) from None
        except ServiceRequestError as error:
            error_type, validation_error = error.error_type, str(error)
            raise
        finally:
            if self.trace_level != "off" and self.trace_dir is not None:
                self._trace_seq += 1
                record = {**self._trace_context, "sequence": self._trace_seq,
                    "call_id": call_id, "created_at": created_at,
                    "request_kind": "external_service", "protocol": "json-service-request/v1",
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "retry_number": 0, "outcome": outcome, "response_valid": outcome == "succeeded",
                    "validation_error": validation_error, "error_type": error_type,
                    "http_status": status_code, "retry_after": retry_after}
                if self.trace_level == "full":
                    record.update(request_text=request_text, raw_response_text=raw_response_text)
                else:
                    record.update(request_sha256=hashlib.sha256(request_text.encode("utf-8")).hexdigest(),
                        response_sha256=(hashlib.sha256(response_bytes).hexdigest()
                                         if response_bytes is not None else None))
                write_service_trace_record(self.trace_dir, _redact(record, api_key))
