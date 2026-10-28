"""Run-owned cooperative cancellation shared by model and tool execution.

Cancellation is an execution fact, never an Agent-selected tool argument. The
Controller owns the token and binds it around a dispatch. Components stop their
managed operation and raise only after cleanup, so paused never means merely
that the caller stopped waiting.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event, Lock
from typing import Callable, Iterator


class OperationCancelled(RuntimeError):
    """An interrupted operation and whether its local resources stopped."""

    def __init__(
        self, reason: str = "user interrupted execution", *, stopped: bool = True,
        details: dict | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stopped = stopped
        self.details = dict(details or {})


class OperationYielded(RuntimeError):
    """A completed dispatch boundary yields to newly queued upstream input."""


class CancellationToken:
    """One-way, thread-safe cancellation for a single active execution scope."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._event = Event()
        self._cancelled = False
        self._reason = "user interrupted execution"
        self._callbacks: dict[object, Callable[[], None]] = {}

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "user interrupted execution") -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            self._reason = reason
            self._event.set()
            callbacks = list(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            callback()

    def raise_if_cancelled(self, *, stopped: bool = True, details: dict | None = None) -> None:
        with self._lock:
            cancelled, reason = self._cancelled, self._reason
        if cancelled:
            raise OperationCancelled(reason, stopped=stopped, details=details)

    def wait(self, timeout: float) -> None:
        """Wait for retry/rate-limit timing, returning early on interruption."""
        self._event.wait(timeout)
        self.raise_if_cancelled()

    def on_cancel(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register cleanup, including the cancel-before-register race.

        Callbacks must be short and nonblocking: they request cancellation;
        the executing component waits for and verifies actual cessation.
        """
        key = object()
        with self._lock:
            if self._cancelled:
                invoke = True
            else:
                invoke = False
                self._callbacks[key] = callback
        if invoke:
            callback()

        def unregister() -> None:
            with self._lock:
                self._callbacks.pop(key, None)

        return unregister


_CURRENT: ContextVar[CancellationToken | None] = ContextVar("execution_cancellation", default=None)


@contextmanager
def bind_cancellation(token: CancellationToken | None) -> Iterator[CancellationToken | None]:
    """Bind the Run-owned token without changing model-facing schemas."""
    binding = _CURRENT.set(token)
    try:
        yield token
    finally:
        _CURRENT.reset(binding)


def current_cancellation() -> CancellationToken | None:
    return _CURRENT.get()


def check_cancelled() -> None:
    token = current_cancellation()
    if token is not None:
        token.raise_if_cancelled()
