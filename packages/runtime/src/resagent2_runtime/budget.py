"""Shared request accounting and deadlines for one synchronous execution scope."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import time
from typing import Callable, Iterator, Literal, Protocol
import uuid


RequestOutcome = Literal["succeeded", "failed", "unknown"]


@dataclass(frozen=True)
class BudgetSnapshot:
    """Read-only model projection of existing limits and charged attempts."""

    scope: Literal["run", "invocation"]
    max_llm_calls: int
    llm_calls_used: int
    remaining_llm_calls: int
    remaining_timeout_seconds: float

    def render(self) -> str:
        owner = (
            "This is the entire Run's shared total, used by Scientific, Compiler, "
            "all execution Agents, permission reviewers and later work, including final delivery. "
            "It is not a dedicated allowance for the current Agent or Task. "
            "Interrupting, resuming or delegating again does not reset it."
            if self.scope == "run" else
            "No Run-wide budget is bound. These are the outer invocation's shared limits, "
            "not an additional allowance for each nested request."
        )
        return (
            "Current execution budget (runtime facts, before dispatching this provider attempt):\n"
            + json.dumps(self.__dict__, ensure_ascii=False) + "\n" + owner + " "
            "Every dispatched provider attempt consumes one call, including failed retries "
            "and requests interrupted after dispatch. This next attempt has not yet been charged."
        )


class BudgetExhaustedError(RuntimeError):
    """The shared request allowance has been consumed."""


class DeadlineExceededError(TimeoutError):
    """The effective execution deadline has elapsed."""


class UsagePersistenceError(RuntimeError):
    """Authoritative model usage could not be loaded or saved; stop execution."""


class UsagePort(Protocol):
    """Persist a charge before permitting its network attempt.

    Implementations raise UsagePersistenceError for authoritative storage errors.
    """

    @property
    def used(self) -> int: ...

    def charge(self, call_id: str, retry_index: int) -> None: ...

    def complete(
        self, call_id: str, retry_index: int, outcome: RequestOutcome,
    ) -> None: ...


@dataclass
class MemoryUsage:
    """The same accounting contract for a standalone module invocation."""

    requests: dict[tuple[str, int], RequestOutcome] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return len(self.requests)

    def charge(self, call_id: str, retry_index: int) -> None:
        key = (call_id, retry_index)
        if key in self.requests:
            raise ValueError("a model request cannot be dispatched twice")
        self.requests[key] = "unknown"

    def complete(
        self, call_id: str, retry_index: int, outcome: RequestOutcome,
    ) -> None:
        key = (call_id, retry_index)
        if key not in self.requests:
            raise ValueError("a model request must be charged before completion")
        self.requests[key] = outcome


@dataclass(frozen=True)
class ExecutionBudget:
    """A narrowed allowance referencing the caller's authoritative usage."""

    usage: UsagePort
    call_limit: int
    deadline: float
    clock: Callable[[], float]
    parent: ExecutionBudget | None = field(default=None, repr=False)

    def snapshot(self) -> BudgetSnapshot:
        """Use the Run owner's facts, or the outer standalone invocation limits."""
        snapshot = getattr(self.usage, "snapshot", None)
        if snapshot is not None:
            return snapshot()
        root = self
        while root.parent is not None:
            root = root.parent
        return BudgetSnapshot(
            scope="invocation", max_llm_calls=root.call_limit,
            llm_calls_used=self.usage.used,
            remaining_llm_calls=max(0, root.call_limit - self.usage.used),
            remaining_timeout_seconds=round(max(0.0, root.deadline - root.clock()), 3),
        )

    @property
    def expired(self) -> bool:
        return self.clock() >= self.deadline

    @property
    def remaining_calls(self) -> int:
        return max(0, self.call_limit - self.usage.used)

    def remaining_timeout(self, operation_timeout: float | None = None) -> float:
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise DeadlineExceededError("execution deadline exceeded")
        return remaining if operation_timeout is None else min(remaining, operation_timeout)

    def charge(self, call_id: str, retry_index: int) -> None:
        self.check()
        self.usage.charge(call_id, retry_index)

    def check(self) -> None:
        """Reject exhausted work before scheduling it or waiting to retry."""
        self.remaining_timeout()
        if self.remaining_calls == 0:
            raise BudgetExhaustedError("model request budget exhausted")


_execution_budget: ContextVar[ExecutionBudget | None] = ContextVar(
    "execution_budget", default=None,
)


def current_budget() -> ExecutionBudget | None:
    """Return the trusted execution context, when one is bound."""
    return _execution_budget.get()


def current_budget_status() -> str:
    """Refresh a disposable P3 projection; never append to durable history."""
    budget = current_budget()
    return "" if budget is None else budget.snapshot().render()


@contextmanager
def execution_budget(
    *, max_llm_calls: int, timeout_seconds: float, usage: UsagePort | None = None,
    clock: Callable[[], float] | None = None,
) -> Iterator[ExecutionBudget]:
    """Bind or narrow an invocation without replacing its parent's shared wallet."""
    if max_llm_calls < 0 or timeout_seconds < 0:
        raise ValueError("execution limits cannot be negative")
    parent = current_budget()
    clock = parent.clock if parent is not None else (clock or time.monotonic)
    deadline = clock() + timeout_seconds
    if parent is not None:
        if usage is not None and usage is not parent.usage:
            raise ValueError("a nested invocation cannot replace shared usage")
        usage = parent.usage
        call_limit = min(parent.call_limit, usage.used + max_llm_calls)
        deadline = min(parent.deadline, deadline)
    else:
        usage = usage if usage is not None else MemoryUsage()
        call_limit = usage.used + max_llm_calls
    budget = ExecutionBudget(usage=usage, call_limit=call_limit, deadline=deadline, clock=clock,
                             parent=parent)
    token = _execution_budget.set(budget)
    try:
        yield budget
    finally:
        _execution_budget.reset(token)


def remaining_timeout(operation_timeout: float) -> float:
    """Clip an operation timeout to the enclosing Run's current remaining time."""
    if operation_timeout <= 0:
        raise DeadlineExceededError("operation deadline exceeded")
    budget = current_budget()
    return operation_timeout if budget is None else budget.remaining_timeout(operation_timeout)


def invoke_model(client, method: str, *args, **kwargs):
    """Charge minimal clients once; transport-aware clients charge each retry."""
    if getattr(client, "manages_usage", False):
        return getattr(client, method)(*args, **kwargs)
    budget = current_budget()
    if budget is None:
        raise RuntimeError("model invocation requires an execution budget")
    call_id = uuid.uuid4().hex
    budget.charge(call_id, 0)
    outcome: RequestOutcome = "unknown"
    try:
        result = getattr(client, method)(*args, **kwargs)
        outcome = "succeeded"
        return result
    finally:
        budget.usage.complete(call_id, 0, outcome)
