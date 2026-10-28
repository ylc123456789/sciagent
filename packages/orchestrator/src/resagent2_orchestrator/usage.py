"""Bind the shared execution budget to the existing atomic Run store."""

from datetime import UTC, datetime

from resagent2_runtime.budget import (
    BudgetExhaustedError, BudgetSnapshot, DeadlineExceededError, RequestOutcome,
    UsagePersistenceError,
)

from .models import ResearchRun
from .store import RunStore


class RunUsagePort:
    """A single Run owner records each request before allowing its dispatch."""

    def __init__(self, run: ResearchRun, store: RunStore) -> None:
        self.run = run
        self.store = store

    @property
    def used(self) -> int:
        return self.run.usage.used

    def snapshot(self) -> BudgetSnapshot:
        """Project the same Run facts used by enforcement; no second ledger."""
        limit = self.run.request.budget.max_llm_calls
        return BudgetSnapshot(
            scope="run", max_llm_calls=limit, llm_calls_used=self.used,
            remaining_llm_calls=max(0, limit - self.used),
            remaining_timeout_seconds=round(
                self.run.remaining_timeout_seconds(datetime.now(UTC)), 3),
        )

    def charge(self, call_id: str, retry_index: int) -> None:
        if self.run.remaining_timeout_seconds(datetime.now(UTC)) <= 0:
            raise DeadlineExceededError("Run execution deadline exceeded")
        if self.used >= self.run.request.budget.max_llm_calls:
            raise BudgetExhaustedError("Run model request budget exhausted")
        key = f"{call_id}:{retry_index}"
        if key in self.run.usage.requests:
            raise ValueError("model request already reserved; dispatch cannot be replayed")
        self.run.usage.requests[key] = "unknown"
        self._save()

    def complete(self, call_id: str, retry_index: int, outcome: RequestOutcome) -> None:
        key = f"{call_id}:{retry_index}"
        if key not in self.run.usage.requests:
            raise ValueError("model request has no reservation")
        self.run.usage.requests[key] = outcome
        self._save()

    def _save(self) -> None:
        try:
            latest = self.store.load(self.run.run_id)
            latest.usage = self.run.usage.model_copy(deep=True)
            latest.updated_at = datetime.now(UTC)
            self.store.save(latest)
        except Exception as error:
            # This is authoritative Run storage, not an optional model service.
            # Keep the cause locally without exposing record contents in feedback.
            raise UsagePersistenceError("Run model usage could not be persisted") from error
