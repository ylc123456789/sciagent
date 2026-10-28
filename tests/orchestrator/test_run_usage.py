"""Run reservations survive interrupted calls without replacing live artifacts."""
from datetime import UTC, datetime

import pytest

from resagent2_contracts import ArtifactCandidate, ResearchRequest, RunBudget, RunPermissions
from resagent2_orchestrator import JsonRunStore, ResearchRun
from resagent2_orchestrator.artifacts import ArtifactRegistry, ScientificArtifactRegistration
from resagent2_orchestrator.usage import RunUsagePort
from resagent2_runtime.budget import (
    BudgetExhaustedError, UsagePersistenceError, execution_budget, invoke_model,
)


def prepared(tmp_path, *, calls=2):
    store = JsonRunStore(tmp_path / "runs")
    now = datetime.now(UTC)
    run = ResearchRun(
        run_id="run_usage", status="running",
        request=ResearchRequest(goal="Test usage",
            permissions=RunPermissions(), budget=RunBudget(max_llm_calls=calls, timeout_seconds=60)),
        created_at=now, updated_at=now,
    )
    store.save(run)
    return store, run


def test_request_reservation_is_on_disk_before_client_is_called(tmp_path):
    store, run = prepared(tmp_path)
    port = RunUsagePort(run, store)
    class Client:
        def next_action(self):
            saved = JsonRunStore(store.root).load(run.run_id)
            assert list(saved.usage.requests.values()) == ["unknown"]
            return "done"
    with execution_budget(max_llm_calls=2, timeout_seconds=60, usage=port):
        assert invoke_model(Client(), "next_action") == "done"
    saved = store.load(run.run_id)
    assert saved.llm_calls_used == 1
    assert list(saved.usage.requests.values()) == ["succeeded"]


@pytest.mark.parametrize("store_method", ["load", "save"])
@pytest.mark.parametrize("storage_error", [OSError, ValueError])
def test_failed_reservation_prevents_dispatch(tmp_path, monkeypatch, store_method, storage_error):
    store, run = prepared(tmp_path)
    port = RunUsagePort(run, store)
    class Client:
        def next_action(self):
            pytest.fail("request must not be sent before durable reservation")
    def fail(_):
        raise storage_error("private record contents must not appear in feedback")
    monkeypatch.setattr(store, store_method, fail)
    with execution_budget(max_llm_calls=2, timeout_seconds=60, usage=port):
        with pytest.raises(UsagePersistenceError, match="Run model usage could not be persisted") as caught:
            invoke_model(Client(), "next_action")
    assert isinstance(caught.value.__cause__, storage_error)
    assert "private record contents" not in str(caught.value)
    assert JsonRunStore(store.root).load(run.run_id).llm_calls_used == 0


@pytest.mark.parametrize("store_method", ["load", "save"])
def test_failed_completion_persistence_remains_unknown_without_replaying(tmp_path, monkeypatch, store_method):
    store, run = prepared(tmp_path)
    port = RunUsagePort(run, store)
    calls = []

    def fail(_):
        raise OSError("completion storage unavailable")

    class Client:
        def next_action(self):
            calls.append("sent")
            assert list(JsonRunStore(store.root).load(run.run_id).usage.requests.values()) == ["unknown"]
            monkeypatch.setattr(store, store_method, fail)
            return "done"

    with execution_budget(max_llm_calls=2, timeout_seconds=60, usage=port):
        with pytest.raises(UsagePersistenceError) as caught:
            invoke_model(Client(), "next_action")

    assert isinstance(caught.value.__cause__, OSError)
    assert calls == ["sent"]
    saved = JsonRunStore(store.root).load(run.run_id)
    assert saved.llm_calls_used == 1
    assert list(saved.usage.requests.values()) == ["unknown"]


def test_unknown_reservations_are_not_refunded_on_restart(tmp_path):
    store, run = prepared(tmp_path, calls=1)
    RunUsagePort(run, store).charge("interrupted", 0)
    reopened = JsonRunStore(store.root)
    saved = reopened.load(run.run_id)
    assert saved.usage.requests == {"interrupted:0": "unknown"}
    port = RunUsagePort(saved, reopened)
    with execution_budget(max_llm_calls=0, timeout_seconds=60, usage=port):
        with pytest.raises(BudgetExhaustedError):
            invoke_model(object(), "next_action")
    assert reopened.load(run.run_id).usage.requests == {"interrupted:0": "unknown"}


def test_usage_updates_preserve_artifacts_registered_during_invocation(tmp_path):
    store, run = prepared(tmp_path)
    port = RunUsagePort(run, store)
    port.charge("first", 0)
    bridge = ScientificArtifactRegistration(ArtifactRegistry(tmp_path / "artifacts"), store)
    ref = bridge.register_scientific(
        ArtifactCandidate(kind="literature_paper", path="papers.md", media_type="text/markdown",
                          summary="Search result", content="A paper"),
        run_id=run.run_id, session_id="session_usage", operation_key="session:session_usage:turn:1:call:fetch", output_position=0,
    )
    port.complete("first", 0, "succeeded")
    port.charge("second", 0)
    persisted = store.load(run.run_id)
    assert persisted.artifacts[ref.id] == ref
    assert persisted.llm_calls_used == 2
    assert persisted.usage.requests == {"first:0": "succeeded", "second:0": "unknown"}


def test_snapshot_uses_run_total_across_narrowed_invocations_and_resume(tmp_path):
    store, run = prepared(tmp_path, calls=12)
    port = RunUsagePort(run, store)
    port.charge("earlier", 0)
    port.complete("earlier", 0, "failed")
    port.charge("interrupted", 0)
    with execution_budget(max_llm_calls=3, timeout_seconds=4, usage=port) as invocation:
        with execution_budget(max_llm_calls=1, timeout_seconds=1) as nested:
            assert nested.remaining_calls == 1
            snapshot = nested.snapshot()
            assert snapshot.scope == "run"
            assert (snapshot.max_llm_calls, snapshot.llm_calls_used,
                    snapshot.remaining_llm_calls) == (12, 2, 10)
            assert 50 < snapshot.remaining_timeout_seconds <= 60
            nested.charge("current", 0)
            nested.usage.complete("current", 0, "succeeded")
        assert invocation.snapshot().remaining_llm_calls == 9
    restored = store.load(run.run_id)
    with execution_budget(max_llm_calls=2, timeout_seconds=2,
                          usage=RunUsagePort(restored, store)) as resumed:
        snapshot = resumed.snapshot()
        assert (snapshot.max_llm_calls, snapshot.llm_calls_used,
                snapshot.remaining_llm_calls) == (12, 3, 9)
        assert restored.usage.requests["interrupted:0"] == "unknown"
