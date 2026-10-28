"""Read-only terminal lookup is distinct from invocation or Session resume."""

from datetime import UTC, datetime

import pytest

from sciagent_contracts import AgentResult, ModuleError, SessionRef, task_session_id
from sciagent_runtime import AgentLoop, InMemorySessionStore, ScriptedLLMClient
from sciagent_runtime.models import AgentState
from sciagent_executor import NativeExecutorAgent


def state(status="completed", owner="executor"):
    now = datetime.now(UTC)
    value = AgentState(work_request_id="work_test",
        session_id=task_session_id("run_lookup", "task_lookup", 1),
        agent_name=owner, owner=owner, run_id="run_lookup", task_id="task_lookup",
        attempt_number=1, status=status, created_at=now, updated_at=now,
    )
    if status in {"completed", "failed"}:
        value.terminal_result = AgentResult(
            status=status, report="Original saved facts", llm_calls=3,
            error=ModuleError(code="contract_error", message="Known failure", retryable=True)
            if status == "failed" else None,
            session=SessionRef(id=value.session_id, module=owner, status=status,
                               state_uri="session://" + value.session_id, created_at=now, updated_at=now),
        )
    return value


@pytest.mark.parametrize("status", ["completed", "failed", "active", "paused", "cancelled", "missing"])
def test_native_lookup_never_prepares_invokes_or_mutates(tmp_path, monkeypatch, status):
    store = InMemorySessionStore()
    saved = state(status if status != "missing" else "active", owner="executor")
    if status != "missing":
        store.save(saved)
    before = saved.model_dump(mode="json")
    client = ScriptedLLMClient([])
    agent = NativeExecutorAgent(lambda _: client, store=store)

    def forbidden(*args, **kwargs):
        raise AssertionError("terminal lookup must not prepare or execute work")

    monkeypatch.setattr(agent.loop, "run", forbidden)
    monkeypatch.setattr("sciagent_components.RepoMaterializer.materialize", forbidden)
    monkeypatch.setattr("sciagent_components.GitWorkspace.snapshot", forbidden)
    monkeypatch.setattr("sciagent_components.GitWorkspace.write_patch_since", forbidden)
    result = agent.read_terminal_result(run_id="run_lookup", task_id="task_lookup", attempt_number=1)
    assert client.contexts == []
    if status in {"completed", "failed"}:
        assert result == saved.terminal_result.model_copy(update={"llm_calls": 0})
        result.report = "Caller modified its copy"
    else:
        assert result is None
    if status != "missing":
        assert store.load(saved.session_id).model_dump(mode="json") == before
    else:
        assert not store.exists(saved.session_id)


@pytest.mark.parametrize("field,value", [("run_id", "run_other"), ("task_id", "task_other"),
                                         ("attempt_number", 2), ("owner", "scientific")])
def test_lookup_checks_owner_and_attempt_identity(field, value):
    saved = state()
    store = InMemorySessionStore()
    store.save(saved)
    fields = dict(session_id=saved.session_id, run_id=saved.run_id, task_id=saved.task_id,
                  attempt_number=saved.attempt_number, owner=saved.owner)
    fields[field] = value
    with pytest.raises(ValueError, match="does not belong"):
        AgentLoop(store=store).read_terminal_result(**fields)
    assert store.load(saved.session_id) == saved


def test_terminal_lookup_does_not_rebuild_unsaved_failure_diagnostics():
    saved = state("failed")
    store = InMemorySessionStore()
    store.save(saved)
    result = NativeExecutorAgent(lambda _: ScriptedLLMClient([]), store=store).read_terminal_result(
        run_id=saved.run_id, task_id=saved.task_id, attempt_number=1,
    )
    assert result.artifacts == []
    assert result.error == saved.terminal_result.error


def test_terminal_checkpoint_storage_failure_propagates_without_publishing_early_result():
    from e2e.native_fixtures import tool_turns
    from sciagent_contracts import AgentRequest, TaskBudget
    from sciagent_runtime import (
        AgentDefinition, AllowListPermissionPolicy, CompletionDecision, FinishTool,
    )
    class FailedTerminalStore(InMemorySessionStore):
        def save(self, current):
            if current.terminal_result is not None:
                raise OSError("terminal checkpoint unavailable")
            super().save(current)
    store = FailedTerminalStore()
    class Completion:
        def evaluate(self, state, candidate):
            return CompletionDecision(complete=True, report=candidate.report)
    definition = AgentDefinition(
        name="executor", owner="executor", system_prompt="Finish the fixture.",
        tools=(FinishTool(),), llm_client=ScriptedLLMClient(tool_turns([
            {"tool": "finish", "arguments": {"report": "Original terminal report"}},
        ])), context_builder=lambda *args: [], permission_policy=AllowListPermissionPolicy({"finish"}),
        completion_check=Completion(),
    )
    request = AgentRequest(work_request_id="work_test", run_id="run_lookup", task_id="task_lookup", attempt_number=1,
                           agent="executor", instruction="Finish", permissions={}, budget=TaskBudget(
                               max_llm_calls=2, timeout_seconds=30))
    loop = AgentLoop(store=store)
    with pytest.raises(OSError, match="terminal checkpoint unavailable"):
        loop.run(definition, request, session_id=task_session_id(request.run_id, request.task_id, 1))
    assert loop.read_terminal_result(
        session_id=task_session_id(request.run_id, request.task_id, 1),
        run_id=request.run_id, task_id=request.task_id, attempt_number=1, owner="executor",
    ) is None


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_native_lookup_rejects_foreign_session_payload_in_expected_file(tmp_path, monkeypatch, status):
    from sciagent_runtime import JsonSessionStore
    store = JsonSessionStore(tmp_path / "sessions")
    saved = state(status, owner="executor")
    expected_session_id = saved.session_id
    foreign_session_id = "session_foreign_saved_result"
    saved.session_id = foreign_session_id
    saved.terminal_result.session = saved.terminal_result.session.model_copy(update={
        "id": foreign_session_id, "state_uri": "session://" + foreign_session_id,
    })
    # The snapshot is internally valid and has the requested Run/Task/Attempt.
    # Only its placement under another Session's filename is incorrect.
    AgentState.model_validate(saved.model_dump(mode="json"))
    path = store.root / (expected_session_id + ".json")
    path.write_text(saved.model_dump_json(), encoding="utf-8")
    before = path.read_bytes()
    files_before = sorted(p.name for p in store.root.iterdir())
    client = ScriptedLLMClient([])
    agent = NativeExecutorAgent(lambda _: client, store=store)
    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched Session lookup must not execute or write")
    monkeypatch.setattr(agent.loop, "run", forbidden)
    monkeypatch.setattr(store, "save", forbidden)
    monkeypatch.setattr("sciagent_components.RepoMaterializer.materialize", forbidden)
    monkeypatch.setattr("sciagent_components.GitWorkspace.snapshot", forbidden)
    monkeypatch.setattr("sciagent_components.GitWorkspace.write_patch_since", forbidden)
    with pytest.raises(ValueError, match="does not belong"):
        agent.read_terminal_result(run_id="run_lookup", task_id="task_lookup", attempt_number=1)
    assert client.contexts == []
    assert path.read_bytes() == before
    assert sorted(p.name for p in store.root.iterdir()) == files_before
