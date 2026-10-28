"""Reconnect saved Native terminal results before considering a new Attempt."""
from e2e.scheduler_fixtures import executor_bindings

from scheduler_fixtures import accept_proposal

from datetime import UTC, datetime
import subprocess
import json

import pytest

from e2e.native_fixtures import tool_turns
from resagent2_contracts import (
    AgentOwner, AgentResult, Attempt, ModuleError, ResearchRequest, RunBudget,
    RunPermissions, SessionRef, TaskProposal, TaskStatus, WorkflowProposal, WorkspaceSpec,
    task_session_id,
)
from resagent2_components import ResourceLayout
from resagent2_coding import NativeCodingAgent
from resagent2_experiment import NativeExperimentAgent
from resagent2_orchestrator import (
    DeterministicWorkInterpreter, ModuleBinding, ResearchController, ResearchRun, WorkflowScheduler,
)
from resagent2_runtime import JsonSessionStore, ScriptedLLMClient


class ProcessCrash(BaseException):
    pass


class CrashAfterTerminal(JsonSessionStore):
    def save(self, state):
        super().save(state)
        if state.terminal_result is not None:
            raise ProcessCrash()


def setup(tmp_path, port, kind="coding"):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "value.txt").write_text("MARK\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "value.txt"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "baseline"], cwd=repo, check=True)
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({kind: ModuleBinding(owner=AgentOwner.EXECUTOR, port=port)}),
        artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data",
        workspaces={"ws_test": WorkspaceSpec(
            workspace_id="ws_test", source_kind="local", location=str(repo),
            access={"read_paths": ["."], "write_paths": ["."]},
        )},
    )
    now = datetime.now(UTC)
    run = ResearchRun(
        run_id="run_terminal_gap", request=ResearchRequest(
            goal="Recover the original result", permissions=RunPermissions(),
            budget=RunBudget(max_llm_calls=20, timeout_seconds=120),
        ), status="running", workspaces=scheduler._resolve_workspaces("run_terminal_gap"),
        created_at=now, updated_at=now,
    )
    scheduler.store.save(run)
    accept_proposal(scheduler, run.run_id, WorkflowProposal(work_request_id="work_original", tasks=[
        TaskProposal(execution_ref=kind + "@1", id="task_result", work_request_id="work_original", workflow_agent_kind="executor",
                     instruction="Apply one edit, then return the actual result"),
    ]))
    return scheduler, repo


@pytest.mark.parametrize("kind", ["coding", "experiment"])
@pytest.mark.parametrize("outcome", ["completed", "failed"])
@pytest.mark.parametrize("boundary", ["loop_terminal", "native_public"])
def test_native_saved_terminal_is_received_without_reexecution(tmp_path, monkeypatch, kind, outcome, boundary):
    actions = ([{"tool": "replace_text", "arguments": {
        "path": "value.txt", "old_text": "MARK", "new_text": "MARK\nACTUAL_EDIT",
    }}] if kind == "coding" else [])
    if kind != "coding" or outcome != "failed":
        actions.append({"tool": "finish", "arguments": {"status": outcome, "report": "Original exact report.  \n"}})
    client = ScriptedLLMClient(tool_turns(actions))
    native_class = {"coding": NativeCodingAgent, "experiment": NativeExperimentAgent}[kind]
    store_class = CrashAfterTerminal if boundary == "loop_terminal" else JsonSessionStore

    class Native(native_class):
        def invoke(self, request):
            result = super().invoke(request)
            if boundary == "native_public":
                raise ProcessCrash()
            return result

    agent = Native(client, store=store_class(tmp_path / "sessions"),
                   resource_layout=ResourceLayout(resource_root=tmp_path / "resources"))
    scheduler, repo = setup(tmp_path, agent, kind)
    with pytest.raises(ProcessCrash):
        scheduler.execute_task("run_terminal_gap", "task_result")
    run = scheduler.store.load("run_terminal_gap")
    used_before = run.llm_calls_used
    run.request.budget.max_llm_calls = used_before  # reception needs no spare model call
    scheduler.store.save(run)
    session_store = JsonSessionStore(tmp_path / "sessions")
    session_id = task_session_id(run.run_id, "task_result", 1)
    saved = session_store.load(session_id)
    saved_dump = saved.model_dump(mode="json")
    if kind == "coding" and outcome == "failed":
        # The first terminal checkpoint already contains the complete public packaging.
        assert any(a.kind == "code_patch" for a in saved.terminal_result.artifacts)

    new_client = ScriptedLLMClient([])
    new_agent = native_class(new_client, store=session_store)
    scheduler.bindings["executor"] = ModuleBinding(owner=AgentOwner.EXECUTOR, port=new_agent)

    def forbidden(*args, **kwargs):
        raise AssertionError("result recovery must not prepare, invoke or diagnose")

    monkeypatch.setattr(new_agent, "invoke", forbidden)
    monkeypatch.setattr("resagent2_components.RepoMaterializer.materialize", forbidden)
    monkeypatch.setattr("resagent2_components.GitWorkspace.snapshot", forbidden)
    monkeypatch.setattr("resagent2_components.GitWorkspace.write_patch_since", forbidden)
    scheduler._recover_interrupted_attempts_in_place(run)
    recovered = scheduler.store.load(run.run_id)
    attempt = recovered.workflow.tasks[0].attempts[0]
    assert len(recovered.workflow.tasks[0].attempts) == 1
    assert attempt.status == outcome
    assert attempt.report == saved.terminal_result.report
    if outcome == "completed" or kind == "experiment":
        assert attempt.report == "Original exact report.  \n"
    assert attempt.session == saved.terminal_result.session
    assert [recovered.artifacts[key].kind for key in attempt.artifact_ids] == [
        artifact.kind for artifact in saved.terminal_result.artifacts
    ]
    assert recovered.llm_calls_used == used_before == len(client.contexts)
    assert new_client.contexts == []
    assert session_store.load(session_id).model_dump(mode="json") == saved_dump
    if kind == "coding":
        assert (repo / "value.txt").read_text().count("ACTUAL_EDIT") == 1
    registered = list(recovered.artifacts)
    scheduler._recover_interrupted_attempts_in_place(recovered)
    assert list(scheduler.store.load(run.run_id).artifacts) == registered
    assert recovered.pending_module_results == {}


@pytest.mark.parametrize("query", [False, True])
def test_no_terminal_and_unsupported_port_keep_existing_retry_policy(tmp_path, query):
    class Port:
        def invoke(self, request):
            raise AssertionError("recovery lookup must not invoke")
    port = Port()
    if query:
        port.read_terminal_result = lambda **kwargs: None
    scheduler, _ = setup(tmp_path, port)
    run = scheduler.store.load("run_terminal_gap")
    task = run.workflow.tasks[0]
    task.status = TaskStatus.RUNNING
    task.attempts = [Attempt(number=1, status="running", started_at=datetime.now(UTC), agent_started=True)]
    scheduler.store.save(run)
    scheduler._recover_interrupted_attempts_in_place(run)
    assert task.attempts[0].status == "failed"
    assert task.attempts[0].error.retryable is True
    assert task.status == "pending"


def test_saved_retryable_failure_still_allows_a_new_attempt(tmp_path):
    now = datetime.now(UTC)
    class Port:
        invokes = 0
        def read_terminal_result(self, **kwargs):
            return AgentResult(status="failed", report="Original retryable failure", error=ModuleError(
                code="contract_error", message="A known failure", retryable=True), session=SessionRef(
                    id="session_original", module="executor", state_uri="session://session_original",
                    status="failed", created_at=now, updated_at=now))
        def invoke(self, request):
            self.invokes += 1
            assert request.attempt_number == 2
            return AgentResult(status="completed", report="Second attempt completed")
    port = Port()
    scheduler, _ = setup(tmp_path, port)
    run = scheduler.store.load("run_terminal_gap")
    task = run.workflow.tasks[0]
    task.status = TaskStatus.RUNNING
    task.attempts = [Attempt(number=1, status="running", started_at=now, agent_started=True)]
    scheduler.store.save(run)
    scheduler._recover_interrupted_attempts_in_place(run)
    final = scheduler.run_until_stable(run.run_id)
    assert [a.status for a in final.workflow.tasks[0].attempts] == ["failed", "completed"]
    assert port.invokes == 1


def test_cancelled_attempt_is_not_queried_or_revived(tmp_path):
    class Port:
        def invoke(self, request):
            raise AssertionError("cancelled work must not execute")
        def read_terminal_result(self, **kwargs):
            raise AssertionError("cancelled work must not be queried")
    scheduler, _ = setup(tmp_path, Port())
    run = scheduler.store.load("run_terminal_gap")
    task = run.workflow.tasks[0]
    task.status = TaskStatus.CANCELLED
    task.attempts = [Attempt(number=1, status="cancelled", started_at=datetime.now(UTC),
                             finished_at=datetime.now(UTC), agent_started=True,
                             error=ModuleError(code="interrupted", message="User cancelled", retryable=False))]
    scheduler.store.save(run)
    scheduler._recover_interrupted_attempts_in_place(run)
    assert task.status == task.attempts[0].status == "cancelled"


def test_existing_pending_result_is_received_before_any_terminal_query(tmp_path, monkeypatch):
    class Port:
        calls = 0
        def invoke(self, request):
            self.calls += 1
            return AgentResult(status="completed", report="Already returned original result")
        def read_terminal_result(self, **kwargs):
            raise AssertionError("existing pending result takes precedence over lookup")
    port = Port()
    scheduler, _ = setup(tmp_path, port)
    with monkeypatch.context() as patch:
        patch.setattr(scheduler, "_persist_attempt_report", lambda *args, **kwargs: (_ for _ in ()).throw(ProcessCrash()))
        with pytest.raises(ProcessCrash):
            scheduler.execute_task("run_terminal_gap", "task_result")
    run = scheduler.store.load("run_terminal_gap")
    assert run.pending_module_results
    scheduler._recover_interrupted_attempts_in_place(run)
    final = scheduler.store.load(run.run_id)
    assert final.workflow.tasks[0].attempts[0].report == "Already returned original result"
    assert final.pending_module_results == {}
    assert port.calls == 1


def test_first_saved_retryable_failure_already_contains_coding_public_facts(tmp_path, monkeypatch):
    class TransportFailure(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            if not self._turns:
                raise OSError("temporary model transport failure")
            return super().next_tool_call(*args, **kwargs)

    client = TransportFailure(tool_turns([{"tool": "replace_text", "arguments": {
        "path": "value.txt", "old_text": "MARK", "new_text": "MARK\nACTUAL_EDIT",
    }}]))
    agent = NativeCodingAgent(client, store=CrashAfterTerminal(tmp_path / "sessions"),
                              resource_layout=ResourceLayout(resource_root=tmp_path / "resources"))
    scheduler, repo = setup(tmp_path, agent)
    with pytest.raises(ProcessCrash):
        scheduler.execute_task("run_terminal_gap", "task_result")
    saved = JsonSessionStore(tmp_path / "sessions").load(
        task_session_id("run_terminal_gap", "task_result", 1))
    assert saved.terminal_result.error.retryable is False
    assert any(a.kind == "code_patch" for a in saved.terminal_result.artifacts)
    fresh = NativeCodingAgent(ScriptedLLMClient([]), store=JsonSessionStore(tmp_path / "sessions"))
    def forbidden(*args, **kwargs):
        raise AssertionError("a saved failure with actual changes must not retry")
    monkeypatch.setattr(fresh, "invoke", forbidden)
    monkeypatch.setattr("resagent2_components.GitWorkspace.write_patch_since", forbidden)
    monkeypatch.setattr("resagent2_components.RepoMaterializer.materialize", forbidden)
    scheduler.bindings["executor"] = ModuleBinding(owner=AgentOwner.EXECUTOR, port=fresh)
    scheduler._recover_interrupted_attempts_in_place(scheduler.store.load("run_terminal_gap"))
    recovered = scheduler.run_until_stable("run_terminal_gap")
    assert recovered.workflow.tasks[0].status == "failed"
    assert len(recovered.workflow.tasks[0].attempts) == 1
    assert (repo / "value.txt").read_text().count("ACTUAL_EDIT") == 1


@pytest.mark.parametrize("kind", ["coding", "experiment"])
def test_controller_resume_receives_complete_result_before_budget_failure(tmp_path, monkeypatch, kind):
    client = ScriptedLLMClient(tool_turns([
        {"tool": "run_shell", "arguments": {"command": "printf saved-command"}},
    ]))
    native_class = {"coding": NativeCodingAgent, "experiment": NativeExperimentAgent}[kind]
    agent = native_class(client, store=CrashAfterTerminal(tmp_path / "sessions"),
                         resource_layout=ResourceLayout(resource_root=tmp_path / "resources"))
    scheduler, _ = setup(tmp_path, agent, kind)
    run = scheduler.store.load("run_terminal_gap")
    run.request.permissions.execute_commands = True
    run.request.permissions.approval_mode = "auto_review"
    scheduler.store.save(run)
    # Permit only this harmless fixture command; no permission-review model is called.
    from resagent2_runtime import ApprovalReviewVerdict
    class Reviewer:
        def review(self, operation, request):
            return ApprovalReviewVerdict(decision="allow", reason="Harmless fixture command")
    agent.operation_reviewer = Reviewer()
    with pytest.raises(ProcessCrash):
        scheduler.execute_task(run.run_id, "task_result")
    run = scheduler.store.load(run.run_id)
    saved = JsonSessionStore(tmp_path / "sessions").load(task_session_id(run.run_id, "task_result", 1))
    record = next(a for a in saved.terminal_result.artifacts if a.kind == "execution_record")
    assert json.loads(record.content)["results"][0]["exit_code"] == 0
    run.request.budget.max_llm_calls = run.llm_calls_used
    usage_before = run.usage.model_dump(mode="json")
    scheduler.store.save(run)
    new_agent = native_class(ScriptedLLMClient([]), store=JsonSessionStore(tmp_path / "sessions"))
    def forbidden(*args, **kwargs):
        raise AssertionError("recovery must not execute or prepare anything")
    monkeypatch.setattr(new_agent, "invoke", forbidden)
    monkeypatch.setattr("resagent2_components.RepoMaterializer.materialize", forbidden)
    monkeypatch.setattr("resagent2_components.GitWorkspace.write_patch_since", forbidden)
    monkeypatch.setattr("resagent2_components.GitWorkspace.snapshot", forbidden)
    scheduler.bindings["executor"] = ModuleBinding(owner=AgentOwner.EXECUTOR, port=new_agent)
    class Scientific:
        invoke = staticmethod(forbidden)
        validate = staticmethod(forbidden)
    controller = ResearchController(
        scientific_port=Scientific(), compiler=None, scheduler=scheduler, registry=None,
        interpreter=DeterministicWorkInterpreter(), gate=Scientific(),
    )
    final = controller.resume_run(run.run_id)
    attempt = final.workflow.tasks[0].attempts[0]
    assert attempt.report == saved.terminal_result.report
    assert len(final.workflow.tasks[0].attempts) == 1
    assert attempt.session == saved.terminal_result.session
    assert any(final.artifacts[key].kind == "execution_record" for key in attempt.artifact_ids)
    assert final.usage.model_dump(mode="json") == usage_before
    assert final.status == "failed" and final.terminal_error.code == "budget_exhausted"


@pytest.mark.parametrize("boundary", ["output_registration", "report_binding"])
def test_recovered_terminal_reception_survives_another_crash(tmp_path, monkeypatch, boundary):
    agent = NativeCodingAgent(ScriptedLLMClient(tool_turns([
        {"tool": "replace_text", "arguments": {
            "path": "value.txt", "old_text": "MARK", "new_text": "MARK\nACTUAL_EDIT",
        }},
        {"tool": "finish", "arguments": {"report": "Original final report"}},
    ])), store=CrashAfterTerminal(tmp_path / "sessions"),
       resource_layout=ResourceLayout(resource_root=tmp_path / "resources"))
    scheduler, repo = setup(tmp_path, agent)
    with pytest.raises(ProcessCrash):
        scheduler.execute_task("run_terminal_gap", "task_result")
    session_store = JsonSessionStore(tmp_path / "sessions")
    saved = session_store.load(task_session_id("run_terminal_gap", "task_result", 1)).terminal_result
    assert [item.kind for item in saved.artifacts] == ["code_patch"]
    fresh = NativeCodingAgent(ScriptedLLMClient([]), store=session_store)
    scheduler.bindings["executor"] = ModuleBinding(owner=AgentOwner.EXECUTOR, port=fresh)
    with monkeypatch.context() as patch:
        if boundary == "output_registration":
            original = scheduler.artifact_registry.register_many
            def register_then_crash(*args, **kwargs):
                original(*args, **kwargs)
                raise ProcessCrash()
            patch.setattr(scheduler.artifact_registry, "register_many", register_then_crash)
        else:
            original = scheduler._save
            def save_then_crash(run):
                original(run)
                if run.workflow.tasks[0].attempts[0].report_artifact_id:
                    raise ProcessCrash()
            patch.setattr(scheduler, "_save", save_then_crash)
        with pytest.raises(ProcessCrash):
            scheduler._recover_interrupted_attempts_in_place(scheduler.store.load("run_terminal_gap"))
    interrupted = scheduler.store.load("run_terminal_gap")
    allocated = {p.ref.id for p in interrupted.pending_registrations.values()} | set(interrupted.artifacts)
    def forbidden(*args, **kwargs):
        raise AssertionError("persisted receiving result takes precedence")
    monkeypatch.setattr(fresh, "invoke", forbidden)
    monkeypatch.setattr(fresh, "read_terminal_result", forbidden)
    scheduler._recover_interrupted_attempts_in_place(interrupted)
    final = scheduler.store.load("run_terminal_gap")
    attempt = final.workflow.tasks[0].attempts[0]
    assert attempt.status == "completed" and attempt.report == "Original final report"
    assert allocated <= set(final.artifacts)
    assert attempt.session == saved.session
    received = [final.artifacts[key] for key in attempt.artifact_ids]
    assert [(item.kind, item.media_type, item.summary, item.output_name) for item in received] == [
        (item.kind, item.media_type, item.summary, item.output_name) for item in saved.artifacts
    ]
    assert set(final.artifacts) == {*attempt.artifact_ids, attempt.report_artifact_id}
    assert not final.pending_module_results and not final.pending_registrations
    assert (repo / "value.txt").read_text().count("ACTUAL_EDIT") == 1


@pytest.mark.parametrize("stopped", [True, False])
def test_diagnostic_cancellation_preserves_saved_failure_and_original_edit(tmp_path, monkeypatch, stopped):
    from resagent2_runtime.cancellation import OperationCancelled
    agent = NativeCodingAgent(ScriptedLLMClient(tool_turns([
        {"tool": "replace_text", "arguments": {
            "path": "value.txt", "old_text": "MARK", "new_text": "MARK\nACTUAL_EDIT",
        }},
    ])), store=JsonSessionStore(tmp_path / "sessions"),
       resource_layout=ResourceLayout(resource_root=tmp_path / "resources"))
    scheduler, repo = setup(tmp_path, agent)
    def interrupted_diagnostic(*args, **kwargs):
        raise OperationCancelled("diagnostic interrupted", stopped=stopped,
                                 details={"operation": "process", "pid": 123})
    monkeypatch.setattr("resagent2_components.GitWorkspace.write_patch_since", interrupted_diagnostic)
    final = scheduler.execute_task("run_terminal_gap", "task_result")
    saved = JsonSessionStore(tmp_path / "sessions").load(task_session_id(final.run_id, "task_result", 1))
    assert saved.terminal_result.error.code == "tool_failed"
    assert "scripted LLM has no remaining turn" in saved.terminal_result.error.message
    assert saved.terminal_result.error.retryable is False
    assert saved.terminal_result.error.details["stop_confirmed"] is stopped
    assert saved.terminal_result.error.details["pid"] == 123
    assert final.workflow.tasks[0].attempts[0].error == saved.terminal_result.error
    assert (repo / "value.txt").read_text().count("ACTUAL_EDIT") == 1
    if not stopped:
        assert final.status == "failed" and final.terminal_error.details["stop_confirmed"] is False


def test_diagnostic_cancellation_cannot_overwrite_original_unconfirmed_process(tmp_path, monkeypatch):
    from resagent2_capabilities import RunShellTool
    from resagent2_runtime import PermissionDecision
    from resagent2_runtime.cancellation import OperationCancelled
    original = {"operation": "process", "pid": 123, "command": "original-running-command",
                "files_retained": True, "exit_code": None}
    def unconfirmed_command(*args, **kwargs):
        (tmp_path / "repo" / "partial.txt").write_text("partial command output")
        raise OperationCancelled("original process stop is unconfirmed", stopped=False, details=original)
    def cancelled_diagnostic(*args, **kwargs):
        raise OperationCancelled("diagnostic did not start", stopped=True, details={
            "operation": "process", "pid": 999, "command": "git diagnostic",
        })
    monkeypatch.setattr(RunShellTool, "execute", unconfirmed_command)
    monkeypatch.setattr("resagent2_capabilities.permissions.OperationPermissionPolicy.check",
                        lambda *_: PermissionDecision(outcome="allow"))
    monkeypatch.setattr("resagent2_components.GitWorkspace.write_patch_since", cancelled_diagnostic)
    agent = NativeCodingAgent(ScriptedLLMClient(tool_turns([
        {"tool": "run_shell", "arguments": {"command": original["command"]}},
    ])), store=JsonSessionStore(tmp_path / "sessions"),
       resource_layout=ResourceLayout(resource_root=tmp_path / "resources"))
    scheduler, _ = setup(tmp_path, agent)
    final = scheduler.execute_task("run_terminal_gap", "task_result")
    saved = JsonSessionStore(tmp_path / "sessions").load(task_session_id(final.run_id, "task_result", 1))
    error = saved.terminal_result.error
    assert error.code == "interrupted" and error.retryable is False
    assert error.details["stop_confirmed"] is False
    assert error.details["pid"] == original["pid"]
    assert error.details["command"] == original["command"]
    assert error.details["diagnostic_patch_details"]["stop_confirmed"] is True
    assert error.details["diagnostic_patch_details"]["pid"] == 999
    assert error.details["diagnostic_patch_details"]["command"] == "git diagnostic"
    record = next(a for a in saved.terminal_result.artifacts if a.kind == "execution_record")
    row = json.loads(record.content)["results"][0]
    assert row["stop_confirmed"] is False and row["pid"] == original["pid"]
    assert final.status == "failed"
    assert final.terminal_error == error
