"""The replaceable ModulePort must obey both envelope and Agent routing contracts."""
from e2e.scheduler_fixtures import executor_bindings

from scheduler_fixtures import accept_proposal
from control_fixtures import question_signal, ScopedScriptedModulePort

from e2e.native_fixtures import tool_turns

from sciagent_contracts import RunPermissions, ExecutionLimits
from sciagent_runtime.budget import current_budget

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from sciagent_contracts import (
    AgentOwner,
    AgentResult,
    ArtifactCandidate,
    ControlSignal,
    ErrorCode,
    ModuleError,
    ModuleStatus,
    QuestionDraft,
    RecordedAnswer,
    ResearchRequest,
    RunBudget,
    RunStatus,
    SessionRef,
    SessionStatus,
    TaskProposal,
    TaskStatus,
    WarningRecord,
    WorkflowAgentKind,
    WorkflowProposal,
)
from sciagent_orchestrator import (
    InMemoryRunStore, ModuleBinding, ResearchRun, ScriptedModulePort, WorkflowScheduler,
)


def _session(status=SessionStatus.COMPLETED):
    now = datetime.now(UTC)
    return SessionRef(
        id="session_boundary", module=AgentOwner.EXECUTOR, state_uri="session://boundary",
        status=status, created_at=now, updated_at=now,
    )




def _execute(tmp_path, result, agent_kind=WorkflowAgentKind.EXECUTOR, *, charged_calls=0):
    class MeteredPort(ScopedScriptedModulePort):
        def invoke(self, request):
            budget = current_budget()
            for index in range(charged_calls):
                budget.charge("test-module", index)
                budget.usage.complete("test-module", index, "succeeded")
            return super().invoke(request)
    port = MeteredPort([result])
    engine = WorkflowScheduler(
        bindings=executor_bindings({agent_kind: ModuleBinding(
            owner=AgentOwner(agent_kind.value),
            port=port,
        )}),
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data",
    )
    now = datetime.now(UTC)
    engine.store.save(ResearchRun(run_id='run_boundary', request=ResearchRequest(goal='Test ModulePort', budget=RunBudget(max_llm_calls=50, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=3, max_attempts_per_task=2)), status=RunStatus.RUNNING, created_at=now, updated_at=now))
    accept_proposal(engine, "run_boundary", WorkflowProposal(
        work_request_id="work_boundary",
        tasks=[TaskProposal(execution_ref=(None if agent_kind == WorkflowAgentKind.SCIENTIFIC else "experiment@1"),
            id="task_boundary", work_request_id="work_boundary", instruction="Test",
            workflow_agent_kind=("scientific" if str(agent_kind) == "scientific" else "executor"),
        )],
    ))
    run = engine.run_until_stable("run_boundary")
    assert len(port.requests) == 1
    return engine, run, run.workflow.tasks[0].attempts[0]


@pytest.mark.parametrize("agent_kind", list(WorkflowAgentKind))
@pytest.mark.parametrize("with_warnings", [False, True])
def test_success_for_each_agent_uses_shared_envelope(tmp_path, agent_kind, with_warnings):
    result = AgentResult(
        status=ModuleStatus.COMPLETED_WITH_WARNINGS if with_warnings else ModuleStatus.COMPLETED,
        report="done",  llm_calls=3,
        warnings=[WarningRecord(code="caveat", message="Limited sample")] if with_warnings else [],
    )
    _, run, attempt = _execute(tmp_path, result, agent_kind, charged_calls=3)
    assert run.workflow.tasks[0].status == TaskStatus.COMPLETED
    assert attempt.error is None
    assert attempt.session is None
    assert run.llm_calls_used == 3




def test_constructed_envelope_is_revalidated(tmp_path):
    result = AgentResult.model_construct(
        status=ModuleStatus.COMPLETED, report="bypassed constructor",
        llm_calls=-10,
    )
    _, run, attempt = _execute(tmp_path, result)
    assert run.workflow.tasks[0].status == TaskStatus.FAILED
    assert attempt.error.code == ErrorCode.CONTRACT_ERROR
    assert attempt.error.retryable is False
    assert run.llm_calls_used == 0


@pytest.mark.parametrize("calls", [True, "3", -1])
def test_raw_usage_must_be_a_nonnegative_integer(tmp_path, calls):
    raw = AgentResult(status="completed", report="Claimed usage").model_dump()
    raw["llm_calls"] = calls
    _, run, attempt = _execute(tmp_path, raw)
    assert run.workflow.tasks[0].status == TaskStatus.FAILED
    assert attempt.error.code == ErrorCode.CONTRACT_ERROR
    assert run.llm_calls_used == 0


@pytest.mark.parametrize("status", [ModuleStatus.FAILED, ModuleStatus.BLOCKED, ModuleStatus.NEEDS_USER_INPUT])
def test_failure_and_pause_use_shared_envelope(tmp_path, status):
    if status == ModuleStatus.NEEDS_USER_INPUT:
        result = AgentResult(
            status=status, report="Need an answer", session=_session(SessionStatus.PAUSED),
            artifacts=[], control=question_signal("session_boundary", 'Which metric?', ['metric']),
        )
    else:
        result = AgentResult(
            status=status, report="Cannot execute",
            error=ModuleError(code=ErrorCode.TOOL_FAILED, message="Original error", retryable=False),
        )
    _, run, attempt = _execute(tmp_path, result)
    assert run.workflow.tasks[0].status.value == status.value
    assert attempt.report.startswith(result.report)
    if status != ModuleStatus.NEEDS_USER_INPUT:
        assert attempt.error.code == ErrorCode.TOOL_FAILED


@pytest.mark.parametrize("failed", [False, True])
def test_prepare_failure_publishes_no_prefix_and_preserves_original_result_facts(tmp_path, failed):
    original = ModuleError(
        code=ErrorCode.TOOL_FAILED, message="Training crashed", retryable=True,
        details={"stderr_tail": "NameError: totla"},
    ) if failed else None
    session = _session(SessionStatus.FAILED if failed else SessionStatus.COMPLETED)
    result = AgentResult(
        status=ModuleStatus.FAILED if failed else ModuleStatus.COMPLETED_WITH_WARNINGS,
        report="Training failed" if failed else "Training finished",

        error=original, session=session, llm_calls=9,
        warnings=[WarningRecord(code="partial_output", message="Only partial evidence available")],
        artifacts=[
            ArtifactCandidate(kind="experiment_result", path="diagnosis.txt", media_type="text/plain",
                              summary="Available diagnostic", content="NameError: totla"),
            ArtifactCandidate(kind="experiment_result", path="missing.json", media_type="application/json",
                              summary="Missing file"),
        ],
    )
    original_result = result.model_dump(mode="json")
    engine, run, attempt = _execute(tmp_path, result, charged_calls=9)
    assert result.model_dump(mode="json") == original_result
    assert run.workflow.tasks[0].status == TaskStatus.FAILED
    assert run.llm_calls_used == 9
    assert attempt.session == session
    assert attempt.report == result.report
    assert attempt.warnings == run.workflow.tasks[0].warnings == result.warnings
    assert attempt.error.retryable is False
    # The missing second file aborts preparation before even the valid inline prefix is published.
    assert attempt.artifact_ids == []
    assert set(run.artifacts) == {attempt.report_artifact_id}
    report = run.artifacts[attempt.report_artifact_id]
    assert report.kind == "module_report"
    assert Path(url2pathname(urlparse(report.uri).path)).read_text() == result.report
    assert report.metadata["output_artifact_ids"] == []
    assert report.metadata["agent_error"] == original_result["error"]
    assert report.metadata["warnings"] == original_result["warnings"]
    assert not run.pending_registrations and not run.pending_module_results
    if failed:
        assert attempt.error.code == original.code
        assert attempt.error.message == original.message
        assert attempt.error.details["stderr_tail"] == "NameError: totla"
        assert "artifact path" in attempt.error.details["artifact_registration_error"]
    else:
        assert attempt.error.code == ErrorCode.ARTIFACT_MISSING
    assert attempt.reception_error == attempt.error
    persisted = engine.run_until_stable(run.run_id)
    assert persisted.llm_calls_used == 9
    assert persisted.workflow.tasks[0].warnings == result.warnings
    restored = persisted.workflow.tasks[0].attempts[0]
    assert restored.report == result.report and restored.session == session
    assert restored.error == attempt.error and restored.warnings == result.warnings
    assert restored.artifact_ids == [] and persisted.artifacts == run.artifacts


@pytest.mark.parametrize("missing_output", [False, True])
def test_failed_experiment_keeps_execution_record_when_candidate_is_missing(
    tmp_path, monkeypatch, missing_output,
):
    """Native failure cleanup preserves command receipts through registration."""
    import shlex
    import subprocess
    import sys
    from pathlib import Path

    from sciagent_contracts import WorkspaceAccess, WorkspaceSpec
    from sciagent_experiment import NativeExperimentAgent
    from sciagent_orchestrator.handoffs import read_json, system_artifact
    from sciagent_runtime import ScriptedLLMClient

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text(
        'import sys\nprint("training failed", file=sys.stderr)\nsys.exit(7)\n'
    )

    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "add", "train.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
         "commit", "-qm", "baseline"], cwd=workspace, check=True,
    )

    command = f"{shlex.quote(sys.executable)} train.py"
    agent = NativeExperimentAgent(ScriptedLLMClient(tool_turns([
        {"tool": "run_shell", "arguments": {"command": command}},
        {"tool": "finish", "arguments": {
            "status": "failed", "report": "Training failed before producing metrics",
            "artifacts": [{
                "kind": "data", "path": "absent.json", "media_type": "application/json",
                "summary": "Requested but absent output",
            }] if missing_output else [],
        }},
    ])))
    results = []

    class Port:
        def invoke(self, request):
            result = agent.invoke(request)
            results.append(result)
            return result

    engine = WorkflowScheduler(
        bindings=executor_bindings({"experiment": ModuleBinding(owner=AgentOwner.EXECUTOR, port=Port())}),
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "data", workspaces={"ws_test": WorkspaceSpec(
            workspace_id="ws_test", source_kind="local", location=str(workspace),
            access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
        )},
    )
    now = datetime.now(UTC)
    run = ResearchRun(
        run_id="run_boundary", status="running",
        request=ResearchRequest(
            goal="Run the provided experiment", permissions=RunPermissions(execute_commands=True),
            budget=RunBudget(max_llm_calls=2, timeout_seconds=30),
            execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=2),
        ), workspaces=engine._resolve_workspaces("run_boundary"),
        created_at=now, updated_at=now,
    )
    engine.store.save(run)
    accept_proposal(engine, run.run_id, WorkflowProposal(
        work_request_id="work_boundary", tasks=[TaskProposal(execution_ref="experiment@1",
            id="task_boundary", work_request_id="work_boundary",
            workflow_agent_kind="executor", instruction="Run train.py and report the outcome",
        )],
    ))
    run = engine.run_until_stable(run.run_id)
    assert run.status == RunStatus.PAUSED
    assert len(results) == 1 and results[0].status == ModuleStatus.NEEDS_USER_INPUT
    question = run.pending_question
    assert question is not None
    paused_session_id = run.workflow.tasks[0].attempts[0].session.id
    assert not any(item.kind == "execution_record" for item in run.artifacts.values())
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, answered_at=datetime.now(UTC),
        run_id=run.run_id, task_id="task_boundary", attempt_number=1, session_id=paused_session_id,
        action=question.action,
    )
    run.answers.append(answer)
    run.pending_question = None
    run.status = RunStatus.RUNNING
    engine.resume_task_in_place(run, "task_boundary")
    engine.store.save(run)
    run = engine.run_until_stable(run.run_id)
    task = run.workflow.tasks[0]
    assert len(task.attempts) == 1
    assert len(results) == 2
    attempt = task.attempts[0]
    result = results[-1]
    assert attempt.session.id == paused_session_id
    state = agent.loop.store.load(attempt.session.id)
    assert task.status == result.status == state.status == "failed"
    expected_code = ErrorCode.BUDGET_EXHAUSTED if missing_output else ErrorCode.AGENT_REPORTED_FAILURE
    assert attempt.error.code == expected_code
    assert attempt.error == result.error
    assert attempt.session == result.session
    assert attempt.report == result.report
    assert run.llm_calls_used == state.llm_calls_used == sum(item.llm_calls for item in results) == 2
    assert result.llm_calls == 1
    assert attempt.error.retryable is False
    if missing_output:
        assert "artifact_path_missing" in state.tool_turns[-1].model_dump_json()
        assert "absent.json" in state.tool_turns[-1].model_dump_json()
    observations = [event.data["value"] for event in state.events
                    if event.type == "observation" and event.tool == "run_shell"
                    and isinstance(event.data.get("value"), dict)
                    and "exit_code" in event.data["value"]]
    assert len(observations) == 1
    assert observations[0]["exit_code"] == 7
    assert observations[0]["stderr_tail"].strip() == "training failed"
    assert len(result.artifacts) == 1
    assert len(attempt.artifact_ids) == 1
    assert attempt.report_artifact_id not in attempt.artifact_ids
    assert {item.kind for item in run.artifacts.values()} == {"module_report", "execution_record"}
    records = [run.artifacts[key] for key in attempt.artifact_ids
               if run.artifacts[key].kind == "execution_record"]
    assert len(records) == 1
    record = records[0]
    assert record.kind == result.artifacts[0].kind == "execution_record"
    rows = read_json(record)["results"]
    assert len(rows) == 1
    assert rows[0]["command"] == command
    assert rows[0]["exit_code"] == 7
    assert rows[0]["stderr_path"] == observations[0]["stderr_path"]
    assert Path(rows[0]["stderr_path"]).read_text().strip() == "training failed"
    assert engine.load(run.run_id).artifacts[record.id] == record


@pytest.mark.parametrize("status", [ModuleStatus.FAILED, ModuleStatus.BLOCKED])
def test_sessionless_first_scientific_failure_does_not_keep_reserved_session(tmp_path, status):
    error = ModuleError(code=ErrorCode.TOOL_FAILED, message="Original failure before a Session started", retryable=False)
    result = AgentResult(status=status, report="Cannot execute", error=error)
    _, run, attempt = _execute(tmp_path, result, WorkflowAgentKind.SCIENTIFIC)
    assert run.workflow.tasks[0].status.value == status.value
    assert attempt.session is None
    assert attempt.error == error
    assert attempt.reception_error is None
