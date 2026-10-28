from e2e.scheduler_fixtures import create_scheduler_run

from e2e.native_fixtures import tool_turns

from sciagent_contracts import AgentPermissions, RunPermissions, ExecutionLimits
import json
from pathlib import Path
import subprocess
import shlex
import sys
from urllib.parse import urlparse
from urllib.request import url2pathname

from datetime import UTC, datetime

from sciagent_coding import NativeCodingAgent
from sciagent_contracts import (
    AgentOwner,
    WorkflowAgentKind,
    ModuleStatus,
    AgentRequest,
    ResearchRequest,
    RunBudget,
    RunStatus,
    TaskBudget,
    TaskProposal,
    TaskStatus,
    WorkflowProposal,
    WorkspaceGrant,
    WorkspaceAccess,
    WorkspaceSourceKind,
    WorkspaceSpec,
)
from sciagent_orchestrator import (
    InMemoryRunStore,
    ModuleBinding,
    WorkflowScheduler,
)
from sciagent_runtime import ScriptedLLMClient



def _create_run(engine, run_id, request, proposal):
    return create_scheduler_run(engine, run_id, request, proposal, resolve_workspaces=True)


def test_scheduler_registers_native_coding_artifacts(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "util.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)

    coding = NativeCodingAgent(
        ScriptedLLMClient(
            tool_turns([
                {
                    "tool": "replace_text",
                    "arguments": {
                        "path": "util.py",
                        "old_text": "VALUE = 1",
                        "new_text": "VALUE = 2",
                    },
                },
                {
                    "tool": "run_shell",
                    "arguments": {"command": f"{shlex.quote(sys.executable)} -m py_compile util.py"},
                },
                {
                    "tool": "finish",
                    "arguments": {"report": "Updated VALUE", "artifacts": [{
                        "kind": "code_change", "path": "util.py", "media_type": "text/x-python",
                        "summary": "Updated source", "output_name": "source",
                    }]},
                },
            ])
        )
    )
    scheduler = WorkflowScheduler(bindings={WorkflowAgentKind.EXECUTOR: ModuleBinding(owner=AgentOwner.EXECUTOR, port=coding)}, store=InMemoryRunStore(), artifact_root=tmp_path / 'artifacts', data_root=tmp_path / 'data', workspaces={'ws_main': WorkspaceSpec(workspace_id='ws_main', source_kind=WorkspaceSourceKind.LOCAL, location=str(repo), access=WorkspaceAccess(read_paths=['.'], write_paths=['.']))})
    request = ResearchRequest(goal='Update one constant', budget=RunBudget(max_llm_calls=10, timeout_seconds=30), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1))
    proposal = WorkflowProposal(
        work_request_id="work_legacy_initial",
        tasks=[
            TaskProposal(
                id="task_code_native",
                work_request_id="work_legacy_initial",
                workflow_agent_kind=WorkflowAgentKind.EXECUTOR, execution_ref="coding@1",
                instruction="Change VALUE from 1 to 2",
            )
        ],
    )

    _create_run(scheduler, "run_native_e2e", request, proposal)
    run = scheduler.run_until_stable("run_native_e2e")
    assert run.status == RunStatus.PAUSED
    question = run.pending_question
    from sciagent_contracts import RecordedAnswer
    run.answers.append(RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, answered_at=datetime.now(UTC),
        run_id=run.run_id, session_id=question.session_id,
        task_id=question.task_id, attempt_number=question.attempt_number,
        action=question.action,
    ))
    run.pending_question = None
    run.status = RunStatus.RUNNING
    scheduler.resume_task_in_place(run, "task_code_native")
    scheduler.store.save(run)
    run = scheduler.run_until_stable(run.run_id)

    assert run.workflow.tasks[0].status == TaskStatus.COMPLETED
    assert {artifact.kind for artifact in run.artifacts.values()} == {
        "module_report",
        "code_patch",
        "code_change",
        "execution_record",
    }
    assert all(len(artifact.sha256) == 64 for artifact in run.artifacts.values())
    source = next(ref for ref in run.artifacts.values() if ref.kind == "code_change")
    assert source.output_name == "source"
    assert Path(url2pathname(urlparse(source.uri).path)).read_text() == "VALUE = 2\n"
    record = next(ref for ref in run.artifacts.values() if ref.kind == "execution_record")
    commands = json.loads(Path(url2pathname(urlparse(record.uri).path)).read_text())["results"]
    assert len(commands) == 1
    assert "py_compile util.py" in commands[0]["command"]
    assert commands[0]["exit_code"] == 0


def test_coding_resume_preserves_attempt_baseline(tmp_path, monkeypatch) -> None:
    """An edit made before a pause must stay attributed to the same Attempt.

    On resume the Git baseline is restored from persisted Session memory, never
    re-snapshotted, so the pre-pause edit is not mistaken for the Attempt's
    starting state (ADR-0011 §2).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "util.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)

    coding = NativeCodingAgent(
        ScriptedLLMClient(
            tool_turns([
                {
                    "tool": "replace_text",
                    "arguments": {
                        "path": "util.py",
                        "old_text": "VALUE = 1",
                        "new_text": "VALUE = 2",
                    },
                },
                {
                    "tool": "ask_user",
                    "arguments": {
                        "text": "Confirm the edit?",
                        "requested_fields": ["approve"],
                    },
                },
                {
                    "tool": "run_shell",
                    "arguments": {"command": f"{shlex.quote(sys.executable)} -m py_compile util.py"},
                },
                {
                    "tool": "finish",
                    "arguments": {"report": "Updated VALUE", "artifacts": [{
                        "kind": "code_change", "path": "util.py", "media_type": "text/x-python",
                        "summary": "Updated source", "output_name": "source",
                    }]},
                },
            ])
        )
    )

    request = AgentRequest(work_request_id="work_direct", run_id='run_resume', task_id='task_code', attempt_number=1, agent=AgentOwner.EXECUTOR, instruction='Change VALUE from 1 to 2', budget=TaskBudget(max_llm_calls=20, timeout_seconds=60), workspace=WorkspaceGrant(root=str(repo), source=WorkspaceSourceKind.LOCAL, access=WorkspaceAccess(read_paths=['.'], write_paths=['.'])), workspace_id='ws_main', workspace_spec=WorkspaceSpec(workspace_id='ws_main', source_kind=WorkspaceSourceKind.LOCAL, location=str(repo), access=WorkspaceAccess(read_paths=['.'], write_paths=['.'])), output_dir=str(tmp_path / 'out'), permissions=AgentPermissions(execute_commands=True))

    first = coding.invoke(request)
    assert first.status == ModuleStatus.NEEDS_USER_INPUT

    from sciagent_contracts import RecordedAnswer
    question = first.control.question
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text, requested_fields=question.requested_fields,
        options=question.options, values={"approve": "yes"}, answered_at=datetime.now(UTC),
        run_id=question.run_id, session_id=question.session_id,
        task_id=question.task_id, attempt_number=question.attempt_number,
    )
    resume_request = request.model_copy(update={
        "parent_session_id": first.session.id, "control_ack": first.control.origin,
        "user_answers": [answer], "resume_answers": [answer], "input_ids": [answer.question_id],
    })
    approval = coding.invoke(resume_request)
    assert approval.status == ModuleStatus.NEEDS_USER_INPUT
    question = approval.control.question
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, answered_at=datetime.now(UTC),
        run_id=question.run_id, session_id=question.session_id,
        task_id=question.task_id, attempt_number=question.attempt_number,
        action=question.action,
    )
    second = coding.invoke(resume_request.model_copy(update={
        "resume_answers": [answer], "control_ack": approval.control.origin,
    }))

    assert second.status == ModuleStatus.COMPLETED
    assert [item.path for item in second.artifacts if item.kind == "code_change"] == ["util.py"]
