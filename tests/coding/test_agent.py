"""Unified Coding invocation, workspace ownership and operation authorization."""

from e2e.native_fixtures import tool_turns

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sciagent_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ErrorCode, ModuleStatus,
    TaskBudget, WorkspaceGrant, WorkspaceAccess, WorkspaceSourceKind,
)
from sciagent_coding import NativeCodingAgent
from sciagent_runtime import AgentAction, ScriptedLLMClient


def init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    (root / "util.py").write_text("def add(a, b):\n    return a + b\n")
    subprocess.run(["git", "add", "util.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)


def request(root: Path, *, writable=False, task_id="task_coding", **updates) -> AgentRequest:
    values = dict(work_request_id="work_test", run_id='run_coding', task_id=task_id, attempt_number=1, agent=AgentOwner.EXECUTOR, instruction='Inspect add and make only needed changes', budget=TaskBudget(max_llm_calls=8, timeout_seconds=30), workspace=WorkspaceGrant(root=str(root), source=WorkspaceSourceKind.LOCAL, access=WorkspaceAccess(read_paths=['.'], write_paths=['.']) if writable else WorkspaceAccess(read_paths=['.'], write_paths=[])), workspace_id='ws_test', output_dir=str(root.parent / 'output'))
    values["permissions"] = AgentPermissions(execute_commands=True)
    values.update(updates)
    return AgentRequest(**values)


def finish(report="add is implemented in util.py", **kwargs):
    return {"tool": "finish", "arguments": {"report": report, **kwargs}}


def edit():
    return {"tool": "replace_text", "arguments": {
        "path": "util.py", "old_text": "def add(a, b):",
        "new_text": "def add(a, b):\n    \"\"\"Return the sum.\"\"\"",
    }}


@pytest.mark.parametrize("writable", [False, True])
def test_same_protocol_can_explain_without_edits(tmp_path, writable):
    init_repo(tmp_path)
    before = (tmp_path / "util.py").read_text()
    client = ScriptedLLMClient(tool_turns([
        {"tool": "read_file", "arguments": {"path": "util.py"}}, finish(),
    ]))
    result = NativeCodingAgent(client).invoke(request(tmp_path, writable=writable))
    assert result.status == ModuleStatus.COMPLETED
    assert result.report == "add is implemented in util.py"
    assert (tmp_path / "util.py").read_text() == before
    assert not any(item.kind == "code_patch" for item in result.artifacts)


def test_read_only_can_deliver_report_content(tmp_path):
    init_repo(tmp_path)
    artifact = {"kind": "module_report", "path": "explanation.md",
                "media_type": "text/markdown", "summary": "Code explanation",
                "content": "The function returns a + b."}
    result = NativeCodingAgent(ScriptedLLMClient(tool_turns([finish(artifacts=[artifact])]))).invoke(request(tmp_path))
    assert result.status == ModuleStatus.COMPLETED
    assert any(item.kind == "module_report" and item.content == artifact["content"] for item in result.artifacts)
    assert not (tmp_path / "explanation.md").exists()


def test_edit_uses_same_finish_and_derives_patch(tmp_path):
    init_repo(tmp_path)
    result = NativeCodingAgent(ScriptedLLMClient(tool_turns([edit(), finish("Added a docstring; tests not run")]))).invoke(
        request(tmp_path, writable=True),
    )
    assert result.status == ModuleStatus.COMPLETED
    patch = next(item for item in result.artifacts if item.kind == "code_patch")
    assert "Return the sum" in (tmp_path.parent / "output" / patch.path).read_text()
    assert not any(item.kind == "code_change" for item in result.artifacts)
    assert not any(item.kind == "verification_result" for item in result.artifacts)


def test_new_file_is_delivered_when_explicitly_submitted(tmp_path):
    init_repo(tmp_path)
    result = NativeCodingAgent(ScriptedLLMClient(tool_turns([
        {"tool": "create_file", "arguments": {"path": "new.py", "content": "VALUE = 1\n"}},
        finish("Added a constant", artifacts=[{
            "kind": "code_change", "path": "new.py", "media_type": "text/x-python",
            "summary": "Reusable constant", "output_name": "source", "metadata": {"purpose": "deliverable"},
        }]),
    ]))).invoke(request(tmp_path, writable=True))
    assert result.status == ModuleStatus.COMPLETED
    source = next(item for item in result.artifacts if item.kind == "code_change")
    assert source.path == "new.py" and source.output_name == "source"
    assert source.media_type == "text/x-python" and source.summary == "Reusable constant"
    assert source.metadata == {"purpose": "deliverable"}
    assert sum(item.kind == "code_patch" for item in result.artifacts) == 1


def test_two_tasks_isolate_changed_files(tmp_path):
    init_repo(tmp_path)
    first = NativeCodingAgent(ScriptedLLMClient(tool_turns([edit(), finish()]))).invoke(request(tmp_path, writable=True))
    assert first.status == ModuleStatus.COMPLETED
    second = NativeCodingAgent(ScriptedLLMClient(tool_turns([
        {"tool": "create_file", "arguments": {"path": "second.py", "content": "X = 2\n"}}, finish(),
    ]))).invoke(request(tmp_path, writable=True, task_id="task_second"))
    assert not any(item.kind == "code_change" for item in second.artifacts)
    patch = (tmp_path.parent / "output" / next(
        item.path for item in second.artifacts if item.kind == "code_patch")).read_text()
    assert "second.py" in patch and "+X = 2" in patch
    assert "Return the sum" not in patch


def test_read_only_inspects_shared_dirty_workspace(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "util.py").write_text("# previous accepted change\n")
    result = NativeCodingAgent(ScriptedLLMClient(tool_turns([
        {"tool": "read_file", "arguments": {"path": "./util.py"}}, finish(),
    ]))).invoke(request(tmp_path))
    assert result.status == ModuleStatus.COMPLETED


def test_write_action_has_one_schema_but_read_only_grant_denies_it(tmp_path):
    init_repo(tmp_path)
    AgentAction.model_validate(edit())
    result = NativeCodingAgent(ScriptedLLMClient(tool_turns([edit()]))).invoke(request(tmp_path))
    assert result.status == ModuleStatus.FAILED
    assert "Return the sum" not in (tmp_path / "util.py").read_text()


def test_removed_verification_tool_does_not_execute(tmp_path):
    init_repo(tmp_path)
    result = NativeCodingAgent(ScriptedLLMClient(tool_turns([
        {"tool": "run_verification", "arguments": {"commands": ["touch unauthorized"]}},
    ]))).invoke(request(tmp_path, writable=True))
    assert result.status == ModuleStatus.FAILED
    assert not (tmp_path / "unauthorized").exists()


def test_process_permission_is_enforced(tmp_path):
    init_repo(tmp_path)
    result = NativeCodingAgent(ScriptedLLMClient(tool_turns([{'tool': 'run_shell', 'arguments': {'command': 'python -m pytest'}}]))).invoke(request(tmp_path, writable=True, permissions=AgentPermissions(execute_commands=False)))
    assert result.status == ModuleStatus.FAILED


def test_failed_call_retains_diagnostic_patch(tmp_path):
    init_repo(tmp_path)
    result = NativeCodingAgent(ScriptedLLMClient(tool_turns([edit()]))).invoke(request(tmp_path, writable=True))
    assert result.status == ModuleStatus.FAILED
    patch = next(item for item in result.artifacts if item.kind == "code_patch")
    assert patch.metadata["diagnostic"]
    assert "Return the sum" in (tmp_path.parent / "output" / patch.path).read_text()
    assert result.error.retryable is False


@pytest.mark.parametrize("kind", ["execution_record", "observation_trace", "code_patch"])
def test_model_cannot_fabricate_system_record(tmp_path, kind):
    init_repo(tmp_path)
    result = NativeCodingAgent(ScriptedLLMClient(tool_turns([finish(artifacts=[{
        "kind": kind, "path": "fake.json", "summary": "fake pass",
        "media_type": "application/json", "content": '{"passed": true}',
    }])]))).invoke(request(tmp_path))
    assert result.status == ModuleStatus.FAILED


def test_missing_workspace_is_blocked(tmp_path):
    result = NativeCodingAgent(ScriptedLLMClient([])).invoke(request(tmp_path, workspace=None))
    assert result.status == ModuleStatus.BLOCKED


def test_explicit_failure_keeps_one_patch_and_does_not_require_a_failed_command(tmp_path):
    init_repo(tmp_path)
    report = "Added documentation, but the requested behavior still needs implementation."
    agent = NativeCodingAgent(ScriptedLLMClient(tool_turns([
        edit(), finish(report, status="failed"),
    ])))
    result = agent.invoke(request(tmp_path, writable=True))
    assert result.status == result.session.status == "failed"
    assert result.error.code == ErrorCode.AGENT_REPORTED_FAILURE
    assert result.error.retryable is False
    assert result.report == report
    patches = [item for item in result.artifacts if item.kind == "code_patch"]
    assert len(patches) == 1
    assert "Return the sum" in (tmp_path.parent / "output" / patches[0].path).read_text()
    assert not any(item.kind == "code_change" for item in result.artifacts)
    state = agent.loop.store.load(result.session.id)
    assert not any(event.tool == "run_verification" for event in state.events)


@pytest.mark.parametrize("preset", ["coding", "experiment"])
@pytest.mark.parametrize("ending", ["completed", "failed", "interrupted", "unconfirmed"])
def test_native_agents_return_actual_execution_record_on_normal_failed_and_interrupted_exit(
    tmp_path, monkeypatch, preset, ending,
):
    from sciagent_capabilities import RunShellTool
    from sciagent_experiment import NativeExperimentAgent
    from sciagent_runtime import PermissionDecision, ToolObservation
    from sciagent_runtime.cancellation import OperationCancelled

    init_repo(tmp_path)
    facts = dict(command="python long_task.py", exit_code=0, timed_out=False,
        stdout_path="actual.stdout", stderr_path="actual.stderr", duration_seconds=0.2)

    def execute(self, state, arguments):
        if ending in {"interrupted", "unconfirmed"}:
            (tmp_path / "partial.txt").write_text("Actual partial output")
            raise OperationCancelled("User stopped work", stopped=ending == "interrupted",
                details={**facts, "exit_code": None, "operation": "process", "pid": 123,
                         "files_retained": True})
        return ToolObservation(summary="Actual command outcome", value=facts)

    monkeypatch.setattr(RunShellTool, "execute", execute)
    monkeypatch.setattr("sciagent_capabilities.permissions.OperationPermissionPolicy.check",
                        lambda *_: PermissionDecision(outcome="allow"))
    client = ScriptedLLMClient(tool_turns([
        {"tool": "run_shell", "arguments": {"command": "python long_task.py"}},
        finish("Exact original report", status="failed" if ending == "failed" else "completed"),
    ]))
    agent = NativeCodingAgent(client) if preset == "coding" else NativeExperimentAgent(client)
    result = agent.invoke(request(tmp_path, writable=True, agent=AgentOwner.EXECUTOR))
    assert result.status == ("failed" if ending == "unconfirmed" else ending)
    records = [item for item in result.artifacts if item.kind == "execution_record"]
    assert len(records) == 1
    row = json.loads(records[0].content)["results"][0]
    assert row["command"] == "python long_task.py" and row["stdout_path"] == "actual.stdout"
    if ending in {"interrupted", "unconfirmed"}:
        assert (tmp_path / "partial.txt").read_text() == "Actual partial output"
        assert row["stop_confirmed"] == (ending == "interrupted")
        assert row["execution_status"] == ("interrupted" if ending == "interrupted" else "unknown")
        assert row["exit_code"] is None
    if ending == "unconfirmed":
        assert result.error.details["stop_confirmed"] is False
    state = agent.loop.store.load(result.session.id)
    assert all(len(turn.tool_calls) == len(turn.tool_results) for turn in state.tool_turns)


@pytest.mark.parametrize("preset", ["coding", "experiment"])
def test_native_model_interruption_without_command_does_not_create_execution(tmp_path, preset):
    from sciagent_experiment import NativeExperimentAgent
    from sciagent_runtime.cancellation import OperationCancelled

    init_repo(tmp_path)
    class InterruptedClient(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            raise OperationCancelled("User stopped model generation", details={"operation": "model"})
    client = InterruptedClient([])
    agent = NativeCodingAgent(client) if preset == "coding" else NativeExperimentAgent(client)
    result = agent.invoke(request(tmp_path, writable=True, agent=AgentOwner.EXECUTOR))
    assert result.status == "interrupted"
    assert not any(item.kind == "execution_record" for item in result.artifacts)


@pytest.mark.parametrize("preset", ["coding", "experiment"])
def test_native_question_wait_already_preserves_completed_command_snapshot(tmp_path, monkeypatch, preset):
    from sciagent_capabilities import RunShellTool
    from sciagent_experiment import NativeExperimentAgent
    from sciagent_runtime import PermissionDecision, ToolObservation
    from sciagent_contracts import VerificationResult

    init_repo(tmp_path)
    facts = dict(command="python prepare.py", exit_code=0, timed_out=False,
        stdout_path="prepare.stdout", stderr_path="prepare.stderr", duration_seconds=0.1)
    monkeypatch.setattr(RunShellTool, "execute", lambda *_: ToolObservation(summary="Command completed", value=facts))
    monkeypatch.setattr("sciagent_capabilities.permissions.OperationPermissionPolicy.check",
                        lambda *_: PermissionDecision(outcome="allow"))
    client = ScriptedLLMClient(tool_turns([
        {"tool": "run_shell", "arguments": {"command": "python prepare.py"}},
        {"tool": "ask_user", "arguments": {"text": "Choose the next step", "requested_fields": ["choice"]}},
    ]))
    agent = NativeCodingAgent(client) if preset == "coding" else NativeExperimentAgent(client)
    result = agent.invoke(request(tmp_path, writable=True, agent=AgentOwner.EXECUTOR))
    assert result.status == "needs_user_input"
    assert json.loads(result.artifacts[0].content) == {"results": [VerificationResult(**facts).model_dump(mode="json")]}
    agent.terminate_session(result.session.id, "User cancelled while waiting")
    assert agent.loop.store.load(result.session.id).status == "cancelled"


@pytest.mark.parametrize("preset", ["coding", "experiment"])
def test_native_execution_agent_can_create_and_explicitly_reuse_environment(tmp_path, preset):
    """Exercise actual shell, approvals and resume, without a lifecycle manager."""
    import shlex
    import sys
    from sciagent_experiment import NativeExperimentAgent
    from sciagent_contracts import RecordedAnswer

    init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".venv/\n")
    env = tmp_path / ".venv"
    target_python = env / "bin" / "python"
    create = f"{shlex.quote(sys.executable)} -m venv --without-pip .venv"
    code = "import pathlib,sys; pathlib.Path('.venv/actual-prefix.txt').write_text(sys.prefix)"
    reuse = f"{shlex.quote(str(target_python))} -c {shlex.quote(code)}"
    client = ScriptedLLMClient(tool_turns([
        {"tool": "run_shell", "arguments": {"command": create}},
        {"tool": "run_shell", "arguments": {"command": reuse}},
        finish("Created the requested environment and observed its actual interpreter prefix."),
    ]))
    agent = NativeCodingAgent(client) if preset == "coding" else NativeExperimentAgent(client)
    req = request(tmp_path, writable=True, agent=AgentOwner.EXECUTOR)
    result = agent.invoke(req)
    assert result.status == "needs_user_input"
    assert not env.exists(), "creating the environment must wait for exact approval"
    for index in range(2):
        assert result.status == "needs_user_input"
        question = result.control.question
        answer = RecordedAnswer(
            question_id=question.id, question_text=question.text,
            requested_fields=question.requested_fields, options=question.options,
            values={"approve": "yes"}, answered_at=datetime.now(UTC),
            run_id=req.run_id, task_id=req.task_id, attempt_number=req.attempt_number,
            session_id=result.session.id, action=question.action,
        )
        req = req.model_copy(update={
            "parent_session_id": result.session.id,
            "resume_answers": [answer], "control_ack": result.control.origin,
        })
        result = agent.invoke(req)
        if index == 0:
            assert target_python.exists()
            assert not (env / "actual-prefix.txt").exists()
    assert result.status == "completed", result.report
    assert Path((env / "actual-prefix.txt").read_text()).resolve() == env.resolve()
    rows = json.loads(next(item.content for item in result.artifacts
                           if item.kind == "execution_record"))["results"]
    assert [item["command"] for item in rows] == [create, reuse]
    assert all(item["exit_code"] == 0 for item in rows)
    names = {schema["function"]["name"] for schema in client.schemas[0]}
    assert "run_shell" in names
    assert names.isdisjoint({"prepare_environment", "run_setup", "audit_env", "run_verification"})
