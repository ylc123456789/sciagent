"""Failed Coding calls retain their facts when optional Git diagnostics fail."""

import json
import subprocess

import pytest

from sciagent_coding import NativeCodingAgent
from sciagent_components import GitWorkspace, GitWorkspaceError, ResourceLayout
from sciagent_contracts import (
    AgentPermissions, AgentRequest, ErrorCode, ModuleStatus, SessionStatus,
    TaskBudget, WorkspaceAccess, WorkspaceGrant, task_session_id,
)
from sciagent_runtime import (
    InMemorySessionStore, LLMExhaustedError, NativeToolCall, ToolCallTurn,
)
from sciagent_runtime.budget import execution_budget


@pytest.mark.parametrize("failure", [
    "loop_deadline", "diagnostic_deadline", "diagnostic_git", "diagnostic_process",
    "diagnostic_unicode", "diagnostic_unexpected", "diagnostic_file",
])
def test_native_failure_keeps_session_usage_and_edits(tmp_path, monkeypatch, failure):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "value.py").write_text("VALUE = 1\n")
    clock = [0.0]

    class Client:
        tool_session_key = "failure-diagnostic-test"
        calls = 0

        def next_tool_call(self, context, schemas, turns, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ToolCallTurn(tool_calls=[NativeToolCall(
                    id="edit", name="replace_text",
                    arguments=json.dumps({
                        "path": "value.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2",
                    }),
                )])
            if failure == "loop_deadline":
                clock[0] = 31.0
                return ToolCallTurn(tool_calls=[NativeToolCall(
                    id="finish", name="finish", arguments='{"report": "Changed value"}',
                )])
            raise LLMExhaustedError("No further model response")

    if failure != "loop_deadline":
        original = GitWorkspace.write_patch_since

        def failed_diagnostic(repository, baseline, path):
            if failure == "diagnostic_git":
                raise GitWorkspaceError("Git diagnostic unavailable")
            if failure == "diagnostic_unicode":
                raise UnicodeDecodeError("utf-8", b"\xac", 0, 1, "invalid start byte")
            if failure == "diagnostic_unexpected":
                raise RuntimeError("unexpected diagnostic failure")
            if failure == "diagnostic_file":
                raise OSError("patch file unavailable")
            if failure == "diagnostic_process":
                def process_timeout(*args, **kwargs):
                    raise subprocess.TimeoutExpired("git", 30)
                monkeypatch.setattr("sciagent_components.git.run_process", process_timeout)
            else:
                clock[0] = 31.0
            return original(repository, baseline, path)

        monkeypatch.setattr(GitWorkspace, "write_patch_since", failed_diagnostic)

    store = InMemorySessionStore()
    client = Client()
    request = AgentRequest(work_request_id="work_test",
        run_id="run_diagnostics", task_id="task_diagnostics", attempt_number=1,
        agent="executor", instruction="Change VALUE from 1 to 2",
        budget=TaskBudget(max_llm_calls=4, timeout_seconds=30),
        permissions=AgentPermissions(),
        output_dir=str(tmp_path / "outputs"),
        workspace=WorkspaceGrant(
            root=str(repo), source="local",
            access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
        ),
    )
    agent = NativeCodingAgent(
        client, store=store, resource_layout=ResourceLayout.from_env(data_root=tmp_path / "data"),
    )
    with execution_budget(max_llm_calls=4, timeout_seconds=30, clock=lambda: clock[0]) as budget:
        result = agent.invoke(request)
        assert result.llm_calls == budget.usage.used == client.calls == 2

    state = store.load(task_session_id(request.run_id, request.task_id, 1))
    assert result.status == ModuleStatus.FAILED
    assert result.session.id == state.session_id
    assert result.session.status == state.status == SessionStatus.FAILED
    assert state.llm_calls_used == 2
    assert state.terminal_result is not None
    assert state.terminal_result.report == result.report
    assert state.terminal_result.error == result.error
    assert state.terminal_result.llm_calls == result.llm_calls
    assert (repo / "value.py").read_text() == "VALUE = 2\n"
    original_error = next(event.data for event in reversed(state.events) if event.type == "error")
    assert result.error.code.value == original_error["code"]
    assert result.error.message == original_error["message"]
    if failure == "loop_deadline":
        assert result.error.code == ErrorCode.TIMEOUT
    assert result.error.retryable is False
    assert "diagnostic_patch_error" in result.error.details
    assert not any(item.kind == "code_patch" for item in result.artifacts)
