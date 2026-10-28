"""Real Coding preparation does not execute repository-selected helpers."""

import json
import shlex
import subprocess

import pytest

from sciagent_coding import NativeCodingAgent
from sciagent_components import ResourceLayout
from sciagent_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ModuleStatus, TaskBudget,
    WorkspaceAccess, WorkspaceGrant, WorkspaceSpec,
)
from sciagent_runtime import ScriptedLLMClient


@pytest.mark.parametrize("source_kind", ["local", "copy"])
@pytest.mark.parametrize("helper", ["clean", "process", "hook", "fsmonitor"])
def test_readonly_coding_does_not_execute_repository_helpers(tmp_path, source_kind, helper):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "payload.txt").write_text("synthetic payload\n")
    subprocess.run(["git", "-C", str(source), "add", "payload.txt"], check=True)
    subprocess.run([
        "git", "-C", str(source), "-c", "user.email=test@example.com",
        "-c", "user.name=test", "commit", "-qm", "initial",
    ], check=True)
    original_index = (source / ".git/index").read_bytes()
    original_head = (source / ".git/HEAD").read_bytes()
    marker = tmp_path / (helper + "-ran")
    command = "printf ran > " + shlex.quote(str(marker)) + "; cat"
    if helper in {"clean", "process"}:
        (source / ".gitattributes").write_text("payload.txt filter=probe\n")
        option, value = f"filter.probe.{helper}", command
        subprocess.run(["git", "-C", str(source), "config", "filter.probe.required", "true"], check=True)
    elif helper == "hook":
        hooks = tmp_path / "hooks"
        hooks.mkdir()
        hook = hooks / "post-index-change"
        hook.write_text("#!/bin/sh\n" + command + "\n")
        hook.chmod(0o755)
        option, value = "core.hooksPath", str(hooks)
    else:
        script = tmp_path / "fsmonitor.sh"
        script.write_text("#!/bin/sh\n" + command + "\n")
        script.chmod(0o755)
        option, value = "core.fsmonitor", str(script)
    subprocess.run(["git", "-C", str(source), "config", option, value], check=True)
    access = WorkspaceAccess(read_paths=["."], write_paths=[])
    destination = source if source_kind == "local" else tmp_path / "copy"
    request = AgentRequest(work_request_id="work_test",
        run_id="run_filter_" + source_kind, task_id="task_inspect", attempt_number=1,
        agent=AgentOwner.EXECUTOR, instruction="Read-only inspection; execute no commands.",
        budget=TaskBudget(max_llm_calls=2, timeout_seconds=20),
        permissions=AgentPermissions(execute_commands=False),
        workspace=WorkspaceGrant(root=str(destination), source=source_kind, access=access),
        workspace_id="ws_inspect",
        workspace_spec=WorkspaceSpec(workspace_id="ws_inspect", source_kind=source_kind,
                                     location=str(source), access=access),
        output_dir=str(tmp_path / "outputs"),
    )
    client = ScriptedLLMClient([{"tool_calls": [{
        "id": "finish_only", "name": "finish",
        "arguments": json.dumps({"report": "Read-only inspection complete"}),
    }]}])
    agent = NativeCodingAgent(client, resource_layout=ResourceLayout(resource_root=tmp_path / "resources"))
    original_files = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    result = agent.invoke(request)
    if helper in {"clean", "process"}:
        assert result.status == ModuleStatus.BLOCKED
        assert "external clean/process" in result.report
        assert result.error is not None and not result.error.retryable
    else:
        assert result.status == ModuleStatus.COMPLETED, result.report
    assert not marker.exists()
    assert (source / ".git/index").read_bytes() == original_index
    assert (source / ".git/HEAD").read_bytes() == original_head
    assert original_files == {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
