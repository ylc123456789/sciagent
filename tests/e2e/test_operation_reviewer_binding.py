"""Execution Agents inject the caller-selected reviewer into their shared loop."""

import subprocess

import pytest

from resagent2_coding import NativeCodingAgent
from resagent2_experiment import NativeExperimentAgent
from resagent2_contracts import AgentPermissions, AgentRequest, TaskBudget, WorkspaceAccess, WorkspaceGrant
from resagent2_runtime import ScriptedLLMClient


class DefinitionCaptured(Exception):
    pass


@pytest.mark.parametrize("agent_type,owner", [(NativeCodingAgent, "coding"),
                                            (NativeExperimentAgent, "experiment")])
@pytest.mark.parametrize("with_reviewer", [False, True])
def test_agent_definition_receives_exact_reviewer_instance(tmp_path, monkeypatch, agent_type, owner, with_reviewer):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=test", "-c",
                    "user.email=test@example.com", "commit", "--allow-empty", "-qm", "baseline"], check=True)
    reviewer = object() if with_reviewer else None
    agent = agent_type(ScriptedLLMClient([]), operation_reviewer=reviewer)
    request = AgentRequest(work_request_id="work_direct",
        run_id="run_binding", task_id="task_binding", attempt_number=1, agent="executor",
        execution_definition=agent.preset_definition,
        instruction="Inspect existing information", budget=TaskBudget(max_llm_calls=2, timeout_seconds=30),
        permissions=AgentPermissions(execute_commands=False, approval_mode="auto_review"),
        workspace=WorkspaceGrant(root=str(tmp_path), source="local", access=WorkspaceAccess(
            read_paths=["."], write_paths=["."],
        )), workspace_id="ws_binding",
        output_dir=str(tmp_path / "outputs") if agent_type is NativeCodingAgent else None,
    )

    def capture(definition, actual_request, **kwargs):
        assert definition.operation_reviewer is reviewer
        assert actual_request == request
        assert actual_request.permissions.approval_mode == "auto_review"
        assert not actual_request.permissions.execute_commands
        raise DefinitionCaptured

    monkeypatch.setattr(agent.loop, "run", capture)
    with pytest.raises(DefinitionCaptured):
        agent.invoke(request)
