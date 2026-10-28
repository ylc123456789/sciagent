"""Real Agent model requests expose delegation only inside the granted boundary."""

import subprocess

import pytest
from pydantic import ValidationError

from e2e.native_fixtures import tool_turns
from resagent2_coding import NativeCodingAgent
from resagent2_components import ResourceLayout
from resagent2_contracts import (
    AgentPermissions, AgentRequest, ModuleStatus, TaskBudget,
    WorkspaceAccess, WorkspaceGrant,
)
from resagent2_experiment import NativeExperimentAgent
from resagent2_runtime import ScriptedLLMClient
from resagent2_scientific import ScientificAgent


@pytest.mark.parametrize("owner", ["scientific", "coding", "experiment"])
@pytest.mark.parametrize(
    "depth,delegation_granted",
    [pytest.param(2, True, id="permitted"),
     pytest.param(2, False, id="capability-disabled"),
     pytest.param(3, False, id="maximum-depth")],
)
def test_real_agent_model_tools_respect_delegation_boundary(
    tmp_path, owner, depth, delegation_granted,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if owner == "coding":
        subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    arguments = {
        "text": "Which dataset should this task use?",
        "requested_fields": ["dataset"],
    }
    if owner == "scientific":
        arguments["assessment"] = {"statement": "A dataset decision is required"}
    client = ScriptedLLMClient(tool_turns([{"tool": "ask_user", "arguments": arguments}]))
    if owner == "scientific":
        agent = ScientificAgent(client)
    else:
        agent_type = NativeCodingAgent if owner == "coding" else NativeExperimentAgent
        agent = agent_type(client, resource_layout=ResourceLayout(resource_root=tmp_path / "resources"))
    request = AgentRequest(
        run_id="run_tool_visibility", work_request_id="work_visibility",
        task_id="task_" + owner, attempt_number=1,
        agent="scientific" if owner == "scientific" else "executor",
        instruction="Ask the direct upstream for the dataset decision before proceeding",
        work_depth=depth, max_work_depth=3,
        permissions=AgentPermissions(request_work=delegation_granted),
        budget=TaskBudget(max_llm_calls=2, timeout_seconds=30),
        workspace=WorkspaceGrant(root=str(workspace), source="local",
            access=WorkspaceAccess(read_paths=["."])),
        output_dir=str(tmp_path / "output"),
    )
    if depth == request.max_work_depth:
        # Scheduler removes the grant at this depth. The public contract rejects
        # a caller which keeps it; Agents must not silently reinterpret that request.
        invalid = request.model_copy(update={"permissions": AgentPermissions(request_work=True)})
        with pytest.raises(ValidationError, match="maximum work depth"):
            AgentRequest.model_validate(invalid)
        assert client.schemas == []

    result = agent.invoke(request)

    assert result.status == ModuleStatus.NEEDS_USER_INPUT, result.report
    assert len(client.schemas) == 1
    names = [schema["function"]["name"] for schema in client.schemas[0]]
    assert len(names) == len(set(names))
    assert ("request_work" in names) is delegation_granted
    # Reaching a delegation boundary must preserve the other common controls.
    assert {"ask_user", "finish", "update_work", "abandon_work"} <= set(names)
