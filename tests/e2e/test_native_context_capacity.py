"""Real execution Agent assembly preserves reads in paired history at capacity."""

import hashlib
import json
import subprocess

import pytest

from e2e.native_fixtures import tool_turns
from sciagent_components import ResourceLayout
from sciagent_contracts import (
    AgentPermissions, AgentRequest, ArtifactRef, ErrorCode, TaskBudget, WorkspaceAccess, WorkspaceGrant,
)
from sciagent_coding import NativeCodingAgent
from sciagent_experiment import NativeExperimentAgent
from sciagent_runtime import ContextComposer, ScriptedLLMClient
from sciagent_runtime.tool_calling import native_input, native_input_text


def _body(label, character):
    prefix, suffix = f"{label}_BEGIN\n", f"\n{label}_END"
    return prefix + character * (128_000 - len(prefix) - len(suffix)) + suffix


FILE_BODY = _body("FILE", "x")
ARTIFACT_BODY = _body("ARTIFACT", "y")


def _agent_with_reads(tmp_path, kind, *, limit=256_000, artifact_first=False):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "train.py").write_text(FILE_BODY)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "train.py"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "baseline"], cwd=root, check=True)
    frozen = tmp_path / "source.txt"
    frozen.write_text(ARTIFACT_BODY)
    ref = ArtifactRef(
        id="artifact_context", kind="code_patch", producer="executor", run_id="run_capacity",
        task_id="task_prior", attempt_number=1, uri=frozen.as_uri(),
        sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(), media_type="text/plain", summary="Prior patch",
    )
    reads = [
        {"tool": "read_file", "arguments": {"path": "train.py"}},
        {"tool": "read_artifact", "arguments": {"artifact_id": ref.id}},
    ]
    if artifact_first:
        reads.reverse()
    client = ScriptedLLMClient(tool_turns([
        *reads, *[{"tool": "list_files", "arguments": {"path": "."}} for _ in range(8)],
        {"tool": "ask_user", "arguments": {"text": "Proceed?", "requested_fields": ["choice"]}},
    ]))
    agent = (NativeCodingAgent if kind == "coding" else NativeExperimentAgent)(
        client, max_context_tokens=limit,
        resource_layout=ResourceLayout(resource_root=tmp_path / "resources"),
    )
    request = AgentRequest(work_request_id="work_direct",
        run_id="run_capacity", task_id="task_capacity", attempt_number=1, agent="executor",
        instruction="Review source and evidence", input_artifacts=[ref],
        workspace=WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(read_paths=["."], write_paths=[])),
        workspace_id="ws_capacity", permissions=AgentPermissions(),
        output_dir=str(tmp_path / "outputs") if kind == "coding" else None,
        budget=TaskBudget(max_llm_calls=12, timeout_seconds=60),
    )
    return agent, client, request


def _wire(client, index=-1, limit=256_000):
    context, schemas, history = client.contexts[index], client.schemas[index], client.histories[index]
    assert context.included_sections == ["system", "task", "runtime_authority", "material_index"]
    assert FILE_BODY not in context.text and ARTIFACT_BODY not in context.text
    for turn in history:
        assert set(turn.tool_results) == {call.id for call in turn.tool_calls}
    measured = ContextComposer.estimate_tokens(native_input_text(context, schemas, history))
    assert measured == context.estimated_tokens
    assert measured <= limit
    wire = native_input(context, schemas, history)
    assert wire["messages"][0]["role"] == "system"
    assert wire["messages"][1]["role"] == wire["messages"][-1]["role"] == "user"
    assert wire["messages"][-1]["content"] == context.material_index
    tree = json.loads(context.material_index.split("\n", 1)[1])
    nodes, indexed_ids, filenames = [tree], [], []
    while nodes:
        node = nodes.pop()
        nodes.extend(node["children"])
        if node["type"] == "artifact":
            indexed_ids.append(node["location"]["artifact_id"])
        elif node["type"] == "file":
            filenames.append(node["name"])
    assert indexed_ids == ["artifact_context"]
    assert tree["artifact_count"] == 1
    assert "train.py" in filenames
    return wire


@pytest.mark.parametrize("kind", ["coding", "experiment"])
@pytest.mark.parametrize("artifact_first", [False, True])
def test_native_default_context_keeps_both_full_read_histories(tmp_path, kind, artifact_first):
    agent, client, request = _agent_with_reads(tmp_path, kind, artifact_first=artifact_first)
    result = agent.invoke(request)
    assert result.status == "needs_user_input", result.report
    assert len(client.contexts) == 11
    wire = _wire(client)
    receipts = [json.loads(message["content"]) for message in wire["messages"] if message["role"] == "tool"]
    reads = [receipt["value"] for receipt in receipts
             if isinstance(receipt.get("value"), dict) and "content" in receipt["value"]]
    assert sorted(read["content"] for read in reads) == sorted([FILE_BODY, ARTIFACT_BODY])
    assert all(read["truncated"] is False for read in reads)
    assert len(reads) == 2


@pytest.mark.parametrize("kind", ["coding", "experiment"])
@pytest.mark.parametrize("limit", [1, 64_000])
def test_explicit_capacity_never_silently_shrinks_reads_or_expands_the_limit(tmp_path, kind, limit):
    agent, client, request = _agent_with_reads(tmp_path, kind, limit=limit)
    result = agent.invoke(request)
    assert result.status == "failed", result.report
    assert result.error.code == ErrorCode.BUDGET_EXHAUSTED
    assert agent.max_context_tokens == limit
    assert result.llm_calls == len(client.contexts) == (0 if limit == 1 else 2)
    if client.contexts:
        _wire(client, limit=limit)
        read = client.histories[-1][0]
        assert json.loads(read.tool_results[read.tool_calls[0].id])["value"]["content"] == FILE_BODY


@pytest.mark.parametrize("kind", ["coding", "experiment"])
def test_model_capacity_is_checked_before_sending_an_oversized_read_history(tmp_path, kind):
    agent, client, request = _agent_with_reads(tmp_path, kind)
    limits = []

    def tool_input_limit(limit):
        limits.append(limit)
        return 32_000

    client.tool_input_limit = tool_input_limit
    result = agent.invoke(request)
    assert result.status == "failed", result.report
    assert result.error.code == ErrorCode.BUDGET_EXHAUSTED
    assert result.llm_calls == len(client.contexts) == 1
    assert all(limit == 256_000 for limit in limits)
    state = agent.loop.store.load(result.session.id)
    read = state.tool_turns[0]
    assert json.loads(read.tool_results[read.tool_calls[0].id])["value"]["content"] == FILE_BODY
    _wire(client, limit=32_000)
