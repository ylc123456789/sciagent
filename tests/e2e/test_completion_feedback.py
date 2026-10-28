"""Native completion feedback keeps one invocation, Attempt and Session."""

from e2e.scheduler_fixtures import accept_proposal

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from resagent2_components import RegisteredArtifactReader
from resagent2_coding import NativeCodingAgent
from resagent2_experiment import NativeExperimentAgent
from resagent2_contracts import (
    AgentOwner, ArtifactCandidate, ErrorCode, ExecutionLimits, ResearchRequest, RunBudget,
    RunPermissions, TaskProposal, WorkflowProposal, WorkspaceAccess, WorkspaceSpec,
)
from resagent2_orchestrator import (
    ArtifactRegistry, InMemoryRunStore, ModuleBinding, ResearchRun, WorkflowScheduler,
)
from resagent2_runtime import NativeToolCall, ToolCallTurn
from resagent2_runtime.tool_calling import native_input


def output(path="metrics.json", **fields):
    return dict(kind="data", path=path, summary="Recorded results",
                media_type="application/json", **fields)


class NativeClient:
    tool_session_key = "completion-feedback-test"

    def __init__(self, submissions, *, prepare=None, statuses=None):
        self.submissions = submissions
        self.prepare = prepare
        self.statuses = statuses or ["completed"] * len(submissions)
        self.contexts = []
        self.wires = []

    def next_tool_call(self, context, schemas, turns, **kwargs):
        index = len(self.contexts)
        self.contexts.append(context)
        self.wires.append(native_input(context, schemas, turns))
        if self.prepare:
            self.prepare(index)
        return ToolCallTurn(tool_calls=[NativeToolCall(
            id=f"finish_{index}", name="finish", arguments=json.dumps({
                "status": self.statuses[index],
                "report": "Analysis ready", "artifacts": self.submissions[index],
            }),
        )])


def execute(tmp_path, kind, submissions, *, budget=8, prepare=None, statuses=None, downstream=False, input_refs=()):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    (workspace / "metrics.json").write_text('{"accuracy": 0.9}')
    for command in [
        ["git", "add", "metrics.json"],
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
         "commit", "-qm", "baseline"],
    ]:
        subprocess.run(command, cwd=workspace, check=True)
    client = NativeClient(submissions, prepare=prepare, statuses=statuses)
    agent = (NativeCodingAgent if kind == "coding" else NativeExperimentAgent)(client)
    requests = []

    class Port:
        def invoke(self, request):
            requests.append(request)
            return agent.invoke(request)

    scheduler = WorkflowScheduler(
        bindings={"executor": ModuleBinding(owner=AgentOwner.EXECUTOR, port=Port())},
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "data", workspaces={"ws_test": WorkspaceSpec(
            workspace_id="ws_test", source_kind="local", location=str(workspace),
            access=WorkspaceAccess(read_paths=["."], write_paths=[]),
        )},
    )
    now = datetime.now(UTC)
    run = ResearchRun(
        run_id="run_completion_feedback", status="running",
        request=ResearchRequest(
            goal="Analyze the provided metrics", permissions=RunPermissions(),
            budget=RunBudget(max_llm_calls=budget, timeout_seconds=30),
            execution_limits=ExecutionLimits(max_tasks=2 if downstream else 1, max_attempts_per_task=2),
        ), workspaces=scheduler._resolve_workspaces("run_completion_feedback"),
        artifacts={ref.id: ref for ref in input_refs},
        created_at=now, updated_at=now,
    )
    scheduler.store.save(run)
    accept_proposal(scheduler, run.run_id, WorkflowProposal(
        work_request_id="work_1", tasks=[TaskProposal(
            id="task_analysis", work_request_id="work_1",
            execution_ref=f"{kind}@1", workflow_agent_kind="executor", instruction="Analyze the provided metrics",
            input_artifacts=[ref.id for ref in input_refs],
        )] + ([TaskProposal(
            id="task_dependent", work_request_id="work_1",
            execution_ref=f"{kind}@1", workflow_agent_kind="executor", instruction="Use the analysis",
            depends_on=["task_analysis"],
        )] if downstream else []),
    ))
    final = scheduler.run_until_stable(run.run_id)
    task = final.workflow.tasks[0]
    assert len(requests) == len(task.attempts) == 1
    attempt = task.attempts[0]
    state = agent.loop.store.load(attempt.session.id)
    assert (state.run_id, state.task_id, state.attempt_number) == (run.run_id, task.id, 1)
    assert final.llm_calls_used == state.llm_calls_used == len(client.contexts)
    return final, task, state, client


@pytest.mark.parametrize("kind", ["coding", "experiment"])
@pytest.mark.parametrize("mistake,code", [
    ([output("wrong-name.json")], "artifact_path_missing"),
    ([output(output_name="scores"), output(output_name="scores")], "duplicate_output_name"),
])
def test_finish_correction_reaches_registration_in_same_attempt(tmp_path, kind, mistake, code):
    run, task, state, client = execute(tmp_path, kind, [mistake, [output()]])
    assert task.status == state.status == "completed"
    assert len(client.contexts) == 2
    assert code in json.dumps(client.wires[1])
    first_receipt = json.loads(state.tool_turns[0].tool_results["finish_0"])
    assert first_receipt["ok"] is False
    assert first_receipt["value"]["completion_check"] == "rejected"
    assert code in first_receipt["summary"]
    assert len(state.tool_turns) == 2
    assert all(len(turn.tool_results) == 1 for turn in state.tool_turns)
    refs = [run.artifacts[key] for key in task.attempts[0].artifact_ids]
    assert len(refs) == 1
    assert Path(refs[0].uri.removeprefix("file://")).read_text() == '{"accuracy": 0.9}'
    assert refs[0].metadata["source_path"] == "metrics.json"


@pytest.mark.parametrize("kind", ["coding", "experiment"])
def test_output_directory_and_registry_use_the_same_source(tmp_path, kind):
    # The native finalizer must recognize the scheduler-controlled attempt root.
    def prepare(index):
        # Obtain the actual root from the existing layout, never guess its shape.
        from resagent2_orchestrator.layout import RunLayout
        attempt_dir = RunLayout(tmp_path / "data").attempt_dir(
            "run_completion_feedback", "task_analysis", 1,
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text("{}")
    run, task, _, _ = execute(tmp_path, kind, [[output("result.json")]], prepare=prepare)
    assert task.status == "completed"
    ref = run.artifacts[task.attempts[0].artifact_ids[0]]
    assert ref.metadata["source_root"] == "output_dir"


@pytest.mark.parametrize("kind", ["coding", "experiment"])
def test_invalid_finish_keeps_feedback_when_budget_exhausts(tmp_path, kind):
    run, task, state, client = execute(tmp_path, kind, [[output("absent.json")]], budget=1)
    assert task.status == state.status == "failed"
    assert task.attempts[0].error.code == ErrorCode.BUDGET_EXHAUSTED
    receipt = json.loads(state.tool_turns[0].tool_results["finish_0"])
    assert receipt["ok"] is False
    assert "artifact_path_missing" in receipt["summary"]
    assert not task.attempts[0].artifact_ids
    assert len(state.tool_turns) == 1
    assert "absent.json" in state.tool_turns[0].tool_calls[0].arguments


@pytest.mark.parametrize("kind", ["coding", "experiment"])
def test_escaped_candidate_does_not_become_correctable_success(tmp_path, kind):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    def prepare(index):
        (tmp_path / "workspace" / "escape.json").symlink_to(outside)
    run, task, state, client = execute(
        tmp_path, kind, [[output("escape.json")], [output()]], prepare=prepare,
    )
    assert task.status == state.status == "failed"
    assert task.attempts[0].error.code == ErrorCode.CONTRACT_ERROR
    assert "outside" in task.attempts[0].error.message
    assert len(client.contexts) == 1
    assert not task.attempts[0].artifact_ids


@pytest.mark.parametrize("kind", ["coding", "experiment"])
def test_agent_declared_failure_keeps_evidence_and_blocks_dependencies(tmp_path, kind):
    run, task, state, client = execute(
        tmp_path, kind, [[output()]], statuses=["failed"], downstream=True,
    )
    assert task.status == state.status == "failed"
    assert task.attempts[0].error.code == ErrorCode.AGENT_REPORTED_FAILURE
    assert task.attempts[0].error.retryable is False
    assert task.attempts[0].report == "Analysis ready"
    assert len(client.contexts) == 1
    dependent = run.workflow.tasks[1]
    assert dependent.status == "blocked"
    assert not dependent.attempts
    refs = [run.artifacts[key] for key in task.attempts[0].artifact_ids]
    assert any(ref.metadata.get("source_path") == "metrics.json" for ref in refs)


@pytest.mark.parametrize("kind", ["coding", "experiment"])
def test_failed_finish_still_corrects_invalid_artifacts_in_same_session(tmp_path, kind):
    run, task, state, client = execute(
        tmp_path, kind, [[output("missing.json")], [output()]],
        statuses=["failed", "failed"],
    )
    assert task.status == state.status == "failed"
    assert task.attempts[0].error.code == ErrorCode.AGENT_REPORTED_FAILURE
    assert len(client.contexts) == 2
    assert "artifact_path_missing" in json.dumps(client.wires[1])
    assert json.loads(state.tool_turns[0].tool_results["finish_0"])["ok"] is False
    assert len(state.tool_turns) == 2
    assert all(len(turn.tool_results) == 1 for turn in state.tool_turns)
    assert len(task.attempts[0].artifact_ids) == 1


@pytest.mark.parametrize("kind", ["coding", "experiment"])
@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["canonical-child-ref", "rewritten-child-ref"])
def test_registered_child_output_is_corrected_before_session_completion(tmp_path, kind, status, rewrite):
    registry = ArtifactRegistry(tmp_path / "child-artifacts")
    child = registry.register(ArtifactCandidate(kind="data", path="child.json",
        media_type="application/json", summary="Original child evidence", content='{"value": 7}'),
        grant=None, producer=AgentOwner.EXECUTOR, run_id="run_completion_feedback",
        task_id="task_child", attempt_number=1, index=1)
    returned = child.model_dump(mode="json")
    if rewrite:
        returned.update(uri="artifact://" + child.id, summary="Rewritten by parent", metadata={})
    run, task, state, client = execute(tmp_path, kind,
        [[returned], [output()]], statuses=[status, status], input_refs=[child])

    assert len(client.contexts) == 2
    first = json.loads(state.tool_turns[0].tool_results["finish_0"])
    assert first["ok"] is False
    assert "completion_check" not in (first.get("value") or {})
    assert child.id in json.dumps(client.wires[1])
    assert task.status == state.status == status
    attempt = task.attempts[0]
    assert attempt.reception_error is None
    assert child.id not in attempt.artifact_ids
    assert run.artifacts[child.id] == child
    RegisteredArtifactReader([child], run_id=run.run_id).verify(child.id)
    assert len(attempt.artifact_ids) == 1
    assert run.artifacts[attempt.artifact_ids[0]].task_id == task.id
