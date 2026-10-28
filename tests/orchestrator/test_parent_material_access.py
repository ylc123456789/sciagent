"""Parent inspection of completed results and mutable files during active work."""
from dataclasses import replace
from sciagent_executor import default_execution_registry
from e2e.scheduler_fixtures import executor_bindings
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from e2e.native_fixtures import tool_turns
from sciagent_contracts import (
    AgentOwner, ResearchRequest, RunBudget, RunPermissions, TaskProposal,
    WorkflowAgentRegistry, WorkflowProposal,
    WorkspaceAccess, WorkspaceSpec, task_session_id,
)
from sciagent_components import (
    ReadOnlyWorkspaceBoundary, RegisteredArtifactReader, ResourceLayout, WorkspacePermissionError,
)
from sciagent_coding import NativeCodingAgent
from sciagent_experiment import NativeExperimentAgent
from sciagent_orchestrator import (
    DeterministicWorkflowCompiler, DeterministicWorkInterpreter, InMemoryRunStore,
    ModuleBinding, ResearchController, WorkflowScheduler,
)
from sciagent_runtime import (
    AgentDefinition, AgentLoop, AllowListPermissionPolicy, AskUserTool,
    ApprovalReviewVerdict, CompletionDecision, FinishTool, InMemorySessionStore, ScriptedLLMClient,
)
from sciagent_runtime.models import ContextSection
from sciagent_components.materials import request_task_context
from sciagent_scientific import ScientificAgent


def action(name, **arguments):
    return {"tool": name, "arguments": arguments}


def active_work_node(context):
    tree = json.loads(context.material_index.split("\n", 1)[1])
    registered = next(node for node in tree["children"] if node["name"] == "registered_materials")
    works = next(node for node in registered["children"] if node["name"] == "work_requests")
    return next(node for node in works["children"] if node["status"] == "executing")


def active_work_id(context):
    authority = json.loads(context.runtime_context.split("\n", 1)[1])
    return authority["child_work"]["active_work_request_id"]


class Completion:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=candidate is not None,
            report=candidate.report if candidate else "", artifacts=candidate.artifacts if candidate else [])


class Child:
    def __init__(self, client):
        self.client = client
        self.requests = []
        self.store = InMemorySessionStore()
        self.loop = AgentLoop(store=self.store)
        tools = (AskUserTool(), FinishTool())
        self.definition = AgentDefinition(
            name="coding", owner=AgentOwner.EXECUTOR, system_prompt="Follow upstream requirements",
            tools=tools, llm_client=client,
            context_builder=lambda request, state, limit: [
                ContextSection(name="task", content=json.dumps(request_task_context(request))),
                ContextSection(name="material_index", content='{"artifacts":[]}')],
            permission_policy=AllowListPermissionPolicy({tool.name for tool in tools}),
            completion_check=Completion(),
        )

    def invoke(self, request):
        self.requests.append(request)
        definition = replace(self.definition,
            execution_ref=request.execution_definition.ref,
            execution_config_hash=request.execution_definition.config_hash)
        return self.loop.run(definition, request,
            session_id=task_session_id(request.run_id, request.task_id, request.attempt_number))


def test_parent_reads_completed_sibling_and_live_path_before_work_finishes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    live = workspace / "partial.csv"
    live.write_text("sequence,value\n0,7\n")
    parent_contexts = []
    artifact_ids = []
    child_question_ids = []

    class ParentClient(ScriptedLLMClient):
        def next_tool_call(self, context, *args, **kwargs):
            stage = len(self.contexts)
            parent_contexts.append(context)
            if stage == 1:
                active = active_work_node(context)
                assert active["status"] == "executing", context.material_index
                assert [task["status"] for task in active["children"]] == ["completed", "needs_user_input"], context.material_index
                entry = next(item for item in active["children"][0]["children"] if item["metadata"]["kind"] == "metrics")
                artifact_ids.append(entry["location"]["artifact_id"])
                questions = json.loads(context.pending_notifications)
                assert len(questions) == 1 and str(live) in questions[0]["text"]
                child_question_ids.append(questions[0]["id"])
                assert "partial.csv" not in context.task_context, context.task_context
                assert "active_work_navigation" not in context.task_context, context.task_context
                assert "accuracy" not in context.task_context, context.task_context
                assert "work_results" not in context.task_context, context.task_context
                self._turns[0] = tool_turns([action("read_artifact", artifact_id=artifact_ids[0])])[0]
            elif stage == 3:
                self._turns[0] = tool_turns([action("update_work", work_request_id=active_work_id(context), answers=[{
                    "task_id": "task_inspect", "question_id": child_question_ids[0],
                    "values": {"decision": "Continue with the partial data only"}}])])[0]
            return super().next_tool_call(context, *args, **kwargs)

    work = action("request_work", assessment={"statement": "Need two steps"}, work_request={
        "objective": "Measure then inspect", "tasks": [
            {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "measure", "instruction": "Measure"},
            {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "inspect", "instruction": "Inspect", "depends_on": ["measure"]}]})
    finish = action("finish", report="Inspected the original measurement and current partial file",
        artifacts=[{"kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json",
            "summary": "Conclusion", "content": {"verdict": "inconclusive", "statement": "Partial data only",
                "limitations": ["Unfinished data is not a complete experiment"]}}])
    parent_client = ParentClient(tool_turns([work, action("read_artifact", artifact_id="placeholder"),
        action("read_file", path=str(live)), action("update_work", work_request_id="placeholder"), finish]))
    child_client = ScriptedLLMClient(tool_turns([
        action("finish", report="Original measurement", artifacts=[{"kind": "metrics", "path": "metrics.json",
            "media_type": "application/json", "summary": "Measured accuracy", "content": '{"accuracy":0.8}'}]),
        action("ask_user",
            text=f"Please compare the completed result with the unfinished file {live}", requested_fields=["decision"]),
        action("finish", report="Used parent's clarification"),
    ]))
    child = Child(child_client)
    scheduler = WorkflowScheduler(bindings=executor_bindings({"coding": ModuleBinding(owner=AgentOwner.EXECUTOR, port=child)}),
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data",
        workspaces={"ws_source": WorkspaceSpec(workspace_id="ws_source", source_kind="local", location=str(workspace),
            access=WorkspaceAccess(read_paths=["."], write_paths=["."]))})
    scientific = ScientificAgent(parent_client)
    controller = ResearchController(scientific_port=scientific, scheduler=scheduler,
        registry=default_execution_registry(),
        compiler=DeterministicWorkflowCompiler(WorkflowProposal(work_request_id="work_fixture", tasks=[
            TaskProposal(execution_ref="coding@1", id="task_measure", work_request_id="work_fixture", workflow_agent_kind="executor", instruction="Measure"),
            TaskProposal(execution_ref="coding@1", id="task_inspect", work_request_id="work_fixture", workflow_agent_kind="executor", instruction="Inspect", depends_on=["task_measure"]),
        ]), patch=None), interpreter=DeterministicWorkInterpreter())
    run = controller.create_run("run_parent_materials", ResearchRequest(goal="Measure then inspect", permissions=RunPermissions(),
        budget=RunBudget(max_llm_calls=20, timeout_seconds=60)))
    assert run.status == "completed", (run.terminal_error, [(task.id, task.status) for task in run.workflow.tasks])
    assert len(child.requests) == 3
    assert child.requests[1].task_id == child.requests[2].task_id == "task_inspect"
    assert child.requests[2].parent_session_id == task_session_id(run.run_id, "task_inspect", 1)
    assert len(run.task("task_inspect").attempts) == 1
    assert child_question_ids[0] == child.requests[2].user_answers[0].question_id
    assert "Continue with the partial data only" in child_client.contexts[-1].task_context
    state = scientific.store.load(run.scientific_session.id)
    reads = [json.loads(next(iter(turn.tool_results.values()))) for turn in state.tool_turns
        if turn.tool_calls and turn.tool_calls[0].name in {"read_artifact", "read_file"}]
    assert all(item["ok"] for item in reads)
    assert "accuracy" in str(reads[0]) and "sequence,value" in str(reads[1])
    assert not any(ref.metadata.get("path") == str(live) for ref in run.artifacts.values())
    assert RegisteredArtifactReader([run.artifacts[artifact_ids[0]]], run_id=run.run_id).read_text(artifact_ids[0])["content"] == '{"accuracy":0.8}'
    assert "partial.csv" in parent_contexts[1].material_index
    assert "sequence,value" not in parent_contexts[1].material_index



class AllowFixtureCommand:
    def review(self, action, request):
        return ApprovalReviewVerdict(decision="allow", reason="Deterministic local fixture write")


@pytest.mark.parametrize("kind", ["coding", "experiment"])
@pytest.mark.parametrize("location", ["workspace", "task_output"])
def test_native_parent_reads_current_file_and_replies_in_original_attempt(tmp_path, kind, location):
    workspace = tmp_path / "source"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    (workspace / "seed.txt").write_text("original source\n")
    subprocess.run(["git", "-C", str(workspace), "add", "seed.txt"], check=True)
    subprocess.run(["git", "-C", str(workspace), "-c", "user.name=test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "baseline"], check=True)
    other = tmp_path / "other_source"
    other.mkdir()
    (other / "public").mkdir()
    (other / "public/reference.txt").write_text("second workspace")
    (other / "public/denied.txt").write_text("private second workspace")
    (other / "ungranted.txt").write_text("outside original read scope")
    requests, parent_requests, live_paths, question_ids, metric_ids = [], [], [], [], []
    answer_text = "Use only rows already present; do not infer missing samples."

    class NativeClient(ScriptedLLMClient):
        def next_tool_call(self, context, *args, **kwargs):
            stage = len(self.contexts)
            if stage == 1:
                request = requests[-1]
                target = (Path(request.workspace.root) if location == "workspace" else Path(request.output_dir)) / "partial.csv"
                live_paths.append(target)
                code = f"from pathlib import Path; p=Path({str(target)!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('row,value\\n0,7\\n')"
                self._turns[0] = tool_turns([action("run_shell", command=shlex.quote(sys.executable) + " -c " + shlex.quote(code))])[0]
            elif stage == 2:
                self._turns[0] = tool_turns([action("ask_user", text=f"The unfinished file is {live_paths[0]}. Which rows should I use?",
                    requested_fields=["decision"])])[0]
            elif stage == 3:
                assert answer_text in context.task_context, context.task_context
                code = f"from pathlib import Path; p=Path({str(live_paths[0])!r}); p.write_text(p.read_text() + '1,8\\n')"
                self._turns[0] = tool_turns([action("run_shell", command=shlex.quote(sys.executable) + " -c " + shlex.quote(code))])[0]
            return super().next_tool_call(context, *args, **kwargs)

    client = NativeClient(tool_turns([
        action("finish", report="Original A report.  \n", artifacts=[{"kind": "metrics", "path": "a_metrics.json",
            "media_type": "application/json", "summary": "Original frozen A measurement", "content": '{"accuracy":0.8}'}]),
        action("run_shell", command="placeholder"),
        action("ask_user", text="placeholder", requested_fields=["decision"]),
        action("run_shell", command="placeholder"), action("finish", report="B continued from the same unfinished file"),
    ]))
    native_class = {"coding": NativeCodingAgent, "experiment": NativeExperimentAgent}[kind]

    class RecordingNative(native_class):
        def invoke(self, request):
            requests.append(request.model_copy(deep=True))
            return super().invoke(request)

    native = RecordingNative(client, operation_reviewer=AllowFixtureCommand(),
        resource_layout=ResourceLayout(resource_root=tmp_path / "resources"))

    class ParentClient(ScriptedLLMClient):
        def next_tool_call(self, context, *args, **kwargs):
            stage = len(self.contexts)
            if stage == 1:
                active = active_work_node(context)
                assert active["status"] == "executing", active
                assert [item["status"] for item in active["children"]] == ["completed", "needs_user_input"], active
                metric = next(item for item in active["children"][0]["children"] if item["metadata"]["kind"] == "metrics")
                metric_ids.append(metric["location"]["artifact_id"])
                question = json.loads(context.pending_notifications)[0]
                question_ids.append(question["id"])
                assert str(live_paths[0]) in question["text"]
                assert "partial.csv" not in context.task_context
                assert "partial.csv" in context.material_index
                self._turns[0] = tool_turns([action("read_artifact", artifact_id=metric_ids[0])])[0]
            elif stage == 2:
                self._turns[0] = tool_turns([action("read_file", path=str(live_paths[0]))])[0]
            elif stage == 3:
                self._turns[0] = tool_turns([action("update_work", work_request_id=active_work_id(context), answers=[{
                    "task_id": "task_b", "question_id": question_ids[0], "values": {"decision": answer_text}}])])[0]
            return super().next_tool_call(context, *args, **kwargs)

    work = action("request_work", assessment={"statement": "Need measurements and a current-file decision"}, work_request={
        "objective": "Inspect A and B original materials", "tasks": [
            {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "a", "instruction": "Return the frozen measurement", "workspace_id": "ws_source"},
            {"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "b", "instruction": "Write a partial file, ask its parent, then continue", "depends_on": ["a"], "workspace_id": "ws_source"}]})
    parent_client = ParentClient(tool_turns([work, action("read_artifact", artifact_id="placeholder"),
        action("read_file", path="placeholder"), action("update_work", work_request_id="placeholder"),
        action("finish", report="Read A frozen metrics and B current data before answering", artifacts=[{
            "kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json", "summary": "Limited conclusion",
            "content": {"verdict": "inconclusive", "statement": "Partial data only", "limitations": ["No complete experiment"]}}])]))

    class RecordingScientific(ScientificAgent):
        def invoke(self, request):
            parent_requests.append(request.model_copy(deep=True))
            return super().invoke(request)

    scientific = RecordingScientific(parent_client, resource_layout=ResourceLayout(resource_root=tmp_path / "resources"))
    scheduler = WorkflowScheduler(bindings=executor_bindings({kind: ModuleBinding(owner=AgentOwner.EXECUTOR, port=native)}),
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data", workspaces={
            "ws_source": WorkspaceSpec(workspace_id="ws_source", source_kind="local", location=str(workspace),
                access=WorkspaceAccess(read_paths=["."], write_paths=["."])),
            "ws_other": WorkspaceSpec(workspace_id="ws_other", source_kind="local", location=str(other),
                access=WorkspaceAccess(read_paths=["public"], write_paths=[], denied_paths=["public/denied.txt"])),
        })
    controller = ResearchController(scientific_port=scientific, scheduler=scheduler,
        registry=default_execution_registry(),
        compiler=DeterministicWorkflowCompiler(WorkflowProposal(work_request_id="work_fixture", tasks=[
            TaskProposal(execution_ref=kind + "@1", id="task_a", work_request_id="work_fixture", workflow_agent_kind="executor", workspace_id="ws_source",
                instruction="Return the frozen measurement"),
            TaskProposal(execution_ref=kind + "@1", id="task_b", work_request_id="work_fixture", workflow_agent_kind="executor", workspace_id="ws_source",
                instruction="Write a partial file, ask its parent, then continue", depends_on=["task_a"]),
        ]), patch=None), interpreter=DeterministicWorkInterpreter())
    run = controller.create_run("run_native_parent", ResearchRequest(goal="Inspect A and B original materials",
        permissions=RunPermissions(execute_commands=True, approval_mode="auto_review"),
        budget=RunBudget(max_llm_calls=30, timeout_seconds=120)))
    assert run.status == "completed", (run.terminal_error, [(task.status, task.attempts) for task in run.workflow.tasks])
    assert len(requests) == 3
    assert requests[1].task_id == requests[2].task_id == "task_b"
    session_id = task_session_id(run.run_id, "task_b", 1)
    assert requests[2].parent_session_id == session_id
    assert len(run.task("task_b").attempts) == 1
    assert requests[2].user_answers[0].question_id == question_ids[0]
    state = native.loop.store.load(session_id)
    assert [turn.tool_calls[0].name for turn in state.tool_turns] == ["run_shell", "ask_user", "run_shell", "finish"]
    assert live_paths[0].exists(), [turn.tool_results for turn in state.tool_turns]
    assert live_paths[0].read_text() == "row,value\n0,7\n1,8\n"
    parent_state = scientific.store.load(run.scientific_session.id)
    assert [turn.tool_calls[0].name for turn in parent_state.tool_turns if turn.tool_calls] == [
        "request_work", "read_artifact", "read_file", "update_work", "finish"]
    receipts = [json.loads(next(iter(turn.tool_results.values()))) for turn in parent_state.tool_turns if turn.tool_calls]
    assert "accuracy" in receipts[1]["value"]["content"]
    assert receipts[2]["value"]["content"] == "row,value\n0,7\n"
    assert all(item["ok"] for item in receipts)
    assert not any(ref.metadata.get("source_path") == str(live_paths[0]) or ref.metadata.get("path") == str(live_paths[0])
        for ref in run.artifacts.values())

    # Controller roots come from actual saved workspaces and started Attempts.
    inspection = parent_requests[1]
    roots = {Path(grant.root) for grant in inspection.read_workspaces}
    assert roots == {workspace, other, Path(requests[0].output_dir), Path(requests[1].output_dir)}
    assert all(not grant.access.write_paths for grant in inspection.read_workspaces)
    assert scheduler.run_layout.run_dir(run.run_id) not in roots
    boundary = ReadOnlyWorkspaceBoundary(inspection.read_workspaces)
    assert boundary.resolve_read_file(str(other / "public/reference.txt")).read_text() == "second workspace"
    assert boundary.resolve_read_file(str(live_paths[0])) == live_paths[0]
    unknown = tmp_path / "unknown" / "private.txt"
    unknown.parent.mkdir()
    unknown.write_text("unknown root")
    private = scheduler.run_layout.state_dir(run.run_id) / "private.txt"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("run private state")
    other_task = scheduler.run_layout.attempt_dir(run.run_id, "task_unrelated", 1) / "private.txt"
    other_task.parent.mkdir(parents=True)
    other_task.write_text("unrelated Task")
    assert boundary.resolve_read_file(str(workspace / ".git/HEAD")) == workspace / ".git/HEAD"
    for path in (other / "public/denied.txt", other / "ungranted.txt", unknown, private, other_task):
        with pytest.raises(WorkspacePermissionError):
            boundary.resolve_read_file(str(path))
    (workspace / "escape.txt").symlink_to(private)
    with pytest.raises(WorkspacePermissionError, match="escapes workspace root"):
        boundary.resolve_read_file(str(workspace / "escape.txt"))

    # Once the delegation ends, only original Run workspace grants remain.
    assert {Path(grant.root) for grant in controller._scientific_read_workspaces(run)} == {workspace, other}
