from sciagent_executor import default_execution_registry

from e2e.native_fixtures import tool_turns

from sciagent_contracts import WorkspaceAccess, RunPermissions, ExecutionLimits
import json
"""Exercise real analysis artifacts across native Agent and controller boundaries."""

import subprocess
import json

from sciagent_components import (
    RegisteredArtifactReader,
    ResourceLayout,
)
from sciagent_contracts import (
    AgentOwner, WorkflowAgentKind, WorkflowAgentRegistry,
    ResearchRequest, RunBudget, RunStatus,
    FutureArtifactBinding, TaskProposal, WorkflowPatch, WorkspaceSourceKind, WorkspaceSpec,
)
from sciagent_coding import NativeCodingAgent
from sciagent_orchestrator import (
    DeterministicWorkInterpreter,
    CompilationResult, JsonRunStore, ModuleBinding, ResearchController,
    WorkflowScheduler,
)
from sciagent_runtime import JsonSessionStore, ScriptedLLMClient
from sciagent_scientific import ScientificAgent


def test_analysis_reaches_dependent_agent_and_scientific_through_frozen_artifact(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "util.py").write_text("VALUE = 42\n", encoding="utf-8")
    for args in [
        ["init", "-q"], ["add", "util.py"],
        ["-c", "user.name=test", "-c", "user.email=test@example.com",
         "commit", "-qm", "baseline"],
    ]:
        subprocess.run(["git", *args], cwd=repo, check=True)

    answer = "The inspected constant is 42; ANALYSIS_HANDOFF_SENTINEL."
    uncertainty = "STATIC_ONLY_SENTINEL: runtime behavior was not tested."
    read_file = {"tool": "read_file", "arguments": {"path": "util.py"}}

    def finish(text):
        return {"tool": "finish", "arguments": {
            "report": text,
            "artifacts": [{
                "kind": "module_report", "path": "module_report.md", "media_type": "text/markdown",
                "output_name": "analysis",
                "summary": "Code explanation", "content": (
                    f"## answer\n\n{text}\n\n## uncertainty\n\n{uncertainty}\n\n## evidence_files\n\nutil.py"
                ),
            }],
        }}

    clients = {
        "task_inspect": ScriptedLLMClient(tool_turns([read_file, finish(answer)])),
        "task_independent": ScriptedLLMClient(tool_turns([read_file, finish("Independent inspection.")])),
    }
    requests = {}
    sessions = JsonSessionStore(tmp_path / "coding_sessions")
    layout = ResourceLayout(resource_root=tmp_path / "resources")

    class CodingPort:
        def invoke(self, request):
            requests[request.task_id] = request
            if request.task_id == "task_followup":
                report = next(ref for ref in request.input_artifacts if ref.output_name == "analysis")
                clients[request.task_id] = ScriptedLLMClient(tool_turns([
                    {"tool": "read_artifact", "arguments": {"artifact_id": report.id}},
                    read_file, finish("Follow-up inspected the shared analysis and source."),
                ]))
            return NativeCodingAgent(
                clients[request.task_id], store=sessions, resource_layout=layout,
            ).invoke(request)

    class Compiler:
        def compile(self, request, **kwargs):
            assert kwargs["current"] is not None
            return CompilationResult(WorkflowPatch(
                work_request_id=request.id, based_on_revision=kwargs["current"].revision,
                add_tasks=[TaskProposal(
                    id=task_id, work_request_id=request.id,
                    execution_ref="coding@1", workflow_agent_kind=WorkflowAgentKind.EXECUTOR,
                    instruction="Inspect the code without changing it. Where is VALUE defined?",
                    input_artifact_bindings=[
                        FutureArtifactBinding(source_task=dependencies[0], output_selector="analysis"),
                    ] if dependencies else [],
                    output_names=["analysis"],
                    workspace_id="ws_main", depends_on=dependencies,
                ) for task_id, dependencies in [
                    ("task_inspect", []),
                    ("task_followup", ["task_inspect"]),
                    ("task_independent", []),
                ]],
            ))

    class ScientificClient:
        tool_session_key = "test-native-tools/v1"

        def __init__(self):
            self.contexts = []
            self.histories = []

        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            self.contexts.append(context)
            self.histories.append([turn.model_copy(deep=True) for turn in turns])
            if len(self.contexts) == 1:
                return tool_turns([{"tool": "request_work", "arguments": {
                    "assessment": {"statement": "Need code inspection, not a measured experiment"},
                    "work_request": {
                        "objective": "Inspect the code and provide the analysis to a follow-up",
                        "tasks": [{"selection": {"kind": "preset", "preset_ref": "coding@1", "reason": "Suitable configured executor"},
                            "key": key,
                            "instruction": "Inspect the code without changing it. Where is VALUE defined?",
                            "depends_on": dependencies,
                            "input_artifact_bindings": [{
                                "source_task": dependencies[0], "output_selector": "analysis",
                            }] if dependencies else [],
                            "output_names": ["analysis"], "workspace_id": "ws_main",
                        } for key, dependencies in [
                            ("inspect", []), ("followup", ["inspect"]), ("independent", []),
                        ]],
                    },
                }}])[0]
            current = scheduler.store.load("run_handoffs")
            report_id = next(
                ref.id for ref in current.artifacts.values()
                if ref.task_id == "task_inspect" and ref.output_name == "analysis"
            )
            if len(self.contexts) == 2:
                return tool_turns([{"tool": "read_artifact", "arguments": {"artifact_id": report_id}}])[0]
            return tool_turns([{"tool": "finish", "arguments": {
                "report": "Scientific conclusion", "artifacts": [{
                    "kind": "scientific_opinion", "path": "opinion.json",
                    "media_type": "application/json", "summary": "Scientific conclusion",
                    "content": {
                        "verdict": "not_applicable", "statement": answer,
                        "limitations": [uncertainty], "evidence_artifact_ids": [report_id],
                    },
                }],
            }}])[0]

    scientific_client = ScientificClient()
    scheduler = WorkflowScheduler(bindings={WorkflowAgentKind.EXECUTOR: ModuleBinding(owner=AgentOwner.EXECUTOR, port=CodingPort())}, store=JsonRunStore(tmp_path / 'runs'), artifact_root=tmp_path / 'artifacts', data_root=tmp_path / 'data', workspaces={'ws_main': WorkspaceSpec(workspace_id='ws_main', source_kind=WorkspaceSourceKind.LOCAL, location=str(repo), access=WorkspaceAccess(read_paths=['.'], write_paths=['.']))})
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(),
        scientific_port=ScientificAgent(
            scientific_client, store=JsonSessionStore(tmp_path / "scientific_sessions"),
            resource_layout=layout,
        ),
        compiler=Compiler(), scheduler=scheduler,
        registry=default_execution_registry(),
    )
    run = controller.create_run('run_handoffs', ResearchRequest(goal='Explain the code, with the analysis available to subsequent work', budget=RunBudget(max_llm_calls=30, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=4, max_attempts_per_task=1)))

    assert run.status == RunStatus.COMPLETED, run.model_dump(mode="json")
    report_id = next(
        ref.id for ref in run.artifacts.values()
        if ref.task_id == "task_inspect" and ref.output_name == "analysis"
    )
    assert all(task.status == "completed" for task in run.workflow.tasks)
    assert not any(a.kind == "module_report" for a in requests["task_inspect"].input_artifacts)
    assert not any(a.kind == "module_report" for a in requests["task_independent"].input_artifacts)
    assert [a.id for a in requests["task_followup"].input_artifacts if a.kind == "module_report"] == [report_id]
    # Check actual native-Agent prompts after read_artifact, not just stored payload.
    for context, history in [(clients["task_followup"].contexts[1], clients["task_followup"].histories[1]),
                             (scientific_client.contexts[-1], scientific_client.histories[-1])]:
        receipts = json.dumps([turn.model_dump(mode="json") for turn in history])
        assert answer in receipts and uncertainty in receipts
        assert answer not in context.material_index and uncertainty not in context.material_index
        assert "artifact_reads" not in context.included_sections
    assert report_id in scientific_client.contexts[1].text
    assert '"kind": "module_report"' in scientific_client.contexts[1].text
    materials = json.loads(scientific_client.contexts[1].material_index.split("\n", 1)[1])
    registered = next(node for node in materials["children"] if node["name"] == "registered_materials")
    work_group = next(node for node in registered["children"] if node["name"] == "work_requests")
    assert len(work_group["children"]) == 1
    work = work_group["children"][0]
    child_work = next(item for item in run.work_requests if item.owner_task_id == "task_root")
    assert work["name"] == child_work.id
    assert {task["name"] for task in work["children"]} == {"task_inspect", "task_followup", "task_independent"}
    inspected = next(task for task in work["children"] if task["name"] == "task_inspect")
    assert report_id in {entry["location"]["artifact_id"] for entry in inspected["children"]}
    original_report_id = run.task("task_inspect").attempts[0].report_artifact_id
    assert original_report_id in {entry["location"]["artifact_id"] for entry in inspected["children"]}
    def artifact_entries(node):
        return ([node] if node["type"] == "artifact" else []) + [
            entry for child in node["children"] for entry in artifact_entries(child)]
    entries = artifact_entries(registered)
    assert materials["artifact_count"] == len(entries) == len({entry["location"]["artifact_id"] for entry in entries})
    assert not any(ref.kind in {"question", "answer", "work_request", "work_feedback", "work_record", "research_index"}
                   for ref in run.artifacts.values())
    assert not any(key in entry for entry in entries for key in ("uri", "sha256", "content"))
    assert run.final_opinion.statement == answer
    assert run.final_opinion.limitations == [uncertainty]

    restored = scheduler.store.load(run.run_id)
    report = restored.artifacts[report_id]
    body = RegisteredArtifactReader([report], run_id=run.run_id).read_text(report_id)
    assert report.media_type == "text/markdown"
    assert f"## answer\n\n{answer}" in body["content"]
    assert f"## uncertainty\n\n{uncertainty}" in body["content"]
    assert "## evidence_files" in body["content"]
    assert "util.py" in body["content"]
    scientific_state = JsonSessionStore(tmp_path / "scientific_sessions").load(restored.scientific_session.id)
    read_ids = {
        event.data["value"]["artifact_id"] for event in scientific_state.events
        if event.type == "observation" and event.tool == "read_artifact" and event.data.get("ok")
    }
    assert report_id in read_ids  # Access history is checked, not used as a completion gate.
    assert restored.llm_calls_used == 10  # 2 + 3 + 2 Coding calls, 3 Scientific calls.
    assert not (repo / "module_report.md").exists()
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=repo) == b""
