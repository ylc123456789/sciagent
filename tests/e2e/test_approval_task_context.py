from resagent2_executor import default_execution_registry
"""An operation approval resumes execution without becoming task dialogue."""

import json
import subprocess
import sys
import shlex
from datetime import UTC, datetime

import pytest

from e2e.native_fixtures import tool_turns, ParentQuestionClient
from resagent2_components import ResourceLayout
from resagent2_contracts import (
    AgentOwner, ExecutionLimits, ResearchRequest, RunBudget, RunPermissions,
    TaskProposal, UserAnswer, WorkflowAgentKind,
    WorkflowAgentRegistry, WorkflowPatch, WorkflowProposal, WorkspaceAccess, WorkspaceSourceKind,
    WorkspaceSpec,
)
from resagent2_coding import NativeCodingAgent
from resagent2_experiment import NativeExperimentAgent
from resagent2_orchestrator import (
    CompilationResult, DeterministicWorkInterpreter, JsonRunStore, ModuleBinding,
    ResearchController, WorkflowScheduler,
)
from resagent2_runtime import JsonSessionStore, ScriptedLLMClient
from resagent2_scientific import ScientificAgent


@pytest.mark.parametrize("kind", ["coding", "experiment"])
def test_business_yes_and_operation_approval_keep_distinct_consumers(
    tmp_path, monkeypatch, kind,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("executed.marker\n")
    (repo / "measure.py").write_text(
        "from pathlib import Path\n"
        "with Path('executed.marker').open('a') as f: f.write('executed\\n')\n"
    )
    for args in [
        ["init", "-q"], ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "-qm", "baseline"],
    ]:
        subprocess.run(["git", *args], cwd=repo, check=True)

    command = {"tool": "run_shell", "arguments": {"command": f"{shlex.quote(sys.executable)} measure.py"}}
    business_text = "Should this task continue? This answer is task guidance, not an operation approval."
    worker_client = ScriptedLLMClient(tool_turns([
        {"tool": "ask_user", "arguments": {
            "text": business_text, "requested_fields": ["approve"],
        }},
        command,
        {"tool": "finish", "arguments": {"report": "Executed the approved command once"}},
    ]))
    scientific_client = ParentQuestionClient(tool_turns([
        {"tool": "request_work", "arguments": {
            "assessment": {"statement": "Need a recorded execution outcome"},
            "work_request": {'objective': 'Execute the local marker script after asking the user', 'tasks': [{"selection": {"kind": "preset", "preset_ref": "coding@1", "reason": "Suitable configured executor"}, 'key': 'work', 'instruction': 'Execute the local marker script after asking the user'}]},
        }},
        {"tool": "finish", "arguments": {
            "report": "The local execution task is complete",
            "artifacts": [{
                "kind": "scientific_opinion", "path": "opinion.json",
                "media_type": "application/json", "summary": "Execution outcome",
                "content": {
                    "verdict": "not_applicable", "statement": "The approved local script executed once",
                    "evidence_artifact_ids": [], "limitations": ["No scientific hypothesis evaluated"],
                },
            }],
        }},
    ]))
    owner = AgentOwner.EXECUTOR
    workflow_kind = WorkflowAgentKind.EXECUTOR
    sessions = JsonSessionStore(tmp_path / "sessions")
    layout = ResourceLayout(resource_root=tmp_path / "resources")
    agent_type = NativeCodingAgent if kind == "coding" else NativeExperimentAgent
    worker = agent_type(worker_client, store=sessions, resource_layout=layout)

    class Compiler:
        def compile(self, request, **kwargs):
            return CompilationResult(WorkflowPatch(
                work_request_id=request.id,
                based_on_revision=kwargs["current"].revision,
                add_tasks=[TaskProposal(
                    id="task_execute", work_request_id=request.id,
                    execution_ref=f"{kind}@1", workflow_agent_kind=workflow_kind, workspace_id="ws_main",
                    instruction=request.request.tasks[0].instruction,
                )],
            ))

    def controller():
        scheduler = WorkflowScheduler(
            bindings={workflow_kind: ModuleBinding(owner=owner, port=worker)},
            store=JsonRunStore(tmp_path / "runs"),
            artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data",
            workspaces={"ws_main": WorkspaceSpec(
                workspace_id="ws_main", source_kind=WorkspaceSourceKind.LOCAL,
                location=str(repo), access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
            )},
        )
        return ResearchController(
            interpreter=DeterministicWorkInterpreter(), compiler=Compiler(),
            scientific_port=ScientificAgent(
                scientific_client, store=JsonSessionStore(tmp_path / "scientific_sessions"),
                resource_layout=layout,
            ),
            scheduler=scheduler,
            registry=default_execution_registry(),
        )

    run = controller().create_run("run_approval_context", ResearchRequest(
        goal="Record one local execution after business guidance and exact operation approval",
        budget=RunBudget(max_llm_calls=20, timeout_seconds=60),
        permissions=RunPermissions(execute_commands=True),
        execution_limits=ExecutionLimits(max_tasks=2, max_attempts_per_task=1),
    ))
    assert run.status == "paused", run.terminal_error
    assert run.pending_question.action is None
    assert run.pending_question.requested_fields == ["approve"]
    session_id = next(task for task in run.workflow.tasks if task.id == "task_execute").attempts[0].session.id
    run = controller().answer_question(run.run_id, UserAnswer(
        question_id=run.pending_question.id, values={"approve": "yes"}, answered_at=datetime.now(UTC),
    ))
    assert run.status == "paused", run.terminal_error
    assert run.pending_question.action.tool == "run_shell"
    assert run.pending_question.action.arguments == command["arguments"]
    assert not (repo / "executed.marker").exists()
    action = run.pending_question.action
    comment = "  只执行本次操作。\nKeep this explanation separate from task guidance.  "
    run = controller().answer_question(run.run_id, UserAnswer(
        question_id=run.pending_question.id, values={"approve": "yes"}, answered_at=datetime.now(UTC),
        comment=comment,
    ))
    assert run.status == "completed", run.terminal_error
    assert (repo / "executed.marker").read_text() == "executed\n"
    task = next(task for task in run.workflow.tasks if task.id == "task_execute")
    assert len(task.attempts) == 1 and task.attempts[0].session.id == session_id
    assert len(run.answers) == 3
    assert run.answers[0].action is None and run.answers[0].source == "user"
    assert run.answers[1].action is None and run.answers[1].source == "parent"
    assert run.answers[2].action == action
    assert JsonRunStore(tmp_path / "runs").load(run.run_id).answers[2].comment == comment

    assert not any(ref.kind in {"question", "answer", "work_request", "work_feedback", "work_record", "research_index"}
                   for ref in run.artifacts.values())
    assert {answer.question_id for answer in run.answers if answer.source == "user"} <= set(run.received_controls.values())
    # Inspect the task section actually sent before execution and before finish.
    # The operation remains in durable control/history, not in task requirements.
    for context in worker_client.contexts[1:]:
        task_text = context.task_context.split("\n", 1)[1]
        dialogue = [e["record"] for e in json.loads(task_text)["business_dialogue"] if e["type"] == "answer"]
        assert len(dialogue) == 1
        assert dialogue[0]["question_text"] == business_text
        assert dialogue[0]["values"] == {"approve": "yes"}
        assert dialogue[0]["question_id"] == run.answers[1].question_id
        assert "comment" not in dialogue[0]
        assert "只执行本次操作" not in context.task_context + context.material_index

    state = sessions.load(session_id)
    assert state.pending_action is None
    assert len(state.tool_turns) == 3
    assert all(set(turn.tool_results) == {call.id for call in turn.tool_calls}
               and turn.executing_call_id is None for turn in state.tool_turns)
    receipts = [json.loads(next(iter(turn.tool_results.values()))) for turn in state.tool_turns]
    assert sum(receipt.get("value", {}).get("exit_code") == 0 for receipt in receipts
               if isinstance(receipt.get("value"), dict)) == 1
    command_receipt = next(receipt for receipt in receipts if "user_comment" in receipt)
    assert command_receipt["user_comment"] == comment
    # It is in the actual paired native history sent to the model after restart.
    assert any(json.loads(raw).get("user_comment") == comment
               for turn in worker_client.histories[-1] for raw in turn.tool_results.values())
    assert run.llm_calls_used == len(worker_client.contexts) + len(scientific_client.contexts) == 7
