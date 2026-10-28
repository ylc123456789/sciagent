"""Shared operation decisions and durable single-use approval."""

from e2e.native_fixtures import tool_turns
from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from resagent2_capabilities import DeletePathTool
from resagent2_capabilities.permissions import OperationPermissionPolicy
from resagent2_components.workspace import WorkspaceBoundary
from resagent2_components.process import ProcessRunner
from resagent2_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ArtifactCandidate, QuestionDraft,
    RecordedAnswer, TaskBudget, WorkspaceAccess, WorkspaceGrant,
)
from resagent2_orchestrator.artifacts import ArtifactRegistry
from resagent2_runtime import (
    AgentDefinition, AgentLoop, CompletionDecision, FinishTool, JsonSessionStore,
    ScriptedLLMClient, ToolObservation,
)


class CommandInput(BaseModel):
    command: str


class Command:
    name = "run_shell"
    input_model = CommandInput

    def __init__(self):
        self.calls = []
        self.before_execute = lambda: None
        self.extra_env = {}

    def prepare(self, arguments):
        return self.approval_context()

    def approval_context(self):
        return self.runner.shell_context(extra_env=self.extra_env)

    def execute_prepared(self, state, arguments, prepared):
        assert prepared == self.approval_context()
        return self.execute(state, arguments)

    def execute(self, state, arguments):
        self.before_execute()
        self.calls.append(arguments.command)
        return ToolObservation(summary="Executed", value={"command": arguments.command})


class Completion:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=candidate is not None, report="Done")


def request(root, **changes):
    fields = dict(
        run_id="run_permissions", work_request_id="work_permissions", task_id="task_permissions", attempt_number=1,
        agent=AgentOwner.EXECUTOR, instruction="Do the authorized work",
        budget=TaskBudget(max_llm_calls=8, timeout_seconds=60),
        permissions=AgentPermissions(execute_commands=True),
        workspace=WorkspaceGrant(root=str(root), source="local",
                                 access=WorkspaceAccess(read_paths=["."], write_paths=["."])),
    )
    return AgentRequest(**(fields | changes))


def execute(req, actions, tool, store, *, client=None):
    boundary = WorkspaceBoundary(req.workspace)
    if isinstance(tool, Command):
        tool.runner = ProcessRunner(boundary)
    tools = (tool, FinishTool())
    definition = AgentDefinition(
        name="permissions", owner=AgentOwner.EXECUTOR, system_prompt="Test",
        tools=tools, llm_client=client or ScriptedLLMClient(tool_turns(actions)),
        context_builder=lambda *_: [],
        permission_policy=OperationPermissionPolicy(
            tools, boundary=boundary,
            request=req,
        ),
        completion_check=Completion(),
    )
    return AgentLoop(store=store).run(definition, req, session_id="session_permissions")


def answer(tmp_path, req, result, *, comment=None):
    question = result.control.question
    recorded = RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, answered_at=datetime.now(UTC),
        comment=comment,
        run_id=req.run_id, session_id=result.session.id,
        task_id=req.task_id, attempt_number=req.attempt_number, action=question.action,
    )
    return req.model_copy(update={"parent_session_id":result.session.id,
        "resume_answers":[recorded], "control_ack":result.control.origin})


def command(text):
    return {"tool": "run_shell", "arguments": {"command": text}}


FINISH = {"tool": "finish", "arguments": {"report": "Done"}}


@pytest.mark.parametrize("second", ["python first.py", "python second.py"])
def test_approval_is_consumed_once_and_never_authorizes_next_command(tmp_path, second):
    for name in ("first.py", "second.py"):
        (tmp_path / name).write_text("pass")
    req = request(tmp_path)
    tool = Command()
    store = JsonSessionStore(tmp_path / "sessions")
    first = execute(req, [command("python first.py")], tool, store)
    assert first.status == "needs_user_input" and tool.calls == []
    first_action = store.load(first.session.id).pending_action
    resumed = answer(tmp_path, req, first)
    next_result = execute(resumed, [command(second)], tool, store)
    assert next_result.status == "needs_user_input"
    assert tool.calls == ["python first.py"]
    assert store.load(first.session.id).pending_action.action_id != first_action.action_id
    final_req = answer(tmp_path, resumed, next_result)
    final = execute(final_req, [FINISH], tool, store)
    assert final.status == "completed"
    assert tool.calls == ["python first.py", second]
    assert store.load(first.session.id).pending_action is None


@pytest.mark.parametrize("restriction", ["operation", "paths"])
def test_approval_cannot_override_current_run_authorization(tmp_path, restriction):
    (tmp_path / "first.py").write_text("pass")
    req, tool = request(tmp_path), Command()
    store = JsonSessionStore(tmp_path / "sessions")
    first = execute(req, [command("python first.py")], tool, store)
    resumed = answer(tmp_path, req, first)
    if restriction == "operation":
        resumed.permissions.execute_commands = False
    else:
        resumed.workspace.access = WorkspaceAccess(read_paths=["."], write_paths=["results"])
    result = execute(resumed, [command("python first.py"), FINISH], tool, store)
    assert result.status == "completed" and tool.calls == []
    events = store.load(first.session.id).events
    assert any(event.type == "observation" and not event.data["ok"] for event in events)


def test_recursive_delete_requires_new_approval_if_target_changes(tmp_path):
    target = tmp_path / "cache"
    target.mkdir()
    (target / "old.txt").write_text("old")
    req = request(tmp_path)
    store = JsonSessionStore(tmp_path / "sessions")
    tool = DeletePathTool(WorkspaceBoundary(req.workspace))
    action = {"tool": "delete_path", "arguments": {"path": "cache", "recursive": True}}
    first = execute(req, [action], tool, store)
    resumed = answer(tmp_path, req, first)
    (target / "new.txt").write_text("new")
    second = execute(resumed, [action], tool, store)
    assert second.status == "needs_user_input"
    assert second.control.source_turn != first.control.source_turn
    original_receipt = json.loads(store.load(first.session.id).tool_turns[
        first.control.source_turn].tool_results[first.control.source_call_id])
    assert original_receipt["value"]["execution_status"] == "not_executed"
    assert sorted(path.name for path in target.iterdir()) == ["new.txt", "old.txt"]
    final = execute(answer(tmp_path, resumed, second), [FINISH], tool, store)
    assert final.status == "completed" and not target.exists()


def test_consumed_approval_is_durable_before_effect_and_not_replayed_after_crash(tmp_path):
    class Interrupted(BaseException):
        pass
    (tmp_path / "first.py").write_text("pass")
    req, tool = request(tmp_path), Command()
    store = JsonSessionStore(tmp_path / "sessions")
    first = execute(req, [command("python first.py")], tool, store)
    comment = "  This approval applies to the interrupted operation only.\n原文保留  "
    resumed = answer(tmp_path, req, first, comment=comment)
    attempts = []
    def interrupt():
        state = store.load(first.session.id)
        assert state.pending_action is None
        assert state.tool_turns[-1].executing_user_comment == comment
        attempts.append("started")
        raise Interrupted()
    tool.before_execute = interrupt
    with pytest.raises(Interrupted):
        execute(resumed, [command("python first.py")], tool, store)
    client = ScriptedLLMClient(tool_turns([FINISH]))
    result = execute(resumed.model_copy(update={"resume_answers": [], "control_ack": None}),
                     [], tool, JsonSessionStore(store.root), client=client)
    assert result.status == "completed" and attempts == ["started"]
    call_id = first.control.source_call_id
    receipts = [json.loads(turn.tool_results[call_id]) for turn in client.histories[0]
                if call_id in turn.tool_results]
    assert len(receipts) == 1 and receipts[0]["user_comment"] == comment
    assert "unknown" in receipts[0]["summary"]
    restored = JsonSessionStore(store.root).load(first.session.id)
    assert all(turn.executing_user_comment is None for turn in restored.tool_turns)


def test_shell_default_mode_asks_with_execution_only_permission(tmp_path):
    req = request(tmp_path, permissions=AgentPermissions(execute_commands=True))
    tool, store = Command(), JsonSessionStore(tmp_path / "sessions")
    script = "  printf 'one\n' | cat\n"
    first = execute(req, [command(script)], tool, store)
    assert first.status == "needs_user_input" and tool.calls == []
    draft = first.control.question
    assert draft.action.arguments == {"command": script}
    assert draft.action.context["cwd"] == str(tmp_path)
    assert draft.action.context["shell"] == "/bin/bash"
    assert draft.action.context["pipefail"] is True
    result = execute(answer(tmp_path, req, first), [FINISH], tool, store)
    assert result.status == "completed" and tool.calls == [script]


def test_changed_shell_script_requires_its_own_approval(tmp_path):
    req, tool = request(tmp_path), Command()
    store = JsonSessionStore(tmp_path / "sessions")
    first = execute(req, [command("printf first")], tool, store)
    result = execute(answer(tmp_path, req, first), [command("printf second")], tool, store)
    assert result.status == "needs_user_input" and tool.calls == ["printf first"]
    assert result.control.question.action.arguments == {
        "command": "printf second",
    }


def test_shell_approval_does_not_follow_a_changed_startup_environment(tmp_path):
    req, tool = request(tmp_path), Command()
    tool.extra_env = {"PATH": "/first/environment/bin"}
    store = JsonSessionStore(tmp_path / "sessions")
    action = command("python measure.py")
    first = execute(req, [action], tool, store)
    resumed = answer(tmp_path, req, first, comment="Only for the original startup")
    original_action = first.control.question.action
    tool.extra_env = {"PATH": "/second/environment/bin"}

    second = execute(resumed, [action], tool, store)
    assert second.status == "needs_user_input" and tool.calls == []
    next_action = second.control.question.action
    assert next_action.arguments == original_action.arguments
    assert next_action.action_id != original_action.action_id
    assert original_action.context["environment_fingerprint"] != next_action.context["environment_fingerprint"]
    assert "/first/environment/bin" not in json.dumps(original_action.context)

    final = execute(answer(tmp_path, resumed, second, comment="New startup reviewed"),
                    [FINISH], tool, store)
    assert final.status == "completed" and tool.calls == ["python measure.py"]
    receipts = [json.loads(raw) for turn in store.load(first.session.id).tool_turns
                for raw in turn.tool_results.values()]
    assert [r["user_comment"] for r in receipts if "user_comment" in r] == [
        "Only for the original startup", "New startup reviewed",
    ]
    original_receipt = json.loads(store.load(first.session.id).tool_turns[
        first.control.source_turn].tool_results[first.control.source_call_id])
    assert original_receipt["value"]["execution_status"] == "not_executed"
    assert "new tool call" in original_receipt["summary"]
    assert second.control.source_turn != first.control.source_turn
