"""Real shell approval binds reviewer materials to the dispatched source version."""

from datetime import UTC, datetime
import json
import sys
from types import SimpleNamespace

import pytest

from e2e.native_fixtures import tool_turns
from resagent2_capabilities import RunShellInput, RunShellTool
from resagent2_capabilities.permissions import OperationPermissionPolicy
from resagent2_components.approval_review import LLMPermissionReviewer
from resagent2_components.process import ProcessRunner
from resagent2_components.workspace import WorkspaceBoundary
from resagent2_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, RecordedAnswer, TaskBudget,
    WorkspaceAccess, WorkspaceGrant,
)
from resagent2_runtime import (
    AgentDefinition, AgentLoop, AgentState, CompletionDecision, FinishTool,
    JsonSessionStore, ScriptedLLMClient,
)


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="actual Linux Bash execution")

COMMAND = "python3 -m unittest -v test_metric"
SHELL = {"tool": "run_shell", "arguments": {"command": COMMAND}}
FINISH = {"tool": "finish", "arguments": {"report": "Recorded the actual shell receipt"}}
METRIC = '''"""METRIC_SOURCE_CANARY: a deterministic local implementation."""
def mse(actual, predicted):
    return sum((a - b) ** 2 for a, b in zip(actual, predicted)) / len(actual)
'''
TEST = '''"""TEST_SOURCE_CANARY: this imports the actual implementation."""
from pathlib import Path
import unittest
from metric import mse

class MetricTests(unittest.TestCase):
    def test_value(self):
        self.assertEqual(mse([1, 2], [0, 1]), 1.0)
        with Path("executed.marker").open("a") as handle:
            handle.write("ran\\n")
'''


class _Complete:
    def evaluate(self, state, candidate):
        return CompletionDecision(
            complete=candidate is not None,
            report=candidate.report if candidate else "",
        )


class _ReviewClient(ScriptedLLMClient):
    def __init__(self, callback=None):
        super().__init__(tool_turns([{
            "tool": "review_operation", "arguments": {
                "decision": "allow", "reason": "The supplied synthetic test is authorized",
            },
        }]))
        self.callback = callback

    def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
        result = super().next_tool_call(
            context, schemas, turns, max_input_tokens=max_input_tokens,
        )
        if self.callback:
            self.callback()
        return result


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "metric.py").write_text(METRIC, encoding="utf-8")
    (root / "test_metric.py").write_text(TEST, encoding="utf-8")
    grant = WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(
        read_paths=["."], write_paths=["."],
    ))
    request = AgentRequest(work_request_id="work_test",
        run_id="run_source_approval", task_id="task_source_approval", attempt_number=1,
        agent=AgentOwner.EXECUTOR, instruction="Run the local metric unittest and record its outcome",
        permissions=AgentPermissions(execute_commands=True, approval_mode="auto_review"),
        budget=TaskBudget(max_llm_calls=10, timeout_seconds=30), workspace=grant,
    )
    boundary = WorkspaceBoundary(grant)
    runner = ProcessRunner(boundary)
    launches = []
    original_launch = runner._run_argv

    def launch(argv, **kwargs):
        launches.append(argv)
        return original_launch(argv, **kwargs)

    monkeypatch.setattr(runner, "_run_argv", launch)
    return SimpleNamespace(
        root=root, boundary=boundary, request=request, launches=launches,
        store=JsonSessionStore(tmp_path / "sessions"),
        tool=RunShellTool(runner, timeout_seconds=5, log_dir=str(root / ".resagent2" / "shell")),
    )


def _invoke(case, *, request=None, reviewer=None, actions=None):
    request = request or case.request
    tools = (case.tool, FinishTool())
    definition = AgentDefinition(
        name="coding", owner=AgentOwner.EXECUTOR, system_prompt="Use the supplied tools.",
        tools=tools, llm_client=ScriptedLLMClient(tool_turns(actions or [SHELL, FINISH])),
        context_builder=lambda *_: [], completion_check=_Complete(),
        permission_policy=OperationPermissionPolicy(tools, boundary=case.boundary, request=request),
        operation_reviewer=reviewer,
    )
    return AgentLoop(store=case.store).run(definition, request, session_id="session_source_approval")


def _reviewer(callback=None):
    client = _ReviewClient(callback)
    return LLMPermissionReviewer(client, max_context_tokens=20_000), client


def _receipts(case):
    state = case.store.load("session_source_approval")
    return [json.loads(turn.tool_results[call.id]) for turn in state.tool_turns
            for call in turn.tool_calls if call.name == "run_shell" and call.id in turn.tool_results]


def _approve(case, request, paused):
    question = paused.control.question
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text,
        requested_fields=question.requested_fields, options=question.options,
        values={"approve": "yes"}, comment="Only the reviewed synthetic operation",
        answered_at=datetime.now(UTC), run_id=request.run_id,
        session_id=paused.session.id, task_id=request.task_id,
        attempt_number=request.attempt_number, action=question.action,
    )
    return request.model_copy(update={
        "parent_session_id": paused.session.id, "resume_answers": [answer],
        "control_ack": paused.control.origin,
    })


def _state(case):
    now = datetime.now(UTC)
    return AgentState(
        session_id="session_source_approval", agent_name="coding", owner=AgentOwner.EXECUTOR,
        run_id=case.request.run_id, task_id=case.request.task_id, attempt_number=1,
        created_at=now, updated_at=now,
    )


def _assert_not_executed(case, receipt):
    assert not case.launches
    assert not (case.root / "executed.marker").exists()
    assert receipt["ok"] is False
    assert receipt["value"]["execution_status"] == "not_executed"


def test_actual_entry_and_local_implementation_reach_stateless_reviewer(case):
    reviewer, client = _reviewer()
    result = _invoke(case, reviewer=reviewer)
    assert result.status == "completed", result.report
    assert len(case.launches) == len(client.contexts) == 1
    assert (case.root / "executed.marker").read_text().splitlines() == ["ran"]
    payload = json.loads(client.contexts[0].task_context.split("\n", 1)[1])
    evidence = json.dumps(payload["implementation_evidence"], ensure_ascii=False)
    assert json.dumps(TEST, ensure_ascii=False) in evidence
    assert json.dumps(METRIC, ensure_ascii=False) in evidence
    assert client.histories == [[]]
    operation = payload["proposed_operation"]
    assert operation["context"]["source_fingerprint"] == case.tool.prepare(RunShellInput(command=COMMAND))["source_fingerprint"]
    receipt, = _receipts(case)
    assert receipt["ok"] and receipt["value"]["exit_code"] == 0
    assert receipt["permission_review"]["status"] == "allow"


@pytest.mark.parametrize("changed", ["test_metric.py", "metric.py"])
def test_change_during_auto_review_invalidates_the_original_operation(case, changed):
    def change_source():
        with (case.root / changed).open("a") as handle:
            handle.write("\n# changed while the provider returned its verdict\n")

    reviewer, client = _reviewer(change_source)
    result = _invoke(case, reviewer=reviewer)
    assert result.status == "completed", result.report
    assert len(client.contexts) == 1
    receipt, = _receipts(case)
    _assert_not_executed(case, receipt)


@pytest.mark.parametrize("changed", [None, "metric.py"])
def test_human_resume_only_executes_the_unchanged_source_version(case, changed):
    request = case.request.model_copy(update={
        "permissions": AgentPermissions(execute_commands=True, approval_mode="manual"),
    })
    reviewer, client = _reviewer()
    paused = _invoke(case, request=request, reviewer=reviewer)
    assert paused.status == "needs_user_input"
    assert not case.launches
    if changed:
        with (case.root / changed).open("a") as handle:
            handle.write("\n# changed during the human pause\n")
    result = _invoke(case, request=_approve(case, request, paused), reviewer=reviewer, actions=[FINISH])
    assert result.status == "completed", result.report
    assert not client.contexts  # Human resume never launches a second automatic review.
    receipt, = _receipts(case)
    assert receipt["user_comment"] == "Only the reviewed synthetic operation"
    if changed:
        _assert_not_executed(case, receipt)
    else:
        assert receipt["ok"] and len(case.launches) == 1
        assert (case.root / "executed.marker").read_text().splitlines() == ["ran"]


def test_final_tool_check_catches_change_after_runtime_permission_recheck(case, monkeypatch):
    original_execute = case.tool.execute_prepared

    def change_before_dispatch(state, arguments, prepared):
        with (case.root / "metric.py").open("a") as handle:
            handle.write("\n# changed immediately before tool execution\n")
        return original_execute(state, arguments, prepared)

    monkeypatch.setattr(case.tool, "execute_prepared", change_before_dispatch)
    reviewer, _ = _reviewer()
    result = _invoke(case, reviewer=reviewer)
    assert result.status == "completed", result.report
    receipt, = _receipts(case)
    _assert_not_executed(case, receipt)


@pytest.mark.parametrize("fingerprint", [None, "not-a-fingerprint", "f" * 64, 7])
def test_missing_invalid_or_unmatched_fingerprint_cannot_launch(case, fingerprint):
    arguments = RunShellInput(command=COMMAND)
    prepared = case.tool.prepare(arguments)
    if fingerprint is None:
        prepared.pop("source_fingerprint")
    else:
        prepared["source_fingerprint"] = fingerprint
    result = case.tool.execute_prepared(_state(case), arguments, prepared)
    _assert_not_executed(case, result.model_dump(mode="json"))


def test_unknown_startup_field_is_not_discarded_when_splitting_source_fingerprint(case):
    arguments = RunShellInput(command=COMMAND)
    prepared = case.tool.prepare(arguments)
    prepared["unknown_startup_flag"] = "must not be silently projected away"
    result = case.tool.execute_prepared(_state(case), arguments, prepared)
    _assert_not_executed(case, result.model_dump(mode="json"))


def test_old_pending_approval_without_fingerprint_does_not_gain_a_new_version(case, monkeypatch):
    request = case.request.model_copy(update={
        "permissions": AgentPermissions(execute_commands=True, approval_mode="manual"),
    })
    prepare = case.tool.prepare

    def legacy_prepare(arguments):
        prepared = prepare(arguments)
        prepared.pop("source_fingerprint")
        return prepared

    monkeypatch.setattr(case.tool, "prepare", legacy_prepare)
    paused = _invoke(case, request=request)
    assert paused.status == "needs_user_input"
    monkeypatch.setattr(case.tool, "prepare", prepare)
    result = _invoke(case, request=_approve(case, request, paused), actions=[FINISH])
    assert result.status == "completed", result.report
    receipt, = _receipts(case)
    _assert_not_executed(case, receipt)


COMPOUND_COMMAND = '''set -o pipefail
echo "=== python -V ==="
python3 -V
echo "=== sha256 BEFORE/AFTER inputs ==="
sha256sum predictions.json measure.py
echo "=== run tests ==="
python3 -m unittest -v test_metric 2>&1; echo "UNITTEST_EXIT=$?"'''


def _compound_action(case):
    (case.root / "predictions.json").write_text("[]\n", encoding="utf-8")
    (case.root / "measure.py").write_text("# fixed fixture entrypoint\n", encoding="utf-8")
    return {"tool": "run_shell", "arguments": {"command": COMPOUND_COMMAND}}


def test_original_compound_shell_is_reviewed_and_executed_as_one_unchanged_operation(case):
    action = _compound_action(case)
    reviewer, client = _reviewer()
    result = _invoke(case, reviewer=reviewer, actions=[action, FINISH])
    assert result.status == "completed", result.report
    assert len(client.contexts) == len(case.launches) == 1
    assert case.launches[0][-1] == COMPOUND_COMMAND
    payload = json.loads(client.contexts[0].task_context.split("\n", 1)[1])
    assert payload["proposed_operation"]["arguments"]["command"] == COMPOUND_COMMAND
    files = {item["path"].rsplit("/", 1)[-1]: item
             for item in payload["implementation_evidence"]["files"]}
    assert files["test_metric.py"]["content"] == TEST
    assert files["metric.py"]["content"] == METRIC
    assert client.histories == [[]]
    assert (case.root / "executed.marker").read_text().splitlines() == ["ran"]
    receipt, = _receipts(case)
    assert receipt["ok"] and receipt["value"]["exit_code"] == 0
    assert receipt["permission_review"]["status"] == "allow"


@pytest.mark.parametrize("changed", ["test_metric.py", "metric.py"])
def test_compound_shell_reuses_existing_source_change_invalidation(case, changed):
    action = _compound_action(case)

    def change_source():
        with (case.root / changed).open("a") as stream:
            stream.write("\n# dependency changed during compound-command review\n")

    reviewer, client = _reviewer(change_source)
    result = _invoke(case, reviewer=reviewer, actions=[action, FINISH])
    assert result.status == "completed", result.report
    assert len(client.contexts) == 1
    receipt, = _receipts(case)
    _assert_not_executed(case, receipt)
