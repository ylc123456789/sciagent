"""Automatic judgment cannot bypass grants, durable dispatch or human answers."""
from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest

from e2e.native_fixtures import tool_turns
from resagent2_capabilities.permissions import OperationPermissionPolicy
from resagent2_components import WorkspaceBoundary
from resagent2_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, RecordedAnswer, TaskBudget,
    WorkspaceAccess, WorkspaceGrant,
)
from resagent2_runtime import (
    AgentDefinition, AgentLoop, CompletionDecision, ContextSection,
    FinishCandidate, FinishTool, JsonSessionStore, ScriptedLLMClient, ToolObservation,
)
from resagent2_runtime.budget import BudgetExhaustedError, DeadlineExceededError, execution_budget
from resagent2_runtime.cancellation import CancellationToken, bind_cancellation
from resagent2_runtime.models import RuntimeModel


class ShellInput(RuntimeModel):
    command: str


class Shell:
    """Actual file effect under the real operation policy, without spawning Bash."""

    name = "run_shell"
    input_model = ShellInput

    def __init__(self, root, store):
        self.root, self.store = root, store
        self.tag = "original"
        self.callback = None

    def prepare(self, arguments):
        return self.approval_context()

    def approval_context(self):
        return {"launch_identity": self.tag}

    def execute_prepared(self, state, arguments, prepared):
        assert prepared == {"cwd": str(self.root), **self.approval_context()}
        saved = self.store.load(state.session_id)
        assert saved.tool_turns[-1].executing_call_id is not None
        with (self.root / "effect.txt").open("a") as handle:
            handle.write(arguments.command + "\n")
        if self.callback:
            self.callback(saved)
        return ToolObservation(summary="Executed", value={"exit_code": 0},
                               finish_candidate=FinishCandidate(report="Actual effect recorded"))


class Complete:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=candidate is not None, report=candidate.report if candidate else "")


class Reviewer:
    def __init__(self, verdict="allow", callback=None):
        self.calls = []
        self.verdict, self.callback = verdict, callback
        self.traces = []

    def set_trace_context(self, **kwargs):
        self.traces.append(kwargs)

    def review(self, operation, request):
        self.calls.append((operation, request))
        if self.callback:
            self.callback(operation, request)
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return {"decision": self.verdict, "reason": "Exact operation is within scope"}


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    store = JsonSessionStore(tmp_path / "sessions")
    grant = WorkspaceGrant(root=str(root), source="local",
                           access=WorkspaceAccess(read_paths=["."], write_paths=["."]))
    request = AgentRequest(work_request_id="work_test",
        run_id="run_review", task_id="task_review", attempt_number=1,
        agent=AgentOwner.EXECUTOR, instruction="Write the agreed marker once",
        permissions=AgentPermissions(execute_commands=True, approval_mode="auto_review"),
        budget=TaskBudget(max_llm_calls=8, timeout_seconds=30), workspace=grant,
    )
    return SimpleNamespace(root=root, store=store, request=request,
                           boundary=WorkspaceBoundary(grant), shell=Shell(root, store))


def invoke(case, reviewer, request=None, actions=None):
    request = request or case.request
    tools = (case.shell, FinishTool())
    client = ScriptedLLMClient(tool_turns(actions or [{"tool": "run_shell", "arguments": {"command": "marker"}}]))
    definition = AgentDefinition(
        name="executor", owner=AgentOwner.EXECUTOR, system_prompt="Use the supplied tools.",
        tools=tools, llm_client=client,
        context_builder=lambda req, state, limit: [ContextSection(name="task", content=req.instruction)],
        permission_policy=OperationPermissionPolicy(tools, boundary=case.boundary, request=request),
        completion_check=Complete(), operation_reviewer=reviewer,
    )
    case.client = client
    return AgentLoop(store=case.store).run(definition, request, session_id="session_review")


def resume_request(case, result, decision="yes", comment="Human decision"):
    question = result.control.question
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text, requested_fields=question.requested_fields,
        options=question.options, values={"approve": decision}, comment=comment,
        answered_at=datetime.now(UTC), run_id=case.request.run_id, session_id=result.session.id,
        task_id=case.request.task_id, attempt_number=1, action=question.action,
    )
    return case.request.model_copy(update={"parent_session_id": result.session.id,
        "resume_answers": [answer], "control_ack": result.control.origin})


def receipts(case):
    state = case.store.load("session_review")
    return [json.loads(value) for turn in state.tool_turns for call in turn.tool_calls
            if call.name == "run_shell" for key, value in turn.tool_results.items() if key == call.id]


def test_allow_is_reviewed_once_after_preflight_and_saved_before_effect(case):
    def reviewing(operation, request):
        saved = case.store.load("session_review")
        assert saved.events[-1].type == "permission_review"
        assert saved.events[-1].data["status"] == "started"
        assert saved.events[-1].data["operation"] == operation.model_dump(mode="json")
        assert not (case.root / "effect.txt").exists()
        assert saved.tool_turns[-1].executing_call_id is None

    def executing(saved):
        review = [event for event in saved.events if event.type == "permission_review"][-1]
        assert review.data["status"] == "allow"
        assert saved.tool_turns[-1].executing_call_id is not None

    case.shell.callback = executing
    reviewer = Reviewer(callback=reviewing)
    result = invoke(case, reviewer)
    assert result.status == "completed"
    assert len(reviewer.calls) == 1
    assert (case.root / "effect.txt").read_text() == "marker\n"
    receipt, = receipts(case)
    assert receipt["permission_review"]["source"] == "auto_review"
    assert receipt["permission_review"]["status"] == "allow"
    assert "user_comment" not in receipt
    native_id = case.store.load(result.session.id).tool_turns[0].tool_calls[0].id
    assert reviewer.traces[0]["operation_key"] == f"session:{result.session.id}:turn:0:call:{native_id}"
    assert reviewer.traces[0]["action_id"] == reviewer.calls[0][0].action_id
    assert reviewer.traces[0]["agent"] == "permission_reviewer"


@pytest.mark.parametrize("decision", ["yes", "no"])
def test_uncertain_review_waits_for_exact_user_answer_and_never_reviews_it_again(case, decision):
    reviewer = Reviewer("ask")
    result = invoke(case, reviewer)
    assert result.status == "needs_user_input"
    assert len(reviewer.calls) == 1
    assert not (case.root / "effect.txt").exists()
    assert result.control.question.action == reviewer.calls[0][0]
    forbidden = Reviewer(callback=lambda *_: pytest.fail("human approval must not be re-reviewed"))
    request = resume_request(case, result, decision)
    final = invoke(case, forbidden, request, actions=[{"tool": "finish", "arguments": {"report": "Done"}}])
    assert final.status == "completed"
    assert (case.root / "effect.txt").exists() == (decision == "yes")
    receipt, = receipts(case)
    assert receipt["permission_review"]["status"] == "ask"
    assert receipt["user_comment"] == "Human decision"
    assert receipt["ok"] == (decision == "yes")
    assert len(forbidden.calls) == 0


@pytest.mark.parametrize("verdict", ["not_a_decision", ValueError("secret must not be copied into the question")])
def test_invalid_or_failed_review_falls_back_to_manual_without_effect(case, verdict):
    result = invoke(case, Reviewer(verdict))
    assert result.status == "needs_user_input"
    assert not (case.root / "effect.txt").exists()
    assert "manual approval required" in result.control.question.text
    assert "secret must not" not in result.control.question.text
    state = case.store.load(result.session.id)
    assert state.events[-1].data["status"] == "error"


def test_missing_reviewer_never_silently_allows(case):
    result = invoke(case, None)
    assert result.status == "needs_user_input"
    assert not (case.root / "effect.txt").exists()


def test_manual_mode_does_not_invoke_reviewer(case):
    request = case.request.model_copy(update={"permissions": AgentPermissions(execute_commands=True)})
    reviewer = Reviewer(callback=lambda *_: pytest.fail("manual mode must not call reviewer"))
    result = invoke(case, reviewer, request)
    assert result.status == "needs_user_input"
    assert not [event for event in case.store.load(result.session.id).events if event.type == "permission_review"]


@pytest.mark.parametrize("denial", ["grant", "workspace"])
def test_deterministic_denial_cannot_be_overridden_by_reviewer(case, denial):
    if denial == "grant":
        case.request.permissions.execute_commands = False
    else:
        restricted = WorkspaceGrant(root=str(case.root), source="local",
            access=WorkspaceAccess(read_paths=["."], write_paths=["subdir"]))
        case.boundary = WorkspaceBoundary(restricted)
    reviewer = Reviewer()
    result = invoke(case, reviewer, actions=[{"tool": "run_shell", "arguments": {"command": "marker"}},
                                           {"tool": "finish", "arguments": {"report": "Denied"}}])
    assert result.status == "completed"
    assert not reviewer.calls
    assert not (case.root / "effect.txt").exists()


@pytest.mark.parametrize("change", ["context", "grant"])
@pytest.mark.parametrize("verdict", ["allow", ValueError("unavailable")])
def test_changes_during_review_invalidate_the_original_operation(case, change, verdict):
    def drift(*_):
        if change == "context":
            case.shell.tag = "changed"
        else:
            case.request.permissions.execute_commands = False
    reviewer = Reviewer(verdict, callback=drift)
    result = invoke(case, reviewer, actions=[{"tool": "run_shell", "arguments": {"command": "marker"}},
                                           {"tool": "finish", "arguments": {"report": "No execution"}}])
    assert result.status == "completed"
    assert not (case.root / "effect.txt").exists()
    receipt, = receipts(case)
    assert receipt["value"]["execution_status"] == "not_executed"
    assert receipt["permission_review"]["status"] == "invalidated"


def test_reviewer_receives_copies_and_cannot_mutate_the_prepared_operation(case):
    def mutate(operation, request):
        operation.arguments["command"] = "different"
        operation.context["launch_identity"] = "fake"
        request.permissions.execute_commands = False
    assert invoke(case, Reviewer(callback=mutate)).status == "completed"
    assert (case.root / "effect.txt").read_text() == "marker\n"
    assert case.request.permissions.execute_commands is True


@pytest.mark.parametrize("error,code", [(BudgetExhaustedError("used"), "budget_exhausted"),
                                      (DeadlineExceededError("expired"), "timeout")])
def test_shared_budget_and_deadline_errors_stop_without_manual_bypass(case, error, code):
    result = invoke(case, Reviewer(error))
    assert result.status == "failed"
    assert result.error.code.value == code
    assert not (case.root / "effect.txt").exists()


def test_authoritative_ledger_failure_does_not_fall_back_to_human_approval(case):
    from resagent2_runtime.budget import UsagePersistenceError
    result = invoke(case, Reviewer(UsagePersistenceError("Run usage could not be saved")))
    assert result.status == "failed"
    assert result.error.code.value == "contract_error"
    assert result.control is None
    assert not (case.root / "effect.txt").exists()


def test_interrupt_during_review_has_one_not_executed_receipt_and_keeps_review_source(case):
    token = CancellationToken()
    with bind_cancellation(token):
        result = invoke(case, Reviewer(callback=lambda *_: token.cancel("User interrupt")))
    assert result.status == "interrupted"
    assert not (case.root / "effect.txt").exists()
    receipt, = receipts(case)
    assert receipt["ok"] is False
    assert receipt["permission_review"]["status"] == "interrupted"


def test_deadline_after_dispatch_does_not_claim_the_operation_never_started(case):
    def deadline(_):
        raise DeadlineExceededError("deadline expired after the effect")
    case.shell.callback = deadline
    result = invoke(case, Reviewer())
    assert result.status == "failed"
    assert result.error.code.value == "timeout"
    assert (case.root / "effect.txt").read_text() == "marker\n"
    receipt, = receipts(case)
    assert "unknown" in receipt["summary"]
    assert "Not executed" not in receipt["summary"]
    assert receipt["permission_review"]["status"] == "allow"


def test_tool_failure_retains_automatic_review_source(case):
    def fail(_):
        raise RuntimeError("the tool failed after writing the marker")
    case.shell.callback = fail
    result = invoke(case, Reviewer(), actions=[
        {"tool": "run_shell", "arguments": {"command": "marker"}},
        {"tool": "finish", "arguments": {"report": "The failed tool needs inspection"}},
    ])
    assert result.status == "completed"
    receipt, = receipts(case)
    assert receipt["ok"] is False
    assert receipt["permission_review"]["source"] == "auto_review"
    assert receipt["permission_review"]["status"] == "allow"


def test_review_audit_persistence_failure_prevents_dispatch(case, monkeypatch):
    original = case.store.save
    failed = False
    def save(state):
        nonlocal failed
        if not failed and any(event.type == "permission_review" and event.data["status"] == "allow"
                              for event in state.events):
            failed = True
            raise OSError("audit storage unavailable")
        original(state)
    monkeypatch.setattr(case.store, "save", save)
    result = invoke(case, Reviewer())
    assert failed and result.status == "failed"
    assert not (case.root / "effect.txt").exists()


def test_crash_after_dispatch_keeps_review_on_unknown_receipt_without_replay(case):
    def crash(_):
        raise SystemExit("simulated owner loss")
    case.shell.callback = crash
    reviewer = Reviewer()
    with pytest.raises(SystemExit):
        invoke(case, reviewer)
    assert (case.root / "effect.txt").read_text() == "marker\n"
    case.shell.callback = None
    request = case.request.model_copy(update={"parent_session_id": "session_review"})
    forbidden = Reviewer(callback=lambda *_: pytest.fail("must not review or replay lost call"))
    result = invoke(case, forbidden, request, actions=[{"tool": "finish", "arguments": {"report": "Lost call not replayed"}}])
    assert result.status == "completed"
    assert (case.root / "effect.txt").read_text() == "marker\n"
    receipt, = receipts(case)
    assert "unknown" in receipt["summary"]
    assert receipt["permission_review"]["status"] == "allow"


def test_real_prompt_reviewer_uses_the_same_wallet_and_session_usage(case, monkeypatch):
    from resagent2_components.approval_review import LLMPermissionReviewer
    from resagent2_components.python_source_evidence import collect_python_source_evidence

    # Only this fixture uses the real reviewer: prepare its actual command evidence.
    original_prepare = case.shell.prepare
    original_execute = case.shell.execute_prepared

    def prepare(arguments):
        evidence = collect_python_source_evidence(arguments.command, case.boundary)
        return {**original_prepare(arguments), "source_fingerprint": evidence.fingerprint}

    def execute_prepared(state, arguments, prepared):
        startup = dict(prepared)
        evidence = collect_python_source_evidence(arguments.command, case.boundary)
        assert startup.pop("source_fingerprint") == evidence.fingerprint
        return original_execute(state, arguments, startup)

    monkeypatch.setattr(case.shell, "prepare", prepare)
    monkeypatch.setattr(case.shell, "execute_prepared", execute_prepared)
    reviewer_client = ScriptedLLMClient(tool_turns([{"tool": "review_operation", "arguments": {
        "decision": "allow", "reason": "The marker is explicitly requested"}}]))
    reviewer = LLMPermissionReviewer(reviewer_client, max_context_tokens=6000)
    with execution_budget(max_llm_calls=2, timeout_seconds=30) as budget:
        result = invoke(case, reviewer)
        assert budget.usage.used == 2
    assert result.status == "completed"
    assert result.llm_calls == case.store.load(result.session.id).llm_calls_used == 2
    assert len(reviewer_client.contexts) == 1
