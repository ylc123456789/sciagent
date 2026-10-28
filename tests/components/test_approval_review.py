"""Bounded review-local inspection, faithful authority inputs and shared controls."""

import json
from datetime import UTC, datetime, timedelta
from io import BytesIO

import httpx
import pytest
from pydantic import ValidationError

from sciagent_components.approval_review import LLMPermissionReviewer
from sciagent_components.python_source_evidence import (
    MAX_FILE_BYTES, MAX_FILES, MAX_TOTAL_BYTES, collect_python_source_evidence,
)
from sciagent_components.workspace import WorkspaceBoundary
from sciagent_contracts import (
    ActionSnapshot, AgentPermissions, AgentRequest, BusinessInput, RecordedAnswer,
    TaskBudget, WorkspaceAccess, WorkspaceGrant, WorkspaceSourceKind,
)
from sciagent_runtime import (
    ApprovalReviewVerdict, NativeToolCall, OpenAICompatibleClient, ScriptedLLMClient,
    ToolCallTurn,
)
from sciagent_runtime.budget import (
    BudgetExhaustedError, DeadlineExceededError, MemoryUsage, UsagePersistenceError, execution_budget,
)
from sciagent_runtime.cancellation import (
    CancellationToken, OperationCancelled, bind_cancellation,
)
from sciagent_runtime.context import ContextBudgetExceeded
from sciagent_runtime.tool_calling import NativeToolCallError


def _turn(value=None, *, name="review_operation", extra_call=False):
    value = value if value is not None else {"decision": "allow", "reason": "Within the stated task and scope"}
    calls = [NativeToolCall(id="review_1", name=name, arguments=json.dumps(value))]
    if extra_call:
        calls.append(NativeToolCall(id="review_2", name="review_operation", arguments=json.dumps(value)))
    return ToolCallTurn(tool_calls=calls)


@pytest.fixture
def operation(agent_request):
    command = "printf '%s' 'literal; $(do-not-run)'"
    evidence = collect_python_source_evidence(command, WorkspaceBoundary(agent_request.workspace))
    return ActionSnapshot(
        action_id="action_review", tool="run_shell", run_id="run_review",
        session_id="session_worker", task_id="task_review", attempt_number=1,
        arguments={"command": command, "timeout_seconds": 12},
        context={"cwd": agent_request.workspace.root, "source_fingerprint": evidence.fingerprint,
                 "launch_env_sha256": "a" * 64,
                 "shell": "/bin/bash", "argv": ["/bin/bash", "-lc"],
                 "visible_side_effects": ["writes only the declared workspace"]},
    )


@pytest.fixture
def agent_request(tmp_path):
    return AgentRequest(work_request_id="work_test",
        run_id="run_review", task_id="task_review", attempt_number=1,
        agent="executor", instruction="  Keep this original instruction.\nUse the granted workspace only. Respect the original goal. No external transfer.  ",
        budget=TaskBudget(max_llm_calls=10, timeout_seconds=30),
        permissions=AgentPermissions(execute_commands=True),
        workspace=WorkspaceGrant(root=str(tmp_path), source=WorkspaceSourceKind.LOCAL,
            access=WorkspaceAccess(read_paths=["."], write_paths=["outputs"], denied_paths=["secrets"])),
        upstream_session_id="session_parent",
    )


def _review(client, operation, agent_request, **kwargs):
    return LLMPermissionReviewer(client, max_context_tokens=20_000, **kwargs).review(operation, agent_request)


@pytest.mark.parametrize("decision,tool,has_workspace", [
    ("allow", "run_shell", True), ("ask", "run_shell", True),
    ("allow", "delete_path", True), ("allow", "delete_path", False),
])
def test_final_output_schema_and_stateless_history(operation, agent_request, decision, tool, has_workspace):
    if not has_workspace:
        agent_request = agent_request.model_copy(update={"workspace": None})
    if tool != "run_shell":
        operation = operation.model_copy(update={"tool": tool, "arguments": {"path": "old.txt"}, "context": {}})
    client = ScriptedLLMClient([_turn({"decision": decision, "reason": "Exact operation reviewed"}), _turn()])
    reviewer = LLMPermissionReviewer(client, max_context_tokens=20_000)
    verdict = reviewer.review(operation, agent_request)
    assert verdict == ApprovalReviewVerdict(decision=decision, reason="Exact operation reviewed")
    reviewer.review(operation, agent_request)
    assert client.histories == [[], []]
    expected = {"review_operation", "review_read_file", "review_list_files"} if tool == "run_shell" else {"review_operation"}
    for schemas in client.schemas:
        by_name = {item["function"]["name"]: item["function"]["parameters"] for item in schemas}
        assert set(by_name) == expected
        schema = by_name["review_operation"]
        assert schema["additionalProperties"] is False
        assert set(schema["properties"]) == {"decision", "reason"}
        assert schema["properties"]["decision"]["enum"] == ["allow", "ask"]
        for name in expected - {"review_operation"}:
            assert by_name[name]["additionalProperties"] is False
            assert set(by_name[name]["properties"]) == {"path"}
            assert by_name[name]["required"] == ["path"]


def test_current_task_and_snapshot_are_lossless_without_lower_history_or_approvals(operation, agent_request, tmp_path):
    now = datetime(2026, 10, 16, tzinfo=UTC)
    original = "  Later user correction\nkeep whitespace.  "
    agent_request.upstream_inputs = [
        BusinessInput(event_id="input_later", text=original, source="user", created_at=now + timedelta(seconds=2)),
        BusinessInput(event_id="input_old", text="Parent guidance", source="parent", source_session_id="session_parent", created_at=now),
    ]
    answer = RecordedAnswer(
        question_id="question_business", values={"choice": "Use the original boundary"},
        answered_at=now + timedelta(seconds=1), question_text="Which boundary?",
        requested_fields=["choice"], source="parent", source_session_id="session_parent",
        run_id=agent_request.run_id, session_id="session_worker", task_id=agent_request.task_id, attempt_number=1,
    )
    agent_request.user_answers = [answer]
    agent_request.input_ids = ["input_old", "question_business", "input_later"]
    agent_request.resume_answers = [answer.model_copy(update={
        "question_id": "question_approval", "action": operation,
        "values": {"choice": "APPROVAL_NOT_AUTHORIZATION"}, "comment": "SECRET_APPROVAL_COMMENT",
    })]
    agent_request.runtime_notices = [{"message": "LOWER_HISTORY_MUST_NOT_ENTER"}]
    agent_request.child_questions = []
    (tmp_path / "unread.txt").write_text("UNREAD_FILE_CANNOT_AUTHORIZE")
    client = ScriptedLLMClient([_turn()])
    _review(client, operation, agent_request)
    context = client.contexts[0]
    payload = json.loads(context.task_context.split("\n", 1)[1])
    task = payload["current_task"]["requirements"]
    assert task["instruction"] == agent_request.instruction
    assert task["task_context"] == agent_request.task_context
    assert payload["current_task"]["authority"]["workspace"] == agent_request.workspace.model_dump(mode="json")
    assert "workspace" not in task
    assert [(item["type"], item["event_id"]) for item in task["business_dialogue"]] == [
        ("input", "input_old"), ("answer", "question_business"), ("input", "input_later"),
    ]
    assert task["business_dialogue"][2]["record"]["text"] == original
    assert task["business_dialogue"][1]["record"] == answer.model_dump(mode="json", exclude_none=True)
    assert payload["proposed_operation"] == operation.model_dump(mode="json")
    assert payload["current_task"]["invocation"] == {
        "agent": "executor", "upstream_session_id": "session_parent", "authorized_input_artifact_ids": [],
    }
    for canary in ["APPROVAL_NOT_AUTHORIZATION", "SECRET_APPROVAL_COMMENT"]:
        assert canary not in json.dumps(payload["current_task"])
        assert canary in json.dumps(payload["prior_operation_decisions"])
    assert payload["prior_operation_decisions"]["records"] == [agent_request.resume_answers[0].model_dump(mode="json")]
    for canary in ["LOWER_HISTORY_MUST_NOT_ENTER", "UNREAD_FILE_CANNOT_AUTHORIZE"]:
        assert canary not in context.text
    assert payload["root_user_context"]["requirements"] is None


def test_root_context_provider_keeps_original_sources_and_is_refreshed_per_review(operation, agent_request):
    original = {"goal": "  User original\n", "constraints": ["do not evaluate withheld data"],
                "business_dialogue": [{"source": "user", "event_id": "user_1", "text": "new boundary", "created_at": "2026-10-16T00:00:00+00:00"}]}
    calls = []

    def provide(run_id):
        calls.append(run_id)
        return original

    client = ScriptedLLMClient([_turn(), _turn()])
    reviewer = LLMPermissionReviewer(client, max_context_tokens=20_000, context_provider=provide)
    reviewer.review(operation, agent_request)
    original["constraints"].append("later requirement")
    reviewer.review(operation, agent_request)
    assert calls == [agent_request.run_id, agent_request.run_id]
    payloads = [json.loads(context.task_context.split("\n", 1)[1]) for context in client.contexts]
    assert payloads[0]["root_user_context"]["requirements"]["constraints"] == ["do not evaluate withheld data"]
    assert payloads[1]["root_user_context"]["requirements"] == original
    assert payloads[1]["current_task"]["requirements"]["instruction"] == agent_request.instruction


@pytest.mark.parametrize("value", [None, {}, [], "not a mapping"])
def test_missing_root_context_never_silently_becomes_empty_authorization(operation, agent_request, value):
    client = ScriptedLLMClient([_turn()])
    with pytest.raises(ValueError, match="root user review context is unavailable"):
        _review(client, operation, agent_request, context_provider=lambda _: value)
    assert not client.contexts


def test_provider_failure_propagates_before_model_call(operation, agent_request):
    client = ScriptedLLMClient([_turn()])

    def fail(_):
        raise OSError("root store unavailable")

    with pytest.raises(OSError, match="root store unavailable"):
        _review(client, operation, agent_request, context_provider=fail)
    assert not client.contexts


def test_exact_denial_and_reason_are_separate_history_not_new_authorization(operation, agent_request):
    original_reason = "  No.\nThis operation exceeds my allowed boundary.  "
    agent_request.resume_answers = [RecordedAnswer(
        question_id="question_denied", values={"approve": "no"}, answered_at=datetime(2026, 10, 16, tzinfo=UTC),
        question_text="Approve this exact operation?", requested_fields=["approve"],
        run_id=agent_request.run_id, session_id="session_worker", task_id=agent_request.task_id,
        attempt_number=1, action=operation, comment=original_reason,
    )]
    client = ScriptedLLMClient([_turn()])
    _review(client, operation, agent_request)
    context = client.contexts[0]
    payload = json.loads(context.task_context.split("\n", 1)[1])
    record = payload["prior_operation_decisions"]["records"][0]
    assert record["action"] == operation.model_dump(mode="json")
    assert record["values"] == {"approve": "no"}
    assert record["comment"] == original_reason
    assert "question_denied" not in json.dumps(payload["current_task"])


def test_cancellation_during_context_loading_stops_before_provider_dispatch(operation, agent_request):
    token = CancellationToken()
    client = ScriptedLLMClient([_turn()])

    def provide(_):
        token.cancel("interrupted while loading review context")
        return {"goal": "User goal"}

    with bind_cancellation(token), pytest.raises(OperationCancelled):
        _review(client, operation, agent_request, context_provider=provide)
    assert not client.contexts


def test_root_operation_history_keeps_cross_task_denials_separate_from_requirements(operation, agent_request):
    historical = {
        "question_id": "question_other_task", "source": "user", "task_id": "task_other",
        "action": operation.model_copy(update={"task_id": "task_other"}).model_dump(mode="json"),
        "values": {"approve": "no"}, "decision": "no", "comment": "  Original refusal\n",
        "answered_at": "2026-10-16T00:00:00Z",
    }
    root_context = {"initial_request": {"goal": "Root goal"}, "business_dialogue": [],
                    "prior_operation_decisions": [historical]}
    client = ScriptedLLMClient([_turn()])
    _review(client, operation, agent_request, context_provider=lambda _: root_context)
    payload = json.loads(client.contexts[0].task_context.split("\n", 1)[1])
    assert payload["prior_operation_decisions"]["root_records"] == [historical]
    assert payload["prior_operation_decisions"]["records"] == []
    assert payload["root_user_context"]["requirements"] == {
        "initial_request": {"goal": "Root goal"}, "business_dialogue": [],
    }
    assert "prior_operation_decisions" in root_context  # the provider's source object is untouched


@pytest.mark.parametrize("value", [
    {"decision": "deny", "reason": "unsupported"},
    {"decision": "allow", "reason": "   "},
    {"decision": "allow"},
    {"decision": "allow", "reason": "ok", "execute": True},
])
def test_invalid_verdict_is_rejected_without_repair_or_allow_fallback(operation, agent_request, value):
    client = ScriptedLLMClient([_turn(value), _turn()])
    with pytest.raises(ValidationError):
        _review(client, operation, agent_request)
    assert len(client.contexts) == 1


@pytest.mark.parametrize("turn", [_turn(name="run_shell"), _turn(extra_call=True), ToolCallTurn(content="allow")])
def test_wrong_tool_multiple_calls_and_plain_text_never_authorize(operation, agent_request, turn):
    client = ScriptedLLMClient([turn])
    with pytest.raises(NativeToolCallError):
        _review(client, operation, agent_request)
    assert len(client.contexts) == 1


def test_context_overflow_preserves_complete_inputs_and_makes_no_model_call(operation, agent_request):
    client = ScriptedLLMClient([_turn()])
    reviewer = LLMPermissionReviewer(client, max_context_tokens=1)
    with pytest.raises(ContextBudgetExceeded):
        reviewer.review(operation, agent_request)
    assert not client.contexts


def test_shared_budget_is_not_reset_or_given_a_separate_wallet(operation, agent_request):
    client = ScriptedLLMClient([_turn(), _turn(), _turn()])
    reviewer = LLMPermissionReviewer(client, max_context_tokens=20_000)
    usage = MemoryUsage()
    with execution_budget(max_llm_calls=2, timeout_seconds=30, usage=usage):
        reviewer.review(operation, agent_request)
        reviewer.review(operation, agent_request)
        with pytest.raises(BudgetExhaustedError):
            reviewer.review(operation, agent_request)
    assert usage.used == len(client.contexts) == 2
    assert set(usage.requests.values()) == {"succeeded"}
    assert '"llm_calls_used": 1' in client.contexts[1].runtime_status
    assert '"max_llm_calls": 2' in client.contexts[1].runtime_status


def test_shared_deadline_is_not_extended(operation, agent_request):
    client = ScriptedLLMClient([_turn()])
    with execution_budget(max_llm_calls=2, timeout_seconds=0):
        with pytest.raises(DeadlineExceededError):
            _review(client, operation, agent_request)
    assert not client.contexts


def test_cancelled_scope_does_not_call_provider_or_model(operation, agent_request):
    client = ScriptedLLMClient([_turn()])
    provider_calls = []
    token = CancellationToken()
    token.cancel()
    with bind_cancellation(token), pytest.raises(OperationCancelled):
        _review(client, operation, agent_request, context_provider=lambda run_id: provider_calls.append(run_id))
    assert provider_calls == client.contexts == []


def _wire_response():
    return BytesIO(json.dumps({"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "tool_calls": [{"id": "review_wire", "type": "function", "function": {
            "name": "review_operation", "arguments": '{"decision":"allow","reason":"Within scope"}',
        }}],
    }}]}).encode())


def _http_client(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_TEST_KEY", "test-review-secret")
    return OpenAICompatibleClient(model="review-model", api_base="https://example.invalid/v1",
        api_key_env="REVIEW_TEST_KEY", trace_dir=tmp_path / "trace", trace_level="full")


@pytest.mark.parametrize("failure", ["transport", "http500"])
def test_each_operation_allows_only_one_http_attempt_even_on_reusable_client(operation, agent_request, monkeypatch, tmp_path, failure):
    client = _http_client(monkeypatch, tmp_path)
    reviewer = LLMPermissionReviewer(client, max_context_tokens=20_000)
    requests = []

    def fail(wire, **kwargs):
        requests.append(wire)
        if failure == "http500":
            raise httpx.HTTPStatusError("server error", request=wire, response=httpx.Response(500))
        raise httpx.TransportError("connection failed")

    monkeypatch.setattr("sciagent_runtime.llm.send_request", fail)
    with execution_budget(max_llm_calls=8, timeout_seconds=30) as budget:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="after 1 attempts"):
                reviewer.review(operation, agent_request)
        assert budget.usage.used == 2
    assert len(requests) == 2
    records = [json.loads(line) for line in (client.trace_dir / "llm_traces.jsonl").read_text().splitlines()]
    assert [len(row["attempts"]) for row in records] == [1, 1]


def test_trace_correlates_exact_operation_and_late_cancelled_verdict_cannot_return(operation, agent_request, monkeypatch, tmp_path):
    client = _http_client(monkeypatch, tmp_path)
    reviewer = LLMPermissionReviewer(client, max_context_tokens=20_000)
    reviewer.set_trace_context(agent="permission_reviewer", run_id=agent_request.run_id,
        task_id=agent_request.task_id, session_id=operation.session_id, action_id=operation.action_id)
    token = CancellationToken()

    def respond(wire, **kwargs):
        token.cancel("interrupted during review")
        return _wire_response()

    monkeypatch.setattr("sciagent_runtime.llm.send_request", respond)
    with bind_cancellation(token), execution_budget(max_llm_calls=3, timeout_seconds=30) as budget:
        with pytest.raises(OperationCancelled, match="interrupted during review"):
            reviewer.review(operation, agent_request)
        assert budget.usage.used == 1
    record = json.loads((client.trace_dir / "llm_traces.jsonl").read_text().splitlines()[0])
    assert record["agent"] == "permission_reviewer"
    assert record["action_id"] == operation.action_id
    assert record["run_id"] == agent_request.run_id
    assert len(record["attempts"]) == 1
    wire = json.loads(record["request_text"])
    assert {tool["function"]["name"] for tool in wire["tools"]} == {
        "review_operation", "review_read_file", "review_list_files",
    }
    assert "test-review-secret" not in json.dumps(record)


def _inspection(name, path, call_id="inspection_1"):
    return ToolCallTurn(tool_calls=[NativeToolCall(id=call_id, name=name,
        arguments=json.dumps({"path": path}))])


def _review_payloads(client):
    return [json.loads(context.task_context.split("\n", 1)[1]) for context in client.contexts]


def _checked_result(result, *, decision):
    from sciagent_runtime.models import ApprovalReviewResult

    assert isinstance(result, ApprovalReviewResult)
    assert result.verdict.decision == decision
    return result.context


def _assert_recollected_context(operation, agent_request, context):
    evidence = collect_python_source_evidence(operation.arguments["command"],
        WorkspaceBoundary(agent_request.workspace), inspections=context["review_inspections"])
    assert context["source_fingerprint"] == evidence.fingerprint
    assert context["source_fingerprint"] != operation.context["source_fingerprint"]
    return evidence


def test_read_code_then_allow_returns_evidence_bound_context_without_executing_or_approving(
    operation, agent_request, tmp_path, monkeypatch,
):
    source = tmp_path / "helper.py"
    marker = tmp_path / "would_execute"
    body = f"from pathlib import Path\nPath({str(marker)!r}).write_text('must not run')\n"
    source.write_text(body)

    def no_approval(*args, **kwargs):
        pytest.fail("review inspection must not recursively request operation approval")

    monkeypatch.setattr("sciagent_capabilities.permissions.OperationPermissionPolicy.check", no_approval)
    client = ScriptedLLMClient([_inspection("review_read_file", "helper.py"), _turn()])
    result = _review(client, operation, agent_request)
    context = _checked_result(result, decision="allow")
    assert context["review_inspections"] == [{"kind": "file", "path": "helper.py"}]
    assert "review_inspections" not in operation.context
    assert not marker.exists()
    payloads = _review_payloads(client)
    assert body not in json.dumps(payloads[0])
    assert payloads[1] == payloads[0]
    evidence = json.loads(client.histories[1][0].tool_results["inspection_1"])
    assert evidence["inspection"] == {
        "kind": "file", "path": "helper.py", "status": "available", "source_path": source.as_posix(),
    }
    assert next(item for item in evidence["files"] if item["path"] == source.as_posix())["content"] == body
    assert [len(history) for history in client.histories] == [0, 1]
    assert len(client.histories[1][0].tool_results) == 1
    _assert_recollected_context(operation, agent_request, context)
    source.write_text("CHANGED = True\n")
    changed = collect_python_source_evidence(operation.arguments["command"],
        WorkspaceBoundary(agent_request.workspace), inspections=context["review_inspections"])
    assert changed.fingerprint != context["source_fingerprint"]


def test_list_directory_then_read_file_then_ask_binds_both_observations(operation, agent_request, tmp_path):
    folder = tmp_path / "inspect"
    folder.mkdir()
    source = folder / "driver.sh"
    body = "# REVIEWED_DRIVER_BODY\nexternal_program --upload\n"
    source.write_text(body)
    client = ScriptedLLMClient([
        _inspection("review_list_files", "inspect", "list_1"),
        _inspection("review_read_file", "inspect/driver.sh", "read_1"),
        _turn({"decision": "ask", "reason": "The external program's effects remain unknown"}),
    ])
    result = _review(client, operation, agent_request)
    context = _checked_result(result, decision="ask")
    assert context["review_inspections"] == [
        {"kind": "directory", "path": "inspect"}, {"kind": "file", "path": "inspect/driver.sh"},
    ]
    payloads = _review_payloads(client)
    assert payloads[0] == payloads[1] == payloads[2]
    listed = json.loads(client.histories[1][0].tool_results["list_1"])
    assert listed["inspection"]["status"] == "available"
    assert listed["inspection"]["listing"]["children"][0]["name"] == "driver.sh"
    assert listed["inspection"]["listing"]["expansion"]["state"] == "complete"
    assert body not in json.dumps(listed)
    read = json.loads(client.histories[2][1].tool_results["read_1"])
    assert read["inspection"]["source_path"] == source.as_posix()
    assert next(item for item in read["files"] if item["path"] == source.as_posix())["content"] == body
    assert [len(history) for history in client.histories] == [0, 1, 2]
    _assert_recollected_context(operation, agent_request, context)
    (folder / "new_after_review.txt").touch()
    changed = collect_python_source_evidence(operation.arguments["command"],
        WorkspaceBoundary(agent_request.workspace), inspections=context["review_inspections"])
    assert changed.fingerprint != context["source_fingerprint"]


@pytest.mark.parametrize("kind,location", [("file", "denied"), ("directory", "outside")])
def test_denied_or_outside_inspection_records_failure_without_disclosure_or_approval(
    operation, agent_request, tmp_path, monkeypatch, kind, location,
):
    denied = tmp_path / "secrets"
    denied.mkdir()
    (denied / "hidden_name_canary.txt").write_text("DENIED_CONTENT_CANARY")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "hidden_outside_name.txt").write_text("OUTSIDE_CONTENT_CANARY")
    path = "secrets" if location == "denied" else f"../{outside.name}"
    if kind == "file":
        path += "/hidden_name_canary.txt" if location == "denied" else "/hidden_outside_name.txt"
    tool = "review_read_file" if kind == "file" else "review_list_files"

    def no_approval(*args, **kwargs):
        pytest.fail("an unavailable inspection must not create an operation approval")

    monkeypatch.setattr("sciagent_capabilities.permissions.OperationPermissionPolicy.check", no_approval)
    client = ScriptedLLMClient([_inspection(tool, path),
        _turn({"decision": "ask", "reason": "The requested implementation evidence is unavailable"})])
    result = _review(client, operation, agent_request)
    context = _checked_result(result, decision="ask")
    assert context["review_inspections"] == [{"kind": kind, "path": path}]
    evidence = json.loads(client.histories[1][0].tool_results["inspection_1"])
    assert evidence["inspection"]["status"] == "unavailable"
    assert not evidence["files"]
    model_input = json.dumps([_review_payloads(client), client.histories[1][0].tool_results])
    assert "DENIED_CONTENT_CANARY" not in model_input
    assert "OUTSIDE_CONTENT_CANARY" not in model_input
    if kind == "directory":
        assert "hidden_name_canary.txt" not in model_input
        assert "hidden_outside_name.txt" not in model_input
    _assert_recollected_context(operation, agent_request, context)


@pytest.mark.parametrize("limit", ["file_bytes", "total_bytes", "files"])
def test_inspection_uses_existing_evidence_limits(operation, agent_request, tmp_path, limit):
    command = operation.arguments["command"]
    if limit != "file_bytes":
        count = MAX_FILES - 1 if limit == "files" else MAX_TOTAL_BYTES // MAX_FILE_BYTES - 1
        entry = "import " + ", ".join(f"part_{i}" for i in range(count)) + "\n"
        if limit == "total_bytes":
            entry += "#" * (MAX_FILE_BYTES - len(entry))
        (tmp_path / "entry.py").write_text(entry)
        for i in range(count):
            (tmp_path / f"part_{i}.py").write_text("#" * MAX_FILE_BYTES if limit == "total_bytes" else "VALUE = 1\n")
        command = "python3 entry.py"
    secret = "OVER_LIMIT_BODY_MUST_NOT_BE_DISCLOSED"
    (tmp_path / "supplement.txt").write_text(secret + "x" * MAX_FILE_BYTES if limit == "file_bytes" else secret)
    boundary = WorkspaceBoundary(agent_request.workspace)
    prepared = collect_python_source_evidence(command, boundary)
    operation = operation.model_copy(update={
        "arguments": {**operation.arguments, "command": command},
        "context": {**operation.context, "source_fingerprint": prepared.fingerprint},
    })
    client = ScriptedLLMClient([_inspection("review_read_file", "supplement.txt"),
        _turn({"decision": "ask", "reason": "Inspection exceeds the bounded evidence allowance"})])
    result = LLMPermissionReviewer(client, max_context_tokens=200_000).review(operation, agent_request)
    context = _checked_result(result, decision="ask")
    evidence = json.loads(client.histories[1][0].tool_results["inspection_1"])
    assert evidence["inspection"]["status"] == "unavailable"
    assert len(evidence["files"]) <= MAX_FILES
    assert sum(item["size_bytes"] for item in evidence["files"]) <= MAX_TOTAL_BYTES
    assert all(item["size_bytes"] <= MAX_FILE_BYTES for item in evidence["files"])
    assert secret not in json.dumps([evidence, client.histories[1][0].tool_results])
    _assert_recollected_context(operation, agent_request, context)


def test_fourth_inspection_request_is_system_ask_without_fifth_provider_call(operation, agent_request, tmp_path):
    (tmp_path / "helper.txt").write_text("Current implementation evidence")
    client = ScriptedLLMClient([
        _inspection("review_read_file", "helper.txt", f"read_{i}") for i in range(5)
    ])
    with execution_budget(max_llm_calls=10, timeout_seconds=30) as budget:
        result = _review(client, operation, agent_request)
        assert budget.usage.used == 4
    context = _checked_result(result, decision="ask")
    assert len(context["review_inspections"]) == 3
    assert len(client.contexts) == 4
    assert [len(history) for history in client.histories] == [0, 1, 2, 3]
    assert {item["function"]["name"] for item in client.schemas[-1]} == {"review_operation"}
    _assert_recollected_context(operation, agent_request, context)


def test_inspection_does_not_reset_shared_budget_before_next_review_turn(operation, agent_request, tmp_path):
    (tmp_path / "helper.txt").write_text("Available evidence")
    client = ScriptedLLMClient([_inspection("review_read_file", "helper.txt"), _turn()])
    with execution_budget(max_llm_calls=1, timeout_seconds=30) as budget:
        with pytest.raises(BudgetExhaustedError):
            _review(client, operation, agent_request)
        assert budget.usage.used == 1
    assert len(client.contexts) == 1


@pytest.mark.parametrize("cancel_at", ["model", "inspection"])
def test_cancellation_between_inspection_turns_never_returns_a_verdict(
    operation, agent_request, tmp_path, monkeypatch, cancel_at,
):
    (tmp_path / "helper.txt").write_text("Available evidence")
    token = CancellationToken()

    class CancelClient(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            result = super().next_tool_call(*args, **kwargs)
            if cancel_at == "model":
                token.cancel("cancel review before inspection")
            return result

    resolve = WorkspaceBoundary.resolve_read_file

    def cancel_read(self, path):
        if cancel_at == "model":
            pytest.fail("cancelled reviewer must not dispatch the requested inspection")
        result = resolve(self, path)
        token.cancel("cancel review during inspection")
        return result

    monkeypatch.setattr(WorkspaceBoundary, "resolve_read_file", cancel_read)
    client = CancelClient([_inspection("review_read_file", "helper.txt"), _turn()])
    with bind_cancellation(token), pytest.raises(OperationCancelled):
        _review(client, operation, agent_request)
    assert len(client.contexts) == 1


def test_review_inspection_history_and_evidence_do_not_cross_review_calls(operation, agent_request, tmp_path):
    body = "UNRELATED_NEXT_REVIEW_MUST_NOT_INHERIT_THIS_BODY"
    (tmp_path / "helper.txt").write_text(body)
    client = ScriptedLLMClient([_inspection("review_read_file", "helper.txt"), _turn(), _turn()])
    reviewer = LLMPermissionReviewer(client, max_context_tokens=20_000)
    first = reviewer.review(operation, agent_request)
    _checked_result(first, decision="allow")
    second = reviewer.review(operation, agent_request)
    assert isinstance(second, ApprovalReviewVerdict)
    assert [len(history) for history in client.histories] == [0, 1, 0]
    last = _review_payloads(client)[2]
    assert not last["implementation_evidence"].get("review_inspections")
    assert body not in json.dumps(last)
    assert "review_inspections" not in operation.context


@pytest.mark.parametrize("value", [{}, {"path": "helper.txt", "execute": True}])
def test_malformed_inspection_parameters_fail_without_repair_or_allow(operation, agent_request, value):
    client = ScriptedLLMClient([_turn(value, name="review_read_file"), _turn()])
    with pytest.raises(ValueError, match="Invalid review inspection arguments"):
        _review(client, operation, agent_request)
    assert len(client.contexts) == 1


@pytest.mark.parametrize("failure", ["format", "provider", "context"])
def test_failure_after_inspection_keeps_latest_binding_for_manual_review(
    operation, agent_request, tmp_path, failure,
):
    (tmp_path / "helper.txt").write_text("Evidence that must remain bound to manual approval")

    class FailAfterInspection(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            turn = super().next_tool_call(*args, **kwargs)
            if len(self.contexts) == 1 and failure == "context":
                reviewer._max_context_tokens = 1
            if len(self.contexts) == 2 and failure == "provider":
                raise RuntimeError("PROVIDER_SECRET_MUST_NOT_ENTER_REASON")
            return turn

    invalid = {"decision": "allow", "reason": "Must not bypass validation", "supplement": {"path": "forged"}}
    client = FailAfterInspection([_inspection("review_read_file", "helper.txt"), _turn(invalid)])
    reviewer = LLMPermissionReviewer(client, max_context_tokens=20_000)
    result = reviewer.review(operation, agent_request)
    context = _checked_result(result, decision="ask")
    assert context["review_inspections"] == [{"kind": "file", "path": "helper.txt"}]
    assert len(client.contexts) == (1 if failure == "context" else 2)
    exception_name = {"format": "ValidationError", "provider": "RuntimeError", "context": "ContextBudgetExceeded"}[failure]
    assert exception_name in result.verdict.reason
    assert "PROVIDER_SECRET" not in result.verdict.reason
    assert "forged" not in json.dumps(context)
    _assert_recollected_context(operation, agent_request, context)


@pytest.mark.parametrize("error_type", [BudgetExhaustedError, DeadlineExceededError,
                                        UsagePersistenceError, OperationCancelled])
def test_control_or_usage_failure_after_inspection_is_never_converted_to_manual_review(
    operation, agent_request, tmp_path, error_type,
):
    (tmp_path / "helper.txt").write_text("Current evidence")
    error = error_type("Stop this review")

    class StopAfterInspection(ScriptedLLMClient):
        def next_tool_call(self, *args, **kwargs):
            turn = super().next_tool_call(*args, **kwargs)
            if len(self.contexts) == 2:
                raise error
            return turn

    client = StopAfterInspection([_inspection("review_read_file", "helper.txt"), _turn()])
    with pytest.raises(error_type) as captured:
        _review(client, operation, agent_request)
    assert captured.value is error
    assert len(client.contexts) == 2



def _inspection_batch(*items):
    return ToolCallTurn(tool_calls=[NativeToolCall(
        id=f"batch_{i}", name=name, arguments=json.dumps({"path": path}),
    ) for i, (name, path) in enumerate(items)])


@pytest.mark.parametrize("items", [
    [("review_read_file", "a.txt"), ("review_list_files", "folder")],
    [("review_list_files", "folder"), ("review_read_file", "a.txt")],
    [("review_read_file", "a.txt"), ("review_read_file", "b.txt")],
    [("review_read_file", "a.txt"), ("review_read_file", "b.txt"),
     ("review_read_file", "c.txt")],
])
def test_inspection_batch_is_sequential_and_fully_paired_before_final_verdict(
    operation, agent_request, tmp_path, monkeypatch, items,
):
    from sciagent_components import approval_review
    from sciagent_runtime.tool_calling import tool_messages

    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("BODY_" + name)
    (tmp_path / "folder").mkdir()
    collect = approval_review.collect_python_source_evidence
    observations = []

    def observe(*args, inspections=None, **kwargs):
        observations.append([dict(item) for item in inspections or []])
        return collect(*args, inspections=inspections, **kwargs)

    monkeypatch.setattr(approval_review, "collect_python_source_evidence", observe)
    client = ScriptedLLMClient([_inspection_batch(*items), _turn()])
    context = _checked_result(_review(client, operation, agent_request), decision="allow")
    expected = [{"kind": "file" if name == "review_read_file" else "directory", "path": path}
                for name, path in items]
    assert context["review_inspections"] == expected
    assert observations == [[]] + [prefix for i in range(len(items))
                                   for prefix in (expected[:i], expected[:i + 1])]
    assert [len(history) for history in client.histories] == [0, 1]
    turn = client.histories[1][0]
    assert list(turn.tool_results) == [call.id for call in turn.tool_calls]
    for call, item in zip(turn.tool_calls, expected, strict=True):
        receipt = json.loads(turn.tool_results[call.id])
        assert receipt["inspection"]["kind"] == item["kind"]
        assert receipt["inspection"]["path"] == item["path"]
        assert receipt["inspection"]["status"] == "available"
        if item["kind"] == "file":
            assert [source["content"] for source in receipt["files"]] == ["BODY_" + item["path"]]
        else:
            assert receipt["inspection"]["listing"]["expansion"]["state"] == "complete"
            assert receipt["files"] == []
    messages = tool_messages(client.histories[1])
    assert [message["tool_call_id"] for message in messages if message["role"] == "tool"] == [
        call.id for call in turn.tool_calls]
    assert _review_payloads(client)[1] == _review_payloads(client)[0]
    assert json.loads(turn.tool_results[turn.tool_calls[-1].id])["source_fingerprint"] == context["source_fingerprint"]
    _assert_recollected_context(operation, agent_request, context)


def test_batch_then_single_inspection_retains_independent_paired_turns(operation, agent_request, tmp_path):
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("BODY_" + name)
    client = ScriptedLLMClient([
        _inspection_batch(("review_read_file", "a.txt"), ("review_read_file", "b.txt")),
        _inspection("review_read_file", "c.txt", "last_read"), _turn(),
    ])
    with execution_budget(max_llm_calls=4, timeout_seconds=30) as budget:
        context = _checked_result(_review(client, operation, agent_request), decision="allow")
        assert budget.usage.used == 3
    assert [len(history) for history in client.histories] == [0, 1, 2]
    assert [len(turn.tool_calls) for turn in client.histories[-1]] == [2, 1]
    assert all(set(turn.tool_results) == {call.id for call in turn.tool_calls}
               for turn in client.histories[-1])
    payloads = _review_payloads(client)
    assert payloads[0] == payloads[1] == payloads[2]
    assert [json.loads(turn.tool_results[call.id])["files"][0]["content"]
            for turn in client.histories[-1] for call in turn.tool_calls] == [
        "BODY_a.txt", "BODY_b.txt", "BODY_c.txt",
    ]
    assert len(context["review_inspections"]) == 3
    assert {schema["function"]["name"] for schema in client.schemas[-1]} == {"review_operation"}
    _assert_recollected_context(operation, agent_request, context)


@pytest.mark.parametrize("names", [
    ["review_operation", "review_read_file"],
    ["review_read_file", "review_operation"],
    ["review_operation", "review_operation"],
    ["review_read_file", "run_shell"],
])
def test_mixed_final_or_unknown_tool_batch_never_reads_any_path(
    operation, agent_request, monkeypatch, names,
):
    from sciagent_components import approval_review

    collect = approval_review.collect_python_source_evidence

    def no_inspection(*args, inspections=None, **kwargs):
        assert not inspections, "an invalid batch must not read its valid prefix"
        return collect(*args, inspections=inspections, **kwargs)

    monkeypatch.setattr(approval_review, "collect_python_source_evidence", no_inspection)
    calls = [NativeToolCall(id=f"mixed_{i}", name=name, arguments=json.dumps(
        {"decision": "allow", "reason": "Must not authorize this batch"}
        if name == "review_operation" else {"path": "unread.txt"})) for i, name in enumerate(names)]
    client = ScriptedLLMClient([ToolCallTurn(tool_calls=calls), _turn()])
    with pytest.raises(NativeToolCallError):
        _review(client, operation, agent_request)
    assert len(client.contexts) == 1


@pytest.mark.parametrize("completed", [0, 1, 2])
def test_batch_larger_than_remaining_inspection_allowance_reads_nothing_from_batch(
    operation, agent_request, tmp_path, monkeypatch, completed,
):
    paths = [f"done_{i}.txt" for i in range(completed)]
    for path in paths:
        (tmp_path / path).write_text("Already reviewed")
    resolve = WorkspaceBoundary.resolve_read_file
    reads = []

    def track_read(self, path):
        reads.append(path)
        return resolve(self, path)

    monkeypatch.setattr(WorkspaceBoundary, "resolve_read_file", track_read)
    turns = ([_inspection_batch(*[("review_read_file", path) for path in paths])] if paths else [])
    excess = [f"overflow_{i}.txt" for i in range(4 - completed)]
    turns.append(_inspection_batch(*[("review_read_file", path) for path in excess]))
    client = ScriptedLLMClient(turns + [_turn()])
    with execution_budget(max_llm_calls=4, timeout_seconds=30) as budget:
        result = _review(client, operation, agent_request)
        assert budget.usage.used == len(turns)
    context = _checked_result(result, decision="ask")
    assert "inspection budget exceeded" in result.verdict.reason.lower()
    assert [item["path"] for item in context.get("review_inspections", [])] == paths
    assert not set(reads).intersection(excess)
    assert len(client.contexts) == len(turns)
    if completed:
        _assert_recollected_context(operation, agent_request, context)
    else:
        assert context == operation.context


@pytest.mark.parametrize("arguments", [
    "{}", '{"path":"b.txt","execute":true}', '{"path":" "}',
    '{"path":[]}', "null", "{broken",
])
def test_batch_prevalidates_all_parameters_before_first_inspection(
    operation, agent_request, monkeypatch, arguments,
):
    from sciagent_components import approval_review

    collect = approval_review.collect_python_source_evidence

    def no_inspection(*args, inspections=None, **kwargs):
        assert not inspections, "a later invalid argument must prevent all batch reads"
        return collect(*args, inspections=inspections, **kwargs)

    monkeypatch.setattr(approval_review, "collect_python_source_evidence", no_inspection)
    batch = _inspection_batch(("review_read_file", "a.txt"), ("review_list_files", "folder"))
    batch.tool_calls[1].arguments = arguments
    client = ScriptedLLMClient([batch, _turn()])
    with pytest.raises(ValueError):
        _review(client, operation, agent_request)
    assert len(client.contexts) == 1


@pytest.mark.parametrize("completed", [1, 2])
def test_mid_batch_failure_keeps_only_latest_completed_binding(
    operation, agent_request, tmp_path, monkeypatch, completed,
):
    from sciagent_components import approval_review

    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("BODY_" + name)
    collect = approval_review.collect_python_source_evidence

    def fail_later(*args, inspections=None, **kwargs):
        if len(inspections or []) == completed + 1:
            raise RuntimeError("PRIVATE_INSPECTION_FAILURE")
        return collect(*args, inspections=inspections, **kwargs)

    monkeypatch.setattr(approval_review, "collect_python_source_evidence", fail_later)
    client = ScriptedLLMClient([_inspection_batch(*[("review_read_file", name)
                              for name in ("a.txt", "b.txt", "c.txt")]), _turn()])
    result = _review(client, operation, agent_request)
    context = _checked_result(result, decision="ask")
    assert len(context["review_inspections"]) == completed
    assert "RuntimeError" in result.verdict.reason
    assert "PRIVATE_INSPECTION_FAILURE" not in result.verdict.reason
    assert len(client.contexts) == 1
    assert client.histories == [[]]  # no unfinished batch is submitted to the provider
    assert "review_inspections" not in operation.context
    _assert_recollected_context(operation, agent_request, context)


@pytest.mark.parametrize("changed", ["original", "inspected"])
def test_mid_batch_drift_preserves_old_binding_and_skips_remaining_inspection(
    operation, agent_request, tmp_path, monkeypatch, changed,
):
    from sciagent_components import approval_review

    entry = tmp_path / "entry.py"
    entry.write_text("ORIGINAL = True\n")
    first = tmp_path / "a.txt"
    first.write_text("FIRST_VERSION")
    boundary = WorkspaceBoundary(agent_request.workspace)
    operation = operation.model_copy(update={
        "arguments": {**operation.arguments, "command": "python3 entry.py"},
        "context": {**operation.context, "source_fingerprint": collect_python_source_evidence(
            "python3 entry.py", boundary).fingerprint},
    })
    collect = approval_review.collect_python_source_evidence
    reads = []
    changed_once = False

    def mutate_after_first(*args, inspections=None, **kwargs):
        nonlocal changed_once
        evidence = collect(*args, inspections=inspections, **kwargs)
        reads.append([dict(item) for item in inspections or []])
        if len(inspections or []) == 1 and not changed_once:
            changed_once = True
            (entry if changed == "original" else first).write_text("CHANGED_VERSION\n")
        return evidence

    monkeypatch.setattr(approval_review, "collect_python_source_evidence", mutate_after_first)
    client = ScriptedLLMClient([_inspection_batch(
        ("review_read_file", "a.txt"), ("review_read_file", "b.txt")), _turn()])
    result = _review(client, operation, agent_request)
    context = _checked_result(result, decision="ask")
    assert context["review_inspections"] == [{"kind": "file", "path": "a.txt"}]
    assert "changed during review" in result.verdict.reason
    assert all(all(item["path"] != "b.txt" for item in items) for items in reads)
    assert context["source_fingerprint"] != collect("python3 entry.py", boundary,
        inspections=context["review_inspections"]).fingerprint
    assert len(client.contexts) == 1


@pytest.mark.parametrize("error_type", [BudgetExhaustedError, DeadlineExceededError,
                                        UsagePersistenceError, OperationCancelled])
def test_control_failure_inside_batch_propagates_original_exception(
    operation, agent_request, tmp_path, monkeypatch, error_type,
):
    from sciagent_components import approval_review

    (tmp_path / "a.txt").write_text("Current evidence")
    error = error_type("Stop before the second inspection")
    collect = approval_review.collect_python_source_evidence

    def stop_second(*args, inspections=None, **kwargs):
        if len(inspections or []) == 2:
            raise error
        return collect(*args, inspections=inspections, **kwargs)

    monkeypatch.setattr(approval_review, "collect_python_source_evidence", stop_second)
    client = ScriptedLLMClient([_inspection_batch(
        ("review_read_file", "a.txt"), ("review_read_file", "b.txt")), _turn()])
    with pytest.raises(error_type) as captured:
        _review(client, operation, agent_request)
    assert captured.value is error
    assert len(client.contexts) == 1


def test_inspection_receipts_only_add_new_dependencies_but_explicit_reread_returns_body(
    operation, agent_request, tmp_path,
):
    (tmp_path / "first.py").write_text("import shared\nFIRST_BODY = 1\n")
    (tmp_path / "second.py").write_text("import shared\nSECOND_BODY = 2\n")
    (tmp_path / "shared.py").write_text("SHARED_DEPENDENCY_BODY = 3\n")
    client = ScriptedLLMClient([
        _inspection("review_read_file", "first.py", "first"),
        _inspection("review_read_file", "second.py", "second"),
        _inspection("review_read_file", "first.py", "reread"), _turn(),
    ])
    context = _checked_result(_review(client, operation, agent_request), decision="allow")
    payloads = _review_payloads(client)
    assert all(payload == payloads[0] for payload in payloads)
    receipts = [json.loads(turn.tool_results[turn.tool_calls[0].id]) for turn in client.histories[-1]]
    assert [{item["path"] for item in receipt["files"]} for receipt in receipts] == [
        {(tmp_path / "first.py").as_posix(), (tmp_path / "shared.py").as_posix()},
        {(tmp_path / "second.py").as_posix()}, {(tmp_path / "first.py").as_posix()},
    ]
    assert len(receipts[0]["imports"]) == len(receipts[1]["imports"]) == 1
    assert receipts[2]["imports"] == []
    assert receipts[-1]["source_fingerprint"] == context["source_fingerprint"]
    assert len(_assert_recollected_context(operation, agent_request, context).review_payload()["files"]) == 3


def test_directory_receipt_preserves_partial_expansion_and_missing_status(operation, agent_request, tmp_path):
    folder = tmp_path / "many"
    folder.mkdir()
    for i in range(101):
        (folder / f"item_{i:03}.txt").touch()
    client = ScriptedLLMClient([
        _inspection("review_list_files", "many", "many"),
        _inspection("review_list_files", "missing", "missing"), _turn(),
    ])
    context = _checked_result(_review(client, operation, agent_request), decision="allow")
    first, second = [json.loads(turn.tool_results[turn.tool_calls[0].id]) for turn in client.histories[-1]]
    listing = first["inspection"]["listing"]
    assert len(listing["children"]) == 100
    assert listing["expansion"]["state"] == "partial"
    assert listing["expansion"]["next_cursor"]
    assert second["inspection"]["status"] == "unavailable"
    assert second["inspection"]["reason"] == "FileNotFoundError"
    _assert_recollected_context(operation, agent_request, context)


def test_native_messages_keep_old_judgment_before_new_body_and_never_rewrite_initial_request(
    operation, agent_request, tmp_path,
):
    from sciagent_runtime.tool_calling import native_input

    body = "NEW_OBSERVED_BODY = 42\n"
    (tmp_path / "helper.py").write_text(body)
    first = _inspection("review_read_file", "helper.py")
    first.content = "Earlier judgment: implementation is missing; inspect it."
    client = ScriptedLLMClient([first, _turn()])
    _checked_result(_review(client, operation, agent_request), decision="allow")
    initial = native_input(client.contexts[0], client.schemas[0], client.histories[0])
    later = native_input(client.contexts[1], client.schemas[1], client.histories[1])
    # Runtime budget status remains the final live message; the original
    # system/request messages stay stable ahead of chronological tool history.
    initial_messages = initial["messages"][:-1]
    assert later["messages"][:len(initial_messages)] == initial_messages
    assert body not in json.dumps(initial["messages"])
    following = later["messages"][len(initial_messages):-1]
    assert [item["role"] for item in following] == ["assistant", "tool"]
    assert following[0]["content"] == first.content
    assert json.loads(following[1]["content"])["files"][0]["content"] == body


def test_raw_bytes_within_evidence_quota_still_require_complete_native_request_capacity(
    operation, agent_request, tmp_path, monkeypatch,
):
    (tmp_path / "helper.txt").write_bytes(b"a" * MAX_FILE_BYTES)
    client = _http_client(monkeypatch, tmp_path)
    sent = []

    def respond(wire, **kwargs):
        sent.append(json.loads(wire.content))
        return BytesIO(json.dumps({"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [{"id": "read_native", "type": "function", "function": {
                "name": "review_read_file", "arguments": '{"path":"helper.txt"}',
            }}],
        }}]}).encode())

    monkeypatch.setattr("sciagent_runtime.llm.send_request", respond)
    result = LLMPermissionReviewer(client, max_context_tokens=20_000).review(operation, agent_request)
    context = _checked_result(result, decision="ask")
    assert "ContextBudgetExceeded" in result.verdict.reason
    assert len(sent) == 1
    evidence = _assert_recollected_context(operation, agent_request, context)
    assert evidence.review_payload()["files"][0]["size_bytes"] == MAX_FILE_BYTES
    assert not evidence.review_payload()["gaps"] or all(
        item["code"] == "unsupported_command" for item in evidence.review_payload()["gaps"])
    assert json.dumps(agent_request.instruction, ensure_ascii=False) in sent[0]["messages"][1]["content"]


@pytest.mark.parametrize("completed", [0, 1])
def test_drift_during_expanded_recollection_cannot_bind_unseen_changed_body(
    operation, agent_request, tmp_path, monkeypatch, completed,
):
    from sciagent_components import approval_review

    source = tmp_path / "entry.py"
    source.write_text("PREVIOUSLY_PROVIDED_BODY = 1\n")
    (tmp_path / "extra.txt").write_text("new inspection")
    command = "python entry.py"
    original = collect_python_source_evidence(command, WorkspaceBoundary(agent_request.workspace))
    operation = operation.model_copy(update={
        "arguments": {**operation.arguments, "command": command},
        "context": {**operation.context, "source_fingerprint": original.fingerprint},
    })
    collect = approval_review.collect_python_source_evidence
    last_bound = []

    def change_between_prefix_and_expanded_read(*args, inspections=None, **kwargs):
        if len(inspections or []) == completed + 1:
            source.write_text("UNSEEN_CHANGED_BODY = 2\n")
        evidence = collect(*args, inspections=inspections, **kwargs)
        if len(inspections or []) == completed:
            last_bound.append(evidence.fingerprint)
        return evidence

    monkeypatch.setattr(approval_review, "collect_python_source_evidence", change_between_prefix_and_expanded_read)
    client = ScriptedLLMClient([
        *[_inspection("review_read_file", "extra.txt", f"read_{i}") for i in range(completed + 1)], _turn(),
    ])
    if completed:
        result = _review(client, operation, agent_request)
        context = _checked_result(result, decision="ask")
        assert context["source_fingerprint"] == last_bound[-1]
        assert len(context["review_inspections"]) == completed
    else:
        with pytest.raises(ValueError, match="Previously provided source evidence changed"):
            _review(client, operation, agent_request)
    assert len(client.contexts) == completed + 1
    assert "UNSEEN_CHANGED_BODY" not in json.dumps(_review_payloads(client))
