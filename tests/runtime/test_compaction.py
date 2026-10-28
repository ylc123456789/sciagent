"""Minimal native-history compaction policy and loss boundaries."""

import json
from io import BytesIO
from httpx import HTTPStatusError, Request, Response

import pytest

from resagent2_runtime.compaction import (
    CompactionPlan,
    compaction_input_text,
    plan_compaction,
)
from resagent2_runtime.context import CONTEXT_TARGET_SHARE, ContextBudgetExceeded, ContextComposer
from resagent2_runtime.llm import LLMTextResponseError, ModelProfile, OpenAICompatibleClient
from resagent2_runtime.models import NativeToolCall, ToolCallTurn
from resagent2_runtime.tool_calling import NativeToolCallError, native_input_text


def _turn(index: int, *, receipt_size: int = 240) -> ToolCallTurn:
    call_id = f"call_{index}"
    return ToolCallTurn(
        content=f"assistant-{index}",
        reasoning_content=f"reasoning-{index}",
        tool_calls=[NativeToolCall(
            id=call_id,
            name="read_file",
            arguments=json.dumps({"path": f"file-{index}.txt"}),
        )],
        tool_results={call_id: json.dumps({
            "ok": True,
            "summary": f"read-{index}",
            "value": "x" * receipt_size,
        })},
    )


def _request_tokens(turns: list[ToolCallTurn]) -> int:
    return ContextComposer.estimate_tokens(native_input_text("current", [], turns))


def test_plan_keeps_recent_complete_turns_and_does_not_mutate_history():
    turns = [_turn(index) for index in range(6)]
    before = [turn.model_dump(mode="json") for turn in turns]
    limit = _request_tokens(turns)

    plan = plan_compaction(
        current_context="current",
        schemas=[],
        turns=turns,
        max_input_tokens=limit,
    )

    assert isinstance(plan, CompactionPlan)
    assert 0 < plan.history_start < len(turns)
    payload = json.loads(plan.prompt)
    target_chars = min(4096, max(1, limit // 20)) * 4
    assert "Aim for " in payload["output_requirement"]
    assert "characters or fewer" in payload["output_requirement"]
    assert "non-empty" in payload["output_requirement"]
    compacted = payload["completed_tool_messages"]
    assert [message["role"] for message in compacted] == [
        role
        for _ in range(plan.history_start)
        for role in ("assistant", "tool")
    ]
    assert compacted[-1]["tool_call_id"] == f"call_{plan.history_start - 1}"
    assert turns[plan.history_start].tool_calls[0].id == f"call_{plan.history_start}"
    assert [turn.model_dump(mode="json") for turn in turns] == before
    assert plan.estimated_tokens == ContextComposer.estimate_tokens(
        compaction_input_text(plan.prompt)
    )


def test_previous_checkpoint_and_absolute_boundary_are_compacted_once():
    turns = [_turn(index) for index in range(8)]
    active = turns[2:]
    limit = _request_tokens(active)

    plan = plan_compaction(
        current_context="current",
        schemas=[],
        turns=turns,
        history_start=2,
        previous_summary="earlier handoff",
        max_input_tokens=limit,
    )

    payload = json.loads(plan.prompt)
    assert payload["previous_handoff"] == "earlier handoff"
    assert plan.history_start > 2
    serialized = json.dumps(payload["completed_tool_messages"])
    assert "call_0" not in serialized and "call_1" not in serialized
    assert "call_2" in serialized


def test_below_trigger_and_single_oversized_turn_have_no_safe_prefix():
    turns = [_turn(0)]
    request_tokens = _request_tokens(turns)
    no_trigger_limit = int(request_tokens / CONTEXT_TARGET_SHARE) + 1
    assert plan_compaction(
        current_context="current", schemas=[], turns=turns,
        max_input_tokens=no_trigger_limit,
    ) is None

    # Even above the trigger, the only recent turn remains one indivisible group.
    assert plan_compaction(
        current_context="current", schemas=[], turns=turns,
        max_input_tokens=max(1, request_tokens // 2),
    ) is None


def test_force_only_bypasses_trigger_and_still_keeps_complete_recent_turn():
    turns = [_turn(index, receipt_size=20) for index in range(3)]
    generous_limit = int(_request_tokens(turns) * 1.5)
    assert plan_compaction(
        current_context="", schemas=[], turns=turns,
        max_input_tokens=generous_limit,
    ) is None

    plan = plan_compaction(
        current_context="", schemas=[], turns=turns,
        max_input_tokens=generous_limit,
        force=True,
    )
    assert plan is not None
    assert 0 < plan.history_start < len(turns)
    compacted = json.loads(plan.prompt)["completed_tool_messages"]
    assert {message.get("tool_call_id") for message in compacted} >= {"call_0"}
    assert turns[plan.history_start].tool_calls[0].id == f"call_{plan.history_start}"


def test_force_cannot_compact_the_only_oversized_recent_turn():
    turn = _turn(0, receipt_size=8000)
    assert plan_compaction(
        current_context="", schemas=[], turns=[turn],
        max_input_tokens=1,
        summary_input_limit=1,
        force=True,
    ) is None


def test_summary_overflow_is_explicit_and_never_truncates_large_turn():
    turns = [_turn(0, receipt_size=8000), _turn(1, receipt_size=20)]
    limit = _request_tokens(turns)
    plan = plan_compaction(
        current_context="current", schemas=[], turns=turns,
        max_input_tokens=limit,
        summary_input_limit=limit * 2,
    )
    assert json.loads(plan.prompt)["completed_tool_messages"][1]["content"].count("x") >= 8000

    with pytest.raises(ContextBudgetExceeded, match="no history boundary was advanced"):
        plan_compaction(
            current_context="current", schemas=[], turns=turns,
            max_input_tokens=limit,
            summary_input_limit=plan.estimated_tokens - 1,
        )


def test_unfinished_tool_group_is_rejected_instead_of_partially_compacted():
    incomplete = _turn(0)
    incomplete.tool_results = {}
    with pytest.raises(NativeToolCallError, match="unfinished tool call"):
        plan_compaction(
            current_context="current",
            schemas=[],
            turns=[incomplete, _turn(1)],
            max_input_tokens=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_input_tokens": 0}, "max_input_tokens"),
        ({"max_input_tokens": 10, "summary_input_limit": 0}, "summary_input_limit"),
        ({"max_input_tokens": 10, "history_start": 3}, "history_start"),
    ],
)
def test_invalid_planning_bounds_fail_early(kwargs, message):
    with pytest.raises(ValueError, match=message):
        plan_compaction(
            current_context="current", schemas=[], turns=[_turn(0)], **kwargs,
        )


def _summary_response(
    content,
    *,
    finish_reason="stop",
    reasoning="private summary reasoning",
    tool_calls=None,
):
    message = {"content": content, "reasoning_content": reasoning}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return BytesIO(json.dumps({
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": {"completion_tokens": 12},
    }).encode())


def _summary_client(monkeypatch, tmp_path, *, trace_level="full", profile=None):
    monkeypatch.setenv("TEST_COMPACTION_KEY", "secret-not-for-trace")
    monkeypatch.setattr("resagent2_runtime.llm.time.sleep", lambda _: None)
    return OpenAICompatibleClient(
        model="summary-test",
        api_base="https://example.invalid/v1",
        api_key_env="TEST_COMPACTION_KEY",
        model_profile=profile,
        trace_dir=tmp_path / f"traces-{trace_level}",
        trace_level=trace_level,
    )


def test_summary_request_uses_exact_bounded_wire_and_shared_full_trace(
    monkeypatch, tmp_path,
):
    profile = ModelProfile(
        context_window=1000,
        reserved_output_tokens=128,
        safety_margin_tokens=64,
    )
    client = _summary_client(monkeypatch, tmp_path, profile=profile)
    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.content))
        return _summary_response("  concise handoff  ", reasoning="summary reasoning")

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    prompt = '{"previous_handoff":null,"completed_tool_messages":[]}'
    result = client.summarize_history(
        prompt,
        max_input_tokens=700,
    )

    assert result == "concise handoff"
    assert client.last_attempts == len(requests) == 1
    body = requests[0]
    status = body["messages"][-1]["content"].split("\n", 1)[1]
    wire = compaction_input_text(prompt, runtime_status=status)
    assert body["messages"] == json.loads(wire)["messages"]
    assert body["max_tokens"] == 128
    assert body["temperature"] == 0
    assert "tools" not in body
    assert "tool_choice" not in body
    assert "response_format" not in body
    row = json.loads((client.trace_dir / "llm_traces.jsonl").read_text())
    assert row["included_sections"] == ["compaction"]
    assert row["estimated_tokens"] == ContextComposer.estimate_tokens(wire)
    assert row["request_text"] == wire
    assert row["request_max_tokens"] == 128
    assert row["raw_response_text"] == "  concise handoff  "
    assert row["raw_reasoning_text"] == "summary reasoning"
    assert row["tool"] is None
    assert row["action_valid"] is None
    assert row["parsed_action"] is None
    assert row["validation_error"] is None


@pytest.mark.parametrize(
    ("content", "finish_reason", "tool_calls", "message"),
    [
        ("handoff", "length", None, "finish with stop"),
        ("", "stop", None, "non-empty text"),
        (None, "stop", None, "non-empty text"),
        ("handoff", "stop", [{"id": "unexpected"}], "must not contain tool calls"),
    ],
)
def test_invalid_summary_output_is_not_retried(
    monkeypatch, tmp_path, content, finish_reason, tool_calls, message,
):
    client = _summary_client(monkeypatch, tmp_path)
    calls = 0

    def respond(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _summary_response(
            content, finish_reason=finish_reason, tool_calls=tool_calls,
        )

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    with pytest.raises(LLMTextResponseError, match=message):
        client.summarize_history("source", max_input_tokens=500)
    assert calls == client.last_attempts == 1
    row = json.loads((client.trace_dir / "llm_traces.jsonl").read_text())
    assert row["included_sections"] == ["compaction"]
    assert row["action_valid"] is None and row["tool"] is None
    assert message in row["validation_error"]


def test_summary_input_budget_fails_before_transport(monkeypatch, tmp_path):
    profile = ModelProfile(
        context_window=256,
        reserved_output_tokens=32,
        safety_margin_tokens=32,
    )
    client = _summary_client(monkeypatch, tmp_path, profile=profile)
    calls = 0

    def respond(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _summary_response("unused")

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    with pytest.raises(ValueError, match="input budget"):
        client.summarize_history("source", max_input_tokens=0)
    with pytest.raises(ContextBudgetExceeded, match="compaction request"):
        client.summarize_history("x" * 1000, max_input_tokens=20)
    assert calls == client.last_attempts == 0
    assert not client.trace_dir.exists()


def test_summary_without_model_profile_leaves_provider_output_limit_unset(
    monkeypatch, tmp_path,
):
    client = _summary_client(monkeypatch, tmp_path)
    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.content))
        return _summary_response("handoff")

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    assert client.summarize_history("source", max_input_tokens=500) == "handoff"
    assert "max_tokens" not in requests[0]
    row = json.loads((client.trace_dir / "llm_traces.jsonl").read_text())
    assert row["request_max_tokens"] is None


def test_summary_http_429_uses_existing_no_retry_and_trace_path(monkeypatch, tmp_path):
    client = _summary_client(monkeypatch, tmp_path)
    calls = 0

    def respond(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise HTTPStatusError(
            "rate limited",
            request=Request("POST", "https://example.invalid/v1/chat/completions"),
            response=Response(429, content=b"slow down"),
        )

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    with pytest.raises(RuntimeError, match="LLM HTTP 429"):
        client.summarize_history("source", max_input_tokens=500)
    assert calls == client.last_attempts == 1
    row = json.loads((client.trace_dir / "llm_traces.jsonl").read_text())
    assert row["included_sections"] == ["compaction"]
    assert row["action_valid"] is None and row["tool"] is None
    assert "429" in row["validation_error"]


@pytest.mark.parametrize("trace_level", ["metadata", "off"])
def test_summary_metadata_and_off_traces_do_not_leak_text(
    monkeypatch, tmp_path, trace_level,
):
    client = _summary_client(monkeypatch, tmp_path, trace_level=trace_level)
    monkeypatch.setattr(
        "resagent2_runtime.llm.send_request",
        lambda *args, **kwargs: _summary_response(
            "PRIVATE_SUMMARY", reasoning="PRIVATE_REASONING",
        ),
    )
    assert client.summarize_history(
        "PRIVATE_PROMPT", max_input_tokens=500,
    ) == "PRIVATE_SUMMARY"
    if trace_level == "off":
        assert not client.trace_dir.exists()
        return
    row = json.loads((client.trace_dir / "llm_traces.jsonl").read_text())
    assert "PRIVATE_" not in json.dumps(row)
    assert row["included_sections"] == ["compaction"]
    assert row["response_sha256"]
    assert row["action_sha256"] is None
    assert row["action_valid"] is None and row["tool"] is None
