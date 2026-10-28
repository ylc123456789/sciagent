"""AgentLoop integration boundaries for minimal native-history checkpoints."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from httpx import ConnectError

import pytest
from pydantic import ValidationError

from sciagent_contracts import (AgentOwner, ErrorCode, ModuleStatus, SessionStatus)
from sciagent_runtime import AgentLoop, JsonSessionStore, ModelProfile, OpenAICompatibleClient
from sciagent_runtime.context import ContextComposer
from sciagent_runtime.models import (
    AgentState,
    HistoryCheckpoint,
    NativeToolCall,
    ToolCallTurn,
)
from sciagent_runtime.tool_calling import contract_fingerprint, native_input_text, native_tool_schemas
from tests.runtime.test_native_tool_calls import (
    _call,
    _reply,
    _request,
    _rows,
    setup,
)


def _paired_turn(index: int, *, receipt_size: int = 700) -> ToolCallTurn:
    call_id = f"history_{index}"
    return ToolCallTurn(
        content=f"assistant-{index}",
        reasoning_content=f"reasoning-{index}",
        tool_calls=[NativeToolCall(
            id=call_id,
            name="write_value",
            arguments=json.dumps({"key": f"old-{index}", "value": index}),
        )],
        tool_results={call_id: json.dumps({
            "ok": True,
            "summary": f"historical receipt {index}",
            "value": "x" * receipt_size,
        })},
    )


def _seed_state(definition, store, turns, *, checkpoint=None) -> AgentState:
    now = datetime.now(UTC)
    state = AgentState(work_request_id="work_test",
        session_id="session_native",
        agent_name=definition.name,
        owner=AgentOwner.EXECUTOR,
        run_id="run_native",
        task_id="task_native",
        attempt_number=1,
        status=SessionStatus.PAUSED,
        tool_protocol_key=definition.llm_client.tool_session_key,
        contract_fingerprint=contract_fingerprint(definition.system_prompt, native_tool_schemas(definition.tools)),
        tool_turns=turns,
        history_checkpoint=checkpoint,
        created_at=now,
        updated_at=now,
    )
    store.save(state)
    return state


def _summary(text="checkpoint summary", *, finish="stop"):
    return {
        "choices": [{
            "finish_reason": finish,
            "message": {
                "content": text,
                "reasoning_content": "summary-only reasoning",
            },
        }],
        "usage": {"completion_tokens": 20},
    }


def _exact_limit(definition, request, state) -> int:
    """Measure the request the loop can compose before its 80% trigger."""

    schemas = native_tool_schemas(definition.tools)
    context = AgentLoop()._compose_context(
        definition,
        request,
        state,
        schemas,
        1_000_000,
    )
    start = state.history_checkpoint.history_start if state.history_checkpoint else 0
    return ContextComposer.estimate_tokens(
        native_input_text(context, schemas, state.tool_turns[start:])
    )


def test_success_keeps_raw_turns_and_sends_checkpoint_with_complete_suffix(setup):
    definition, store, requests, install = setup
    definition.llm_client.model_profile = ModelProfile(
        context_window=20_000,
        reserved_output_tokens=4096,
        safety_margin_tokens=1024,
    )
    turns = [_paired_turn(index) for index in range(6)]
    before = [turn.model_dump(mode="json") for turn in turns]
    state = _seed_state(definition, store, turns)
    request = _request(parent="session_native", calls=3)
    definition = replace(definition, max_context_tokens=_exact_limit(definition, request, state))
    install([_summary(), _reply()])

    result = AgentLoop(store=store).run(
        definition, request, session_id="ignored_on_resume",
    )

    assert result.status == ModuleStatus.COMPLETED
    saved = store.load("session_native")
    checkpoint = saved.history_checkpoint
    assert checkpoint is not None
    assert 0 < checkpoint.history_start < len(turns)
    assert [turn.model_dump(mode="json") for turn in saved.tool_turns[:len(turns)]] == before
    assert len(saved.tool_turns) == len(turns) + 1  # the final finish turn is new
    assert saved.llm_calls_used == result.llm_calls == len(requests) == 2

    summary_body, action_body = requests
    assert "tools" not in summary_body and "response_format" not in summary_body
    assert summary_body["max_tokens"] == action_body["max_tokens"] == 4096
    action_wire = json.dumps(action_body, ensure_ascii=False)
    for index in range(checkpoint.history_start):
        assert f"history_{index}" not in action_wire
    for index in range(checkpoint.history_start, len(turns)):
        assert f"history_{index}" in action_wire
    current = json.dumps(action_body["messages"])
    assert "Lossy recap" in current
    assert "checkpoint summary" in current
    assert "not exact evidence" in current

    rows = _rows(definition)
    assert rows[0]["included_sections"] == ["compaction"]
    assert rows[0]["action_valid"] is None
    assert rows[0]["tool"] is None and rows[0]["parsed_action"] is None
    assert rows[0]["estimated_tokens"] == ContextComposer.estimate_tokens(
        rows[0]["request_text"]
    )
    assert rows[0]["request_max_tokens"] == 4096
    assert rows[1]["action_valid"] is True


@pytest.mark.parametrize("character", ["x", "文"])
def test_summary_above_writing_target_fits_complete_request_without_truncation(setup, character):
    definition, store, requests, install = setup
    turns = [_paired_turn(i, receipt_size=4500) for i in range(7)]
    previous = HistoryCheckpoint(history_start=1, summary="earlier checkpoint")
    _seed_state(definition, store, turns, checkpoint=previous)
    definition = replace(definition, max_context_tokens=8192)
    target_chars = min(4096, 8192 // 20) * 4
    sentinel = "END_OF_HANDOFF"
    summary = character * (target_chars + 61 - len(sentinel)) + sentinel
    goal = "Current user decision: use mul; verification is still required."
    install([_summary(summary), _reply()])

    result = AgentLoop(store=store).run(
        definition, _request(parent="session_native", calls=3, goal=goal), session_id="ignored",
    )

    assert result.status == ModuleStatus.COMPLETED
    saved = store.load("session_native")
    assert saved.history_checkpoint.history_start > previous.history_start
    assert saved.history_checkpoint.summary == summary
    assert saved.tool_turns[:len(turns)] == turns
    assert result.llm_calls == saved.llm_calls_used == len(requests) == 2
    current = json.dumps(requests[-1]["messages"], ensure_ascii=False)
    assert summary in current and goal in current
    assert current.count("Lossy recap") == 1
    row = _rows(definition)[-1]
    assert row["estimated_tokens"] == ContextComposer.estimate_tokens(row["request_text"])
    assert row["estimated_tokens"] <= 8192


def test_summary_http_retry_shares_action_call_ledger(setup):
    definition, store, requests, install = setup
    state = _seed_state(definition, store, [_paired_turn(i) for i in range(6)])
    request = _request(parent="session_native", calls=3)
    definition = replace(definition, max_context_tokens=_exact_limit(definition, request, state))
    install([ConnectError("temporary"), _summary(), _reply()])
    result = AgentLoop(store=store).run(definition, request, session_id="ignored")
    assert result.status == ModuleStatus.COMPLETED
    assert result.llm_calls == store.load("session_native").llm_calls_used == len(requests) == 3
    rows = _rows(definition)
    assert len(rows) == 2
    assert [len(row["attempts"]) for row in rows] == [2, 1]
    assert sum(row["retry_number"] + 1 for row in rows) == 3
    assert rows[0]["action_valid"] is None


def test_required_current_context_still_too_large_does_not_commit_summary(setup):
    definition, store, requests, install = setup
    turns = [_paired_turn(i) for i in range(6)]
    _seed_state(definition, store, turns)
    definition = replace(definition, max_context_tokens=2000)
    install([_summary()])
    result = AgentLoop(store=store).run(
        definition, _request(parent="session_native", calls=3, goal="x" * 15_000),
        session_id="ignored",
    )
    saved = store.load("session_native")
    assert result.error.code == ErrorCode.BUDGET_EXHAUSTED
    assert saved.history_checkpoint is None
    assert saved.tool_turns == turns
    assert result.llm_calls == len(requests) == 0
    assert not any(event.type == "compaction" for event in saved.events)


def test_checkpoint_survives_pause_and_disk_resume(setup, tmp_path):
    definition, _, requests, install = setup
    disk = JsonSessionStore(tmp_path / "sessions")
    turns = [_paired_turn(index) for index in range(6)]
    state = _seed_state(definition, disk, turns)
    first_request = _request(parent="session_native", calls=3)
    definition = replace(
        definition,
        max_context_tokens=_exact_limit(definition, first_request, state),
    )
    ask = _call(
        "ask_user",
        {"text": "Which mode?", "requested_fields": ["mode"]},
        call_id="call_pause",
    )
    install([_summary("durable checkpoint"), _reply([ask])])

    first = AgentLoop(store=disk).run(
        definition, first_request, session_id="ignored_on_resume",
    )
    assert first.status == ModuleStatus.NEEDS_USER_INPUT
    paused = JsonSessionStore(disk.root).load("session_native")
    assert paused.history_checkpoint is not None
    boundary = paused.history_checkpoint.history_start
    assert len(paused.tool_turns) == len(turns) + 1

    next_client = OpenAICompatibleClient(
        model="native-test",
        api_base="https://example.invalid/v1",
        api_key_env="TEST_NATIVE_KEY",
        trace_level="off",
    )
    resumed_definition = replace(
        definition,
        llm_client=next_client,
        max_context_tokens=128_000,
    )
    install([_reply()])
    second = AgentLoop(store=JsonSessionStore(disk.root)).run(
        resumed_definition,
        _request(parent="session_native", calls=3, goal="mode=fast").model_copy(update={"resume_answers": [_answer(first.control.question)], "control_ack": first.control.origin}),
        session_id="ignored_again",
    )

    assert second.status == ModuleStatus.COMPLETED
    resumed = JsonSessionStore(disk.root).load("session_native")
    assert resumed.history_checkpoint.history_start == boundary
    assert resumed.history_checkpoint.summary == "durable checkpoint"
    assert resumed.llm_calls_used == 3
    wire = json.dumps(requests[-1], ensure_ascii=False)
    assert "durable checkpoint" in wire and "call_pause" in wire
    assert "mode=fast" in json.dumps(requests[-1]["messages"])
    assert all(f"history_{index}" not in wire for index in range(boundary))


@pytest.mark.parametrize(
    ("response", "error_code"),
    [
        (ConnectError("offline"), ErrorCode.TOOL_FAILED),
        (_summary("x" * 20_000), ErrorCode.BUDGET_EXHAUSTED),
        (_summary(""), ErrorCode.TOOL_FAILED),
        (_summary("   "), ErrorCode.TOOL_FAILED),
        (_summary("partial handoff", finish="length"), ErrorCode.TOOL_FAILED),
    ],
)
def test_invalid_summary_or_total_context_overflow_does_not_advance_existing_boundary(
    setup, response, error_code,
):
    definition, store, requests, install = setup
    turns = [_paired_turn(index) for index in range(7)]
    original = HistoryCheckpoint(history_start=1, summary="old checkpoint")
    state = _seed_state(definition, store, turns, checkpoint=original)
    request = _request(parent="session_native", calls=2)
    definition = replace(definition, max_context_tokens=_exact_limit(definition, request, state))
    install([response])

    result = AgentLoop(store=store).run(
        definition, request, session_id="ignored_on_resume",
    )

    assert result.error.code == error_code
    saved = store.load("session_native")
    assert saved.history_checkpoint == original
    assert saved.llm_calls_used == result.llm_calls == len(requests) == 1
    assert saved.tool_turns == turns
    assert not any(event.type == "compaction" for event in saved.events)


def test_one_remaining_call_never_starts_compaction(setup):
    definition, store, requests, install = setup
    turns = [_paired_turn(index) for index in range(6)]
    state = _seed_state(definition, store, turns)
    request = _request(parent="session_native", calls=1)
    definition = replace(definition, max_context_tokens=_exact_limit(definition, request, state))
    install([])

    result = AgentLoop(store=store).run(
        definition, request, session_id="ignored_on_resume",
    )

    assert result.error.code == ErrorCode.BUDGET_EXHAUSTED
    saved = store.load("session_native")
    assert saved.history_checkpoint is None
    assert saved.llm_calls_used == result.llm_calls == 0
    assert requests == []


def test_timeout_after_summary_keeps_checkpoint_and_skips_action_call(setup):
    definition, store, requests, install = setup
    turns = [_paired_turn(index) for index in range(6)]
    state = _seed_state(definition, store, turns)
    request = _request(parent="session_native", calls=3)
    definition = replace(definition, max_context_tokens=_exact_limit(definition, request, state))
    clock = [0.0]

    def finish_summary(_body):
        clock[0] = 61.0
        return _summary("checkpoint before timeout")

    install([finish_summary])
    result = AgentLoop(store=store, clock=lambda: clock[0]).run(
        definition, request, session_id="ignored_on_resume",
    )

    assert result.error.code == ErrorCode.TIMEOUT
    saved = store.load("session_native")
    assert saved.history_checkpoint is not None
    assert saved.history_checkpoint.summary == "checkpoint before timeout"
    assert saved.llm_calls_used == result.llm_calls == len(requests) == 1
    assert any(event.type == "compaction" for event in saved.events)
    assert len(saved.tool_turns) == len(turns)


def test_checkpoint_model_requires_positive_boundary_complete_prefix_and_latest_suffix(setup):
    definition, store, _, _ = setup
    complete = [_paired_turn(0), _paired_turn(1)]
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        HistoryCheckpoint(history_start=0, summary="invalid")

    base = _seed_state(definition, store, complete)
    payload = base.model_dump(mode="json")
    payload["history_checkpoint"] = {"history_start": len(complete), "summary": "none kept"}
    with pytest.raises(ValidationError, match="retain a recent complete turn"):
        AgentState.model_validate(payload)

    incomplete = _paired_turn(0)
    incomplete.tool_results = {}
    payload["tool_turns"] = [
        incomplete.model_dump(mode="json"),
        _paired_turn(1).model_dump(mode="json"),
    ]
    payload["history_checkpoint"] = {"history_start": 1, "summary": "bad prefix"}
    with pytest.raises(ValidationError, match="unfinished tool calls"):
        AgentState.model_validate(payload)


def test_single_giant_latest_turn_fails_without_lossy_compaction(setup):
    definition, store, requests, install = setup
    latest = _paired_turn(0, receipt_size=20_000)
    _seed_state(definition, store, [latest])
    definition = replace(definition, max_context_tokens=1000)
    install([])

    result = AgentLoop(store=store).run(
        definition,
        _request(parent="session_native", calls=3),
        session_id="ignored_on_resume",
    )

    assert result.error.code == ErrorCode.BUDGET_EXHAUSTED
    saved = store.load("session_native")
    assert saved.history_checkpoint is None
    assert saved.tool_turns[0] == latest
    assert saved.llm_calls_used == result.llm_calls == 0
    assert requests == []


def _answer(question):
    from tests.runtime.test_resume import _answer as answer
    return answer(question)
