from dataclasses import replace
from sciagent_contracts import ExecutionConfig, ExecutionDefinition
"""Native argument errors recover in place; transport failures may retry a Task."""

from sciagent_contracts import AgentPermissions, RunPermissions, ExecutionLimits

import json
from uuid import uuid4
from datetime import UTC, datetime
from unittest import mock
from httpx import TransportError

import pytest

from sciagent_contracts import (AgentOwner, ErrorCode, ModuleStatus, AgentRequest, ResearchRequest, RunBudget, RunStatus, TaskProposal, TaskStatus, TaskBudget, WorkflowProposal, WorkRequest, WorkRequestDraft)
from sciagent_orchestrator import (
    InMemoryRunStore,
    ModuleBinding,
    ResearchRun,
    WorkflowScheduler,
)
from tool_fixtures import WriteValueTool

from sciagent_runtime import (
    AgentDefinition,
    AgentLoop,
    AllowListPermissionPolicy,
    CompletionDecision,
    ContextSection,
    FinishCandidate,
    FinishTool,
    InMemorySessionStore,
    OpenAICompatibleClient,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _AcceptFinish:
    def evaluate(self, state, candidate: FinishCandidate | None) -> CompletionDecision:
        if candidate is None:
            return CompletionDecision(complete=False)
        return CompletionDecision(complete=True, report="done")


def _context(request, state, max_context_tokens) -> list[ContextSection]:
    return [
        ContextSection(name="task", content=request.instruction)
    ]


@pytest.mark.parametrize("call_budget, expected_code, calls", [
    (50, ErrorCode.TOOL_FAILED, 5),
    (2, ErrorCode.BUDGET_EXHAUSTED, 2),
])
def test_bad_json_stops_at_existing_limits(
    monkeypatch, tmp_path, call_budget, expected_code, calls,
) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    client = OpenAICompatibleClient(
        model="test-model",
        api_base="https://example.com/v1",
        api_key_env="TEST_LLM_KEY",
        trace_dir=tmp_path / "traces",
        trace_level="full",
    )
    bad = _response("not valid json")
    definition = AgentDefinition(
        name="recovery",
        owner=AgentOwner.EXECUTOR,
        system_prompt="Use the provided tools only.",
        tools=(FinishTool(),),
        llm_client=client,
        context_builder=_context,
        permission_policy=AllowListPermissionPolicy({"finish"}),
        completion_check=_AcceptFinish(),
    )
    request = AgentRequest(work_request_id="work_test", run_id='run_r', task_id='task_r', attempt_number=1, agent=AgentOwner.EXECUTOR, instruction='q', budget=TaskBudget(max_llm_calls=call_budget, timeout_seconds=60), permissions=AgentPermissions(execute_commands=True))

    with (
        mock.patch("sciagent_runtime.llm.time.sleep"),
        mock.patch("sciagent_runtime.llm.send_request", side_effect=lambda *a, **k: _response("not valid json")),
    ):
        result = AgentLoop(store=InMemorySessionStore()).run(
            definition, request, session_id="session_recovery"
        )

    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    assert result.error.code == expected_code
    assert result.error.retryable is False
    assert result.llm_calls == calls
    assert client.last_attempts == 1
    assert "not valid JSON" in result.error.details["last_observation"]["summary"]

    trace_file = tmp_path / "traces" / "llm_traces.jsonl"
    record = json.loads(trace_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["raw_tool_calls"][0]["function"]["arguments"] == "not valid json"
    assert record["action_valid"] is False
    assert len(trace_file.read_text().splitlines()) == calls


class _LoopPort:
    def __init__(self, definition: AgentDefinition) -> None:
        self.definition = definition
        self.loop = AgentLoop(store=InMemorySessionStore())
        self.requests: list[AgentRequest] = []

    def invoke(self, request: AgentRequest):
        self.requests.append(request)
        frozen = request.execution_definition
        definition = replace(self.definition, execution_ref=frozen.ref, execution_config_hash=frozen.config_hash)
        return self.loop.run(
            definition, request, session_id=f"session_{request.attempt_number}"
        )


@pytest.mark.parametrize("network_failure", [False, True])
def test_scheduler_keeps_attempt_for_json_but_retries_transport(
    monkeypatch, tmp_path, network_failure,
) -> None:
    """Correct JSON in-place; unchanged transport exhaustion retries the Task."""
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    client = OpenAICompatibleClient(
        model="test-model",
        api_base="https://example.com/v1",
        api_key_env="TEST_LLM_KEY",
        trace_dir=tmp_path / "traces",
        trace_level="full",
    )
    definition = AgentDefinition(
        name="recovery",
        owner=AgentOwner.EXECUTOR,
        system_prompt="Use the provided tools only.",
        tools=(FinishTool(),),
        llm_client=client,
        context_builder=_context,
        permission_policy=AllowListPermissionPolicy({"finish"}),
        completion_check=_AcceptFinish(),
    )
    port = _LoopPort(definition)
    store = InMemoryRunStore()
    engine = WorkflowScheduler(
        bindings={
            AgentOwner.EXECUTOR: ModuleBinding(
                owner=AgentOwner.EXECUTOR,
                port=port,
            )
        },
        store=store,
        data_root=tmp_path / "data",
    )
    request = ResearchRequest(goal='exercise retry recovery', budget=RunBudget(max_llm_calls=10, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=2))
    now = datetime.now(UTC)
    store.save(
        ResearchRun(
            execution_definitions={"test@1": ExecutionDefinition(ref="test@1", config=ExecutionConfig(
                system_prompt=definition.system_prompt, tools=["finish"], preparation="fixture@1",
                completion="fixture@1", finalizer="fixture@1", context_builder="fixture@1",
                implementation_version="1", model_profile="fixture@1", max_context_tokens=256000))},
            run_id="run_recovery_chain",
            work_requests=[WorkRequest(
                id="work_recovery", run_id="run_recovery_chain", status="executing", workflow_revision=1,
                request=WorkRequestDraft(objective="exercise retry recovery", tasks=[{"selection": {"kind": "preset", "preset_ref": "test@1", "reason": "Test task capability"},
                    "key": "recovery", "instruction": "Finish after a provider failure",
                }]), created_at=now, updated_at=now,
            )],
            request=request,
            status=RunStatus.RUNNING,
            created_at=now,
            updated_at=now,
        )
    )
    engine.accept_proposal(
        "run_recovery_chain",
        WorkflowProposal(
            work_request_id="work_recovery",
            tasks=[
                TaskProposal(execution_ref="test@1",
                    id="task_recovery",
                    work_request_id="work_recovery",
                    workflow_agent_kind=AgentOwner.EXECUTOR,
                    instruction="Finish after a provider failure",

                )
            ],
        ),
    )
    bad = _response("not valid json")
    if network_failure:
        bad = TransportError("unavailable")
    valid = _response(_FINISH)
    failures = [bad, bad, bad] if network_failure else [_response("not valid json") for _ in range(3)]

    with (
        mock.patch("sciagent_runtime.llm.time.sleep"),
        mock.patch(
            "sciagent_runtime.llm.send_request",
            side_effect=[*failures, valid],
        ),
    ):
        run = engine.run_until_stable("run_recovery_chain")

    workflow_task = run.workflow.tasks[0]
    assert workflow_task.status == TaskStatus.COMPLETED
    assert [item.status.value for item in workflow_task.attempts] == (
        ["failed", "completed"] if network_failure else ["completed"]
    )
    assert [item.attempt_number for item in port.requests] == (
        [1, 2] if network_failure else [1]
    )
    assert run.llm_calls_used == 4

    records = [
        json.loads(line)
        for line in (tmp_path / "traces" / "llm_traces.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["action_valid"] for record in records] == (
        [False, True] if network_failure else [False, False, False, True]
    )
    assert sum(len(record["attempts"]) for record in records) == 4
    assert records[-1]["parsed_action"]["tool"] == "finish"
    if not network_failure:
        assert len({record["session_id"] for record in records}) == 1
        assert "not valid JSON" in records[-1]["request_text"]


@pytest.fixture
def recovery(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    monkeypatch.setattr("sciagent_runtime.llm.time.sleep", lambda _: None)
    client = OpenAICompatibleClient(
        model="test-model", api_base="https://example.invalid/v1",
        api_key_env="TEST_LLM_KEY", trace_dir=tmp_path / "traces", trace_level="full",
    )
    definition = AgentDefinition(
        name="recovery", owner=AgentOwner.EXECUTOR,
        system_prompt="Use the provided tools only.",
        tools=(WriteValueTool(), FinishTool()), llm_client=client,
        context_builder=_context,
        permission_policy=AllowListPermissionPolicy({"write_value", "finish"}),
        completion_check=_AcceptFinish(),
    )
    request = AgentRequest(work_request_id="work_test", run_id='run_r', task_id='task_r', attempt_number=1, agent=AgentOwner.EXECUTOR, instruction='q', budget=TaskBudget(max_llm_calls=10, timeout_seconds=60), permissions=AgentPermissions(execute_commands=True))
    return definition, request, InMemorySessionStore()


def _response(content):
    try:
        action = json.loads(content)
    except json.JSONDecodeError:
        name, arguments = "write_value", content
    else:
        name, arguments = action["tool"], json.dumps(action.get("arguments", {}))
    return _FakeResponse({
        "choices": [{"finish_reason": "stop", "message": {
            "content": None, "reasoning_content": "PRIVATE_REASONING",
            "tool_calls": [{"id": f"call_{uuid4().hex}", "type": "function",
                            "function": {"name": name, "arguments": arguments}}],
        }}],
        "usage": {"completion_tokens": 20},
    })


_FINISH = '{"tool":"finish","arguments":{"report":"Done"}}'
_WRITE = '{"tool":"write_value","arguments":{"key":"kept","value":7}}'


@pytest.mark.parametrize("bad", [
    "   ", "PRIVATE_BROKEN_JSON",
    _WRITE + "\n<DSML>PRIVATE_SECOND_ACTION</DSML>",
    _WRITE + "\nLet me continue.", _WRITE + "\n" + _FINISH,
])
def test_bad_json_feedback_preserves_state_and_never_executes_prefix(recovery, bad):
    definition, request, store = recovery
    with mock.patch("sciagent_runtime.llm.send_request", side_effect=[
        _response(bad), _response(_WRITE), _response(_FINISH),
    ]):
        result = AgentLoop(store=store).run(definition, request, session_id="session_r")

    state = store.load("session_r")
    assert result.status == ModuleStatus.COMPLETED
    assert result.llm_calls == state.llm_calls_used == 3
    assert state.step == 2
    assert state.memory["kept"] == 7
    assert [e.tool for e in state.events if e.type == "action"] == ["write_value", "finish"]
    assert state.last_observation.ok
    assert all(len(turn.tool_results) == len(turn.tool_calls) for turn in state.tool_turns)
    assert not json.loads(state.tool_turns[0].tool_results[state.tool_turns[0].tool_calls[0].id])["ok"]
    records = [json.loads(line) for line in
               (definition.llm_client.trace_dir / "llm_traces.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert records[0]["raw_tool_calls"][0]["function"]["arguments"] == bad
    assert records[0]["action_valid"] is False
    assert records[0]["finish_reason"] == "stop"
    assert records[0]["usage"] == {"completion_tokens": 20}
    assert "runtime_feedback" not in records[1]["included_sections"]
    assert "Return native tool calls with JSON objects of arguments" in records[1]["request_text"]
    assert "No tool was executed" in records[1]["request_text"]
    assert "runtime_feedback" not in records[2]["included_sections"]


@pytest.mark.parametrize("call_budget", [2, 3])
def test_transport_then_bad_json_counts_all_attempts_without_exceeding_budget(
    recovery, call_budget,
):
    definition, request, store = recovery
    request = request.model_copy(update={"budget": TaskBudget(
        max_llm_calls=call_budget, timeout_seconds=60,
    )})
    with mock.patch("sciagent_runtime.llm.send_request", side_effect=[
        TransportError("transient"), _response("bad JSON"), _response(_FINISH),
    ]) as provider:
        result = AgentLoop(store=store).run(definition, request, session_id="session_r")
    assert result.llm_calls == provider.call_count == call_budget
    assert store.load("session_r").llm_calls_used == call_budget
    if call_budget == 2:
        assert result.error.code == ErrorCode.BUDGET_EXHAUSTED
    else:
        assert result.status == ModuleStatus.COMPLETED
    records = [json.loads(line) for line in
               (definition.llm_client.trace_dir / "llm_traces.jsonl").read_text().splitlines()]
    assert len(records[0]["attempts"]) == 2
    assert records[0]["attempts"][0]["raw_response_text"] is None
    assert records[0]["attempts"][1]["raw_tool_calls"][0]["function"]["arguments"] == "bad JSON"
    assert sum(r["retry_number"] + 1 for r in records) == result.llm_calls


def test_json_and_schema_errors_share_feedback_and_failure_limit(recovery):
    definition, request, store = recovery
    bad_schema = '{"tool":"finish","report":"Done"}'
    with mock.patch("sciagent_runtime.llm.send_request", side_effect=[
        _response("bad JSON"), _response(bad_schema), _response("bad JSON"),
        _response(bad_schema), _response("bad JSON"), _response(_FINISH),
    ]) as provider:
        result = AgentLoop(store=store).run(definition, request, session_id="session_r")
    assert result.error.code == ErrorCode.TOOL_FAILED
    assert result.error.retryable is False
    assert result.llm_calls == provider.call_count == 5
    records = [json.loads(line) for line in
               (definition.llm_client.trace_dir / "llm_traces.jsonl").read_text().splitlines()]
    assert sum("model" in r for r in records) == 5
    assert sum(r["action_valid"] for r in records) == 2


def test_json_correction_still_respects_wall_clock(recovery):
    definition, request, store = recovery
    clock = [0.0]
    def respond(*args, **kwargs):
        clock[0] = 61.0
        return _response("bad JSON")
    with mock.patch("sciagent_runtime.llm.send_request", side_effect=respond) as provider:
        result = AgentLoop(store=store, clock=lambda: clock[0]).run(
            definition, request, session_id="session_r",
        )
    assert result.error.code == ErrorCode.TIMEOUT
    assert result.llm_calls == provider.call_count == 1
    assert store.load("session_r").step == 0


def test_json_feedback_is_provider_neutral(recovery):
    from dataclasses import replace

    class OtherClient:
        def __init__(self):
            self.contexts = []

        tool_session_key = "other-native/v1"

        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            from sciagent_runtime.tool_calling import parse_tool_turn
            self.contexts.append(context)
            response = _response("bad JSON" if len(self.contexts) == 1 else _FINISH)
            return parse_tool_turn(response._payload["choices"][0]["message"], "stop")

    definition, request, store = recovery
    client = OtherClient()
    result = AgentLoop(store=store).run(
        replace(definition, llm_client=client), request, session_id="session_r",
    )
    assert result.status == ModuleStatus.COMPLETED
    assert result.llm_calls == 2
    assert "runtime_feedback" not in client.contexts[1].included_sections
    assert "not valid JSON" in json.dumps(store.load("session_r").tool_turns[0].tool_results)


def test_correction_keeps_previously_completed_work(recovery):
    definition, request, store = recovery
    with mock.patch("sciagent_runtime.llm.send_request", side_effect=[
        _response(_WRITE), _response("bad JSON"), _response(_FINISH),
    ]):
        result = AgentLoop(store=store).run(definition, request, session_id="session_r")
    state = store.load("session_r")
    assert result.status == ModuleStatus.COMPLETED
    assert result.llm_calls == state.llm_calls_used == 3
    assert state.memory["kept"] == 7
    assert [e.tool for e in state.events if e.type == "action"] == ["write_value", "finish"]


def test_trace_storage_failure_does_not_interrupt_schema_recovery(recovery, caplog):
    definition, request, store = recovery
    trace_dir = definition.llm_client.trace_dir
    trace_dir.mkdir()
    (trace_dir / "llm_traces.jsonl").mkdir()
    bad_schema = '{"tool":"finish","report":"Done"}'
    with mock.patch("sciagent_runtime.llm.send_request", side_effect=[
        _response(bad_schema), _response(_FINISH),
    ]) as provider:
        result = AgentLoop(store=store).run(definition, request, session_id="session_r")

    state = store.load("session_r")
    assert result.status == ModuleStatus.COMPLETED
    assert result.llm_calls == state.llm_calls_used == provider.call_count == 2
    # The schema-validation trace also fails; feedback still reaches the next request.
    request_body = json.loads(provider.call_args_list[1].args[0].content)
    assert "Field required" in json.dumps(request_body["messages"])
    assert "runtime_feedback" not in state.model_dump()
    assert state.last_observation.ok
    assert caplog.text.count("Could not write optional LLM trace") == 2
    assert "PRIVATE_REASONING" not in caplog.text
