"""Plain-prompt callers share runtime budgets and provider accounting."""

import json
from unittest import mock
from httpx import TransportError

import pytest
from pydantic import BaseModel

from resagent2_runtime import (
    AgentAction, ContextBudgetExceeded, ModelProfile, OpenAICompatibleClient,
    PromptLLMClient,
)


class StructuredClient:
    """Stateless structured-output fixture, separate from Agent tool calls."""
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.contexts = []

    def next_action(self, context, output_type):
        self.contexts.append(context)
        return next(self.outputs)


def _adapter(client, *, limit=512):
    return PromptLLMClient(
        client, system_prompt="Return one action.", max_context_tokens=limit,
    )


def test_plain_prompt_uses_required_context_without_agent_loop():
    client = StructuredClient([{"tool": "finish"}])
    adapter = _adapter(client)
    # These hooks are optional for minimal clients.
    adapter.set_trace_context(run_id="run_plain")
    adapter.set_attempt_limit(1)

    assert adapter.next_action("Complete this request", AgentAction) == {"tool": "finish"}
    context = client.contexts[0]
    assert context.included_sections == ["system", "request"]
    assert context.omitted_sections == []
    assert "Complete this request" in context.text
    assert 0 < context.estimated_tokens <= 512
    assert adapter.last_attempts == 1


def test_plain_prompt_accepts_a_schema_without_agent_action_fields():
    class Decision(BaseModel):
        accepted: bool

    response = Decision(accepted=True)
    adapter = _adapter(StructuredClient([response]))
    assert adapter.next_action("Review this proposal", Decision) is response

    instruction = OpenAICompatibleClient._action_instruction(Decision)
    schema = json.loads(instruction.split("\n", 1)[1])
    assert schema["required"] == ["accepted"]
    assert "tool" not in schema["properties"]


def test_plain_prompt_rejects_invalid_component_limit():
    with pytest.raises(ValueError, match="must be positive"):
        _adapter(StructuredClient([]), limit=0)


def test_over_budget_prompt_does_not_call_provider_or_reuse_previous_count():
    client = StructuredClient([{"tool": "finish"}])
    # Leave room for the current invocation status as well as the short task.
    adapter = _adapter(client, limit=256)
    adapter.next_action("Complete", AgentAction)
    assert adapter.last_attempts == 1

    with pytest.raises(ContextBudgetExceeded):
        adapter.next_action("x" * 1000, AgentAction)

    assert len(client.contexts) == 1
    assert adapter.last_attempts == 0


def test_plain_prompt_respects_provider_model_capacity(monkeypatch):
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1", api_key_env="NO_KEY",
        model_profile=ModelProfile(
            context_window=1000, reserved_output_tokens=100, safety_margin_tokens=100,
        ),
    )
    available = client.context_budget(AgentAction, component_limit=4096)
    assert 0 < available < 1000
    monkeypatch.setattr(
        client, "next_action", lambda *_: pytest.fail("must reject before provider"),
    )

    adapter = _adapter(client, limit=4096)
    with pytest.raises(ContextBudgetExceeded):
        adapter.next_action("x" * (4 * (available + 1)), AgentAction)
    assert adapter.last_attempts == 0


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": '{"tool": "finish"}'}}],
        }).encode()


@pytest.mark.parametrize("succeeds", [True, False])
def test_plain_prompt_forwards_retry_limit_trace_and_real_attempt_count(
    monkeypatch, tmp_path, succeeds,
):
    monkeypatch.setenv("PROMPT_TEST_KEY", "dummy-test-key")
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1", api_key_env="PROMPT_TEST_KEY",
        trace_dir=tmp_path / "traces", trace_level="full",
    )
    adapter = _adapter(client)
    adapter.set_trace_context(agent="plain_caller", run_id="run_plain", step="review")
    adapter.set_attempt_limit(2)
    responses = [TransportError("transient"), _Response() if succeeds else TransportError("again")]

    with (
        mock.patch("resagent2_runtime.llm.time.sleep"),
        mock.patch("resagent2_runtime.llm.send_request", side_effect=responses) as request,
    ):
        if succeeds:
            assert adapter.next_action("Complete", AgentAction) == {"tool": "finish"}
        else:
            with pytest.raises(RuntimeError, match="after 2 attempts"):
                adapter.next_action("Complete", AgentAction)
    assert request.call_count == adapter.last_attempts == 2
    records = [
        json.loads(line) for line in (tmp_path / "traces" / "llm_traces.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["run_id"] == "run_plain"
    assert records[0]["agent"] == "plain_caller"
    assert records[0]["step"] == "review"
    assert records[0]["retry_number"] == 1
    assert records[0]["included_sections"] == ["system", "request"]
    assert records[0]["estimated_tokens"] > 0
    assert "Complete" in records[0]["request_text"]

    # The previous two HTTP attempts must not leak into a pre-call rejection.
    with pytest.raises(ContextBudgetExceeded):
        adapter.next_action("x" * 10000, AgentAction)
    assert adapter.last_attempts == 0
    assert client.last_attempts == 2


class _OutputDecision(BaseModel):
    value: int


def _output_response(calls):
    from io import BytesIO

    return BytesIO(json.dumps({"choices": [{"finish_reason": "tool_calls", "message": {
        "tool_calls": [{"id": f"output_{index}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments),
        }} for index, (name, arguments) in enumerate(calls)],
    }}]}).encode())


@pytest.mark.parametrize("stateless_output", [False, True])
def test_native_output_and_agent_wire_have_distinct_protocols_with_shared_retry_budget(monkeypatch, tmp_path, stateless_output):
    from resagent2_runtime import ComposedContext
    from resagent2_runtime.budget import execution_budget
    from resagent2_runtime.tool_calling import NATIVE_TOOL_INSTRUCTION

    monkeypatch.setenv("PROMPT_TEST_KEY", "dummy-test-key")
    monkeypatch.setattr("resagent2_runtime.llm.time.sleep", lambda _: None)
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1", api_key_env="PROMPT_TEST_KEY",
        trace_dir=tmp_path / "traces", trace_level="full",
    )
    caller_system = "Return the requested decision with return_decision."
    adapter = PromptLLMClient(client, system_prompt=caller_system, max_context_tokens=4096,
                              output_tool_name="return_decision")
    bodies = []

    def respond(request, **kwargs):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            raise TransportError("transient")
        return _output_response([("return_decision", {"value": 7})])

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    with execution_budget(max_llm_calls=9, timeout_seconds=60) as budget:
        budget.charge("earlier_request", 0)
        if stateless_output:
            adapter.set_attempt_limit(2)
            assert adapter.next_action("Assess the supplied data", _OutputDecision).value == 7
            assert adapter.last_attempts == 2
        else:
            client.set_attempt_limit(2)
            context = ComposedContext(text="Assess the supplied data", system_prompt=caller_system,
                                      task_context="Assess the supplied data", included_sections=[], estimated_tokens=1)
            schemas = [{"type": "function", "function": {
                "name": "return_decision", "parameters": _OutputDecision.model_json_schema(),
            }}]
            assert len(client.next_tool_call(context, schemas, [], max_input_tokens=4096).tool_calls) == 1
        assert budget.usage.used == 3
    assert client.last_attempts == len(bodies) == 2
    for body in bodies:
        expected_system = caller_system if stateless_output else caller_system + "\n\n" + NATIVE_TOOL_INSTRUCTION
        assert body["messages"][0] == {"role": "system", "content": expected_system}
        assert [tool["function"]["name"] for tool in body["tools"]] == ["return_decision"]
        assert body["tools"][0]["function"]["parameters"] == _OutputDecision.model_json_schema()
        assert "response_format" not in body
    statuses = [next(message["content"].split("\n", 1)[1] for message in body["messages"]
                     if message["content"].startswith("Runtime status (")) for body in bodies]
    facts = [json.loads(status.splitlines()[1]) for status in statuses]
    assert [item["max_llm_calls"] for item in facts] == [9, 9]
    assert [item["llm_calls_used"] for item in facts] == [1, 2]
    row = json.loads((client.trace_dir / "llm_traces.jsonl").read_text())
    for body, attempt in zip(bodies, row["attempts"]):
        assert attempt["request_text"] == json.dumps({"messages": body["messages"], "tools": body["tools"]}, ensure_ascii=False)


@pytest.mark.parametrize("calls,error", [
    ([], "tool"),
    ([("return_decision", {"value": 7}), ("return_decision", {"value": 8})], "tool"),
    ([("other_output", {"value": 7})], "tool"),
    ([("return_decision", {"missing_value": True})], "schema"),
])
def test_native_output_keeps_exactly_one_named_typed_result_validation(monkeypatch, calls, error):
    from pydantic import ValidationError
    from resagent2_runtime.budget import execution_budget
    from resagent2_runtime.tool_calling import NativeToolCallError

    monkeypatch.setenv("PROMPT_TEST_KEY", "dummy-test-key")
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1", api_key_env="PROMPT_TEST_KEY",
    )
    adapter = PromptLLMClient(client, system_prompt="Return one typed result.", max_context_tokens=4096,
                              output_tool_name="return_decision")
    bodies = []

    def respond(request, **kwargs):
        bodies.append(json.loads(request.content))
        return _output_response(calls)

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    expected_error = NativeToolCallError if error == "tool" else ValidationError
    with execution_budget(max_llm_calls=3, timeout_seconds=60) as budget:
        with pytest.raises(expected_error):
            adapter.next_action("Assess the supplied data", _OutputDecision)
        assert budget.usage.used == 1
    assert adapter.last_attempts == len(bodies) == 1
    assert bodies[0]["messages"][0]["content"] == "Return one typed result."


def test_native_output_schema_and_budget_are_counted_before_transport(monkeypatch):
    from pydantic import Field
    from resagent2_runtime.budget import execution_budget

    class LargeOutput(BaseModel):
        value: str = Field(description="Complete output contract. " * 1000)

    monkeypatch.setenv("PROMPT_TEST_KEY", "dummy-test-key")
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1", api_key_env="PROMPT_TEST_KEY",
    )
    adapter = PromptLLMClient(client, system_prompt="Return one typed result.", max_context_tokens=1000,
                              output_tool_name="return_decision")
    monkeypatch.setattr("resagent2_runtime.llm.send_request", lambda *a, **kw: pytest.fail("must not send"))
    with execution_budget(max_llm_calls=3, timeout_seconds=60) as budget:
        with pytest.raises(ContextBudgetExceeded):
            adapter.next_action("Assess the supplied data", LargeOutput)
        assert budget.usage.used == 0
    assert adapter.last_attempts == client.last_attempts == 0
