"""Model budget facts use the shared ledger, including retries and compaction."""

from datetime import UTC, datetime
from io import BytesIO
import json

import httpx
import pytest

from e2e.native_fixtures import tool_turns
from resagent2_components import request_task_context
from resagent2_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ResearchRequest, RunBudget, RunPermissions, TaskBudget,
)
from resagent2_orchestrator import JsonRunStore, ResearchRun
from resagent2_orchestrator.usage import RunUsagePort
from resagent2_runtime import (
    AgentDefinition, AgentLoop, AllowListPermissionPolicy, CompletionDecision,
    ComposedContext, ContextComposer, ContextSection, FinishTool, InMemorySessionStore,
    OpenAICompatibleClient, PromptLLMClient, ScriptedLLMClient,
)
from resagent2_runtime.budget import DeadlineExceededError, current_budget_status, execution_budget
from resagent2_runtime.models import AgentAction
from resagent2_runtime.tool_calling import native_tool_schemas
from tool_fixtures import WriteValueTool


def _run_usage(tmp_path):
    store = JsonRunStore(tmp_path / "runs")
    now = datetime.now(UTC)
    run = ResearchRun(
        run_id="run_visible_budget", status="running", created_at=now, updated_at=now,
        request=ResearchRequest(goal="Deliver a report", permissions=RunPermissions(),
                                budget=RunBudget(max_llm_calls=12, timeout_seconds=60)),
    )
    store.save(run)
    usage = RunUsagePort(run, store)
    usage.charge("previous", 0)
    usage.complete("previous", 0, "failed")
    usage.charge("cancelled", 0)  # An unknown charged request is not refunded.
    return store, run, usage


def _facts(status):
    return json.loads(status.splitlines()[1])


def _wire_status(body):
    return next(message["content"].split("\n", 1)[1] for message in body["messages"]
                if message["content"].startswith("Runtime status ("))


class _Completion:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=candidate is not None)


def test_each_agent_turn_refreshes_run_total_without_persisting_budget_history(tmp_path):
    _, _, usage = _run_usage(tmp_path)
    client = ScriptedLLMClient(tool_turns([
        {"tool": "write_value", "arguments": {"key": "x", "value": 1}},
        {"tool": "finish", "arguments": {"report": "Delivered"}},
    ]))
    definition = AgentDefinition(
        name="budget_test", owner=AgentOwner.EXECUTOR, system_prompt="Do the assigned work",
        tools=(WriteValueTool(), FinishTool()), llm_client=client,
        context_builder=lambda req, state, limit: [
            ContextSection(name="task", content=json.dumps(request_task_context(req))),
            ContextSection(name="material_index", content="Complete empty directory"),
        ],
        permission_policy=AllowListPermissionPolicy({"write_value", "finish"}),
        completion_check=_Completion(),
    )
    request = AgentRequest(work_request_id="work_test",
        run_id="run_visible_budget", task_id="task_report", attempt_number=1,
        agent="executor", instruction="Deliver a report", permissions=AgentPermissions(),
        budget=TaskBudget(max_llm_calls=4, timeout_seconds=10),
    )
    sessions = InMemorySessionStore()
    with execution_budget(max_llm_calls=4, timeout_seconds=10, usage=usage):
        result = AgentLoop(store=sessions).run(definition, request, session_id="session_budget")
    assert result.status == "completed"
    assert [_facts(ctx.runtime_status)["llm_calls_used"] for ctx in client.contexts] == [2, 3]
    assert [_facts(ctx.runtime_status)["remaining_llm_calls"] for ctx in client.contexts] == [10, 9]
    assert all(_facts(ctx.runtime_status)["max_llm_calls"] == 12 for ctx in client.contexts)
    assert "not a dedicated allowance" in client.contexts[-1].runtime_status
    assert "budget" not in json.loads(client.contexts[-1].task_context.split("\n", 1)[1])
    state = sessions.load("session_budget")
    assert len(state.tool_turns) == 2
    assert "remaining_llm_calls" not in state.model_dump_json()


def test_native_retry_refreshes_budget_before_each_charge_and_traces_exact_input(tmp_path, monkeypatch):
    _, _, usage = _run_usage(tmp_path)
    monkeypatch.setenv("VISIBLE_BUDGET_KEY", "test-secret")
    monkeypatch.setattr("resagent2_runtime.llm.time.sleep", lambda _: None)
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1", api_key_env="VISIBLE_BUDGET_KEY",
        trace_dir=tmp_path / "trace", trace_level="full",
    )
    bodies = []

    def respond(request, **kwargs):
        body = json.loads(request.content)
        bodies.append(body)
        facts = _facts(_wire_status(body))
        assert facts["llm_calls_used"] == usage.used - 1
        if len(bodies) == 1:
            raise httpx.HTTPStatusError("transient", request=request,
                                        response=httpx.Response(503, content=b"retry"))
        return BytesIO(json.dumps({"choices": [{"finish_reason": "tool_calls", "message": {
            "tool_calls": [{"id": "finish_current", "type": "function", "function": {
                "name": "finish", "arguments": '{"report":"Done"}',
            }}],
        }}]}).encode())

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    authority = '{"current_agent":"scientific","run_execution_permissions":{"execute_commands":true,"approval_mode":"auto_review"}}'
    context = ContextComposer().compose("rules", [
        ContextSection(name="task", content="unchanged business requirements"),
        ContextSection(name="runtime_authority", content=authority),
        ContextSection(name="material_index", content="complete directory"),
    ], max_tokens=4000)
    with execution_budget(max_llm_calls=4, timeout_seconds=10, usage=usage):
        client.next_tool_call(context, native_tool_schemas((FinishTool(),)), [], max_input_tokens=4000)
    snapshots = [_facts(_wire_status(body)) for body in bodies]
    assert [item["llm_calls_used"] for item in snapshots] == [2, 3]
    assert [item["remaining_llm_calls"] for item in snapshots] == [10, 9]
    assert all(item["scope"] == "run" and item["max_llm_calls"] == 12 for item in snapshots)
    assert usage.used == 4
    assert list(usage.run.usage.requests.values()) == ["failed", "unknown", "failed", "succeeded"]
    row = json.loads((client.trace_dir / "llm_traces.jsonl").read_text().splitlines()[0])
    for body, attempt in zip(bodies, row["attempts"]):
        assert attempt["request_text"] == json.dumps(
            {"messages": body["messages"], "tools": body["tools"]}, ensure_ascii=False)
        assert body["messages"][-1]["content"] == "## material_index\ncomplete directory"
        authority_messages = [message["content"] for message in body["messages"]
                              if message["content"].startswith("Runtime facts (")]
        assert authority_messages == ["Runtime facts (not task requirements):\n## runtime_authority\n" + authority]
        assert body["messages"][1]["content"] == "## task\nunchanged business requirements"
    assert row["request_text"] == row["attempts"][-1]["request_text"]


def test_standalone_nested_transport_does_not_label_retry_cap_as_run_budget():
    clock = [10.0]
    with execution_budget(max_llm_calls=20, timeout_seconds=100, clock=lambda: clock[0]) as parent:
        parent.charge("existing", 0)
        with execution_budget(max_llm_calls=2, timeout_seconds=3) as provider:
            clock[0] = 12.0
            snapshot = provider.snapshot()
            assert snapshot.scope == "invocation"
            assert (snapshot.max_llm_calls, snapshot.llm_calls_used,
                    snapshot.remaining_llm_calls) == (20, 1, 19)
            assert snapshot.remaining_timeout_seconds == 98
            assert "No Run-wide budget" in snapshot.render()


def test_prompt_adapter_gives_compiler_the_same_run_snapshot(tmp_path):
    _, _, usage = _run_usage(tmp_path)

    class CompilerFixture:
        def next_action(self, context, output_type):
            assert _facts(context.runtime_status)["scope"] == "run"
            assert _facts(context.runtime_status)["max_llm_calls"] == 12
            assert "Original task remains unchanged" in context.task_context
            return {"tool": "finish"}

    adapter = PromptLLMClient(CompilerFixture(), system_prompt="Route tasks", max_context_tokens=4000)
    with execution_budget(max_llm_calls=3, timeout_seconds=3, usage=usage):
        assert adapter.next_action("Original task remains unchanged", AgentAction) == {"tool": "finish"}


def test_summary_retries_and_next_model_request_share_fresh_run_budget(tmp_path, monkeypatch):
    _, _, usage = _run_usage(tmp_path)
    monkeypatch.setenv("SUMMARY_BUDGET_KEY", "test-secret")
    monkeypatch.setattr("resagent2_runtime.llm.time.sleep", lambda _: None)
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1", api_key_env="SUMMARY_BUDGET_KEY",
        trace_dir=tmp_path / "trace", trace_level="full",
    )
    snapshots = []

    def respond(request, **kwargs):
        body = json.loads(request.content)
        snapshots.append(_facts(_wire_status(body)))
        if len(snapshots) == 1:
            raise httpx.HTTPStatusError("transient", request=request,
                                        response=httpx.Response(503, content=b"retry"))
        message = {"content": "Earlier actions retained"}
        if "tools" in body:
            message = {"tool_calls": [{"id": "finish_after_summary", "type": "function", "function": {
                "name": "finish", "arguments": '{"report":"Done"}',
            }}]}
        return BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": message}]}).encode())

    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    with execution_budget(max_llm_calls=6, timeout_seconds=10, usage=usage):
        assert client.summarize_history("Summarize completed turns", max_input_tokens=4000) == "Earlier actions retained"
        client.next_tool_call(
            ComposedContext(text="task", included_sections=[], estimated_tokens=1),
            native_tool_schemas((FinishTool(),)), [], max_input_tokens=4000,
        )
    assert [item["llm_calls_used"] for item in snapshots] == [2, 3, 4]
    assert [item["remaining_llm_calls"] for item in snapshots] == [10, 9, 8]
    assert all(item["scope"] == "run" and item["max_llm_calls"] == 12 for item in snapshots)
    assert usage.used == 5


def test_retry_input_is_not_traced_as_sent_when_preparation_exhausts_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_BOUNDARY_KEY", "test-secret")
    monkeypatch.setattr("resagent2_runtime.llm.time.sleep", lambda _: None)
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1", api_key_env="TRACE_BOUNDARY_KEY",
        trace_dir=tmp_path / "trace", trace_level="full",
    )
    sent = []
    clock = [0.0]

    def status():
        value = current_budget_status()
        if sent:
            clock[0] = 11.0
            return "UNSENT_SECOND_INPUT " + value
        return value

    def respond(request, **kwargs):
        sent.append(json.loads(request.content))
        raise httpx.HTTPStatusError("transient", request=request,
                                    response=httpx.Response(503, content=b"retry"))

    monkeypatch.setattr("resagent2_runtime.llm.current_budget_status", status)
    monkeypatch.setattr("resagent2_runtime.llm.send_request", respond)
    with execution_budget(max_llm_calls=3, timeout_seconds=10, clock=lambda: clock[0]) as budget:
        with pytest.raises(DeadlineExceededError):
            client.next_tool_call(
                ComposedContext(text="task", included_sections=[], estimated_tokens=1),
                native_tool_schemas((FinishTool(),)), [], max_input_tokens=4000,
            )
        assert budget.usage.used == client.last_attempts == len(sent) == 1
    row = json.loads((client.trace_dir / "llm_traces.jsonl").read_text().splitlines()[0])
    first_wire = json.dumps({"messages": sent[0]["messages"], "tools": sent[0]["tools"]}, ensure_ascii=False)
    assert row["request_text"] == row["attempts"][0]["request_text"] == first_wire
    assert len(row["attempts"]) == 1 and row["retry_number"] == 0
    assert "UNSENT_SECOND_INPUT" not in json.dumps(row)


def test_no_trace_is_created_when_first_prepared_attempt_cannot_be_reserved(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_BOUNDARY_KEY", "test-secret")
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1", api_key_env="TRACE_BOUNDARY_KEY",
        trace_dir=tmp_path / "trace", trace_level="full",
    )
    clock = [0.0]
    snapshots = [0]

    def status():
        snapshots[0] += 1
        value = current_budget_status()
        if snapshots[0] == 2:  # Initial capacity check passed; dispatch preparation expires.
            clock[0] = 11.0
        return value

    monkeypatch.setattr("resagent2_runtime.llm.current_budget_status", status)
    monkeypatch.setattr("resagent2_runtime.llm.send_request", lambda *a, **kw: pytest.fail("must not send"))
    with execution_budget(max_llm_calls=3, timeout_seconds=10, clock=lambda: clock[0]) as budget:
        with pytest.raises(DeadlineExceededError):
            client.next_tool_call(
                ComposedContext(text="task", included_sections=[], estimated_tokens=1),
                native_tool_schemas((FinishTool(),)), [], max_input_tokens=4000,
            )
        assert budget.usage.used == client.last_attempts == 0
    assert not (client.trace_dir / "llm_traces.jsonl").exists()


def test_native_capacity_cannot_silently_omit_required_runtime_authority(monkeypatch):
    from resagent2_runtime import ContextBudgetExceeded

    monkeypatch.setenv("VISIBLE_BUDGET_KEY", "test-secret")
    client = OpenAICompatibleClient(
        model="test", api_base="https://example.invalid/v1", api_key_env="VISIBLE_BUDGET_KEY",
    )
    monkeypatch.setattr("resagent2_runtime.llm.send_request", lambda *a, **kw: pytest.fail("must not send"))
    context = ContextComposer().compose("rules", [
        ContextSection(name="task", content="business requirement"),
        ContextSection(name="runtime_authority", content="complete runtime facts " * 2000),
    ], max_tokens=100000)
    with execution_budget(max_llm_calls=3, timeout_seconds=10) as budget:
        with pytest.raises(ContextBudgetExceeded):
            client.next_tool_call(context, native_tool_schemas((FinishTool(),)), [], max_input_tokens=1000)
        assert budget.usage.used == 0
