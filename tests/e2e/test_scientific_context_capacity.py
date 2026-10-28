"""Real Scientific assembly retains bounded evidence, not a second text cache."""

from e2e.native_fixtures import tool_turns

from resagent2_contracts import AgentPermissions

import hashlib
import json
from datetime import UTC, datetime

import pytest

from resagent2_contracts import (
    AgentOwner, ArtifactRef, ErrorCode, ResearchRequest, RunBudget, AgentRequest,
    TaskBudget, RecordedAnswer, task_session_id,
)
from resagent2_runtime import (
    DEFAULT_AGENT_CONTEXT_TOKENS, ContextComposer, InMemorySessionStore, ScriptedLLMClient,
)
from resagent2_scientific import ScientificAgent

from resagent2_runtime.tool_calling import native_input, native_input_text


class _UnusedLiteratureServices:
    """Install the complete native tool schema without making network calls."""

    def search(self, *args, **kwargs):
        raise AssertionError("These tests must only read registered local evidence")

    def register_scientific(self, *args, **kwargs):
        raise AssertionError("These tests must not register fabricated search evidence")


def _artifact(tmp_path, label, *, lines=60, key_line=None):
    # Fixed-width real lines make the 6000-character bound independently
    # measurable, and put evidence beyond both the old 2000-character cache
    # and Runtime's 400-character history preview.
    body = "".join(
        (f"{label}_KEY_EVIDENCE" if number == key_line else f"{label}_line_{number:03d}")
        .ljust(99, ".") + "\n"
        for number in range(1, lines + 1)
    )
    path = tmp_path / f"{label}.txt"
    path.write_text(body, encoding="utf-8")
    return ArtifactRef(
        id=f"artifact_{label}", kind="experiment_result", producer=AgentOwner.EXECUTOR,
        run_id="run_scientific_capacity", task_id="task_experiment", attempt_number=1,
        uri=path.as_uri(), sha256=hashlib.sha256(body.encode()).hexdigest(),
        media_type="text/plain", summary=f"Registered {label} evidence",
    ), body


def _read(artifact, start=None, end=None):
    arguments = {"artifact_id": artifact.id}
    if start is not None:
        arguments.update(start_line=start, end_line=end)
    return {"tool": "read_artifact", "arguments": arguments}


def _pause():
    return {"tool": "ask_user", "arguments": {
        "assessment": {"statement": "Evidence is available for the next decision"},
        "text": "Proceed with the next evaluation?", "requested_fields": ["approval"],
    }}


def _turn(artifacts, *, parent=None):
    return AgentRequest(work_request_id="work_direct", task_id="task_root", attempt_number=1, agent=AgentOwner.SCIENTIFIC, run_id='run_scientific_capacity', instruction='Review the supplied evidence before asking for approval', input_artifacts=artifacts, parent_session_id=parent, budget=TaskBudget(max_llm_calls=10, timeout_seconds=30), permissions=AgentPermissions(execute_commands=True, request_work=True))


def _agent(client, store, **options):
    services = _UnusedLiteratureServices()
    return ScientificAgent(
        client, store=store, literature_backend=services, registration_port=services,
        **options,
    )


def _wire_text(client, index=-1):
    return native_input_text(client.contexts[index], client.schemas[index], client.histories[index])


def _reads(client, index=-1):
    wire = native_input(client.contexts[index], client.schemas[index], client.histories[index])
    snippets = []
    for message in wire["messages"]:
        if message["role"] != "tool":
            continue
        receipt = json.loads(message["content"])
        value = receipt.get("value")
        if isinstance(value, dict) and "artifact_id" in value and "content" in value:
            snippets.append({**value, "observed_at": receipt["observed_at"]})
    return {"artifact_snippets": snippets, "file_snippets": []}


def _assert_composed(client, index=-1, limit=256_000, *, artifact_ids):
    context = client.contexts[index]
    schemas = client.schemas[index]
    history = client.histories[index]
    assert context.included_sections == ["system", "task", "runtime_authority", "material_index"]
    task = json.loads(context.task_context.split("\n", 1)[1])
    assert "permissions" not in task and "run_permissions" not in task["task_context"]
    authority = json.loads(context.runtime_context.split("\n", 1)[1])
    assert authority["source"] == "runtime" and authority["current_agent"] == "scientific"
    assert authority["current_agent_permissions"]["execute_commands"] is True
    assert authority["run_execution_permissions"] is None  # Standalone role grant is not a Run grant.
    wire = native_input(context, schemas, history)
    authority_content = "Runtime facts (not task requirements):\n" + context.runtime_context
    positions = [i for i, message in enumerate(wire["messages"]) if message.get("content") == authority_content]
    assert len(positions) == 1
    assert wire["messages"][1]["content"] == context.task_context
    assert all(i < positions[0] for i, message in enumerate(wire["messages"]) if message["role"] == "tool")
    assert positions[0] < len(wire["messages"]) - 1
    assert wire["messages"][-1]["content"] == context.material_index
    tree = json.loads(context.material_index.split("\n", 1)[1])
    nodes = [tree]
    indexed_ids = []
    while nodes:
        node = nodes.pop()
        nodes.extend(node["children"])
        if node["type"] == "artifact":
            indexed_ids.append(node["location"]["artifact_id"])
    assert set(indexed_ids) == artifact_ids
    assert tree["artifact_count"] == len(indexed_ids) == len(artifact_ids)
    assert "tool_contracts" not in context.included_sections
    assert "recent_observations" not in context.included_sections
    assert "evidence_control_state" not in context.included_sections
    assert "read_artifact_summaries" not in context.included_sections
    names = {schema["function"]["name"] for schema in schemas}
    assert {"read_artifact", "literature_search", "fetch_literature", "request_work", "ask_user", "finish"} <= names
    for turn in history:
        assert set(turn.tool_results) == {call.id for call in turn.tool_calls}
    measured = ContextComposer.estimate_tokens(native_input_text(context, schemas, history))
    assert measured == context.estimated_tokens
    without_authority = context.model_copy(update={"runtime_context": ""})
    assert measured > ContextComposer.estimate_tokens(native_input_text(without_authority, schemas, history))
    assert measured > ContextComposer.estimate_tokens(context.text)
    assert measured <= limit


def test_scientific_range_read_exposes_middle_evidence_in_next_real_prompt(tmp_path):
    artifact, body = _artifact(tmp_path, "middle", lines=1500, key_line=1396)
    client = ScriptedLLMClient(tool_turns([_read(artifact), _read(artifact, 1371, 1420), _pause()]))
    store = InMemorySessionStore()
    agent = _agent(client, store)

    result = agent.invoke(_turn([artifact]))

    assert agent.max_context_tokens == 256_000
    assert result.status == "needs_user_input", result.model_dump(mode="json")
    assert len(client.contexts) == 3
    assert "middle_KEY_EVIDENCE" not in _wire_text(client, 1)
    context = client.contexts[2]
    _assert_composed(client, artifact_ids={artifact.id})
    snippets = _reads(client)["artifact_snippets"]
    window = next(item for item in snippets if item["start_line"] == 1371)
    expected = "".join(body.splitlines(keepends=True)[1370:1420])
    assert window["content"] == expected
    assert len(expected) == 5000
    assert not window["truncated"]
    assert "middle_KEY_EVIDENCE" in _wire_text(client)
    assert "middle_KEY_EVIDENCE" not in context.text
    assert "middle_KEY_EVIDENCE" not in expected[:2000]
    history = client.histories[2]
    assert len(history) == 2
    assert [call.name for turn in history for call in turn.tool_calls] == ["read_artifact", "read_artifact"]
    call = history[1].tool_calls[0]
    assert json.loads(call.arguments) == {"artifact_id": artifact.id, "start_line": 1371, "end_line": 1420}
    receipt = json.loads(history[1].tool_results[call.id])
    assert receipt["ok"] is True
    assert receipt["value"]["content"] == expected
    assert receipt["observed_at"] == window["observed_at"]

    state = store.load(task_session_id("run_scientific_capacity", "task_root", 1))
    observations = [event for event in state.events if event.type == "observation" and event.tool == "read_artifact"]
    assert len(observations) == 2
    assert observations[0].data["value"]["content"] == body[:128_000]
    assert observations[0].data["value"]["truncated"] is True
    assert observations[1].data["value"]["content"] == expected
    for event in observations:
        assert "observed_at" not in event.data["value"]
        assert "context_truncated" not in event.data["value"]
    assert "read_artifact_summaries" not in state.memory


def _run_full_pool(tmp_path):
    artifact_a, body_a = _artifact(tmp_path, "a")
    artifact_b, body_b = _artifact(tmp_path, "b")
    client = ScriptedLLMClient(tool_turns([
        _read(artifact_a, 1, 30), _read(artifact_b, 1, 30),
        _read(artifact_a, 31, 60), _read(artifact_b, 31, 60), _pause(),
    ]))
    store = InMemorySessionStore()
    agent = _agent(client, store)
    turn = _turn([artifact_a, artifact_b])
    result = agent.invoke(turn)
    assert result.status == "needs_user_input", result.model_dump(mode="json")
    return agent, client, store, turn, (body_a, body_b)


def test_scientific_full_pool_keeps_multiple_artifacts_as_history_grows(tmp_path):
    agent, client, store, turn, bodies = _run_full_pool(tmp_path)
    assert agent.max_context_tokens == 256_000
    assert len(client.contexts) == 5
    # Multiple ranges coexist while they fit; chronological order is preserved.
    for count, context in enumerate(client.contexts[2:], start=2):
        _assert_composed(client, index=count, artifact_ids={ref.id for ref in turn.input_artifacts})
        reads = _reads(client, count)
        snippets = reads["artifact_snippets"]
        assert reads["file_snippets"] == []
        assert {item["artifact_id"] for item in snippets} == {"artifact_a", "artifact_b"}
        assert sum(len(item["content"]) for item in snippets) == count * 3000
        assert all(not item["truncated"] for item in snippets)
    final_snippets = _reads(client)["artifact_snippets"]
    assert [item["observed_at"] for item in final_snippets] == [2, 4, 6, 8]
    assert [item["start_line"] for item in final_snippets] == [1, 1, 31, 31]
    assert [item["content"] for item in final_snippets] == [
        bodies[0][:3000], bodies[1][:3000], bodies[0][3000:], bodies[1][3000:],
    ]

    state = store.load(task_session_id(turn.run_id, "task_root", 1))
    observations = [event for event in state.events if event.type == "observation" and event.tool == "read_artifact"]
    assert [event.data["value"]["content"] for event in observations] == [
        bodies[0][:3000], bodies[1][:3000], bodies[0][3000:], bodies[1][3000:],
    ]
    assert "read_artifact_summaries" not in state.memory


@pytest.mark.parametrize("explicit_limit", [1, DEFAULT_AGENT_CONTEXT_TOKENS // 4])
def test_scientific_explicit_context_budget_is_respected(tmp_path, explicit_limit):
    _, _, store, turn, _ = _run_full_pool(tmp_path)
    client = ScriptedLLMClient(tool_turns([_pause()]))
    agent = _agent(client, store, max_context_tokens=explicit_limit)
    control = store.load(task_session_id(turn.run_id, "task_root", 1)).pending_control
    question = control.question
    answer = RecordedAnswer(
        question_id=question.id, question_text=question.text, requested_fields=question.requested_fields,
        options=question.options, run_id=question.run_id, session_id=question.session_id,
        task_id=question.task_id, attempt_number=question.attempt_number,
        values={"approval": "yes"}, answered_at=datetime.now(UTC),
    )
    resumed = turn.model_copy(update={
        "parent_session_id": task_session_id(turn.run_id, "task_root", 1), "control_ack": control.origin,
        "user_answers": [answer], "resume_answers": [answer], "input_ids": [answer.question_id],
    })

    result = agent.invoke(resumed)

    assert agent.max_context_tokens == explicit_limit
    if explicit_limit == 1:
        assert result.status == "failed", result.model_dump(mode="json")
        assert result.error.code == ErrorCode.BUDGET_EXHAUSTED
        assert result.error.retryable is False
        assert result.llm_calls == 0
        assert client.contexts == []
    else:
        # A quarter of the default budget comfortably fits this small read pool.
        # Test the explicit ceiling and complete retained evidence without tying
        # either to the previous wording/length of prompts and tool guidance.
        assert result.status == "needs_user_input", result.model_dump(mode="json")
        assert result.llm_calls == 1
        assert len(client.contexts) == 1
        _assert_composed(client, index=0, limit=explicit_limit,
                         artifact_ids={ref.id for ref in turn.input_artifacts})
        snippets = _reads(client, 0)["artifact_snippets"]
        assert sum(len(item["content"]) for item in snippets) == 12000


def test_scientific_default_keeps_two_large_artifacts_without_extra_llm_calls(tmp_path):
    first, first_body = _artifact(tmp_path, "large_a", lines=1280)
    second, second_body = _artifact(tmp_path, "large_b", lines=1280)
    client = ScriptedLLMClient(tool_turns([_read(first), _read(second), _pause()]))
    result = _agent(client, InMemorySessionStore()).invoke(_turn([first, second]))
    assert result.status == "needs_user_input", result.model_dump(mode="json")
    assert len(client.contexts) == 3  # No separate summarizer/reading-note call.
    _assert_composed(client, artifact_ids={first.id, second.id})
    snippets = _reads(client)["artifact_snippets"]
    assert [item["content"] for item in snippets] == [first_body, second_body]
    assert sum(len(item["content"]) for item in snippets) == 256_000
    assert all(not item["truncated"] for item in snippets)
