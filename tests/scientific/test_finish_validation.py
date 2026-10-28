"""Scientific output mistakes are corrected inside the existing Agent loop."""

from datetime import UTC, datetime
import json

import pytest

from resagent2_components import RegisteredArtifactReader
from resagent2_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ArtifactCandidate, ErrorCode,
    SessionStatus, TaskBudget, WorkResult, task_session_id,
)
from resagent2_orchestrator.artifacts import ArtifactRegistry
from resagent2_runtime import (
    AgentState, FinishCandidate, InMemorySessionStore, NativeToolCall, ToolCallTurn,
)
from resagent2_runtime.budget import execution_budget
from resagent2_scientific import ScientificAgent
from resagent2_scientific.completion import ScientificCompletionCheck


RUN_ID = "run_finish_validation"
SESSION_ID = task_session_id(RUN_ID, "task_root", 1)


def opinion():
    return ArtifactCandidate(
        kind="scientific_opinion", path="opinion.json", media_type="application/json",
        summary="Scientific judgment", content=json.dumps({
            "verdict": "inconclusive", "statement": "No measurements supplied",
        }),
    )


def finish(*extras):
    return FinishCandidate(report="Scientific conclusion", artifacts=[opinion(), *extras])


def state():
    now = datetime.now(UTC)
    return AgentState(work_request_id="work_root", task_id="task_root", attempt_number=1,
        session_id=SESSION_ID, agent_name="scientific", owner=AgentOwner.SCIENTIFIC,
        run_id=RUN_ID, created_at=now, updated_at=now,
    )


class NativeClient:
    tool_session_key = "scientific-finish-test"

    def __init__(self, finishes):
        self.finishes = finishes
        self.contexts = []
        self.histories = []

    def next_tool_call(self, context, schemas, turns, **kwargs):
        self.contexts.append(context)
        self.histories.append([turn.model_dump(mode="json") for turn in turns])
        index = len(self.contexts) - 1
        payload = self.finishes[index].model_dump(mode="json")
        for item in payload["artifacts"]:
            if item["kind"] == "scientific_opinion" and "content" in item:
                item["content"] = json.loads(item["content"])
        return ToolCallTurn(tool_calls=[NativeToolCall(
            id=f"finish_{index}", name="finish",
            arguments=json.dumps(payload),
        )])


def invoke(client, *, budget=5, artifacts=()):
    store = InMemorySessionStore()
    request = AgentRequest(work_request_id="work_root", task_id="task_root", attempt_number=1,
        run_id=RUN_ID, agent="scientific", instruction="Judge the supplied evidence",
        budget=TaskBudget(max_llm_calls=budget, timeout_seconds=30),
        permissions=AgentPermissions(), input_artifacts=list(artifacts),
    )
    agent = ScientificAgent(client, store=store)
    with execution_budget(max_llm_calls=budget, timeout_seconds=30) as scope:
        result = agent.invoke(request)
        assert result.llm_calls == scope.usage.used == len(client.contexts)
    return result, store.load(SESSION_ID)


def unsupported(kind="data"):
    return ArtifactCandidate(
        kind=kind, path="copied_evidence.json", media_type="application/json",
        summary="Copied evidence", content='{"accuracy": 0.52}',
    )


def test_native_finish_feedback_corrects_kind_in_same_session_and_budget():
    client = NativeClient([finish(unsupported()), finish()])
    result, persisted = invoke(client)

    assert result.status == "completed"
    assert result.session.id == persisted.session_id == SESSION_ID
    assert result.llm_calls == persisted.llm_calls_used == 2
    assert "Unsupported Scientific finish artifact kind: data" in str(client.histories[1])
    assert "scientific_opinion.evidence_artifact_ids" in str(client.histories[1])
    assert {item.kind for item in result.artifacts} == {"scientific_opinion", "observation_trace"}
    assert persisted.status == SessionStatus.COMPLETED
    assert len(persisted.tool_turns) == 2
    assert all(len(turn.tool_results) == 1 for turn in persisted.tool_turns)
    assert persisted.last_observation.ok


@pytest.mark.parametrize("budget,expected_code,calls", [
    (2, ErrorCode.BUDGET_EXHAUSTED, 2),
    (10, ErrorCode.TOOL_FAILED, 5),
])
def test_invalid_finishes_stop_under_existing_budget_and_failure_limit(budget, expected_code, calls):
    client = NativeClient([finish(unsupported())] * 6)
    result, persisted = invoke(client, budget=budget)

    assert result.status == "failed"
    assert result.error.code == expected_code
    assert result.llm_calls == persisted.llm_calls_used == calls
    assert persisted.status == SessionStatus.FAILED
    assert "data" in persisted.last_observation.summary
    assert not any(item.kind == "data" for item in result.artifacts)


@pytest.mark.parametrize("kind", ["invented_kind", "observation_trace", "literature_search"])
def test_model_cannot_author_unknown_or_tool_generated_scientific_artifacts(kind):
    decision = ScientificCompletionCheck([]).evaluate(state(), finish(unsupported(kind)))
    assert not decision.complete
    assert kind in decision.report


@pytest.mark.parametrize("ownership", ["input", "foreign_session", "unregistered", "modified_ref"])
def test_existing_refs_cannot_be_relabelled_as_new_outputs(tmp_path, ownership):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    ref = registry.register_scientific(
        ArtifactCandidate(kind="web_page", path="report.md", media_type="text/markdown",
                          summary="Existing report", content="Existing evidence"),
        run_id=RUN_ID, session_id="session_other" if ownership == "foreign_session" else SESSION_ID,
    )
    output = ref.model_copy(update={"summary": "Rewritten metadata"}) if ownership == "modified_ref" else ref
    check = ScientificCompletionCheck(
        [], resolve_artifact=lambda _: None if ownership == "unregistered" else ref,
        input_artifact_ids=[ref.id] if ownership == "input" else [],
    )
    decision = check.evaluate(state(), finish(output))
    assert not decision.complete
    assert "cite existing evidence IDs" in decision.report


def test_live_owned_tool_ref_remains_a_valid_output(tmp_path):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    ref = registry.register_scientific(
        ArtifactCandidate(kind="web_page", path="page.md", media_type="text/markdown",
                          summary="Retrieved page", content="Frozen source text"),
        run_id=RUN_ID, session_id=SESSION_ID,
    )
    reader = RegisteredArtifactReader([], run_id=RUN_ID, resolve=lambda _: ref)
    check = ScientificCompletionCheck([], resolve_artifact=reader.resolve_ref, reader=reader)
    decision = check.evaluate(state(), finish(ref))
    assert decision.complete
    assert decision.artifacts[-1] == ref
    duplicate = check.evaluate(state(), finish(ref, ref))
    assert not duplicate.complete
    assert "only once" in duplicate.report


def test_duplicate_logical_output_name_gets_completion_feedback():
    candidate = finish(ArtifactCandidate(
        kind="web_page", path="report.md", media_type="text/markdown",
        summary="Report", content="Conclusion", output_name="conclusion",
    ))
    candidate.artifacts[0].output_name = "conclusion"
    decision = ScientificCompletionCheck([]).evaluate(state(), candidate)
    assert not decision.complete
    assert "unique output_name" in decision.report


def test_native_finish_rejects_even_same_session_input_ref_then_recovers(tmp_path):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    ref = registry.register_scientific(
        ArtifactCandidate(kind="web_page", path="prior_report.md", media_type="text/markdown",
                          summary="Earlier result", content="Earlier evidence"),
        run_id=RUN_ID, session_id=SESSION_ID,
    )
    client = NativeClient([finish(ref), finish()])
    result, persisted = invoke(client, artifacts=[ref])

    assert result.status == "completed"
    assert result.llm_calls == persisted.llm_calls_used == 2
    assert "Input, foreign or unregistered artifacts" in str(client.histories[1])
    assert all(getattr(item, "id", None) != ref.id for item in result.artifacts)


class NativeActionClient:
    tool_session_key = "required-output-recovery"

    def __init__(self, actions):
        self.actions = actions
        self.contexts = []
        self.histories = []

    def next_tool_call(self, context, schemas, turns, **kwargs):
        index = len(self.contexts)
        self.contexts.append(context)
        self.histories.append([turn.model_dump(mode="json") for turn in turns])
        name, arguments = self.actions[index]
        arguments = json.loads(json.dumps(arguments))
        if name == "finish":
            for item in arguments["artifacts"]:
                if item["kind"] == "scientific_opinion" and "content" in item:
                    item["content"] = json.loads(item["content"])
        return ToolCallTurn(tool_calls=[NativeToolCall(
            id=f"delivery_{index}", name=name, arguments=json.dumps(arguments),
        )])


def required_output_request(tmp_path):
    registry = ArtifactRegistry(tmp_path / "required-artifacts")
    request = AgentRequest(work_request_id="work_root", task_id="task_root", attempt_number=1,
        run_id=RUN_ID, agent="scientific", instruction="Deliver the requested metrics",
        budget=TaskBudget(max_llm_calls=5, timeout_seconds=30),
        permissions=AgentPermissions(request_work=True),
        task_context={"conclusion_requirements": {"required_artifacts": ["metrics"]}},
    )
    return registry, request


def test_native_missing_output_feedback_can_request_work_and_resume_same_session(tmp_path):
    registry, request = required_output_request(tmp_path)
    client = NativeActionClient([
        ("finish", finish().model_dump(mode="json")),
        ("request_work", {
            "assessment": {"statement": "The required metrics have not been produced"},
            "work_request": {'objective': 'Produce metrics with output_name metrics', 'tasks': [{"selection": {"kind": "preset", "preset_ref": "coding@1", "reason": "Suitable configured executor"}, 'key': 'work', 'instruction': 'Produce metrics with output_name metrics'}]},
        }),
        ("finish", finish().model_dump(mode="json")),
    ])
    store = InMemorySessionStore()
    agent = ScientificAgent(client, store=store)
    with execution_budget(max_llm_calls=5, timeout_seconds=30):
        paused = agent.invoke(request)
    persisted = store.load(SESSION_ID)

    assert paused.status == "request_work"
    assert paused.control.action == "request_work"
    assert paused.session.id == persisted.session_id == SESSION_ID
    assert paused.llm_calls == persisted.llm_calls_used == 2
    assert persisted.status == SessionStatus.PAUSED
    assert len(persisted.tool_turns) == 2
    assert "required_artifact_missing" in str(client.histories[1])
    assert "subject=metrics" in str(client.histories[1])

    delivered = registry.register(
        ArtifactCandidate(
            kind="experiment_result", path="measured.json", media_type="application/json",
            summary="Measured result", content='{"accuracy": 0.8}', output_name="metrics",
        ),
        grant=None, producer=AgentOwner.EXECUTOR, run_id=RUN_ID,
        task_id="task_metrics", attempt_number=1, index=1,
    )
    resumed = request.model_copy(update={
        "parent_session_id": SESSION_ID,
        "input_artifacts": [*request.input_artifacts, delivered],
        "control_ack": paused.control.origin,
        "work_results": [WorkResult(
            run_id=RUN_ID, session_id=SESSION_ID, work_request_id=paused.control.work_request.id,
            objective=paused.control.work_request.request.objective, status="stable", artifacts=[delivered],
        )],
    })
    with execution_budget(max_llm_calls=3, timeout_seconds=30):
        completed = agent.invoke(resumed)
    persisted = store.load(SESSION_ID)

    assert completed.status == "completed"
    assert completed.session.id == paused.session.id == persisted.session_id
    assert completed.llm_calls == 1
    assert persisted.llm_calls_used == 3
    assert persisted.status == SessionStatus.COMPLETED
    assert len(persisted.tool_turns) == 3
    assert persisted.last_observation.ok
    trace = next(item for item in completed.artifacts if item.kind == "observation_trace")
    assert json.loads(trace.content)["observed_artifact_ids"] == []
    assert all(getattr(item, "id", None) != delivered.id for item in completed.artifacts)


def test_native_missing_output_feedback_can_ask_user_in_same_session(tmp_path):
    _, request = required_output_request(tmp_path)
    client = NativeActionClient([
        ("finish", finish().model_dump(mode="json")),
        ("ask_user", {
            "assessment": {"statement": "Required metrics are unavailable"},
            "text": "Which source should produce the required metrics?",
            "requested_fields": ["source"],
        }),
    ])
    store = InMemorySessionStore()
    agent = ScientificAgent(client, store=store)
    with execution_budget(max_llm_calls=5, timeout_seconds=30):
        result = agent.invoke(request)
    persisted = store.load(SESSION_ID)

    assert result.status == "needs_user_input"
    assert result.control.action == "ask_user"
    assert result.session.id == persisted.session_id == SESSION_ID
    assert result.llm_calls == persisted.llm_calls_used == 2
    assert persisted.status == SessionStatus.PAUSED
    assert len(persisted.tool_turns) == 2
    assert "required_artifact_missing" in str(client.histories[1])
    assert "subject=metrics" in str(client.histories[1])
    assert result.control.question.text == "Which source should produce the required metrics?"
    assert not result.artifacts
