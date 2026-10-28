from sciagent_runtime.tool_calling import contract_fingerprint, native_tool_schemas
import json
from datetime import UTC, datetime

import pytest

from e2e.native_fixtures import tool_turns
from sciagent_contracts import (
    AgentOwner,
    AgentPermissions,
    AgentRequest,
    ModuleStatus,
    SessionStatus,
    TaskBudget,
)
from sciagent_runtime import (
    AgentAction,
    AgentDefinition,
    AgentLoop,
    AllowListPermissionPolicy,
    AskUserTool,
    CompletionDecision,
    ContextSection,
    FinishTool,
    InMemorySessionStore,
    JsonSessionStore,
    ScriptedLLMClient,
    AgentState,
)


def test_json_session_round_trip_records_current_schema(tmp_path):
    from sciagent_contracts import SCHEMA_VERSION

    now = datetime.now(UTC)
    state = AgentState(
        session_id="session_schema", agent_name="scientific", owner=AgentOwner.SCIENTIFIC,
        run_id="run_schema", created_at=now, updated_at=now,
    )
    store = JsonSessionStore(tmp_path)
    store.save(state)
    value = json.loads((tmp_path / "session_schema.json").read_text(encoding="utf-8"))
    assert value["schema_version"] == SCHEMA_VERSION == "32.0"
    assert store.load(state.session_id) == state


@pytest.mark.parametrize("version", [None, "24.0", "25.0", "26.0", "27.0", "28.0", "29.0", "30.0"])
def test_json_session_rejects_old_or_missing_schema_without_changing_file(tmp_path, version):
    now = datetime.now(UTC)
    state = AgentState(
        session_id="session_schema", agent_name="scientific", owner=AgentOwner.SCIENTIFIC,
        run_id="run_schema", created_at=now, updated_at=now,
    )
    value = state.model_dump(mode="json")
    if version is None:
        value.pop("schema_version")
    else:
        value["schema_version"] = version
    path = tmp_path / "session_schema.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    before = path.read_bytes()
    store = JsonSessionStore(tmp_path)
    with pytest.raises(ValueError, match="Session schema_version must be 32.0"):
        store.load(state.session_id)
    assert path.read_bytes() == before


def _context(request, state, max_context_tokens) -> list[ContextSection]:
    return [
        ContextSection(name="task", content=request.instruction)
    ]


class _AcceptFinish:
    def evaluate(self, state, candidate) -> CompletionDecision:
        return CompletionDecision(
            complete=True,
            report="done",

        )


def _request(*, attempt: int, parent: str | None = None) -> AgentRequest:
    return AgentRequest(work_request_id="work_test", run_id='run_resume', task_id='task_experiment', attempt_number=attempt, agent=AgentOwner.EXECUTOR, instruction='Which dataset?', budget=TaskBudget(max_llm_calls=5, timeout_seconds=60), parent_session_id=parent, permissions=AgentPermissions(execute_commands=True))


@pytest.mark.parametrize("receipt_saved", [True, False])
def test_ask_user_resume_reuses_session_and_resets_budget(receipt_saved) -> None:
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    definition = AgentDefinition(
        name="needs-input",
        owner=AgentOwner.SCIENTIFIC,
        system_prompt="Ask then finish.",
        tools=(AskUserTool(), FinishTool()),
        llm_client=ScriptedLLMClient(
            tool_turns([
                AgentAction(
                    tool="ask_user",
                    arguments={
                        "text": "Which dataset?",
                        "requested_fields": ["dataset"],
                    },
                ),
                AgentAction(tool="finish", arguments={'report': '{"dataset": "demo"}'}),
            ])
        ),
        context_builder=_context,
        permission_policy=AllowListPermissionPolicy({"ask_user", "finish"}),
        completion_check=_AcceptFinish(),
    )

    first = loop.run(definition, _request(attempt=1), session_id="session_child")
    assert first.status == ModuleStatus.NEEDS_USER_INPUT
    assert first.session is not None and first.session.id == "session_child"

    if not receipt_saved:
        # Crash window: the paused control was saved, but not its accepted receipt.
        checkpoint = store.load(first.session.id)
        checkpoint.tool_turns[first.control.source_turn].tool_results.pop(first.control.source_call_id)
        store.save(checkpoint)
    resumed = loop.run(
        definition,
        _request(attempt=1, parent="session_child").model_copy(update={
            "resume_answers": [_answer(first.control.question)], "control_ack": first.control.origin,
            "runtime_notices": [{"event_id": "notice_after_question", "summary": "New runtime fact"}],
        }),
        session_id="session_child",
    )

    assert resumed.status == ModuleStatus.COMPLETED
    assert resumed.session is not None and resumed.session.id == "session_child"
    assert resumed.report == "done"

    state = store.load("session_child")
    assert state.attempt_number == 1
    assert state.step == 2  # cumulative across resumed calls in this Attempt
    receipt = json.loads(state.tool_turns[first.control.source_turn].tool_results[first.control.source_call_id])
    assert receipt["value"]["question_id"] == first.control.question.id
    assert all(len(turn.tool_results) == len(turn.tool_calls) for turn in state.tool_turns)


def test_resume_rejects_another_attempt_without_mutating_checkpoint() -> None:
    store = InMemorySessionStore()
    now = datetime.now(UTC)
    state = AgentState(work_request_id="work_test",
        session_id="session_same_attempt",
        agent_name="finisher",
        owner=AgentOwner.SCIENTIFIC,
        run_id="run_resume",
        task_id="task_experiment",
        attempt_number=1,
        status=SessionStatus.PAUSED,
        created_at=now,
        updated_at=now,
    )
    store.save(state)
    client = ScriptedLLMClient([])
    definition = AgentDefinition(
        name="finisher", owner=AgentOwner.SCIENTIFIC,
        system_prompt="unused", tools=(FinishTool(),), llm_client=client,
        context_builder=_context,
        permission_policy=AllowListPermissionPolicy({"finish"}),
        completion_check=_AcceptFinish(),
    )
    result = AgentLoop(store=store).run(
        definition, _request(attempt=2, parent=state.session_id),
        session_id=state.session_id,
    )
    assert result.status == ModuleStatus.FAILED
    assert result.error.code.value == "contract_error"
    assert store.load(state.session_id) == state
    assert client.contexts == []


def test_resume_unknown_session_fails_cleanly() -> None:
    loop = AgentLoop(store=InMemorySessionStore())
    definition = AgentDefinition(
        name="needs-input",
        owner=AgentOwner.SCIENTIFIC,
        system_prompt="unused",
        tools=(FinishTool(),),
        llm_client=ScriptedLLMClient([]),
        context_builder=_context,
        permission_policy=AllowListPermissionPolicy({"finish"}),
        completion_check=_AcceptFinish(),
    )

    result = loop.run(
        definition,
        _request(attempt=2, parent="session_missing"),
        session_id="session_missing",
    )

    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    assert result.error.code.value == "contract_error"


def test_resume_rejects_non_paused_session() -> None:
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    definition = AgentDefinition(
        name="finisher",
        owner=AgentOwner.SCIENTIFIC,
        system_prompt="finish only",
        tools=(FinishTool(),),
        llm_client=ScriptedLLMClient(tool_turns([AgentAction(tool="finish", arguments={'report': '{}'})])),
        context_builder=_context,
        permission_policy=AllowListPermissionPolicy({"finish"}),
        completion_check=_AcceptFinish(),
    )

    first = loop.run(definition, _request(attempt=1), session_id="session_done")
    assert first.status == ModuleStatus.COMPLETED

    result = loop.run(
        definition,
        _request(attempt=2, parent="session_done"),
        session_id="session_done",
    )

    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    assert result.error.code.value == "contract_error"


def test_resume_recovers_active_session_after_interruption() -> None:
    """ACTIVE is a persisted checkpoint after a process interruption, not a finish."""
    store = InMemorySessionStore()
    now = datetime.now(UTC)
    store.save(
        AgentState(work_request_id="work_test",
            session_id="session_interrupted",
            tool_protocol_key=ScriptedLLMClient.tool_session_key,
            contract_fingerprint=contract_fingerprint("finish only", native_tool_schemas((FinishTool(),))),
            agent_name="finisher",
            owner=AgentOwner.SCIENTIFIC,
            run_id="run_resume",
            task_id="task_experiment",
            attempt_number=1,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    loop = AgentLoop(store=store)
    definition = AgentDefinition(
        name="finisher",
        owner=AgentOwner.SCIENTIFIC,
        system_prompt="finish only",
        tools=(FinishTool(),),
        llm_client=ScriptedLLMClient(
            tool_turns([AgentAction(tool="finish", arguments={'report': '{}'})])
        ),
        context_builder=_context,
        permission_policy=AllowListPermissionPolicy({"finish"}),
        completion_check=_AcceptFinish(),
    )

    result = loop.run(
        definition,
        _request(attempt=1, parent="session_interrupted"),
        session_id="session_interrupted",
    )

    assert result.status == ModuleStatus.COMPLETED
    assert store.load("session_interrupted").status == SessionStatus.COMPLETED


def test_resume_rejects_mismatched_task() -> None:
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    definition = AgentDefinition(
        name="needs-input",
        owner=AgentOwner.SCIENTIFIC,
        system_prompt="ask then finish",
        tools=(AskUserTool(), FinishTool()),
        llm_client=ScriptedLLMClient(
            tool_turns([
                AgentAction(
                    tool="ask_user",
                    arguments={"text": "Which?", "requested_fields": ["x"]},
                ),
                AgentAction(tool="finish", arguments={'report': '{}'}),
            ])
        ),
        context_builder=_context,
        permission_policy=AllowListPermissionPolicy({"ask_user", "finish"}),
        completion_check=_AcceptFinish(),
    )

    first = loop.run(definition, _request(attempt=1), session_id="session_child")
    assert first.status == ModuleStatus.NEEDS_USER_INPUT

    other = AgentRequest(work_request_id="work_test", run_id='run_resume', task_id='task_other', attempt_number=2, agent=AgentOwner.EXECUTOR, instruction='Which dataset?', budget=TaskBudget(max_llm_calls=5, timeout_seconds=60), parent_session_id='session_child', permissions=AgentPermissions(execute_commands=True))
    result = loop.run(definition, other, session_id="session_child")

    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    assert result.error.code.value == "contract_error"


def _resume_fixture(owner):
    client = ScriptedLLMClient([])
    definition = AgentDefinition(
        name=owner.value, owner=owner, system_prompt="unused",
        tools=(FinishTool(),), llm_client=client, context_builder=_context,
        permission_policy=AllowListPermissionPolicy({"finish"}),
        completion_check=_AcceptFinish(),
    )
    task_scope = {"task_id": "task_root", "attempt_number": 1} if owner == AgentOwner.SCIENTIFIC else {
        "task_id": "task_load", "attempt_number": 1,
    }
    request = AgentRequest(work_request_id="work_test",
        run_id="run_load", agent=owner, instruction="Continue the existing task",
        parent_session_id="session_load", permissions=AgentPermissions(),
        budget=TaskBudget(max_llm_calls=5, timeout_seconds=60),
        **task_scope,
    )
    now = datetime.now(UTC)
    state = AgentState(work_request_id="work_test",
        session_id="session_load", agent_name=owner.value, owner=owner,
        run_id=request.run_id, task_id=request.task_id,
        attempt_number=request.attempt_number, status=SessionStatus.PAUSED,
        tool_protocol_key=client.tool_session_key, created_at=now, updated_at=now,
    )
    return definition, request, state


@pytest.mark.parametrize("owner", [AgentOwner.SCIENTIFIC, AgentOwner.EXECUTOR])
@pytest.mark.parametrize("damage", [
    "invalid_json", "invalid_utf8", "old_schema", "missing_schema",
    "future_schema", "invalid_state", "non_object",
])
def test_resume_persisted_load_error_is_structured_without_rewriting_record(tmp_path, owner, damage):
    from sciagent_runtime.store import SessionLoadError

    definition, request, state = _resume_fixture(owner)
    value = state.model_dump(mode="json")
    if damage == "old_schema":
        value["schema_version"] = "23.0"
    elif damage == "future_schema":
        value["schema_version"] = "999.0"
    elif damage == "missing_schema":
        value.pop("schema_version")
    elif damage == "invalid_state":
        value["llm_calls_used"] = "PRIVATE_CORRUPT_VALUE"
    elif damage == "non_object":
        value = []
    content = json.dumps(value).encode("utf-8")
    if damage == "invalid_json":
        content = b'{"PRIVATE_CORRUPT_VALUE":'
    elif damage == "invalid_utf8":
        content = b"PRIVATE_CORRUPT_VALUE\xff"
    store = JsonSessionStore(tmp_path)
    path = tmp_path / "session_load.json"
    path.write_bytes(content)

    with pytest.raises(SessionLoadError):
        store.load(state.session_id)
    result = AgentLoop(store=store).run(definition, request, session_id=state.session_id)

    assert result.status == ModuleStatus.FAILED
    assert result.error.code.value == "contract_error"
    assert result.error.retryable is False
    assert result.report == result.error.message
    assert "cannot resume session" in result.report
    if damage in {"old_schema", "future_schema", "missing_schema", "non_object"}:
        assert "schema_version must be 32.0" in result.report
    assert result.llm_calls == 0
    assert result.session is None
    assert definition.llm_client.contexts == []
    assert "PRIVATE_CORRUPT_VALUE" not in result.model_dump_json()
    assert path.read_bytes() == content
    assert list(tmp_path.iterdir()) == [path]


def test_resume_unreadable_record_is_structured_without_rewriting_record(tmp_path, monkeypatch):
    from pathlib import Path
    from sciagent_runtime.store import SessionLoadError

    definition, request, state = _resume_fixture(AgentOwner.EXECUTOR)
    store = JsonSessionStore(tmp_path)
    store.save(state)
    path = tmp_path / "session_load.json"
    original = path.read_bytes()
    original_read = Path.read_text

    def unreadable(source, *args, **kwargs):
        if source == path:
            raise PermissionError("record is unreadable")
        return original_read(source, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(SessionLoadError):
        store.load(state.session_id)
    result = AgentLoop(store=store).run(definition, request, session_id=state.session_id)

    assert result.status == ModuleStatus.FAILED
    assert result.error.code.value == "contract_error"
    assert result.error.retryable is False
    assert result.llm_calls == 0
    assert definition.llm_client.contexts == []
    assert path.read_bytes() == original


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_resume_does_not_hide_custom_store_programming_errors(error_type):
    definition, request, state = _resume_fixture(AgentOwner.EXECUTOR)

    class BrokenStore(InMemorySessionStore):
        def load(self, session_id):
            raise error_type("custom store bug")

    store = BrokenStore()
    store.save(state)
    with pytest.raises(error_type, match="custom store bug"):
        AgentLoop(store=store).run(definition, request, session_id=state.session_id)
    assert definition.llm_client.contexts == []


def _answer(question):
    from sciagent_contracts import RecordedAnswer
    return RecordedAnswer(question_id=question.id, question_text=question.text, requested_fields=question.requested_fields, options=question.options, values={field:"demo" for field in question.requested_fields}, answered_at=datetime.now(UTC), run_id=question.run_id, session_id=question.session_id, task_id=question.task_id, attempt_number=question.attempt_number)
