import json
from e2e.native_fixtures import tool_turns

from resagent2_contracts import AgentPermissions
from resagent2_contracts import ArtifactCandidate, QuestionDraft

from resagent2_contracts import (AgentOwner, ErrorCode, ModuleError, ModuleStatus, AgentRequest, TaskBudget)
from tool_fixtures import ReadValueTool, WriteValueTool

from resagent2_runtime import (
    AgentAction,
    AgentDefinition,
    AgentLoop,
    AllowListPermissionPolicy,
    CompletionDecision,
    ContextSection,
    FinishCandidate,
    FinishTool,
    InMemorySessionStore,
    ScriptedLLMClient,
    AskUserTool,
)


class AcceptFinish:
    """Test finalizer that accepts only an explicit finish candidate."""

    def evaluate(self, state, candidate: FinishCandidate | None) -> CompletionDecision:
        if candidate is None:
            return CompletionDecision(complete=False)
        return CompletionDecision(
            complete=True,
            report="Deterministic completion check passed",

        )


class NeverFinish:
    """Test finalizer that rejects every proposed finish."""

    def evaluate(self, state, candidate: FinishCandidate | None) -> CompletionDecision:
        return CompletionDecision(complete=False)


class BudgetedScriptedLLM(ScriptedLLMClient):
    """Scripted client exposing the optional model-aware budget hook."""

    def tool_input_limit(self, component_limit: int) -> int:
        return 1


def build_context(request, state, max_context_tokens) -> list[ContextSection]:
    return [
        ContextSection(
            name="task",
            content=f"Instruction: {request.instruction}",
        ),
        ContextSection(
            name="memory",
            content=str(state.memory),
        ),
    ]


def request(agent: AgentOwner) -> AgentRequest:
    return AgentRequest(work_request_id="work_test", run_id='run_runtime', task_id='task_runtime', attempt_number=1, agent=agent, instruction='Read or store the reference value', budget=TaskBudget(max_llm_calls=5, timeout_seconds=60), permissions=AgentPermissions(execute_commands=True))


def definition(
    *,
    name: str,
    llm: ScriptedLLMClient,
    tools: tuple,
    allowed_tools: set[str],
    completion_check=None,
) -> AgentDefinition:
    return AgentDefinition(
        name=name,
        owner=AgentOwner.EXECUTOR,
        system_prompt="Follow the typed task and use only the provided tools.",
        tools=tools,
        llm_client=llm,
        context_builder=build_context,
        permission_policy=AllowListPermissionPolicy(allowed_tools),
        completion_check=completion_check or AcceptFinish(),
    )


def test_same_loop_respects_different_tool_grants() -> None:
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    read_definition = definition(
        name="reader",
        llm=ScriptedLLMClient(
            tool_turns([
                AgentAction(tool="read_value", arguments={"key": "answer"}),
                AgentAction(
                    tool="finish",
                    arguments={'report': '{"answer": 42}'},
                ),
            ])
        ),
        tools=(ReadValueTool(), FinishTool()),
        allowed_tools={"read_value", "finish"},
    )
    write_definition = definition(
        name="writer",
        llm=ScriptedLLMClient(
            tool_turns([
                AgentAction(
                    tool="write_value",
                    arguments={"key": "verified", "value": True},
                ),
                AgentAction(
                    tool="finish",
                    arguments={'report': '{"written": true}'},
                ),
            ])
        ),
        tools=(WriteValueTool(), FinishTool()),
        allowed_tools={"write_value", "finish"},
    )

    read_result = loop.run(
        read_definition,
        request(AgentOwner.EXECUTOR),
        session_id="session_reader",
        initial_memory={"answer": 42},
    )
    write_result = loop.run(
        write_definition,
        request(AgentOwner.EXECUTOR),
        session_id="session_writer",
    )

    assert read_result.status == ModuleStatus.COMPLETED
    assert read_result.report == "Deterministic completion check passed"
    assert write_result.status == ModuleStatus.COMPLETED
    assert store.load("session_writer").memory["verified"] is True
    assert type(loop) is AgentLoop


def test_loop_uses_native_schema_without_json_contract_section() -> None:
    client = ScriptedLLMClient(
        tool_turns([AgentAction(tool="finish", arguments={'report': '{"answer": 42}'})])
    )
    result = AgentLoop(store=InMemorySessionStore()).run(
        definition(
            name="contracts",
            llm=client,
            tools=(FinishTool(),),
            allowed_tools={"finish"},
        ),
        request(AgentOwner.EXECUTOR),
        session_id="session_contracts",
    )

    assert result.status == ModuleStatus.COMPLETED
    assert client.contexts
    assert "tool_contracts" not in client.contexts[0].included_sections
    assert "finish: report" not in client.contexts[0].text
    assert client.schemas[0][0]["function"]["parameters"]["required"] == ["report"]


def test_agent_loop_rejects_json_only_client_without_model_call() -> None:
    class JsonOnlyClient:
        def next_action(self, *args):
            raise AssertionError("JSON Agent action fallback must not execute")

    result = AgentLoop().run(
        definition(name="json-only", llm=JsonOnlyClient(), tools=(FinishTool(),),
                   allowed_tools={"finish"}),
        request(AgentOwner.EXECUTOR), session_id="session_json_only",
    )
    assert result.error.code == ErrorCode.CONTRACT_ERROR
    assert result.llm_calls == 0


def test_loop_uses_model_aware_context_budget_when_client_provides_it() -> None:
    llm = BudgetedScriptedLLM(tool_turns([AgentAction(tool="finish", arguments={})]))
    result = AgentLoop(store=InMemorySessionStore()).run(
        definition(
            name="budgeted",
            llm=llm,
            tools=(FinishTool(),),
            allowed_tools={"finish"},
        ),
        request(AgentOwner.EXECUTOR),
        session_id="session_budgeted",
    )

    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    assert result.error.code == ErrorCode.BUDGET_EXHAUSTED
    assert llm.contexts == []


def test_ask_user_returns_signal_without_reading_a_terminal() -> None:
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    ask_definition = definition(
        name="needs-input",
        llm=ScriptedLLMClient(
            tool_turns([
                AgentAction(
                    tool="ask_user",
                    arguments={
                        "text": "Which dataset should be used?",
                        "requested_fields": ["dataset"],
                    },
                )
            ])
        ),
        tools=(AskUserTool(),),
        allowed_tools={"ask_user"},
        completion_check=NeverFinish(),
    )

    result = loop.run(
        ask_definition,
        request(AgentOwner.EXECUTOR),
        session_id="session_question",
    )

    assert result.status == ModuleStatus.NEEDS_USER_INPUT
    assert result.control is not None and result.control.action == "ask_user"
    assert result.artifacts == []
    question = result.control.question
    assert question.requested_fields == ["dataset"]
    assert result.session is not None
    assert result.session.status.value == "paused"


class RequireWriteBeforeFinish:
    """Reject finish until memory has a ``ready`` key, with an actionable summary."""

    def evaluate(self, state, candidate: FinishCandidate | None) -> CompletionDecision:
        if candidate is None:
            return CompletionDecision(complete=False)
        if not state.memory.get("ready"):
            return CompletionDecision(
                complete=False,
                report="Run the write_value tool before finishing",
            )
        return CompletionDecision(complete=True, report="done")


def test_completion_rejected_then_corrective_action_then_finish_succeeds() -> None:
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    profile = definition(
        name="recover-finish",
        llm=ScriptedLLMClient(
            tool_turns([
                AgentAction(tool="finish", arguments={'report': '{"ok": true}'}),
                AgentAction(tool="write_value", arguments={"key": "ready", "value": True}),
                AgentAction(tool="finish", arguments={'report': '{"ok": true}'}),
            ])
        ),
        tools=(WriteValueTool(), FinishTool()),
        allowed_tools={"write_value", "finish"},
        completion_check=RequireWriteBeforeFinish(),
    )

    result = loop.run(
        profile,
        request(AgentOwner.EXECUTOR),
        session_id="session_recover_finish",
    )

    assert result.status == ModuleStatus.COMPLETED
    assert store.load("session_recover_finish").memory["ready"] is True


def test_rejection_receipt_survives_an_intervening_observation() -> None:
    """A rejection must remain visible even after an ordinary observation
    (read_value) overwrites last_observation; the durable slot is separate."""
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    llm = ScriptedLLMClient(
        tool_turns([
            AgentAction(tool="finish", arguments={'report': '{"ok": true}'}),
            AgentAction(tool="read_value", arguments={"key": "missing"}),
            AgentAction(tool="write_value", arguments={"key": "ready", "value": True}),
            AgentAction(tool="finish", arguments={'report': '{"ok": true}'}),
        ])
    )
    profile = definition(
        name="feedback-persists",
        llm=llm,
        tools=(ReadValueTool(), WriteValueTool(), FinishTool()),
        allowed_tools={"read_value", "write_value", "finish"},
        completion_check=RequireWriteBeforeFinish(),
    )

    result = loop.run(
        profile,
        request(AgentOwner.EXECUTOR),
        session_id="session_feedback_persists",
    )

    assert result.status == ModuleStatus.COMPLETED
    assert all("runtime_feedback" not in context.included_sections for context in llm.contexts)
    assert "Run the write_value tool before finishing" in json.dumps(llm.histories[2][0].tool_results)
    assert store.load("session_feedback_persists").last_observation.ok


def test_recoverable_tool_error_sets_feedback_and_continues() -> None:
    """A Tool that raises (recoverable) must not kill the session immediately:
    it becomes a historical rejection receipt and the next scripted action still runs."""
    from resagent2_runtime import ToolObservation

    class FailingOnceTool:
        name = "flaky"
        input_model = FinishTool.input_model  # reuse a valid schema

        def __init__(self) -> None:
            self.calls = 0

        def execute(self, state, arguments):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("transient failure")
            return ToolObservation(summary="recovered", value={"ok": True})

    flaky = FailingOnceTool()
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    profile = definition(
        name="recover-tool",
        llm=ScriptedLLMClient(
            tool_turns([
                AgentAction(tool="flaky", arguments={'report': '{"ok": true}'}),
                AgentAction(tool="flaky", arguments={'report': '{"ok": true}'}),
                AgentAction(tool="finish", arguments={'report': '{"ok": true}'}),
            ])
        ),
        tools=(flaky, FinishTool()),
        allowed_tools={"flaky", "finish"},
    )

    result = loop.run(
        profile,
        request(AgentOwner.EXECUTOR),
        session_id="session_recover_tool",
    )

    assert result.status == ModuleStatus.COMPLETED
    assert flaky.calls == 2
    # The first failure was recorded as durable feedback before the recovery.
    assert store.load("session_recover_tool").last_observation.ok


def test_consecutive_failures_stop_before_budget() -> None:
    from resagent2_runtime import ToolObservation

    class AlwaysFailTool:
        name = "always_fail"
        input_model = FinishTool.input_model

        def execute(self, state, arguments):
            return ToolObservation(summary="failed", ok=False, value={})

    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    profile = definition(
        name="always-fail",
        llm=ScriptedLLMClient(
            tool_turns([AgentAction(tool="always_fail", arguments={'report': '{}'})] * 50)
        ),
        tools=(AlwaysFailTool(), FinishTool()),
        allowed_tools={"always_fail", "finish"},
    )
    req = AgentRequest(work_request_id="work_test", run_id='run_x', task_id='task_x', attempt_number=1, agent=AgentOwner.EXECUTOR, instruction='i', budget=TaskBudget(max_llm_calls=50, timeout_seconds=60), permissions=AgentPermissions(execute_commands=True))
    result = loop.run(profile, req, session_id="session_fail")

    assert result.status == ModuleStatus.FAILED
    assert result.error.code == ErrorCode.TOOL_FAILED
    # It stopped at the recoverable-failure limit, not by exhausting 50 calls.
    assert store.load("session_fail").step < 50


def test_native_tool_receipts_are_passed_without_preview_section() -> None:
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    llm = ScriptedLLMClient(
        tool_turns([
            AgentAction(tool="write_value", arguments={"key": "a", "value": 1}),
            AgentAction(tool="finish", arguments={'report': '{"ok": true}'}),
        ])
    )
    profile = definition(
        name="recent",
        llm=llm,
        tools=(WriteValueTool(), FinishTool()),
        allowed_tools={"write_value", "finish"},
    )
    loop.run(profile, request(AgentOwner.EXECUTOR), session_id="session_recent")

    # Receipts reach the next native request through paired history.
    assert "recent_observations" not in llm.contexts[1].included_sections
    previous = llm.histories[1][0]
    assert previous.tool_calls[0].name == "write_value"
    assert json.loads(previous.tool_results[previous.tool_calls[0].id])["value"] == 1


class RejectWithReason:
    """Reject every proposed finish with an actionable summary."""

    def evaluate(self, state, candidate: FinishCandidate | None) -> CompletionDecision:
        return CompletionDecision(
            complete=False, report="Missing required evidence; keep working"
        )


def test_completion_rejection_counts_as_failure() -> None:
    """A model that keeps proposing finish while the completion check keeps
    rejecting must be stopped by the consecutive-failure guard, not the budget."""
    store = InMemorySessionStore()
    loop = AgentLoop(store=store)
    profile = definition(
        name="reject-finish",
        llm=ScriptedLLMClient(
            tool_turns([AgentAction(tool="finish", arguments={'report': '{"ok": true}'})] * 50)
        ),
        tools=(FinishTool(),),
        allowed_tools={"finish"},
        completion_check=RejectWithReason(),
    )
    req = AgentRequest(work_request_id="work_test", run_id='run_x', task_id='task_x', attempt_number=1, agent=AgentOwner.EXECUTOR, instruction='i', budget=TaskBudget(max_llm_calls=50, timeout_seconds=60), permissions=AgentPermissions(execute_commands=True))
    result = loop.run(profile, req, session_id="session_reject_finish")

    assert result.status == ModuleStatus.FAILED
    assert result.error.code == ErrorCode.TOOL_FAILED
    state = store.load("session_reject_finish")
    assert state.step < 50
    # Every rejection was persisted as a failed observation, not ok=True.
    rejections = [
        event.data for event in state.events
        if event.type == "observation" and event.tool == "finish"
    ]
    assert rejections
    assert all(not data.get("ok") for data in rejections if isinstance(data, dict))


def test_native_history_preserves_all_error_bodies() -> None:
    """Full paired history preserves both errors without a second preview path."""
    from resagent2_runtime import ToolObservation

    pad = "x" * 600
    errors = ["AlphaError: alpha broke", "BravoError: bravo broke"]

    class FailingTool:
        name = "flaky"
        input_model = FinishTool.input_model

        def __init__(self) -> None:
            self.calls = 0

        def execute(self, state, arguments):
            err = errors[min(self.calls, len(errors) - 1)]
            self.calls += 1
            return ToolObservation(
                summary="command failed",
                ok=False,
                value={"stdout_tail": pad, "stderr_tail": err},
            )

    llm = ScriptedLLMClient(
        tool_turns([
            AgentAction(tool="flaky", arguments={'report': '{}'}),
            AgentAction(tool="flaky", arguments={'report': '{}'}),
            AgentAction(tool="flaky", arguments={'report': '{}'}),
        ])
    )
    profile = definition(
        name="errors",
        llm=llm,
        tools=(FailingTool(), FinishTool()),
        allowed_tools={"flaky", "finish"},
        completion_check=NeverFinish(),
    )
    loop = AgentLoop(store=InMemorySessionStore())
    loop.run(profile, request(AgentOwner.EXECUTOR), session_id="session_errors")

    assert "recent_observations" not in llm.contexts[2].included_sections
    receipts = [json.loads(turn.tool_results[turn.tool_calls[0].id]) for turn in llm.histories[2]]
    assert [receipt["value"]["stderr_tail"] for receipt in receipts] == errors
    assert all(receipt["value"]["stdout_tail"] == pad for receipt in receipts)


class VerifiedFailure:
    """Finalizer that returns a deterministic verified failure (not completion)."""

    def evaluate(self, state, candidate: FinishCandidate | None) -> CompletionDecision:
        return CompletionDecision(
            complete=False,
            report="Experiment failed; retained the execution log",
            artifacts=[ArtifactCandidate(
                kind="execution_log", path="run.log", media_type="text/plain",
                summary="Recorded command failure", content="NameError: totla",
            )],
            failure=ModuleError(
                code=ErrorCode.TOOL_FAILED,
                message="Experiment command failed with exit code 1",
                retryable=False,
                details={
                    "command": "python train.py",
                    "exit_code": 1,
                    "stderr_tail": "NameError: totla",
                },
            ),
        )


def test_deterministic_failure_exit_returns_failed() -> None:
    store = InMemorySessionStore()
    profile = definition(
        name="verified-failure",
        llm=ScriptedLLMClient(
            tool_turns([
                AgentAction(
                    tool="finish",
                    arguments={'report': '{}'},
                )
            ])
        ),
        tools=(FinishTool(),),
        allowed_tools={"finish"},
        completion_check=VerifiedFailure(),
    )

    result = AgentLoop(store=store).run(
        profile,
        request(AgentOwner.EXECUTOR),
        session_id="session_verified_failure",
    )

    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    assert result.error.code == ErrorCode.TOOL_FAILED
    assert result.error.message == "Experiment command failed with exit code 1"
    assert result.error.details["stderr_tail"] == "NameError: totla"
    assert result.report == "Experiment failed; retained the execution log"
    assert result.artifacts[0].content == "NameError: totla"
    assert store.load("session_verified_failure").status.value == "failed"


def test_failure_details_preserve_latest_observation_and_historical_receipt():
    from resagent2_runtime import AgentState, ToolObservation
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    state = AgentState(session_id="session_failure", agent_name="executor", owner=AgentOwner.EXECUTOR,
                       run_id="run_runtime", created_at=now, updated_at=now,
                       last_observation=ToolObservation(summary="command failed", ok=False, value={"stderr":"failure"}))
    assert AgentLoop._failure_details(state)["last_observation"]["value"] == {"stderr":"failure"}
    assert "runtime_feedback" not in AgentLoop._failure_details(state)




def test_malformed_action_is_recoverable() -> None:
    """An action that fails schema validation is fed back, then corrected."""
    store = InMemorySessionStore()
    profile = definition(
        name="recover-invalid-action",
        llm=ScriptedLLMClient(
            tool_turns([
                {
                    "tool": "finish",
                    "arguments": {'report': '{}', "undocumented_field": True},
                },
                AgentAction(tool="finish", arguments={'report': '{"ok": true}'}),
            ])
        ),
        tools=(FinishTool(),),
        allowed_tools={"finish"},
        completion_check=AcceptFinish(),
    )
    result = AgentLoop(store=store).run(
        profile,
        request(AgentOwner.EXECUTOR),
        session_id="session_recover_invalid_action",
    )
    assert result.status == ModuleStatus.COMPLETED


def test_llm_calls_counts_malformed_actions() -> None:
    """A malformed action still consumed an LLM call, so it must count toward the
    reported total (ADR-0011 §7); the Scientific Agent no longer infers this
    from the step counter, which does not advance on schema errors."""
    store = InMemorySessionStore()
    profile = definition(
        name="count-calls",
        llm=ScriptedLLMClient(
            tool_turns([
                {
                    "tool": "finish",
                    "arguments": {'report': '{}', "undocumented_field": True},
                },
                AgentAction(tool="finish", arguments={'report': '{"ok": true}'}),
            ])
        ),
        tools=(FinishTool(),),
        allowed_tools={"finish"},
    )
    result = AgentLoop(store=store).run(
        profile,
        request(AgentOwner.EXECUTOR),
        session_id="session_count_calls",
    )

    assert result.status == ModuleStatus.COMPLETED
    assert result.llm_calls == 2
    assert store.load("session_count_calls").llm_calls_used == 2


def test_agent_loop_passes_remaining_call_budget_to_client() -> None:
    class BudgetAwareLLM(ScriptedLLMClient):
        def __init__(self) -> None:
            super().__init__(tool_turns([AgentAction(tool="finish", arguments={'report': '{}'})]))
            self.attempt_limits: list[int] = []

        def set_attempt_limit(self, max_attempts: int) -> None:
            self.attempt_limits.append(max_attempts)

    llm = BudgetAwareLLM()
    profile = definition(
        name="bounded-client",
        llm=llm,
        tools=(FinishTool(),),
        allowed_tools={"finish"},
    )
    bounded_request = request(AgentOwner.EXECUTOR).model_copy(
        update={
            "budget": TaskBudget(
                max_llm_calls=1, timeout_seconds=60
            )
        }
    )
    result = AgentLoop(store=InMemorySessionStore()).run(
        profile,
        bounded_request,
        session_id="session_bounded_client",
    )

    assert result.status == ModuleStatus.COMPLETED
    assert llm.attempt_limits == [1]


def test_scripted_agent_client_rejects_json_action_fixture() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ScriptedLLMClient([{"tool": "finish", "arguments": {"report": "done"}}])
