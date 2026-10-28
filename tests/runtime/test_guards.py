from e2e.native_fixtures import tool_turns

from sciagent_contracts import AgentPermissions
import pytest

from sciagent_contracts import (AgentOwner, ErrorCode, ModuleStatus, AgentRequest, TaskBudget)
from tool_fixtures import WriteValueTool

from sciagent_runtime import (
    AgentAction,
    AgentDefinition,
    AgentLoop,
    AllowListPermissionPolicy,
    CompletionDecision,
    ContextComposer,
    ContextBudgetExceeded,
    ContextSection,
    FinishCandidate,
    FinishTool,
    InMemorySessionStore,
    ScriptedLLMClient,
)


class NeverFinish:
    def evaluate(self, state, candidate: FinishCandidate | None) -> CompletionDecision:
        return CompletionDecision(complete=False)


def context_builder(request, state, max_context_tokens) -> list[ContextSection]:
    return [ContextSection(name="instruction", content=request.instruction)]


def request(*, max_llm_calls: int = 2, timeout_seconds: int = 60) -> AgentRequest:
    return AgentRequest(work_request_id="work_test", run_id='run_guard', task_id='task_guard', attempt_number=1, agent=AgentOwner.EXECUTOR, instruction='Can the action run?', budget=TaskBudget(max_llm_calls=max_llm_calls, timeout_seconds=timeout_seconds), permissions=AgentPermissions(execute_commands=True))


def definition(actions, *, allowed_tools: set[str]) -> AgentDefinition:
    return AgentDefinition(
        name="guard-test",
        owner=AgentOwner.EXECUTOR,
        system_prompt="Test runtime guards.",
        tools=(WriteValueTool(), FinishTool()),
        llm_client=ScriptedLLMClient(tool_turns(actions)),
        context_builder=context_builder,
        permission_policy=AllowListPermissionPolicy(allowed_tools),
        completion_check=NeverFinish(),
    )


def test_permission_is_checked_before_tool_execution() -> None:
    store = InMemorySessionStore()
    result = AgentLoop(store=store).run(
        definition(
            [
                AgentAction(
                    tool="write_value",
                    arguments={"key": "forbidden", "value": True},
                )
            ],
            allowed_tools=set(),
        ),
        request(max_llm_calls=1),
        session_id="session_denied",
    )

    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    assert result.error.code == ErrorCode.BUDGET_EXHAUSTED
    assert "not allowed" in store.load("session_denied").last_observation.summary
    assert "forbidden" not in store.load("session_denied").memory


def test_tool_arguments_are_validated_before_execution() -> None:
    store = InMemorySessionStore()
    result = AgentLoop(store=store).run(
        definition(
            [AgentAction(tool="write_value", arguments={"key": "missing-value"})],
            allowed_tools={"write_value"},
        ),
        request(),
        session_id="session_invalid_input",
    )

    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    # Invalid arguments are recoverable within the loop. This scripted client
    # has no corrective action left, so its exhaustion becomes TOOL_FAILED while
    # the original actionable rejection remains persisted for diagnosis.
    assert result.error.code == ErrorCode.TOOL_FAILED
    assert store.load("session_invalid_input").memory == {}
    assert store.load("session_invalid_input").last_observation is not None


def test_llm_action_must_match_the_typed_action_schema() -> None:
    store = InMemorySessionStore()
    result = AgentLoop(store=store).run(
        definition(
            [
                {
                    "tool": "finish",
                    "arguments": {'report': '{}', "undocumented_field": True},
                }
            ],
            allowed_tools={"finish"},
        ),
        request(),
        session_id="session_invalid_action",
    )

    # The malformed action is rejected (never executed) and recorded as durable
    # feedback; with no corrective action left the loop fails on exhaustion.
    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    state = store.load("session_invalid_action")
    assert state.last_observation is not None
    assert "undocumented_field" in state.last_observation.summary


def test_rejected_finish_exhausts_budget_instead_of_completing() -> None:
    result = AgentLoop(store=InMemorySessionStore()).run(
        definition(
            [
                AgentAction(
                    tool="finish",
                    arguments={'report': '{}'},
                )
            ],
            allowed_tools={"finish"},
        ),
        request(max_llm_calls=1),
        session_id="session_rejected_finish",
    )

    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    assert result.error.code == ErrorCode.BUDGET_EXHAUSTED


def test_state_is_saved_incrementally_for_each_step() -> None:
    store = InMemorySessionStore()
    AgentLoop(store=store).run(
        definition(
            [
                AgentAction(
                    tool="write_value",
                    arguments={"key": "saved", "value": True},
                ),
                AgentAction(tool="finish", arguments={'report': '{}'}),
            ],
            allowed_tools={"write_value", "finish"},
        ),
        request(max_llm_calls=2),
        session_id="session_persisted",
    )

    history = store.history("session_persisted")
    assert len(history) >= 5
    assert any(item.memory.get("saved") is True for item in history)
    assert history[-1].status.value == "failed"


def test_timeout_is_returned_as_structured_error() -> None:
    ticks = iter([0.0, 10.0])
    loop = AgentLoop(store=InMemorySessionStore(), clock=lambda: next(ticks))

    result = loop.run(
        definition([], allowed_tools=set()),
        request(timeout_seconds=5),
        session_id="session_timeout",
    )

    assert result.status == ModuleStatus.FAILED
    assert result.error is not None
    assert result.error.code == ErrorCode.TIMEOUT


def test_context_composer_retains_every_section_or_reports_capacity():
    composer = ContextComposer()
    sections = [ContextSection(name="task", content="must keep"),
                ContextSection(name="material_index", content="x" * 100)]
    result = composer.compose("system", sections, max_tokens=1000)
    assert result.omitted_sections == []
    assert result.included_sections == ["system", "task", "material_index"]
    with pytest.raises(ContextBudgetExceeded):
        composer.compose("system", sections, max_tokens=15)




def test_required_context_cannot_silently_overflow_budget() -> None:
    with pytest.raises(ContextBudgetExceeded, match="required"):
        ContextComposer().compose(
            "system prompt that is too long",
            [],
            max_tokens=1,
        )
