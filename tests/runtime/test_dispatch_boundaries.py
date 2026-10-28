from e2e.native_fixtures import tool_turn
"""Dispatch deadlines and unambiguous control signals, without real time or LLMs."""

from resagent2_contracts import AgentPermissions

from itertools import combinations

from pydantic import BaseModel, ValidationError
import pytest

from resagent2_contracts import (AgentOwner, ErrorCode, ModuleStatus, AgentRequest, SessionStatus, TaskBudget)
from resagent2_runtime import (
    AgentDefinition, AgentLoop, CompletionDecision, FinishCandidate,
    InMemorySessionStore, PermissionDecision, ToolObservation,
)


class FakeClock:
    now = 0.0

    def __call__(self):
        return self.now


class EmptyInput(BaseModel):
    pass


class RecordingTool:
    name = "write"
    input_model = EmptyInput

    def __init__(self):
        self.executions = 0

    def execute(self, state, arguments):
        self.executions += 1
        return ToolObservation(
            summary="Executed write", memory_updates={"written": True},
            finish_candidate=FinishCandidate(report="Written"),
        )


class AcceptCompletion:
    def evaluate(self, state, candidate):
        return CompletionDecision(complete=candidate is not None)


@pytest.mark.parametrize("expired_stage", ["llm", "permission"])
@pytest.mark.parametrize("elapsed", [10.0, 10.1])
def test_elapsed_deadline_blocks_dispatch_but_keeps_llm_calls(expired_stage, elapsed):
    _run_dispatch(expired_stage, elapsed, should_execute=False)


def test_action_with_remaining_time_still_executes():
    _run_dispatch("llm", 9.0, should_execute=True)


def _run_dispatch(stage, elapsed, *, should_execute):
    clock = FakeClock()
    tool = RecordingTool()
    store = InMemorySessionStore()

    class Client:
        # Include provider-internal retries in the real-call count; timeout
        # before dispatch must not erase them.
        last_attempts = 2

        tool_session_key = "dispatch-native/v1"

        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            if stage == "llm":
                clock.now = elapsed
            return tool_turn("write")

    class Policy:
        def check(self, action, state, request):
            if stage == "permission":
                clock.now = elapsed
            return PermissionDecision(outcome='allow' if True else 'deny')

    definition = AgentDefinition(
        name="writer", owner=AgentOwner.EXECUTOR, system_prompt="Use write.",
        tools=(tool,), llm_client=Client(), context_builder=lambda request, state, limit: [],
        permission_policy=Policy(), completion_check=AcceptCompletion(),
    )
    request = AgentRequest(work_request_id="work_test", run_id='run_deadline', task_id='task_deadline', attempt_number=1, agent=AgentOwner.EXECUTOR, instruction='Write once', budget=TaskBudget(max_llm_calls=3, timeout_seconds=10), permissions=AgentPermissions(execute_commands=True))

    result = AgentLoop(store=store, clock=clock).run(
        definition, request, session_id="session_deadline"
    )

    persisted = store.load("session_deadline")
    assert result.llm_calls == persisted.llm_calls_used == 1
    assert tool.executions == int(should_execute)
    if should_execute:
        assert result.status == ModuleStatus.COMPLETED
        assert persisted.memory["written"] is True
    else:
        assert result.status == ModuleStatus.FAILED
        assert result.error.code == ErrorCode.TIMEOUT
        assert "before tool dispatch" in result.error.message
        assert persisted.status == SessionStatus.FAILED
        assert "written" not in persisted.memory
        assert persisted.last_observation is None
        assert not any(event.type == "observation" for event in persisted.events)


CONTROL_SIGNALS = {
    "finish_candidate": {'report': '{"done": true}'},
    "question": {"text": "Choose", "requested_fields": ["choice"]},
    # Empty objects are still present control signals, not falsey absence.
    "request_work": {},
}


@pytest.mark.parametrize("names", [(), *[(name,) for name in CONTROL_SIGNALS]])
def test_tool_observation_accepts_zero_or_one_control_signal(names):
    observation = ToolObservation(
        summary="One action", **{name: CONTROL_SIGNALS[name] for name in names}
    )
    assert observation.summary == "One action"


@pytest.mark.parametrize(
    "names", [*combinations(CONTROL_SIGNALS, 2), tuple(CONTROL_SIGNALS)]
)
def test_tool_observation_rejects_multiple_control_signals(names):
    with pytest.raises(ValidationError, match="at most one"):
        ToolObservation(
            summary="Ambiguous action", **{name: CONTROL_SIGNALS[name] for name in names}
        )
