"""Controller preserves unconfirmed stops instead of turning them into pauses."""
from datetime import UTC, datetime

import pytest

from resagent2_contracts import (
    AgentOwner, AgentResult, ErrorCode, ModuleError, ModuleStatus, RunStatus,
    SessionRef, SessionStatus, WorkRequestStatus, WorkflowAgentKind,
)
from resagent2_orchestrator import ModuleBinding
from resagent2_runtime.cancellation import CancellationToken, OperationCancelled, bind_cancellation

from test_controller import build_controller, request_work_action, research_request


@pytest.mark.parametrize("failure_form", ["agent_result", "component_exception"])
def test_unconfirmed_child_stop_is_terminal_failure_and_never_a_confirmed_pause(
    tmp_path, monkeypatch, failure_form,
):
    monkeypatch.chdir(tmp_path)
    token = CancellationToken()
    controller = build_controller(actions=[request_work_action()])

    class UnstoppedPort:
        requests = []

        def invoke(self, request):
            self.requests.append(request)
            token.cancel("user requested interruption")
            if failure_form == "component_exception":
                raise OperationCancelled("process still running", stopped=False,
                                         details={"operation": "process", "pid": 45678})
            now = datetime.now(UTC)
            return AgentResult(
                status=ModuleStatus.FAILED, report="process still running",
                error=ModuleError(code=ErrorCode.INTERRUPTED, message="process still running",
                                  retryable=False, details={"stop_confirmed": False, "pid": 45678}),
                session=SessionRef(id="session_unstopped", module=AgentOwner.EXECUTOR,
                                   state_uri="session://session_unstopped", status=SessionStatus.FAILED,
                                   created_at=now, updated_at=now),
            )

    port = UnstoppedPort()
    controller.scheduler.bindings["executor"] = ModuleBinding(
        owner=AgentOwner.EXECUTOR, port=port,
    )
    with bind_cancellation(token):
        run = controller.create_run("run_unstopped", research_request())
    assert run.status == RunStatus.FAILED
    assert run.terminal_error.code == ErrorCode.INTERRUPTED
    assert run.terminal_error.details["stop_confirmed"] is False
    assert run.terminal_error.details["pid"] == 45678
    assert run.manually_paused_at is None
    assert all(work.status != WorkRequestStatus.CANCELLED for work in run.work_requests)
    assert not any(notice.get("stop_confirmed") is True for notice in run.runtime_notices)
    assert len(port.requests) == 1
    # A later observation cannot resume or turn this failure into safe pause.
    again = controller.run_until_stable(run.run_id)
    assert again.status == RunStatus.FAILED
    assert again.terminal_error.details["stop_confirmed"] is False
    assert len(port.requests) == 1
