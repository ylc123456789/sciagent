"""Direct-control fixtures; bind synthetic question scopes at their test Port boundary."""
from datetime import UTC, datetime
from uuid import uuid4
from resagent2_contracts import AgentResult, ControlSignal, PendingQuestion
from resagent2_orchestrator import ScriptedModulePort


def question_signal(session_id, text, requested_fields):
    return ControlSignal(action="ask_user", source_session_id=session_id, source_turn=1,
                         source_call_id="call_" + uuid4().hex,
                         question=PendingQuestion(id="question_" + uuid4().hex, run_id="run_fixture",
                                                  task_id="task_fixture", attempt_number=1,
                                                  session_id=session_id, text=text,
                                                  requested_fields=requested_fields, created_at=datetime.now(UTC)))


class ScopedScriptedModulePort(ScriptedModulePort):
    def invoke(self, request):
        result = super().invoke(request)
        if isinstance(result, AgentResult) and result.control and result.control.question:
            signal = result.control.model_copy(deep=True)
            signal.question = signal.question.model_copy(update={
                "run_id": request.run_id, "task_id": request.task_id,
                "attempt_number": request.attempt_number, "recipient_session_id": request.upstream_session_id, "created_at": __import__("resagent2_orchestrator.scheduler", fromlist=["datetime"]).datetime.now(UTC),
            })
            result = result.model_copy(update={"control": signal})
        return result


from e2e.native_fixtures import ParentQuestionClient
