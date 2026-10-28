"""A task answer resumes its Attempt; a retry starts a separate answer scope."""
from resagent2_executor import default_execution_registry
from e2e.scheduler_fixtures import executor_bindings

from datetime import UTC, datetime
import json

import pytest

from control_fixtures import question_signal, ScopedScriptedModulePort, ParentQuestionClient
from e2e.native_fixtures import tool_turns
from resagent2_contracts import (
    AgentOwner, AgentResult, ArtifactCandidate, AttemptStatus, ControlSignal,
    ErrorCode, ExecutionLimits, ModuleError, ModuleStatus, QuestionDraft,
    RecordedAnswer, ResearchRequest, RunBudget, RunPermissions, RunStatus, SessionRef,
    SessionStatus, UserAnswer, WorkflowAgentKind,
    WorkflowAgentRegistry, WorkflowProposal, TaskProposal,
)
from resagent2_orchestrator import (
    DeterministicWorkInterpreter, DeterministicWorkflowCompiler, InMemoryRunStore,
    ModuleBinding, ResearchController, ScriptedModulePort, WorkflowScheduler,
)
from resagent2_orchestrator.handoffs import read_json
from resagent2_runtime import InMemorySessionStore, ScriptedLLMClient
from resagent2_scientific import ScientificAgent


NOW = datetime(2026, 10, 9, tzinfo=UTC)


def session(attempt_number, status):
    session_id = f"session_attempt_{attempt_number}"
    return SessionRef(
        id=session_id, module=AgentOwner.EXECUTOR, state_uri=f"memory://{session_id}",
        status=status, created_at=NOW, updated_at=NOW,
    )


def question_result(attempt_number):
    draft = QuestionDraft(text="Which dataset?", requested_fields=["dataset"])
    return AgentResult(
        status=ModuleStatus.NEEDS_USER_INPUT, report=draft.text,
        artifacts=[], control=question_signal(f"session_attempt_{attempt_number}", draft.text, draft.requested_fields),
        session=session(attempt_number, SessionStatus.PAUSED),
    )


@pytest.mark.parametrize("retry_pauses", [False, True])
def test_answer_resumes_original_attempt_and_is_not_inherited_by_retry(tmp_path, retry_pauses):
    retry_result = question_result(2) if retry_pauses else AgentResult(
        status=ModuleStatus.COMPLETED, report="Retry complete",
        session=session(2, SessionStatus.COMPLETED),
    )
    port = ScopedScriptedModulePort([
        question_result(1),
        AgentResult(
            status=ModuleStatus.FAILED, report="Retryable execution failure after the answer",
            error=ModuleError(code=ErrorCode.TOOL_FAILED, message="temporary failure", retryable=True),
            session=session(1, SessionStatus.FAILED),
        ),
        retry_result,
    ])
    scheduler = WorkflowScheduler(
        bindings=executor_bindings({"experiment": ModuleBinding(owner=AgentOwner.EXECUTOR, port=port)}),
        store=InMemoryRunStore(), artifact_root=tmp_path / "artifacts",
    )
    proposal = WorkflowProposal(
        work_request_id="work_1", tasks=[TaskProposal(execution_ref="experiment@1",
            id="task_experiment", work_request_id="work_1",
            workflow_agent_kind="executor", instruction="Run the experiment",
        )],
    )
    scientific = ScientificAgent(ParentQuestionClient(tool_turns([
        {"tool": "request_work", "arguments": {
            "assessment": {"statement": "Need experiment results"},
            "work_request": {'objective': 'Run the experiment', 'tasks': [{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Run the experiment'}]},
        }},
        {"tool": "finish", "arguments": {
            "report": "Reviewed the task outcome", "artifacts": [{'kind': "scientific_opinion", 'path': "opinion.json", 'media_type': "application/json", 'summary': "Scientific conclusion", 'content': {
                    "verdict": "inconclusive", "statement": "No strong scientific claim from this task",
                }}],
        }},
    ])), store=InMemorySessionStore())
    controller = ResearchController(
        interpreter=DeterministicWorkInterpreter(), scientific_port=scientific,
        compiler=DeterministicWorkflowCompiler(proposal, patch=None), scheduler=scheduler,
        registry=default_execution_registry(),
    )
    paused = controller.create_run("run_answer_retry_scope", ResearchRequest(
        goal="Run and review an experiment", budget=RunBudget(max_llm_calls=20, timeout_seconds=60),
        execution_limits=ExecutionLimits(max_tasks=2, max_attempts_per_task=2),
        permissions=RunPermissions(),
    ))
    assert paused.status == RunStatus.PAUSED
    assert len(paused.task("task_experiment").attempts) == 1
    question = paused.pending_question
    child_question, = paused.child_questions.values()
    assert question.task_id == "task_root" and question.id != child_question.id
    assert child_question.attempt_number == 1
    assert child_question.task_id == "task_experiment"
    first = port.requests[0]
    assert first.attempt_number == 1 and first.parent_session_id is None
    assert not any(ref.kind == "answer" for ref in first.input_artifacts)

    run = controller.answer_question(paused.run_id, UserAnswer(
        question_id=question.id, values={"dataset": "demo"}, answered_at=datetime.now(UTC),
    ))
    assert run.status == (RunStatus.PAUSED if retry_pauses else RunStatus.COMPLETED), run.terminal_error
    assert len(port.requests) == 3
    resumed, retried = port.requests[1:]
    assert resumed.attempt_number == 1
    assert resumed.parent_session_id == "session_attempt_1"
    recorded, = resumed.resume_answers
    assert recorded.question_id == child_question.id and recorded.values == {"dataset": "demo"}
    assert recorded.source == "parent"
    assert recorded.task_id == first.task_id and recorded.attempt_number == 1
    assert resumed.user_answers == [recorded]
    assert retried.attempt_number == 2 and retried.task_id == first.task_id
    assert retried.parent_session_id is None and retried.resume_answers == [] and retried.user_answers == []
    assert not any(ref.kind in {"question", "answer"} for ref in run.artifacts.values())
    assert len(run.answers) == 2 and run.answers[-1] == recorded
    assert run.answers[0].source == "user"
    assert child_question.id in run.delivered_answer_ids and question.id in run.delivered_answer_ids
    attempts = run.task("task_experiment").attempts
    assert [attempt.number for attempt in attempts] == [1, 2]
    assert attempts[0].status == AttemptStatus.FAILED and attempts[0].error.retryable
    assert attempts[0].session.id == "session_attempt_1"
    assert attempts[1].session.id == "session_attempt_2"
    if retry_pauses:
        assert run.pending_question.task_id == "task_root"
        assert next(iter(run.child_questions.values())).attempt_number == 2
        assert run.pending_question.id != question.id
