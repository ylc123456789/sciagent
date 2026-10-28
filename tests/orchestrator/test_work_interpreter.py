"""Reverse handoff carries actual refs and preserves every terminal attempt."""
from datetime import UTC, datetime
import pytest
from sciagent_contracts import ArtifactRef,Attempt,ModuleError,WorkRequest,WorkRequestDraft,WorkflowTask
from sciagent_orchestrator import DeterministicWorkInterpreter
from sciagent_runtime.budget import execution_budget
NOW=datetime.now(UTC)
ERROR=ModuleError(code="tool_failed",message="command failed",retryable=False)


def work(**kw):
    return WorkRequest(owner_task_id="task_root", owner_attempt_number=1, depth=2, id="work_test",run_id="run_test",owner_session_id="session_test",
        request=WorkRequestDraft(objective='Compare methods', tasks=[{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Compare methods'}]),
        created_at=NOW,updated_at=NOW,**kw)


def artifact(n,attempt=1,kind="module_report"):
    return ArtifactRef(id=f"artifact_{n}",run_id="run_test",task_id="task_measure",attempt_number=attempt,
        producer="orchestrator" if kind=="module_report" else "executor",kind=kind,
        uri=f"file:///frozen/{n}.md",sha256="0"*64,media_type="text/plain",summary="Original report" if kind=="module_report" else "Measured output",
        metadata={"source_type":"task_report"} if kind=="module_report" else {})


def test_handoff_preserves_failed_attempt_and_latest_outputs_without_rewriting_body():
    failed=artifact("failed");report=artifact("report",2);metric=artifact("metric",2,"metrics")
    task=WorkflowTask(execution_ref="experiment@1", id="task_measure",work_request_id="work_test",workflow_agent_kind="executor",
        instruction="Measure calibration",status="completed",attempts=[
            Attempt(number=1,status="failed",started_at=NOW,finished_at=NOW,error=ERROR,
                    agent_started=True,report="  EXACT first report\n",report_artifact_id=failed.id),
            Attempt(number=2,status="completed",started_at=NOW,finished_at=NOW,
                    agent_started=True,report="EXACT final report\n",report_artifact_id=report.id,artifact_ids=[metric.id])])
    current=work().model_copy(update={"status":"stable","workflow_revision":1})
    before=task.model_dump_json();refs={ref.id:ref for ref in [failed,report,metric]}
    with execution_budget(max_llm_calls=1,timeout_seconds=30) as budget:
        result=DeterministicWorkInterpreter().interpret(current,tasks=[task],artifacts=refs)
        assert budget.usage.used==0
    assert task.model_dump_json()==before
    assert result.artifacts==list(refs.values())
    assert [a.report_artifact_id for a in result.tasks[0].attempts]==[failed.id,report.id]
    assert "EXACT" not in result.model_dump_json() and "summary" not in type(result).model_fields


def test_zero_task_compiler_failure_is_truthful_and_has_no_fake_report():
    result=DeterministicWorkInterpreter().interpret(work(status="failed",error=ERROR),tasks=[],artifacts={})
    assert result.status=="failed" and result.error==ERROR and result.tasks==[] and result.artifacts==[]


def test_unexecuted_dependency_task_has_navigation_but_no_agent_report():
    task=WorkflowTask(execution_ref="experiment@1", id="task_measure",work_request_id="work_test",workflow_agent_kind="executor",
        instruction="Measure",status="blocked")
    current=work().model_copy(update={"status":"stable","workflow_revision":1})
    result=DeterministicWorkInterpreter().interpret(current,tasks=[task],artifacts={})
    assert result.tasks[0].status=="blocked" and result.tasks[0].attempts==[] and result.artifacts==[]
    assert result.tasks[0].error is not None


def test_running_attempt_cannot_be_delivered_as_terminal():
    task=WorkflowTask(execution_ref="experiment@1", id="task_measure",work_request_id="work_test",workflow_agent_kind="executor",
        instruction="Measure",status="running",attempts=[Attempt(number=1,status="running",started_at=NOW)])
    current=work().model_copy(update={"status":"stable","workflow_revision":1})
    with pytest.raises(ValueError,match="nonterminal"):
        DeterministicWorkInterpreter().interpret(current,tasks=[task],artifacts={})


def test_wrong_producer_scope_cannot_be_presented_under_an_attempt():
    report=artifact("report").model_copy(update={"task_id":"task_other"})
    task=WorkflowTask(execution_ref="experiment@1", id="task_measure",work_request_id="work_test",workflow_agent_kind="executor",
        instruction="Measure",status="completed",attempts=[Attempt(number=1,status="completed",started_at=NOW,
            finished_at=NOW,agent_started=True,report="raw",report_artifact_id=report.id)])
    current=work().model_copy(update={"status":"stable","workflow_revision":1})
    with pytest.raises(ValueError,match="producing Attempt"):
        DeterministicWorkInterpreter().interpret(current,tasks=[task],artifacts={report.id:report})
