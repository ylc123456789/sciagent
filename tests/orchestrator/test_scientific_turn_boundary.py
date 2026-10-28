"""Unified root result reception retains delivered child results and validates exact scope."""
from sciagent_executor import default_execution_registry
from e2e.scheduler_fixtures import executor_bindings

from sciagent_contracts import RunPermissions, ExecutionLimits
from sciagent_orchestrator.models import RunUsage
import json
from datetime import UTC, datetime

import pytest

from sciagent_contracts import (
    AgentOwner,
    AgentResult,
    ArtifactCandidate,
    Attempt,
    ConclusionRequirements,
    ControlSignal,
    ModuleError,
    QuestionDraft,
    ResearchRequest,
    RunBudget,
    SessionRef,
    SessionStatus,
    WorkResult,
    WorkRequest,
    WorkRequestDraft,
    WorkRequestStatus,
    WorkTaskResult,
    PendingQuestion,
    WorkflowAgentRegistry,
    Workflow,
    WorkflowTask,
    task_session_id,
    WorkspaceRecord,
    WorkspaceSpec,
    WorkspaceAccess,
)
from sciagent_orchestrator import ModuleBinding, DeterministicWorkInterpreter, JsonRunStore, ResearchController, ResearchRun, WorkflowScheduler
from sciagent_orchestrator.handoffs import system_artifact


def candidate(kind, data):
    return ArtifactCandidate(kind=kind, path=kind+".json", media_type="application/json", summary=kind,
        content=data.model_dump_json() if hasattr(data,"model_dump_json") else json.dumps(data))


def prepared(tmp_path):
    now = datetime.now(UTC)
    controller = ResearchController(interpreter=DeterministicWorkInterpreter(), scientific_port=None, compiler=None,
        scheduler=WorkflowScheduler(bindings=executor_bindings({}), store=JsonRunStore(tmp_path/"runs"), artifact_root=tmp_path/"artifacts"),
        registry=default_execution_registry())
    session = SessionRef(id=task_session_id("run_boundary", "task_root", 1), module="scientific", status="paused",
        state_uri="session://"+task_session_id("run_boundary", "task_root", 1), created_at=now, updated_at=now)
    run = ResearchRun(run_id='run_boundary', status='running', request=ResearchRequest(goal='Evaluate', budget=RunBudget(max_llm_calls=20, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=4, max_attempts_per_task=2)), scientific_session=session, usage=RunUsage(requests={f'fixture_{i}:0': 'succeeded' for i in range(2)}), work_requests=[WorkRequest(owner_task_id="task_root", owner_attempt_number=1, depth=2, id='work_1', run_id='run_boundary', owner_session_id=session.id, request=WorkRequestDraft(objective='Run', tasks=[{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Run'}]), status='stable', workflow_revision=1, outcome=WorkResult(run_id="run_boundary", session_id=session.id, objective="Run", status="stable",work_request_id="work_1",workflow_revision=1,tasks=[WorkTaskResult(execution_ref="experiment@1", task_id="task_one", workflow_agent_kind="executor",instruction="Run",status="completed")]), created_at=now, updated_at=now)], created_at=now, updated_at=now)
    run.execution_definitions = default_execution_registry().execution_definitions
    run.execution_presets = default_execution_registry().presets
    run.conclusion_requirements_ref = system_artifact(controller.scheduler.artifact_registry, run,
        "conclusion_requirements", ConclusionRequirements())
    run.workflow = Workflow(
        run_id=run.run_id, revision=1, created_from="work_1",
        tasks=[WorkflowTask(execution_ref="experiment@1",
            id="task_one", work_request_id="work_1", workflow_agent_kind="executor",
            instruction="Run", status="completed",
            attempts=[Attempt(
                number=1, status="completed", started_at=now, finished_at=now, report="Done",
            )],
        )],
    )
    root_work = WorkRequest(id="work_root", run_id=run.run_id,
        request=WorkRequestDraft(objective="Evaluate", tasks=[{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key":"root", "instruction":"Evaluate"}]),
        status="executing", workflow_revision=1, created_at=now, updated_at=now)
    run.work_requests.append(root_work)
    run.workflow.tasks.insert(0, WorkflowTask(execution_ref=None, id="task_root", work_request_id="work_root",
        workflow_agent_kind="scientific", instruction="Evaluate", status="waiting_for_work",
        attempts=[Attempt(number=1, status="waiting_for_work", started_at=now, session=session)]))
    controller._deliver_child_results(run)
    controller.scheduler.store.save(run)
    request = root_request(controller, run)
    return controller, run, request


def root_request(controller, run):
    return controller.scheduler._module_request(run, run.task("task_root"), 1,
                                                parent_session_id=run.scientific_session.id)


def apply_result(controller, run_id, request, result):
    class Port:
        def invoke(self, actual):
            return result
    controller.scheduler.bindings[request.agent] = ModuleBinding(owner=request.agent, port=Port())
    return controller.scheduler.execute_task(run_id, request.task_id)


def test_scientific_sees_declared_logical_workspaces_without_physical_locations(tmp_path):
    from sciagent_components import request_task_context

    controller, run, _ = prepared(tmp_path)
    for workspace_id, kind in (("ws_source", "local"), ("ws_results", "generated")):
        source = WorkspaceSpec(
            workspace_id=workspace_id, source_kind=kind,
            location=str(tmp_path / "private-source") if kind == "local" else None,
            access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
        )
        run.workspaces[workspace_id] = WorkspaceRecord(
            workspace_id=workspace_id, root=str(tmp_path / workspace_id),
            source=source, managed=kind != "local",
        )
    context = request_task_context(root_request(controller, run))
    assert context["task_context"]["workspaces"] == [
        {"workspace_id": "ws_source", "source_kind": "local"},
        {"workspace_id": "ws_results", "source_kind": "generated"},
    ]
    assert str(tmp_path) not in json.dumps(context)


def reply(run, status):
    session_status = {"request_work":"paused", "needs_user_input":"paused", "completed":"completed", "failed":"failed"}[status]
    fields = dict(status=status, report="Scientific turn report", llm_calls=3,
        session=run.scientific_session.model_copy(update={"status":SessionStatus(session_status)}))
    artifacts = [candidate("observation_trace", {"observed_artifact_ids":[]})]
    if status == "completed": artifacts.append(candidate("scientific_opinion", {"verdict":"inconclusive","statement":"Not enough evidence"}))
    elif status == "failed": fields["error"] = ModuleError(code="tool_failed",message="DISTINCT_ROOT_CAUSE",retryable=False,details={"stderr_tail":"Original diagnostic"})
    elif status == "needs_user_input":
        fields["control"] = ControlSignal(action="ask_user", source_session_id=run.scientific_session.id,
            source_turn=1, source_call_id="ask_metric", question=PendingQuestion(
                id="question_metric", run_id=run.run_id, session_id=run.scientific_session.id, task_id="task_root", attempt_number=1,
                text="Which metric?", requested_fields=["metric"], created_at=datetime.now(UTC)))
    else:
        from sciagent_contracts import ScientificAssessment
        fields["control"] = ControlSignal(action="request_work", source_session_id=run.scientific_session.id,
            source_turn=2, source_call_id="work_repeat", assessment=ScientificAssessment(statement="Need more evidence"),
            work_request=WorkRequest(owner_task_id="task_root", owner_attempt_number=1, depth=2, id="work_repeat", run_id=run.run_id, owner_session_id=run.scientific_session.id,
                request=WorkRequestDraft(objective='Repeat', tasks=[{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Repeat'}]),
                created_at=datetime.now(UTC), updated_at=datetime.now(UTC)))
    return AgentResult(artifacts=artifacts, **fields)


def assert_prior_results_retained(run):
    assert run.status == "failed"
    assert run.llm_calls_used == 2
    assert run.work_requests[0].status == "consumed"
    assert run.scientific_session.id == task_session_id(run.run_id, "task_root", 1)
    assert run.pending_question is None
    assert run.delivered_answer_ids == []


@pytest.mark.parametrize("stage", ["handoff", "reception"])
@pytest.mark.parametrize("message", ["", "   ", "Original message"],
                         ids=["empty", "whitespace-only", "with-message"])
def test_boundary_exceptions_persist_terminal_error(tmp_path, monkeypatch, stage, message):
    controller, run, request = prepared(tmp_path)

    def reject(*args, **kwargs):
        raise RuntimeError(message)

    if stage == "handoff":
        monkeypatch.setattr(controller.scheduler, "request_enricher", reject)
        actual = controller.scheduler.execute_task(run.run_id, "task_root")
    elif stage == "reception":
        monkeypatch.setattr("sciagent_orchestrator.scheduler.receive_artifacts", reject)
        actual = apply_result(controller, run.run_id, request, reply(run, "needs_user_input"))
    else:
        class RejectingCompiler:
            compile = staticmethod(reject)

        controller.compiler = RejectingCompiler()
        run.work_requests[0] = run.work_requests[0].model_copy(update={
            "status": WorkRequestStatus.REQUESTED, "workflow_revision": None, "outcome": None,
        })
        controller.scheduler.store.save(run)
        actual = controller.run_until_stable(run.run_id)

    durable = JsonRunStore(controller.scheduler.store.root).load(run.run_id)
    assert actual == durable
    assert durable.status == "failed"
    assert durable.terminal_error.code == "contract_error"
    assert durable.terminal_error.message == (message.strip() or "RuntimeError")
    assert durable.terminal_error.retryable is False
    assert durable.pending_question is None
    assert durable.delivered_answer_ids == []
    if stage == "compiler":
        assert durable.work_requests[0].status == "failed"
        assert durable.work_requests[0].error == durable.terminal_error
    else:
        assert durable.work_requests[0].status == "consumed"


@pytest.mark.parametrize("status", ["request_work","needs_user_input","completed","failed"])
@pytest.mark.parametrize("field,value", [("id","session_other"),("module",AgentOwner.EXECUTOR),("status",SessionStatus.ACTIVE)])
def test_all_statuses_reject_wrong_session_before_acknowledgement(tmp_path,status,field,value):
    controller,run,request = prepared(tmp_path)
    result = reply(run,status)
    result.session = result.session.model_copy(update={field:value})
    actual = apply_result(controller, run.run_id,request,result)
    assert_prior_results_retained(actual)
    assert actual.terminal_error.code == "contract_error"


def test_invalid_envelope_is_revalidated_and_not_consumed(tmp_path):
    controller,run,request = prepared(tmp_path)
    result = reply(run,"completed").model_copy(update={"report":""})
    actual = apply_result(controller, run.run_id,request,result)
    assert_prior_results_retained(actual)


def test_valid_question_acknowledges_prior_work_once(tmp_path):
    controller,run,request = prepared(tmp_path)
    actual = apply_result(controller, run.run_id,request,reply(run,"needs_user_input"))
    assert actual.status == "paused"
    assert actual.work_requests[0].status == "consumed"
    assert actual.llm_calls_used == 2
    assert actual.pending_question.requested_fields == ["metric"]
    assert not any(ref.kind == "question" for ref in actual.artifacts.values())
    assert actual.terminal_error is None


@pytest.mark.parametrize("with_session", [False,True])
def test_failed_turn_keeps_root_cause_on_disk(tmp_path,with_session):
    controller,run,request = prepared(tmp_path)
    result = reply(run,"failed")
    if not with_session:
        result.session=None
        result.artifacts=[]
    apply_result(controller, run.run_id,request,result)
    actual = controller.scheduler.store.load(run.run_id)
    assert actual.status == "failed" and actual.llm_calls_used == 2
    assert actual.terminal_error == result.error


def test_final_report_storage_failure_has_durable_reason(tmp_path,monkeypatch):
    controller,run,request = prepared(tmp_path)
    def fail(*args,**kwargs): raise OSError("REPORT_STORAGE_UNAVAILABLE")
    monkeypatch.setattr(controller.scheduler.artifact_registry,"register_final_report",fail)
    apply_result(controller, run.run_id,request,reply(run,"completed"))
    actual = controller.scheduler.store.load(run.run_id)
    assert actual.status == "failed" and actual.llm_calls_used == 2
    assert "REPORT_STORAGE_UNAVAILABLE" in actual.terminal_error.message
    assert actual.final_opinion is None and actual.final_report_artifact_id is None


def test_unknown_observation_log_does_not_block_completion(tmp_path):
    controller,run,request = prepared(tmp_path)
    result = reply(run,"completed")
    result.artifacts[0].content=json.dumps({"observed_artifact_ids":["artifact_forged"]})
    actual = apply_result(controller, run.run_id,request,result)
    assert actual.status == "completed"
    assert actual.work_requests[0].status == "consumed"
    assert actual.terminal_error is None


@pytest.mark.parametrize("calls", [-1,True,"3"])
def test_invalid_usage_does_not_charge_fabricated_calls(tmp_path,calls):
    controller,run,request = prepared(tmp_path)
    raw=reply(run,"completed").model_dump()
    raw["llm_calls"]=calls
    actual=apply_result(controller, run.run_id,request,raw)
    assert actual.status == "failed"
    assert actual.llm_calls_used == 2


def test_missing_delivery_gate_preserves_registered_artifacts_and_unconsumed_work(tmp_path):
    from sciagent_orchestrator.handoffs import read_json

    controller, run, _ = prepared(tmp_path)
    run.conclusion_requirements_ref = system_artifact(
        controller.scheduler.artifact_registry, run, "conclusion_requirements",
        ConclusionRequirements(required_artifacts=["metrics"]),
        operation_key="test:require_outputs",
    )
    controller.scheduler.store.save(run)
    request = root_request(controller, run)
    before = dict(run.artifacts)
    result = reply(run, "completed")
    result.artifacts.append(ArtifactCandidate(
        kind="module_report", path="partial.json", media_type="application/json",
        summary="Partial work is preserved", content='{"partial": true}', output_name="partial",
    ))

    actual = apply_result(controller, run.run_id, request, result)
    assert_prior_results_retained(actual)
    assert actual.terminal_error.code == "contract_error"
    assert actual.final_opinion is None
    assert actual.final_report_artifact_id is None
    assert [(item.code.value, item.subject) for item in actual.completion_violations] == [
        ("required_artifact_missing", "metrics"),
    ]
    assert all(actual.artifacts[key] == ref for key, ref in before.items())
    new_refs = [ref for key, ref in actual.artifacts.items() if key not in before]
    assert {ref.kind for ref in new_refs} == {"scientific_opinion", "observation_trace", "module_report"}
    partial = next(ref for ref in new_refs if ref.output_name == "partial")
    assert read_json(partial) == {"partial": True}
    durable = controller.scheduler.store.load(run.run_id)
    assert durable.artifacts == actual.artifacts
    assert durable.completion_violations == actual.completion_violations
    assert durable.work_requests[0].status == "consumed"
