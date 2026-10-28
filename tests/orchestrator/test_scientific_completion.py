"""Final completion consumes registered opinion, observation and requirement snapshots."""
from sciagent_executor import default_execution_registry

from sciagent_contracts import RunPermissions, ExecutionLimits
import json
from datetime import UTC, datetime

import pytest

from sciagent_contracts import (
    AgentOwner,
    AgentResult,
    ArtifactCandidate,
    Attempt,
    ConclusionRequirements,
    ModuleError,
    SYSTEM_ARTIFACT_PROVENANCE,
    ScientificOpinion,
    SessionRef,
    WorkResult,
    WorkRequest,
    WorkRequestDraft,
    WorkTaskResult,
    Workflow,
    WorkflowAgentRegistry,
    WorkflowTask,
)
from sciagent_orchestrator import (
    ArtifactRegistry, ArtifactRegistrationError, CompletionViolationCode, FinalReportRenderer,
    ResearchRun, ScientificCompletionValidator,
)
from sciagent_contracts import (
    ResearchRequest,
    RunBudget,
)
from sciagent_orchestrator.handoffs import system_artifact


def candidate(kind, data):
    return ArtifactCandidate(kind=kind, path=kind+".json", media_type="application/json", summary=kind,
        content=data.model_dump_json() if hasattr(data, "model_dump_json") else json.dumps(data))


@pytest.fixture
def prepared(tmp_path):
    now = datetime.now(UTC)
    registry = ArtifactRegistry(tmp_path / "artifacts")
    session = SessionRef(id="session_scientific", module="scientific", state_uri="memory://scientific",
        status="completed", created_at=now, updated_at=now)
    run = ResearchRun(run_id='run_gate', request=ResearchRequest(goal='Evaluate the method', budget=RunBudget(max_llm_calls=20, timeout_seconds=60), permissions=RunPermissions(execute_commands=True), execution_limits=ExecutionLimits(max_tasks=5, max_attempts_per_task=2)), status='running', scientific_session=session, created_at=now, updated_at=now)
    run.execution_definitions = default_execution_registry().execution_definitions
    run.execution_presets = default_execution_registry().presets
    run.conclusion_requirements_ref = system_artifact(registry, run, "conclusion_requirements", ConclusionRequirements())
    gate = ScientificCompletionValidator(registry)
    return registry, run, gate


def registered_result(prepared, *, verdict="inconclusive", cited=(), observed=(), limitations=()):
    registry, run, gate = prepared
    opinion = ScientificOpinion(verdict=verdict, statement="The evidence supports this judgment",
        evidence_artifact_ids=list(cited), limitations=list(limitations))
    refs = [registry.register_scientific(item, run_id=run.run_id, session_id=run.scientific_session.id)
        for item in [candidate("scientific_opinion", opinion), candidate("observation_trace", {"observed_artifact_ids": list(observed)})]]
    run.artifacts.update({ref.id: ref for ref in refs})
    return AgentResult(status="completed", report="Conclusion", artifacts=refs, session=run.scientific_session), refs


def evidence(prepared, *, run_id=None):
    registry, run, _ = prepared
    ref = registry.register_scientific(candidate("literature_paper", {"papers": ["paper"]}),
        run_id=run_id or run.run_id, session_id=run.scientific_session.id)
    run.artifacts[ref.id] = ref
    return ref


@pytest.mark.parametrize("verdict", ["supports", "refutes", "inconclusive", "not_applicable"])
def test_all_verdicts_pass_with_valid_registered_evidence(prepared, verdict):
    _, run, gate = prepared
    ref = evidence(prepared)
    result, refs = registered_result(prepared, verdict=verdict, cited=[ref.id], observed=[ref.id])
    accepted = gate.validate(run, result, refs)
    assert accepted.ok
    assert accepted.report.evidence == [ref]


@pytest.mark.parametrize("run_observed,trace_observed,cited", [(False, True, True), (True, False, True), (True, True, False)])
def test_required_evidence_depends_on_registration_and_citation_only(prepared, run_observed, trace_observed, cited):
    registry, run, gate = prepared
    ref = evidence(prepared)
    run.conclusion_requirements_ref = system_artifact(registry, run, "conclusion_requirements",
        ConclusionRequirements(required_evidence_kinds=["literature_paper"]),operation_key="test:required_evidence")
    result, refs = registered_result(prepared, cited=[ref.id] if cited else [], observed=[ref.id] if trace_observed else [])
    actual = gate.validate(run, result, refs)
    assert actual.ok is cited
    assert (CompletionViolationCode.MISSING_EVIDENCE_KIND in {v.code for v in actual.violations}) is (not cited)


def test_unknown_observation_log_does_not_reject_completion(prepared):
    _, run, gate = prepared
    result, refs = registered_result(prepared, observed=["artifact_fake"])
    assert gate.validate(run, result, refs).ok


def test_cross_run_evidence_is_rejected(prepared):
    _, run, gate = prepared
    ref = evidence(prepared, run_id="run_other")
    result, refs = registered_result(prepared, verdict="supports", cited=[ref.id], observed=[ref.id])
    assert not gate.validate(run, result, refs).ok


@pytest.mark.parametrize("status", ["pending", "running", "needs_user_input"])
def test_nonterminal_task_prevents_completion(prepared, status):
    _, run, gate = prepared
    run.workflow = Workflow(run_id=run.run_id, revision=1, created_from="work_test", tasks=[WorkflowTask(execution_ref="experiment@1",
        id="task_test", work_request_id="work_test", workflow_agent_kind="executor", instruction="Measure", status=status)])
    result, refs = registered_result(prepared)
    actual = gate.validate(run, result, refs)
    assert CompletionViolationCode.ACTIVE_CONTROL_STATE in {v.code for v in actual.violations}


def test_active_work_request_prevents_completion(prepared):
    _, run, gate = prepared
    run.work_requests = [WorkRequest(owner_task_id="task_root", owner_attempt_number=1, depth=2, id="work_test", run_id=run.run_id, owner_session_id=run.scientific_session.id,
        request=WorkRequestDraft(objective='Measure', tasks=[{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Measure'}]), created_at=run.created_at, updated_at=run.updated_at)]
    result, refs = registered_result(prepared)
    assert not gate.validate(run, result, refs).ok


def test_completed_task_requires_terminal_attempt_history(prepared):
    _, run, gate = prepared
    task = WorkflowTask(execution_ref="experiment@1", id="task_test", work_request_id="work_test", workflow_agent_kind="executor", instruction="Measure", status="completed")
    run.workflow = Workflow(run_id=run.run_id, revision=1, created_from="work_test", tasks=[task])
    result, refs = registered_result(prepared)
    assert not gate.validate(run, result, refs).ok
    run.workflow.tasks[0].attempts = [Attempt(number=1, status="completed", started_at=run.created_at, finished_at=run.updated_at, report="Executed")]
    assert gate.validate(run, result, refs).ok


def test_failed_task_requires_limitations_and_is_reported(prepared):
    _, run, gate = prepared
    issue = WorkTaskResult(execution_ref="experiment@1", task_id="task_failed", status="failed", instruction="Command failed", workflow_agent_kind="executor",
        error=ModuleError(code="tool_failed", message="exit 1", retryable=False))
    run.work_requests = [WorkRequest(owner_task_id="task_root", owner_attempt_number=1, depth=2, id="work_test", run_id=run.run_id, owner_session_id=run.scientific_session.id,
        request=WorkRequestDraft(objective='Measure', tasks=[{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Measure'}]), status="consumed", workflow_revision=1,
        outcome=WorkResult(run_id=run.run_id, session_id=run.scientific_session.id, objective="Measure", status="stable", work_request_id="work_test", workflow_revision=1, tasks=[issue]),
        created_at=run.created_at, updated_at=run.updated_at)]
    result, refs = registered_result(prepared)
    assert not gate.validate(run, result, refs).ok
    result, refs = registered_result(prepared, limitations=["Execution failed"])
    actual = gate.validate(run, result, refs)
    assert actual.ok and actual.report.execution_issues == [issue]


def test_compilation_failure_without_tasks_requires_limitations_and_is_reported(prepared):
    _, run, gate = prepared
    error = ModuleError(code="contract_error", message="Compilation did not produce a valid graph", retryable=False)
    run.work_requests = [WorkRequest(owner_task_id="task_root", owner_attempt_number=1, depth=2,
        id="work_uncompiled", run_id=run.run_id, owner_session_id=run.scientific_session.id,
        request=WorkRequestDraft(objective='Measure', tasks=[{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, 'key': 'work', 'instruction': 'Measure'}]), status="failed", error=error,
        outcome=WorkResult(run_id=run.run_id, session_id=run.scientific_session.id, work_request_id="work_uncompiled",
                           objective="Measure", status="failed", error=error),
        created_at=run.created_at, updated_at=run.updated_at,
    )]
    result, refs = registered_result(prepared)
    assert CompletionViolationCode.MISSING_LIMITATIONS in {item.code for item in gate.validate(run, result, refs).violations}
    result, refs = registered_result(prepared, limitations=["No measurement was possible because compilation failed"])
    validation = gate.validate(run, result, refs)
    assert validation.ok
    assert validation.report.execution_issues == []
    rendered = FinalReportRenderer().render(validation.report).content
    assert "work_uncompiled" in rendered and error.message in rendered


@pytest.mark.parametrize("fault", ["missing_requirement", "missing_opinion", "foreign_session", "corrupt_opinion"])
def test_completion_rejects_missing_or_corrupt_authority(prepared, fault):
    from pathlib import Path
    _, run, gate = prepared
    result, refs = registered_result(prepared)
    if fault == "missing_requirement": run.conclusion_requirements_ref = None
    elif fault == "missing_opinion": refs = refs[1:]
    elif fault == "foreign_session": result.session = result.session.model_copy(update={"id": "session_foreign"})
    else: Path(refs[0].uri.removeprefix("file://")).write_text("{}")
    assert not gate.validate(run, result, refs).ok


def test_renderer_and_registration_are_deterministic(prepared):
    registry, run, gate = prepared
    result, refs = registered_result(prepared)
    accepted = gate.validate(run, result, refs)
    first = FinalReportRenderer().render(accepted.report)
    assert first == FinalReportRenderer().render(accepted.report)
    one = registry.register_final_report(first.candidate, first.content, run_id=run.run_id)
    two = registry.register_final_report(first.candidate, first.content, run_id=run.run_id)
    assert one.id != two.id and one.sha256 == two.sha256


def test_scientific_registration_is_idempotent_and_rejects_system_kind(prepared):
    registry, run, _ = prepared
    first = evidence(prepared)
    assert first.id != evidence(prepared).id
    assert first.task_id is None and first.producer == AgentOwner.SCIENTIFIC
    with pytest.raises(ArtifactRegistrationError):
        registry.register_scientific(candidate("acceptance_requirements", {}), run_id=run.run_id, session_id=run.scientific_session.id)


@pytest.mark.parametrize("kind", sorted(SYSTEM_ARTIFACT_PROVENANCE))
def test_each_system_kind_supports_its_defined_scope_and_replay(prepared, kind):
    registry, run, _ = prepared
    source, scopes = SYSTEM_ARTIFACT_PROVENANCE[kind]
    for scope in scopes:
        fields = {"run": {}, "task": {"task_id": "task_test"}, "attempt": {"task_id": "task_test", "attempt_number": 1}, "session": {"session_id": run.scientific_session.id}}[scope]
        first = registry.register_system_artifact(candidate(kind, {}), run_id=run.run_id, source_type=source, **fields)
        assert first.id != registry.register_system_artifact(candidate(kind, {}), run_id=run.run_id, source_type=source, **fields).id
        with pytest.raises(ValueError):
            registry.register_system_artifact(candidate(kind, {}), run_id=run.run_id, source_type="forged", **fields)


def require_outputs(prepared, names):
    registry, run, _ = prepared
    run.conclusion_requirements_ref = system_artifact(
        registry, run, "conclusion_requirements", ConclusionRequirements(required_artifacts=names),operation_key="test:required_outputs",
    )


def named_delivery(prepared, *, name="metrics", run_id=None, path="measurements.json"):
    registry, run, _ = prepared
    ref = registry.register_scientific(
        ArtifactCandidate(
            kind="web_page", path=path, media_type="application/json",
            summary="Delivered report", content='{"measurement": 1}', output_name=name,
        ),
        run_id=run_id or run.run_id, session_id=run.scientific_session.id,
    )
    run.artifacts[ref.id] = ref
    return ref


def test_required_output_does_not_force_observation_or_citation(prepared):
    _, run, gate = prepared
    require_outputs(prepared, ["metrics"])
    ref = named_delivery(prepared)
    result, refs = registered_result(prepared)
    accepted = gate.validate(run, result, refs)
    assert accepted.ok
    assert accepted.report.evidence == []
    assert ref.id not in accepted.report.opinion.evidence_artifact_ids


@pytest.mark.parametrize("source", ["missing", "wrong_case", "filename", "foreign_run", "unregistered"])
def test_gate_requires_registered_exact_name_in_same_run(prepared, tmp_path, source):
    _, run, gate = prepared
    require_outputs(prepared, ["metrics"])
    if source == "wrong_case":
        named_delivery(prepared, name="Metrics")
    elif source == "filename":
        named_delivery(prepared, name=None, path="metrics")
    elif source == "foreign_run":
        named_delivery(prepared, run_id="run_foreign")
    elif source == "unregistered":
        ref = named_delivery(prepared)
        del run.artifacts[ref.id]
    else:
        (tmp_path / "metrics").write_text("A workspace file alone is insufficient", encoding="utf-8")
    result, refs = registered_result(prepared)
    actual = gate.validate(run, result, refs)
    assert not actual.ok
    assert [(item.code.value, item.message, item.subject, item.related_ids) for item in actual.violations] == [
        ("required_artifact_missing", "required artifact was not produced", "metrics", []),
    ]


@pytest.mark.parametrize("fault", ["missing", "corrupt"])
def test_gate_rejects_unreadable_registered_delivery_as_authority_error(prepared, fault):
    from pathlib import Path

    _, run, gate = prepared
    require_outputs(prepared, ["metrics"])
    ref = named_delivery(prepared)
    path = Path(ref.uri.removeprefix("file://"))
    if fault == "missing":
        path.unlink()
    else:
        path.write_text("changed", encoding="utf-8")
    result, refs = registered_result(prepared)
    actual = gate.validate(run, result, refs)
    assert not actual.ok
    assert CompletionViolationCode.INVALID_OPINION in {item.code for item in actual.violations}
    assert CompletionViolationCode.REQUIRED_ARTIFACT_MISSING not in {item.code for item in actual.violations}


def test_registered_deliveries_from_different_rounds_can_share_a_required_name(prepared):
    registry, run, gate = prepared
    require_outputs(prepared, ["metrics"])
    named_delivery(prepared)
    later = registry.register(
        ArtifactCandidate(
            kind="experiment_result", path="later.json", media_type="application/json",
            summary="Later measured metrics", content='{"measurement": 2}', output_name="metrics",
        ),
        grant=None, producer=AgentOwner.EXECUTOR, run_id=run.run_id,
        task_id="task_later", attempt_number=1, index=1,
    )
    run.artifacts[later.id] = later
    result, refs = registered_result(prepared)
    assert gate.validate(run, result, refs).ok


def test_pending_scientific_output_only_counts_after_receiving_registration(prepared):
    registry, run, gate = prepared
    require_outputs(prepared, ["analysis"])
    pending = ArtifactCandidate(
        kind="web_page", path="analysis.md", media_type="text/markdown",
        summary="Requested analysis", content="The completed analysis", output_name="analysis",
    )
    result, refs = registered_result(prepared)
    result.artifacts.append(pending)
    rejected = gate.validate(run, result, refs)
    assert not rejected.ok
    assert any(item.subject == "analysis" for item in rejected.violations)

    registered = registry.register_scientific(
        pending, run_id=run.run_id, session_id=run.scientific_session.id,
    )
    run.artifacts[registered.id] = registered
    result.artifacts[-1] = registered
    assert gate.validate(run, result, [*refs, registered]).ok


def test_gate_without_output_requirements_does_not_read_unrelated_files(prepared):
    from pathlib import Path

    _, run, gate = prepared
    ref = named_delivery(prepared)
    Path(ref.uri.removeprefix("file://")).unlink()
    result, refs = registered_result(prepared)
    assert gate.validate(run, result, refs).ok


def test_registry_uses_supplied_run_map_and_verifies_frozen_delivery(prepared):
    registry, run, _ = prepared
    ref = named_delivery(prepared)
    assert registry.missing_required_artifacts(
        ["metrics"], run_id=run.run_id, artifacts=run.artifacts,
    ) == []
    assert registry.missing_required_artifacts(
        ["metrics"], run_id=run.run_id, artifacts={},
    ) == ["metrics"]
    assert registry.missing_required_artifacts(
        ["metrics"], run_id="run_other", artifacts={ref.id: ref},
    ) == ["metrics"]


def test_completion_does_not_require_an_observation_trace(prepared):
    _, run, gate = prepared
    ref = evidence(prepared)
    result, refs = registered_result(prepared, cited=[ref.id])
    refs = [item for item in refs if item.kind != "observation_trace"]
    result.artifacts = refs
    assert gate.validate(run, result, refs).ok


@pytest.mark.parametrize("fault", ["unknown", "missing", "corrupt"])
def test_unread_citations_still_require_registered_frozen_content(prepared, fault):
    from pathlib import Path

    _, run, gate = prepared
    ref = evidence(prepared)
    cited = "artifact_unknown" if fault == "unknown" else ref.id
    if fault == "missing":
        Path(ref.uri.removeprefix("file://")).unlink()
    elif fault == "corrupt":
        Path(ref.uri.removeprefix("file://")).write_text("changed")
    result, refs = registered_result(prepared, cited=[cited])
    assert not gate.validate(run, result, refs).ok
