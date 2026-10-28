"""Scientific uses the common result while retaining evidence and recovery rules."""

from e2e.native_fixtures import tool_turns

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from resagent2_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ArtifactRef, ErrorCode,
    ScientificOpinion, TaskBudget, RecordedAnswer, WorkResult, WorkTaskResult, task_session_id,
)
from resagent2_scientific import ScientificAgent
from resagent2_scientific.tools import RequestWorkTool
from resagent2_runtime import AgentState, ScriptedLLMClient, ToolRegistry


def request(*, artifacts=(), parent=None, budget=10, request_work=True, **extra):
    return AgentRequest(work_request_id="work_root", task_id="task_root", attempt_number=1, run_id='run_example', agent=AgentOwner.SCIENTIFIC, instruction='Evaluate the method', input_artifacts=list(artifacts), parent_session_id=parent, budget=TaskBudget(max_llm_calls=budget, timeout_seconds=60), permissions=AgentPermissions(request_work=request_work, execute_commands=True), **extra)


def artifact(root, artifact_id="artifact_1", *, kind="experiment_result", content=None, session=None):
    content = json.dumps({"value": 1} if content is None else content)
    path = root / (artifact_id + ".json")
    path.write_text(content)
    kwargs = dict(
        id=artifact_id, kind=kind, run_id="run_example", uri=path.as_uri(),
        sha256=hashlib.sha256(content.encode()).hexdigest(), media_type="application/json", summary=kind,
    )
    if kind == "experiment_result":
        kwargs.update(producer=AgentOwner.EXECUTOR, task_id="task_exp", attempt_number=1)
    else:
        kwargs.update(producer=AgentOwner.ORCHESTRATOR, metadata={"source_type": "import"})
        if session:
            kwargs["session_id"] = session
    return ArtifactRef(**kwargs)


def finish(*, evidence=(), verdict="inconclusive", **opinion):
    body = {"verdict": verdict, "statement": "A statement",
            "evidence_artifact_ids": list(evidence), **opinion}
    return {"tool": "finish", "arguments": {
        "report": "Scientific conclusion is ready",
        "artifacts": [{"kind": "scientific_opinion", "path": "scientific_opinion.json",
                       "media_type": "application/json", "summary": "Scientific judgment",
                       "content": body}],
    }}


def content(result, kind):
    item = next(item for item in result.artifacts if item.kind == kind)
    return json.loads(item.content)


def work():
    return {"tool": "request_work", "arguments": {
        "assessment": {"statement": "Need more evidence"},
        "work_request": {'objective': 'Run the experiment', 'tasks': [{"selection": {"kind": "preset", "preset_ref": "coding@1", "reason": "Suitable configured executor"}, 'key': 'work', 'instruction': 'Run the experiment'}]},
    }}


def ask():
    return {"tool": "ask_user", "arguments": {
        "assessment": {"statement": "Need a dataset choice"},
        "text": "No dataset is selected. Which dataset should we use?", "requested_fields": ["dataset"],
    }}


def test_finish_with_existing_evidence_completes(tmp_path):
    ref = artifact(tmp_path)
    agent = ScientificAgent(ScriptedLLMClient(tool_turns([
        {"tool": "read_artifact", "arguments": {"artifact_id": ref.id}},
        finish(evidence=[ref.id], verdict="supports"),
    ])))
    result = agent.invoke(request(artifacts=[ref]))
    assert result.status == "completed", result.report
    assert content(result, "scientific_opinion")["verdict"] == "supports"
    assert content(result, "observation_trace")["observed_artifact_ids"] == [ref.id]


def test_request_work_has_direct_stable_control_without_artifact():
    result = ScientificAgent(ScriptedLLMClient(tool_turns([work()]))).invoke(request())
    assert result.status == "request_work", result.report
    assert result.control.action == "request_work"
    assert result.control.work_request.request.tasks[0].instruction == "Run the experiment"
    assert result.artifacts == []
    assert result.control.work_request.id


def test_cached_control_does_not_bypass_fixed_tool_contract_validation():
    client = ScriptedLLMClient(tool_turns([work()]))
    agent = ScientificAgent(client)
    first = agent.invoke(request())
    agent.web_search_backend = object()
    changed = agent.invoke(request())
    assert first.status == "request_work"
    assert changed.status == "failed"
    assert "different fixed contract" in changed.report
    assert len(client.contexts) == 1


def test_unanswered_resume_does_not_cache_away_a_later_business_answer():
    client = ScriptedLLMClient(tool_turns([ask(), finish()]))
    agent = ScientificAgent(client)
    first = agent.invoke(request())
    still_waiting = agent.invoke(request(parent=first.session.id))
    assert still_waiting.control == first.control
    question = first.control.question
    answer = RecordedAnswer(task_id=question.task_id, attempt_number=question.attempt_number,
        question_id=question.id, question_text=question.text, requested_fields=question.requested_fields,
        options=question.options, values={"dataset": "CIFAR-10"}, answered_at=datetime.now(UTC),
        run_id=question.run_id, session_id=question.session_id,
    )
    final = agent.invoke(request(parent=first.session.id, input_ids=[answer.question_id], user_answers=[answer], resume_answers=[answer]))
    assert final.status == "completed", final.report
    assert len(client.contexts) == 2


def test_failed_work_without_tasks_requires_limitations_and_corrects_in_same_session():
    from resagent2_contracts import ModuleError
    client = ScriptedLLMClient(tool_turns([
        work(), finish(), finish(limitations=["Requested implementation could not be compiled"]),
    ]))
    agent = ScientificAgent(client)
    first = agent.invoke(request())
    result = WorkResult(
        run_id="run_example", session_id=first.session.id,
        work_request_id=first.control.work_request.id, objective="Run the experiment", status="failed",
        error=ModuleError(code="contract_error", message="Compilation failed before tasks existed", retryable=False),
    )
    final = agent.invoke(request(parent=first.session.id, control_ack=first.control.origin, work_results=[result]))
    assert final.status == "completed", final.report
    state = agent.store.load(first.session.id)
    rejected = state.tool_turns[1]
    receipt = json.loads(rejected.tool_results[rejected.tool_calls[0].id])
    assert receipt["ok"] is False
    assert "at least one limitation" in receipt["summary"]
    assert final.llm_calls == 2
    assert len(state.tool_turns) == 3
    assert content(final, "scientific_opinion")["limitations"] == ["Requested implementation could not be compiled"]




def test_request_work_permission_is_enforced():
    result = ScientificAgent(ScriptedLLMClient(tool_turns([work()]))).invoke(request(request_work=False))
    assert result.status == "failed"


def test_empty_tasks_rejected_before_tool_execution(monkeypatch):
    tool = RequestWorkTool()
    monkeypatch.setattr(tool, "execute", lambda *_: pytest.fail("Invalid arguments executed"))
    now = datetime.now(UTC)
    state = AgentState(work_request_id="work_root", task_id="task_root", attempt_number=1, session_id="session_v", agent_name="scientific",
                       owner=AgentOwner.SCIENTIFIC, run_id="run_example", created_at=now, updated_at=now)
    action = work()["arguments"]
    action["work_request"]["tasks"] = []
    with pytest.raises(ValidationError):
        ToolRegistry((tool,)).dispatch("request_work", action, state)


def test_imported_literature_can_satisfy_requirements_after_reading(tmp_path):
    ref = artifact(tmp_path, kind="literature_paper")
    client = ScriptedLLMClient(tool_turns([
        {"tool": "read_artifact", "arguments": {"artifact_id": ref.id}},
        finish(evidence=[ref.id], verdict="supports"),
    ]))
    result = ScientificAgent(client).invoke(request(artifacts=[ref], task_context={
        "conclusion_requirements": {"required_evidence_kinds": ["literature_paper"]},
    }))
    assert result.status == "completed", result.report
    assert content(result, "observation_trace")["observed_artifact_ids"] == [ref.id]
    assert result.llm_calls == 2
    assert "conclusion_requirements" in client.contexts[0].task_context


def test_literature_search_returns_temporary_candidates_without_registration(tmp_path):
    from resagent2_components import LiteraturePaper
    class Backend:
        def search(self, query, **kwargs):
            return _search_result([LiteraturePaper(paper_id="arxiv:1234.5678", title="Paper", authors=["Author"],abstract="Result",source_url="https://arxiv.org/abs/1234.5678")])
    client = ScriptedLLMClient(tool_turns([
        {"tool":"literature_search","arguments":{"query":"method"}}, finish(),
    ]))
    result = ScientificAgent(client,literature_backend=Backend()).invoke(request())
    assert result.status == "completed", result.report
    assert not any(item.kind.startswith("literature_") for item in result.artifacts)
    assert "Paper" in str(client.histories[1][0].tool_results)
    assert "Paper" not in client.contexts[1].material_index




def test_question_pauses_with_content_reference():
    result = ScientificAgent(ScriptedLLMClient(tool_turns([ask()]))).invoke(request())
    assert result.status == "needs_user_input"
    assert result.control.action == "ask_user"
    assert result.control.question.requested_fields == ["dataset"]
    assert result.artifacts == []


def test_question_schema_retains_visible_background_guidance():
    from resagent2_scientific.models import AskUserInput
    from resagent2_runtime.tools import AskUserToolInput
    assert AskUserInput.model_fields["text"].description == AskUserToolInput.model_fields["text"].description
    assert "background" in AskUserInput.model_fields["text"].description


def test_work_result_resume_reuses_session_and_is_idempotent(tmp_path):
    client = ScriptedLLMClient(tool_turns([work(),finish()]))
    agent=ScientificAgent(client)
    first=agent.invoke(request())
    outcome=WorkResult(run_id="run_example",session_id=first.session.id,
                       work_request_id=first.control.work_request.id,objective="Run the experiment",status="stable")
    resumed=request(parent=first.session.id,work_results=[outcome],control_ack=first.control.origin)
    result=agent.invoke(resumed)
    duplicate=agent.invoke(resumed)
    assert result.status == duplicate.status == "completed"
    assert duplicate.llm_calls == 0 and len(client.contexts)==2
    assert first.control.work_request.id in client.contexts[1].material_index




def test_answer_resume_projects_full_question_answer_without_artifacts(tmp_path):
    client=ScriptedLLMClient(tool_turns([ask(),finish()]))
    agent=ScientificAgent(client);first=agent.invoke(request());q=first.control.question
    answer=RecordedAnswer(task_id=q.task_id, attempt_number=q.attempt_number, question_id=q.id,question_text=q.text,requested_fields=q.requested_fields,
                          options=q.options,values={"dataset":"CIFAR-10"},answered_at=datetime.now(UTC),
                          run_id="run_example",session_id=first.session.id)
    result=agent.invoke(request(parent=first.session.id,input_ids=[answer.question_id], user_answers=[answer],resume_answers=[answer],control_ack=first.control.origin))
    assert result.status=="completed",result.report
    assert "CIFAR-10" in client.contexts[-1].task_context
    assert q.text in client.contexts[-1].task_context
    assert not any(item.kind in {"question","answer"} for item in result.artifacts)




def test_unread_citation_is_rejected(tmp_path):
    result = ScientificAgent(ScriptedLLMClient(tool_turns([finish(evidence=["artifact_unread"], verdict="supports")]))).invoke(
        request(artifacts=[artifact(tmp_path)]),
    )
    assert result.status == "failed"
    assert result.error.code == ErrorCode.TOOL_FAILED


def test_unread_citation_can_be_recovered_by_reading(tmp_path):
    ref = artifact(tmp_path)
    agent = ScientificAgent(ScriptedLLMClient(tool_turns([
        finish(evidence=[ref.id], verdict="supports"),
        {"tool": "read_artifact", "arguments": {"artifact_id": ref.id}},
        finish(evidence=[ref.id], verdict="supports"),
    ])))
    result = agent.invoke(request(artifacts=[ref]))
    assert result.status == "completed"


def test_failed_work_requires_limitation_in_opinion(tmp_path):
    from resagent2_contracts import ModuleError
    agent=ScientificAgent(ScriptedLLMClient(tool_turns([work(),finish()])))
    first=agent.invoke(request())
    failed=WorkTaskResult(task_id="task_failed",workflow_agent_kind="executor",instruction="Measure",status="failed",
                          error=ModuleError(code=ErrorCode.TOOL_FAILED,message="Crashed",retryable=False))
    outcome=WorkResult(run_id="run_example",session_id=first.session.id,work_request_id=first.control.work_request.id,
                       objective="Run the experiment",status="stable",tasks=[failed])
    result=agent.invoke(request(parent=first.session.id,work_results=[outcome],control_ack=first.control.origin))
    assert result.status == "failed"




def test_budget_exhaustion_counts_real_calls(tmp_path):
    ref = artifact(tmp_path)
    result = ScientificAgent(ScriptedLLMClient(tool_turns([
        {"tool": "read_artifact", "arguments": {"artifact_id": ref.id}},
    ] * 4))).invoke(request(artifacts=[ref], budget=2))
    assert result.status == "failed"
    assert result.error.code == ErrorCode.BUDGET_EXHAUSTED
    assert result.llm_calls == 2


@pytest.mark.parametrize("tool", ["ask_user", "request_work"])
def test_assessment_cannot_cite_unobserved_evidence(tool):
    action = ask() if tool == "ask_user" else work()
    action["arguments"]["assessment"]["evidence_artifact_ids"] = ["artifact_unread"]
    result = ScientificAgent(ScriptedLLMClient(tool_turns([action]))).invoke(request())
    assert result.status == "failed"


def test_repeated_first_request_uses_persisted_result():
    client = ScriptedLLMClient(tool_turns([work(), finish()]))
    agent = ScientificAgent(client)
    first = agent.invoke(request())
    duplicate = agent.invoke(request())
    assert first.status == duplicate.status == "request_work"
    assert first.control == duplicate.control
    assert duplicate.llm_calls == 0
    assert len(client.contexts) == 1


def test_earlier_reads_remain_in_context(tmp_path):
    a = artifact(tmp_path, "artifact_a")
    b = artifact(tmp_path, "artifact_b")
    client = ScriptedLLMClient(tool_turns([
        {"tool": "read_artifact", "arguments": {"artifact_id": a.id}},
        {"tool": "read_artifact", "arguments": {"artifact_id": b.id}}, finish(evidence=[a.id, b.id]),
    ]))
    result = ScientificAgent(client).invoke(request(artifacts=[a, b]))
    assert result.status == "completed"
    assert "artifact_reads" not in client.contexts[-1].included_sections
    assert all(ref.id in str(client.histories[-1]) for ref in (a,b))
    assert content(result, "observation_trace")["observed_artifact_ids"] == [a.id, b.id]


def test_obsolete_opinion_fields_are_rejected():
    result = ScientificAgent(ScriptedLLMClient(tool_turns([
        finish(acknowledged_task_ids=["task_unknown"]),
    ]))).invoke(request())
    assert result.status == "failed"


def _search_result(papers):
    from resagent2_components import LiteratureSearchResult
    return LiteratureSearchResult(papers=papers, source="arxiv", executed_query="test query",
                                  page=1, next_page=None, source_attempts=[{
                                      "source": "arxiv", "status": "success",
                                      "error_type": None, "error": None,
                                  }])


def test_standalone_role_permissions_do_not_imply_a_run_execution_grant():
    client = ScriptedLLMClient(tool_turns([finish()]))
    result = ScientificAgent(client).invoke(request())
    assert result.status == "completed", result.report
    context = client.contexts[0]
    task = json.loads(context.task_context.split("\n", 1)[1])
    assert task["instruction"] == "Evaluate the method"
    assert "permissions" not in task
    authority = json.loads(context.runtime_context.split("\n", 1)[1])
    assert authority["current_agent_permissions"]["execute_commands"] is True
    assert authority["run_execution_permissions"] is None


def test_finish_schema_uses_canonical_opinion_and_serializes_before_completion():
    from resagent2_runtime import FinishTool
    from resagent2_runtime.tool_calling import native_tool_schemas

    client = ScriptedLLMClient(tool_turns([finish(statement="不确定的结论")]))
    agent = ScientificAgent(client)
    result = agent.invoke(request())
    assert result.status == "completed", result.report
    schema = next(item["function"]["parameters"] for item in client.schemas[0]
                  if item["function"]["name"] == "finish")
    canonical = ScientificOpinion.model_json_schema()
    canonical.pop("$defs")
    assert schema["$defs"]["ScientificOpinion"] == canonical
    assert schema["$defs"]["ScientificOpinion"]["additionalProperties"] is False
    opinion_candidate = schema["$defs"]["ScientificOpinionCandidate"]
    assert opinion_candidate["properties"]["content"] == {"$ref": "#/$defs/ScientificOpinion"}
    assert schema["$defs"]["OtherScientificArtifactCandidate"]["properties"]["kind"]["not"] == {
        "const": "scientific_opinion",
    }
    assert "ScientificOpinion" not in native_tool_schemas((FinishTool(),))[0]["function"]["parameters"]["$defs"]
    item = next(item for item in result.artifacts if item.kind == "scientific_opinion")
    expected = ScientificOpinion(verdict="inconclusive", statement="不确定的结论")
    assert item.content == expected.model_dump_json()


@pytest.mark.parametrize("invalid", [
    {"verdict": "inconclusive", "statement": "Conclusion", "extra": "forbidden"},
    {"verdict": "supports", "statement": "Missing evidence"},
    {"verdict": "invented", "statement": "Invalid verdict"},
    '{"verdict":"inconclusive","statement":"String bypass"}',
    None,
])
def test_opinion_schema_rejection_is_correctable_without_generic_candidate_escape(invalid):
    bad = finish()
    bad["arguments"]["artifacts"][0]["content"] = invalid
    client = ScriptedLLMClient(tool_turns([bad, finish()]))
    result = ScientificAgent(client).invoke(request())
    assert result.status == "completed", result.report
    assert result.llm_calls == 2
    rejected = client.histories[1][0]
    receipt = json.loads(rejected.tool_results[rejected.tool_calls[0].id])
    assert receipt["ok"] is False
    assert "validation" in receipt["summary"].lower()


def test_structured_opinion_keeps_registered_citation_checks_and_failed_rejection():
    invalid_ref = finish(evidence=["artifact_missing"], verdict="supports")
    failed = finish()
    failed["arguments"]["status"] = "failed"
    client = ScriptedLLMClient(tool_turns([invalid_ref, failed, finish()]))
    result = ScientificAgent(client).invoke(request())
    assert result.status == "completed", result.report
    assert result.llm_calls == 3
    history = client.histories[2]
    receipts = [json.loads(turn.tool_results[turn.tool_calls[0].id]) for turn in history]
    assert "Invalid evidence citation" in receipts[0]["summary"]
    assert "requires a completed opinion" in receipts[1]["summary"]


def test_legacy_finish_schema_session_is_rejected_without_rewriting_saved_record():
    from resagent2_runtime.tools import FinishInput
    from resagent2_runtime.tool_calling import contract_fingerprint
    from resagent2_scientific.context import SCIENTIFIC_PROMPT

    client = ScriptedLLMClient(tool_turns([work()]))
    agent = ScientificAgent(client)
    first = agent.invoke(request())
    saved = agent.store.load(first.session.id)
    old_schemas = json.loads(json.dumps(client.schemas[0]))
    old_finish = next(item["function"] for item in old_schemas if item["function"]["name"] == "finish")
    old_finish["parameters"] = FinishInput.model_json_schema()
    saved.contract_fingerprint = contract_fingerprint(SCIENTIFIC_PROMPT, old_schemas)
    agent.store.save(saved)
    before = agent.store.load(first.session.id).model_dump_json()
    resumed = agent.invoke(request(parent=first.session.id))
    assert resumed.status == "failed"
    assert "different fixed contract" in resumed.report
    assert len(client.contexts) == 1
    assert agent.store.load(first.session.id).model_dump_json() == before


@pytest.mark.parametrize("content", [None, "Ordinary UTF-8 text"])
def test_scientific_adapter_preserves_ordinary_file_and_text_candidate_semantics(content):
    from resagent2_contracts import ArtifactCandidate
    from resagent2_runtime import FinishInput
    from resagent2_scientific.models import ScientificFinishInput

    ordinary = ArtifactCandidate(
        kind="scientific_assessment", path="reports/assessment.json",
        media_type="application/json", summary="Supporting output", content=content,
    )
    arguments = finish()["arguments"]
    arguments["artifacts"].append(ordinary.model_dump(mode="json"))
    normalized = ScientificFinishInput.model_validate(arguments).to_candidate()
    assert normalized.artifacts[1] == ordinary
    # Executor uses the ordinary schema and string/null content without the domain adapter.
    generic = FinishInput(report="Executor output", artifacts=[ordinary]).to_candidate()
    assert generic.artifacts == [ordinary]
