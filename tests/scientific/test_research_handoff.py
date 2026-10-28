"""Direct work navigation replaces persisted ResearchIndex and feedback snapshots."""
import hashlib
import json
from datetime import UTC,datetime
from resagent2_contracts import AgentOwner,AgentPermissions,AgentRequest,ArtifactRef,TaskBudget,WorkResult,WorkTaskResult,WorkAttemptResult
from resagent2_runtime import AgentState
from resagent2_scientific.context import build_context

def test_work_tasks_and_attempt_reports_are_fully_visible_without_report_bodies(tmp_path):
    now=datetime.now(UTC)
    path=tmp_path/"report.md";path.write_text("BODY_MUST_BE_READ")
    ref=ArtifactRef(id="artifact_report",kind="module_report",producer="orchestrator",run_id="run_handoff",
                    task_id="task_a",attempt_number=1,uri=path.as_uri(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    media_type="text/markdown",summary="Implementation findings and limitations",metadata={"source_type":"task_report"})
    result=WorkResult(run_id="run_handoff",session_id="session_handoff",work_request_id="work_a",objective="Implement and evaluate",status="stable",
                      artifacts=[ref],tasks=[WorkTaskResult(task_id="task_a",workflow_agent_kind="executor",instruction="Implement method",status="completed",
                        attempts=[WorkAttemptResult(number=1,status="completed",report_artifact_id=ref.id,artifact_ids=[])])])
    request=AgentRequest(work_request_id="work_root", task_id="task_root", attempt_number=1, run_id="run_handoff",agent="scientific",instruction="Judge evidence",parent_session_id="session_handoff",permissions=AgentPermissions(),
                         budget=TaskBudget(max_llm_calls=10,timeout_seconds=60),input_artifacts=[ref],work_results=[result])
    state=AgentState(work_request_id="work_root", task_id="task_root", attempt_number=1, session_id="session_handoff",agent_name="scientific",owner=AgentOwner.SCIENTIFIC,run_id="run_handoff",created_at=now,updated_at=now)
    sections=build_context(request,state)
    assert [section.name for section in sections]==["task","runtime_authority","material_index"]
    directory=next(section.content for section in sections if section.name=="material_index")
    assert all(text in directory for text in ("work_a","Implement and evaluate","task_a",ref.id,ref.summary))
    assert "BODY_MUST_BE_READ" not in directory
    assert "work_record" not in directory and "research_index" not in directory
    tree = json.loads(directory)
    def nodes(node):
        yield node
        for child in node["children"]:
            yield from nodes(child)
    artifact_nodes = [node for node in nodes(tree) if node["type"] == "artifact"]
    assert len(artifact_nodes) == tree["artifact_count"] == 1
    assert artifact_nodes[0]["location"] == {"artifact_id": "artifact_report"}
