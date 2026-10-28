"""Reading logs never qualify citations or force Scientific control actions."""

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from resagent2_components import RegisteredArtifactReader
from resagent2_contracts import AgentOwner, AgentPermissions, AgentRequest, ArtifactCandidate, TaskBudget
from resagent2_orchestrator import ArtifactRegistry
from resagent2_runtime import AgentState, FinishCandidate
from resagent2_scientific.completion import ScientificCompletionCheck
from resagent2_scientific.context import build_context
from resagent2_scientific.models import AskUserInput, RequestWorkInput
from resagent2_scientific.tools import AskUserTool, RequestWorkTool


def _state(memory=None):
    now = datetime.now(UTC)
    return AgentState(work_request_id="work_root", task_id="task_root", attempt_number=1,
        session_id="session_s", agent_name="scientific", owner=AgentOwner.SCIENTIFIC,
        run_id="run_s", created_at=now, updated_at=now, memory=memory or {},
    )


def _finish(artifact_id):
    return FinishCandidate(report="Supported", artifacts=[ArtifactCandidate(
        kind="scientific_opinion", path="opinion.json", media_type="application/json",
        summary="Conclusion", content=json.dumps({
            "verdict": "supports", "statement": "Supported", "evidence_artifact_ids": [artifact_id],
        }),
    )])


def _source(tmp_path, *, run_id="run_s"):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    ref = registry.register_scientific(ArtifactCandidate(
        kind="web_page", path="evidence.txt", media_type="text/plain",
        summary="Source report", content="Source content",
    ), run_id=run_id, session_id="session_s")
    return ref, RegisteredArtifactReader([ref], run_id="run_s")


@pytest.mark.parametrize("logged", [False, True])
def test_completion_accepts_registered_citation_independently_of_reading_log(tmp_path, logged):
    ref, reader = _source(tmp_path)
    state = _state({"read_artifact_ids": [ref.id]} if logged else {})
    decision = ScientificCompletionCheck([], reader=reader).evaluate(state, _finish(ref.id))
    assert decision.complete
    assert state.memory == ({"read_artifact_ids": [ref.id]} if logged else {})


@pytest.mark.parametrize("fault", ["unknown", "foreign_run", "corrupt"])
def test_read_log_never_bypasses_citation_authority(tmp_path, fault):
    ref, reader = _source(tmp_path, run_id="run_other" if fault == "foreign_run" else "run_s")
    cited = "artifact_unknown" if fault == "unknown" else ref.id
    if fault == "corrupt":
        Path(ref.uri.removeprefix("file://")).write_text("changed")
    decision = ScientificCompletionCheck([], reader=reader).evaluate(
        _state({"read_artifact_ids": [cited]}), _finish(cited),
    )
    assert not decision.complete
    assert "Invalid evidence citation" in decision.report


@pytest.mark.parametrize("action", ["request_work", "ask_user"])
def test_registered_unread_citation_does_not_block_control_tools(tmp_path, action):
    ref, reader = _source(tmp_path)
    assessment = {"statement": "Need a research decision", "evidence_artifact_ids": [ref.id]}
    if action == "request_work":
        tool = RequestWorkTool(reader=reader)
        args = RequestWorkInput.model_validate({
            "assessment": assessment,
            "work_request": {'objective': 'Measure', 'tasks': [{"selection": {"kind": "preset", "preset_ref": "coding@1", "reason": "Suitable configured executor"}, 'key': 'work', 'instruction': 'Measure'}]},
        })
    else:
        tool = AskUserTool(reader)
        args = AskUserInput.model_validate({
            "assessment": assessment, "text": "Which measurement?", "requested_fields": ["measurement"],
        })
    result = tool.execute(_state(), args)
    assert result.ok
    assert result.request_work is not None if action == "request_work" else result.question is not None


def test_context_has_no_pending_citation_action():
    request = AgentRequest(work_request_id="work_root", task_id="task_root", attempt_number=1,
        run_id="run_s", agent=AgentOwner.SCIENTIFIC, instruction="Judge supplied evidence",
        budget=TaskBudget(max_llm_calls=10, timeout_seconds=60), permissions=AgentPermissions(),
    )
    sections = build_context(request, _state({
        "read_artifact_ids": ["artifact_old"],
        "pending_citation_artifact_ids": ["artifact_unread"],
    }))
    assert "evidence_control_state" not in {section.name for section in sections}
    assert "read_artifact_or_remove_citation" not in str(sections)


@pytest.mark.parametrize("fault", ["unknown", "foreign_run", "corrupt", "missing"])
def test_work_inputs_are_rejected_without_emitting_control_or_observing(tmp_path, fault):
    ref, reader = _source(tmp_path, run_id="run_other" if fault == "foreign_run" else "run_s")
    artifact_id = "artifact_unknown" if fault == "unknown" else ref.id
    path = Path(ref.uri.removeprefix("file://"))
    if fault == "corrupt":
        path.write_text("changed")
    elif fault == "missing":
        path.unlink()
    state = _state()
    observation = RequestWorkTool(reader=reader).execute(state, RequestWorkInput(
        assessment={"statement": "Need a measurement"},
        work_request={'objective': 'Measure', 'tasks': [{"selection": {"kind": "preset", "preset_ref": "coding@1", "reason": "Suitable configured executor"}, 'key': 'work', 'instruction': 'Measure', 'input_artifact_ids': [artifact_id]}]},
    ))
    assert not observation.ok
    expected = {"unknown": "not authorized", "foreign_run": "not authorized",
                "corrupt": "sha256", "missing": "file is missing"}[fault]
    assert "Work input artifact" in observation.summary
    assert expected in observation.summary
    assert observation.request_work is None
    assert not observation.memory_updates
    assert not state.memory


def test_work_inputs_verify_frozen_bytes_without_requiring_or_recording_a_read(tmp_path):
    ref, reader = _source(tmp_path)
    state = _state()
    observation = RequestWorkTool(reader=reader).execute(state, RequestWorkInput(
        assessment={"statement": "Need a measurement"},
        work_request={'objective': 'Measure', 'tasks': [{"selection": {"kind": "preset", "preset_ref": "coding@1", "reason": "Suitable configured executor"}, 'key': 'work', 'instruction': 'Measure', 'input_artifact_ids': [ref.id]}]},
    ))
    assert observation.ok
    assert observation.request_work["work_request"]["tasks"][0]["input_artifact_ids"] == [ref.id]
    assert "read_artifact_ids" not in observation.memory_updates
    assert not state.memory
