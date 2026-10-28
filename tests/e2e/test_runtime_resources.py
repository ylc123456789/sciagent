"""Resource refresh uses artifact inputs and persisted Sessions across processes."""

from e2e.native_fixtures import tool_turns

from resagent2_contracts import AgentPermissions

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from resagent2_components import DatasetCatalog, ResourceLayout, dataset_context, resolve_dataset_refs
from resagent2_contracts import (
    AgentOwner, AgentRequest, ArtifactCandidate, TaskBudget, RecordedAnswer,
    WorkspaceGrant, WorkspaceAccess, WorkspaceSourceKind, task_session_id,
)
from resagent2_coding import NativeCodingAgent
from resagent2_experiment import NativeExperimentAgent
from resagent2_orchestrator import ArtifactRegistry
from resagent2_runtime import JsonSessionStore, ScriptedLLMClient
from resagent2_scientific import ScientificAgent


def _ask(scientific=False):
    args = {"text": "Provision demo under the dataset root and register it, then confirm.",
            "requested_fields": ["dataset_ready"]}
    if scientific:
        args["assessment"] = {"statement": "Check dataset availability"}
    return {"tool": "ask_user", "arguments": args}


def _system_artifact(registry, kind, content, *, run_id="run_resources", session_id=None, scientific=False):
    candidate = ArtifactCandidate(
        kind=kind, path=kind + ".json", media_type="application/json", summary=kind,
        content=json.dumps(content),
    )
    return registry.register_system_artifact(
        candidate, run_id=run_id, source_type="dataset_catalog",
    )



def _probe(root, kind, phase):
    root = Path(root)
    layout = ResourceLayout(resource_root=root / "resources")
    datasets = DatasetCatalog(layout.dataset_root).references()
    scientific = kind == "scientific"
    task_id = "task_root" if scientific else "task_resources"
    session_id = task_session_id("run_resources", task_id, 1)
    sessions = JsonSessionStore(root / "sessions")
    before = sessions.load(session_id) if phase else None
    actions = [_ask(scientific)]
    if phase and not scientific:
        actions.insert(0, {"tool": "read_file", "arguments": {"path": "util.py"}})
    client = ScriptedLLMClient(tool_turns(actions))
    registry = ArtifactRegistry(root / "artifacts")
    refs = [_system_artifact(registry, "dataset_catalog", {"datasets": [ref.model_dump(mode="json") for ref in datasets]})]
    resume = []
    answers = []
    if phase:
        question = before.pending_control.question
        saved = root / "answers.json"
        answers = json.loads(saved.read_text()) if saved.exists() else []
        answer = RecordedAnswer(
            question_id=question.id, question_text=question.text,
            requested_fields=question.requested_fields, options=question.options,
            values={"dataset_ready": f"yes, reply_phase_{phase}"},
            answered_at=datetime.now(UTC), run_id="run_resources", session_id=session_id,
            task_id=task_id, attempt_number=1,
        )
        resume = [answer]
        answers.append(answer.model_dump(mode="json"))
        saved.write_text(json.dumps(answers))
    common = dict(
        run_id="run_resources",
        agent=AgentOwner.SCIENTIFIC if scientific else AgentOwner.EXECUTOR, input_artifacts=refs,
        work_request_id="work_direct", task_id=task_id, attempt_number=1,
        input_ids=[answer["question_id"] for answer in answers],
        resume_answers=resume, user_answers=answers,
        control_ack=before.pending_control.origin if phase else None,
        parent_session_id=session_id if phase else None,
        budget=TaskBudget(max_llm_calls=10, timeout_seconds=30),
    )
    if scientific:
        request = AgentRequest(**common, instruction='Check availability without downloading data', permissions=AgentPermissions(execute_commands=True))
        agent = ScientificAgent(client, store=sessions, resource_layout=layout)
    else:
        request = AgentRequest(**common, instruction='Inspect the provided resources', workspace=WorkspaceGrant(root=str(root / 'repo'), source=WorkspaceSourceKind.LOCAL, access=WorkspaceAccess(read_paths=['.'], write_paths=[])), workspace_id='ws_resources', output_dir=str(root / 'output'), permissions=AgentPermissions(execute_commands=True))
        cls = NativeExperimentAgent if kind == "experiment" else NativeCodingAgent
        agent = cls(client, store=sessions, resource_layout=layout)
    result = agent.invoke(request)
    assert result.status == "needs_user_input", result
    assert result.session.id == session_id
    text = client.contexts[0].text
    expected = '["demo"]' if phase == 2 else "[]"
    assert f'"available_dataset_ids": {expected}' in text
    missing = '["demo"]' if phase == 1 else "[]"
    assert f'"unavailable_dataset_ids": {missing}' in text
    shared = dataset_context(resolve_dataset_refs(layout.dataset_root, datasets))
    assert shared["availability_basis"] in text
    assert shared["missing_dataset_guidance"] in text
    for context in client.contexts:
        if phase:
            task = json.loads(context.task_context.split("\n", 1)[-1])
            expected_answers = [
                RecordedAnswer.model_validate(value).model_dump(mode="json", exclude_none=True)
                for value in answers
            ]
            assert [event["record"] for event in task["business_dialogue"] if event["type"] == "answer"] == expected_answers
            for previous in range(1, phase + 1):
                assert f"reply_phase_{previous}" in context.task_context
            assert context.included_sections == ["system", "task", "runtime_authority", "material_index"]
        tree = json.loads(context.material_index.split("\n", 1)[1])
        catalog = next(node for node in tree["children"] if node["name"] == "datasets")
        assert [node["location"]["dataset_id"] for node in catalog["children"]] == (["demo"] if phase else [])
        if phase:
            dataset, = catalog["children"]
            assert dataset["status"] == ("available" if phase == 2 else "missing")
            assert dataset["metadata"]["directory_listing_supported"] is False
    after = sessions.load(session_id)
    assert after.task_id == task_id
    assert after.attempt_number == 1
    if before:
        assert after.created_at == before.created_at
        assert len(after.events) > len(before.events)
        if "workspace_snapshot" in before.memory:
            assert after.memory["workspace_snapshot"] == before.memory["workspace_snapshot"]


@pytest.mark.parametrize("kind", ["scientific", "coding", "experiment"])
def test_native_agents_recheck_resources_across_processes(tmp_path, kind):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "util.py").write_text("VALUE = 1\n")
    for args in [["init", "-q"], ["add", "."],
                 ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "baseline"]]:
        subprocess.run(["git", *args], cwd=repo, check=True)
    dataset_root = tmp_path / "resources" / "datasets"
    dataset_root.mkdir(parents=True)
    for phase in range(3):
        if phase == 1:
            (dataset_root / "catalog.json").write_text('{"demo": "demo"}')
        if phase == 2:
            (dataset_root / "demo").mkdir()
        code = (f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r}); import runpy; " + f"ns = runpy.run_path({str(Path(__file__).resolve())!r}); "
                + f"ns['_probe']({str(tmp_path)!r}, {kind!r}, {phase})")
        proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=30)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(list((tmp_path / "sessions").glob("*.json"))) == 1


def test_scientific_invalid_resource_is_controlled_failure(tmp_path):
    layout = ResourceLayout(resource_root=tmp_path)
    layout.dataset_root.mkdir()
    (layout.dataset_root / "link").symlink_to(tmp_path / "outside", target_is_directory=True)
    ref = _system_artifact(
        ArtifactRegistry(tmp_path / "artifacts"), "dataset_catalog",
        {"datasets": [{"dataset_id": "x", "relative_path": "link"}]}, run_id="run_invalid",
    )
    client = ScriptedLLMClient([])
    result = ScientificAgent(client, resource_layout=layout).invoke(AgentRequest(work_request_id="work_direct", task_id="task_root", attempt_number=1, run_id='run_invalid', agent=AgentOwner.SCIENTIFIC, input_artifacts=[ref], instruction='No unsafe resource paths', budget=TaskBudget(max_llm_calls=5, timeout_seconds=30), permissions=AgentPermissions(execute_commands=True)))
    assert result.status == "failed"
    assert result.error.code == "invalid_input"
    assert not client.contexts
