"""Uniform material nodes preserve artifact provenance and file discoverability."""

import json
from datetime import UTC, datetime

import pytest

from resagent2_components import (
    DatasetAvailability, ReadOnlyWorkspaceBoundary, RegisteredArtifactReader,
    WorkspaceBoundary, material_index_context,
)
from resagent2_components.artifacts import ArtifactReadError
from resagent2_contracts import (
    AgentPermissions, AgentRequest, ArtifactRef, ModuleError, TaskBudget,
    WorkAttemptResult, WorkResult, WorkTaskResult, WorkspaceAccess, WorkspaceGrant,
)
from resagent2_runtime import AgentState, ContextComposer
from resagent2_runtime.context import ContextBudgetExceeded


NOW = datetime.now(UTC)


def ref(name, **updates):
    values = dict(id=f"artifact_{name}", kind="web_page", producer="scientific", run_id="run_test",
                  session_id="session_test", uri=f"file:///private/{name}", sha256="a" * 64,
                  summary=f"Material {name}", media_type="text/markdown")
    return ArtifactRef(**(values | updates))


def request(refs=(), **updates):
    return AgentRequest(run_id="run_test", work_request_id="work_root", task_id="task_scientific",
                        attempt_number=1, agent="scientific", instruction="Inspect materials",
                        input_artifacts=list(refs), permissions=AgentPermissions(),
                        budget=TaskBudget(max_llm_calls=4, timeout_seconds=60), **updates)


def state(ids=()):
    return AgentState(session_id="session_test", agent_name="scientific", owner="scientific", run_id="run_test",
                      created_at=NOW, updated_at=NOW, authorized_artifact_ids=list(ids))


def grant(path, *, read=(".",), denied=()):
    return WorkspaceGrant(root=str(path), source="local", access=WorkspaceAccess(
        read_paths=list(read), denied_paths=list(denied)))


def descendants(node):
    yield node
    for child in node["children"]:
        yield from descendants(child)


def branch(tree, name):
    return next(node for node in tree["children"] if node["name"] == name)


def render(req=None, current=None, **kwargs):
    section = material_index_context(req or request(), current or state(), **kwargs)
    assert section.name == "material_index"
    return json.loads(section.content)


def test_all_authorized_artifacts_remain_complete_and_keep_standard_metadata():
    refs = [ref(index) for index in range(80)]
    req, current = request(refs[:2]), state([item.id for item in refs[1:]])
    before = req.model_dump_json(), current.model_dump_json()
    reader = RegisteredArtifactReader(refs, run_id=req.run_id)
    section = material_index_context(req, current, reader=reader, max_file_entries=0)
    tree = json.loads(section.content)
    artifacts = [node for node in descendants(tree) if node["type"] == "artifact"]
    assert tree["artifact_count"] == 80
    assert [node["location"]["artifact_id"] for node in artifacts] == [item.id for item in refs]
    assert all(node["expansion"] == {"state": "complete"} for node in artifacts)
    assert all(node["content_nature"] == "frozen" for node in artifacts)
    assert "/private/" not in section.content
    assert "a" * 64 not in section.content
    assert (req.model_dump_json(), current.model_dump_json()) == before
    with pytest.raises(ContextBudgetExceeded):
        ContextComposer().compose("System", [section], max_tokens=30)


def test_missing_accepted_ref_is_not_silently_omitted():
    with pytest.raises(ArtifactReadError, match="unavailable"):
        render(current=state(["artifact_missing"]))


def test_produced_artifacts_stay_under_actual_task_and_reuse_is_only_referenced():
    report = ref("report", kind="module_report", producer="orchestrator", session_id=None,
                 task_id="task_code", attempt_number=1, metadata={"source_type": "task_report"})
    output = ref("code", kind="code", producer="executor", session_id=None, task_id="task_code", attempt_number=1)
    paper = ref("paper", kind="literature_paper")
    work = WorkResult(run_id="run_test", session_id="session_test", work_request_id="work_one", objective="Implement",
        status="stable", workflow_revision=1, artifacts=[report, output, paper], tasks=[WorkTaskResult(
            task_id="task_code", workflow_agent_kind="executor", status="completed", instruction="Implement method",
            attempts=[WorkAttemptResult(number=1, status="completed", report_artifact_id=report.id, artifact_ids=[output.id])])])
    tree = render(request([paper, report, output], parent_session_id="session_test", work_results=[work]))
    work_node = next(node for node in descendants(tree) if node["type"] == "work_request")
    task = work_node["children"][0]
    assert task["metadata"]["instruction"] == "Implement method"
    assert task["status"] == "completed"
    assert [node["location"]["artifact_id"] for node in task["children"]] == [report.id, output.id]
    assert work_node["metadata"]["referenced_artifact_ids"] == [paper.id]
    artifact_nodes = [node for node in descendants(tree) if node["type"] == "artifact"]
    assert len(artifact_nodes) == 3


def test_navigation_preserves_error_facts_but_not_private_inputs_or_tool_bodies():
    active = {"work_request_id": "work_failed", "objective": "Measure", "status": "executing", "tasks": [{
        "task_id": "task_measure", "workflow_agent_kind": "experiment", "status": "failed", "instruction": "Measure",
        "upstream_inputs": [{"text": "OTHER_TASK_PRIVATE_INPUT"}],
        "error": {"code": "tool_failed", "message": "Execution failed", "details": {"content": "PRIVATE_TOOL_BODY"}},
        "attempts": [{"number": 1, "status": "failed", "reception_error": {
            "message": "Reception failed", "details": {"content": "PRIVATE_RECEPTION_BODY"}}}]}]}
    tree = render(request(task_context={"active_work_navigation": active}))
    text = json.dumps(tree)
    assert "Execution failed" in text and "Reception failed" in text
    assert "OTHER_TASK_PRIVATE_INPUT" not in text and "PRIVATE_TOOL_BODY" not in text
    assert "PRIVATE_RECEPTION_BODY" not in text


def test_small_existing_project_is_fully_discoverable_on_initial_projection(tmp_path):
    (tmp_path / "README.md").write_text("ORIGINAL_RULE_BODY")
    (tmp_path / "main.py").write_text("pass")
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "split.json").write_text('{"split": "sequential"}')
    for index in range(21):
        (tmp_path / f"module_{index}.py").write_text("pass")
    tree = render(request(read_workspaces=[grant(tmp_path)]))
    files = branch(tree, "file_spaces")
    nodes = list(descendants(files))
    assert {"README.md", "main.py", "split.json"} <= {node["name"] for node in nodes}
    assert all(node["expansion"]["state"] == "complete" for node in nodes)
    assert "ORIGINAL_RULE_BODY" not in json.dumps(tree)
    split = next(node for node in nodes if node["name"] == "split.json")
    assert split["location"]["path"] == str(tmp_path / "resources" / "split.json")


def test_all_roots_remain_visible_when_file_preview_budget_is_exhausted(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        root.mkdir()
        for name in ("a", "b", "c"):
            (root / name).touch()
    tree = render(request(read_workspaces=[grant(root) for root in roots]), max_file_entries=1)
    spaces = branch(tree, "file_spaces")["children"]
    assert len(spaces) == 2
    assert [node["location"]["path"] for node in spaces] == list(map(str, roots))
    assert spaces[0]["expansion"]["state"] == "partial"
    assert spaces[0]["expansion"]["listing"]["cursor"]
    assert spaces[1]["expansion"]["state"] == "not_expanded"
    assert spaces[1]["status"] == "available"


def test_character_limited_preview_never_publishes_cursor_after_hidden_entries(tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / name).touch()
    tree = render(request(read_workspaces=[grant(tmp_path)]), max_file_chars=400)
    root = branch(tree, "file_spaces")["children"][0]
    assert len(root["children"]) == 1
    assert root["expansion"]["state"] == "partial"
    assert root["expansion"]["listing"] == {"path": str(tmp_path)}
    assert "next_cursor" not in root["expansion"]


def test_cached_and_symlinked_directories_keep_explicit_unexpanded_entries(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "body.pyc").touch()
    (tmp_path / "self").symlink_to(tmp_path, target_is_directory=True)
    tree = render(request(read_workspaces=[grant(tmp_path)]))
    nodes = branch(tree, "file_spaces")["children"][0]["children"]
    assert [node["name"] for node in nodes] == ["__pycache__", "self"]
    assert all(node["expansion"]["state"] == "not_expanded" for node in nodes)
    assert all(node["expansion"]["listing"]["path"] for node in nodes)


def test_missing_empty_and_failed_file_roots_are_distinct(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = tmp_path / "missing"
    tree = render(request(read_workspaces=[grant(empty), grant(missing)]))
    first, second = branch(tree, "file_spaces")["children"]
    assert first["status"] == "available" and first["children"] == []
    assert first["expansion"]["state"] == "complete"
    assert second["status"] == "missing"
    scope = ReadOnlyWorkspaceBoundary([grant(empty)])
    monkeypatch.setattr(scope, "list_directory", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("PRIVATE")))
    tree = render(read_boundary=scope)
    node = branch(tree, "file_spaces")["children"][0]
    assert node["status"] == "read_error"
    assert "PRIVATE" not in json.dumps(tree)


def test_default_projection_respects_partial_scope_and_refreshes_real_file_observations(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "allowed.py").touch()
    (tmp_path / "hidden").touch()
    req = request(read_workspaces=[grant(tmp_path, read=("src",))])
    first = render(req)
    assert "hidden" not in json.dumps(first)
    (tmp_path / "src" / "new.py").touch()
    second = render(req)
    assert "new.py" not in json.dumps(first)
    assert "new.py" in json.dumps(second)


def test_supplied_boundary_is_authoritative_and_publishes_absolute_paths(tmp_path):
    (tmp_path / "file").touch()
    scope = WorkspaceBoundary(grant(tmp_path))
    tree = render(request(), read_boundary=scope)
    root = branch(tree, "file_spaces")["children"][0]
    assert root["location"] == {"path": str(tmp_path)}
    assert root["children"][0]["location"] == {"path": str(tmp_path / "file")}


def test_dataset_catalog_does_not_create_a_workspace_file_grant(tmp_path):
    availability = DatasetAvailability(available=[{"dataset_id": "cifar", "path": str(tmp_path), "access": "read_only"}],
                                       unavailable_ids=["missing_dataset"])
    tree = render(datasets=availability)
    assert branch(tree, "file_spaces")["children"] == []
    datasets = branch(tree, "datasets")["children"]
    assert [(node["name"], node["status"]) for node in datasets] == [("cifar", "available"), ("missing_dataset", "missing")]
    assert all(node["expansion"] == {"state": "complete"} for node in datasets)
    assert datasets[0]["location"] == {"dataset_id": "cifar"}
    assert datasets[0]["metadata"]["directory_listing_supported"] is False
    assert str(tmp_path) not in json.dumps(datasets)


def test_unsupported_directory_name_is_not_given_an_expandable_file_address(tmp_path):
    (tmp_path / "folder ").mkdir()
    (tmp_path / "folder " / "body.py").write_text("PRIVATE_BODY")
    tree = render(request(read_workspaces=[grant(tmp_path)]))
    root = branch(tree, "file_spaces")["children"][0]
    node = root["children"][0]
    assert node["name"] == "folder "
    assert node["status"] == "unsupported_filename"
    assert "path" not in node["location"]
    assert node["children"] == []
    assert node["expansion"] == {"state": "complete"}
    assert "PRIVATE_BODY" not in json.dumps(tree)


def test_import_metadata_cannot_remove_material_or_break_context_projection():
    imported = ref("import_metadata", kind="input", producer="orchestrator", session_id=None,
        metadata={"source_type": "import", "source_artifact_id": "descriptive caller value",
                  "agent_started": "false", "source_url": 17,
                  "final_url": {"custom": "value"}, "paper_id": "paper-exact"})
    before = imported.model_dump_json()

    tree = render(request([imported]))

    artifacts = [node for node in descendants(tree) if node["type"] == "artifact"]
    assert tree["artifact_count"] == len(artifacts) == 1
    assert artifacts[0]["location"] == {"artifact_id": imported.id}
    assert artifacts[0]["metadata"]["source_type"] == "import"
    assert artifacts[0]["metadata"]["paper_id"] == "paper-exact"
    assert not {"source_artifact_id", "agent_started", "source_url", "final_url"} & artifacts[0]["metadata"].keys()
    assert imported.model_dump_json() == before
