"""Tool addresses share authorization without widening storage namespaces."""

import json
from datetime import UTC, datetime

import pytest

from sciagent_capabilities import (
    CreateFileTool, DeletePathTool, ListFilesTool, ReadFileTool, ReplaceTextTool, SearchTextTool,
)
from sciagent_capabilities.permissions import OperationPermissionPolicy
from sciagent_components import ReadOnlyWorkspaceBoundary, WorkspaceBoundary, WorkspacePermissionError
from sciagent_components import material_index_context
from sciagent_components.read_scope import request_read_boundary
from sciagent_components.workspace import (
    DeleteChangedError, FILESYSTEM_PATH_DESCRIPTION, MAIN_WORKSPACE_PATH_DESCRIPTION,
    _normalize_relative,
)
from sciagent_contracts import (
    ActionSnapshot, AgentPermissions, AgentRequest, RecordedAnswer, TaskBudget, WorkspaceAccess, WorkspaceGrant,
)
from sciagent_runtime import AgentAction, AgentState


def grant(root, *, read=(".",), write=(".",), denied=()):
    return WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(
        read_paths=list(read), write_paths=list(write), denied_paths=list(denied),
    ))


def state():
    now = datetime.now(UTC)
    return AgentState(session_id="session_addresses", agent_name="test", owner="executor",
                      run_id="run_addresses", task_id="task_addresses", attempt_number=1,
                      created_at=now, updated_at=now)


@pytest.mark.parametrize("absolute", [False, True])
def test_filesystem_workflow_accepts_the_same_absolute_and_relative_addresses(tmp_path, absolute):
    scope = WorkspaceBoundary(grant(tmp_path))
    address = str(tmp_path / "nested/source.txt") if absolute else "nested/source.txt"
    create = CreateFileTool(scope)
    create.execute(state(), create.input_model(path=address, content="original needle\n"))
    with pytest.raises(FileExistsError):
        create.execute(state(), create.input_model(path=str(tmp_path / "nested/source.txt"), content="other"))
    read = ReadFileTool(scope)
    assert read.execute(state(), read.input_model(path=address)).value["content"] == "original needle\n"
    list_tool = ListFilesTool(scope)
    directory = str(tmp_path / "nested") if absolute else "nested"
    page = list_tool.execute(state(), list_tool.input_model(path=directory)).value
    assert page["location"]["path"] == str(tmp_path / "nested")
    copied = page["children"][0]["location"]["path"]
    assert copied == str(tmp_path / "nested/source.txt")
    replace = ReplaceTextTool(scope)
    replace.execute(state(), replace.input_model(path=copied, old_text="original", new_text="updated"))
    search = SearchTextTool(scope)
    for search_path in (address, directory):
        result = search.execute(state(), search.input_model(path=search_path, query="needle"))
        assert result.value["matches"] == [{"path": copied, "line": 1, "text": "updated needle"}]
    delete = DeletePathTool(scope)
    delete.execute(state(), delete.input_model(path=copied))
    assert not (tmp_path / "nested/source.txt").exists()


def test_relative_shorthand_stays_at_default_root_when_process_cwd_changes(tmp_path, monkeypatch):
    root, other = tmp_path / "root", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    (root / "file").write_text("trusted")
    (other / "file").write_text("different")
    scope = WorkspaceBoundary(grant(root))
    reads = ReadOnlyWorkspaceBoundary([grant(root), grant(other)], relative_root=root)
    monkeypatch.chdir(other)
    assert scope.resolve_read_file("file").read_text() == "trusted"
    assert reads.resolve_read_file("file").read_text() == "trusted"
    CreateFileTool(scope).execute(state(), CreateFileTool.input_model(path="new", content="created"))
    assert (root / "new").read_text() == "created"
    assert not (other / "new").exists()


def test_extra_read_root_does_not_enable_search_or_mutation(tmp_path):
    main, readonly = tmp_path / "main", tmp_path / "readonly"
    main.mkdir()
    readonly.mkdir()
    source = readonly / "source"
    source.write_text("needle")
    scope = WorkspaceBoundary(grant(main))
    reads = ReadOnlyWorkspaceBoundary([grant(main), grant(readonly, write=())], relative_root=main)
    assert reads.resolve_read_file(str(source)) == source
    assert reads.list_directory(str(readonly))["children"][0]["location"]["path"] == str(source)
    with pytest.raises(FileNotFoundError):
        reads.resolve_read_file("source")  # Never search other roots for shorthand.
    operations = [
        (SearchTextTool(scope), {"path": str(source), "query": "needle"}),
        (CreateFileTool(scope), {"path": str(readonly / "new"), "content": "new"}),
        (ReplaceTextTool(scope), {"path": str(source), "old_text": "needle", "new_text": "changed"}),
        (DeletePathTool(scope), {"path": str(source)}),
    ]
    for tool, values in operations:
        with pytest.raises(WorkspacePermissionError, match="outside"):
            tool.execute(state(), tool.input_model(**values))
    assert source.read_text() == "needle"
    assert not (readonly / "new").exists()
    readonly_scope = WorkspaceBoundary(grant(readonly, write=()))
    with pytest.raises(WorkspacePermissionError, match="write scope"):
        readonly_scope.resolve_write_file(str(source))
    with pytest.raises(WorkspacePermissionError, match="write scope"):
        readonly_scope.prepare_delete(str(source))


def test_absolute_addresses_recheck_lexical_denials_before_resolving_aliases(tmp_path):
    (tmp_path / "public").mkdir()
    source = tmp_path / "public/source"
    source.write_text("keep")
    denied = tmp_path / "denied"
    denied.symlink_to(tmp_path / "public", target_is_directory=True)
    scope = WorkspaceBoundary(grant(tmp_path, denied=("denied",)))
    for operation in (scope.resolve_read_file, scope.resolve_write_file, scope.prepare_delete):
        with pytest.raises(WorkspacePermissionError, match="denied_paths"):
            operation(str(denied / "source"))
    with pytest.raises(WorkspacePermissionError, match="denied_paths"):
        scope.list_directory(str(denied))
    assert source.read_text() == "keep"


def test_absolute_addresses_recheck_resolved_denials_and_private_or_git_targets(tmp_path):
    for name in ("denied", ".sciagent", ".git"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "file").write_text(name)
        (tmp_path / f"alias_{name}").symlink_to(tmp_path / name, target_is_directory=True)
    scope = WorkspaceBoundary(grant(tmp_path, denied=("denied",)))
    for name in ("denied", ".sciagent", ".git"):
        for path in (tmp_path / name / "file", tmp_path / f"alias_{name}/file"):
            if name == ".git":
                assert scope.resolve_read_file(str(path)).read_text() == ".git"
            else:
                with pytest.raises(WorkspacePermissionError):
                    scope.resolve_read_file(str(path))
            with pytest.raises(WorkspacePermissionError):
                scope.resolve_write_file(str(path))
            with pytest.raises(WorkspacePermissionError):
                scope.prepare_delete(str(path))


def test_root_escape_and_external_alias_into_root_are_rejected_lexically(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "file").write_text("keep")
    alias = tmp_path / "external_alias"
    alias.symlink_to(root, target_is_directory=True)
    scope = WorkspaceBoundary(grant(root))
    for path in (str(alias / "file"), str(tmp_path / "root-sibling/file"),
                 str(root / "../root/file"), "../root/file", "C:/file", "C:\\file", "\\\\server\\share\\file"):
        for operation in (scope.resolve_read_file, scope.resolve_write_file, scope.prepare_delete):
            with pytest.raises(WorkspacePermissionError):
                operation(path)
    for path in (str(root), "."):
        with pytest.raises(WorkspacePermissionError):
            scope.prepare_delete(path, recursive=True)


def test_absolute_delete_uses_relative_snapshot_and_detects_target_drift(tmp_path):
    target = tmp_path / "build"
    target.mkdir()
    file = target / "old"
    file.write_text("old")
    tool = DeletePathTool(WorkspaceBoundary(grant(tmp_path)))
    args = tool.input_model(path=str(target), recursive=True)
    snapshot = tool.prepare(args)
    assert snapshot == tool.prepare(tool.input_model(path="build", recursive=True))
    assert snapshot["path"] == "build"
    assert [item["path"] for item in snapshot["entries"]] == ["build", "build/old"]
    with pytest.raises(WorkspacePermissionError, match="snapshot"):
        tool.execute_prepared(state(), tool.input_model(path=str(tmp_path / "other"), recursive=True), snapshot)
    file.write_text("changed")
    with pytest.raises(DeleteChangedError):
        tool.execute_prepared(state(), args, snapshot)
    assert file.read_text() == "changed"
    result = tool.execute_prepared(state(), args, tool.prepare(args))
    assert result.ok and result.value["deleted_paths"] == ["build/old", "build"]


def test_absolute_delete_unlinks_final_symlink_without_following_outside_target(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("keep")
    alias = root / "alias"
    alias.symlink_to(outside)
    tool = DeletePathTool(WorkspaceBoundary(grant(root)))
    args = tool.input_model(path=str(alias))
    assert tool.prepare(args)["entries"][0]["kind"] == "symlink"
    assert tool.execute(state(), args).ok
    assert outside.read_text() == "keep"
    assert not alias.is_symlink()


def test_grant_and_system_namespaces_stay_relative(tmp_path):
    scope = WorkspaceBoundary(grant(tmp_path))
    with pytest.raises(WorkspacePermissionError, match="relative"):
        _normalize_relative(str(tmp_path / "file"))
    with pytest.raises(WorkspacePermissionError, match="relative"):
        scope.resolve_system_write(str(tmp_path / ".sciagent/output"))
    assert scope.resolve_system_write(".sciagent/output") == tmp_path / ".sciagent/output"


@pytest.mark.parametrize("tool", [ReadFileTool, ListFilesTool, SearchTextTool, CreateFileTool, ReplaceTextTool, DeletePathTool])
def test_path_schema_and_guidance_use_shared_contract(tool):
    expected = FILESYSTEM_PATH_DESCRIPTION if tool in (ReadFileTool, ListFilesTool) else MAIN_WORKSPACE_PATH_DESCRIPTION
    assert tool.input_model.model_json_schema()["properties"]["path"]["description"] == expected
    assert tool.model_guidance.startswith(expected)


def test_equivalent_delete_address_cannot_reuse_approval_for_another_raw_invocation(tmp_path):
    target = tmp_path / "build"
    target.mkdir()
    (target / "file").write_text("keep")
    scope = WorkspaceBoundary(grant(tmp_path))
    tool = DeletePathTool(scope)
    current = state()
    request = AgentRequest(
        run_id=current.run_id, work_request_id="work_addresses", agent="executor",
        task_id=current.task_id, attempt_number=current.attempt_number, permissions=AgentPermissions(),
        instruction="Remove obsolete build files", workspace=scope.grant,
        budget=TaskBudget(max_llm_calls=2, timeout_seconds=30),
    )
    relative = AgentAction(tool="delete_path", arguments={"path": "build", "recursive": True})
    absolute = AgentAction(tool="delete_path", arguments={"path": str(target), "recursive": True})
    policy = OperationPermissionPolicy((tool,), boundary=scope, request=request)
    prepared = policy.check(relative, current, request)
    current.pending_action = ActionSnapshot(
        action_id="action_addresses", tool="delete_path", arguments=relative.arguments,
        context=prepared.context, run_id=current.run_id, session_id=current.session_id,
        task_id=current.task_id, attempt_number=current.attempt_number,
    )
    answer = RecordedAnswer(
        question_id="question_addresses", question_text="Delete build?", requested_fields=["approve"],
        values={"approve": "yes"}, answered_at=datetime.now(UTC),
        run_id=current.run_id, session_id=current.session_id, action=current.pending_action,
        task_id=current.task_id, attempt_number=current.attempt_number,
    )
    request = request.model_copy(update={"resume_answers": [answer]})
    policy = OperationPermissionPolicy((tool,), boundary=scope, request=request)
    assert policy.check(relative, current, request).outcome == "allow"
    equivalent = policy.check(absolute, current, request)
    assert equivalent.context == prepared.context
    assert equivalent.outcome == "ask"
    assert (target / "file").read_text() == "keep"


def test_workspace_root_preserves_symlink_then_parent_resolution_semantics(tmp_path):
    base = tmp_path / "base"
    elsewhere = tmp_path / "elsewhere"
    base.mkdir()
    (elsewhere / "child").mkdir(parents=True)
    intended = elsewhere / "workspace"
    intended.mkdir()
    wrong = base / "workspace"
    wrong.mkdir()
    (intended / "source").write_text("authorized source")
    (wrong / "source").write_text("wrong source")
    (base / "alias").symlink_to(elsewhere / "child", target_is_directory=True)
    configured_root = base / "alias/../workspace"
    scope = WorkspaceBoundary(grant(configured_root))
    assert scope.root == intended
    assert scope.resolve_read_file("source").read_text() == "authorized source"
    assert scope.resolve_read_file(str(intended / "source")) == intended / "source"
    assert scope.list_directory()["location"]["path"] == str(intended)
    assert scope.list_directory()["children"][0]["location"]["path"] == str(intended / "source")
    CreateFileTool(scope).execute(state(), CreateFileTool.input_model(path="new", content="new"))
    assert (intended / "new").read_text() == "new"
    assert not (wrong / "new").exists()
    with pytest.raises(WorkspacePermissionError):
        scope.resolve_read_file(str(wrong / "source"))


def executor_request(workspace, *, read_workspaces=()):
    return AgentRequest(
        run_id="run_addresses", work_request_id="work_addresses", agent="executor",
        task_id="task_addresses", attempt_number=1, permissions=AgentPermissions(),
        instruction="Use authorized filesystem addresses", workspace=workspace,
        read_workspaces=list(read_workspaces),
        budget=TaskBudget(max_llm_calls=2, timeout_seconds=30),
    )


@pytest.fixture
def symlink_workspace(tmp_path):
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "workspace_alias"
    alias.symlink_to(physical, target_is_directory=True)
    return physical, alias


@pytest.mark.parametrize("address_kind", ["relative", "alias", "physical"])
def test_executor_symlink_root_addresses_round_trip_across_tools(symlink_workspace, address_kind):
    physical, alias = symlink_workspace
    request = executor_request(grant(alias))
    scope, reads = WorkspaceBoundary(request.workspace), request_read_boundary(request)
    root = {"relative": "", "alias": str(alias), "physical": str(physical)}[address_kind]
    address = f"{root}/nested/source.txt" if root else "nested/source.txt"
    directory = f"{root}/nested" if root else "nested"
    create = CreateFileTool(scope)
    assert create.execute(state(), create.input_model(path=address, content="original needle\n")).ok
    read = ReadFileTool(reads)
    for path in (address, str(alias / "nested/source.txt"), str(physical / "nested/source.txt")):
        assert read.execute(state(), read.input_model(path=path)).value["content"] == "original needle\n"
    listing = ListFilesTool(reads)
    page = listing.execute(state(), listing.input_model(path=directory)).value
    assert page["location"]["path"] == str(physical / "nested")
    copied = page["children"][0]["location"]["path"]
    assert copied == str(physical / "nested/source.txt")
    replace = ReplaceTextTool(scope)
    assert replace.execute(state(), replace.input_model(
        path=address, old_text="original", new_text="updated",
    )).ok
    search = SearchTextTool(scope)
    for path in (address, directory):
        assert search.execute(state(), search.input_model(path=path, query="needle")).value["matches"] == [
            {"path": copied, "line": 1, "text": "updated needle"},
        ]
    # The index follows the same read-boundary construction as Executor. Its
    # default construction must publish the same usable physical addresses.
    for boundary in (None, reads):
        tree = json.loads(material_index_context(request, state(), read_boundary=boundary).content)
        files = next(node for node in tree["children"] if node["name"] == "file_spaces")
        root_node = files["children"][0]
        assert root_node["status"] == "available"
        assert root_node["location"]["path"] == str(physical)
        assert root_node["children"][0]["location"]["path"] == str(physical / "nested")
        indexed = root_node["children"][0]["children"][0]["location"]["path"]
        assert indexed == copied
        assert read.execute(state(), read.input_model(path=indexed)).value["content"] == "updated needle\n"
    # A returned physical address remains writable; configured aliases also
    # remain accepted when creating descendants whose parents do not exist.
    replace.execute(state(), replace.input_model(path=copied, old_text="updated", new_text="final"))
    assert read.execute(state(), read.input_model(path=address)).value["content"] == "final needle\n"
    delete = DeletePathTool(scope)
    assert delete.execute(state(), delete.input_model(path=address)).ok
    assert not (physical / "nested/source.txt").exists()
    create.execute(state(), create.input_model(path=str(alias / "new/child"), content="new"))
    assert delete.execute(state(), delete.input_model(path=str(physical / "new/child"))).ok


@pytest.mark.parametrize("address_kind", ["alias", "physical"])
def test_symlink_root_keeps_lexical_and_resolved_exclusions(symlink_workspace, address_kind):
    physical, alias = symlink_workspace
    for name in ("public", "denied", ".sciagent", ".git"):
        (physical / name).mkdir()
        (physical / name / "source").write_text("keep")
    (physical / "denied_alias").symlink_to(physical / "public", target_is_directory=True)
    (physical / "public_alias").symlink_to(physical / "denied", target_is_directory=True)
    request = executor_request(grant(alias, denied=("denied", "denied_alias")))
    scope, reads = WorkspaceBoundary(request.workspace), request_read_boundary(request)
    root = alias if address_kind == "alias" else physical
    for suffix in ("denied", "denied_alias", "public_alias", ".sciagent", ".git"):
        path = str(root / suffix / "source")
        if suffix == ".git":
            assert reads.resolve_read_file(path).read_text() == "keep"
        else:
            with pytest.raises(WorkspacePermissionError):
                reads.resolve_read_file(path)
            with pytest.raises(WorkspacePermissionError):
                reads.list_directory(str(root / suffix))
        for operation in (scope.resolve_write_file, scope.prepare_delete):
            with pytest.raises(WorkspacePermissionError):
                operation(path)
    page = reads.list_directory(str(root))
    assert {node["name"] for node in page["children"]} == {".git", "public"}
    assert (physical / "public/source").read_text() == "keep"


@pytest.mark.parametrize("address_kind", ["alias", "physical"])
def test_symlink_root_delete_unlinks_final_symlink(symlink_workspace, tmp_path, address_kind):
    physical, alias = symlink_workspace
    outside = tmp_path / "outside"
    outside.write_text("keep")
    link = physical / "final_link"
    link.symlink_to(outside)
    root = alias if address_kind == "alias" else physical
    delete = DeletePathTool(WorkspaceBoundary(grant(alias)))
    args = delete.input_model(path=str(root / "final_link"))
    assert delete.prepare(args)["entries"][0]["kind"] == "symlink"
    assert delete.execute(state(), args).ok
    assert outside.read_text() == "keep"
    assert not link.is_symlink()


@pytest.mark.parametrize("narrow_alias", [False, True])
def test_read_grants_apply_most_specific_physical_scope_through_root_aliases(
    symlink_workspace, tmp_path, narrow_alias,
):
    physical, alias = symlink_workspace
    nested = physical / "nested"
    nested.mkdir()
    (nested / "visible").write_text("allowed")
    (nested / "secret").write_text("denied")
    nested_alias = tmp_path / "nested_alias"
    nested_alias.symlink_to(nested, target_is_directory=True)
    (physical / "jump").symlink_to(nested, target_is_directory=True)
    narrow = nested_alias if narrow_alias else nested
    request = executor_request(grant(alias), read_workspaces=[grant(narrow, write=(), denied=("secret",))])
    reads = request_read_boundary(request)
    roots = [physical / "nested", alias / "nested", physical / "jump", alias / "jump"]
    if narrow_alias:
        roots.append(nested_alias)
    for root in roots:
        assert reads.resolve_read_file(str(root / "visible")) == nested / "visible"
        with pytest.raises(WorkspacePermissionError, match="denied_paths"):
            reads.resolve_read_file(str(root / "secret"))
        assert [node["name"] for node in reads.list_directory(str(root))["children"]] == ["visible"]
    assert reads.resolve_read_file("nested/visible") == nested / "visible"
    outside_alias = tmp_path / "undeclared_alias"
    outside_alias.symlink_to(physical, target_is_directory=True)
    for operation in (reads.resolve_read_file, WorkspaceBoundary(request.workspace).resolve_write_file,
                      WorkspaceBoundary(request.workspace).prepare_delete):
        with pytest.raises(WorkspacePermissionError, match="outside"):
            operation(str(outside_alias / "nested/visible"))
    with pytest.raises(WorkspacePermissionError, match="outside"):
        reads.list_directory(str(outside_alias))


def test_request_read_root_resolves_symlink_then_parent_before_matching(tmp_path):
    base, elsewhere = tmp_path / "base", tmp_path / "elsewhere"
    base.mkdir()
    (elsewhere / "child").mkdir(parents=True)
    physical = elsewhere / "workspace"
    physical.mkdir()
    wrong = base / "workspace"
    wrong.mkdir()
    (physical / "source").write_text("authorized")
    (wrong / "source").write_text("wrong")
    (base / "alias").symlink_to(elsewhere / "child", target_is_directory=True)
    request = executor_request(grant(base / "alias/../workspace"))
    reads = request_read_boundary(request)
    for path in ("source", str(physical / "source")):
        assert reads.resolve_read_file(path) == physical / "source"
    for path in (".", str(physical)):
        page = reads.list_directory(path)
        assert page["location"]["path"] == str(physical)
        assert page["children"][0]["location"]["path"] == str(physical / "source")
    with pytest.raises(WorkspacePermissionError):
        reads.resolve_read_file(str(wrong / "source"))


def test_same_physical_root_alias_grants_intersect_for_every_address(symlink_workspace, tmp_path):
    physical, alias = symlink_workspace
    restrictive_alias = tmp_path / "restrictive_alias"
    restrictive_alias.symlink_to(physical, target_is_directory=True)
    (physical / "visible").write_text("allowed")
    (physical / "secret").write_text("denied")
    request = executor_request(grant(alias), read_workspaces=[
        grant(restrictive_alias, write=(), denied=("secret",)),
    ])
    reads = request_read_boundary(request)
    for root in (physical, alias, restrictive_alias):
        assert reads.resolve_read_file(str(root / "visible")) == physical / "visible"
        with pytest.raises(WorkspacePermissionError, match="denied_paths"):
            reads.resolve_read_file(str(root / "secret"))
        page = reads.list_directory(str(root))
        assert page["location"]["path"] == str(physical)
        assert [node["name"] for node in page["children"]] == ["visible"]


def test_retargeted_declared_root_alias_is_rejected_while_canonical_root_stays_pinned(
    symlink_workspace, tmp_path,
):
    physical, alias = symlink_workspace
    other = tmp_path / "other"
    other.mkdir()
    (physical / "source").write_text("original")
    (other / "source").write_text("different")
    request = executor_request(grant(alias))
    scope, reads = WorkspaceBoundary(request.workspace), request_read_boundary(request)
    alias.unlink()
    alias.symlink_to(other, target_is_directory=True)
    # Constructed boundaries capture physical authority and its default root.
    for path in ("source", str(physical / "source")):
        assert reads.resolve_read_file(path).read_text() == "original"
        assert scope.resolve_read_file(path).read_text() == "original"
    create = CreateFileTool(scope)
    for path in ("relative_new", str(physical / "canonical_new")):
        assert create.execute(state(), create.input_model(path=path, content="created")).ok
    assert (physical / "relative_new").read_text() == "created"
    assert (physical / "canonical_new").read_text() == "created"
    assert sorted(path.name for path in other.iterdir()) == ["source"]
    for operation in (reads.resolve_read_file, scope.resolve_read_file,
                      scope.resolve_write_file, scope.prepare_delete):
        with pytest.raises(WorkspacePermissionError, match="changed"):
            operation(str(alias / "source"))
    for boundary in (reads, scope):
        with pytest.raises(WorkspacePermissionError, match="changed"):
            boundary.list_directory(str(alias))
        assert boundary.list_directory(".")["location"]["path"] == str(physical)
    assert (physical / "source").read_text() == "original"
    assert (other / "source").read_text() == "different"
