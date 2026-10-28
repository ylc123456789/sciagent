"""Permission-scoped, nonrecursive directory observations and continuation."""

import base64
import json
import os

import pytest

from sciagent_components import WorkspaceBoundary, ReadOnlyWorkspaceBoundary, WorkspacePermissionError
from sciagent_components.workspace import DirectoryChangedError
from sciagent_contracts import WorkspaceAccess, WorkspaceGrant


def grant(root, *, read=(".",), denied=()):
    return WorkspaceGrant(root=str(root), source="local",
                          access=WorkspaceAccess(read_paths=list(read), denied_paths=list(denied)))


def boundary(root, **kwargs):
    return WorkspaceBoundary(grant(root, **kwargs))


def names(page):
    return [child["name"] for child in page["children"]]


def test_single_level_nodes_keep_git_and_cache_but_skip_runtime_or_denied_entries(tmp_path):
    (tmp_path / "z.py").write_text("pass")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "nested.py").write_text("pass")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".sciagent").mkdir()
    (tmp_path / "private.txt").write_text("secret")
    page = boundary(tmp_path, denied=("private.txt",)).list_directory()
    assert names(page) == [".git", "__pycache__", "src", "z.py"]
    assert page["expansion"] == {"state": "complete"}
    directory, source = page["children"][2:]
    assert directory["location"] == {"path": str(tmp_path / "src")}
    assert directory["expansion"] == {"state": "not_expanded"}
    assert source["content_nature"] == "live_file"
    assert source["children"] == []
    assert source["expansion"] == {"state": "complete"}
    assert names(boundary(tmp_path).list_directory("src")) == ["nested.py"]


def test_empty_directory_is_complete_not_missing(tmp_path):
    page = boundary(tmp_path).list_directory()
    assert page["status"] == "available"
    assert page["children"] == []
    assert page["expansion"]["state"] == "complete"
    with pytest.raises(FileNotFoundError):
        boundary(tmp_path).list_directory("missing")


def test_more_than_two_thousand_entries_can_be_paged_without_overlap(tmp_path):
    expected = [f"entry_{index:04d}" for index in range(2005)]
    for name in reversed(expected):
        (tmp_path / name).touch()
    scope = boundary(tmp_path)
    cursor = None
    actual = []
    observations = set()
    while True:
        page = scope.list_directory(max_entries=1000, cursor=cursor)
        actual.extend(names(page))
        observations.add(page["observation"]["directory_fingerprint"])
        assert len(page["children"]) <= 1000
        cursor = page["expansion"].get("next_cursor")
        if cursor is None:
            assert page["expansion"]["state"] == "complete"
            break
        assert page["expansion"]["state"] == "partial"
    assert actual == expected
    assert len(observations) == 1


def test_unauthorized_entries_do_not_fill_page_or_hide_later_authorized_files(tmp_path):
    for name in ("a", "b", "c", "d", "e", "f"):
        (tmp_path / name).touch()
    scope = boundary(tmp_path, read=("e", "f"))
    first = scope.list_directory(max_entries=1)
    assert names(first) == ["e"]
    second = scope.list_directory(max_entries=1, cursor=first["expansion"]["next_cursor"])
    assert names(second) == ["f"]
    assert second["expansion"] == {"state": "complete"}


def test_exact_file_read_scope_discovers_only_its_ancestor_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "allowed.py").touch()
    (tmp_path / "src" / "hidden.py").touch()
    (tmp_path / "hidden.txt").touch()
    scope = boundary(tmp_path, read=("src/allowed.py",))
    assert names(scope.list_directory()) == ["src"]
    assert names(scope.list_directory("src")) == ["allowed.py"]
    with pytest.raises(WorkspacePermissionError):
        boundary(tmp_path, read=()).list_directory()


def test_cursor_rejects_other_path_new_authority_and_directory_changes(tmp_path):
    for folder in ("one", "two"):
        (tmp_path / folder).mkdir()
        for name in ("a", "b"):
            (tmp_path / folder / name).touch()
    scope = boundary(tmp_path)
    first = scope.list_directory("one", max_entries=1)
    cursor = first["expansion"]["next_cursor"]
    with pytest.raises(DirectoryChangedError, match="restart"):
        scope.list_directory("two", cursor=cursor)
    with pytest.raises(DirectoryChangedError, match="scope changed"):
        boundary(tmp_path, denied=("elsewhere",)).list_directory("one", cursor=cursor)
    (tmp_path / "one" / "a").write_text("Existing content can change")
    assert names(scope.list_directory("one", cursor=cursor)) == ["b"]
    (tmp_path / "one" / "new").touch()
    with pytest.raises(DirectoryChangedError, match="restart"):
        scope.list_directory("one", cursor=cursor)


@pytest.mark.parametrize("cursor", ["invalid", "", "====", "e30", "[]", "x" * 9000])
def test_invalid_cursor_is_a_clear_input_error(tmp_path, cursor):
    with pytest.raises(ValueError, match="Invalid directory cursor"):
        boundary(tmp_path).list_directory(cursor=cursor)


def test_change_during_observation_rejects_apparently_complete_page(tmp_path, monkeypatch):
    from sciagent_components import workspace

    (tmp_path / "a").touch()
    original = workspace._directory_names

    def names_then_mutate(*args):
        found = original(*args)
        (tmp_path / "added_during_scan").touch()
        return found

    monkeypatch.setattr(workspace, "_directory_names", names_then_mutate)
    with pytest.raises(DirectoryChangedError, match="during listing"):
        boundary(tmp_path).list_directory()


def test_no_descendant_scan_or_content_read_is_needed(tmp_path, monkeypatch):
    from sciagent_components import workspace

    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "secret").write_text("BODY_MUST_NOT_BE_READ")
    original = workspace.os.scandir

    def scan_only_root(path):
        assert os.fspath(path) == str(tmp_path)
        return original(path)

    monkeypatch.setattr(workspace.os, "scandir", scan_only_root)
    page = boundary(tmp_path).list_directory(max_entries=1)
    assert names(page) == ["nested"]
    assert "BODY_MUST_NOT_BE_READ" not in json.dumps(page)


def test_multi_root_uses_absolute_paths_and_keeps_missing_root_lazy(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    (root / "file").touch()
    missing = tmp_path / "not_created"
    scope = ReadOnlyWorkspaceBoundary([grant(root), grant(missing)])
    page = scope.list_directory(str(root))
    assert page["children"][0]["location"] == {"path": str(root / "file")}
    with pytest.raises(WorkspacePermissionError, match="absolute"):
        scope.list_directory(".")
    with pytest.raises(FileNotFoundError):
        scope.list_directory(str(missing))


def test_narrow_root_and_same_root_restrictions_cannot_fall_back_to_wider_grant(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "private").touch()
    (nested / "public").touch()
    scope = ReadOnlyWorkspaceBoundary([grant(tmp_path), grant(nested, denied=("private",))])
    assert names(scope.list_directory(str(nested))) == ["public"]
    same = ReadOnlyWorkspaceBoundary([grant(tmp_path), grant(tmp_path, denied=("nested",))])
    assert names(same.list_directory(str(tmp_path))) == []
    with pytest.raises(WorkspacePermissionError):
        same.list_directory(str(nested))


def test_symlink_targets_are_rechecked_against_narrower_grants(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret").touch()
    (tmp_path / "alias").symlink_to(private, target_is_directory=True)
    scope = ReadOnlyWorkspaceBoundary([grant(tmp_path), grant(private, read=())])
    assert names(scope.list_directory(str(tmp_path))) == []
    with pytest.raises(WorkspacePermissionError):
        scope.list_directory(str(tmp_path / "alias"))


def test_internal_symlinks_are_visible_but_escaping_ones_are_not(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "inside").mkdir()
    (root / "alias").symlink_to(root / "inside", target_is_directory=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    page = boundary(root).list_directory()
    assert names(page) == ["alias", "inside"]
    assert page["children"][0]["is_symlink"] is True


def test_metadata_errors_are_visible_without_exception_path_leaks(tmp_path, monkeypatch):
    (tmp_path / "unreadable").touch()
    scope = boundary(tmp_path)
    resolve = scope._resolve_listing_entry

    def failing_child(path):
        if path == str(tmp_path / "unreadable"):
            raise PermissionError("PRIVATE_EXCEPTION_PATH")
        return resolve(path)

    monkeypatch.setattr(scope, "_resolve_listing_entry", failing_child)
    page = scope.list_directory()
    assert page["children"][0]["status"] == "read_error"
    assert page["children"][0]["error"] == "PermissionError"
    assert "PRIVATE_EXCEPTION_PATH" not in json.dumps(page)


@pytest.mark.skipif(os.name != "posix", reason="Unix byte filenames")
def test_non_utf8_filename_is_reported_without_a_fictitious_text_path(tmp_path):
    raw = os.fsencode(tmp_path) + b"/bad_\xff"
    descriptor = os.open(raw, os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(descriptor)
    page = boundary(tmp_path).list_directory()
    node = page["children"][0]
    assert node["status"] == "unsupported_filename"
    assert "path" not in node["location"]
    assert base64.b64decode(node["location"]["path_bytes_base64"]) == raw
    json.dumps(page, ensure_ascii=False).encode("utf-8")


def test_optional_relative_root_is_only_addressing_and_keeps_absolute_locations(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.py").write_text("pass")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "command.stdout").write_text("result")
    scope = ReadOnlyWorkspaceBoundary([grant(workspace), grant(outputs)], relative_root=workspace)
    assert scope.resolve_read_file("source.py") == workspace / "source.py"
    assert scope.resolve_read_file(str(outputs / "command.stdout")) == outputs / "command.stdout"
    page = scope.list_directory(".")
    assert page["location"]["path"] == str(workspace)
    assert page["children"][0]["location"]["path"] == str(workspace / "source.py")
    empty = ReadOnlyWorkspaceBoundary([], relative_root=workspace)
    with pytest.raises(WorkspacePermissionError):
        empty.resolve_read_file("source.py")
    with pytest.raises(WorkspacePermissionError):
        empty.list_directory(".")


@pytest.mark.parametrize("path", ["../outside", "nested/../source.py", "C:/source.py", "C:\\source.py", "\\\\server\\share\\file"])
def test_relative_root_cannot_enable_traversal_or_windows_drives(tmp_path, path):
    scope = ReadOnlyWorkspaceBoundary([grant(tmp_path)], relative_root=tmp_path)
    with pytest.raises(WorkspacePermissionError):
        scope.resolve_read_file(path)
    with pytest.raises(WorkspacePermissionError):
        scope.list_directory(path)


def test_relative_root_cannot_override_nested_or_same_root_denials(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "secret").touch()
    scope = ReadOnlyWorkspaceBoundary([grant(tmp_path), grant(nested, read=())], relative_root=tmp_path)
    with pytest.raises(WorkspacePermissionError):
        scope.resolve_read_file("nested/secret")
    assert names(scope.list_directory(".")) == []


def test_public_grant_descriptors_cannot_mutate_read_authority(tmp_path):
    (tmp_path / "secret").touch()
    scope = ReadOnlyWorkspaceBoundary([grant(tmp_path, denied=("secret",))])
    view = scope.grants
    assert isinstance(view, tuple)
    view[0].access.denied_paths.clear()
    assert names(scope.list_directory(str(tmp_path))) == []


def test_many_denied_names_return_bounded_advancing_pages_instead_of_looping(tmp_path):
    for index in range(2100):
        (tmp_path / f"denied_{index:04d}").touch()
    (tmp_path / "z_allowed").touch()
    scope = boundary(tmp_path, read=("z_allowed",))
    first = scope.list_directory(max_entries=1)
    assert first["children"] == []
    assert first["expansion"]["state"] == "partial"
    second = scope.list_directory(max_entries=1, cursor=first["expansion"]["next_cursor"])
    assert names(second) == ["z_allowed"]
    assert second["expansion"]["state"] == "complete"


@pytest.mark.skipif(os.name != "posix", reason="Unix text filenames")
@pytest.mark.parametrize("multi_root", [False, True])
def test_noncanonical_filenames_are_discoverable_without_misleading_read_addresses(tmp_path, multi_root):
    (tmp_path / "a").write_text("CANONICAL")
    names = ["a ", " b", "c\\d", "e:f"]
    for name in names:
        (tmp_path / name).write_text("DIFFERENT_FILE")
    scope = (ReadOnlyWorkspaceBoundary([grant(tmp_path)], relative_root=tmp_path) if multi_root
             else boundary(tmp_path))
    path = str(tmp_path) if multi_root else "."
    cursor, nodes = None, []
    while True:
        page = scope.list_directory(path, max_entries=1, cursor=cursor)
        nodes.extend(page["children"])
        cursor = page["expansion"].get("next_cursor")
        if cursor is None:
            assert page["expansion"]["state"] == "complete"
            break
    by_name = {node["name"]: node for node in nodes}
    assert set(by_name) == {"a", *names}
    assert scope.resolve_read_file(by_name["a"]["location"]["path"]).read_text() == "CANONICAL"
    for name in names:
        node = by_name[name]
        assert node["status"] == "unsupported_filename"
        assert "path" not in node["location"]
        expected = str(tmp_path / name)
        assert os.fsdecode(base64.b64decode(node["location"]["path_bytes_base64"])) == expected
        assert "DIFFERENT_FILE" not in json.dumps(node)


@pytest.mark.skipif(os.name != "posix", reason="Unix text filenames")
def test_exact_name_metadata_keeps_denials_and_narrow_symlink_targets_private(tmp_path):
    (tmp_path / "a").write_text("safe")
    denied = tmp_path / "denied"
    denied.mkdir()
    (denied / "e:f").touch()
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret").touch()
    (tmp_path / "alias ").symlink_to(private, target_is_directory=True)
    (tmp_path / "escape ").symlink_to(tmp_path.parent, target_is_directory=True)
    scope = ReadOnlyWorkspaceBoundary([
        grant(tmp_path, denied=("denied",)), grant(private, read=())
    ])
    assert names(scope.list_directory(str(tmp_path))) == ["a"]
    intersected = ReadOnlyWorkspaceBoundary([grant(tmp_path), grant(tmp_path, read=("a",))])
    assert names(intersected.list_directory(str(tmp_path))) == ["a"]


@pytest.mark.skipif(os.name != "posix", reason="Unix text filenames")
def test_exact_name_metadata_does_not_reveal_a_file_outside_partial_read_scope(tmp_path):
    (tmp_path / "a").touch()
    (tmp_path / "a ").touch()
    (tmp_path / "e:f").touch()
    assert names(boundary(tmp_path, read=("a",)).list_directory()) == ["a"]


@pytest.mark.parametrize("read,denied,expected", [
    ((".",), (), [".git", "src"]),
    ((".git/HEAD",), (), [".git"]),
    (("src",), (), ["src"]),
    ((".",), (".git",), ["src"]),
])
def test_git_listing_uses_the_normal_read_scope(tmp_path, read, denied, expected):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "src").mkdir()
    scope = boundary(tmp_path, read=read, denied=denied)
    assert names(scope.list_directory()) == expected
    if ".git" in expected:
        assert names(scope.list_directory(".git")) == ["HEAD"]
    else:
        with pytest.raises(WorkspacePermissionError):
            scope.list_directory(".git")


def test_git_listing_alias_cannot_bypass_a_denied_target(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "git-alias").symlink_to(tmp_path / ".git", target_is_directory=True)
    allowed = boundary(tmp_path)
    assert names(allowed.list_directory("git-alias")) == ["HEAD"]
    denied = boundary(tmp_path, denied=(".git",))
    assert names(denied.list_directory()) == []
    with pytest.raises(WorkspacePermissionError, match="denied_paths"):
        denied.list_directory("git-alias")
