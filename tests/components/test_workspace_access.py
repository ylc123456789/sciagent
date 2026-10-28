"""Workspace grants constrain reads, writes and exact prepared deletions."""

import json
import os

import pytest

from resagent2_components.workspace import (
    DeleteChangedError, DeleteInterruptedError, WorkspaceBoundary, WorkspacePermissionError,
)
from resagent2_contracts import WorkspaceAccess, WorkspaceGrant


def boundary(root, *, read=(".",), write=(".",), denied=()):
    return WorkspaceBoundary(WorkspaceGrant(
        root=str(root), source="local", access=WorkspaceAccess(
            read_paths=list(read), write_paths=list(write), denied_paths=list(denied),
        ),
    ))


def test_empty_access_authorizes_no_files(tmp_path):
    (tmp_path / "source.py").write_text("x = 1")
    scope = WorkspaceBoundary(WorkspaceGrant(
        root=str(tmp_path), source="local", access=WorkspaceAccess(),
    ))
    with pytest.raises(WorkspacePermissionError, match="read scope"):
        scope.resolve_read_file("source.py")
    with pytest.raises(WorkspacePermissionError, match="write scope"):
        scope.resolve_write_file("source.py")


def test_partial_scope_can_list_authorized_descendants_without_parent_access(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "visible.py").write_text("pass")
    (tmp_path / "secret.py").write_text("secret")
    scope = boundary(tmp_path, read=("src/visible.py",), write=())
    assert scope.iter_files() == ["src/visible.py"]
    with pytest.raises(WorkspacePermissionError):
        scope.resolve_read_file("secret.py")


def test_new_authorized_directory_does_not_require_authorizing_parent(tmp_path):
    scope = boundary(tmp_path, read=("results",), write=("results",))
    assert scope.resolve_write_file("results/nested/out.json") == tmp_path / "results/nested/out.json"
    with pytest.raises(WorkspacePermissionError):
        scope.resolve_write_file("other.json")


def test_parent_symlink_rechecks_effective_write_scope(tmp_path):
    (tmp_path / "private").mkdir()
    (tmp_path / "results").symlink_to(tmp_path / "private", target_is_directory=True)
    scope = boundary(tmp_path, denied=("private",))
    with pytest.raises(WorkspacePermissionError):
        scope.resolve_write_file("results/new.txt")


def test_system_outputs_are_separate_from_empty_file_grants(tmp_path):
    scope = boundary(tmp_path, read=(), write=())
    assert scope.resolve_system_write(".resagent2/results/a.txt") == tmp_path / ".resagent2/results/a.txt"
    with pytest.raises(WorkspacePermissionError):
        scope.resolve_system_write("source.py")


def test_system_output_link_cannot_enter_user_source(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / ".resagent2").symlink_to(tmp_path / "src", target_is_directory=True)
    with pytest.raises(WorkspacePermissionError):
        boundary(tmp_path).resolve_system_write(".resagent2/source.py")


def test_exact_file_and_empty_directory_delete_without_confirmation(tmp_path):
    (tmp_path / "old.py").write_text("pass")
    (tmp_path / "empty").mkdir()
    scope = boundary(tmp_path)
    for path in ("old.py", "empty"):
        snapshot = scope.prepare_delete(path)
        assert snapshot["requires_confirmation"] is False
        assert scope.delete_prepared(json.loads(json.dumps(snapshot))) == [path]
        assert not (tmp_path / path).exists()


def test_link_delete_is_authorized_by_link_path_and_never_follows_target(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.write_text("untouched")
    (root / "alias").symlink_to(outside)
    scope = boundary(root, read=("alias",), write=("alias",))
    with pytest.raises(WorkspacePermissionError):
        scope.resolve_read_file("alias")
    assert scope.delete_prepared(scope.prepare_delete("alias")) == ["alias"]
    assert outside.read_text() == "untouched"
    assert not (root / "alias").is_symlink()


def test_broken_link_delete_does_not_resolve_missing_target(tmp_path):
    (tmp_path / "broken").symlink_to(tmp_path / "absent")
    scope = boundary(tmp_path)
    assert scope.delete_prepared(scope.prepare_delete("broken")) == ["broken"]


def test_directory_link_is_unlinked_even_when_recursive(tmp_path):
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "keep").write_text("keep")
    (tmp_path / "alias").symlink_to(tmp_path / "target", target_is_directory=True)
    scope = boundary(tmp_path, denied=("target",))
    snapshot = scope.prepare_delete("alias", recursive=True)
    assert snapshot["requires_confirmation"] is False
    scope.delete_prepared(snapshot)
    assert (tmp_path / "target" / "keep").read_text() == "keep"


def test_recursive_cache_delete_is_snapshotted_and_requires_confirmation(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_bytes(b"cache")
    scope = boundary(tmp_path)
    assert scope.iter_files() == []
    with pytest.raises(WorkspacePermissionError, match="recursive=True"):
        scope.prepare_delete("__pycache__")
    snapshot = scope.prepare_delete("__pycache__", recursive=True)
    assert snapshot["requires_confirmation"] is True
    assert scope.delete_prepared(snapshot) == ["__pycache__/a.pyc", "__pycache__"]


@pytest.mark.parametrize("path", [".", "../outside", ".git", ".resagent2"])
def test_delete_rejects_root_traversal_and_metadata(tmp_path, path):
    with pytest.raises(WorkspacePermissionError):
        boundary(tmp_path).prepare_delete(path, recursive=True)


def test_recursive_delete_validates_all_descendants_before_deleting_anything(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "a.txt").write_text("visible")
    (tmp_path / "build" / "private").write_text("denied")
    scope = boundary(tmp_path, denied=("build/private",))
    with pytest.raises(WorkspacePermissionError):
        scope.prepare_delete("build", recursive=True)
    assert (tmp_path / "build" / "a.txt").read_text() == "visible"


@pytest.mark.parametrize("change", ["add", "edit", "replace", "remove"])
def test_changed_delete_snapshot_cannot_be_used(tmp_path, change):
    (tmp_path / "build").mkdir()
    path = tmp_path / "build" / "file"
    path.write_text("original")
    scope = boundary(tmp_path)
    snapshot = scope.prepare_delete("build", recursive=True)
    if change == "add":
        (tmp_path / "build" / "new").write_text("new")
    elif change == "edit":
        path.write_text("changed")
    elif change == "replace":
        path.rename(tmp_path / "original")
        path.write_text("original")
    else:
        path.unlink()
    with pytest.raises(DeleteChangedError):
        scope.delete_prepared(snapshot)
    assert (tmp_path / "build").is_dir()


def test_revoked_scope_is_rechecked_when_using_prepared_delete(tmp_path):
    path = tmp_path / "old"
    path.write_text("keep")
    snapshot = boundary(tmp_path).prepare_delete("old")
    with pytest.raises(WorkspacePermissionError):
        boundary(tmp_path, write=()).delete_prepared(snapshot)
    assert path.read_text() == "keep"


def test_delete_parent_symlink_cannot_escape_after_snapshot(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "build").mkdir()
    (root / "build" / "file").write_text("old")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("keep")
    scope = boundary(root)
    snapshot = scope.prepare_delete("build/file")
    (root / "build").rename(root / "original")
    (root / "build").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspacePermissionError):
        scope.delete_prepared(snapshot)
    assert (outside / "file").read_text() == "keep"


def test_interrupted_delete_reports_exact_progress(tmp_path, monkeypatch):
    (tmp_path / "build").mkdir()
    for name in ("a", "b"):
        (tmp_path / "build" / name).write_text(name)
    scope = boundary(tmp_path)
    snapshot = scope.prepare_delete("build", recursive=True)
    unlink = os.unlink

    def interrupted(path, **kwargs):
        if path == "a":
            raise OSError("interrupted")
        return unlink(path, **kwargs)

    monkeypatch.setattr(os, "unlink", interrupted)
    with pytest.raises(DeleteInterruptedError) as result:
        scope.delete_prepared(snapshot)
    assert result.value.deleted == ["build/b"]
    assert result.value.remaining == ["build/a", "build"]
    assert (tmp_path / "build" / "a").is_file()


@pytest.mark.parametrize("read,denied,allowed", [
    ((".",), (), True),
    ((".git",), (), True),
    ((".git/HEAD",), (), True),
    (("src",), (), False),
    ((".",), (".git",), False),
    ((".git",), (".git/HEAD",), False),
])
def test_git_reads_follow_the_normal_read_scope(tmp_path, read, denied, allowed):
    (tmp_path / ".git").mkdir()
    head = tmp_path / ".git/HEAD"
    head.write_text("ref: refs/heads/main\n")
    scope = boundary(tmp_path, read=read, write=(), denied=denied)
    assert scope.allows_read(".git/HEAD") is allowed
    if allowed:
        assert scope.resolve_read_file(".git/HEAD") == head
        assert scope.iter_files() == [".git/HEAD"]
    else:
        with pytest.raises(WorkspacePermissionError):
            scope.resolve_read_file(".git/HEAD")


@pytest.mark.parametrize("path", [".git/HEAD", "git-alias/HEAD"])
def test_git_reads_do_not_authorize_writes_or_deletes_through_aliases(tmp_path, path):
    (tmp_path / ".git").mkdir()
    head = tmp_path / ".git/HEAD"
    head.write_text("ref: refs/heads/main\n")
    (tmp_path / "git-alias").symlink_to(tmp_path / ".git", target_is_directory=True)
    scope = boundary(tmp_path)
    assert scope.resolve_read_file(path) == head
    with pytest.raises(WorkspacePermissionError, match="runtime-reserved"):
        scope.resolve_write_file(path)
    with pytest.raises(WorkspacePermissionError, match="runtime-reserved"):
        scope.prepare_delete(path)
    assert head.read_text() == "ref: refs/heads/main\n"


def test_recursive_parent_delete_cannot_remove_git_metadata(tmp_path):
    repository = tmp_path / "vendor/repository"
    (repository / ".git").mkdir(parents=True)
    (repository / ".git/HEAD").write_text("ref: refs/heads/main\n")
    source = repository / "source.py"
    source.write_text("value = 1\n")
    scope = boundary(tmp_path)
    with pytest.raises(WorkspacePermissionError, match="runtime-reserved"):
        scope.prepare_delete("vendor", recursive=True)
    assert source.read_text() == "value = 1\n"
    assert (repository / ".git/HEAD").exists()


def test_git_pointer_file_is_readable_but_still_cannot_be_replaced_or_deleted(tmp_path):
    pointer = tmp_path / ".git"
    pointer.write_text("gitdir: /outside/repository/.git/worktrees/linked\n")
    scope = boundary(tmp_path)
    assert scope.resolve_read_file(".git") == pointer
    with pytest.raises(NotADirectoryError):
        scope.resolve_read_directory(".git")
    with pytest.raises(WorkspacePermissionError, match="runtime-reserved"):
        scope.resolve_write_file(".git")
    with pytest.raises(WorkspacePermissionError, match="runtime-reserved"):
        scope.prepare_delete(".git")


@pytest.mark.parametrize("path", [".resagent2/state", "runtime-alias/state"])
def test_runtime_metadata_remains_unreadable_and_unwritable(tmp_path, path):
    (tmp_path / ".resagent2").mkdir()
    (tmp_path / ".resagent2/state").write_text("private")
    (tmp_path / "runtime-alias").symlink_to(tmp_path / ".resagent2", target_is_directory=True)
    scope = boundary(tmp_path)
    with pytest.raises(WorkspacePermissionError, match="runtime-reserved"):
        scope.resolve_read_file(path)
    with pytest.raises(WorkspacePermissionError, match="runtime-reserved"):
        scope.resolve_write_file(path)
    with pytest.raises(WorkspacePermissionError, match="runtime-reserved"):
        scope.prepare_delete(path)



def test_git_file_alias_can_be_unlinked_but_cannot_overwrite_its_target(tmp_path):
    (tmp_path / ".git").mkdir()
    head = tmp_path / ".git/HEAD"
    head.write_text("ref: refs/heads/main\n")
    alias = tmp_path / "git-head"
    alias.symlink_to(head)
    scope = boundary(tmp_path)
    with pytest.raises(WorkspacePermissionError, match="runtime-reserved"):
        scope.resolve_write_file("git-head")
    assert scope.delete_prepared(scope.prepare_delete("git-head")) == ["git-head"]
    assert not alias.is_symlink()
    assert head.read_text() == "ref: refs/heads/main\n"
