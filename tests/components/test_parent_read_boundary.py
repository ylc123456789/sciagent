"""Parent reads use only explicit roots and preserve each workspace boundary."""

import pytest

from sciagent_components import ReadOnlyWorkspaceBoundary, WorkspacePermissionError
from sciagent_contracts import WorkspaceAccess, WorkspaceGrant


def grant(root, *, read=(".",), denied=()):
    return WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(
        read_paths=list(read), write_paths=[], denied_paths=list(denied),
    ))


def test_parent_can_read_multiple_roots_but_not_siblings(tmp_path):
    roots = [tmp_path / name for name in ("source", "outputs")]
    for root in roots:
        root.mkdir()
        (root / "current.txt").write_text(root.name)
    outside = tmp_path / "ungranted.txt"
    outside.write_text("private")
    boundary = ReadOnlyWorkspaceBoundary([grant(root) for root in roots])
    for root in roots:
        assert boundary.resolve_read_file(str(root / "current.txt")).read_text() == root.name
    with pytest.raises(WorkspacePermissionError, match="authorized read workspaces"):
        boundary.resolve_read_file(str(outside))


@pytest.mark.parametrize("path", ["relative.txt", "../outside.txt", "/tmp/../outside.txt"])
def test_parent_requires_absolute_nontraversing_paths(tmp_path, path):
    with pytest.raises(WorkspacePermissionError, match="absolute path"):
        ReadOnlyWorkspaceBoundary([grant(tmp_path)]).resolve_read_file(path)


@pytest.mark.parametrize("restricted", ["private.txt", ".sciagent/state"])
def test_parent_keeps_denials_and_reserved_paths(tmp_path, restricted):
    path = tmp_path / restricted
    path.parent.mkdir(exist_ok=True)
    path.write_text("private")
    boundary = ReadOnlyWorkspaceBoundary([grant(tmp_path, denied=("private.txt",))])
    with pytest.raises(WorkspacePermissionError):
        boundary.resolve_read_file(str(path))


def test_parent_uses_specific_root_without_broad_fallback(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    path = nested / "private.txt"
    path.write_text("private")
    boundary = ReadOnlyWorkspaceBoundary([grant(tmp_path), grant(nested, denied=("private.txt",))])
    with pytest.raises(WorkspacePermissionError, match="denied_paths"):
        boundary.resolve_read_file(str(path))


def test_equal_roots_cannot_remove_an_existing_denial(tmp_path):
    path = tmp_path / "private.txt"
    path.write_text("private")
    for grants in ([grant(tmp_path), grant(tmp_path, denied=("private.txt",))],
                   [grant(tmp_path, denied=("private.txt",)), grant(tmp_path)]):
        with pytest.raises(WorkspacePermissionError, match="denied_paths"):
            ReadOnlyWorkspaceBoundary(grants).resolve_read_file(str(path))


def test_missing_root_is_lazy_and_not_replaced_by_parent(tmp_path):
    missing = tmp_path / "not_created"
    boundary = ReadOnlyWorkspaceBoundary([grant(tmp_path), grant(missing)])
    with pytest.raises(FileNotFoundError):
        boundary.resolve_read_file(str(missing / "current.txt"))
    missing.mkdir()
    path = missing / "current.txt"
    path.write_text("created later")
    assert boundary.resolve_read_file(str(path)) == path


def test_parent_rechecks_symlink_target_and_does_not_cross_roots(tmp_path):
    source, outputs = tmp_path / "source", tmp_path / "outputs"
    source.mkdir()
    outputs.mkdir()
    target = outputs / "current.txt"
    target.write_text("current")
    (source / "alias.txt").symlink_to(target)
    boundary = ReadOnlyWorkspaceBoundary([grant(source), grant(outputs)])
    with pytest.raises(WorkspacePermissionError, match="escapes workspace root"):
        boundary.resolve_read_file(str(source / "alias.txt"))
    assert boundary.resolve_read_file(str(target)) == target


def test_parent_keeps_resolved_target_denials_within_root(tmp_path):
    target = tmp_path / "private.txt"
    target.write_text("private")
    (tmp_path / "alias.txt").symlink_to(target)
    with pytest.raises(WorkspacePermissionError, match="denied_paths"):
        ReadOnlyWorkspaceBoundary([grant(tmp_path, denied=("private.txt",))]).resolve_read_file(
            str(tmp_path / "alias.txt"))


@pytest.mark.parametrize("restriction", ["denied", "read_scope"])
def test_parent_alias_cannot_bypass_a_more_specific_target_grant(tmp_path, restriction):
    nested = tmp_path / "private"
    nested.mkdir()
    target = nested / "secret.txt"
    target.write_text("private")
    (tmp_path / "alias").symlink_to(nested, target_is_directory=True)
    narrow = grant(nested, denied=("secret.txt",)) if restriction == "denied" else grant(nested, read=("public",))
    boundary = ReadOnlyWorkspaceBoundary([grant(tmp_path), narrow])
    with pytest.raises(WorkspacePermissionError):
        boundary.resolve_read_file(str(tmp_path / "alias" / "secret.txt"))


def test_parent_target_matching_preserves_same_root_grant_intersection(tmp_path):
    nested = tmp_path / "private"
    nested.mkdir()
    target = nested / "secret.txt"
    target.write_text("private")
    (tmp_path / "alias").symlink_to(nested, target_is_directory=True)
    for grants in ([grant(tmp_path), grant(nested), grant(nested, denied=("secret.txt",))],
                   [grant(tmp_path), grant(nested, denied=("secret.txt",)), grant(nested)]):
        with pytest.raises(WorkspacePermissionError, match="denied_paths"):
            ReadOnlyWorkspaceBoundary(grants).resolve_read_file(str(tmp_path / "alias" / "secret.txt"))


@pytest.mark.parametrize("read,denied,allowed", [
    ((".",), (), True),
    ((".git/HEAD",), (), True),
    (("src",), (), False),
    ((".",), (".git",), False),
    ((".",), (".git/HEAD",), False),
])
def test_parent_git_reads_keep_workspace_read_scope(tmp_path, read, denied, allowed):
    (tmp_path / ".git").mkdir()
    head = tmp_path / ".git/HEAD"
    head.write_text("ref: refs/heads/main\n")
    boundary = ReadOnlyWorkspaceBoundary([grant(tmp_path, read=read, denied=denied)])
    if allowed:
        assert boundary.resolve_read_file(str(head)) == head
    else:
        with pytest.raises(WorkspacePermissionError):
            boundary.resolve_read_file(str(head))


def test_parent_git_pointer_does_not_grant_its_external_metadata_root(tmp_path):
    workspace = tmp_path / "linked"
    workspace.mkdir()
    metadata = tmp_path / "repository/.git/worktrees/linked"
    metadata.mkdir(parents=True)
    head = metadata / "HEAD"
    head.write_text("ref: refs/heads/main\n")
    pointer = workspace / ".git"
    pointer.write_text(f"gitdir: {metadata}\n")
    boundary = ReadOnlyWorkspaceBoundary([grant(workspace)])
    assert boundary.resolve_read_file(str(pointer)) == pointer
    with pytest.raises(WorkspacePermissionError, match="authorized read workspaces"):
        boundary.resolve_read_file(str(head))


def test_parent_git_alias_cannot_bypass_a_narrower_target_grant(tmp_path):
    metadata = tmp_path / ".git"
    metadata.mkdir()
    head = metadata / "HEAD"
    head.write_text("ref: refs/heads/main\n")
    (tmp_path / "git-alias").symlink_to(metadata, target_is_directory=True)
    boundary = ReadOnlyWorkspaceBoundary([grant(tmp_path), grant(metadata, read=())])
    with pytest.raises(WorkspacePermissionError, match="read scope"):
        boundary.resolve_read_file(str(tmp_path / "git-alias/HEAD"))


@pytest.mark.parametrize("initially_exists", [False, True])
@pytest.mark.parametrize("operation", ["read", "list"])
def test_parent_does_not_accept_root_replaced_by_link_to_another_grant(
    tmp_path, initially_exists, operation,
):
    selected, other = tmp_path / "selected", tmp_path / "other"
    if initially_exists:
        selected.mkdir()
    other.mkdir()
    (other / "private.txt").write_text("another root")
    boundary = ReadOnlyWorkspaceBoundary([grant(selected), grant(other)])
    if initially_exists:
        selected.rmdir()
    selected.symlink_to(other, target_is_directory=True)
    with pytest.raises(WorkspacePermissionError, match="changed"):
        if operation == "read":
            boundary.resolve_read_file(str(selected / "private.txt"))
        else:
            boundary.list_directory(str(selected))
    assert boundary.resolve_read_file(str(other / "private.txt")).read_text() == "another root"
