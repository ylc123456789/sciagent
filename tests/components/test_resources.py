
from resagent2_contracts import WorkspaceAccess
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from resagent2_components import (
    DatasetCatalog,
    DatasetResolutionError,
    RepoMaterializer,
    RepoMaterializerError,
    ResourceLayout,
    dataset_context,
    dataset_env_overrides,
    resolve_dataset_refs,
)
from resagent2_components.process import _descendant_pids
from resagent2_contracts import DatasetRef, WorkspaceSourceKind, WorkspaceSpec


def _init_repo(root: Path, *, commit_file: str = "tracked.txt") -> str:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    (root / commit_file).write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


# ── RepoMaterializer ────────────────────────────────────────────────


def _spec(kind: WorkspaceSourceKind, location: str | None = None) -> WorkspaceSpec:
    return WorkspaceSpec(workspace_id='ws_main', source_kind=kind, location=location, access=WorkspaceAccess(read_paths=['.'], write_paths=['.']))


def test_materialize_clones_git_source(tmp_path) -> None:
    source = tmp_path / "source"
    commit = _init_repo(source)
    materialized = RepoMaterializer().materialize(
        workspace=tmp_path / "work", source=_spec(WorkspaceSourceKind.GIT, str(source))
    )

    assert materialized.repo_path.is_dir()
    assert materialized.commit == commit
    assert materialized.source == "git"


def test_materialize_copy_preserves_worktree(tmp_path) -> None:
    source = tmp_path / "source"
    commit = _init_repo(source)
    (source / "uncommitted.py").write_text("dirty = True\n", encoding="utf-8")
    materialized = RepoMaterializer().materialize(
        workspace=tmp_path / "work", source=_spec(WorkspaceSourceKind.COPY, str(source))
    )

    assert materialized.commit == commit
    assert materialized.source == "copy"
    assert (materialized.repo_path / "uncommitted.py").read_text(encoding="utf-8") == (
        "dirty = True\n"
    )


def test_materialize_copy_is_idempotent_on_resume(tmp_path) -> None:
    source = tmp_path / "source"
    commit = _init_repo(source)
    materializer = RepoMaterializer()
    spec = _spec(WorkspaceSourceKind.COPY, str(source))

    first = materializer.materialize(workspace=tmp_path / "work", source=spec)
    second = materializer.materialize(workspace=tmp_path / "work", source=spec)

    assert first.repo_path == second.repo_path
    assert first.commit == second.commit == commit


def test_materialize_copy_into_preexisting_empty_dir(tmp_path) -> None:
    source = tmp_path / "source"
    _init_repo(source)
    workspace = tmp_path / "work"
    workspace.mkdir()

    materialized = RepoMaterializer().materialize(
        workspace=workspace, source=_spec(WorkspaceSourceKind.COPY, str(source))
    )

    assert materialized.repo_path.is_dir()
    assert materialized.source == "copy"


def test_copy_cannot_reuse_workspace_from_other_source(tmp_path) -> None:
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    _init_repo(source_a)
    _init_repo(source_b)
    workspace = tmp_path / "work"
    RepoMaterializer().materialize(
        workspace=workspace, source=_spec(WorkspaceSourceKind.COPY, str(source_a))
    )

    with pytest.raises(RepoMaterializerError, match="does not match"):
        RepoMaterializer().materialize(
            workspace=workspace, source=_spec(WorkspaceSourceKind.COPY, str(source_b))
        )


def test_missing_metadata_cannot_pretend_source_match(tmp_path) -> None:
    source = tmp_path / "source"
    _init_repo(source)
    workspace = tmp_path / "work"
    _init_repo(workspace)  # a git repo, but no materialization metadata

    with pytest.raises(RepoMaterializerError, match="metadata"):
        RepoMaterializer().materialize(
            workspace=workspace, source=_spec(WorkspaceSourceKind.COPY, str(source))
        )


def test_corrupt_metadata_cannot_pretend_source_match(tmp_path) -> None:
    source = tmp_path / "source"
    _init_repo(source)
    workspace = tmp_path / "work"
    _init_repo(workspace)
    (workspace.parent / "workspace.json").write_text("not json", encoding="utf-8")

    with pytest.raises(RepoMaterializerError, match="metadata"):
        RepoMaterializer().materialize(
            workspace=workspace, source=_spec(WorkspaceSourceKind.COPY, str(source))
        )


def test_source_mismatch_is_a_structured_error(tmp_path) -> None:
    source = tmp_path / "source"
    _init_repo(source)
    workspace = tmp_path / "work"
    RepoMaterializer().materialize(
        workspace=workspace, source=_spec(WorkspaceSourceKind.COPY, str(source))
    )

    with pytest.raises(RepoMaterializerError, match="materialized as"):
        RepoMaterializer().materialize(
            workspace=workspace,
            source=_spec(WorkspaceSourceKind.GIT, "https://example.com/other.git"),
        )


def test_materialized_commit_matches_actual_repo(tmp_path) -> None:
    source = tmp_path / "source"
    commit = _init_repo(source)
    materialized = RepoMaterializer().materialize(
        workspace=tmp_path / "work", source=_spec(WorkspaceSourceKind.COPY, str(source))
    )

    actual = subprocess.run(
        ["git", "-C", str(materialized.repo_path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    assert materialized.commit == commit == actual


def test_materialize_binds_local_in_place(tmp_path) -> None:
    source = tmp_path / "source"
    commit = _init_repo(source)
    materialized = RepoMaterializer().materialize(
        workspace=tmp_path / "work", source=_spec(WorkspaceSourceKind.LOCAL, str(source))
    )

    assert materialized.repo_path.resolve() == source.resolve()
    assert materialized.commit == commit
    assert materialized.source == "local"


@pytest.mark.parametrize("kind", [WorkspaceSourceKind.COPY, WorkspaceSourceKind.LOCAL])
def test_materialize_rejects_repository_subdirectory_before_creating_workspace(tmp_path, kind):
    source = tmp_path / "source"
    _init_repo(source)
    subdirectory = source / "package"
    subdirectory.mkdir()
    (subdirectory / "model.py").write_text("x = 1\n")
    workspace = tmp_path / "work"

    with pytest.raises(RepoMaterializerError, match="must be a Git repository root"):
        RepoMaterializer().materialize(
            workspace=workspace, source=_spec(kind, str(subdirectory)),
        )

    assert not workspace.exists()
    assert not (workspace.parent / "workspace.json").exists()


def test_copy_rejects_linked_worktree_without_touching_source_git_state(tmp_path):
    source = tmp_path / "source"
    commit = _init_repo(source)
    linked = tmp_path / "linked"
    subprocess.run(
        ["git", "-C", str(source), "worktree", "add", "-b", "linked", str(linked)],
        check=True, capture_output=True,
    )
    git_pointer = (linked / ".git").read_bytes()
    index = subprocess.run(
        ["git", "-C", str(linked), "rev-parse", "--git-path", "index"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    index_bytes = Path(index).read_bytes()
    workspace = tmp_path / "work"

    with pytest.raises(RepoMaterializerError, match="independent .git directory"):
        RepoMaterializer().materialize(
            workspace=workspace, source=_spec(WorkspaceSourceKind.COPY, str(linked)),
        )

    assert not workspace.exists()
    assert not (workspace.parent / "workspace.json").exists()
    assert (linked / ".git").read_bytes() == git_pointer
    assert Path(index).read_bytes() == index_bytes
    assert subprocess.run(
        ["git", "-C", str(linked), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    ).stdout.strip() == commit
    assert (linked / "tracked.txt").read_text() == "baseline\n"


def test_local_can_bind_linked_worktree_root(tmp_path):
    source = tmp_path / "source"
    commit = _init_repo(source)
    linked = tmp_path / "linked"
    subprocess.run(
        ["git", "-C", str(source), "worktree", "add", "-b", "linked", str(linked)],
        check=True, capture_output=True,
    )

    materialized = RepoMaterializer().materialize(
        workspace=tmp_path / "work", source=_spec(WorkspaceSourceKind.LOCAL, str(linked)),
    )

    assert materialized.repo_path == linked.resolve()
    assert materialized.commit == commit


def test_copy_rejects_linked_git_metadata_directory(tmp_path):
    source = tmp_path / "source"
    _init_repo(source)
    external_metadata = tmp_path / "external-git"
    (source / ".git").rename(external_metadata)
    (source / ".git").symlink_to(external_metadata, target_is_directory=True)

    with pytest.raises(RepoMaterializerError, match="independent .git directory"):
        RepoMaterializer().materialize(
            workspace=tmp_path / "work", source=_spec(WorkspaceSourceKind.COPY, str(source)),
        )

    assert not (tmp_path / "work").exists()
    assert (source / ".git").is_symlink()
    assert external_metadata.is_dir()


def test_materialize_generated_creates_empty_managed_workspace(tmp_path) -> None:
    materialized = RepoMaterializer().materialize(
        workspace=tmp_path / "work", source=_spec(WorkspaceSourceKind.GENERATED)
    )

    assert materialized.repo_path.is_dir()
    assert materialized.source == "generated"
    assert (materialized.repo_path / ".git").is_dir()


def test_git_source_requires_location() -> None:
    with pytest.raises(ValidationError, match="requires a location"):
        WorkspaceSpec(workspace_id='ws_main', source_kind=WorkspaceSourceKind.GIT, access=WorkspaceAccess(read_paths=['.'], write_paths=['.']))


def test_local_source_requires_location() -> None:
    with pytest.raises(ValidationError, match="requires a location"):
        WorkspaceSpec(workspace_id='ws_main', source_kind=WorkspaceSourceKind.LOCAL, access=WorkspaceAccess(read_paths=['.'], write_paths=['.']))


def test_copy_source_requires_location() -> None:
    with pytest.raises(ValidationError, match="requires a location"):
        WorkspaceSpec(workspace_id='ws_main', source_kind=WorkspaceSourceKind.COPY, access=WorkspaceAccess(read_paths=['.'], write_paths=['.']))


def test_generated_source_forbids_location() -> None:
    with pytest.raises(ValidationError, match="must not have a location"):
        WorkspaceSpec(workspace_id='ws_main', source_kind=WorkspaceSourceKind.GENERATED, location='/tmp/x', access=WorkspaceAccess(read_paths=['.'], write_paths=['.']))


def test_resource_layout_respects_resource_root_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RESAGENT2_RESOURCE_ROOT", str(tmp_path / "shared"))
    layout = ResourceLayout.from_env(data_root=tmp_path / "data")
    assert layout.resource_root == tmp_path / "shared"


def test_resource_layout_defaults_derive_from_resource_root(tmp_path) -> None:
    layout = ResourceLayout(resource_root=tmp_path / "resources")

    assert layout.dataset_root == tmp_path / "resources" / "datasets"


def test_resource_layout_accepts_independent_roots(tmp_path) -> None:
    layout = ResourceLayout(
        resource_root=tmp_path / "resources",
        dataset_root=tmp_path / "elsewhere" / "datasets",
    )

    assert layout.dataset_root == tmp_path / "elsewhere" / "datasets"


def test_resource_layout_env_vars_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RESAGENT2_DATASET_ROOT", str(tmp_path / "ds"))

    layout = ResourceLayout.from_env(data_root=tmp_path / "data")

    assert layout.resource_root == tmp_path / "data" / "resources"
    assert layout.dataset_root == tmp_path / "ds"


def test_resource_layout_reads_data_root_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RESAGENT2_RESOURCE_ROOT", raising=False)
    monkeypatch.setenv("RESAGENT2_DATA_ROOT", str(tmp_path / "data"))

    layout = ResourceLayout.from_env()

    assert layout.resource_root == tmp_path / "data" / "resources"


def test_resource_layout_defaults_to_data_root_resources(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RESAGENT2_RESOURCE_ROOT", raising=False)
    monkeypatch.delenv("RESAGENT2_DATASET_ROOT", raising=False)

    layout = ResourceLayout.from_env(data_root=tmp_path / "data")

    assert layout.resource_root == tmp_path / "data" / "resources"
    assert layout.dataset_root == tmp_path / "data" / "resources" / "datasets"
