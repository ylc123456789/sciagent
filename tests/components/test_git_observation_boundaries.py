"""Fixed Git observations never opt into repository-selected executables."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from resagent2_components import RepoMaterializer, RepoMaterializerError, WorkspaceBoundary
from resagent2_components.git import GitWorkspace, GitWorkspaceError
from resagent2_contracts import WorkspaceAccess, WorkspaceGrant, WorkspaceSourceKind, WorkspaceSpec


def git(root, *arguments, text=True):
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True,
        capture_output=True, text=text,
    ).stdout


def init(root):
    root.mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    (root / "tracked.txt").write_text("initial\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "initial")
    return root


def repository(root, *, read_paths=None):
    return GitWorkspace(WorkspaceBoundary(WorkspaceGrant(
        root=str(root), source=WorkspaceSourceKind.LOCAL,
        access=WorkspaceAccess(read_paths=read_paths or ["."], write_paths=[]),
    )))


def marker_script(root, name):
    marker = root / (name + "-ran")
    script = root / (name + ".sh")
    script.write_text("#!/bin/sh\nprintf ran > " + shlex.quote(str(marker)) + "\ncat\n")
    script.chmod(0o755)
    return str(script), marker


@pytest.mark.parametrize("attribute_source", ["nested", "info", "global_include", "worktree"])
def test_snapshot_rejects_effective_external_clean_without_running_it(tmp_path, monkeypatch, attribute_source):
    root = init(tmp_path / "repo")
    command, marker = marker_script(tmp_path, "clean")
    payload = "tracked.txt"
    if attribute_source == "nested":
        nested = root / "nested"
        nested.mkdir()
        (nested / "tracked.txt").write_text("payload\n")
        (nested / ".gitattributes").write_text("tracked.txt filter=probe\n")
        payload = "nested/tracked.txt"
    elif attribute_source == "info":
        (root / ".git/info/attributes").write_text("tracked.txt filter=probe\n")
    elif attribute_source == "global_include":
        home = tmp_path / "home"
        home.mkdir()
        attributes = home / "attributes"
        attributes.write_text("tracked.txt filter=probe\n")
        included = home / "filters"
        included.write_text("[filter \"probe\"]\n clean = " + command + "\n")
        (home / ".gitconfig").write_text(
            "[include]\n path = " + str(included) + "\n[core]\n attributesFile = " + str(attributes) + "\n"
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    else:
        (root / ".gitattributes").write_text("tracked.txt filter=probe\n")
        git(root, "config", "extensions.worktreeConfig", "true")
        git(root, "config", "--worktree", "filter.probe.clean", command)
    if attribute_source not in {"global_include", "worktree"}:
        git(root, "config", "filter.probe.clean", command)

    with pytest.raises(GitWorkspaceError, match="external clean/process.*" + payload):
        repository(root).snapshot()
    assert not marker.exists()


@pytest.mark.parametrize("driver,kind", [("set", "clean"), ("unset", "clean"), ("unspecified", "process"), ("a=b.nested", "process")])
def test_snapshot_handles_literal_filter_names_and_process_drivers(tmp_path, driver, kind):
    root = init(tmp_path / "repo")
    command, marker = marker_script(tmp_path, "filter")
    (root / ".gitattributes").write_text(f"tracked.txt filter={driver}\n")
    git(root, "config", f"filter.{driver}.{kind}", command)
    with pytest.raises(GitWorkspaceError, match="external clean/process"):
        repository(root).snapshot()
    assert not marker.exists()


def test_snapshot_does_not_block_unused_or_out_of_scope_filters(tmp_path):
    root = init(tmp_path / "repo")
    command, marker = marker_script(tmp_path, "unused")
    git(root, "config", "filter.unused.clean", command)
    (root / "hidden.txt").write_text("not in the grant\n")
    (root / ".gitattributes").write_text("hidden.txt filter=unused\n")
    observed = repository(root, read_paths=["tracked.txt"])
    baseline = observed.snapshot()
    (root / "tracked.txt").write_text("visible update\n")
    assert observed.changed_paths_since(baseline) == ["tracked.txt"]
    assert not marker.exists()


def test_snapshot_checks_attributes_against_the_same_empty_index_as_add(tmp_path):
    root = init(tmp_path / "repo")
    # Attribute fallback to the user's index would select an obsolete filter.
    (root / ".gitattributes").write_text("tracked.txt filter=old\n")
    git(root, "add", ".gitattributes")
    (root / ".gitattributes").unlink()
    command, marker = marker_script(tmp_path, "old")
    git(root, "config", "filter.old.clean", command)
    baseline = repository(root).snapshot()
    assert baseline.tree_hash
    assert not marker.exists()


@pytest.mark.parametrize("helper", ["hook", "fsmonitor", "external_diff", "textconv"])
def test_fixed_observations_do_not_execute_configured_helpers(tmp_path, helper):
    root = init(tmp_path / "repo")
    observed = repository(root)
    baseline = observed.snapshot()
    command, marker = marker_script(tmp_path, helper)
    if helper == "hook":
        hooks = tmp_path / "hooks"
        hooks.mkdir()
        hook = hooks / "post-index-change"
        hook.write_text(Path(command).read_text())
        hook.chmod(0o755)
        git(root, "config", "core.hooksPath", str(hooks))
    elif helper == "fsmonitor":
        git(root, "config", "core.fsmonitor", command)
    elif helper == "external_diff":
        git(root, "config", "diff.external", command)
    else:
        (root / ".gitattributes").write_text("tracked.txt diff=probe\n")
        git(root, "config", "diff.probe.textconv", command)
    (root / "tracked.txt").write_text("after\n")
    assert observed.changed_paths_since(baseline)
    assert "+after" in observed.diff_since(baseline)
    assert not marker.exists()


def test_snapshot_preserves_index_files_and_attempt_increment_with_builtin_eol(tmp_path):
    root = init(tmp_path / "repo")
    (root / "tracked.txt").write_text("already staged\n")
    git(root, "add", "tracked.txt")
    (root / "tracked.txt").write_text("before attempt\n")
    (root / ".gitattributes").write_text("*.txt text eol=lf\n")
    (root / "untracked.txt").write_bytes(b"before\r\n")
    original_index = (root / ".git/index").read_bytes()
    original_head = git(root, "rev-parse", "HEAD")
    original_files = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    observed = repository(root)
    baseline = observed.snapshot()
    assert git(root, "show", baseline.tree_hash + ":untracked.txt", text=False) == b"before\n"
    assert original_files == {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    (root / "tracked.txt").write_text("after attempt\n")
    patch = observed.diff_since(baseline)
    assert "-before attempt" in patch and "+after attempt" in patch
    assert "already staged" not in patch
    assert (root / ".git/index").read_bytes() == original_index
    assert git(root, "rev-parse", "HEAD") == original_head
    assert (root / "untracked.txt").read_bytes() == b"before\r\n"


def test_snapshot_ignores_inherited_git_repository_and_config_controls(tmp_path, monkeypatch):
    root = init(tmp_path / "repo")
    foreign = init(tmp_path / "foreign")
    foreign_index = (foreign / ".git/index").read_bytes()
    command, marker = marker_script(tmp_path, "injected")
    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(foreign))
    monkeypatch.setenv("GIT_INDEX_FILE", str(foreign / ".git/index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(foreign / ".git/objects"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", command)
    baseline = repository(root).snapshot()
    assert baseline.tree_hash
    assert (foreign / ".git/index").read_bytes() == foreign_index
    assert not marker.exists()


def test_snapshot_supports_linked_worktree_without_changing_its_index(tmp_path):
    source = init(tmp_path / "source")
    linked = tmp_path / "linked"
    git(source, "worktree", "add", "-b", "linked", str(linked))
    index = Path(git(linked, "rev-parse", "--git-path", "index").strip())
    original_index = index.read_bytes()
    observed = repository(linked)
    baseline = observed.snapshot()
    (linked / "tracked.txt").write_text("linked update\n")
    assert observed.changed_paths_since(baseline) == ["tracked.txt"]
    assert "+linked update" in observed.diff_since(baseline)
    assert index.read_bytes() == original_index


@pytest.mark.parametrize("used", [True, False])
def test_clone_checks_actual_checkout_filters_and_does_not_execute_them(tmp_path, monkeypatch, used):
    source = init(tmp_path / "source")
    (source / ".gitattributes").write_text("tracked.txt filter=probe\n" if used else "unused.bin filter=probe\n")
    git(source, "add", ".gitattributes")
    git(source, "commit", "-qm", "attributes")
    command, marker = marker_script(tmp_path, "smudge")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[filter \"probe\"]\n smudge = " + command + "\n")
    monkeypatch.setenv("HOME", str(home))
    destination = tmp_path / "clone"
    spec = WorkspaceSpec(workspace_id="ws_clone", source_kind=WorkspaceSourceKind.GIT,
                         location=str(source), access=WorkspaceAccess(read_paths=["."], write_paths=["."]))
    if used:
        with pytest.raises(RepoMaterializerError, match="external smudge/process"):
            RepoMaterializer().materialize(workspace=destination, source=spec)
        assert not destination.exists()
        assert not (tmp_path / ".clone.resagent2-staging").exists()
    else:
        result = RepoMaterializer().materialize(workspace=destination, source=spec)
        assert result.commit
        assert (destination / "tracked.txt").read_text() == "initial\n"
    assert not marker.exists()


def test_clone_preserves_an_empty_repository(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    git(source, "init", "-q")
    spec = WorkspaceSpec(workspace_id="ws_empty", source_kind=WorkspaceSourceKind.GIT,
                         location=str(source), access=WorkspaceAccess(read_paths=["."], write_paths=["."]))
    result = RepoMaterializer().materialize(workspace=tmp_path / "clone", source=spec)
    assert result.commit == "" and (result.repo_path / ".git").is_dir()


def test_observation_does_not_lazy_fetch_missing_objects(tmp_path):
    source = init(tmp_path / "remote")
    root = init(tmp_path / "repo")
    observed = repository(root)
    baseline = observed.snapshot()
    blob = git(root, "rev-parse", "HEAD:tracked.txt").strip()
    (root / ".git/objects" / blob[:2] / blob[2:]).unlink()
    marker = tmp_path / "upload-pack-ran"
    uploader = tmp_path / "upload-pack"
    uploader.write_text(
        "#!/bin/sh\nprintf ran > " + shlex.quote(str(marker)) + '\nexec git-upload-pack "$@"\n'
    )
    uploader.chmod(0o755)
    git(root, "config", "extensions.partialClone", "origin")
    git(root, "config", "remote.origin.url", str(source))
    git(root, "config", "remote.origin.promisor", "true")
    git(root, "config", "remote.origin.uploadpack", str(uploader))
    (root / "tracked.txt").write_text("after\n")
    with pytest.raises(GitWorkspaceError):
        observed.diff_since(baseline)
    assert not marker.exists()


@pytest.mark.parametrize("kind", [WorkspaceSourceKind.GENERATED, WorkspaceSourceKind.GIT])
def test_materializer_does_not_install_or_run_template_hooks(tmp_path, monkeypatch, kind):
    source = init(tmp_path / "source")
    template = tmp_path / "template"
    hooks = template / "hooks"
    hooks.mkdir(parents=True)
    command, marker = marker_script(tmp_path, "template")
    for name in ("post-checkout", "post-index-change"):
        hook = hooks / name
        hook.write_text(Path(command).read_text())
        hook.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[init]\n templateDir = " + str(template) + "\n")
    monkeypatch.setenv("HOME", str(home))
    spec = WorkspaceSpec(
        workspace_id="ws_template", source_kind=kind,
        location=str(source) if kind == WorkspaceSourceKind.GIT else None,
        access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
    )
    materialized = RepoMaterializer().materialize(workspace=tmp_path / "work", source=spec)
    assert not marker.exists()
    assert not (materialized.repo_path / ".git/hooks/post-checkout").exists()
    assert not (materialized.repo_path / ".git/hooks/post-index-change").exists()


def test_configured_sentinel_driver_without_a_filter_attribute_is_unused(tmp_path):
    root = init(tmp_path / "repo")
    command, marker = marker_script(tmp_path, "sentinel")
    for name in ("set", "unset", "unspecified"):
        git(root, "config", f"filter.{name}.clean", command)
    assert repository(root).snapshot().tree_hash
    assert not marker.exists()


@pytest.mark.parametrize("attribute,driver", [("filter", "set"), ("-filter", "unset")])
def test_boolean_filter_attributes_with_ambiguous_configured_driver_fail_explicitly(tmp_path, attribute, driver):
    root = init(tmp_path / "repo")
    command, marker = marker_script(tmp_path, "ambiguous")
    git(root, "config", f"filter.{driver}.clean", command)
    (root / ".gitattributes").write_text(f"tracked.txt {attribute}\n")
    with pytest.raises(GitWorkspaceError, match="ambiguously named"):
        repository(root).snapshot()
    assert not marker.exists()


@pytest.mark.parametrize("direction", ["clean", "process"])
def test_empty_filter_name_cannot_escape_snapshot_preflight(tmp_path, direction):
    root = init(tmp_path / "repo")
    command, marker = marker_script(tmp_path, "empty-filter")
    git(root, "config", f"filter..{direction}", command)
    (root / ".gitattributes").write_text("tracked.txt filter=\n")
    with pytest.raises(GitWorkspaceError, match="external clean/process"):
        repository(root).snapshot()
    assert not marker.exists()


@pytest.mark.parametrize("direction", ["smudge", "process"])
def test_empty_filter_name_cannot_escape_clone_preflight(tmp_path, monkeypatch, direction):
    source = init(tmp_path / "source")
    (source / ".gitattributes").write_text("tracked.txt filter=\n")
    git(source, "add", ".gitattributes")
    git(source, "commit", "-qm", "empty filter attributes")
    command, marker = marker_script(tmp_path, "empty-checkout-filter")
    home = tmp_path / "home"
    home.mkdir()
    config = home / ".gitconfig"
    config.write_text(f'[filter ""]\n {direction} = {command}\n')
    monkeypatch.setenv("HOME", str(home))
    spec = WorkspaceSpec(workspace_id="ws_empty_filter", source_kind=WorkspaceSourceKind.GIT,
        location=str(source), access=WorkspaceAccess(read_paths=["."], write_paths=["."]))
    with pytest.raises(RepoMaterializerError, match="external smudge/process"):
        RepoMaterializer().materialize(workspace=tmp_path / "clone", source=spec)
    assert not (tmp_path / "clone").exists()
    assert not marker.exists()


def test_snapshot_preserves_required_filter_without_a_clean_direction(tmp_path):
    root = init(tmp_path / "repo")
    command, marker = marker_script(tmp_path, "smudge-only")
    (root / ".gitattributes").write_text("tracked.txt filter=probe\n")
    git(root, "config", "filter.probe.smudge", command)
    git(root, "config", "filter.probe.required", "true")
    baseline = repository(root).snapshot()
    assert git(root, "show", baseline.tree_hash + ":tracked.txt") == "initial\n"
    assert not marker.exists()


def test_clone_preserves_required_filter_without_a_smudge_direction(tmp_path, monkeypatch):
    source = init(tmp_path / "source")
    (source / ".gitattributes").write_text("tracked.txt filter=probe\n")
    git(source, "add", ".gitattributes")
    git(source, "commit", "-qm", "attributes")
    command, marker = marker_script(tmp_path, "clean-only")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text(
        '[filter "probe"]\n clean = ' + command + '\n required = true\n'
    )
    monkeypatch.setenv("HOME", str(home))
    spec = WorkspaceSpec(workspace_id="ws_clean_only", source_kind=WorkspaceSourceKind.GIT,
        location=str(source), access=WorkspaceAccess(read_paths=["."], write_paths=["."]))
    result = RepoMaterializer().materialize(workspace=tmp_path / "clone", source=spec)
    assert (result.repo_path / "tracked.txt").read_text() == "initial\n"
    assert not marker.exists()
