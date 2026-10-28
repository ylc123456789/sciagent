import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from resagent2_contracts import (
    AgentOwner,
    ArtifactRef,
    VerificationResult,
    WorkspaceGrant,
    WorkspaceAccess,
    WorkspaceSourceKind,
)
from resagent2_components import (
    ArtifactReadError,
    GitWorkspace,
    ProcessRunner,
    RegisteredArtifactReader,
    WorkspaceBoundary,
    WorkspacePermissionError,
)
from resagent2_capabilities import (
    ReadFileTool,
    ReplaceTextTool,
)
from resagent2_runtime import AgentState


def grant(root: Path, *, read_only: bool = False, denied_paths=("denied",)) -> WorkspaceGrant:
    return WorkspaceGrant(
        root=str(root),
        access=WorkspaceAccess(read_paths=["."], write_paths=[] if read_only else ["."],
                               denied_paths=list(denied_paths)),
        source=WorkspaceSourceKind.LOCAL,
    )


def init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    (root / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)


def test_workspace_rejects_traversal_reserved_paths_and_escaping_symlink(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)
    boundary = WorkspaceBoundary(grant(root))

    with pytest.raises(WorkspacePermissionError):
        boundary.resolve_read_file("../outside.txt")
    with pytest.raises(WorkspacePermissionError):
        boundary.resolve_read_file(".resagent2/state")
    with pytest.raises(WorkspacePermissionError):
        boundary.resolve_read_file("link.txt")


def test_read_only_workspace_cannot_resolve_a_write(tmp_path) -> None:
    boundary = WorkspaceBoundary(grant(tmp_path, read_only=True))

    with pytest.raises(WorkspacePermissionError, match="write scope"):
        boundary.resolve_write_file("new.py", must_be_new=True)








def test_process_runner_writes_shell_logs(tmp_path) -> None:
    boundary = WorkspaceBoundary(grant(tmp_path))
    result = ProcessRunner(boundary).run_shell(
        f'{sys.executable} -c "print(42)"',
        log_dir=".resagent2/test",
        index=1,
        timeout_seconds=10,
    )

    assert result.exit_code == 0
    assert (tmp_path / result.stdout_path).read_text(encoding="utf-8").strip() == "42"


def test_process_runner_marks_timeout_and_stops_process(tmp_path) -> None:
    boundary = WorkspaceBoundary(grant(tmp_path))
    result = ProcessRunner(boundary).run_shell(
        f'{sys.executable} -c "import time; time.sleep(5)"',
        log_dir=".resagent2/timeout",
        index=1,
        timeout_seconds=1,
    )

    assert result.timed_out is True
    assert result.exit_code != 0


def test_process_runner_extra_env_cannot_inject_credentials(tmp_path) -> None:
    boundary = WorkspaceBoundary(grant(tmp_path))
    result = ProcessRunner(boundary).run_shell(
        "env",
        log_dir=".resagent2/env",
        index=1,
        timeout_seconds=10,
        extra_env={
            "DEEPSEEK_API_KEY": "sk-secret-deepseek",
            "OPENAI_API_KEY": "sk-secret-openai",
            "SSH_AUTH_SOCK": "/tmp/ssh-agent-secret.sock",
            "AWS_SECRET_ACCESS_KEY": "cloud-secret-aws",
            "MY_CUSTOM_VAR": "visible-value",
        },
    )

    assert result.exit_code == 0
    stdout = Path(result.stdout_path).read_text(encoding="utf-8")
    for secret in (
        "sk-secret-deepseek",
        "sk-secret-openai",
        "ssh-agent-secret.sock",
        "cloud-secret-aws",
    ):
        assert secret not in stdout
    assert "MY_CUSTOM_VAR=visible-value" in stdout


def test_git_workspace_snapshot_detects_new_files(tmp_path) -> None:
    init_repo(tmp_path)
    repository = GitWorkspace(WorkspaceBoundary(grant(tmp_path)))
    baseline = repository.snapshot()
    (tmp_path / "new.py").write_text("value = 1\n", encoding="utf-8")

    assert repository.changed_paths_since(baseline) == ["new.py"]
    assert "new.py" in repository.diff_since(baseline)


def test_git_baseline_isolates_attempt_increment(tmp_path) -> None:
    init_repo(tmp_path)
    repository = GitWorkspace(WorkspaceBoundary(grant(tmp_path)))

    # Task A leaves tracked + untracked changes in the shared workspace.
    (tmp_path / "tracked.txt").write_text("changed-a\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    baseline = repository.snapshot()

    # Task B's increment is only tracked.txt (again) and the new b.py.
    (tmp_path / "tracked.txt").write_text("changed-b\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 1\n", encoding="utf-8")

    changed = repository.changed_paths_since(baseline)
    assert changed == ["b.py", "tracked.txt"]
    assert "a.py" not in changed

    diff = repository.diff_since(baseline)
    assert "changed-b" in diff
    assert "a.py" not in diff


def test_git_baseline_detects_untracked_file_content_change(tmp_path) -> None:
    init_repo(tmp_path)
    repository = GitWorkspace(WorkspaceBoundary(grant(tmp_path)))

    # Task A creates an untracked file.
    (tmp_path / "helper.py").write_text("H=1\n", encoding="utf-8")
    baseline = repository.snapshot()

    # Task B modifies that same untracked file (name unchanged, content changed).
    (tmp_path / "helper.py").write_text("H=2\n", encoding="utf-8")

    patch = repository.diff_since(baseline)
    changed = repository.changed_paths_since(baseline)
    assert changed == ["helper.py"]
    assert "-H=1" in patch
    assert "+H=2" in patch
    assert "new file mode" not in patch

    # The incremental patch must replay against Task A's workspace state.
    (tmp_path / "helper.py").write_text("H=1\n", encoding="utf-8")
    check = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=tmp_path,
        input=patch,
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr


def test_git_baseline_detects_untracked_file_deletion(tmp_path) -> None:
    init_repo(tmp_path)
    repository = GitWorkspace(WorkspaceBoundary(grant(tmp_path)))
    (tmp_path / "helper.py").write_text("H=1\n", encoding="utf-8")
    baseline = repository.snapshot()

    (tmp_path / "helper.py").unlink()

    assert repository.changed_paths_since(baseline) == ["helper.py"]
    assert repository.deleted_paths_since(baseline) == ["helper.py"]
    patch = repository.diff_since(baseline)
    assert "deleted file mode" in patch
    assert "-H=1" in patch


def test_symlink_cannot_bypass_denied_paths(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    init_repo(root)
    (root / "denied").mkdir()
    secret = root / "denied" / "secret.txt"
    secret.write_text("secret\n", encoding="utf-8")
    (root / "alias.txt").symlink_to(secret)
    boundary = WorkspaceBoundary(grant(root))

    with pytest.raises(WorkspacePermissionError):
        boundary.resolve_read_file("alias.txt")
    with pytest.raises(WorkspacePermissionError):
        boundary.resolve_write_file("alias.txt")


def test_registered_artifact_reader_verifies_hash(tmp_path) -> None:
    frozen = tmp_path / "artifact.txt"
    frozen.write_text("evidence", encoding="utf-8")
    artifact = ArtifactRef(
        id="artifact_input",
        kind="text",
        producer=AgentOwner.SCIENTIFIC,
        run_id="run_artifact",
        task_id="task_artifact",
        attempt_number=1,
        uri=frozen.as_uri(),
        sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        media_type="text/plain",
        summary="input",
    )
    reader = RegisteredArtifactReader([artifact], run_id=artifact.run_id)

    assert reader.read_text("artifact_input")["content"] == "evidence"
    frozen.write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactReadError, match="sha256"):
        reader.read_text("artifact_input")


def _agent_state() -> AgentState:
    now = datetime.now(UTC)
    return AgentState(
        session_id="session_test",
        agent_name="reader",
        owner=AgentOwner.EXECUTOR,
        run_id="run_test",
        task_id="task_test",
        attempt_number=1,
        created_at=now,
        updated_at=now,
    )


def test_git_diff_respects_denied_paths(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    init_repo(root)
    (root / "denied").mkdir()
    (root / "denied" / "secret.txt").write_text("secret content\n", encoding="utf-8")
    repository = GitWorkspace(WorkspaceBoundary(grant(root)))
    baseline = repository.snapshot()
    (root / "tracked.txt").write_text("changed\n", encoding="utf-8")

    changed = repository.changed_paths_since(baseline)
    assert "tracked.txt" in changed
    assert "denied/secret.txt" not in changed
    assert "secret content" not in repository.diff_since(baseline)


def test_resolve_system_write_is_separate_from_source_scope(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    boundary = WorkspaceBoundary(
        WorkspaceGrant(
            root=str(root),
            access=WorkspaceAccess(read_paths=["src"], write_paths=["src"]),
            source=WorkspaceSourceKind.LOCAL,
        )
    )
    resolved = boundary.resolve_system_write(".resagent2/runs/run_x/task_x/a_1/x.patch")
    assert resolved.is_relative_to(root)


def test_read_file_rejects_oversized_files(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "big.txt").write_text("x" * 2_000_000, encoding="utf-8")
    boundary = WorkspaceBoundary(grant(root))
    tool = ReadFileTool(boundary, max_bytes=1_000_000)
    with pytest.raises(ValueError, match="too large"):
        tool.execute(_agent_state(), tool.input_model(path="big.txt"))


def test_read_file_returns_line_range_metadata(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("l1\nl2\nl3\nl4\n", encoding="utf-8")
    boundary = WorkspaceBoundary(grant(root))
    tool = ReadFileTool(boundary)
    result = tool.execute(
        _agent_state(), tool.input_model(path="m.py", start_line=2, end_line=3)
    )
    assert result.value["path"] == "m.py"
    assert result.value["start_line"] == 2
    assert result.value["end_line"] == 3
    assert result.value["content"] == "l2\nl3\n"
    assert result.value["truncated"] is False


def test_read_file_unbounded_read_has_null_range(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("l1\n", encoding="utf-8")
    boundary = WorkspaceBoundary(grant(root))
    tool = ReadFileTool(boundary)
    result = tool.execute(_agent_state(), tool.input_model(path="m.py"))
    assert result.value["start_line"] is None
    assert result.value["end_line"] is None
    assert result.value["content"] == "l1\n"


def test_replace_text_preserves_leading_whitespace_in_old_text(tmp_path) -> None:
    # old_text is an exact needle: indentation must survive model validation.
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("class A:\n    def f(self):\n        return 1\n", encoding="utf-8")
    boundary = WorkspaceBoundary(grant(root))
    tool = ReplaceTextTool(boundary)

    args = tool.input_model(
        path="m.py",
        old_text="    def f(self):\n        return 1",
        new_text="    def f(self):\n        return 2",
    )
    assert args.old_text == "    def f(self):\n        return 1"

    tool.execute(_agent_state(), args)
    assert (root / "m.py").read_text(encoding="utf-8") == (
        "class A:\n    def f(self):\n        return 2\n"
    )


def test_replace_text_does_not_grow_indentation(tmp_path) -> None:
    # The historical bug stripped old_text's indentation, so the replacement
    # left the original spaces AND prepended new ones (4 -> 8). This guards it.
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("class A:\n    def f(self):\n        return 1\n", encoding="utf-8")
    tool = ReplaceTextTool(WorkspaceBoundary(grant(root)))

    tool.execute(
        _agent_state(),
        tool.input_model(
            path="m.py",
            old_text="    def f(self):\n        return 1",
            new_text="    def f(self):\n        return 2",
        ),
    )
    assert (root / "m.py").read_text(encoding="utf-8") == (
        "class A:\n    def f(self):\n        return 2\n"
    )


def test_replace_text_rejects_ambiguous_match_without_touching_file(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    original = "def f(self):\n    pass\ndef g(self):\n    pass\n"
    (root / "m.py").write_text(original, encoding="utf-8")
    tool = ReplaceTextTool(WorkspaceBoundary(grant(root)))

    with pytest.raises(ValueError, match="found 2"):
        tool.execute(
            _agent_state(),
            tool.input_model(
                path="m.py",
                old_text="    pass",
                new_text="    return 1",
            ),
        )
    # The file must be untouched after a rejected ambiguous replacement.
    assert (root / "m.py").read_text(encoding="utf-8") == original


def test_replace_text_rejects_empty_old_text(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("x = 1\n", encoding="utf-8")
    tool = ReplaceTextTool(WorkspaceBoundary(grant(root)))

    with pytest.raises(Exception, match="old_text must not be empty"):
        tool.input_model(path="m.py", old_text="", new_text="y")


@pytest.mark.parametrize(
    ("payload", "replaced"),
    [(b"opaque non-UTF-8: \xac\n", True), ("UTF-8 replacement character: \ufffd\n".encode("utf-8"), False)],
)
def test_git_diff_preview_marks_lossy_decoding_without_altering_patch(tmp_path, payload, replaced) -> None:
    from resagent2_capabilities import GitDiffInput, GitDiffTool

    init_repo(tmp_path)
    repository = GitWorkspace(WorkspaceBoundary(grant(tmp_path)))
    baseline = repository.snapshot()
    (tmp_path / "opaque.txt").write_bytes(payload)
    patch = repository.diff_bytes_since(baseline)
    observation = GitDiffTool(repository, baseline=baseline).execute(
        _agent_state(), GitDiffInput(max_chars=20_000),
    )
    assert observation.value["decoding_errors_replaced"] is replaced
    assert observation.value["encoding"] == "utf-8"
    assert observation.value["diff"] == patch.decode("utf-8", errors="replace")
    assert observation.value["truncated"] is False
    assert payload in patch

    limited = GitDiffTool(repository, baseline=baseline).execute(
        _agent_state(), GitDiffInput(max_chars=10),
    )
    assert limited.value["truncated"] is True
    assert limited.value["diff"] == observation.value["diff"][:10]
    assert repository.diff_bytes_since(baseline) == patch
