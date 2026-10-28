import pytest

from resagent2_orchestrator import RunLayout


def test_run_layout_resolves_standard_directories(tmp_path) -> None:
    layout = RunLayout(tmp_path / "data")

    assert layout.run_dir("run_x") == tmp_path / "data" / "runs" / "run_x"
    assert layout.workspace_repo_dir("run_x", "ws_a") == (
        tmp_path / "data" / "runs" / "run_x" / "workspaces" / "ws_a" / "repo"
    )
    assert layout.workspace_meta_path("run_x", "ws_a") == (
        tmp_path / "data" / "runs" / "run_x" / "workspaces" / "ws_a" / "workspace.json"
    )
    assert layout.attempt_dir("run_x", "task_a", 2) == (
        tmp_path / "data" / "runs" / "run_x" / "attempts" / "task_a" / "attempt_2"
    )
    assert layout.artifacts_dir("run_x") == (
        tmp_path / "data" / "runs" / "run_x" / "artifacts"
    )


def test_run_layout_reads_data_root_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RESAGENT2_DATA_ROOT", str(tmp_path / "custom"))
    assert RunLayout.from_env().data_root == tmp_path / "custom"


def test_run_layout_workspace_dir_rejects_escape(tmp_path) -> None:
    layout = RunLayout(tmp_path / "data")
    with pytest.raises(ValueError, match="escapes"):
        layout.workspace_dir("run_x", "../escape")
