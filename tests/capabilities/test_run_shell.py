"""Unified shell execution preserves actual outcomes without managed environments."""

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from sciagent_capabilities import RunShellInput, RunShellTool
from sciagent_components import GitWorkspace, ProcessRunner, WorkspaceBoundary
from sciagent_components.workspace import ReadOnlyWorkspaceBoundary
from sciagent_contracts import AgentOwner, WorkspaceAccess, WorkspaceGrant
from sciagent_runtime import AgentState
from sciagent_runtime.cancellation import OperationCancelled


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="fixed Linux Bash execution")


def _state():
    now = datetime.now(UTC)
    return AgentState(
        session_id="session_shell", agent_name="coding", owner=AgentOwner.EXECUTOR,
        run_id="run_shell", task_id="task_shell", attempt_number=1,
        created_at=now, updated_at=now,
    )


def _boundary(root, access=None):
    return WorkspaceBoundary(WorkspaceGrant(
        root=str(root), source="local",
        access=access or WorkspaceAccess(read_paths=["."], write_paths=["."]),
    ))


def _tool(root, **kwargs):
    return RunShellTool(
        ProcessRunner(_boundary(root)), log_dir=str(root / ".sciagent" / "shell"),
        timeout_seconds=kwargs.pop("timeout_seconds", 3), **kwargs,
    )


def _repository(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "tracked.txt").write_text("before\n", encoding="utf-8")
    return GitWorkspace(_boundary(root))


def test_shell_input_preserves_script_whitespace():
    script = "\n  cat <<'EOF'\n  unchanged text\nEOF\n\n"
    assert RunShellInput(command=script).command == script


@pytest.mark.parametrize("command", ["", " \t\n", "echo bad\x00"])
def test_shell_input_rejects_empty_or_nul_commands(command):
    with pytest.raises(ValidationError):
        RunShellInput(command=command)


@pytest.mark.parametrize("access,allowed", [
    (WorkspaceAccess(read_paths=["."], write_paths=["."]), False),
    (WorkspaceAccess(read_paths=["."], write_paths=[]), True),
    (WorkspaceAccess(read_paths=["."], write_paths=["src"]), True),
    (WorkspaceAccess(read_paths=["."], write_paths=["."], denied_paths=["secret"]), True),
])
def test_shell_denies_before_execution(tmp_path, access, allowed):
    tool = RunShellTool(
        ProcessRunner(_boundary(tmp_path, access)),
        log_dir=str(tmp_path / ".sciagent"), timeout_seconds=3, allowed=allowed,
    )
    with pytest.raises(PermissionError):
        tool.execute(_state(), RunShellInput(command="touch should_not_exist"))
    assert not (tmp_path / "should_not_exist").exists()


def test_shell_runs_without_environment_binding_and_records_startup(tmp_path):
    tool = _tool(tmp_path, extra_env={"SHELL_TEST_INPUT": "explicit value"})
    startup = tool.approval_context()
    script = 'printf "%s\\n" "$SHELL_TEST_INPUT"\n'
    result = tool.execute(_state(), RunShellInput(command=script))
    assert result.ok
    assert result.value["command"] == script
    assert result.value["stdout_tail"] == "explicit value\n"
    assert {key: result.value[key] for key in startup} == startup
    assert "explicit value" not in str(startup)
    assert result.memory_updates == {}


def test_shell_state_is_not_preserved_between_calls(tmp_path, monkeypatch):
    monkeypatch.delenv("SCIAGENT_TEMP_STATE", raising=False)
    (tmp_path / "sub").mkdir()
    tool = _tool(tmp_path)
    assert tool.execute(_state(), RunShellInput(command="export SCIAGENT_TEMP_STATE=old; cd sub")).ok
    second = tool.execute(_state(), RunShellInput(
        command='printf "%s\\n" "${SCIAGENT_TEMP_STATE-unset}" "$PWD"',
    ))
    assert second.value["stdout_tail"].splitlines() == ["unset", str(tmp_path)]


def test_shell_does_not_overwrite_logs_when_state_is_unchanged(tmp_path):
    tool = _tool(tmp_path)
    first = tool.execute(_state(), RunShellInput(command="echo first"))
    second = tool.execute(_state(), RunShellInput(command="echo second"))
    assert first.value["stdout_path"] != second.value["stdout_path"]
    assert Path(first.value["stdout_path"]).read_text() == "first\n"
    assert Path(second.value["stdout_path"]).read_text() == "second\n"


@pytest.mark.parametrize("tail,timeout,expected_exit", [
    ("exit 7", 3, 7), ("sleep 2", 0.2, None),
])
def test_shell_records_coding_changes_after_failure_or_timeout(tmp_path, tail, timeout, expected_exit):
    repository = _repository(tmp_path)
    before = repository.snapshot()
    tool = _tool(tmp_path, repository=repository, timeout_seconds=timeout)
    script = "printf changed > tracked.txt\nprintf diagnostic >&2\n" + tail
    result = tool.execute(_state(), RunShellInput(command=script))
    assert not result.ok
    assert result.value["changed_paths"] == ["tracked.txt"]
    assert result.value["code_before"] == before.tree_hash
    assert result.value["code_after"] == repository.snapshot().tree_hash
    assert result.value["code_before"] != result.value["code_after"]
    assert result.value["stderr_tail"] == "diagnostic"
    assert result.value["command"] == script
    assert Path(result.value["stdout_path"]).is_file()
    if expected_exit is None:
        assert result.value["timed_out"] is True
    else:
        assert result.value["exit_code"] == expected_exit
        assert result.value["timed_out"] is False


def test_shell_read_records_unchanged_code(tmp_path):
    repository = _repository(tmp_path)
    result = _tool(tmp_path, repository=repository).execute(
        _state(), RunShellInput(command="cat tracked.txt"),
    )
    assert result.ok and result.value["changed_paths"] == []
    assert result.value["code_before"] == result.value["code_after"]
    assert result.value["stdout_tail"] == "before\n"


def test_shell_retains_receipt_when_change_diagnostic_fails(tmp_path, monkeypatch):
    repository = _repository(tmp_path)

    def failed_diagnostic(before):
        raise ValueError("cannot inspect repository after command")

    monkeypatch.setattr(repository, "changed_paths_since", failed_diagnostic)
    result = _tool(tmp_path, repository=repository).execute(
        _state(), RunShellInput(command="echo actual-output; exit 9"),
    )
    assert not result.ok and result.value["exit_code"] == 9
    assert result.value["stdout_tail"] == "actual-output\n"
    assert "cannot inspect repository" in result.value["workspace_change_error"]
    assert "changed_paths" not in result.value


def test_prepared_shell_refuses_changed_startup_before_side_effects(tmp_path):
    tool = _tool(tmp_path, extra_env={"SCIAGENT_LAUNCH_VALUE": "first"})
    approved = tool.prepare(RunShellInput(command="touch never"))
    tool.extra_env["SCIAGENT_LAUNCH_VALUE"] = "second"
    result = tool.execute_prepared(_state(), RunShellInput(command="touch never"), approved)
    assert not result.ok and result.value["execution_status"] == "not_executed"
    assert "changed after approval" in result.summary
    assert not (tmp_path / "never").exists()
    assert not (tmp_path / ".sciagent").exists()


def test_receipt_records_actual_startup_instead_of_an_earlier_observation(tmp_path, monkeypatch):
    tool = _tool(tmp_path, extra_env={"SCIAGENT_LAUNCH_VALUE": "before"})
    before = tool.approval_context()
    original_run = tool.runner.run_shell

    def changed_startup(script, **kwargs):
        kwargs["extra_env"]["SCIAGENT_LAUNCH_VALUE"] = "actual"
        return original_run(script, **kwargs)

    monkeypatch.setattr(tool.runner, "run_shell", changed_startup)
    result = tool.execute(_state(), RunShellInput(command='printf "%s" "$SCIAGENT_LAUNCH_VALUE"'))
    assert result.ok and result.value["stdout_tail"] == "actual"
    assert result.value["environment_fingerprint"] == tool.approval_context()["environment_fingerprint"]
    assert result.value["environment_fingerprint"] != before["environment_fingerprint"]


def test_prepared_shell_passes_the_checked_environment_unchanged_to_child(tmp_path, monkeypatch):
    monkeypatch.setenv("SCIAGENT_LAUNCH_VALUE", "checked")
    tool = _tool(tmp_path)
    approved = tool.prepare(RunShellInput(
        command='printf "%s" "$SCIAGENT_LAUNCH_VALUE"',
    ))
    original_run = tool.runner._run_argv

    def change_parent_after_check(*args, **kwargs):
        monkeypatch.setenv("SCIAGENT_LAUNCH_VALUE", "later-parent-value")
        return original_run(*args, **kwargs)

    monkeypatch.setattr(tool.runner, "_run_argv", change_parent_after_check)
    result = tool.execute_prepared(_state(), RunShellInput(
        command='printf "%s" "$SCIAGENT_LAUNCH_VALUE"',
    ), approved)
    assert result.ok and result.value["stdout_tail"] == "checked"
    assert result.value["environment_fingerprint"] == approved["environment_fingerprint"]


@pytest.mark.parametrize("stopped", [True, False])
def test_cancelled_git_diagnostic_preserves_stop_status_and_completed_script(tmp_path, monkeypatch, stopped):
    repository = _repository(tmp_path)

    def interrupted_diagnostic(before):
        raise OperationCancelled("interrupted Git check", stopped=stopped, details={
            "operation": "process", "command": ["git", "diff"],
            "unconfirmed_processes": [] if stopped else [123],
        })

    monkeypatch.setattr(repository, "changed_paths_since", interrupted_diagnostic)
    with pytest.raises(OperationCancelled) as caught:
        _tool(tmp_path, repository=repository).execute(
            _state(), RunShellInput(command="printf retained > actual-file; printf main-completed"),
        )
    assert caught.value.stopped is stopped
    assert caught.value.details["unconfirmed_processes"] == ([] if stopped else [123])
    completed = caught.value.details["completed_command"]
    assert completed["exit_code"] == 0 and completed["stdout_tail"] == "main-completed"
    assert (tmp_path / "actual-file").read_text() == "retained"


def test_shell_keeps_actual_result_when_new_filter_blocks_post_command_snapshot(tmp_path):
    repository = _repository(tmp_path)
    script = (
        "git config filter.later.clean 'printf ran > filter-ran; cat'\n"
        "printf 'tracked.txt filter=later\\n' > .gitattributes\n"
        "printf actual-command-output\nexit 7"
    )
    result = _tool(tmp_path, repository=repository).execute(
        _state(), RunShellInput(command=script),
    )
    assert not result.ok
    assert result.value["exit_code"] == 7
    assert result.value["stdout_tail"] == "actual-command-output"
    assert "external clean/process" in result.value["workspace_change_error"]
    assert Path(result.value["stdout_path"]).read_text() == "actual-command-output"
    assert not (tmp_path / "filter-ran").exists()
    assert "changed_paths" not in result.value



def test_shell_receipt_has_complete_short_output_and_explicit_access_facts(tmp_path):
    result = _tool(tmp_path).execute(_state(), RunShellInput(command="printf 'out'; printf 'err' >&2"))
    assert result.ok
    for stream, expected in (("stdout", "out"), ("stderr", "err")):
        output = result.value[f"{stream}_output"]
        assert output["path"] == result.value[f"{stream}_path"]
        assert output["content"] == expected
        assert output["total_bytes"] == len(expected)
        assert output["truncated"] is False
        assert output["text_status"] == "supported"
        # Default logs are reserved; their existence does not grant read access.
        assert output["read_access"] == "not_authorized"
        assert "read_path" not in output


def test_shell_receipt_retains_both_ends_with_configurable_preview_limit(tmp_path):
    text = "START-" + "x" * 100 + "-END"
    tool = _tool(tmp_path, output_preview_chars=20)
    result = tool.execute(_state(), RunShellInput(command=f"printf '{text}'; exit 7"))
    output = result.value["stdout_output"]
    assert not result.ok and result.value["exit_code"] == 7
    assert output["preview_kind"] == "head_tail"
    assert output["head"] == text[:10] and output["tail"] == text[-10:]
    assert output["preview_chars"] == 20
    assert output["total_chars"] == len(text)
    assert output["omitted_chars"] == len(text) - 20
    assert output["truncated"] is True
    assert Path(output["path"]).read_text() == text
    assert result.value["stdout_tail"] == text


def test_shell_reports_only_its_explicit_log_read_grant(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    logs = tmp_path / "own-command-logs"
    read_boundary = ReadOnlyWorkspaceBoundary([WorkspaceGrant(
        root=str(logs), source="local", access=WorkspaceAccess(read_paths=["."], write_paths=[]),
    )])
    tool = RunShellTool(
        ProcessRunner(_boundary(workspace)), timeout_seconds=3,
        log_dir=str(logs), log_read_boundary=read_boundary,
    )
    result = tool.execute(_state(), RunShellInput(command="printf retained"))
    output = result.value["stdout_output"]
    assert output["read_access"] == "authorized"
    assert output["read_path"] == output["path"]
    assert read_boundary.resolve_read_file(output["read_path"]).read_text() == "retained"
    with pytest.raises(PermissionError):
        read_boundary.resolve_read_file(str(tmp_path / "unrelated.log"))


@pytest.mark.parametrize("raw,status", [(b"\xff", "invalid_utf8"), (b"a\x00b", "contains_nul")])
def test_shell_preserves_raw_nontext_output_and_reports_limit(tmp_path, raw, status):
    source = tmp_path / "raw-source"
    source.write_bytes(raw)
    result = _tool(tmp_path).execute(_state(), RunShellInput(command="cat raw-source"))
    output = result.value["stdout_output"]
    assert result.ok
    assert output["text_status"] == status
    assert output["total_bytes"] == len(raw)
    assert Path(output["path"]).read_bytes() == raw


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_shell_rejects_invalid_output_preview_configuration(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        _tool(tmp_path, output_preview_chars=limit)



def test_default_shell_output_budget_previews_beginning_and_end(tmp_path):
    result = _tool(tmp_path).execute(_state(), RunShellInput(
        command="printf BEGIN; printf '%012100d' 0; printf END",
    ))
    output = result.value["stdout_output"]
    assert result.ok
    assert output["preview_limit_chars"] == 12_000
    assert output["preview_chars"] == 12_000
    assert len(output["head"]) == len(output["tail"]) == 6_000
    assert output["head"].startswith("BEGIN") and output["tail"].endswith("END")
    assert output["total_bytes"] == output["total_chars"] == 12_108
    assert output["omitted_chars"] == 108
    assert output["text_status"] == "supported"
    assert len(result.value["stdout_tail"]) == 2_000
