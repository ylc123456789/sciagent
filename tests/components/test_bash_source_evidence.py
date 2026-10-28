"""Regression contracts for source evidence from actual compound Bash commands.

These tests inspect authorized fixture bytes only. The collector must neither
execute the command nor treat successful syntax parsing as an approval verdict.
"""
import hashlib
from pathlib import Path

import pytest

from sciagent_components.python_source_evidence import collect_python_source_evidence
from sciagent_components.workspace import WorkspaceBoundary
from sciagent_contracts import WorkspaceAccess, WorkspaceGrant


REAL_FAILURE_COMMAND = '''set -o pipefail
echo "=== python -V ==="
python3 -V
echo "=== sha256 BEFORE/AFTER inputs ==="
sha256sum predictions.json measure.py
echo "=== run tests ==="
python3 -m unittest -v test_metric 2>&1; echo "UNITTEST_EXIT=$?"'''


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "test_metric.py").write_text("from metric import mse\nassert mse([0], [1]) == 1\n")
    (tmp_path / "metric.py").write_text(
        "SOURCE_ID = 'ORIGINAL_METRIC_SOURCE'\n"
        "def mse(actual, predicted):\n"
        "    return sum((a-b)**2 for a,b in zip(actual,predicted))/len(actual)\n"
    )
    return tmp_path


def collect(root, command):
    grant = WorkspaceGrant(root=str(root), source="local", access=WorkspaceAccess(
        read_paths=["."], write_paths=[],
    ))
    return collect_python_source_evidence(command, WorkspaceBoundary(grant))


def sources(evidence):
    return {Path(item["path"]).name: item for item in evidence.review_payload()["files"]}


def gaps(evidence):
    return evidence.review_payload()["gaps"]


def test_real_paused_command_exposes_test_and_imported_implementation(workspace):
    before = {path.name: path.read_bytes() for path in workspace.iterdir()}
    evidence = collect(workspace, REAL_FAILURE_COMMAND)
    actual = sources(evidence)
    assert set(actual) == {"test_metric.py", "metric.py"}
    for name, source in actual.items():
        assert source["content"] == before[name].decode("utf-8")
        assert source["sha256"] == hashlib.sha256(before[name]).hexdigest()
    assert {path.name: path.read_bytes() for path in workspace.iterdir()} == before
    assert "decision" not in evidence.review_payload()


@pytest.mark.parametrize("command", [
    "echo start; python3 -m unittest -v test_metric; echo done",
    "echo start\npython3 -m unittest -v test_metric\necho done",
    "python3 -m unittest -v test_metric 2>&1",
    'python3 -m unittest -v test_metric 2>&1; echo "UNITTEST_EXIT=$?"',
    "python3 -V && python3 -m unittest -v test_metric || echo failed",
])
def test_shell_scaffolding_does_not_erase_literal_python_sources(workspace, command):
    evidence = collect(workspace, command)
    assert set(sources(evidence)) == {"test_metric.py", "metric.py"}
    assert evidence.review_payload()["coverage"]
    assert "decision" not in evidence.review_payload()


def test_quoted_and_commented_python_text_is_not_an_executable_entrypoint(workspace):
    (workspace / "decoy.py").write_text("raise AssertionError('must not be selected')\n")
    command = (
        "echo 'python3 decoy.py; python3 $(echo decoy.py)'\n"
        "# python3 decoy.py\n"
        "python3 -m unittest -v test_metric"
    )
    evidence = collect(workspace, command)
    assert set(sources(evidence)) == {"test_metric.py", "metric.py"}
    assert "decoy.py" not in sources(evidence)


def test_quoted_adjacent_words_keep_the_actual_literal_filename(workspace):
    evidence = collect(workspace, "python3 'test_'metric.py")
    assert set(sources(evidence)) == {"test_metric.py", "metric.py"}


@pytest.mark.parametrize("command,wrong_entrypoint", [
    ("python3 'met'\\\nric.py", "met"),
    ("python\\\n3 metric.py", "3"),
])
def test_backslash_newline_joining_never_selects_a_partial_literal_entrypoint(
    workspace, command, wrong_entrypoint,
):
    (workspace / wrong_entrypoint).write_text("PARTIAL_WORD_IS_NOT_THE_ENTRYPOINT = True\n")
    evidence = collect(workspace, command)
    actual = sources(evidence)
    assert wrong_entrypoint not in actual
    assert set(actual) <= {"metric.py"}
    assert not any("PARTIAL_WORD_IS_NOT_THE_ENTRYPOINT" in file["content"]
                   for file in evidence.review_payload()["files"])
    if not actual:
        assert gaps(evidence)


@pytest.mark.parametrize("name", ["semi;colon.py", "space name.py", "literal$entry.py", "测量.py"])
def test_quoted_script_path_is_data_and_preserves_its_actual_bytes(workspace, name):
    (workspace / name).write_text("ACTUAL_LITERAL_SOURCE = True\n")
    evidence = collect(workspace, f"python3 '{name}'")
    assert set(sources(evidence)) == {name}
    assert sources(evidence)[name]["content"] == "ACTUAL_LITERAL_SOURCE = True\n"


def test_unrelated_log_redirect_is_reported_without_executing_or_erasing_python_sources(workspace):
    evidence = collect(workspace, "python3 -m unittest -v test_metric > results.log")
    assert set(sources(evidence)) == {"test_metric.py", "metric.py"}
    assert not (workspace / "results.log").exists()
    assert any(item.get("redirections") for item in evidence.review_payload()["coverage"])


@pytest.mark.parametrize("command", [
    "python3 2>&1 -m unittest -v test_metric",
    "python3 > run.log -m unittest -v test_metric",
    "python3 -m > run.log unittest -v test_metric",
    "python3 -m unittest -v > run.log test_metric",
])
def test_redirect_inside_argv_preserves_python_arguments_and_excludes_destination(workspace, command):
    evidence = collect(workspace, command)
    assert set(sources(evidence)) == {"test_metric.py", "metric.py"}
    coverage = evidence.review_payload()["coverage"]
    assert len(coverage) == 1
    assert coverage[0]["argv"] == ["python3", "-m", "unittest", "-v", "test_metric"]
    assert coverage[0]["redirections"]
    assert not (workspace / "run.log").exists()


def test_unresolved_shell_entrypoint_never_reads_a_same_named_literal_file(workspace):
    (workspace / "$ENTRY").write_text("THIS_IS_NOT_THE_EXPANDED_ENTRYPOINT = True\n")
    evidence = collect(workspace, 'python3 "$ENTRY"')
    assert not sources(evidence)
    assert gaps(evidence)


def test_syntax_analysis_never_runs_substitution_or_generated_python(workspace):
    marker = workspace / "executed.marker"
    (workspace / "entry.py").write_text(
        "from pathlib import Path\n"
        "Path('executed.marker').write_text('ran')\n"
    )
    evidence = collect(workspace, 'echo "$(touch executed.marker)"; python3 entry.py')
    assert not marker.exists()
    assert gaps(evidence)
    assert "decision" not in evidence.review_payload()


@pytest.mark.parametrize("command", [
    "python3 -m unittest -v test_metric &&",
    "python3 -m unittest -v test_metric; echo 'unterminated",
    "if true; then python3 -m unittest -v test_metric",
])
def test_invalid_whole_command_is_not_presented_as_complete_source_evidence(workspace, command):
    evidence = collect(workspace, command)
    assert gaps(evidence)
    assert not sources(evidence)
    assert "decision" not in evidence.review_payload()


@pytest.mark.parametrize("prefix", [
    "cd elsewhere; ",
    "export PYTHONPATH=elsewhere; ",
    "PYTHONPATH=elsewhere ",
    "PATH=elsewhere ",
    "alias python3='echo'; ",
    "source environment.sh; ",
    ". environment.sh; ",
    "eval 'true'; ",
    "python3() { :; }; ",
])
def test_prior_shell_context_changes_do_not_claim_original_cwd_source(workspace, prefix):
    (workspace / "elsewhere").mkdir()
    (workspace / "environment.sh").write_text("export PYTHONPATH=elsewhere\n")
    evidence = collect(workspace, prefix + "python3 -m unittest -v test_metric")
    assert gaps(evidence)
    assert not any("ORIGINAL_METRIC_SOURCE" in file["content"]
                   for file in evidence.review_payload()["files"])


@pytest.mark.parametrize("command,written_source", [
    ("printf replacement > metric.py; python3 -m unittest -v test_metric", "metric.py"),
    (": > metric.py; python3 -m unittest -v test_metric", "metric.py"),
    ("python3 -m unittest -v test_metric > metric.py", "metric.py"),
    ("printf replacement > test_metric.py; python3 -m unittest -v test_metric", "test_metric.py"),
])
def test_prior_or_own_redirection_does_not_prove_the_old_executed_source(workspace, command, written_source):
    before = {path.name: path.read_bytes() for path in workspace.iterdir()}
    evidence = collect(workspace, command)
    assert gaps(evidence)
    assert written_source not in sources(evidence)
    assert not any("ORIGINAL_METRIC_SOURCE" in file["content"]
                   for file in evidence.review_payload()["files"])
    assert {path.name: path.read_bytes() for path in workspace.iterdir()} == before


def test_unknown_later_command_keeps_already_identified_source_with_visible_gap(workspace):
    evidence = collect(workspace, "python3 -m unittest -v test_metric; unknown_external_program")
    assert set(sources(evidence)) == {"test_metric.py", "metric.py"}
    assert gaps(evidence)
    assert "decision" not in evidence.review_payload()


def test_compound_fingerprint_tracks_dependency_bytes_and_returns_independent_payload(workspace):
    first = collect(workspace, REAL_FAILURE_COMMAND)
    assert first.fingerprint == collect(workspace, REAL_FAILURE_COMMAND).fingerprint
    metric = workspace / "metric.py"
    metric.write_text(metric.read_text() + "\n# changed dependency bytes\n")
    second = collect(workspace, REAL_FAILURE_COMMAND)
    assert first.fingerprint != second.fingerprint
    display = second.review_payload()
    display["files"].clear()
    assert set(sources(second)) == {"test_metric.py", "metric.py"}


@pytest.mark.parametrize("budget,value,command", [
    ("MAX_COMMAND_BYTES", 64, "echo '" + "测" * 20 + "'; python3 test_metric.py"),
    ("MAX_AST_NODES", 8, "echo a; echo b; echo c; python3 test_metric.py"),
    ("MAX_AST_DEPTH", 3, "((((python3 test_metric.py))))"),
])
def test_structure_budget_exhaustion_is_explicit_and_does_not_claim_partial_tree_complete(
    workspace, monkeypatch, budget, value, command,
):
    from sciagent_components import _bash_source_structure

    monkeypatch.setattr(_bash_source_structure, budget, value)
    evidence = collect(workspace, command)
    assert not sources(evidence)
    assert any("limit" in gap["code"] for gap in gaps(evidence))
    assert "decision" not in evidence.review_payload()


def test_actual_second_paused_command_with_literal_current_cwd_retains_sources(workspace):
    # Exact v1 server command, with only its cwd replaced by this granted fixture.
    command = (
        f"cd {workspace} && python3 -m unittest -v test_metric 2>&1; "
        'echo "EXIT_CODE=$?"; echo "--- checksums ---"; '
        'sha256sum predictions.json measure.py metric.py test_metric.py; '
        'echo "--- git ---"; git status --porcelain'
    )
    before = {path.name: path.read_bytes() for path in workspace.iterdir()}
    evidence = collect(workspace, command)
    assert set(sources(evidence)) == {"test_metric.py", "metric.py"}
    assert {path.name: path.read_bytes() for path in workspace.iterdir()} == before
    assert "decision" not in evidence.review_payload()


@pytest.mark.parametrize("redirect", ["2>&1", "> results.log"])
def test_current_cwd_list_trailing_redirect_belongs_to_python_only(workspace, redirect):
    evidence = collect(workspace, f"cd {workspace} && python3 -m unittest -v test_metric {redirect}")
    assert set(sources(evidence)) == {"test_metric.py", "metric.py"}
    coverage = evidence.review_payload()["coverage"]
    assert len(coverage) == 2
    assert coverage[0]["argv"] == ["cd", str(workspace)]
    assert coverage[0]["redirections"] == []
    assert coverage[1]["argv"] == ["python3", "-m", "unittest", "-v", "test_metric"]
    assert len(coverage[1]["redirections"]) == 1
    assert not (workspace / "results.log").exists()


def test_current_cwd_list_source_write_redirect_does_not_reuse_old_dependency(workspace):
    evidence = collect(workspace, f"cd {workspace} && python3 -m unittest -v test_metric > metric.py")
    assert "metric.py" not in sources(evidence)
    assert gaps(evidence)
    coverage = evidence.review_payload()["coverage"]
    assert len(coverage) == 2 and coverage[0]["redirections"] == []
    assert any(redirect["kind"] == "file_write" and redirect["destination"] == "metric.py"
               for redirect in coverage[1]["redirections"])


@pytest.mark.parametrize("cd_command", [
    "cd elsewhere",
    "cd -P {cwd}",
    "cd -- {cwd}",
    'cd "$PWD"',
    'cd "$' '{PWD}"',
    'cd "$(pwd)"',
])
def test_current_cwd_noop_exception_does_not_extend_to_other_options_or_expansions(workspace, cd_command):
    (workspace / "elsewhere").mkdir()
    command = cd_command.replace("{cwd}", str(workspace)) + " && python3 -m unittest -v test_metric"
    evidence = collect(workspace, command)
    assert gaps(evidence)
    assert not any("ORIGINAL_METRIC_SOURCE" in file["content"]
                   for file in evidence.review_payload()["files"])


@pytest.mark.parametrize("prefix", ["unknown_external_program; ", "export PYTHONPATH=elsewhere; "])
def test_literal_current_cwd_cd_does_not_reset_preceding_context_uncertainty(workspace, prefix):
    command = prefix + f"cd {workspace} && python3 -m unittest -v test_metric"
    evidence = collect(workspace, command)
    assert gaps(evidence)
    assert not any("ORIGINAL_METRIC_SOURCE" in file["content"]
                   for file in evidence.review_payload()["files"])
