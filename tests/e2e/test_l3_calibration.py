"""Offline L3 preparation checks; no CIFAR download, Torch, training or model calls."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from e2e.l3_calibration import prepare as l3


DATA_FILES = [*(f"data_batch_{index}" for index in range(1, 6)), "test_batch", "batches.meta"]
BUDGETS = {"max_tasks": 12, "max_attempts": 2, "max_llm_calls": 200, "timeout_seconds": 14400}


def fingerprint(path):
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    """Deliberately non-pickle bytes ensure verification needs no data deserialization."""
    root = tmp_path / "datasets" / "cifar10"
    batches = root / "cifar-10-batches-py"
    batches.mkdir(parents=True)
    expected = {}
    for name in DATA_FILES:
        path = batches / name
        path.write_bytes(f"not a pickle: {name}\n".encode())
        expected[name] = fingerprint(path)
    manifest = tmp_path / "tiny-hashes.json"
    manifest.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(l3, "DATA_MANIFEST", manifest)
    return root, expected


def test_verify_dataset_checks_all_seven_raw_files(tiny_dataset):
    root, expected = tiny_dataset
    assert l3.verify_dataset(root) == expected


@pytest.mark.parametrize("damage", ["same_size_corruption", "wrong_size", "missing"])
def test_verify_dataset_rejects_changed_or_missing_bytes(tiny_dataset, damage):
    root, _ = tiny_dataset
    path = root / "cifar-10-batches-py" / "test_batch"
    if damage == "same_size_corruption":
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    elif damage == "wrong_size":
        path.write_bytes(path.read_bytes() + b"extra")
    else:
        path.unlink()
    with pytest.raises(ValueError):
        l3.verify_dataset(root)


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    dataset = tmp_path / "datasets" / "cifar10"
    dataset.mkdir(parents=True)
    catalog = dataset.parent / "catalog.json"
    catalog.write_text(json.dumps({"cifar10": "cifar10", "unrelated": "other-dataset"}), encoding="utf-8")
    verified = {name: {"sha256": "a" * 64, "bytes": index + 1} for index, name in enumerate(DATA_FILES)}
    monkeypatch.setattr(l3, "verify_dataset", lambda path: verified)
    return dataset, catalog, verified


def test_prepare_rejects_existing_root_without_touching_contents(tmp_path, prepared_inputs):
    dataset, catalog, _ = prepared_inputs
    root = tmp_path / "existing-run"
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_bytes(b"existing experiment must survive")
    before = catalog.read_bytes()
    with pytest.raises(ValueError):
        l3.prepare_workspace(root, dataset)
    assert list(root.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"existing experiment must survive"
    assert catalog.read_bytes() == before


def test_prepare_rejects_dangling_symlink_root(tmp_path, prepared_inputs):
    dataset, catalog, _ = prepared_inputs
    root = tmp_path / "symlink-run"
    missing_target = tmp_path / "missing-target"
    root.symlink_to(missing_target, target_is_directory=True)
    original_catalog = catalog.read_bytes()
    with pytest.raises(ValueError):
        l3.prepare_workspace(root, dataset)
    assert root.is_symlink()
    assert root.readlink() == missing_target
    assert not missing_target.exists()
    assert catalog.read_bytes() == original_catalog


@pytest.mark.parametrize("location", ["inside_dataset", "contains_dataset", "inside_source"])
def test_prepare_rejects_overlapping_data_or_product_paths(tmp_path, prepared_inputs, location):
    dataset, catalog, _ = prepared_inputs
    if location == "inside_dataset":
        root = dataset / "forbidden-run"
    elif location == "contains_dataset":
        root = dataset.parent
    else:
        root = Path(l3.__file__).resolve().parent / "forbidden-test-run"
    assert not root.exists() or location == "contains_dataset"
    before = catalog.read_bytes()
    with pytest.raises(ValueError):
        l3.prepare_workspace(root, dataset)
    assert catalog.read_bytes() == before
    if location != "contains_dataset":
        assert not root.exists()


@pytest.mark.parametrize("catalog_state", ["missing", "wrong_dataset"])
def test_prepare_does_not_create_or_rewrite_shared_catalog(tmp_path, prepared_inputs, catalog_state):
    dataset, catalog, _ = prepared_inputs
    if catalog_state == "missing":
        catalog.unlink()
        before = None
    else:
        catalog.write_text(json.dumps({"cifar10": "different-location"}), encoding="utf-8")
        before = catalog.read_bytes()
    root = tmp_path / "rejected-run"
    with pytest.raises(ValueError):
        l3.prepare_workspace(root, dataset)
    assert not root.exists()
    assert (catalog.read_bytes() if catalog.exists() else None) == before


def test_prepare_records_reproducible_neutral_workspace(tmp_path, prepared_inputs, monkeypatch):
    dataset, catalog, verified = prepared_inputs
    root = tmp_path / "fresh-run"
    original_catalog = catalog.read_bytes()
    monkeypatch.setenv("RESAGENT2_MODEL", "offline-fixture-model")
    manifest = l3.prepare_workspace(root, dataset)
    workspace = root / "workspace"
    source = Path(l3.__file__).resolve().parent

    assert manifest == json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_id"] == "l3-calibration-public-v1"
    assert manifest["budgets"] == BUDGETS
    assert Path(manifest["workspace"]).resolve() == workspace.resolve()
    assert Path(manifest["dataset_root"]).resolve() == dataset.resolve()
    assert manifest["dataset_files"] == verified
    assert manifest["prepared_with_python"] == str(Path(sys.executable).absolute())
    assert (root / "python-path.txt").read_text(encoding="utf-8").strip() == manifest["prepared_with_python"]
    assert manifest["preparation_performed_training_or_model_calls"] is False
    assert catalog.read_bytes() == original_catalog
    assert (root / "goal.txt").read_bytes() == (source / "goal.txt").read_bytes()
    assert manifest["goal_sha256"] == fingerprint(root / "goal.txt")["sha256"]

    baseline = source / "baseline"
    baseline_files = {path.relative_to(baseline).as_posix() for path in baseline.rglob("*") if path.is_file()}
    assert baseline_files <= manifest["input_files"].keys()
    for relative in baseline_files:
        assert (workspace / relative).read_bytes() == (baseline / relative).read_bytes()
    for relative, expected in manifest["input_files"].items():
        assert fingerprint(workspace / relative) == expected
    assert fingerprint(workspace / "resources/split.json") == manifest["input_files"]["resources/split.json"]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(workspace), *args]).decode().strip()

    assert git("rev-parse", "HEAD") == manifest["workspace_commit"]
    assert git("rev-list", "--count", "HEAD") == "1"
    assert git("status", "--porcelain") == ""
    assert git("show", "-s", "--format=%an|%ae|%cn|%ce", "HEAD") == (
        "L3 fixture|l3@example.invalid|L3 fixture|l3@example.invalid"
    )
    tracked = set(git("ls-files").splitlines())
    assert tracked == baseline_files | {"README.md", ".gitignore"}
    assert "goal.txt" not in tracked
    for relative in manifest["input_files"]:
        committed = subprocess.check_output(["git", "-C", str(workspace), "show", f"HEAD:{relative}"])
        assert hashlib.sha256(committed).hexdigest() == manifest["input_files"][relative]["sha256"]
    assert not any(path.suffix in {".pt", ".pth", ".npz", ".csv"} for path in workspace.rglob("*"))

    # Source only the generated assignment file, never the training entrypoint.
    script = 'source "$1"; printf "%s\\n" "$RESAGENT2_MODEL" "$RESAGENT2_DATASET_ROOT" "$RESAGENT2_DATASETS_JSON"'
    config = subprocess.check_output(["bash", "--noprofile", "--norc", "-c", script, "l3-test", str(root / "run-config.sh")])
    model, shared_root, mapping = config.decode().splitlines()
    assert model == "offline-fixture-model"
    assert Path(shared_root).resolve() == dataset.parent.resolve()
    assert json.loads(mapping) == {"cifar10": str(dataset.resolve())}


def test_prepare_copies_only_baseline_allowlist(tmp_path, prepared_inputs, monkeypatch):
    dataset, _, _ = prepared_inputs
    baseline = tmp_path / "baseline-with-local-files"
    shutil.copytree(l3.BASELINE, baseline)
    junk = {".env", "private-notes.txt", "__pycache__/main.cpython-312.pyc", "checkpoint/old.pth"}
    for relative in junk:
        path = baseline / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"UNRELATED_LOCAL_FILE_MUST_NOT_BE_COPIED")
    monkeypatch.setattr(l3, "BASELINE", baseline)
    root = tmp_path / "allowlisted-run"
    manifest = l3.prepare_workspace(root, dataset)
    for relative in junk:
        assert not (root / "workspace" / relative).exists()
        assert relative not in manifest["input_files"]
    assert (root / "workspace/main.py").read_bytes() == (baseline / "main.py").read_bytes()


@pytest.fixture
def fake_launcher(tmp_path, monkeypatch, prepared_inputs):
    """The executable records arguments instead of importing any production CLI."""
    root = tmp_path / "run root with spaces"
    dataset, _, _ = prepared_inputs
    monkeypatch.setenv("RESAGENT2_MODEL", "frozen fixture model")
    l3.prepare_workspace(root, dataset)
    interpreter = tmp_path / "fake python"
    capture = tmp_path / "invocation.bin"
    interpreter.write_text(
        '#!/bin/bash\n'
        'if [ "${1-}" = "$L3_TEST_PREPARE" ]; then exec "$L3_TEST_REAL_PYTHON" "$@"; fi\n'
        'printf "%s\\0" "$PWD" "$RESAGENT2_MODEL" "$@" > "$L3_TEST_CAPTURE"\n'
        'exit "$L3_TEST_EXIT"\n',
        encoding="utf-8",
    )
    interpreter.chmod(0o700)
    (root / "python-path.txt").write_text(str(interpreter) + "\n", encoding="utf-8")
    monkeypatch.setenv("L3_TEST_CAPTURE", str(capture))
    monkeypatch.setenv("L3_TEST_PREPARE", str(Path(l3.__file__).resolve()))
    monkeypatch.setenv("L3_TEST_REAL_PYTHON", sys.executable)
    script = Path(l3.__file__).resolve().parent / "run.sh"

    def invoke(command, *arguments, exit_code):
        monkeypatch.setenv("L3_TEST_EXIT", str(exit_code))
        result = subprocess.run(
            ["bash", str(script), str(root), command, "run_public_l3", *arguments],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert capture.exists(), result.stderr
        fields = capture.read_bytes().decode().split("\0")
        assert fields.pop() == ""
        assert Path(fields[0]) == root
        assert fields[1] == "frozen fixture model"
        return result, fields[2:]

    return root, invoke


def test_run_launcher_preserves_protocol_arguments_and_failure_exit(fake_launcher):
    root, invoke = fake_launcher
    result, argv = invoke("run", exit_code=7)
    assert result.returncode == 7
    assert argv[:3] == ["-m", "resagent2_cli", "run"]
    expected = {
        "--run-id": "run_public_l3",
        "--workspace": str(root / "workspace"),
        "--data-root": str(root / "data"),
        "--goal-file": str(root / "goal.txt"),
        "--required-artifact": "comparison_results",
        "--approval-mode": "auto_review",
        "--max-tasks": "12",
        "--max-attempts": "2",
        "--max-llm-calls": "200",
        "--timeout-seconds": "14400",
    }
    for flag, value in expected.items():
        assert argv.count(flag) == 1
        assert argv[argv.index(flag) + 1] == value
    assert argv.count("--constraint") == 2
    assert "--execute-commands" in argv
    assert (root / ".started/run-id.txt").read_text(encoding="utf-8") == "run_public_l3\n"


def test_answer_launcher_forwards_fields_with_spaces_and_exit_code(fake_launcher):
    root, invoke = fake_launcher
    answer = ["--field", "actual_answer=two words = preserved", "--comment", "checked current operation"]
    result, argv = invoke("answer", *answer, exit_code=3)
    assert result.returncode == 3
    assert argv == [
        "-m", "resagent2_cli", "answer", "run_public_l3", "--data-root", str(root / "data"), *answer,
    ]


def test_resume_launcher_reuses_original_root_and_failure_exit(fake_launcher):
    root, invoke = fake_launcher
    result, argv = invoke("resume", exit_code=7)
    assert result.returncode == 7
    assert argv == ["-m", "resagent2_cli", "resume", "run_public_l3", "--data-root", str(root / "data")]


@pytest.mark.parametrize("changed_input", ["goal", "ignored_checkpoint"])
def test_claim_rejects_modified_inputs_without_consuming_root(tmp_path, prepared_inputs, changed_input):
    dataset, _, _ = prepared_inputs
    root = tmp_path / "modified-input-run"
    l3.prepare_workspace(root, dataset)
    if changed_input == "goal":
        goal = root / "goal.txt"
        goal.write_bytes(goal.read_bytes() + b"\nChanged research goal\n")
    else:
        checkpoint = root / "workspace/checkpoint/old.pth"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"PREVIOUS_RUN_CHECKPOINT")
        assert subprocess.check_output(
            ["git", "-C", str(root / "workspace"), "status", "--porcelain"],
        ) == b""
    with pytest.raises(ValueError):
        l3.claim_start(root, "run_changed")
    assert not (root / ".started").exists()
    assert not (root / "data").exists()


def test_claim_rejects_second_run_id_and_preserves_first_claim(tmp_path, prepared_inputs):
    dataset, _, _ = prepared_inputs
    root = tmp_path / "single-use-run"
    l3.prepare_workspace(root, dataset)
    l3.claim_start(root, "run_first")
    marker = root / ".started/run-id.txt"
    assert marker.read_bytes() == b"run_first\n"
    with pytest.raises(ValueError):
        l3.claim_start(root, "run_different")
    assert marker.read_bytes() == b"run_first\n"
    assert not (root / "data").exists()
