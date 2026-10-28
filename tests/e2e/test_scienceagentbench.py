"""SAB bridge tests use native production agents; only model transport is scripted."""
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx
import pytest

from e2e import scienceagentbench as sab


GOLD_CANARY = "PRIVATE_GOLD_DO_NOT_IMPORT"
KNOWLEDGE_CANARY = "KNOWLEDGE_NOT_IN_BASE"
PROGRAM = (
    "# Native coding mechanism fixture; not a benchmark result.\n"
    "import json\n"
    "from pathlib import Path\n"
    "number = int(Path('benchmark/datasets/toy/input.txt').read_text())\n"
    "result = Path('pred_results/toy.json')\n"
    "result.parent.mkdir(parents=True, exist_ok=True)\n"
    "result.write_text(json.dumps({'answer': number + 5}) + '\\n')\n"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    benchmark = tmp_path / "private-benchmark"
    for relative, text in {
        "datasets/toy/input.txt": "37\n",
        "datasets/unselected/private.txt": GOLD_CANARY,
        "gold_programs/toy.py": GOLD_CANARY,
        "eval_programs/toy_eval.py": GOLD_CANARY,
    }.items():
        path = benchmark / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    row = {
        "instance_id": 1,
        "domain": "mechanism-fixture",
        "subtask_categories": ["data analysis"],
        "github_name": "fixture/toy",
        "task_inst": "  Write a Python program using the public toy input to save the requested answer.\n",
        "domain_knowledge": KNOWLEDGE_CANARY,
        "dataset_folder_tree": "└── toy\n    └── input.txt",
        "dataset_preview": "input.txt contains an integer",
        "src_file_or_path": "private/upstream/source.py",
        "gold_program_name": "toy.py",
        "output_fname": "pred_results/toy.json",
        "eval_script_name": "toy_eval.py",
    }
    annotations = tmp_path / "private-annotations.json"
    annotations.write_text(json.dumps([row], ensure_ascii=False, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(sab, "DATASET_REVISION", "a" * 40, raising=False)
    monkeypatch.setattr(sab, "ANNOTATIONS_SHA256", digest(annotations), raising=False)
    return {
        "annotations": annotations,
        "annotations_sha256": digest(annotations),
        "dataset_revision": "a" * 40,
        "benchmark": benchmark,
        "instance_id": 1,
    }


def _identity(inputs):
    return {
        "benchmark": "ScienceAgentBench", "revision": sab.REVISION,
        "dataset_revision": inputs["dataset_revision"], "annotations_sha256": inputs["annotations_sha256"],
        "split": "verified", "instance_id": str(inputs["instance_id"]), "knowledge": False,
        "program_filename": "pred_toy.py",
    }


def _prepare(tmp_path, inputs):
    return Path(sab.prepare(**inputs, output=tmp_path / "public"))


def test_prepare_copies_only_selected_public_dataset_and_original_task(tmp_path, inputs):
    manifest = _prepare(tmp_path, inputs)
    assert manifest.name == "manifest.json"
    environment = manifest.parent / "environment"
    assert (environment / "benchmark/datasets/toy/input.txt").read_bytes() == b"37\n"
    row = json.loads(inputs["annotations"].read_text())[0]
    assert row["task_inst"].encode() in (environment / "task.txt").read_bytes()
    assert (manifest.parent / "original-task.txt").read_bytes() == row["task_inst"].encode()
    assert not (environment / "benchmark/datasets/unselected").exists()
    assert not (environment / "benchmark/gold_programs").exists()
    assert not (environment / "benchmark/eval_programs").exists()
    for path in manifest.parent.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            content = path.read_bytes()
            assert GOLD_CANARY.encode() not in content
            assert KNOWLEDGE_CANARY.encode() not in content
    with pytest.raises(FileExistsError):
        sab.prepare(**inputs, output=manifest.parent)


def test_prepare_rejects_changed_annotations_and_duplicate_instance(tmp_path, inputs, monkeypatch):
    inputs["annotations"].write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        sab.prepare(**inputs, output=tmp_path / "bad-hash")
    rows = [{"instance_id": 1}, {"instance_id": 1}]
    inputs["annotations"].write_text(json.dumps(rows), encoding="utf-8")
    inputs["annotations_sha256"] = digest(inputs["annotations"])
    monkeypatch.setattr(sab, "ANNOTATIONS_SHA256", inputs["annotations_sha256"], raising=False)
    with pytest.raises(ValueError):
        sab.prepare(**inputs, output=tmp_path / "duplicate")


@pytest.mark.parametrize("field,value", [
    ("gold_program_name", "../outside.py"),
    ("gold_program_name", "/absolute.py"),
    ("dataset_folder_tree", "└── ../outside"),
    ("dataset_folder_tree", "└── /absolute"),
])
def test_prepare_rejects_unsafe_paths_before_publishing(tmp_path, inputs, monkeypatch, field, value):
    rows = json.loads(inputs["annotations"].read_text())
    rows[0][field] = value
    inputs["annotations"].write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    inputs["annotations_sha256"] = digest(inputs["annotations"])
    monkeypatch.setattr(sab, "ANNOTATIONS_SHA256", inputs["annotations_sha256"], raising=False)
    output = tmp_path / "unsafe"
    with pytest.raises(ValueError):
        sab.prepare(**inputs, output=output)
    assert not output.exists()


def test_prepare_rejects_selected_dataset_symlink(tmp_path, inputs):
    secret = tmp_path / "secret.txt"
    secret.write_text(GOLD_CANARY)
    (inputs["benchmark"] / "datasets/toy/leak.txt").symlink_to(secret)
    with pytest.raises(ValueError):
        sab.prepare(**inputs, output=tmp_path / "linked")
    assert not (tmp_path / "linked").exists()


@pytest.mark.parametrize("damage", ["extra-file", "input-bytes", "private-field"])
def test_run_rejects_adversarial_public_manifest_before_model(tmp_path, inputs, monkeypatch, damage):
    manifest = _prepare(tmp_path, inputs)
    if damage == "extra-file":
        (manifest.parent / "environment/private-gold.json").write_text(GOLD_CANARY)
    elif damage == "input-bytes":
        (manifest.parent / "environment/benchmark/datasets/toy/input.txt").write_text("999")
    else:
        value = json.loads(manifest.read_text())
        value["private_gold"] = GOLD_CANARY
        manifest.write_text(json.dumps(value), encoding="utf-8")
    def no_model(*args, **kwargs):
        pytest.fail("invalid public input must fail before model transport")
    monkeypatch.setattr("resagent2_runtime.llm.send_request", no_model)
    with pytest.raises(ValueError):
        sab.run(manifest=manifest, output=tmp_path / "invalid-run", execution_mode="offline-mechanism")


def _native_chain(tmp_path, inputs, monkeypatch, *, root_program=None, fail_child=False):
    from resagent2_cli.composition import build_application
    from resagent2_contracts import UserAnswer
    from resagent2_orchestrator import JsonRunStore

    manifest = _prepare(tmp_path, inputs)
    output = tmp_path / "run"
    counts = Counter()
    monkeypatch.setenv("RESAGENT2_MODEL", "offline-sab-mechanism")
    monkeypatch.setenv("RESAGENT2_API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("RESAGENT2_API_KEY_ENV", "SAB_MECHANISM_KEY")
    monkeypatch.setenv("SAB_MECHANISM_KEY", "dummy-test-only")
    monkeypatch.setenv("RESAGENT2_WEB_SEARCH_PROVIDER", "off")
    monkeypatch.setenv("RESAGENT2_LLM_TRACE_LEVEL", "off")
    monkeypatch.setenv("RESAGENT2_RESOURCE_ROOT", str(tmp_path / "resources"))

    def evidence():
        current = JsonRunStore(output / "data/state").load("run_sab_mechanism")
        return next(ref.id for ref in current.artifacts.values() if ref.output_name == sab.OUTPUT_NAME)

    def reply(request, **kwargs):
        body = json.loads(request.content)
        serialized = json.dumps(body)
        assert GOLD_CANARY not in serialized
        assert KNOWLEDGE_CANARY not in serialized
        names = {tool["function"]["name"] for tool in body.get("tools", [])}
        role = "executor" if "run_shell" in names else "scientific"
        counts[role] += 1
        index = counts[role]
        if role == "scientific":
            if index == 1:
                tool, args = "request_work", {
                    "assessment": {"statement": "Implement and run the requested public-input Python program"},
                    "work_request": {"objective": "Implement and verify the toy program", "tasks": [{
                        "key": "implement", "instruction": "Create pred_toy.py, execute it against the public input, and register the exact Python file as output_name=scienceagentbench_program.",
                        "selection": {"kind": "preset", "preset_ref": "coding@1", "reason": "Python source implementation and execution are one coherent task"},
                        "output_names": [sab.OUTPUT_NAME],
                    }]},
                }
            elif index == 2:
                tool, args = "read_artifact", {"artifact_id": evidence()}
            elif index == 3:
                tool, args = "finish", {"report": "Implemented a public-input program and retained its frozen source.", "artifacts": [{
                    "kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json", "summary": "Offline mechanism opinion",
                    "content": {"verdict": "inconclusive", "statement": "Offline transport mechanism completed", "evidence_artifact_ids": [evidence()],
                                "limitations": ["Scripted transport; not a real model benchmark score."]},
                }]}
                if root_program is not None:
                    args["artifacts"].append({"kind": "scientific_assessment", "path": "pred_toy.py", "media_type": "text/x-python",
                        "summary": "Accepted top-level Python program", "output_name": sab.OUTPUT_NAME, "content": root_program})
            else:
                raise AssertionError(body)
        else:
            assert "create_file" in names, "program implementation must use native coding@1"
            if index == 1:
                tool, args = "create_file", {"path": str(output / "environment/pred_toy.py"), "content": PROGRAM}
            elif index == 2:
                tool, args = "run_shell", {"command": shlex.quote(sys.executable) + " pred_toy.py"}
            elif index == 3:
                assert json.loads((output / "environment/pred_results/toy.json").read_text()) == {"answer": 42}
                tool, args = "finish", {"report": "Created and executed the actual public-input program.", "artifacts": [{
                    "kind": "source_program", "path": "pred_toy.py", "media_type": "text/x-python",
                    "summary": "Actual implemented Python program", "output_name": sab.OUTPUT_NAME,
                }]}
                if fail_child:
                    args.update(status="failed", report="Fixture task failed after producing its source and result.")
            else:
                raise AssertionError(body)
        return httpx.Response(200, request=request, json={"choices": [{"finish_reason": "tool_calls", "message": {"content": None,
            "tool_calls": [{"id": f"call_{role}_{index}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]}}]})

    def no_network(*args, **kwargs):
        pytest.fail("offline mechanism tests must never send real network requests")
    monkeypatch.setattr("httpx.Client.send", no_network)
    monkeypatch.setattr("httpx.AsyncClient.send", no_network)
    monkeypatch.setattr("resagent2_runtime.llm.send_request", reply)
    run = sab.run(manifest=manifest, output=output, run_id="run_sab_mechanism", max_llm_calls=16,
        timeout_seconds=60, execution_mode="offline-mechanism")
    assert run.status.value == "paused", run.terminal_error
    assert run.pending_question.action.tool == "run_shell"
    app = build_application(data_root=output / "data")
    question = run.pending_question
    run = app.controller.answer_question(run.run_id, UserAnswer(question_id=question.id,
        values={question.requested_fields[0]: "yes"}, answered_at=datetime.now(UTC)))
    assert run.status.value == "completed", run.terminal_error
    assert counts == {"scientific": 3, "executor": 3}
    assert run.workflow.tasks[0].workflow_agent_kind.value == "scientific"
    child = next(task for task in run.workflow.tasks if task.id != "task_root")
    assert child.execution_ref == "coding@1"
    assert run.usage.used == 6
    assert run.request.required_artifacts == [sab.OUTPUT_NAME]
    assert json.loads(run.request.context) == _identity(inputs)
    assert any(ref.kind == "execution_record" for ref in run.artifacts.values())
    assert any(ref.kind == "code_patch" for ref in run.artifacts.values())
    assert json.loads((output / "run.json").read_text())["execution_mode"] == "offline-mechanism"
    destination = output / "submission/pred_toy.py"
    sab.export_saved(data_root=output / "data", run_id=run.run_id, output=destination)
    return run, output, destination


def test_native_coding_chain_exports_frozen_program_not_report_or_mutable_workspace(tmp_path, inputs, monkeypatch):
    run, output, destination = _native_chain(tmp_path, inputs, monkeypatch)
    assert destination.read_bytes() == PROGRAM.encode()
    receipt = json.loads((output / "export.json").read_text())
    frozen = run.artifacts[receipt["program_artifact_id"]]
    assert frozen.output_name == sab.OUTPUT_NAME
    assert frozen.producer.value == "executor"
    assert receipt["program_sha256"] == digest(destination) == frozen.sha256
    assert receipt["usage"] == run.usage.model_dump(mode="json")
    assert receipt["benchmark_identity"] == json.loads(run.request.context) == _identity(inputs)
    (output / "environment/pred_toy.py").write_text("raise RuntimeError('mutable tampering')\n")
    sab.export_program(run, destination)
    assert destination.read_bytes() == PROGRAM.encode()
    assert json.loads((output / "environment/pred_results/toy.json").read_text()) == {"answer": 42}


@pytest.mark.parametrize("fail_child", [False, True])
@pytest.mark.parametrize("program", ["\n# accepted final program: π\nprint(  'root program'  )\n\n", "def broken( :\n# Official VER, not the adapter, determines syntax validity.\n"])
def test_accepted_root_inline_program_wins_with_exact_bytes_and_truthful_provenance(tmp_path, inputs, monkeypatch, fail_child, program):
    run, output, destination = _native_chain(tmp_path, inputs, monkeypatch, root_program=program, fail_child=fail_child)
    frozen = sab.export_program(run, destination)
    assert destination.read_bytes() == program.encode()
    assert frozen.producer.value == "scientific"
    assert frozen.task_id == "task_root"
    assert frozen.session_id == run.scientific_session.id
    assert frozen.id not in run.final_opinion.evidence_artifact_ids
    assert "source_path" not in frozen.metadata
    receipt = json.loads((output / "export.json").read_text())
    assert receipt["program_provenance"]["producer"] == "scientific"
    assert receipt["program_sha256"] == frozen.sha256
    invalid = run.model_copy(deep=True)
    invalid.artifacts[frozen.id] = frozen.model_copy(update={"session_id": "session_foreign"})
    with pytest.raises(ValueError):
        sab.export_program(invalid, tmp_path / "foreign.py")
    invalid = run.model_copy(deep=True)
    duplicate = frozen.model_copy(update={"id": "artifact_duplicate_root"})
    invalid.artifacts[duplicate.id] = duplicate
    invalid.task("task_root").attempts[-1].artifact_ids.append(duplicate.id)
    with pytest.raises(ValueError):
        sab.export_program(invalid, tmp_path / "ambiguous-root.py")


def test_export_rejects_unaccepted_uncited_missing_and_corrupt_programs(tmp_path, inputs, monkeypatch):
    run, output, destination = _native_chain(tmp_path, inputs, monkeypatch)
    frozen = next(ref for ref in run.artifacts.values() if ref.output_name == sab.OUTPUT_NAME)
    for damage in ("run-status", "uncited", "task-status", "attempt-error", "missing-final-report"):
        invalid = run.model_copy(deep=True)
        if damage == "run-status":
            invalid.status = type(invalid.status).FAILED
        elif damage == "uncited":
            invalid.final_opinion.evidence_artifact_ids.remove(frozen.id)
        elif damage == "task-status":
            task = invalid.task(frozen.task_id)
            task.status = type(task.status).FAILED
        elif damage == "attempt-error":
            invalid.task(frozen.task_id).attempts[-1].finished_at = None
        else:
            invalid.final_report_artifact_id = None
        with pytest.raises(ValueError):
            sab.export_program(invalid, tmp_path / (damage + ".py"))
    victim = tmp_path / "victim.py"
    victim.write_text("preserve me")
    linked = tmp_path / "linked.py"
    linked.symlink_to(victim)
    with pytest.raises(ValueError):
        sab.export_program(run, linked)
    assert victim.read_text() == "preserve me"
    frozen_path = Path(url2pathname(urlparse(frozen.uri).path))
    frozen_path.write_text("print('corrupt frozen source')\n")
    with pytest.raises(ValueError):
        sab.export_program(run, tmp_path / "corrupt.py")


def test_export_ignores_failed_retry_but_rejects_multiple_accepted_cited_programs(tmp_path, inputs, monkeypatch):
    from resagent2_contracts import AttemptStatus
    run, output, destination = _native_chain(tmp_path, inputs, monkeypatch)
    frozen = next(ref for ref in run.artifacts.values() if ref.output_name == sab.OUTPUT_NAME)
    task = run.task(frozen.task_id)
    successful = task.attempts[-1]
    old_ref = frozen.model_copy(update={"id": "artifact_failed_old", "attempt_number": 1})
    failed = successful.model_copy(update={"number": 1, "status": AttemptStatus.FAILED, "artifact_ids": [old_ref.id]})
    successful.number = 2
    task.attempts = [failed, successful]
    new_ref = frozen.model_copy(update={"attempt_number": 2})
    run.artifacts[frozen.id] = new_ref
    run.artifacts[old_ref.id] = old_ref
    run.final_opinion.evidence_artifact_ids.append(old_ref.id)
    assert sab.export_program(run, destination).id == new_ref.id
    second_task = task.model_copy(deep=True, update={"id": "task_second_success"})
    second = new_ref.model_copy(update={"id": "artifact_second_success", "task_id": second_task.id})
    second_task.attempts[-1].artifact_ids = [second.id]
    run.workflow.tasks.append(second_task)
    run.artifacts[second.id] = second
    run.final_opinion.evidence_artifact_ids.append(second.id)
    with pytest.raises(ValueError):
        sab.export_program(run, tmp_path / "ambiguous.py")

@pytest.mark.parametrize("official_output", [None, [], [1, 0.5, 1]])
def test_grade_missing_or_malformed_actual_result_is_infrastructure_error(tmp_path, inputs, monkeypatch, official_output):
    """Check the receipt boundary only; this stub is never an official score."""
    import ast
    import subprocess
    import tarfile

    program = tmp_path / "pred_toy.py"
    program.write_text(PROGRAM, encoding="utf-8")
    export_receipt = tmp_path / "export.json"
    export_receipt.write_text(json.dumps({"status": "completed", "program_sha256": digest(program), "benchmark_identity": _identity(inputs)}))
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-offline-grader-key")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    calls = []

    def official_process_boundary(command, **kwargs):
        calls.append(command)
        if command[0] == "git" and "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout=sab.REVISION + "\n")
        if command[0] == "docker":
            return subprocess.CompletedProcess(command, 0, stdout="offline-fixture-daemon\n")
        if command[0] == "git" and "archive" in command:
            with tarfile.open(fileobj=kwargs["stdout"], mode="w"):
                pass
            return subprocess.CompletedProcess(command, 0)
        assert command[:2] == [sys.executable, "-c"]
        assert "run_instances([e]" in command[2]
        assert "build_base_images" in command[2]
        if official_output is not None:
            invocation = next(node for node in ast.walk(ast.parse(command[2])) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_instances")
            evaluation_run_id = ast.literal_eval(invocation.args[7])
            target = Path(kwargs["cwd"]) / "logs/run_evaluation" / evaluation_run_id / "1/output/result.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(official_output))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sab.subprocess, "run", official_process_boundary)
    output = tmp_path / "grade"
    with pytest.raises(RuntimeError):
        sab.grade(**inputs, upstream=upstream, program=program, export_receipt=export_receipt, output=output)
    receipt = json.loads((output / "grade.json").read_text())
    assert receipt["status"] == "infrastructure_error"
    assert receipt["official_result"] is None
    assert receipt["valid_program"] is None
    assert receipt["success_rate"] is None
    assert receipt["error"]
    assert any(command[0] == "git" and "archive" in command for command in calls)
    assert any(command[:2] == [sys.executable, "-c"] for command in calls)

def test_prepare_and_run_reject_symlinked_output_ancestors(tmp_path, inputs, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked-parent"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        sab.prepare(**inputs, output=link / "public")
    assert not (outside / "public").exists()
    manifest = _prepare(tmp_path, inputs)
    def no_model(*args, **kwargs):
        pytest.fail("unsafe output path must fail before model invocation")
    monkeypatch.setattr("resagent2_runtime.llm.send_request", no_model)
    with pytest.raises(ValueError):
        sab.run(manifest=manifest, output=link / "run", execution_mode="offline-mechanism")
    assert not (outside / "run").exists()


def _grade_submission(tmp_path, inputs):
    program = tmp_path / "pred_toy.py"
    program.write_text(PROGRAM, encoding="utf-8")
    receipt = tmp_path / "export.json"
    receipt.write_text(json.dumps({"status": "completed", "program_sha256": digest(program), "benchmark_identity": _identity(inputs)}))
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    return {"program": program, "export_receipt": receipt, "upstream": upstream}


@pytest.mark.parametrize("field,value", [
    ("benchmark", "CORE-Bench"),
    ("revision", "b" * 40),
    ("dataset_revision", "b" * 40),
    ("annotations_sha256", "b" * 64),
    ("split", "train"),
    ("instance_id", "2"),
    ("program_filename", "pred_other.py"),
    ("knowledge", "false"),
])
def test_grade_rejects_cross_task_or_version_export_before_processes(tmp_path, inputs, monkeypatch, field, value):
    submission = _grade_submission(tmp_path, inputs)
    path = submission["export_receipt"]
    receipt = json.loads(path.read_text())
    receipt["benchmark_identity"][field] = value
    path.write_text(json.dumps(receipt))
    def no_process(*args, **kwargs):
        pytest.fail("cross-task export must be rejected before Git or Docker")
    monkeypatch.setattr(sab.subprocess, "run", no_process)
    with pytest.raises(ValueError):
        sab.grade(**inputs, **submission, output=tmp_path / "wrong-grade")


@pytest.mark.parametrize("identity", [None, [], "invalid"])
def test_grade_rejects_malformed_benchmark_identity_before_processes(tmp_path, inputs, monkeypatch, identity):
    submission = _grade_submission(tmp_path, inputs)
    path = submission["export_receipt"]
    receipt = json.loads(path.read_text())
    receipt["benchmark_identity"] = identity
    path.write_text(json.dumps(receipt))
    def no_process(*args, **kwargs):
        pytest.fail("invalid identity must be rejected before Git or Docker")
    monkeypatch.setattr(sab.subprocess, "run", no_process)
    with pytest.raises(ValueError):
        sab.grade(**inputs, **submission, output=tmp_path / "invalid-grade")


def test_grade_missing_judge_credentials_retains_infrastructure_receipt_without_worker(tmp_path, inputs, monkeypatch):
    import subprocess
    import tarfile
    submission = _grade_submission(tmp_path, inputs)
    for name in ("OPENAI_API_KEY", "AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    def only_git(command, **kwargs):
        calls.append(command)
        if command[0] == "git" and "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout=sab.REVISION + "\n")
        if command[0] == "git" and "archive" in command:
            with tarfile.open(fileobj=kwargs["stdout"], mode="w"):
                pass
            return subprocess.CompletedProcess(command, 0)
        pytest.fail("missing credentials must not start Docker grading worker")
    monkeypatch.setattr(sab.subprocess, "run", only_git)
    output = tmp_path / "grade-no-credentials"
    with pytest.raises(RuntimeError, match="credentials"):
        sab.grade(**inputs, **submission, output=output)
    receipt = json.loads((output / "grade.json").read_text())
    assert receipt["status"] == "infrastructure_error"
    assert receipt["official_result"] is None
    assert receipt["valid_program"] is None
    assert receipt["success_rate"] is None
    assert "credentials" in receipt["error"]
    assert len(calls) == 2

def _stub_grader_process(monkeypatch, on_worker):
    """Keep grading boundary tests wholly local; never contact a Docker daemon."""
    import subprocess
    import tarfile
    calls = []
    def process(command, **kwargs):
        calls.append(command)
        if command[0] == "git" and "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout=sab.REVISION + "\n")
        if command[0] == "git" and "archive" in command:
            with tarfile.open(fileobj=kwargs["stdout"], mode="w"):
                pass
            return subprocess.CompletedProcess(command, 0)
        assert command[:2] == [sys.executable, "-c"]
        on_worker(command, kwargs)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(sab.subprocess, "run", process)
    return calls


def test_grade_outer_timeout_cleans_only_exact_invocation_container(tmp_path, inputs, monkeypatch):
    import subprocess
    submission = _grade_submission(tmp_path, inputs)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-offline-grader-key")
    def timeout_worker(command, kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    _stub_grader_process(monkeypatch, timeout_worker)
    removed = []
    def cleanup(name):
        removed.append(name)
        return {"container": name, "status": "removed"}
    monkeypatch.setattr(sab, "_cleanup_container", cleanup)
    output = tmp_path / "grade-timeout"
    with pytest.raises(RuntimeError, match="outer deadline"):
        sab.grade(**inputs, **submission, output=output, timeout_seconds=1)
    receipt = json.loads((output / "grade.json").read_text())
    expected = "sab.eval.1." + receipt["grading_run_id"]
    assert removed == [expected]
    assert receipt["timeout_container_cleanup"] == {"container": expected, "status": "removed"}
    assert receipt["status"] == "infrastructure_error"
    assert receipt["official_result"] is None
    assert receipt["success_rate"] is None


def test_grade_rejects_program_changed_since_export_before_processes(tmp_path, inputs, monkeypatch):
    submission = _grade_submission(tmp_path, inputs)
    submission["program"].write_bytes(b"not the frozen program\n")
    def no_process(*args, **kwargs):
        pytest.fail("changed program must fail before Git or Docker")
    monkeypatch.setattr(sab.subprocess, "run", no_process)
    with pytest.raises(ValueError, match="program"):
        sab.grade(**inputs, **submission, output=tmp_path / "changed-program")


def test_grade_asset_mutation_cannot_be_reported_as_score(tmp_path, inputs, monkeypatch):
    import ast
    submission = _grade_submission(tmp_path, inputs)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-offline-grader-key")
    def altered_assets(command, kwargs):
        invocation = next(node for node in ast.walk(ast.parse(command[2])) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_instances")
        grading_run_id = ast.literal_eval(invocation.args[7])
        result = Path(kwargs["cwd"]) / "logs/run_evaluation" / grading_run_id / "1/output/result.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps([1, 0.5, 1, "boundary fixture; never an official score"]))
        (inputs["benchmark"] / "datasets/toy/input.txt").write_text("changed during grading")
    _stub_grader_process(monkeypatch, altered_assets)
    output = tmp_path / "changed-assets"
    with pytest.raises(RuntimeError, match="assets changed"):
        sab.grade(**inputs, **submission, output=output)
    receipt = json.loads((output / "grade.json").read_text())
    assert receipt["status"] == "infrastructure_error"
    assert receipt["official_result"] is None
    assert receipt["valid_program"] is None
    assert receipt["success_rate"] is None

@pytest.mark.parametrize("preview", [None, ""])
def test_verified_optional_preview_and_hyphenated_program_names(tmp_path, inputs, monkeypatch, preview):
    # Verified ID11 uses a hyphen; 18 official rows have null previews. Preserve
    # the upstream representation rather than rejecting or normalizing them.
    row = json.loads(inputs["annotations"].read_text())[0]
    row.update(dataset_preview=preview, gold_program_name="fixture-cell.py", eval_script_name="fixture-cell_eval.py")
    inputs["annotations"].write_text(json.dumps([row]))
    inputs["annotations_sha256"] = digest(inputs["annotations"])
    monkeypatch.setattr(sab, "ANNOTATIONS_SHA256", inputs["annotations_sha256"])
    manifest = _prepare(tmp_path, inputs)
    _, public = sab._public(manifest)
    assert public["program_filename"] == "pred_fixture-cell.py"
    text = (manifest.parent / "environment/task.txt").read_text()
    assert text.endswith("Here are some helpful previews for the dataset file(s):\n" + str(preview))


def test_public_workspace_mutation_during_clone_is_rejected_before_execution(tmp_path, inputs, monkeypatch):
    manifest = _prepare(tmp_path, inputs)
    real_copy = sab.shutil.copytree
    def changed_copy(source, destination, *args, **kwargs):
        result = real_copy(source, destination, *args, **kwargs)
        target = Path(destination) / "benchmark/datasets/toy/input.txt"
        if target.exists():
            target.write_text("different bytes after manifest validation")
        return result
    monkeypatch.setattr(sab.shutil, "copytree", changed_copy)
    def forbidden(*args, **kwargs):
        pytest.fail("changed input must fail before any Git or execution process")
    monkeypatch.setattr(sab.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="changed while copying"):
        sab.run(manifest=manifest, output=tmp_path / "raced-run", execution_mode="offline-mechanism")


def test_snapshot_rejects_unpinned_file_before_optional_arrow_import(tmp_path):
    parquet = tmp_path / "wrong.parquet"
    parquet.write_bytes(b"not the official parquet")
    output = tmp_path / "annotations.json"
    with pytest.raises(ValueError, match="parquet"):
        sab.snapshot(parquet=parquet, output=output)
    assert not output.exists()
