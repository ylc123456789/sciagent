"""CORE bridge tests use native agents; scripted responses are transport-only."""
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import UTC, datetime

import httpx
import pytest

from e2e import corebench


def _archive(path, files, *, link=None):
    with tarfile.open(path, "w:gz") as bundle:
        for name, content in files.items():
            encoded = content.encode()
            member = tarfile.TarInfo(name)
            member.size = len(encoded)
            bundle.addfile(member, io.BytesIO(encoded))
        if link:
            member = tarfile.TarInfo(link)
            member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
            bundle.addfile(member)
    return path


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    dataset, prompts = tmp_path / "private-train.json", tmp_path / "prompts.json"
    dataset.write_text(json.dumps([{"capsule_id": "capsule-0000001", "task_prompt": "  Execute the supplied script and report answer.\n",
        "results": [{"answer": "MECHANISM_RESULT"}], "private_audit": "GOLD_CANARY"}]))
    prompts.write_text(json.dumps({"codeocean_hard": "Task: codeocean_hard\n{task_prompt}\nQuestions: {json_fields}"}))
    monkeypatch.setattr(corebench, "TRAIN_SHA256", corebench.digest(dataset))
    monkeypatch.setattr(corebench, "PROMPTS_SHA256", corebench.digest(prompts))
    archive = _archive(tmp_path / "capsule.tar.gz", {
        "capsule-0000001/code/main.py": "import json\nfrom pathlib import Path\nPath('report.json').write_text(json.dumps({'answer': 'mechanism_result'.upper()}))\n",
        "capsule-0000001/README.md": "Run code/main.py from the environment root.",
        "capsule-0000001/data/input.txt": "public data",
        "capsule-0000001/results/leak.txt": "GOLD_CANARY",
        "capsule-0000001/environment/Dockerfile": "hidden setup",
        "capsule-0000001/REPRODUCING.md": "removed instruction",
        "capsule-0000001/code/run.sh": "removed run",
        "capsule-0000001/code/run": "removed run",
    })
    return dict(dataset=dataset, prompts=prompts, capsule_id="capsule-0000001", archive=archive, archive_sha256=corebench.digest(archive))


def test_prepare_preserves_public_bytes_and_official_hard_boundary(tmp_path, inputs):
    manifest = corebench.prepare(**inputs, output=tmp_path / "public")
    public = json.loads(manifest.read_text())
    assert public["question_keys"] == ["answer"]
    assert "GOLD_CANARY" not in manifest.read_text()
    assert (manifest.parent / "original-task-prompt.txt").read_bytes() == b"  Execute the supplied script and report answer.\n"
    environment = manifest.parent / "environment"
    capsule = environment / inputs["capsule_id"]
    assert list((capsule / "results").iterdir()) == []
    for name in ("environment", "REPRODUCING.md", "code/run.sh", "code/run"):
        assert not (capsule / name).exists()
    assert "dict_keys(['answer'])" in (environment / "task.txt").read_text()
    assert (environment / ".git").is_dir()
    with pytest.raises(FileExistsError):
        corebench.prepare(**inputs, output=manifest.parent)
    (environment / "task.txt").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        corebench._public(manifest)


@pytest.mark.parametrize("member", ["../escape", "/absolute", "capsule-0000001/../escape", "capsule-0000001/a\\b", "other/a", "capsule-0000001/a:b"])
def test_prepare_rejects_unsafe_tar_before_writing(tmp_path, inputs, member):
    inputs["archive"] = _archive(tmp_path / "bad.tar.gz", {member: "bad"})
    inputs["archive_sha256"] = corebench.digest(inputs["archive"])
    with pytest.raises(ValueError, match="unsafe"):
        corebench.prepare(**inputs, output=tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_prepare_rejects_link_and_wrong_upstream_pin(tmp_path, inputs):
    inputs["archive"] = _archive(tmp_path / "bad.tar.gz", {}, link="capsule-0000001/link")
    inputs["archive_sha256"] = corebench.digest(inputs["archive"])
    with pytest.raises(ValueError, match="unsafe"):
        corebench.prepare(**inputs, output=tmp_path / "public")
    inputs["dataset"].write_text("[]")
    with pytest.raises(ValueError, match="pinned"):
        corebench.prepare(**inputs, output=tmp_path / "public")


def test_public_manifest_rejects_extra_files_and_private_fields(tmp_path, inputs):
    manifest = corebench.prepare(**inputs, output=tmp_path / "public")
    leak = manifest.parent / "environment" / "private-gold.json"
    leak.write_text("private")
    with pytest.raises(ValueError, match="unexpected or missing"):
        corebench._public(manifest)
    leak.unlink()
    outside = tmp_path / "private.txt"
    outside.write_text("must not be copied")
    leak.symlink_to(outside)
    with pytest.raises(ValueError, match="links"):
        corebench._public(manifest)
    leak.unlink()
    public = json.loads(manifest.read_text())
    public["results"] = [{"answer": "GOLD_CANARY"}]
    manifest.write_text(json.dumps(public))
    with pytest.raises(ValueError, match="unexpected public manifest"):
        corebench._public(manifest)


def _native_chain(tmp_path, inputs, monkeypatch, *, root_report=None, fail_child=False):
    from sciagent_cli.composition import build_application
    from sciagent_contracts import UserAnswer
    manifest = corebench.prepare(**inputs, output=tmp_path / "public")
    output = tmp_path / "run"
    counts = Counter()
    monkeypatch.setenv("SCIAGENT_MODEL", "offline-corebench-mechanism")
    monkeypatch.setenv("SCIAGENT_API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("SCIAGENT_API_KEY_ENV", "COREBENCH_MECHANISM_KEY")
    monkeypatch.setenv("COREBENCH_MECHANISM_KEY", "dummy-test-only")
    monkeypatch.setenv("SCIAGENT_WEB_SEARCH_PROVIDER", "off")
    monkeypatch.setenv("SCIAGENT_LLM_TRACE_LEVEL", "off")
    monkeypatch.setenv("SCIAGENT_RESOURCE_ROOT", str(tmp_path / "resources"))

    def evidence():
        from sciagent_orchestrator import JsonRunStore
        current = JsonRunStore(output / "data" / "state").load("run_corebench_mechanism")
        return next(ref.id for ref in current.artifacts.values() if ref.output_name == corebench.OUTPUT_NAME)

    def reply(request, **kwargs):
        body = json.loads(request.content)
        assert "GOLD_CANARY" not in json.dumps(body)
        names = {tool["function"]["name"] for tool in body.get("tools", [])}
        role = "executor" if "run_shell" in names else "scientific"
        counts[role] += 1
        index = counts[role]
        if role == "scientific":
            if index == 1:
                tool, args = "request_work", {"assessment": {"statement": "Execute the public capsule and retain measured answer"},
                    "work_request": {"objective": "Execute the supplied capsule", "tasks": [{"key": "measure",
                        "instruction": "Run capsule-0000001/code/main.py and register report.json as output_name=corebench_report.",
                        "selection": {"kind": "preset", "preset_ref": "experiment@1", "reason": "Execute scientific code"},
                        "output_names": [corebench.OUTPUT_NAME]}]}}
            elif index == 2:
                tool, args = "read_artifact", {"artifact_id": evidence()}
            elif index == 3:
                tool, args = "finish", {"report": "Executed the supplied capsule and recorded the answer.", "artifacts": [{
                    "kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json", "summary": "Mechanism test opinion",
                    "content": {"verdict": "inconclusive", "statement": "Offline mechanism fixture completed", "evidence_artifact_ids": [evidence()],
                        "limitations": ["Scripted transport; this is not a real model benchmark result."]}}]}
                if root_report is not None:
                    args["artifacts"].append({"kind": "scientific_assessment", "path": "report.json",
                        "media_type": "application/json", "summary": "Top-level measured report",
                        "output_name": corebench.OUTPUT_NAME, "content": root_report})
            else:
                raise AssertionError(body)
        elif index == 1:
            tool, args = "run_shell", {"command": shlex.quote(sys.executable) + " capsule-0000001/code/main.py"}
        elif index == 2:
            assert (output / "environment" / "report.json").is_file()
            tool, args = "finish", {"report": "Executed capsule script successfully.", "artifacts": [{"kind": "benchmark_report", "path": "report.json",
                "media_type": "application/json", "summary": "Answer from actual script execution", "output_name": corebench.OUTPUT_NAME}]}
            if fail_child:
                args["status"] = "failed"
                args["report"] = "Fixture child failed after producing its measured output."
        else:
            raise AssertionError(body)
        return httpx.Response(200, request=request, json={"choices": [{"finish_reason": "tool_calls", "message": {"content": None,
            "tool_calls": [{"id": f"call_{role}_{index}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]}}]})

    def no_http(*args, **kwargs):
        pytest.fail("offline mechanism tests must not send network requests")
    monkeypatch.setattr("httpx.Client.send", no_http)
    monkeypatch.setattr("httpx.AsyncClient.send", no_http)
    monkeypatch.setattr("sciagent_runtime.llm.send_request", reply)
    run = corebench.run(manifest=manifest, output=output, run_id="run_corebench_mechanism", max_llm_calls=12, timeout_seconds=60, execution_mode="offline-mechanism")
    if run.status.value == "paused":
        assert run.pending_question.action.tool == "run_shell"
        app = build_application(data_root=output / "data")
        question = run.pending_question
        run = app.controller.answer_question(run.run_id, UserAnswer(question_id=question.id,
            values={question.requested_fields[0]: "yes"}, answered_at=datetime.now(UTC)))
    assert run.status.value == "completed", run.terminal_error
    assert counts == {"scientific": 3, "executor": 2}
    assert run.workflow.tasks[0].workflow_agent_kind.value == "scientific"
    assert run.usage.used == 5
    assert any(ref.kind == "execution_record" for ref in run.artifacts.values())
    assert json.loads((output / "run.json").read_text())["execution_mode"] == "offline-mechanism"
    corebench.export_saved(data_root=output / "data", run_id=run.run_id, output=output / "environment" / "report.json")
    return run, output


def test_native_production_chain_exports_only_registered_workspace_report(tmp_path, inputs, monkeypatch):
    run, output = _native_chain(tmp_path, inputs, monkeypatch)
    assert json.loads((output / "environment" / "report.json").read_text()) == {"answer": "MECHANISM_RESULT"}
    receipt = json.loads((output / "export.json").read_text())
    assert receipt["usage"] == run.usage.model_dump(mode="json")
    frozen = next(ref for ref in run.artifacts.values() if ref.output_name == corebench.OUTPUT_NAME)
    assert corebench.digest(output / "environment" / "report.json") == frozen.sha256
    # Existing workspace output is not authoritative after registration.
    (output / "environment" / "report.json").write_text('{"answer":"tampered"}')
    corebench.export_report(run, output / "environment" / "report.json")
    assert corebench.digest(output / "environment" / "report.json") == frozen.sha256
    invalid = run.model_copy(deep=True)
    invalid.final_opinion.evidence_artifact_ids.remove(frozen.id)
    with pytest.raises(ValueError, match="exactly one"):
        corebench.export_report(invalid, tmp_path / "uncited.json")
    invalid = run.model_copy(deep=True)
    invalid.task(frozen.task_id).status = type(invalid.task(frozen.task_id).status).FAILED
    with pytest.raises(ValueError, match="exactly one"):
        corebench.export_report(invalid, tmp_path / "failed-child.json")
    victim = tmp_path / "victim.json"
    victim.write_text("preserve me")
    linked = tmp_path / "linked-report.json"
    linked.symlink_to(victim)
    with pytest.raises(ValueError, match="symlinks"):
        corebench.export_report(run, linked)
    assert victim.read_text() == "preserve me"


@pytest.mark.parametrize("fail_child", [False, True])
def test_top_level_inline_report_wins_with_truthful_provenance(tmp_path, inputs, monkeypatch, fail_child):
    # Distinct whitespace proves export copies the root's exact frozen bytes;
    # the child/workspace report remains a separate source even on failure.
    report = '{ "answer": "MECHANISM_RESULT" }\n'
    run, output = _native_chain(tmp_path, inputs, monkeypatch, root_report=report, fail_child=fail_child)
    frozen = corebench.export_report(run, output / "environment" / "report.json")
    assert frozen.producer.value == "scientific"
    assert frozen.task_id == "task_root"
    assert frozen.session_id == run.scientific_session.id
    assert frozen.id not in run.final_opinion.evidence_artifact_ids
    assert (output / "environment" / "report.json").read_bytes() == report.encode()
    child = next(task for task in run.workflow.tasks if task.id != "task_root")
    assert child.status.value == ("failed" if fail_child else "completed")
    assert child.attempts[-1].status.value == ("failed" if fail_child else "completed")
    if fail_child:
        assert child.attempts[-1].error is not None
    receipt = json.loads((output / "export.json").read_text())
    assert receipt["report_provenance"]["producer"] == "scientific"
    assert receipt["report_provenance"]["task_id"] == "task_root"
    assert receipt["report_sha256"] == frozen.sha256
    # A damaged root binding cannot silently fall back to a valid cited child.
    invalid = run.model_copy(deep=True)
    invalid.artifacts[frozen.id] = frozen.model_copy(update={"session_id": "session_foreign"})
    with pytest.raises(ValueError, match="provenance"):
        corebench.export_report(invalid, tmp_path / "foreign-session.json")
    invalid = run.model_copy(deep=True)
    duplicate = frozen.model_copy(update={"id": "artifact_duplicate_root"})
    invalid.artifacts[duplicate.id] = duplicate
    invalid.task("task_root").attempts[-1].artifact_ids.append(duplicate.id)
    with pytest.raises(ValueError, match="exactly one"):
        corebench.export_report(invalid, tmp_path / "ambiguous-root.json")
    # A matching workspace file is never a replacement for Registry bytes.
    from urllib.parse import urlparse
    from urllib.request import url2pathname
    Path(url2pathname(urlparse(frozen.uri).path)).write_text('{"answer":"changed frozen bytes"}')
    with pytest.raises(ValueError, match="hash|sha256"):
        corebench.export_report(run, tmp_path / "tampered-export.json")


def test_export_ignores_failed_retry_output_but_rejects_successful_cited_ambiguity(tmp_path, inputs, monkeypatch):
    from sciagent_contracts import AttemptStatus
    run, output = _native_chain(tmp_path, inputs, monkeypatch)
    retried = run.model_copy(deep=True)
    frozen = next(ref for ref in retried.artifacts.values() if ref.output_name == corebench.OUTPUT_NAME)
    task = retried.task(frozen.task_id)
    successful = task.attempts[-1]
    old_ref = frozen.model_copy(update={"id": "artifact_old_failed", "attempt_number": 1})
    failed = successful.model_copy(update={"number": 1, "status": AttemptStatus.FAILED, "artifact_ids": [old_ref.id]})
    successful.number = 2
    task.attempts = [failed, successful]
    new_ref = frozen.model_copy(update={"attempt_number": 2})
    retried.artifacts[frozen.id] = new_ref
    retried.artifacts[old_ref.id] = old_ref
    assert corebench.export_report(retried, output / "environment" / "report.json").id == new_ref.id
    # Even citing a retained failed report cannot turn it into final delivery.
    retried.final_opinion.evidence_artifact_ids.append(old_ref.id)
    assert corebench.export_report(retried, output / "environment" / "report.json").id == new_ref.id
    second_task = task.model_copy(deep=True, update={"id": "task_second_success"})
    second = new_ref.model_copy(update={"id": "artifact_second_success", "task_id": second_task.id})
    second_task.attempts[-1].artifact_ids = [second.id]
    retried.workflow.tasks.append(second_task)
    retried.artifacts[second.id] = second
    retried.final_opinion.evidence_artifact_ids.append(second.id)
    with pytest.raises(ValueError, match="exactly one"):
        corebench.export_report(retried, tmp_path / "ambiguous.json")


@pytest.mark.parametrize("root_recovery", [False, True])
def test_native_chain_to_separate_official_grader(tmp_path, inputs, monkeypatch, root_recovery):
    evaluator = os.environ.get("COREBENCH_OFFICIAL_EVALUATOR")
    if not evaluator:
        pytest.skip("set COREBENCH_OFFICIAL_EVALUATOR to the pinned official evaluations.py")
    grader_python = os.environ.get("COREBENCH_GRADER_PYTHON")
    if not grader_python:
        for dependency in ("numpy", "scipy", "openai", "backoff", "tqdm"):
            pytest.importorskip(dependency)
        grader_python = sys.executable
    run, output = _native_chain(tmp_path, inputs, monkeypatch,
        root_report='{ "answer": "MECHANISM_RESULT" }\n' if root_recovery else None,
        fail_child=root_recovery)
    grade_output = tmp_path / "private-grade" / "results.json"
    # Only the toy input's pin changes in this test process. Production has no
    # bypass flag; official evaluator SHA/source/rules remain exactly pinned.
    fixture_entry = "import sys; from e2e import corebench; corebench.TRAIN_SHA256 = sys.argv[1]; raise SystemExit(corebench.main(sys.argv[2:]))"
    subprocess.run([grader_python, "-c", fixture_entry, corebench.digest(inputs["dataset"]), "grade",
        "--dataset", str(inputs["dataset"]), "--evaluator", evaluator, "--capsule-id", inputs["capsule_id"],
        "--report", str(output / "environment" / "report.json"), "--export-receipt", str(output / "export.json"),
        "--output", str(grade_output)], cwd=Path(corebench.__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    result = json.loads(grade_output.read_text())
    assert result["summary"]["correct_tasks"] == result["summary"]["total_tasks"] == 1
    receipt = json.loads((output / "export.json").read_text())
    selected = run.artifacts[receipt["report_artifact_id"]]
    assert selected.producer.value == ("scientific" if root_recovery else "executor")
    assert result["provenance"]["report_sha256"] == selected.sha256
    assert "Scripted transport" in run.final_opinion.limitations[0]
