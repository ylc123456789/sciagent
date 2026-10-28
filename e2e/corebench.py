"""Small CORE-Bench Hard bridge; research execution stays in production packages."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime

REVISION = "e32a2980e72fe6eb04ee04eb749458f570625663"
TRAIN_SHA256 = "3df47f1b3fa1cb60045018eb1a0f1ad4ecf6a53f72318c845a879ce0313b0730"
PROMPTS_SHA256 = "e70b63c40671f3924a8d71aac0e1c64a5dae3c8cf0bf391471c488581fff53b8"
GRADER_SHA256 = "88a132c1196bfd9f1f8cf32d23e64cb6e0b26d98437ab5ff72f5e77ef56b53bb"
OUTPUT_NAME = "corebench_report"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_receipt(path, value):
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("receipt destination must not contain symlinks")
    descriptor, temporary = tempfile.mkstemp(prefix=".corebench-receipt-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _pinned_json(path, expected):
    if digest(path) != expected:
        raise ValueError("upstream file does not match pinned CORE-Bench revision")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _record(dataset, capsule_id):
    if not re.fullmatch(r"capsule-\d{7}", capsule_id):
        raise ValueError("invalid capsule id")
    matches = [row for row in dataset if row.get("capsule_id") == capsule_id]
    if len(matches) != 1:
        raise ValueError("capsule must occur exactly once in train")
    row = matches[0]
    if not isinstance(row.get("task_prompt"), str) or not row["task_prompt"].strip():
        raise ValueError("task_prompt must be nonblank text")
    answers = row.get("results")
    if not isinstance(answers, list) or not answers or not isinstance(answers[0], dict) or not answers[0]:
        raise ValueError("results must contain nonempty question dictionaries")
    keys = set(answers[0])
    for answer in answers:
        if not isinstance(answer, dict) or set(answer) != keys:
            raise ValueError("inconsistent question keys")
        for key, value in answer.items():
            if not isinstance(key, str) or not key.strip() or isinstance(value, bool) or not isinstance(value, (str, int, float, list)):
                raise ValueError("unsupported gold schema")
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("nonfinite gold")
    return row


def _unpack(archive, destination, capsule_id):
    """Validate every tar member before writing; never execute archive contents."""
    if Path(archive).stat().st_size > 256 * 1024 * 1024:
        raise ValueError("compressed capsule exceeds 256 MiB limit")
    with tarfile.open(archive, "r:gz") as bundle:
        members, expanded_size = [], 0
        seen = set()
        for member in bundle:
            members.append(member)
            expanded_size += member.size
            if len(members) > 20000 or expanded_size > 1024 * 1024 * 1024:
                raise ValueError("capsule exceeds extraction limits")
            name = member.name.rstrip("/")
            path = PurePosixPath(name)
            if (not name or "\\" in name or ":" in name or any(ord(c) < 32 for c in name)
                    or path.is_absolute() or ".." in path.parts or path.parts[0] != capsule_id
                    or path.as_posix() in seen or not (member.isdir() or member.isfile())):
                raise ValueError("unsafe capsule archive member")
            seen.add(path.as_posix())
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)


def prepare(*, dataset, prompts, capsule_id, archive, archive_sha256, output):
    """Create only the public Hard input; never persist the train/gold record."""
    row = _record(_pinned_json(dataset, TRAIN_SHA256), capsule_id)
    template = _pinned_json(prompts, PROMPTS_SHA256)["codeocean_hard"]
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha256) or digest(archive) != archive_sha256:
        raise ValueError("capsule archive does not match the trusted source hash")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".corebench-", dir=output.parent) as temporary:
        stage = Path(temporary) / "public"
        environment = stage / "environment"
        environment.mkdir(parents=True)
        _unpack(archive, environment, capsule_id)
        capsule = environment / capsule_id
        if not capsule.is_dir():
            raise ValueError("capsule archive has no capsule directory")
        # Match official benchmark.py Hard removal rules, including empty results.
        for relative in ("results", "REPRODUCING.md", "environment", "code/run.sh", "code/run"):
            target = capsule / relative
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        (capsule / "results").mkdir()
        if any(path.name == ".git" for path in capsule.rglob("*")):
            raise ValueError("capsule must not supply Git metadata")
        task = template.replace("{task_prompt}", row["task_prompt"]).replace("{json_fields}", str(row["results"][0].keys()))
        (environment / "task.txt").write_bytes(task.encode("utf-8"))
        (stage / "original-task-prompt.txt").write_bytes(row["task_prompt"].encode("utf-8"))
        files = {path.relative_to(stage).as_posix(): digest(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        manifest = {"benchmark": "CORE-Bench", "revision": REVISION, "split": "train", "difficulty": "codeocean_hard",
                    "capsule_id": capsule_id, "question_keys": list(row["results"][0]),
                    "archive_sha256": digest(archive), "files": files}
        (stage / "task.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        subprocess.run(["git", "init", "--template=", "-q", str(environment)], check=True)
        stage.rename(output)
    return output / "task.json"


def _public(manifest):
    manifest = Path(manifest).resolve()
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if (value.get("revision"), value.get("split"), value.get("difficulty")) != (REVISION, "train", "codeocean_hard"):
        raise ValueError("unsupported public task manifest")
    expected_fields = {"benchmark", "revision", "split", "difficulty", "capsule_id", "question_keys", "archive_sha256", "files"}
    if set(value) != expected_fields or value["benchmark"] != "CORE-Bench" or not re.fullmatch(r"capsule-\d{7}", value["capsule_id"]):
        raise ValueError("unexpected public manifest fields")
    actual = set()
    for path in manifest.parent.rglob("*"):
        if ".git" in path.relative_to(manifest.parent).parts:
            continue
        if path.is_symlink():
            raise ValueError("public inputs must not contain links")
        if path.is_file() and path != manifest:
            actual.add(path.relative_to(manifest.parent).as_posix())
    if actual != set(value["files"]):
        raise ValueError("unexpected or missing public input files")
    for relative, expected in value["files"].items():
        path = (manifest.parent / relative).resolve()
        if not path.is_relative_to(manifest.parent) or digest(path) != expected:
            raise ValueError("public input hash mismatch")
    return manifest, value


def run(*, manifest, output, run_id=None, max_llm_calls=200, timeout_seconds=7200,
        max_tasks=8, max_attempts=2, max_work_depth=3, approval_mode="manual", execution_mode="live-model"):
    """Invoke the production application, fixed Scientific root and real Registry."""
    from sciagent_cli.composition import build_application
    from sciagent_contracts import ArtifactImport, ResearchRequest, RunBudget, RunPermissions, ExecutionLimits, WorkspaceSpec, WorkspaceSourceKind, WorkspaceAccess

    manifest, public = _public(manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Every run has its own mutable workspace; preserve prepared public input.
    environment = output / "environment"
    shutil.copytree(manifest.parent / "environment", environment, ignore=shutil.ignore_patterns(".git"))
    subprocess.run(["git", "init", "--template=", "-q", str(environment)], check=True)
    run_id = run_id or "run_corebench_" + public["capsule_id"].replace("-", "_") + "_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    app = build_application(data_root=output / "data", workspaces={"ws_main": WorkspaceSpec(
        workspace_id="ws_main", source_kind=WorkspaceSourceKind.LOCAL, location=str(environment),
        access=WorkspaceAccess(read_paths=["."], write_paths=["."]))})
    imports = [ArtifactImport(uri=str(path), kind="benchmark_input", media_type=media, summary=summary, expected_sha256=digest(path))
               for path, media, summary in [(environment / "task.txt", "text/plain", "Original CORE-Bench Hard task"),
                    (manifest.parent / "original-task-prompt.txt", "text/plain", "Original CORE-Bench task prompt bytes"),
                    (manifest, "application/json", "Public CORE-Bench input manifest")]]
    request = ResearchRequest(goal=(environment / "task.txt").read_text(encoding="utf-8"), input_artifacts=imports,
        constraints=["The workspace root is the environment directory containing the capsule. Deliver the JSON report as a registered artifact with output_name=corebench_report and media_type=application/json. The accepted root Scientific attempt may supply the report directly; otherwise cite the accepted delegated report in the final scientific opinion. The adapter exports its frozen bytes to the outer task environment/report.json. Preserve the original task and answer all its questions."],
        required_artifacts=[OUTPUT_NAME], budget=RunBudget(max_llm_calls=max_llm_calls, timeout_seconds=timeout_seconds),
        execution_limits=ExecutionLimits(max_tasks=max_tasks, max_attempts_per_task=max_attempts, max_work_depth=max_work_depth),
        permissions=RunPermissions(execute_commands=True, approval_mode=approval_mode))
    started = datetime.now(UTC)
    result = app.controller.create_run(run_id, request)
    receipt = {"run_id": run_id, "capsule_id": public["capsule_id"], "revision": REVISION, "execution_mode": execution_mode,
        "status": result.status.value, "started_at": started.isoformat(), "finished_at": datetime.now(UTC).isoformat(),
        "request_budget": request.budget.model_dump(mode="json"), "execution_limits": request.execution_limits.model_dump(mode="json"),
        "model": os.environ.get("SCIAGENT_MODEL", "deepseek-v4-flash"), "usage": result.usage.model_dump(mode="json"),
        "user_wait_seconds": result.user_wait_seconds, "answers": [answer.model_dump(mode="json") for answer in result.answers],
        "pending_question": result.pending_question.model_dump(mode="json") if result.pending_question else None,
        "token_and_cost_accounting": "See configured LLM metadata traces; absent provider usage is unknown, never zero.",
        "final_report_artifact_id": result.final_report_artifact_id}
    _write_receipt(output / "run.json", receipt)
    if result.status.value == "completed":
        export_saved(data_root=output / "data", run_id=run_id, output=environment / "report.json")
    return result


def export_report(run, destination):
    """Export frozen bytes, never trust an unregistered workspace report.json."""
    from sciagent_components import RegisteredArtifactReader
    if run.status.value != "completed" or run.final_opinion is None or run.final_report_artifact_id not in run.artifacts:
        raise ValueError("only accepted completed research runs may export")
    def accepted_output(ref):
        # The completed final Attempt is the public receiving-gate record.
        # Retained failed/old outputs do not become delivery through citation.
        try:
            task = run.task(ref.task_id)
        except (StopIteration, ValueError):
            return False
        last = task.attempts[-1] if task.attempts else None
        return (ref.run_id == run.run_id and task.status.value == "completed"
                and ref.producer.value == task.workflow_agent_kind.value
                and last is not None and ref.id in last.artifact_ids
                and ref.attempt_number == last.number
                and last.status.value in {"completed", "completed_with_warnings"}
                and last.finished_at is not None and last.error is None
                and (ref.producer.value != "scientific" or (
                    last.session is not None and run.scientific_session is not None
                    and ref.session_id == last.session.id == run.scientific_session.id)))

    try:
        root = run.task("task_root")
        root_ids = set(root.attempts[-1].artifact_ids)
    except (StopIteration, ValueError, IndexError):
        raise ValueError("accepted root Attempt is missing") from None
    outputs = [ref for ref in run.artifacts.values() if ref.output_name == OUTPUT_NAME]
    # Prefer the top-level accepted result. Its inline candidate is registered
    # at reception, so the opinion cannot be required to know its future ID.
    refs = [ref for ref in outputs if ref.id in root_ids]
    if refs and any(ref.task_id != root.id or not accepted_output(ref) for ref in refs):
        raise ValueError("root report lacks accepted Task/Attempt/Session provenance")
    if not refs:
        refs = [ref for ref in outputs
                if ref.id in run.final_opinion.evidence_artifact_ids and accepted_output(ref)]
    if len(refs) != 1:
        raise ValueError("corebench_report must identify exactly one accepted registered output")
    ref = refs[0]
    if ref.media_type != "application/json":
        raise ValueError("report must have application/json media type")
    reader = RegisteredArtifactReader(list(run.artifacts.values()), run_id=run.run_id)
    source = reader.verified_path(ref.id)
    if not isinstance(json.loads(source.read_text(encoding="utf-8")), dict):
        raise ValueError("report must contain a JSON object")
    destination = Path(destination)
    if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("export destination must not contain symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".corebench-report-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
        if digest(temporary) != ref.sha256:
            raise ValueError("export changed report bytes")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return ref


def export_saved(*, data_root, run_id, output):
    """Export after the normal production CLI has answered/resumed a paused run."""
    from sciagent_orchestrator import JsonRunStore
    run = JsonRunStore(Path(data_root) / "state").load(run_id)
    ref = export_report(run, output)
    receipt = {"run_id": run_id, "status": run.status.value, "report_artifact_id": ref.id, "report_sha256": ref.sha256,
        "report_provenance": ref.model_dump(mode="json", include={"producer", "task_id", "attempt_number", "session_id", "metadata"}),
        "budget": run.request.budget.model_dump(mode="json"), "usage": run.usage.model_dump(mode="json"),
        "answers": [answer.model_dump(mode="json") for answer in run.answers], "user_wait_seconds": run.user_wait_seconds,
        "created_at": run.created_at.isoformat(), "updated_at": run.updated_at.isoformat(),
        "final_report_artifact_id": run.final_report_artifact_id}
    receipt_path = Path(data_root).resolve().parent / "export.json"
    _write_receipt(receipt_path, receipt)
    return ref


def grade(*, dataset, evaluator, capsule_id, report, output, export_receipt=None):
    """Run the hash-pinned official evaluator locally, outside the Agent host."""
    gold = _record(_pinned_json(dataset, TRAIN_SHA256), capsule_id)["results"]
    if digest(evaluator) != GRADER_SHA256:
        raise ValueError("evaluator does not match pinned official revision")
    report = Path(report)
    answer = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(answer, dict):
        raise ValueError("reported result must be a JSON object")
    if export_receipt:
        receipt = json.loads(Path(export_receipt).read_text(encoding="utf-8"))
        if receipt.get("status") != "completed" or receipt.get("report_sha256") != digest(report):
            raise ValueError("report does not match accepted export receipt")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec = importlib.util.spec_from_file_location("corebench_official_evaluations", evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    counts = official.eval_result_json(copy.deepcopy(gold), copy.deepcopy(answer))
    value = {"capsule_results": [{"capsule_id": capsule_id, "result_report": answer, **counts}],
             "provenance": {"revision": REVISION, "grader_sha256": GRADER_SHA256, "report_sha256": digest(report), "llm_summary": False,
                            "export_receipt_sha256": digest(export_receipt) if export_receipt else None}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    official.score_results(str(output), llm_summary=False)
    return json.loads(output.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in ("dataset", "prompts", "capsule-id", "archive", "archive-sha256", "output"):
        prepare_parser.add_argument("--" + name, required=True)
    run_parser = commands.add_parser("run")
    for name in ("manifest", "output"):
        run_parser.add_argument("--" + name, required=True)
    run_parser.add_argument("--run-id")
    for name, default in (("max-llm-calls", 200), ("timeout-seconds", 7200), ("max-tasks", 8), ("max-attempts", 2), ("max-work-depth", 3)):
        run_parser.add_argument("--" + name, type=int, default=default)
    run_parser.add_argument("--approval-mode", choices=["manual", "auto_review"], default="manual")
    run_parser.add_argument("--execution-mode", choices=["live-model", "offline-mechanism"], default="live-model")
    export_parser = commands.add_parser("export")
    for name in ("data-root", "run-id", "output"):
        export_parser.add_argument("--" + name, required=True)
    grade_parser = commands.add_parser("grade")
    for name in ("dataset", "evaluator", "capsule-id", "report", "output"):
        grade_parser.add_argument("--" + name, required=True)
    grade_parser.add_argument("--export-receipt", help="bind report bytes to the retained accepted export receipt")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = (export_saved if command == "export" else globals()[command])(**args)
    if command == "run":
        print(json.dumps({"run_id": result.run_id, "status": result.status.value}))
        return 0 if result.status.value == "completed" else 2
    print(result if command == "prepare" else result.id if command == "export" else json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
