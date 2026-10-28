"""ScienceAgentBench verified bridge; execution uses the production application."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from uuid import uuid4

from e2e.corebench import digest, _write_receipt

REVISION = "c26e151ed601ba109dc4d35e057ff8e73fec469d"
OUTPUT_NAME = "scienceagentbench_program"
DATASET = "osunlp/ScienceAgentBench"
DATASET_REVISION = "9c6e96c9e74572e979b0930ee735041cef528cb7"
PARQUET_SHA256 = "c6f937863a220bd1762a00c20a0f79cc8dfca900b819bdb552150310731ae147"
ANNOTATIONS_SHA256 = "0095b7e258828fd2ebe2d3f876ba46894388e6e2259409ef2e85f1d71c4c5129"
MAX_PUBLIC_FILES = 50000
MAX_PUBLIC_BYTES = 8 * 1024**3

# These are the upstream base prompts (agent.py at REVISION), without its
# model-specific context truncation or its optional self-debug conversation.
SYSTEM_PROMPT = """You are an expert Python programming assistant that helps scientist users to write high-quality code to solve their tasks.
Given a user request, you are expected to write a complete program that accomplishes the requested task and save any outputs in the correct format.
Please wrap your program in a code block that specifies the script type, python. For example:
```python
print("Hello World!")
```"""
FORMAT_PROMPT = """Please keep your response concise and do not use a code block if it's not intended to be executed.
Please do not suggest a few line changes, incomplete program outline, or partial code that requires the user to modify.
Please do not use any interactive Python commands in your program, such as `!pip install numpy`, which will cause execution errors."""
REQUEST_PROMPT = "Here's the user request you need to work on:"
DATA_INFO_PROMPT = """You can access the dataset at `{dataset_path}`. Here is the directory structure of the dataset:
```
{dataset_folder_tree}
```
Here are some helpful previews for the dataset file(s):
{dataset_preview}"""


def _relative(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(c) < 32 for c in value)):
        raise ValueError("unsafe relative benchmark path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", "..", ".git"} for part in value.split("/")):
        raise ValueError("unsafe relative benchmark path")
    return path


def _no_links(path):
    path = Path(path).absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("benchmark paths must not contain symlinks")
    return path


def _row(annotations, annotations_sha256, dataset_revision, instance_id):
    if dataset_revision != DATASET_REVISION:
        raise ValueError("dataset revision does not match the pinned verified split")
    if annotations_sha256 != ANNOTATIONS_SHA256 or digest(annotations) != ANNOTATIONS_SHA256:
        raise ValueError("annotations do not match the pinned verified snapshot")
    rows = json.loads(Path(annotations).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("annotations must be the JSON array exported from the verified split")
    matches = [row for row in rows if str(row.get("instance_id")) == str(instance_id)]
    if len(matches) != 1 or not re.fullmatch(r"[0-9]+", str(instance_id)):
        raise ValueError("instance must occur exactly once in verified annotations")
    row = matches[0]
    for key in ("task_inst", "dataset_folder_tree", "gold_program_name", "output_fname", "eval_script_name"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError("missing benchmark annotation: " + key)
    if "dataset_preview" not in row or row["dataset_preview"] is not None and not isinstance(row["dataset_preview"], str):
        raise ValueError("invalid benchmark annotation: dataset_preview")
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.py", row["gold_program_name"]) or not re.fullmatch(r"[A-Za-z0-9_-]+\.py", row["eval_script_name"]):
        raise ValueError("invalid official program filename")
    _relative(row["output_fname"].rstrip("/"))
    return row


def snapshot(*, parquet, output):
    """Convert the pinned public verified parquet to deterministic JSON."""
    if digest(parquet) != PARQUET_SHA256:
        raise ValueError("parquet does not match the pinned verified split")
    import pyarrow as pa
    import pyarrow.parquet as pq
    if int(pa.__version__.split(".")[0]) < 22:
        raise ValueError("snapshot requires pyarrow>=22; use a separate preparation environment")
    rows = pq.read_table(parquet).to_pylist()
    data = (json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    import hashlib
    if hashlib.sha256(data).hexdigest() != ANNOTATIONS_SHA256:
        raise ValueError("verified annotation export does not match the canonical snapshot")
    output = _no_links(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(data)
    return output


def _files(root):
    root = _no_links(root)
    files, total = {}, 0
    if not root.is_dir():
        raise ValueError("public dataset directory does not exist")
    for path in sorted(root.rglob("*")):
        _no_links(path)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("public dataset must contain regular files only")
        total += path.stat().st_size
        if len(files) >= MAX_PUBLIC_FILES or total > MAX_PUBLIC_BYTES:
            raise ValueError("public dataset exceeds preparation limits")
        relative = path.relative_to(root).as_posix()
        _relative(relative)
        files[relative] = digest(path)
    return files


def _task(row, knowledge):
    # Preserve upstream's first-line [4:] interpretation; do not guess a dataset
    # from the gold script or copy the entire benchmark folder.
    raw_folder = row["dataset_folder_tree"].split("\n")[0][4:]
    folder = _relative(raw_folder.rstrip("/")).as_posix()
    text = SYSTEM_PROMPT + "\n\n" + FORMAT_PROMPT + "\n\n" + REQUEST_PROMPT + "\n" + row["task_inst"]
    if knowledge:
        if "domain_knowledge" not in row:
            raise ValueError("expert knowledge is missing")
        text += "\n" + str(row["domain_knowledge"])
    text += "\n" + DATA_INFO_PROMPT.format(dataset_path="benchmark/datasets/" + raw_folder,
        dataset_folder_tree=row["dataset_folder_tree"], dataset_preview=row["dataset_preview"])
    return text, folder


def prepare(*, annotations, annotations_sha256, dataset_revision, benchmark, instance_id, output, knowledge=False):
    """Project only the selected task's public data from benchmark_verified."""
    row = _row(annotations, annotations_sha256, dataset_revision, instance_id)
    text, folder = _task(row, knowledge)
    source = _no_links(Path(benchmark) / "datasets" / folder)
    before = _files(source)
    output = _no_links(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix=".sab-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary) / "public"
        target = stage / "environment" / "benchmark" / "datasets" / folder
        shutil.copytree(source, target, symlinks=True)
        if _files(target) != before or _files(source) != before:
            raise ValueError("public inputs changed during preparation")
        (stage / "environment" / "task.txt").write_text(text, encoding="utf-8")
        (stage / "original-task.txt").write_text(row["task_inst"], encoding="utf-8")
        files = _files(stage)
        value = {"benchmark": "ScienceAgentBench", "revision": REVISION, "dataset": DATASET,
            "dataset_revision": dataset_revision, "annotations_sha256": annotations_sha256,
            "split": "verified", "instance_id": str(instance_id), "knowledge": knowledge,
            "program_filename": "pred_" + row["gold_program_name"], "output_fname": row["output_fname"],
            "dataset_folder": folder, "files": files}
        _write_receipt(stage / "manifest.json", value)
        os.rename(stage, output)
    return output / "manifest.json"


def _public(manifest):
    manifest = _no_links(manifest)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {"benchmark", "revision", "dataset", "dataset_revision", "annotations_sha256", "split",
                "instance_id", "knowledge", "program_filename", "output_fname", "dataset_folder", "files"}
    if (set(value) != expected or value["benchmark"] != "ScienceAgentBench" or value["revision"] != REVISION
            or value["dataset"] != DATASET or value["split"] != "verified" or not isinstance(value["knowledge"], bool)
            or value["dataset_revision"] != DATASET_REVISION or value["annotations_sha256"] != ANNOTATIONS_SHA256
            or not re.fullmatch(r"[0-9]+", value["instance_id"])
            or not re.fullmatch(r"pred_[A-Za-z0-9_-]+\.py", value["program_filename"])):
        raise ValueError("unexpected public manifest fields")
    folder = _relative(value["dataset_folder"]).as_posix()
    _relative(value["output_fname"].rstrip("/"))
    allowed = {"original-task.txt", "environment/task.txt"}
    prefix = "environment/benchmark/datasets/" + folder + "/"
    for relative in value["files"]:
        _relative(relative)
        if relative not in allowed and not relative.startswith(prefix):
            raise ValueError("unexpected public input file")
    actual = _files(manifest.parent)
    actual.pop(manifest.name, None)
    if set(actual) != set(value["files"]):
        raise ValueError("unexpected or missing public input files")
    if actual != value["files"]:
        raise ValueError("public input hash mismatch")
    return manifest, value


def _identity(public):
    return {key: public[key] for key in ("benchmark", "revision", "dataset_revision", "annotations_sha256",
        "split", "instance_id", "knowledge", "program_filename")}


def run(*, manifest, output, run_id=None, max_llm_calls=200, timeout_seconds=7200,
        max_tasks=8, max_attempts=2, max_work_depth=3, approval_mode="manual", execution_mode="live-model"):
    from sciagent_cli.composition import build_application
    from sciagent_contracts import ArtifactImport, ResearchRequest, RunBudget, RunPermissions, ExecutionLimits, WorkspaceSpec, WorkspaceSourceKind, WorkspaceAccess

    manifest, public = _public(manifest)
    output = _no_links(output)
    output.mkdir(parents=True, exist_ok=False)
    environment = output / "environment"
    shutil.copytree(manifest.parent / "environment", environment)
    expected = {name.removeprefix("environment/"): sha for name, sha in public["files"].items() if name.startswith("environment/")}
    if _files(environment) != expected:
        raise ValueError("public inputs changed while copying the execution workspace")
    subprocess.run(["git", "init", "--template=", "-q", str(environment)], check=True)
    run_id = run_id or "run_sab_" + public["instance_id"] + "_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    app = build_application(data_root=output / "data", workspaces={"ws_main": WorkspaceSpec(
        workspace_id="ws_main", source_kind=WorkspaceSourceKind.LOCAL, location=str(environment),
        access=WorkspaceAccess(read_paths=["."], write_paths=["."]))})
    imports = [ArtifactImport(uri=str(path), kind="benchmark_input", media_type=media, summary=summary, expected_sha256=digest(path))
        for path, media, summary in [(environment / "task.txt", "text/plain", "ScienceAgentBench verified task"),
            (manifest.parent / "original-task.txt", "text/plain", "Original unmodified task instruction"),
            (manifest, "application/json", "Public benchmark input manifest")]]
    request = ResearchRequest(goal=(environment / "task.txt").read_text(encoding="utf-8"), input_artifacts=imports,
        context=json.dumps(_identity(public), sort_keys=True),
        constraints=["Run the complete research workflow from this workspace root. For code implementation, select the available coding preset. "
            "Deliver one complete, self-contained Python program as a registered artifact with output_name=" + OUTPUT_NAME +
            "; media_type=text/x-python. Its official export filename is " + public["program_filename"] +
            ". Register the actual program file, not a fenced response or a Git patch. It must run from a fresh benchmark root containing only the public dataset, "
            "write the exact outputs requested by the original task (output_fname=" + public["output_fname"] +
            "), and not rely on temporary absolute workspace paths or undeclared helper scripts. "
            "The final Scientific opinion must cite the accepted program artifact unless it delivers its own accepted program. "
            "The independent official evaluator will rerun the frozen program; do not search for reference programs, evaluation scripts or reference results."],
        required_artifacts=[OUTPUT_NAME], budget=RunBudget(max_llm_calls=max_llm_calls, timeout_seconds=timeout_seconds),
        execution_limits=ExecutionLimits(max_tasks=max_tasks, max_attempts_per_task=max_attempts, max_work_depth=max_work_depth),
        permissions=RunPermissions(execute_commands=True, approval_mode=approval_mode))
    started = datetime.now(UTC)
    result = app.controller.create_run(run_id, request)
    receipt = {"run_id": run_id, "benchmark": "ScienceAgentBench", "public": public,
        "execution_mode": execution_mode, "status": result.status.value, "started_at": started.isoformat(),
        "returned_at": datetime.now(UTC).isoformat(), "budget": request.budget.model_dump(mode="json"),
        "execution_limits": request.execution_limits.model_dump(mode="json"), "model": os.environ.get("SCIAGENT_MODEL", "deepseek-v4-flash"),
        "usage": result.usage.model_dump(mode="json"), "answers": [item.model_dump(mode="json") for item in result.answers],
        "pending_question": result.pending_question.model_dump(mode="json") if result.pending_question else None}
    _write_receipt(output / "run.json", receipt)
    if result.status.value == "completed":
        export_saved(data_root=output / "data", run_id=run_id, output=output / "pred_programs" / public["program_filename"])
    return result


def export_program(run, destination):
    """Select the accepted top-level result, copying exact Registry bytes."""
    from sciagent_components import RegisteredArtifactReader
    if run.status.value != "completed" or run.final_opinion is None or run.final_report_artifact_id not in run.artifacts:
        raise ValueError("only accepted completed research runs may export")

    def accepted(ref):
        try:
            task = run.task(ref.task_id)
        except (StopIteration, ValueError):
            return False
        last = task.attempts[-1] if task.attempts else None
        return (ref.run_id == run.run_id and task.status.value == "completed" and ref.producer.value == task.workflow_agent_kind.value
            and last is not None and ref.id in last.artifact_ids and ref.attempt_number == last.number
            and last.status.value in {"completed", "completed_with_warnings"} and last.finished_at is not None and last.error is None
            and (ref.producer.value != "scientific" or (last.session is not None and run.scientific_session is not None
                and ref.session_id == last.session.id == run.scientific_session.id)))

    try:
        root = run.task("task_root")
        root_ids = set(root.attempts[-1].artifact_ids)
    except (StopIteration, ValueError, IndexError):
        raise ValueError("accepted root Attempt is missing") from None
    outputs = [ref for ref in run.artifacts.values() if ref.output_name == OUTPUT_NAME]
    refs = [ref for ref in outputs if ref.id in root_ids]
    if refs and any(ref.task_id != root.id or not accepted(ref) for ref in refs):
        raise ValueError("root program lacks accepted Task/Attempt/Session provenance")
    if not refs:
        refs = [ref for ref in outputs if ref.id in run.final_opinion.evidence_artifact_ids and accepted(ref)]
    if len(refs) != 1:
        raise ValueError("scienceagentbench_program must identify exactly one accepted registered output")
    ref = refs[0]
    if ref.media_type not in {"text/x-python", "text/plain", "application/x-python-code"}:
        raise ValueError("program must have a Python text media type")
    source = RegisteredArtifactReader(list(run.artifacts.values()), run_id=run.run_id).verified_path(ref.id)
    destination = _no_links(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".sab-program-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
        if digest(temporary) != ref.sha256:
            raise ValueError("export changed program bytes")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return ref


def export_saved(*, data_root, run_id, output):
    from sciagent_orchestrator import JsonRunStore
    run = JsonRunStore(Path(data_root) / "state").load(run_id)
    ref = export_program(run, output)
    receipt = {"run_id": run_id, "status": run.status.value, "program_artifact_id": ref.id, "program_sha256": ref.sha256,
        "benchmark_identity": json.loads(run.request.context),
        "program_provenance": ref.model_dump(mode="json", include={"producer", "task_id", "attempt_number", "session_id", "metadata"}),
        "budget": run.request.budget.model_dump(mode="json"), "usage": run.usage.model_dump(mode="json"),
        "answers": [answer.model_dump(mode="json") for answer in run.answers], "user_wait_seconds": run.user_wait_seconds,
        "created_at": run.created_at.isoformat(), "updated_at": run.updated_at.isoformat(), "final_report_artifact_id": run.final_report_artifact_id}
    _write_receipt(Path(data_root).resolve().parent / "export.json", receipt)
    return ref


def grade(*, annotations, annotations_sha256, dataset_revision, benchmark, instance_id, upstream,
          program, export_receipt, output, timeout_seconds=7200):
    """Run upstream's unchanged Docker evaluator on an independent grading host.

    Only one pinned verified row is submitted, and only its actual result.json
    counts. Upstream main()'s zero placeholders for unselected rows are avoided.
    Gold/evaluators must never be placed on the Agent execution host.
    """
    row = _row(annotations, annotations_sha256, dataset_revision, instance_id)
    receipt = json.loads(Path(export_receipt).read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("status") != "completed" or receipt.get("program_sha256") != digest(program):
        raise ValueError("program does not match accepted export receipt")
    identity = receipt.get("benchmark_identity", {})
    expected = {"benchmark": "ScienceAgentBench", "revision": REVISION, "dataset_revision": dataset_revision,
        "annotations_sha256": annotations_sha256, "split": "verified", "instance_id": str(instance_id),
        "program_filename": "pred_" + row["gold_program_name"]}
    if not isinstance(identity, dict) or not isinstance(identity.get("knowledge"), bool) or any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("export belongs to a different benchmark task or version")
    upstream = _no_links(upstream)
    revision = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if revision != REVISION:
        raise ValueError("evaluator does not match pinned official revision")
    benchmark = _no_links(benchmark)
    # A filename alone is not proof that the private assets are the verified
    # release. Retain their exact hashes with the grading receipt.
    private_files = _files(benchmark)
    for name in ("gold_programs/" + row["gold_program_name"], "eval_programs/" + row["eval_script_name"]):
        if name not in private_files:
            raise ValueError("official verified benchmark assets are missing")
    output = _no_links(output)
    output.mkdir(parents=True, exist_ok=False)
    evaluator = output / "upstream"
    evaluator.mkdir()
    # Export the immutable Git tree, not potentially modified working files or
    # untracked Python modules. No monkeypatch of upstream loaders or scoring.
    archive = output / "upstream.tar"
    with archive.open("xb") as handle:
        subprocess.run(["git", "-C", str(upstream), "archive", REVISION], check=True, stdout=handle)
    with tarfile.open(archive) as bundle:
        bundle.extractall(evaluator, filter="data")
    archive.unlink()
    predictions = output / "pred_programs"
    predictions.mkdir()
    prediction = predictions / ("pred_" + row["gold_program_name"])
    shutil.copyfile(program, prediction)
    if digest(prediction) != receipt["program_sha256"]:
        raise ValueError("program changed before grading")
    # Official image names only use instance_id. A fresh grade explicitly
    # rebuilds instance images so another candidate cannot reuse stale code.
    row = dict(row, instance_id=str(instance_id))
    _write_receipt(output / "example.json", row)
    grading_run_id = "sciagent_" + uuid4().hex
    worker = "\n".join([
        "import fcntl, hashlib, json, tempfile",
        "from pathlib import Path",
        "from evaluation.harness.run_evaluation import run_instances",
        "from evaluation.harness.docker_build import build_base_images",
        "import docker",
        "e = json.loads(Path('../example.json').read_text())",
        "c = docker.from_env()",
        "daemon = hashlib.sha256(c.info()['ID'].encode()).hexdigest()[:24]",
        "lock = open(Path(tempfile.gettempdir()) / ('sciagent-sab-' + daemon + '-' + e['instance_id'] + '.lock'), 'a')",
        "fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)",
        "build_base_images(c, [e], " + repr(str(benchmark)) + ", " + repr(str(predictions)) + ", False)",
        "run_instances([e], " + repr(str(benchmark)) + ", " + repr(str(predictions)) + ", 'base', False, True, 1, " + repr(grading_run_id) + ", 1800)",
    ])
    result_path = evaluator / "logs/run_evaluation" / grading_run_id / str(instance_id) / "output/result.json"
    started = datetime.now(UTC)
    result, error, cleanup = None, None, None
    try:
        if not os.environ.get("OPENAI_API_KEY") and not all(os.environ.get(name) for name in (
                "AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME")):
            raise RuntimeError("official grader requires complete OpenAI or Azure judge credentials")
        with (output / "grader.log").open("xb") as log:
            completed = subprocess.run([sys.executable, "-c", worker], cwd=evaluator, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_seconds)
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError("official grader failed or did not produce result.json; inspect grader.log")
        candidate = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(candidate, list) or len(candidate) != 4 or candidate[0] not in (0, 1) or candidate[2] not in (0, 1):
            raise ValueError("unexpected official grader result")
        if _files(benchmark) != private_files:
            raise ValueError("official benchmark assets changed during grading")
        result = candidate
    except subprocess.TimeoutExpired as exc:
        error = type(exc).__name__ + ": official grader exceeded its outer deadline"
        # Killing the worker does not kill a daemon-owned container. Remove only
        # this invocation's exact container; never prune shared Docker state.
        cleanup = _cleanup_container("sab.eval." + str(instance_id) + "." + grading_run_id)
    except (OSError, RuntimeError, ValueError) as exc:
        error = type(exc).__name__ + ": " + str(exc)
    value = {"benchmark": "ScienceAgentBench", "split": "verified", "revision": REVISION,
        "dataset_revision": dataset_revision, "annotations_sha256": annotations_sha256, "instance_id": str(instance_id), "knowledge": identity["knowledge"],
        "grading_run_id": grading_run_id,
        "timeout_container_cleanup": cleanup,
        "program_sha256": receipt["program_sha256"], "export_receipt_sha256": digest(export_receipt), "private_asset_hashes": private_files,
        "status": "graded" if result is not None else "infrastructure_error", "error": error,
        "official_result": result, "valid_program": result[0] if result else None,
        "codebert_score": result[1] if result else None, "success_rate": result[2] if result else None,
        "log_info": result[3] if result else None, "attempts": 1, "aggregation": "single attempt; not official best-of-three",
        "started_at": started.isoformat(), "finished_at": datetime.now(UTC).isoformat(),
        "judge_cost": None, "generation_usage": receipt.get("usage"), "generation_answers": receipt.get("answers")}
    _write_receipt(output / "grade.json", value)
    if error:
        raise RuntimeError(error)
    return value


def _cleanup_container(name):
    try:
        import docker
        with docker.from_env(timeout=15) as client:
            try:
                client.containers.get(name).remove(force=True)
            except docker.errors.NotFound:
                return {"container": name, "status": "not_found"}
        return {"container": name, "status": "removed"}
    except Exception as exc:
        return {"container": name, "status": "cleanup_failed", "error_type": type(exc).__name__}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = commands.add_parser("snapshot")
    for name in ("parquet", "output"):
        snapshot_parser.add_argument("--" + name, required=True)
    prepare_parser = commands.add_parser("prepare")
    grade_parser = commands.add_parser("grade")
    for command in (prepare_parser, grade_parser):
        for name in ("annotations", "annotations-sha256", "dataset-revision", "benchmark", "instance-id", "output"):
            command.add_argument("--" + name, required=True)
    prepare_parser.add_argument("--knowledge", action="store_true")
    for name in ("upstream", "program", "export-receipt"):
        grade_parser.add_argument("--" + name, required=True)
    grade_parser.add_argument("--timeout-seconds", type=int, default=7200)
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
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = (export_saved if command == "export" else globals()[command])(**args)
    if command == "run":
        print(json.dumps({"run_id": result.run_id, "status": result.status.value}))
        return 0 if result.status.value == "completed" else 2
    print(result.id if command == "export" else json.dumps(result) if command == "grade" else str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
