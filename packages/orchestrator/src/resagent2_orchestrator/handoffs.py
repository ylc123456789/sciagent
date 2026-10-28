"""Shared result reception and immutable structured handoffs."""

import json
import math
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from resagent2_contracts import (
    AgentOwner, ArtifactCandidate, ArtifactRef, TaskAcceptanceSpec,
    SYSTEM_ARTIFACT_PROVENANCE,
)
from .artifacts import ArtifactRegistrationError, _sha256


def read_json(ref: ArtifactRef, model=None):
    parsed = urlparse(ref.uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ArtifactRegistrationError("only registered local artifacts may be read")
    path = Path(url2pathname(parsed.path))
    if not path.is_file() or _sha256(path) != ref.sha256:
        raise ArtifactRegistrationError("registered artifact is missing or its hash changed")
    data = json.loads(path.read_text(encoding="utf-8"))
    return model.model_validate(data) if model else data


def system_artifact(registry, run, kind, data, *, session_id=None, task_id=None, attempt_number=None,
                    operation_key=None):
    source = SYSTEM_ARTIFACT_PROVENANCE[kind][0]
    content = data.model_dump(mode="json") if hasattr(data, "model_dump") else data
    key = operation_key or f"system:{kind}:{session_id or task_id or run.run_id}:{attempt_number or 0}"
    candidate = ArtifactCandidate(kind=kind, path=f"{kind}.json", media_type="application/json",
                                  summary=kind.replace("_", " "),
                                  content=json.dumps(content, ensure_ascii=False, sort_keys=True))
    ref = registry.register_system_artifact(candidate, run_id=run.run_id, source_type=source,
                                           session_id=session_id, task_id=task_id,
                                           attempt_number=attempt_number, operation_key=key, run=run)
    run.artifacts[ref.id] = ref
    return ref


def receive_artifacts(registry, run, request, result, *, previous_ids=(), checkpoint=None):
    """Validate the entire handoff before preparing and freezing its output group."""
    offset = len(previous_ids)
    names = {run.artifacts[key].output_name for key in previous_ids
             if key in run.artifacts and run.artifacts[key].output_name is not None}
    returned_ids = set()
    keys = []
    for index, output in enumerate(result.artifacts):
        if output.output_name is not None:
            if output.output_name in names:
                raise ArtifactRegistrationError("duplicate logical output name")
            names.add(output.output_name)
        if isinstance(output, ArtifactRef):
            ref = run.artifacts.get(output.id)
            if ref is None or ref != output or ref.run_id != run.run_id:
                raise ArtifactRegistrationError("returned Ref is not in the trusted registration record")
            if request.task_id is not None:
                owned = ref.task_id == request.task_id and ref.attempt_number == request.attempt_number
            else:
                owned = result.session is not None and ref.session_id == result.session.id
            if not owned or ref.producer != request.agent or output.id in {a.id for a in request.input_artifacts}:
                raise ArtifactRegistrationError("input or foreign artifact cannot be returned as a new output")
            if ref.id in returned_ids:
                raise ArtifactRegistrationError("duplicate returned artifact reference")
            returned_ids.add(ref.id)
        if request.task_id is None and result.session is None:
            raise ArtifactRegistrationError("session output requires its owning session")
        keys.append(f"task:{request.task_id}:attempt:{request.attempt_number}:output:{offset + index + 1}"
                    if request.task_id is not None else
                    f"session:{result.session.id}:result:{result.session.updated_at.isoformat()}:{index}")
    return registry.register_many(result.artifacts, grant=request.workspace, producer=request.agent,
        run_id=run.run_id, task_id=request.task_id, attempt_number=request.attempt_number,
        start_index=offset + 1, output_dir=request.output_dir, operation_keys=keys,
        session_id=result.session.id if result.session is not None else None,
        run=run, checkpoint=checkpoint)


def control_key(control):
    """Provider IDs are unique only within their original native response."""
    return f"session:{control.source_session_id}:turn:{control.source_turn}:call:{control.source_call_id}"


def receive_control(run, request, result):
    """Receive a native control from its actual Task/Attempt owner."""
    signal = result.control
    if signal is None:
        return None
    if result.session is None or signal.source_session_id != result.session.id:
        raise ArtifactRegistrationError("control source session mismatch")
    if request.parent_session_id and signal.source_session_id != request.parent_session_id:
        raise ArtifactRegistrationError("control does not belong to resumed session")
    key = control_key(signal)
    payload = signal.question if signal.action == "ask_user" else signal.work_request
    if payload is None:
        raise ArtifactRegistrationError("parent updates require the owner receiving gate")
    existing = run.received_controls.get(key)
    if existing is not None:
        if existing != payload.id:
            raise ArtifactRegistrationError("control identity conflicts with previously received call")
        return payload
    if signal.action == "ask_user":
        question = signal.question
        if (question.run_id, question.task_id, question.attempt_number, question.session_id) != (
                request.run_id, request.task_id, request.attempt_number, signal.source_session_id):
            raise ArtifactRegistrationError("question control scope mismatch")
        if question.action is None and request.upstream_session_id is not None:
            if question.recipient_session_id != request.upstream_session_id:
                raise ArtifactRegistrationError("question recipient is not the direct parent")
            run.child_questions[question.id] = question
            owner = run.work(request.work_request_id).owner_task_id
            if owner is not None and owner not in run.attention_task_ids:
                run.attention_task_ids.append(owner)
            run.runtime_notices.append({"event_id": "question:" + question.id,
                "type": "child_question", "source_session_id": question.session_id,
                "recipient_session_id": question.recipient_session_id,
                "question": question.model_dump(mode="json")})
        else:
            if run.pending_question is not None:
                raise ArtifactRegistrationError("a user question is already pending")
            run.pending_question = question
    else:
        work = signal.work_request
        if not request.permissions.request_work or request.work_depth >= request.max_work_depth:
            raise ArtifactRegistrationError("delegation is not granted at this depth")
        if (work.run_id, work.owner_task_id, work.owner_attempt_number, work.owner_session_id, work.depth) != (
                run.run_id, request.task_id, request.attempt_number, signal.source_session_id, request.work_depth + 1):
            raise ArtifactRegistrationError("work control owner or depth mismatch")
        if run.active_child(request.task_id, request.attempt_number) is not None:
            raise ArtifactRegistrationError("this Attempt already has unconsumed child work")
        if work.processed_input_ids != request.input_ids:
            raise ArtifactRegistrationError("child work input baseline differs from its owner request")
        run.work_requests.append(work)
        if request.agent == AgentOwner.SCIENTIFIC:
            run.latest_scientific_assessment = signal.assessment
    run.received_controls[key] = payload.id
    return payload


def check_acceptance(run, task, attempt, refs):
    if attempt.acceptance_ref != task.acceptance_ref:
        raise ArtifactRegistrationError("Attempt acceptance binding differs from Task")
    if task.acceptance_ref is None:
        return
    ref = task.acceptance_ref
    if run.artifacts.get(ref.id) != ref or ref.task_id != task.id or ref.kind != "acceptance_requirements":
        raise ArtifactRegistrationError("invalid acceptance binding")
    spec = read_json(ref, TaskAcceptanceSpec)
    if any(item.run_id != run.run_id or item.task_id != task.id or item.attempt_number != attempt.number for item in refs):
        raise ArtifactRegistrationError("acceptance can inspect only this Attempt's outputs")
    names = [item.output_name for item in refs if item.output_name is not None]
    if len(names) != len(set(names)) or not set(spec.required_output_names) <= set(names):
        raise ArtifactRegistrationError("required logical outputs missing or ambiguous")
    if not set(spec.required_artifact_kinds) <= {item.kind for item in refs}:
        raise ArtifactRegistrationError("required artifact kinds missing")
    paths = {item.metadata.get("source_path") for item in refs}
    if not set(spec.required_artifact_paths) <= paths:
        raise ArtifactRegistrationError("required artifact paths missing")
    numeric = set()
    for item in refs:
        if item.media_type != "application/json":
            continue
        data = read_json(item)
        if isinstance(data, dict):
            numeric.update(key for key, value in data.items() if isinstance(value, (int, float))
                           and not isinstance(value, bool) and math.isfinite(value))
    if not set(spec.required_metric_keys) <= numeric:
        raise ArtifactRegistrationError("required numeric metric keys missing")
