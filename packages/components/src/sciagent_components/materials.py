"""Complete task facts and direct artifact navigation shared by all Agents."""

from __future__ import annotations

import json
from collections.abc import Sequence
from sciagent_contracts import AgentRequest, BusinessInput, DatasetRef, RecordedAnswer
from sciagent_runtime import AgentState, ContextSection
from .artifacts import ArtifactReadError, RegisteredArtifactReader


def read_artifact_json(reader: RegisteredArtifactReader, artifact_id: str, model=None):
    window = reader.read_text(artifact_id, max_chars=16_000_000)
    if window.get("truncated"):
        raise ArtifactReadError("structured artifact exceeds the reading limit")
    value = json.loads(window["content"])
    return model.model_validate(value) if model is not None else value


_RUNTIME_TASK_KEYS = {
    "active_work_navigation", "active_work_request_id", "child_tasks", "run_permissions",
}


def runtime_authority_context(request: AgentRequest) -> ContextSection:
    """Actual authority and child-control facts, separate from business instructions."""
    value = request.model_dump(mode="json", exclude_none=True, include={
        "agent", "permissions", "workspace", "read_workspaces", "workspace_id", "workspace_spec", "output_dir",
        "work_depth", "max_work_depth",
    })
    value["source"] = "runtime"
    value["current_agent"] = request.agent.value
    value["current_agent_permissions"] = value.pop("permissions")
    value["run_execution_permissions"] = request.task_context.get("run_permissions")
    if request.permissions.request_work and request.work_depth < request.max_work_depth:
        value["available_execution_presets"] = [item.model_dump(mode="json") for item in request.execution_catalog]
    if request.execution_definition is not None:
        value["execution_ref"] = request.execution_definition.ref
    value["child_work"] = {key: request.task_context[key] for key in (
        "active_work_request_id", "child_tasks"
    ) if key in request.task_context}
    return ContextSection(name="runtime_authority", content=json.dumps(value, ensure_ascii=False))


def request_task_context(request: AgentRequest) -> dict:
    """Initial facts and complete business dialogue; no operation approvals."""
    value = request.model_dump(mode="json", exclude_none=True, include={
        "run_id", "task_id", "attempt_number", "work_request_id", "instruction", "task_context",
    })
    value["task_context"] = {key: item for key, item in value["task_context"].items()
                             if key not in _RUNTIME_TASK_KEYS}
    value["business_dialogue"] = ordered_business_dialogue(
        upstream_inputs=request.upstream_inputs, answers=request.user_answers, input_ids=request.input_ids)
    return value


def ordered_business_dialogue(*, upstream_inputs: Sequence[BusinessInput],
                              answers: Sequence[RecordedAnswer], input_ids: Sequence[str]) -> list[dict]:
    """Project complete business records in the caller's trusted reception order."""
    if any(answer.action is not None for answer in answers):
        raise ValueError("operation approvals are not business dialogue")
    records = (
        [{"event_id": event.event_id, "created_at": event.created_at.isoformat(),
          "type": "input", "record": event.model_dump(mode="json", exclude_none=True)}
         for event in upstream_inputs]
        + [{"event_id": answer.question_id, "created_at": answer.answered_at.isoformat(),
            "type": "answer", "record": answer.model_dump(mode="json", exclude_none=True)}
           for answer in answers]
    )
    by_id = {item["event_id"]: item for item in records}
    if (len(by_id) != len(records) or len(set(input_ids)) != len(input_ids)
            or set(by_id) != set(input_ids)):
        raise ValueError("provided input identities must exactly match the complete business dialogue")
    # Reception order is authoritative. Source/client timestamps are retained
    # for provenance and cannot move a later amendment before an earlier one.
    return [by_id[identity] for identity in input_ids]


def request_dataset_refs(request: AgentRequest) -> list[DatasetRef]:
    """Resolve the declared catalog without projecting material bodies to context."""
    values = request.task_context.get("dataset_refs")
    if values is not None:
        return [DatasetRef.model_validate(value) for value in values]
    reader = RegisteredArtifactReader(request.input_artifacts, run_id=request.run_id)
    refs = {}
    for artifact in request.input_artifacts:
        if artifact.kind != "dataset_catalog":
            continue
        for item in read_artifact_json(reader, artifact.id)["datasets"]:
            ref = DatasetRef.model_validate(item)
            if ref.dataset_id in refs and refs[ref.dataset_id] != ref:
                raise ArtifactReadError("dataset remapped during invocation")
            refs[ref.dataset_id] = ref
    return list(refs.values())
