"""Direct-parent business controls shared by task Agents."""
from __future__ import annotations

from pydantic import Field, model_validator
from sciagent_contracts import WorkRequestDraft
from sciagent_runtime import ToolObservation
from sciagent_runtime.models import RuntimeModel


class RequestWorkInput(RuntimeModel):
    work_request: WorkRequestDraft


class RequestWorkTool:
    """Delegate a self-contained set of tasks and wait for their actual results."""
    name = "request_work"
    input_model = RequestWorkInput
    model_guidance = (
        "Provide an objective for managing the work and explicit tasks. Each task has a local "
        "key, a self-contained instruction, optional depends_on local keys, input_artifact_ids, "
        "future input_artifact_bindings, output_names, and a declared logical workspace_id. "
        "Choose selection.kind=preset with a preset_ref from the runtime execution catalog and a reason, or selection.kind=no_match with the specific missing capability. Put every requirement a task must follow into that task's instruction; the objective "
        "is not automatically supplied as a child requirement. Select only declared workspaces "
        "and authorized registered inputs. Do not invent preset references, runtime identities, "
        "physical workspace paths or permissions. The Compiler deterministically binds your explicit choices "
        "without rewriting tasks; it cannot add, split or repair your tasks. A no_match selection "
        "returns the original task and reason without execution; it does not generate a configuration. Acceptance means "
        "the delegation was recorded, not that it has finished. Only one active delegation is "
        "allowed per attempt. All delegated work shares the Run budget and permission limits."
    )

    def __init__(self, *, request=None, reader=None, allowed=True):
        self.request = request
        self.reader = reader
        self.allowed = allowed

    def execute(self, state, arguments):
        if not self.allowed:
            raise PermissionError("This task cannot delegate work")
        if self.request is not None:
            if not self.request.permissions.request_work or self.request.work_depth >= self.request.max_work_depth:
                raise PermissionError("Delegation capability or depth limit does not allow this work")
            if self.request.task_context.get("active_work_request_id"):
                raise PermissionError("Consume or end your active delegation before requesting another")
        for task in arguments.work_request.tasks:
            for artifact_id in task.input_artifact_ids:
                ref = self.reader.resolve_ref(artifact_id) if self.reader else None
                if ref is None or ref.run_id != state.run_id:
                    return ToolObservation(ok=False, summary=f"Work input artifact is not authorized: {artifact_id}")
                try:
                    self.reader.verify(artifact_id)
                except (ValueError, OSError) as error:
                    return ToolObservation(ok=False, summary=f"Work input artifact is unavailable: {error}")
        return ToolObservation(summary="Work delegation submitted", request_work=arguments.model_dump(mode="json"))


class WorkInput(RuntimeModel):
    task_ids: list[str] = Field(min_length=1)
    text: str | None = None
    source_input_id: str | None = None

    @model_validator(mode="after")
    def validate_content(self):
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")
        if (self.text is None) == (self.source_input_id is None):
            raise ValueError("provide exactly one text or source_input_id")
        if self.text is not None and not self.text.strip():
            raise ValueError("text must contain non-whitespace characters")
        return self


class WorkAnswer(RuntimeModel):
    task_id: str
    question_id: str
    values: dict[str, str] | None = None
    source_answer_id: str | None = None

    @model_validator(mode="after")
    def validate_content(self):
        if (self.values is None) == (self.source_answer_id is None):
            raise ValueError("provide exactly one values or source_answer_id")
        return self


class UpdateWorkInput(RuntimeModel):
    work_request_id: str
    inputs: list[WorkInput] = Field(default_factory=list)
    answers: list[WorkAnswer] = Field(default_factory=list)


class UpdateWorkTool:
    """Append direct guidance or exact business answers to your selected child tasks."""
    name = "update_work"
    input_model = UpdateWorkInput
    model_guidance = (
        "Choose the exact active work_request_id and explicit recipient task_ids. Use inputs "
        "for new parent text or a source_input_id received by you, and answers for a specific "
        "child question_id with its exact requested values or a received source_answer_id. "
        "One submission may inform all live children, selected children, or no children, and "
        "may answer questions at the same time. Empty inputs and answers acknowledge that "
        "the provided upstream input requires no child update; they do not clear existing "
        "requirements. Inputs append in order: explicitly state what an amendment replaces. "
        "Initial instructions and earlier input remain available. Only exact question answers "
        "resolve question waits. Completed tasks cannot receive execution updates; request new "
        "work when they need to be redone. All selected updates are saved before child work "
        "continues. This tool grants no execution permission."
    )

    def __init__(self, request):
        self.request = request

    def execute(self, state, arguments):
        if arguments.work_request_id != self.request.task_context.get("active_work_request_id"):
            raise PermissionError("Can update only your own active delegation")
        allowed = {item["task_id"] for item in self.request.task_context.get("child_tasks", [])
                   if item.get("status") in {"pending", "running", "needs_user_input", "waiting_for_work"}}
        for item in arguments.inputs:
            if not set(item.task_ids) <= allowed:
                raise PermissionError("Update targets must be live direct child tasks")
            if item.source_input_id and not any(e.event_id == item.source_input_id for e in self.request.upstream_inputs):
                raise PermissionError("Input reference was not received by this task")
        seen = set()
        for item in arguments.answers:
            if item.question_id in seen:
                raise ValueError("A question can be answered only once per submission")
            seen.add(item.question_id)
            if item.task_id not in allowed:
                raise PermissionError("Answer target must be a live direct child")
            question = next((q for q in self.request.child_questions if q.id == item.question_id), None)
            if question is None or question.task_id != item.task_id or question.recipient_session_id != state.session_id or question.action is not None:
                raise PermissionError("Question does not belong to this parent and child")
            values = item.values
            if item.source_answer_id:
                answer = next((a for a in self.request.user_answers if a.question_id == item.source_answer_id and a.action is None), None)
                if answer is None:
                    raise PermissionError("Answer reference was not received by this task")
                values = answer.values
            if set(values) != set(question.requested_fields):
                raise ValueError("Answer must supply the exact requested fields")
        return ToolObservation(summary="Work update submitted", work_update=arguments.model_dump(mode="json"))


class AbandonWorkInput(RuntimeModel):
    work_request_id: str
    reason: str = Field(min_length=1)


class AbandonWorkTool:
    """End your own pending delegation while preserving actual results and files."""
    name = "abandon_work"
    input_model = AbandonWorkInput

    def __init__(self, request):
        self.request = request

    def execute(self, state, arguments):
        if arguments.work_request_id != self.request.task_context.get("active_work_request_id"):
            raise PermissionError("Can end only your own active delegation")
        return ToolObservation(summary="Delegation termination requested", abandon_work_id=arguments.work_request_id,
                               control_reason=arguments.reason)


def delegation_tools(request, reader):
    tools = [UpdateWorkTool(request), AbandonWorkTool(request)]
    if request.permissions.request_work and request.work_depth < request.max_work_depth:
        tools.insert(0, RequestWorkTool(request=request, reader=reader))
    return tools
