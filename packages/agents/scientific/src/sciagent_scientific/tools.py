"""Scientific control tools; the runtime persists stable request identities."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel

from sciagent_contracts import QuestionDraft
from sciagent_components import RegisteredArtifactReader
from sciagent_runtime import AgentState, ToolObservation
from sciagent_runtime.tools import AskUserTool as RuntimeAskUserTool, FinishTool

from sciagent_capabilities.delegation import RequestWorkTool as DelegationRequestWorkTool

from .completion import cited_artifacts
from .models import AskUserInput, RequestWorkInput


def _citation_error(reader, state: AgentState, cited_ids: list[str]) -> ToolObservation | None:
    try:
        cited_artifacts(cited_ids, run_id=state.run_id, reader=reader)
    except (ValueError, OSError) as error:
        return ToolObservation(summary=f"Invalid evidence citation: {error}", ok=False)
    return None


class RequestWorkTool(DelegationRequestWorkTool):
    """Delegate work with a scientific assessment using the shared parent protocol."""
    input_model = RequestWorkInput
    model_guidance = (DelegationRequestWorkTool.model_guidance +
        " assessment contains statement, evidence_artifact_ids, limitations and "
        "unresolved_questions. Cite registered evidence IDs.")

    def execute(self, state, arguments):
        error = _citation_error(self.reader, state, arguments.assessment.evidence_artifact_ids)
        if error is not None:
            return error
        observation = super().execute(state, arguments)
        if not observation.ok:
            return observation
        return observation.model_copy(update={
            "summary": arguments.assessment.statement,
            "memory_updates": {"latest_assessment": arguments.assessment.model_dump(mode="json")},
        })


class AskUserTool:
    """Ask for a research decision with the current scientific assessment."""

    model_guidance = (
        RuntimeAskUserTool.model_guidance
        + " assessment contains statement, evidence_artifact_ids, limitations, and "
        "unresolved_questions; cite registered evidence IDs."
    )
    name = "ask_user"
    input_model = AskUserInput

    def __init__(self, reader: RegisteredArtifactReader | None = None) -> None:
        self.reader = reader

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(AskUserInput, arguments)
        error = _citation_error(self.reader, state, args.assessment.evidence_artifact_ids)
        if error is not None:
            return error
        return ToolObservation(
            summary=args.assessment.statement,
            question=QuestionDraft(text=args.text, requested_fields=args.requested_fields, options=args.options, related_question_id=args.related_question_id),
            memory_updates={"latest_assessment": args.assessment.model_dump(mode="json")},
        )
