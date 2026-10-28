"""Scientific actions and content-bearing control tools."""

from typing import Literal

from pydantic import Field, field_validator

from sciagent_contracts import (
    ArtifactCandidate, ArtifactRef, ScientificAssessment, ScientificOpinion, WorkRequestDraft,
)
from sciagent_runtime.models import FinishCandidate, RuntimeModel
from sciagent_runtime.tools import AskUserToolInput, FinishInput


class RequestWorkInput(RuntimeModel):
    assessment: ScientificAssessment
    work_request: WorkRequestDraft


class AskUserInput(AskUserToolInput):
    assessment: ScientificAssessment


class ScientificOpinionCandidate(ArtifactCandidate):
    """An opinion object serialized to JSON at the shared finish boundary."""

    kind: Literal["scientific_opinion"]
    content: ScientificOpinion


class OtherScientificArtifactCandidate(ArtifactCandidate):
    """Ordinary text/file candidates cannot bypass the opinion object schema."""

    kind: str = Field(min_length=1, json_schema_extra={"not": {"const": "scientific_opinion"}})

    @field_validator("kind")
    @classmethod
    def exclude_opinion(cls, value: str) -> str:
        value = value.strip()
        if not value or value == "scientific_opinion":
            raise ValueError("scientific_opinion requires structured opinion content")
        return value


class ScientificFinishInput(FinishInput):
    """Scientific uses the shared result with a typed opinion candidate."""

    artifacts: list[ScientificOpinionCandidate | OtherScientificArtifactCandidate | ArtifactRef] = Field(
        default_factory=list,
    )

    def to_candidate(self) -> FinishCandidate:
        artifacts = []
        for item in self.artifacts:
            if isinstance(item, ScientificOpinionCandidate):
                artifacts.append(ArtifactCandidate(**{
                    **item.model_dump(exclude={"content"}),
                    "content": item.content.model_dump_json(),
                }))
            else:
                artifacts.append(item)
        return FinishCandidate(status=self.status, report=self.report, artifacts=artifacts)
