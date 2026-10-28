"""Tool dispatch registry and the shared finish/ask_user tools."""

from __future__ import annotations

from typing import Protocol, cast

from pydantic import BaseModel, Field, JsonValue

from resagent2_contracts import AnswerFieldName, QuestionDraft

from .models import AgentState, FinishCandidate, NonEmptyStr, RuntimeModel, ToolObservation


class Tool(Protocol):
    """Protocol implemented by every runtime Tool."""

    name: str
    input_model: type[BaseModel]

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        """Execute validated arguments without mutating state directly."""


class ToolNotFoundError(LookupError):
    """Raised when an action names a Tool outside its Agent definition."""


class ToolRegistry:
    """Validates Tool names and argument schemas before dispatch."""

    def __init__(self, tools: tuple[Tool, ...]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def contains(self, name: str) -> bool:
        """Return whether a Tool name belongs to this registry."""

        return name in self._tools

    def dispatch(
        self,
        name: str,
        arguments: dict[str, JsonValue],
        state: AgentState,
        *,
        prepared: dict[str, JsonValue] | None = None,
    ) -> ToolObservation:
        """Validate raw arguments and execute the selected Tool."""

        from .cancellation import check_cancelled
        parsed = self.validate(name, arguments)
        check_cancelled()
        if prepared is not None:
            return self._tools[name].execute_prepared(state, parsed, prepared)
        return self._tools[name].execute(state, parsed)

    def validate(self, name: str, arguments: dict[str, JsonValue]) -> BaseModel:
        """Preflight arguments without executing a tool or changing state."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool.input_model.model_validate(arguments)


class FinishInput(FinishCandidate):
    """Input schema and normalization boundary for the shared finish tool."""

    def to_candidate(self) -> FinishCandidate:
        """Normalize model-facing input before the common completion checks."""
        return FinishCandidate(**self.model_dump())


class FinishTool:
    """Submit a report and artifact outputs as a finish candidate for validation."""

    name = "finish"
    input_model = FinishInput
    model_guidance = (
        "Use status and report to state the assigned work's outcome; the Agent's role "
        "determines its completion requirements. For each new artifact, supply kind, "
        "path, media_type and summary, with optional output_name. Supply content "
        "in the form defined by the input schema. "
        "Without content, path must identify exactly one existing file relative to "
        "the supplied workspace or output roots. With content, path is the relative "
        "filename for the stored content. Use files for substantive source/results and "
        "content for short structured outputs. Preserve explicit output_name values "
        "verbatim; a filename alone is not a logical output_name. Never invent "
        "ArtifactIds or fabricate system-generated execution or verification records."
    )

    def __init__(self, input_model: type[FinishInput] = FinishInput) -> None:
        self.input_model = input_model

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(FinishInput, arguments)
        return ToolObservation(
            summary="Produced a finish candidate",
            finish_candidate=args.to_candidate(),
        )


class AskUserToolInput(RuntimeModel):
    """Input schema for AskUserTool."""

    related_question_id: str | None = Field(default=None, description="When escalating your direct child question, reference its exact question ID")
    text: NonEmptyStr = Field(
        description="Self-contained question shown to your direct superior (root asks the user). Include the "
        "background needed to answer in this text."
    )
    requested_fields: list[AnswerFieldName] = Field(min_length=1)
    options: dict[AnswerFieldName, list[NonEmptyStr]] | None = None


class AskUserTool:
    """Request direct-superior input with a self-contained question and named answer fields."""

    name = "ask_user"
    input_model = AskUserToolInput
    model_guidance = (
        "Put the background, question and explanations in text so the user can answer "
        "without seeing internal context. requested_fields contains short machine keys "
        "such as mode or file_choice: 1-64 ASCII letters, digits or underscores, "
        "starting with a letter. Do not put sentences or option explanations in keys. "
        "Optional options maps those same keys to the offered answer values; explain "
        "the choices in text."
    )

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(AskUserToolInput, arguments)
        return ToolObservation(
            summary="User input is required",
            question=QuestionDraft(**args.model_dump()),
        )
