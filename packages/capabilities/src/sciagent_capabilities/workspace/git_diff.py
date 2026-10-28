"""Git diff tool."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from sciagent_runtime import AgentState, ToolObservation
from sciagent_runtime.models import RuntimeModel
from sciagent_components.git import GitBaseline, GitWorkspace

class GitDiffInput(RuntimeModel):
    """Bound the Git diff returned to the Agent context."""

    max_chars: int = Field(default=8_000, ge=1, le=20_000)


class GitDiffTool:
    """Expose the Attempt-relative Git patch (the increment since baseline).

    The Coding Agent always supplies its Attempt baseline, so the Agent sees the
    same increment the deterministic finalizer attributes (ADR-0011 §4).
    """

    name = "git_diff"
    input_model = GitDiffInput
    model_guidance = (
        "Read the current Attempt-relative patch. This reports actual code changes; "
        "use recorded command outcomes to assess what has been checked. "
        "This is a bounded UTF-8 preview; undecodable bytes are replaced and marked. "
        "The registered patch preserves the original bytes."
    )

    def __init__(
        self, repository: GitWorkspace, *, baseline: GitBaseline,
    ) -> None:
        self.repository = repository
        self.baseline = baseline

    def execute(self, state: AgentState, arguments: BaseModel) -> ToolObservation:
        args = cast(GitDiffInput, arguments)
        patch = self.repository.diff_bytes_since(self.baseline)
        try:
            diff = patch.decode("utf-8")
            decoding_errors_replaced = False
        except UnicodeDecodeError:
            diff = patch.decode("utf-8", errors="replace")
            decoding_errors_replaced = True
        truncated = len(diff) > args.max_chars
        value = {
            "diff": diff[: args.max_chars],
            "truncated": truncated,
            "encoding": "utf-8",
            "decoding_errors_replaced": decoding_errors_replaced,
        }
        return ToolObservation(
            summary="Read current Git diff",
            value=value,
        )
