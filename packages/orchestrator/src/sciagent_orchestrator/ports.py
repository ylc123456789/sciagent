"""Child-module boundary and deterministic fake implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sciagent_contracts import (
    AgentOwner, AgentCheckpoint,
    AgentResult,
    AgentRequest,
)


class ModulePort(Protocol):
    """Uniform boundary implemented by native Agents or other task adapters."""

    def invoke(self, request: AgentRequest) -> AgentResult:
        """Execute until a result or pause; resume may invoke the same Attempt again."""


@runtime_checkable
class CheckpointResultPort(Protocol):
    """Optional read-only recovery of a saved result and its actual input boundary."""

    def read_checkpoint_result(self, *, run_id: str, task_id: str, attempt_number: int) -> AgentCheckpoint | None:
        """Return only saved facts, without model, tool, preparation or finalizer calls."""


@runtime_checkable
class TerminalResultPort(Protocol):
    """Optional read-only lookup for an already persisted task-attempt outcome.

    Implementations own their Session identity and persistence. None means no
    saved terminal outcome; lookup must never invoke, resume or diagnose work.
    """

    def read_terminal_result(self, *, run_id: str, task_id: str, attempt_number: int) -> AgentResult | None:
        """Return saved facts without consuming model budget or executing tools."""


@dataclass(frozen=True, slots=True)
class ModuleBinding:
    """Execution Agent owner and invocation port.

    The workspace is no longer fixed here: the Scheduler resolves each task's
    ``workspace_id`` against ``ResearchRun.workspaces`` and derives a
    per-attempt ``WorkspaceGrant``.
    """

    owner: AgentOwner
    port: ModulePort
    can_delegate: bool = True


class ScriptedModulePort:
    """Deterministic fake ModulePort that returns predefined results."""

    def __init__(self, results: list[AgentResult]) -> None:
        self._results = deque(results)
        self.requests: list[AgentRequest] = []

    def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if not self._results:
            raise RuntimeError("scripted ModulePort has no remaining result")
        return self._results.popleft()
