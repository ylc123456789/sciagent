"""The sciagent shared Agentic Loop runtime (how an Agent runs).

This package owns the execution engine only: the Agentic Loop, Agent state,
context composition, LLM calls, session persistence and the Tool protocol plus
the loop's own finish/ask_user tools. Concrete abilities (filesystem, Git,
process, repository, environment, dataset, hardware) live in
``sciagent_components``. Model-facing wrappers live in
``sciagent_capabilities``; Agents choose tools through their Tool Profile.
"""

from .context import (
    DEFAULT_AGENT_CONTEXT_TOKENS,
    ContextBudgetExceeded,
    ContextComposer,
)
from .llm import (
    LLMClient,
    LLMExhaustedError,
    ModelProfile,
    OpenAICompatibleClient,
    PromptLLMClient,
    ScriptedLLMClient,
    StructuredLLMClient,
)
from .loop import (
    AgentDefinition,
    AgentLoop,
    AllowListPermissionPolicy,
    CompletionCheck,
    ContextBuilder,
    PermissionPolicy,
    OperationReviewer,
)
from .service_request import ServiceRequestClient, ServiceRequestError, ServiceRequestHTTPError
from .models import (
    AgentAction,
    AgentEvent,
    AgentState,
    ApprovalReviewResult,
    ApprovalReviewVerdict,
    CompletionDecision,
    ComposedContext,
    ContextSection,
    FinishCandidate,
    NativeToolCall,
    PermissionDecision,
    ToolCallTurn,
    ToolObservation,
)
from .store import InMemorySessionStore, JsonSessionStore, SessionStore
from .tools import (
    AskUserTool,
    AskUserToolInput,
    FinishInput,
    FinishTool,
    Tool,
    ToolNotFoundError,
    ToolRegistry,
)

__all__ = [
    "DEFAULT_AGENT_CONTEXT_TOKENS",
    "AgentAction",
    "AgentDefinition",
    "AgentEvent",
    "AgentLoop",
    "AgentState",
    "ApprovalReviewResult",
    "ApprovalReviewVerdict",
    "AllowListPermissionPolicy",
    "AskUserTool",
    "AskUserToolInput",
    "CompletionCheck",
    "CompletionDecision",
    "ComposedContext",
    "ContextBudgetExceeded",
    "ContextBuilder",
    "ContextComposer",
    "ContextSection",
    "FinishCandidate",
    "FinishInput",
    "FinishTool",
    "InMemorySessionStore",
    "JsonSessionStore",
    "LLMClient",
    "LLMExhaustedError",
    "ModelProfile",
    "ServiceRequestClient",
    "ServiceRequestError",
    "ServiceRequestHTTPError",
    "NativeToolCall",
    "OpenAICompatibleClient",
    "PermissionDecision",
    "PermissionPolicy",
    "OperationReviewer",
    "PromptLLMClient",
    "ScriptedLLMClient",
    "StructuredLLMClient",
    "SessionStore",
    "Tool",
    "ToolCallTurn",
    "ToolNotFoundError",
    "ToolObservation",
    "ToolRegistry",
]
