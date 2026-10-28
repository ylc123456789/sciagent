"""Production composition root for the standalone CLI.

This module only creates and connects existing ResAgent2 components. Research
control, task scheduling, Agent behavior, persistence semantics, and evidence
validation remain in their owning packages.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from resagent2_components import (
    ArxivLiteratureBackend,
    MultiSourceLiteratureBackend,
    OpenAlexLiteratureBackend,
    DeepSeekWebSearchBackend,
    WebPageFetcher,
)
from resagent2_components import (
    DatasetCatalog,
    ResourceLayout,
)
from resagent2_components.approval_review import LLMPermissionReviewer
from resagent2_components.literature.fulltext import DEFAULT_PDF_PARSE_TIMEOUT_SECONDS, parse_pdf
from resagent2_contracts import (
    AgentOwner,
    WorkflowAgentKind,
    WorkspaceSpec,
)
from resagent2_executor import (
    NativeExecutorAgent, default_execution_registry, validate_execution_definition,
)
from resagent2_orchestrator import (
    JsonRunStore,
    FixedWorkflowCompiler,
    DeterministicWorkInterpreter,
    ModuleBinding,
    ResearchController,
    ScientificArtifactRegistration,
    WorkflowScheduler,
)
from resagent2_runtime import (
    DEFAULT_AGENT_CONTEXT_TOKENS,
    JsonSessionStore,
    ModelProfile,
    ServiceRequestClient,
    OpenAICompatibleClient,
)
from resagent2_runtime.controls import RunControlInbox
from resagent2_scientific import ScientificAgent

from .approval_context import approval_context


@dataclass(frozen=True, slots=True)
class CliApplication:
    """Objects the CLI needs after composition."""

    controller: ResearchController
    run_store: JsonRunStore


def _trace_dir() -> Path | None:
    value = os.environ.get("RESAGENT2_LLM_TRACE_DIR")
    return Path(value).expanduser().resolve() if value else None


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _web_search_provider() -> str | None:
    """Validate explicit provider credentials before creating clients or files."""

    explicit_provider = os.environ.get("RESAGENT2_WEB_SEARCH_PROVIDER")
    provider = "deepseek" if explicit_provider is None else explicit_provider
    if provider not in {"deepseek", "off"}:
        raise ValueError("RESAGENT2_WEB_SEARCH_PROVIDER must be deepseek or off")
    if provider == "off":
        return None
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        if explicit_provider is not None:
            raise ValueError("RESAGENT2_WEB_SEARCH_PROVIDER=deepseek requires DEEPSEEK_API_KEY")
        return None
    if "RESAGENT2_WEB_SEARCH_MODEL" in os.environ:
        raise ValueError(
            "RESAGENT2_WEB_SEARCH_MODEL is unsupported by the Search API; remove it"
        )
    return provider


def _model_profile() -> ModelProfile:
    """Declare model capacity separately from module input allowances."""

    return ModelProfile(
        context_window=_positive_int_env("RESAGENT2_CONTEXT_WINDOW", 1_000_000),
        reserved_output_tokens=_positive_int_env(
            "RESAGENT2_RESERVED_OUTPUT_TOKENS", 256_000
        ),
        safety_margin_tokens=_positive_int_env(
            "RESAGENT2_CONTEXT_SAFETY_MARGIN_TOKENS", 1024
        ),
    )


def _component_context_limit(component: str) -> int:
    return _positive_int_env(
        f"RESAGENT2_{component.upper()}_CONTEXT_TOKENS",
        DEFAULT_AGENT_CONTEXT_TOKENS,
    )


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        model=os.environ.get("RESAGENT2_MODEL", "deepseek-v4-flash"),
        api_base=os.environ.get("RESAGENT2_API_BASE", "https://api.deepseek.com/v1"),
        api_key_env=os.environ.get("RESAGENT2_API_KEY_ENV", "DEEPSEEK_API_KEY"),
        model_profile=_model_profile(),
        timeout_seconds=_positive_int_env("RESAGENT2_LLM_TIMEOUT_SECONDS", 600),
        trace_dir=_trace_dir(),
        trace_level=os.environ.get("RESAGENT2_LLM_TRACE_LEVEL", "off"),
    )


def _execution_client(profile: str):
    """Resolve only host-supported model profiles; config cannot supply credentials."""
    if profile != "default":
        raise ValueError(f"unsupported execution model profile: {profile}")
    return _client()


def build_application(
    *,
    data_root: str | Path,
    workspaces: dict[str, WorkspaceSpec] | None = None,
) -> CliApplication:
    """Create the existing system behind the CLI boundary."""

    pdf_parse_timeout_seconds = _positive_int_env(
        "RESAGENT2_PDF_PARSE_TIMEOUT_SECONDS", DEFAULT_PDF_PARSE_TIMEOUT_SECONDS,
    )
    web_timeout_seconds = _positive_int_env(
        "RESAGENT2_WEB_TIMEOUT_SECONDS", 30,
    )
    web_max_response_bytes = _positive_int_env(
        "RESAGENT2_WEB_MAX_RESPONSE_BYTES", 4 * 1024 * 1024,
    )
    web_search_timeout_seconds = _positive_int_env(
        "RESAGENT2_WEB_SEARCH_TIMEOUT_SECONDS", 60,
    )
    web_search_provider = _web_search_provider()
    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_store = JsonRunStore(root / "state")
    resource_layout = ResourceLayout.from_env(data_root=root)
    dataset_catalog = DatasetCatalog(resource_layout.dataset_root)
    executor_store = JsonSessionStore(root / "sessions" / "executor")
    scientific_store = JsonSessionStore(root / "sessions" / "scientific")
    coding_context_tokens = _component_context_limit("coding")
    experiment_context_tokens = _component_context_limit("experiment")
    scientific_context_tokens = _component_context_limit("scientific")
    operation_reviewer = LLMPermissionReviewer(
        _client(),
        max_context_tokens=_component_context_limit("permission_reviewer"),
        context_provider=partial(approval_context, run_store),
    )

    registry = default_execution_registry(
        coding_max_context_tokens=coding_context_tokens,
        experiment_max_context_tokens=experiment_context_tokens,
    )
    for definition in registry.execution_definitions.values():
        validate_execution_definition(definition)
    bindings = {
        WorkflowAgentKind.EXECUTOR: ModuleBinding(
            owner=AgentOwner.EXECUTOR,
            port=NativeExecutorAgent(
                _execution_client,
                store=executor_store,
                resource_layout=resource_layout,
                operation_reviewer=operation_reviewer,
            ),
        ),
    }
    scheduler = WorkflowScheduler(
        bindings=bindings,
        store=run_store,
        artifact_root=root / "artifacts",
        data_root=root,
        workspaces=workspaces,
    )
    registration = ScientificArtifactRegistration(
        scheduler.artifact_registry,
        run_store,
    )
    web_search_backend = None
    if web_search_provider == "deepseek":
        web_search_backend = DeepSeekWebSearchBackend(
            ServiceRequestClient(
                endpoint="https://api.deepseek.com/v1/tools/search",
                api_key_env="DEEPSEEK_API_KEY",
                timeout_seconds=web_search_timeout_seconds,
                max_response_bytes=web_max_response_bytes,
                trace_dir=_trace_dir(),
                trace_level=os.environ.get("RESAGENT2_LLM_TRACE_LEVEL", "off"),
            ),
            count=10,
        )
    scientific = ScientificAgent(
        _client(),
        literature_backend=MultiSourceLiteratureBackend(
            ArxivLiteratureBackend(),
            OpenAlexLiteratureBackend(api_key=os.environ.get("OPENALEX_API_KEY")),
        ),
        literature_parser=partial(parse_pdf, timeout_seconds=pdf_parse_timeout_seconds),
        web_search_backend=web_search_backend,
        web_page_fetcher=WebPageFetcher(
            timeout_seconds=web_timeout_seconds,
            max_response_bytes=web_max_response_bytes,
        ),
        registration_port=registration,
        store=scientific_store,
        max_context_tokens=scientific_context_tokens,
        resource_layout=resource_layout,
    )
    controller = ResearchController(
        scientific_port=scientific,
        compiler=FixedWorkflowCompiler(validate_execution_definition),
        interpreter=DeterministicWorkInterpreter(),
        scheduler=scheduler,
        registry=registry,
        dataset_ref_source=dataset_catalog,
        control_inbox=RunControlInbox(root / "controls"),
    )
    trace_level = os.environ.get("RESAGENT2_LLM_TRACE_LEVEL", "off")
    print(f"LLM trace: {trace_level}; directory: {_trace_dir() or 'unset'}", file=sys.stderr)
    return CliApplication(controller=controller, run_store=run_store)
