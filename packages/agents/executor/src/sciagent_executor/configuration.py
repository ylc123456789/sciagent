"""Trusted configuration resolution; configuration never grants authorization."""
from __future__ import annotations

from sciagent_contracts import (
    ExecutionConfig, ExecutionDefinition, ExecutionPreset, WorkflowAgentRegistry,
)
from sciagent_runtime import DEFAULT_AGENT_CONTEXT_TOKENS

from .context import (
    CODING_CAPABILITY_DESCRIPTION, CODING_PROMPT,
    EXPERIMENT_CAPABILITY_DESCRIPTION, EXPERIMENT_PROMPT,
)


IMPLEMENTATION_VERSION = "1"
CONTEXT_BUILDER = "executor@1"
WORKSPACE_POLICY = ("workspace@1", "artifacts@1", "execution@1")
GIT_POLICY = ("workspace.git@1", "artifacts.git@1", "execution.git@1")
OPERATION_TOOLS = frozenset({
    "list_files", "read_file", "search_text", "read_artifact", "git_diff",
    "create_file", "replace_text", "delete_path", "run_shell", "request_work",
})
SOURCE_CHANGE_TOOLS = frozenset({"create_file", "replace_text", "delete_path", "git_diff"})
COMMON_TOOLS = ["list_files", "read_file", "search_text", "read_artifact", "run_shell", "request_work"]


def validate_execution_definition(definition: ExecutionDefinition) -> ExecutionDefinition:
    """Reject unsupported components before repository work or model invocation."""
    definition = ExecutionDefinition.model_validate(definition)
    config = definition.config
    if config.implementation_version != IMPLEMENTATION_VERSION:
        raise ValueError(f"Unsupported executor implementation version: {config.implementation_version}")
    if config.context_builder != CONTEXT_BUILDER:
        raise ValueError(f"Unsupported context builder: {config.context_builder}")
    policy = (config.preparation, config.completion, config.finalizer)
    if policy not in (WORKSPACE_POLICY, GIT_POLICY):
        raise ValueError(f"Unsupported or inconsistent preparation/completion/finalizer policies: {policy}")
    unknown = set(config.tools) - OPERATION_TOOLS
    if unknown:
        raise ValueError(f"Unknown executor tools: {sorted(unknown)}")
    if len(config.tools) != len(set(config.tools)):
        raise ValueError("Executor tool references must be unique")
    if SOURCE_CHANGE_TOOLS.intersection(config.tools) and policy != GIT_POLICY:
        raise ValueError("Source-editing and Git tools require the Git preparation, completion and finalizer policies")
    return definition


def build_coding_definition(*, model_profile: str = "default",
                            max_context_tokens: int = DEFAULT_AGENT_CONTEXT_TOKENS) -> ExecutionDefinition:
    return validate_execution_definition(ExecutionDefinition(ref="coding@1", config=ExecutionConfig(
        system_prompt=CODING_PROMPT,
        tools=[*COMMON_TOOLS, "git_diff", "create_file", "replace_text", "delete_path"],
        preparation=GIT_POLICY[0], completion=GIT_POLICY[1], finalizer=GIT_POLICY[2],
        context_builder=CONTEXT_BUILDER, implementation_version=IMPLEMENTATION_VERSION,
        model_profile=model_profile, max_context_tokens=max_context_tokens,
    )))


def build_experiment_definition(*, model_profile: str = "default",
                                max_context_tokens: int = DEFAULT_AGENT_CONTEXT_TOKENS) -> ExecutionDefinition:
    return validate_execution_definition(ExecutionDefinition(ref="experiment@1", config=ExecutionConfig(
        system_prompt=EXPERIMENT_PROMPT, tools=list(COMMON_TOOLS),
        preparation=WORKSPACE_POLICY[0], completion=WORKSPACE_POLICY[1], finalizer=WORKSPACE_POLICY[2],
        context_builder=CONTEXT_BUILDER, implementation_version=IMPLEMENTATION_VERSION,
        model_profile=model_profile, max_context_tokens=max_context_tokens,
    )))


def default_execution_registry(*, model_profile: str = "default",
                               coding_max_context_tokens: int = DEFAULT_AGENT_CONTEXT_TOKENS,
                               experiment_max_context_tokens: int = DEFAULT_AGENT_CONTEXT_TOKENS) -> WorkflowAgentRegistry:
    coding = build_coding_definition(model_profile=model_profile, max_context_tokens=coding_max_context_tokens)
    experiment = build_experiment_definition(model_profile=model_profile, max_context_tokens=experiment_max_context_tokens)
    return WorkflowAgentRegistry(execution_definitions={coding.ref: coding, experiment.ref: experiment}, presets=[
        ExecutionPreset(preset_ref="coding@1", name="Coding",
            description=CODING_CAPABILITY_DESCRIPTION,
            execution_ref=coding.ref),
        ExecutionPreset(preset_ref="experiment@1", name="Experiment",
            description=EXPERIMENT_CAPABILITY_DESCRIPTION,
            execution_ref=experiment.ref),
    ])
