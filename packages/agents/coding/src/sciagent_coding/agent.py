"""Direct factory entry for the trusted coding execution preset."""
from sciagent_contracts import AgentRequest
from sciagent_executor import NativeExecutorAgent, build_coding_definition
from sciagent_runtime import DEFAULT_AGENT_CONTEXT_TOKENS


class NativeCodingAgent(NativeExecutorAgent):
    description = 'Understand, implement and repair code; continue with fitting, training and research evaluation when requested, with Git change and execution evidence.'

    def __init__(self, llm_client, *, store=None, resource_layout=None,
                 max_context_tokens=DEFAULT_AGENT_CONTEXT_TOKENS, operation_reviewer=None):
        self.preset_definition = build_coding_definition(max_context_tokens=max_context_tokens)
        def resolve(profile):
            if profile != self.preset_definition.config.model_profile:
                raise ValueError(f"Unknown model profile: {profile}")
            return llm_client
        super().__init__(resolve, store=store, resource_layout=resource_layout,
                         operation_reviewer=operation_reviewer)

    @property
    def max_context_tokens(self):
        return self.preset_definition.config.max_context_tokens

    def invoke(self, request):
        try:
            request = AgentRequest.model_validate(request)
        except ValueError as error:
            return self._failure(str(error))
        if request.execution_definition is None:
            request = request.model_copy(update={"execution_definition": self.preset_definition})
        elif request.execution_definition != self.preset_definition:
            return self._failure("The coding preset entry received a different frozen execution definition", blocked=True)
        return super().invoke(request)
