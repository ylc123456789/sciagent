"""Direct factory entry for the trusted experiment execution preset."""
from resagent2_contracts import AgentRequest
from resagent2_executor import NativeExecutorAgent, build_experiment_definition
from resagent2_runtime import DEFAULT_AGENT_CONTEXT_TOKENS


class NativeExperimentAgent(NativeExecutorAgent):
    description = 'Analyze existing results and use available code for experiments, fitting and evaluation, with actual command and result evidence; no source-editing tools or Git delivery.'

    def __init__(self, llm_client, *, store=None, resource_layout=None,
                 max_context_tokens=DEFAULT_AGENT_CONTEXT_TOKENS, operation_reviewer=None):
        self.preset_definition = build_experiment_definition(max_context_tokens=max_context_tokens)
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
            return self._failure("The experiment preset entry received a different frozen execution definition", blocked=True)
        return super().invoke(request)
