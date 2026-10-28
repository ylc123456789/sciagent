"""Persisted Run and WorkRequest setup for isolated Scheduler tests.

These graphs contain only the work under test, not Controller.create_run's
complete root Scientific/delegation graph.
"""
from datetime import UTC, datetime
from sciagent_contracts import WorkRequest, WorkRequestDraft, AgentOwner, WorkflowAgentKind, task_session_id
from sciagent_executor import default_execution_registry
from sciagent_orchestrator import ModuleBinding
from sciagent_orchestrator import ResearchRun


def accept_proposal(engine, run_id, proposal):
    if not proposal.tasks:
        return engine.accept_proposal(run_id, proposal)
    run = engine.store.load(run_id)
    registry = default_execution_registry()
    run.execution_definitions.update(registry.execution_definitions)
    run.execution_presets = list(registry.presets)
    if not any(work.id == proposal.work_request_id for work in run.work_requests):
        now = datetime.now(UTC)
        keys = {task.id: f"item_{index}" for index, task in enumerate(proposal.tasks)}
        drafts = [{"key": keys[task.id], "instruction": task.instruction,
                   "selection": {"kind": "preset", "preset_ref": task.execution_ref or "coding@1", "reason": "Explicit scheduler fixture configuration"},
                   "depends_on": [keys[item] for item in task.depends_on]}
                  for task in proposal.tasks]
        run.work_requests.append(WorkRequest(
            id=proposal.work_request_id, run_id=run_id, status="executing", workflow_revision=1,
            request=WorkRequestDraft(objective=run.request.goal, tasks=drafts),
            created_at=now, updated_at=now,
        ))
    engine.store.save(run)
    return engine.accept_proposal(run_id, proposal)


def create_scheduler_run(engine, run_id, request, proposal, *, resolve_workspaces=False):
    """Save a bare running Run, then attach the explicit work under test."""
    now = datetime.now(UTC)
    fields = {"workspaces": engine._resolve_workspaces(run_id)} if resolve_workspaces else {}
    engine.store.save(ResearchRun(
        run_id=run_id, request=request, status="running",
        created_at=now, updated_at=now, **fields,
    ))
    return accept_proposal(engine, run_id, proposal)


class _PresetPorts:
    """Only a test double: preserve separately scripted preset behaviors."""
    def __init__(self, ports):
        self.ports = ports
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        ref = request.execution_definition.ref
        key = ref.split("@", 1)[0]
        port = self.ports.get(key)
        if port is None and len(self.ports) == 1:
            port = next(iter(self.ports.values()))
        if port is None:
            raise ValueError(f"No explicit scripted preset port for {ref}")
        return port.invoke(request)


    def __getattr__(self, name):
        if name not in {"terminate_session", "cancel_question", "read_checkpoint_result", "read_terminal_result"}:
            raise AttributeError(name)
        if not any(callable(getattr(port, name, None)) for port in self.ports.values()):
            raise AttributeError(name)

        def forward(*args, **kwargs):
            for port in self.ports.values():
                callback = getattr(port, name, None)
                if callback is None:
                    continue
                for request in getattr(port, "requests", []):
                    if name in {"terminate_session", "cancel_question"}:
                        matches = task_session_id(request.run_id, request.task_id, request.attempt_number) == args[0]
                    else:
                        matches = (request.run_id, request.task_id, request.attempt_number) == (
                            kwargs["run_id"], kwargs["task_id"], kwargs["attempt_number"])
                    if matches:
                        return callback(*args, **kwargs)
            return None
        return forward


def executor_bindings(bindings):
    """Convert preset-keyed scripted bindings to the single Executor boundary."""
    ports = {}
    result = {}
    for key, binding in bindings.items():
        if str(key) == "scientific":
            result[WorkflowAgentKind.SCIENTIFIC] = binding
        else:
            ports[str(key)] = binding.port
    if ports:
        if len(ports) == 1:
            port = next(iter(ports.values()))
        else:
            port = _PresetPorts(ports)
        result[WorkflowAgentKind.EXECUTOR] = ModuleBinding(AgentOwner.EXECUTOR, port,
            can_delegate=all(binding.can_delegate for key,binding in bindings.items() if str(key) != "scientific"))
    return result
