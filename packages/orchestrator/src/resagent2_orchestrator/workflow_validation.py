"""Pure round-boundary checks shared by compilation and graph acceptance.

Schema validators own shape, identity and DAG consistency. This predicate owns
the execution policy: a request must add work now.
"""

from resagent2_contracts import WorkflowPatch, WorkflowProposal


def validate_workflow_candidate(candidate: WorkflowProposal | WorkflowPatch) -> None:
    """Reject an empty work request; graph consistency belongs to the schema."""
    tasks = candidate.tasks if isinstance(candidate, WorkflowProposal) else candidate.add_tasks
    if not tasks:
        raise ValueError("compiler produced an empty task graph")
