"""Initial and appended work use the same candidate graph contract."""

import pytest
from pydantic import ValidationError

from sciagent_contracts import FutureArtifactBinding, TaskProposal, WorkflowPatch, WorkflowProposal


def candidate(kind, tasks):
    if kind == "proposal":
        return WorkflowProposal(work_request_id="work_test", tasks=tasks)
    return WorkflowPatch(work_request_id="work_test", based_on_revision=1, add_tasks=tasks)


def tasks():
    producer = TaskProposal(execution_ref="test@1",
        id="task_producer", work_request_id="work_test", workflow_agent_kind="executor",
        instruction="Produce an analysis", output_names=["analysis"],
    )
    consumer = TaskProposal(execution_ref="test@1",
        id="task_consumer", work_request_id="work_test", workflow_agent_kind="executor",
        instruction="Use the analysis", depends_on=[producer.id],
        input_artifact_bindings=[FutureArtifactBinding(source_task=producer.id, output_selector="analysis")],
    )
    return producer, consumer


@pytest.mark.parametrize("kind", ["proposal", "patch"])
@pytest.mark.parametrize("invalid", ["undeclared_output", "missing_dependency", "unknown_source", "cycle"])
def test_candidate_rejects_invalid_future_output_graph(kind, invalid):
    producer, consumer = tasks()
    if invalid == "undeclared_output":
        consumer.input_artifact_bindings[0].output_selector = "typo"
        message = "undeclared output name"
    elif invalid == "missing_dependency":
        consumer.depends_on = []
        message = "requires direct dependency"
    elif invalid == "unknown_source":
        consumer.input_artifact_bindings[0].source_task = "task_missing"
        message = "unknown task"
    else:
        producer.depends_on = [consumer.id]
        message = "cycle"
    with pytest.raises(ValidationError, match=message):
        candidate(kind, [producer, consumer])


@pytest.mark.parametrize("kind", ["proposal", "patch"])
def test_candidate_accepts_valid_output_binding_in_unordered_graph(kind):
    producer, consumer = tasks()
    graph = candidate(kind, [consumer, producer])
    assert type(graph).model_validate_json(graph.model_dump_json()) == graph
