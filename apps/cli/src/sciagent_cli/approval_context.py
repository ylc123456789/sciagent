"""Read original user requirements for an operation reviewer at composition."""

from __future__ import annotations

from sciagent_components.materials import ordered_business_dialogue
from sciagent_contracts import WorkflowAgentKind
from sciagent_orchestrator.store import RunStore


def approval_context(run_store: RunStore, run_id: str) -> dict:
    """Project root user records without promoting delegated text to authority.

    Loading errors deliberately propagate: unavailable original requirements are
    a reason to request manual review, not permission to omit those requirements.
    This reads only the Run snapshot, never artifacts or private Agent sessions.
    """
    run = run_store.load(run_id)
    roots = [task for task in run.workflow.tasks
             if run.work(task.work_request_id).owner_task_id is None] if run.workflow else []
    if len(roots) != 1 or roots[0].workflow_agent_kind != WorkflowAgentKind.SCIENTIFIC:
        raise ValueError("review context requires one root Scientific task")
    root = roots[0]
    answers = [answer for answer in run.answers if answer.task_id == root.id and answer.action is None]
    if any(answer.run_id != run.run_id for answer in answers):
        raise ValueError("root business answer belongs to another Run")
    # Validate the complete root dialogue before selecting real user records.
    # input_ids() repairs/deduplicates an order for scheduling; the reviewer must
    # not let that hide missing or conflicting authorization evidence.
    dialogue = ordered_business_dialogue(upstream_inputs=run.task_inputs.get(root.id, []),
        answers=answers, input_ids=run.task_input_order.get(root.id, []))
    dialogue = [item for item in dialogue
                if item["record"]["source"] == "user"
                and item["record"].get("source_session_id") is None
                and item["record"].get("original_event_id") is None
                and item["record"].get("original_question_id") is None]
    return {
        "run_id": run.run_id,
        "initial_request": {
            "source": "user",
            "created_at": run.created_at.isoformat(),
            "goal": run.request.goal,
            "hypothesis": run.request.hypothesis,
            "context": run.request.context,
            "constraints": list(run.request.constraints),
            "required_artifacts": list(run.request.required_artifacts),
            "required_evidence_kinds": list(run.request.required_evidence_kinds),
        },
        "business_dialogue": dialogue,
        # Historical decisions describe what the user actually reviewed. They
        # never grant permission to this or any subsequent operation.
        "prior_operation_decisions": sorted(
            [
                {
                    **answer.model_dump(mode="json", exclude_none=True),
                    "decision": answer.values.get("approve"),
                }
                for answer in run.answers
                if answer.source == "user"
                and answer.source_session_id is None
                and answer.original_question_id is None
                and answer.action is not None
            ],
            key=lambda item: (item["answered_at"], item["question_id"]),
        ),
    }
