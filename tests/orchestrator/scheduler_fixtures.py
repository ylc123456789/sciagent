"""Explicit persisted WorkRequest setup for isolated Scheduler unit tests.

These inputs intentionally contain only the task graph under test. They do not
represent Controller.create_run's complete root Scientific/delegation graph.
"""
from resagent2_contracts import WorkRequest, WorkRequestDraft
from e2e.scheduler_fixtures import accept_proposal, create_scheduler_run


def add_root_owner(run, session_id):
    """Complete a hand-built delegated graph with its actual waiting root owner.

    Keep the subject tasks first so Scheduler assertions use their original
    positions; Controller tests should select the named task explicitly.
    """
    from resagent2_contracts import Attempt, SessionRef, Workflow, WorkflowTask
    if run.workflow is None:
        run.workflow = Workflow(run_id=run.run_id, revision=1, created_from="work_root", tasks=[])
    root = next((task for task in run.workflow.tasks if task.id == "task_root"), None)
    if root is None:
        session = SessionRef(id=session_id, module="scientific", status="paused",
                             state_uri=f"memory://{session_id}", created_at=run.created_at, updated_at=run.updated_at)
        run.scientific_session = session
        run.workflow.tasks.append(WorkflowTask(execution_ref=None,
            id="task_root", work_request_id="work_root", workflow_agent_kind="scientific",
            instruction=run.request.goal, status="waiting_for_work", attempts=[Attempt(
                number=1, status="waiting_for_work", started_at=run.created_at, session=session,
            )],
        ))
    if not any(work.id == "work_root" for work in run.work_requests):
        run.work_requests.append(WorkRequest(
            id="work_root", run_id=run.run_id, request=WorkRequestDraft(
                objective=run.request.goal, tasks=[{"selection":{"kind":"preset","preset_ref":"coding@1","reason":"Explicit test preset"}, "key": "root", "instruction": run.request.goal}]),
            status="executing", workflow_revision=1,
            created_at=run.created_at, updated_at=run.updated_at,
        ))
    return run


def execution_tasks(run):
    return [task for task in run.workflow.tasks if task.id != "task_root"]
