from resagent2_contracts import WorkflowAgentKind, RunStatus

from e2e.mock_e2e import run_mock_e2e


def test_mock_e2e_runs_the_golden_loop_to_completion(tmp_path) -> None:
    run = run_mock_e2e(workdir=tmp_path)

    assert run.status == RunStatus.COMPLETED
    assert [task.workflow_agent_kind for task in run.workflow.tasks] == [
        WorkflowAgentKind.SCIENTIFIC,
        WorkflowAgentKind.EXECUTOR,
        WorkflowAgentKind.EXECUTOR,
    ]
    assert run.final_opinion is not None
    assert run.artifacts[run.final_report_artifact_id].kind == "final_report"
    assert all(task.attempts for task in run.workflow.tasks)
