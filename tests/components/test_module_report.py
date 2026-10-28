"""Original report bytes remain intact, including long physical lines."""
from resagent2_contracts import AgentResult,ArtifactCandidate
from resagent2_components import RegisteredArtifactReader
from resagent2_orchestrator import ArtifactRegistry


def test_raw_agent_report_preserves_leading_trailing_whitespace():
    text="  Exact report\n\n"+" 长行"*90000+"\n\tEND  "
    result=AgentResult(status="completed",report=text)
    assert result.report==text
    assert AgentResult.model_validate_json(result.model_dump_json()).report==text


def test_original_task_report_is_frozen_without_wrapping_and_continuously_readable(tmp_path):
    text="  first\n\n"+" x\t"*90000+"TAIL_SENTINEL\n\tfinal  "
    ref=ArtifactRegistry(tmp_path).register_system_artifact(ArtifactCandidate(kind="module_report",
        path="report.md",media_type="text/markdown",summary="Original coding report",content=text),
        run_id="run_report",task_id="task_code",attempt_number=1,source_type="task_report")
    reader=RegisteredArtifactReader([ref],run_id="run_report")
    chunks=[];start=0
    while True:
        window=reader.read_text(ref.id,start_char=start,max_chars=10000)
        chunks.append(window["content"])
        if window["next_start_char"] is None: break
        start=window["next_start_char"]
    assert "".join(chunks)==text
