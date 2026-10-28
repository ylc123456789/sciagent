"""Scientific forwards a parser chosen by its caller to the full-text tool."""

from e2e.native_fixtures import tool_turns

from sciagent_components import RegisteredArtifactReader
from sciagent_components.literature import LiteraturePaper
from sciagent_components.literature.fulltext import PdfText
from sciagent_contracts import (
    AgentOwner, AgentPermissions, AgentRequest, ArtifactCandidate, ScientificOpinion,
    TaskBudget, task_session_id,
)
from sciagent_orchestrator import ArtifactRegistry
from sciagent_runtime import ScriptedLLMClient
from sciagent_scientific import ScientificAgent


def test_injected_parser_reads_frozen_pdf_and_registers_its_text(tmp_path):
    run_id = "run_parser"
    session_id = task_session_id(run_id, "task_root", 1)
    registry = ArtifactRegistry(tmp_path / "artifacts")

    class Registration:
        def __init__(self):
            self.refs = {}

        def register_scientific(self, candidate, **kwargs):
            position = kwargs.pop("output_position", None)
            if position is not None and kwargs.get("operation_key") is not None:
                kwargs["operation_key"] += f":output:{position}"
            ref = registry.register_scientific(candidate, **kwargs)
            self.refs[ref.id] = ref
            return ref

        def list_artifacts(self, *, run_id, session_id):
            return [ref for ref in self.refs.values() if ref.run_id == run_id and ref.session_id == session_id]

        def resolve(self, artifact_id, *, run_id, session_id):
            ref = self.refs.get(artifact_id)
            return ref if ref and ref.run_id == run_id and ref.session_id == session_id else None

    register = Registration()
    paper = LiteraturePaper(
        paper_id="test:paper", title="A paper", source_url="https://example.test/paper",
    )
    paper_ref = register.register_scientific(ArtifactCandidate(
        kind="literature_paper", path="paper.json", summary=paper.title,
        content=paper.model_dump_json(),
        media_type="application/json", metadata={"paper": paper.model_dump(mode="json")},
    ), run_id=run_id, session_id=session_id)
    original = b"%PDF-1.7\nOriginal frozen bytes"
    pdf_ref = register.register_scientific(ArtifactCandidate(
        kind="literature_pdf", path="paper.pdf", media_type="application/pdf",
        summary="Original PDF",
        metadata={"paper_artifact_id": paper_ref.id, "source_artifact_id": paper_ref.id},
    ), run_id=run_id, session_id=session_id, content_bytes=original)
    parsed_paths = []

    def parser(path):
        parsed_paths.append(path)
        assert path.read_bytes() == original
        return PdfText("## Page 1\nInjected parser text.", 1, [], "injected-test")

    client = ScriptedLLMClient(tool_turns([
        {"tool": "fetch_literature", "arguments": {"source": f"artifact:{paper_ref.id}"}},
        {"tool": "finish", "arguments": {
            "report": "Parser injection checked",
            "artifacts": [{
                "kind": "scientific_opinion", "path": "scientific_opinion.json",
                "media_type": "application/json", "summary": "Scientific judgment",
                "content": ScientificOpinion(
                    verdict="inconclusive", statement="Full text is available.",
                ).model_dump(mode="json"),
            }],
        }},
    ]))
    result = ScientificAgent(
        client, registration_port=register, literature_parser=parser,
    ).invoke(AgentRequest(work_request_id="work_root", task_id="task_root", attempt_number=1,
        run_id=run_id, agent=AgentOwner.SCIENTIFIC, instruction="Read the paper",
        input_artifacts=[paper_ref, pdf_ref],
        budget=TaskBudget(max_llm_calls=2, timeout_seconds=30),
        permissions=AgentPermissions(),
    ))

    assert result.status == "completed", result.report
    assert len(parsed_paths) == 1
    text = next(ref for ref in result.artifacts if ref.kind == "literature_fulltext")
    assert text.metadata["parser_version"] == "injected-test"
    assert text.metadata["source_artifact_id"] == pdf_ref.id
    reader = RegisteredArtifactReader([text], run_id=run_id)
    assert "Injected parser text." in reader.read_text(text.id)["content"]
