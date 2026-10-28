"""Offline external-paper import through Controller, Scientific and the CLI-shaped manifest."""

from e2e.native_fixtures import tool_turns
import json
from datetime import UTC, datetime
from pathlib import Path

from resagent2_capabilities.literature import fulltext as fulltext_tool
from resagent2_components.literature import load_literature_manifest
from resagent2_components.literature.fulltext import PdfText
from resagent2_contracts import (
    AgentOwner, ExecutionLimits, ResearchRequest, RunBudget, RunPermissions,
    UserAnswer, WorkflowAgentRegistry,
)
from resagent2_orchestrator import (
    DeterministicWorkInterpreter, ResearchController, ScientificArtifactRegistration,
    WorkflowScheduler,
)
from resagent2_scientific import ScientificAgent


def _manifest(tmp_path: Path) -> Path:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nimported source")
    manifest = tmp_path / "papers.json"
    manifest.write_text(json.dumps({"papers": [{
        "title": "Imported paper",
        "doi": "10.1234/imported",
        "abstract": "User supplied abstract",
        "pdf_path": "paper.pdf",
    }]}), encoding="utf-8")
    return manifest


def _finish(evidence: str):
    return {"tool": "finish", "arguments": {
        "report": "Read the imported paper.",
        "artifacts": [{
            "kind": "scientific_opinion", "path": "opinion.json",
            "media_type": "application/json", "summary": "Conclusion",
            "content": {
                "verdict": "supports", "statement": "Imported source supports the claim.",
                "evidence_artifact_ids": [evidence],
            },
        }],
    }}


def _controller(tmp_path, client):
    scheduler = WorkflowScheduler(
        bindings={}, artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data",
    )
    registration = ScientificArtifactRegistration(scheduler.artifact_registry, scheduler.store)
    client.registration = registration
    client.store = scheduler.store
    agent = ScientificAgent(client, registration_port=registration)
    return ResearchController(
        scientific_port=agent, compiler=None, scheduler=scheduler,
        registry=WorkflowAgentRegistry(),
        interpreter=DeterministicWorkInterpreter(),
    ), registration


def test_imported_pdf_is_read_offline_and_can_complete(tmp_path, monkeypatch):
    calls = []

    def parse(path):
        calls.append(Path(path))
        return PdfText("# PDF text\n\n## Page 1\n\nImported full text.", 1, [], "test-parser")

    monkeypatch.setattr(fulltext_tool, "parse_pdf", parse)
    class Client:
        tool_session_key = "test-native-tools/v1"

        step = 0

        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            self.step += 1
            papers = [r for r in self.store.load("run_imported").artifacts.values()
                      if r.kind == "literature_paper"]
            assert len(papers) == 1
            if self.step == 1:
                return tool_turns([{"tool": "fetch_literature",
                        "arguments": {"source": f"artifact:{papers[0].id}"}}])[0]
            texts = [r for r in self.store.load("run_imported").artifacts.values()
                     if r.kind == "literature_fulltext"]
            if self.step == 2:
                assert len(texts) == 1
                return tool_turns([{"tool": "read_artifact", "arguments": {
                    "artifact_id": next(r.id for r in self.store.load("run_imported").artifacts.values()
                                        if r.kind == "literature_pdf")
                }}])[0]
            if self.step == 3:
                assert len(texts) == 1
                return tool_turns([{"tool": "read_artifact", "arguments": {"artifact_id": texts[0].id}}])[0]
            assert self.step == 4
            return tool_turns([_finish(texts[0].id)])[0]

    controller, registration = _controller(tmp_path, Client())
    run = controller.create_run(
        "run_imported",
        ResearchRequest(
            goal="Read the imported source",
            required_evidence_kinds=["literature_fulltext"],
            budget=RunBudget(max_llm_calls=8, timeout_seconds=60),
            execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1),
            permissions=RunPermissions(execute_commands=False),
        ),
        literature=load_literature_manifest(_manifest(tmp_path)),
    )
    assert run.status.value == "completed", run.terminal_error
    assert len(calls) == 1
    kinds = {ref.kind for ref in run.artifacts.values()}
    assert {"literature_paper", "literature_pdf", "literature_fulltext"} <= kinds
    paper = next(ref for ref in run.artifacts.values() if ref.kind == "literature_paper")
    pdf = next(ref for ref in run.artifacts.values() if ref.kind == "literature_pdf")
    text = next(ref for ref in run.artifacts.values() if ref.kind == "literature_fulltext")
    assert paper.producer == AgentOwner.ORCHESTRATOR
    assert pdf.metadata["source_artifact_id"] == paper.id
    assert text.metadata["source_artifact_id"] == pdf.id
    assert run.final_opinion.evidence_artifact_ids == [text.id]


def test_paused_run_import_refreshes_index_without_resuming(tmp_path):
    class Client:
        tool_session_key = "test-native-tools/v1"

        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            return tool_turns([{"tool": "ask_user", "arguments": {
                "assessment": {"statement": "Need a source"},
                "text": "Import a source before continuing?",
                "requested_fields": ["ready"],
            }}])[0]

    controller, registration = _controller(tmp_path, Client())
    run = controller.create_run(
        "run_paused_import",
        ResearchRequest(
            goal="Wait for imported source",
            budget=RunBudget(max_llm_calls=4, timeout_seconds=60),
            execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1),
            permissions=RunPermissions(execute_commands=False),
        ),
    )
    assert run.status.value == "paused"
    session_before = run.scientific_session
    updated = controller.import_literature(
        run.run_id, load_literature_manifest(_manifest(tmp_path)),
    )
    assert updated.status.value == "paused"
    assert updated.scientific_session == session_before
    assert updated.llm_calls_used == run.llm_calls_used
    assert updated.pending_question.id == run.pending_question.id
    paper = next(ref for ref in updated.artifacts.values() if ref.kind == "literature_paper")
    assert paper.metadata["source_type"] == "import"
    assert paper in controller._authorized_artifacts(updated)
    assert not any(ref.kind in {"research_index", "question", "answer"} for ref in updated.artifacts.values())



def test_paused_import_then_answer_continues_same_scientific_session(tmp_path, monkeypatch):
    parse_calls = []

    def parse(path):
        parse_calls.append(Path(path))
        return PdfText("# Imported full text\n\n## Page 1\n\nEvidence after resume.", 1, [], "test-parser")

    monkeypatch.setattr(fulltext_tool, "parse_pdf", parse)
    manifest = _manifest(tmp_path)
    client_state = {"step": 0}

    class Client:
        tool_session_key = "test-native-tools/v1"

        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            client_state["step"] += 1
            step = client_state["step"]
            run = self.store.load("run_paused_resume")
            papers = [ref for ref in run.artifacts.values() if ref.kind == "literature_paper"]
            texts = [ref for ref in run.artifacts.values() if ref.kind == "literature_fulltext"]
            if step == 1:
                assert not papers
                return tool_turns([{"tool": "ask_user", "arguments": {
                    "assessment": {"statement": "Need a paper"},
                    "text": "Import a paper before continuing?",
                    "requested_fields": ["ready"],
                }}])[0]
            assert len(papers) == 1
            # The resumed prompt contains the refreshed index and imported title.
            assert "Imported paper" in context.text
            if step == 2:
                return tool_turns([{"tool": "fetch_literature",
                        "arguments": {"source": f"artifact:{papers[0].id}"}}])[0]
            if step == 3:
                assert len(texts) == 1
                return tool_turns([{"tool": "read_artifact", "arguments": {"artifact_id": texts[0].id}}])[0]
            assert step == 4
            assert len(texts) == 1
            return tool_turns([_finish(texts[0].id)])[0]

    controller, registration = _controller(tmp_path, Client())
    run = controller.create_run(
        "run_paused_resume",
        ResearchRequest(
            goal="Use a paper supplied while paused",
            required_evidence_kinds=["literature_fulltext"],
            budget=RunBudget(max_llm_calls=8, timeout_seconds=60),
            execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1),
            permissions=RunPermissions(execute_commands=False),
        ),
    )
    assert run.status.value == "paused"
    session_id = run.scientific_session.id

    imported = controller.import_literature(run.run_id, load_literature_manifest(manifest))
    assert imported.status.value == "paused"
    assert imported.scientific_session.id == session_id

    resumed = controller.answer_question(
        run.run_id,
        UserAnswer(
            question_id=imported.pending_question.id,
            values={"ready": "yes"},
            answered_at=datetime.now(UTC),
        ),
    )
    assert resumed.status.value == "completed", resumed.terminal_error
    assert resumed.scientific_session.id == session_id
    assert len(parse_calls) == 1
    fulltext = next(ref for ref in resumed.artifacts.values() if ref.kind == "literature_fulltext")
    assert resumed.final_opinion.evidence_artifact_ids == [fulltext.id]


def test_repeated_import_reuses_snapshot_and_replaced_pdf_creates_new_chain(tmp_path):
    class Client:
        tool_session_key = "test-native-tools/v1"

        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            return tool_turns([{"tool": "ask_user", "arguments": {
                "assessment": {"statement": "Need a source"},
                "text": "Supply papers before continuing?", "requested_fields": ["ready"],
            }}])[0]

    controller, _ = _controller(tmp_path, Client())
    run = controller.create_run(
        "run_import_snapshots",
        ResearchRequest(
            goal="Wait for supplied papers",
            budget=RunBudget(max_llm_calls=5, timeout_seconds=60),
            execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1),
            permissions=RunPermissions(execute_commands=False),
        ),
    )
    manifest = _manifest(tmp_path)
    imported = controller.import_literature(run.run_id, load_literature_manifest(manifest))
    paper = next(ref for ref in imported.artifacts.values() if ref.kind == "literature_paper")
    pdf = next(ref for ref in imported.artifacts.values() if ref.kind == "literature_pdf")
    ids_before = set(imported.artifacts)

    repeated = controller.import_literature(run.run_id, load_literature_manifest(manifest))
    assert set(repeated.artifacts) == ids_before
    assert repeated.artifacts[paper.id] == paper
    assert repeated.artifacts[pdf.id] == pdf

    renamed = (tmp_path / "paper.pdf").rename(tmp_path / "renamed.pdf")
    entries = json.loads(manifest.read_text())
    entries["papers"][0]["pdf_path"] = renamed.name
    manifest.write_text(json.dumps(entries))
    same_source = controller.import_literature(run.run_id, load_literature_manifest(manifest))
    assert set(same_source.artifacts) == ids_before
    assert same_source.artifacts[paper.id] == paper
    assert same_source.artifacts[pdf.id] == pdf

    renamed.write_bytes(b"%PDF-1.7\nreplacement source")
    replaced = controller.import_literature(run.run_id, load_literature_manifest(manifest))
    papers = [ref for ref in replaced.artifacts.values() if ref.kind == "literature_paper"]
    pdfs = [ref for ref in replaced.artifacts.values() if ref.kind == "literature_pdf"]
    assert len(papers) == len(pdfs) == 2
    new_paper = next(ref for ref in papers if ref.id != paper.id)
    new_pdf = next(ref for ref in pdfs if ref.id != pdf.id)
    assert new_paper.metadata["paper"] == paper.metadata["paper"]
    assert new_paper.metadata["import_pdf_sha256"] != paper.metadata["import_pdf_sha256"]
    assert new_pdf.metadata["paper_artifact_id"] == new_paper.id
    assert new_pdf.metadata["source_artifact_id"] == new_paper.id
    assert replaced.artifacts[paper.id] == paper
    assert replaced.artifacts[pdf.id] == pdf
    assert Path(pdf.uri.removeprefix("file://")).read_bytes() == b"%PDF-1.7\nimported source"
    assert replaced.status.value == "paused"
    assert replaced.pending_question.id == run.pending_question.id
    assert replaced.scientific_session == run.scientific_session
