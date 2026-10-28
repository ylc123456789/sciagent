"""Scientific search → fetch → explicit read through real stores and receipts."""

import hashlib
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from e2e.native_fixtures import tool_turns
from sciagent_components import ArtifactReadError, LiteraturePaper, LiteratureSearchResult, RegisteredArtifactReader
from sciagent_components.literature.fulltext import PdfText
from sciagent_contracts import ResearchRequest, RunBudget, RunPermissions, WorkflowAgentRegistry
from sciagent_orchestrator import (
    DeterministicWorkInterpreter, ResearchController, ScientificArtifactRegistration, WorkflowScheduler,
)
from sciagent_scientific import ScientificAgent


TAIL_EVIDENCE = "TAIL_PAPER_RESULT: matched evaluation improved the metric by 1.2 percentage points."
RUN_ID = "run_literature_windows"


def _papers():
    return [LiteraturePaper(
        paper_id=f"2401.0000{index}v1", title=f"Controlled comparison {index}",
        authors=["Example Researcher"], published_at=date(2024, 1, 1),
        abstract="Controlled settings and matched baselines. " * 40
                 + (TAIL_EVIDENCE if index == 5 else "No tail result in this paper."),
        source_url=f"https://arxiv.org/abs/2401.0000{index}v1",
        pdf_url=f"https://arxiv.org/pdf/2401.0000{index}v1",
    ) for index in range(6)]


def _path(ref):
    return Path(url2pathname(urlparse(ref.uri).path))


def _finish(evidence, statement):
    return {"tool": "finish", "arguments": {
        "report": statement, "artifacts": [{
            "kind": "scientific_opinion", "path": "opinion.json", "media_type": "application/json",
            "summary": "Scientific conclusion", "content": {
                "verdict": "supports", "statement": statement, "evidence_artifact_ids": [evidence],
            },
        }],
    }}


def _receipt(turns):
    return json.loads(next(iter(turns[-1].tool_results.values())))["value"]


def _controller(tmp_path, client, papers):
    class Backend:
        def search(self, query, **kwargs):
            limit, page = kwargs["max_results"], kwargs["page"]
            return LiteratureSearchResult(papers[(page - 1) * limit:page * limit], "arxiv", query,
                                          page, page + 1 if page * limit < len(papers) else None,
                                          total_results=len(papers))
        def resolve(self, source):
            return next(paper for paper in papers if paper.key == source)
    scheduler = WorkflowScheduler(bindings={}, artifact_root=tmp_path / "artifacts", data_root=tmp_path / "data")
    registration = ScientificArtifactRegistration(scheduler.artifact_registry, scheduler.store)
    client.store = scheduler.store
    agent = ScientificAgent(client, literature_backend=Backend(), registration_port=registration)
    controller = ResearchController(scientific_port=agent, compiler=None, scheduler=scheduler,
                                    registry=WorkflowAgentRegistry(), interpreter=DeterministicWorkInterpreter())
    return controller, agent


def _request(kind):
    return ResearchRequest(goal="Locate, preserve and read the relevant comparison", required_evidence_kinds=[kind],
                           budget=RunBudget(max_llm_calls=8, timeout_seconds=30),
                           permissions=RunPermissions(execute_commands=False))


def test_search_pagination_returns_leads_then_only_chosen_fetch_is_preserved(tmp_path, monkeypatch):
    from sciagent_capabilities.literature import fulltext as tool
    from sciagent_components.literature.fulltext import PdfFetchError
    monkeypatch.setattr(tool, "fetch_pdf", lambda *args: (_ for _ in ()).throw(PdfFetchError("PDF unavailable")))
    class Client:
        tool_session_key = "test-native-tools/v1"
        step = 0
        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            self.step += 1
            if self.step == 1:
                return tool_turns([{"tool": "literature_search", "arguments": {"query": "comparison"}}])[0]
            run = self.store.load(RUN_ID)
            if self.step == 2:
                assert not any(ref.kind.startswith("literature_") for ref in run.artifacts.values())
                receipt = _receipt(turns)
                assert len(receipt["papers"]) == 5 and receipt["next_request"]["page"] == 2
                return tool_turns([{"tool": "literature_search", "arguments": receipt["next_request"]}])[0]
            if self.step == 3:
                receipt = _receipt(turns)
                assert receipt["next_request"] is None and receipt["total_results"] == 6
                assert TAIL_EVIDENCE in receipt["papers"][0]["abstract"]
                assert not any(ref.kind.startswith("literature_") for ref in run.artifacts.values())
                return tool_turns([{"tool": "fetch_literature", "arguments": {"source": receipt["papers"][0]["fetch_source"]}}])[0]
            paper, = [ref for ref in run.artifacts.values() if ref.kind == "literature_paper"]
            if self.step == 4:
                assert TAIL_EVIDENCE not in context.material_index
                self.paper_id = paper.id
                lines = _path(paper).read_text().splitlines()
                self.line = next(i for i, line in enumerate(lines, 1) if TAIL_EVIDENCE in line)
                return tool_turns([{"tool": "read_artifact", "arguments": {
                    "artifact_id": paper.id, "start_line": self.line, "end_line": self.line,
                }}])[0]
            assert self.step == 5
            read = _receipt(turns)
            assert TAIL_EVIDENCE in read["content"] and not read["truncated"]
            assert "artifact_reads" not in context.included_sections
            assert TAIL_EVIDENCE not in context.material_index
            return tool_turns([_finish(paper.id, TAIL_EVIDENCE)])[0]
    client = Client()
    controller, agent = _controller(tmp_path, client, _papers())
    run = controller.create_run(RUN_ID, _request("literature_paper"))
    assert run.status == "completed", run.terminal_error
    assert run.completion_violations == []
    assert len([ref for ref in run.artifacts.values() if ref.kind == "literature_paper"]) == 1
    assert not any(ref.kind == "literature_search" for ref in run.artifacts.values())
    ref = run.artifacts[client.paper_id]
    frozen = _path(ref).read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == ref.sha256
    _path(ref).write_bytes(b" " + frozen[1:])
    with pytest.raises(ArtifactReadError, match="sha256"):
        RegisteredArtifactReader([ref], run_id=RUN_ID).read_text(ref.id, start_line=client.line, end_line=client.line)


def test_search_fetch_read_finish_preserves_source_chain_and_history_only_body(tmp_path, monkeypatch):
    from sciagent_capabilities.literature import fulltext as tool
    evidence = "FULLTEXT_ONLY: held-out accuracy was 0.83 under a fixed evaluation protocol."
    calls = []
    def download(url, destination):
        calls.append(("download", url))
        destination.write_bytes(b"%PDF-1.7\ncontrolled source bytes\n%%EOF")
        return destination
    def parse(path):
        calls.append(("parse", Path(path)))
        return PdfText(f"## Page 1\n\n{evidence}\n", 1, [], "controlled-test-parser")
    monkeypatch.setattr(tool, "fetch_pdf", download)
    monkeypatch.setattr(tool, "parse_pdf", parse)
    class Client:
        tool_session_key = "test-native-tools/v1"
        step = 0
        def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
            self.step += 1
            if self.step == 1:
                return tool_turns([{"tool": "literature_search", "arguments": {"query": "controlled result"}}])[0]
            if self.step == 2:
                assert evidence not in context.text
                return tool_turns([{"tool": "fetch_literature", "arguments": {"source": _receipt(turns)["papers"][0]["fetch_source"]}}])[0]
            text, = [ref for ref in self.store.load(RUN_ID).artifacts.values() if ref.kind == "literature_fulltext"]
            if self.step == 3:
                assert evidence not in context.text and evidence not in json.dumps(turns, default=lambda x: x.model_dump())
                return tool_turns([{"tool": "read_artifact", "arguments": {"artifact_id": text.id}}])[0]
            assert self.step == 4 and evidence in _receipt(turns)["content"]
            assert evidence not in context.material_index and "artifact_reads" not in context.included_sections
            return tool_turns([_finish(text.id, evidence)])[0]
    controller, agent = _controller(tmp_path, Client(), _papers()[:1])
    run = controller.create_run(RUN_ID, _request("literature_fulltext"))
    assert run.status == "completed", run.terminal_error
    assert [name for name, _ in calls] == ["download", "parse"]
    by_kind = {ref.kind: ref for ref in run.artifacts.values()}
    paper, pdf, text = [by_kind[kind] for kind in ("literature_paper", "literature_pdf", "literature_fulltext")]
    assert pdf.metadata["source_artifact_id"] == paper.id and text.metadata["source_artifact_id"] == pdf.id
    assert run.final_opinion.evidence_artifact_ids == [text.id]
    state = agent.store.load(run.scientific_session.id)
    assert [event.tool for event in state.events if event.type == "observation"] == [
        "literature_search", "fetch_literature", "read_artifact", "finish",
    ]
    assert len(state.tool_turns) == 4 and all(len(turn.tool_results) == len(turn.tool_calls) for turn in state.tool_turns)
    assert set(state.authorized_artifact_ids) == {paper.id, pdf.id, text.id}
