"""CLI and real E2E share literature backend and parser configuration."""

import pytest

from sciagent_components import (
    ArxivLiteratureBackend,
    MultiSourceLiteratureBackend,
    OpenAlexLiteratureBackend,
)
from sciagent_cli import composition


@pytest.mark.parametrize("api_key", [None, "private-test-key"])
def test_cli_and_e2e_wire_same_literature_backends(tmp_path, monkeypatch, api_key):
    from e2e import real_e2e

    if api_key:
        monkeypatch.setenv("OPENALEX_API_KEY", api_key)
    else:
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.setattr(composition, "_client", lambda: object())
    app = composition.build_application(data_root=tmp_path / "cli")
    e2e_agent = real_e2e._application(tmp_path / "e2e").controller.scientific_port
    for agent in (app.controller.scientific_port, e2e_agent):
        backend = agent.literature_backend
        assert isinstance(backend, MultiSourceLiteratureBackend)
        assert len(backend.backends) == 2
        assert isinstance(backend.backends[0], ArxivLiteratureBackend)
        assert isinstance(backend.backends[1], OpenAlexLiteratureBackend)
        assert backend.backends[1]._api_key == api_key


@pytest.mark.parametrize("configured,expected", [(None, 300), ("180", 180)])
def test_cli_binds_pdf_parser_timeout(tmp_path, monkeypatch, configured, expected):
    import json
    import subprocess
    from sciagent_components.literature import fulltext

    if configured is None:
        monkeypatch.delenv("SCIAGENT_PDF_PARSE_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("SCIAGENT_PDF_PARSE_TIMEOUT_SECONDS", configured)
    monkeypatch.setattr(composition, "_client", lambda: object())
    calls = []

    def parse_worker(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "markdown": "## Page 1\nText", "page_count": 1,
            "warnings": [], "parser_version": "test",
        }), "")

    monkeypatch.setattr(fulltext, "run_process", parse_worker)
    app = composition.build_application(data_root=tmp_path / "data")
    parsed = app.controller.scientific_port.literature_parser(tmp_path / "source.pdf")
    assert parsed.markdown == "## Page 1\nText"
    assert calls[0][1]["timeout"] == expected


@pytest.mark.parametrize("configured", ["0", "-1", "invalid", "1.5", ""])
def test_invalid_pdf_timeout_fails_before_io_or_client_creation(tmp_path, monkeypatch, configured):
    monkeypatch.setenv("SCIAGENT_PDF_PARSE_TIMEOUT_SECONDS", configured)
    monkeypatch.setattr(composition, "_client", lambda: pytest.fail("unexpected client creation"))
    data = tmp_path / "data"
    with pytest.raises(ValueError, match="SCIAGENT_PDF_PARSE_TIMEOUT_SECONDS must be"):
        composition.build_application(data_root=data)
    assert not data.exists()
