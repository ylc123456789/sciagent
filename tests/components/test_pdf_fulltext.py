"""PDF download bounds and real page extraction do not create artifact authority."""

import json
import os
import subprocess
import sys
import time

import httpx
import pymupdf
import pymupdf4llm
import pytest

from resagent2_components.literature import fulltext
from resagent2_runtime.budget import DeadlineExceededError, execution_budget
from resagent2_runtime.http import ResponseTooLargeError


PDF_BYTES = b"%PDF-1.7\nsource bytes"


def response(content=PDF_BYTES):
    return httpx.Response(200, content=content)


def test_download_preserves_pdf_bytes_and_passes_limits(monkeypatch, tmp_path):
    calls = []

    def send(request, **kwargs):
        calls.append((request, kwargs))
        return response()

    monkeypatch.setattr(fulltext, "send_request", send)
    target = tmp_path / "source.pdf"
    assert fulltext.fetch_pdf(
        "https://example.test/paper.pdf", target, timeout_seconds=12, max_bytes=1024,
    ) == target
    assert target.read_bytes() == PDF_BYTES
    request, kwargs = calls[0]
    assert request.headers["accept"] == "application/pdf"
    assert "ResAgent2" in request.headers["user-agent"]
    assert kwargs == {"timeout": 12, "max_response_bytes": 1024, "follow_redirects": True}


@pytest.mark.parametrize("url", ["file:///tmp/x", "ftp://example.test/x", "https:///x",
                                  "https://name:secret@example.test/x"])
def test_download_rejects_non_http_or_credentialed_urls(monkeypatch, tmp_path, url):
    monkeypatch.setattr(fulltext, "send_request", lambda *a, **k: pytest.fail("unexpected HTTP"))
    with pytest.raises(fulltext.PdfFetchError, match="HTTP\\(S\\)"):
        fulltext.fetch_pdf(url, tmp_path / "source.pdf")


def test_download_rejects_html_and_does_not_create_file(monkeypatch, tmp_path):
    monkeypatch.setattr(fulltext, "send_request", lambda *a, **k: response(b"<html>Login</html>"))
    target = tmp_path / "source.pdf"
    with pytest.raises(fulltext.PdfFetchError, match="PDF signature"):
        fulltext.fetch_pdf("https://example.test/paper", target)
    assert not target.exists()


@pytest.mark.parametrize("url", ["https://[invalid", "https://example.test:wrong/paper.pdf"])
def test_download_reports_malformed_url(monkeypatch, tmp_path, url):
    monkeypatch.setattr(fulltext, "send_request", lambda *a, **k: pytest.fail("unexpected HTTP"))
    with pytest.raises(fulltext.PdfFetchError, match="Invalid PDF URL"):
        fulltext.fetch_pdf(url, tmp_path / "source.pdf")


@pytest.mark.parametrize("failure", [ResponseTooLargeError("over 20 decoded bytes"),
                                      httpx.ConnectError("offline"), TimeoutError("slow")])
def test_download_failures_keep_cause_and_no_file(monkeypatch, tmp_path, failure):
    def send(*args, **kwargs):
        raise failure

    monkeypatch.setattr(fulltext, "send_request", send)
    target = tmp_path / "source.pdf"
    with pytest.raises(fulltext.PdfFetchError) as caught:
        fulltext.fetch_pdf("https://example.test/paper", target)
    assert caught.value.__cause__ is failure
    assert not target.exists()


def test_download_keeps_http_status(monkeypatch, tmp_path):
    def send(request, **kwargs):
        httpx.Response(429, request=request).raise_for_status()

    monkeypatch.setattr(fulltext, "send_request", send)
    with pytest.raises(fulltext.PdfFetchError, match="HTTP 429"):
        fulltext.fetch_pdf("https://example.test/paper", tmp_path / "source.pdf")


def test_download_does_not_overwrite_existing_destination(monkeypatch, tmp_path):
    monkeypatch.setattr(fulltext, "send_request", lambda *a, **k: response())
    target = tmp_path / "source.pdf"
    target.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        fulltext.fetch_pdf("https://example.test/paper", target)
    assert target.read_bytes() == b"original"


def test_download_does_not_swallow_persistence_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(fulltext, "send_request", lambda *a, **k: response())
    with pytest.raises(FileNotFoundError):
        fulltext.fetch_pdf("https://example.test/paper", tmp_path / "absent" / "paper.pdf")


def pdf(tmp_path, texts):
    path = tmp_path / "source.pdf"
    with pymupdf.open() as document:
        for text in texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)
    return path


def test_real_pdf_extraction_preserves_pages_and_warns_about_empty_page(tmp_path):
    path = pdf(tmp_path, ["First page scientific result.", "", "Third page limitation."])
    original = path.read_bytes()
    parsed = fulltext.parse_pdf(path)
    assert parsed.page_count == 3
    assert parsed.parser_version
    assert parsed.warnings == ["Page 2 has no extracted text; OCR is disabled."]
    assert "First page scientific result." in parsed.markdown
    assert "Third page limitation." in parsed.markdown
    assert "## Page 1" in parsed.markdown
    assert "## Page 2" in parsed.markdown
    assert "## Page 3" in parsed.markdown
    assert "OCR: disabled" in parsed.markdown
    assert path.read_bytes() == original


def test_pdf_extraction_uses_no_ocr_or_image_writes(monkeypatch, tmp_path):
    path = pdf(tmp_path, ["content"])
    calls = []

    def parse(source, **kwargs):
        calls.append((source, kwargs))
        return [{"text": "content"}]

    monkeypatch.setattr(pymupdf4llm, "to_markdown", parse)
    assert fulltext._parse_pdf_local(path).page_count == 1
    assert calls == [(str(path), {
        "page_chunks": True, "use_ocr": False, "show_progress": False,
        "write_images": False, "embed_images": False,
    })]


def test_no_text_is_a_parse_failure_and_preserves_pdf(tmp_path):
    path = pdf(tmp_path, [""])
    original = path.read_bytes()
    with pytest.raises(fulltext.PdfParseError, match="no extractable text; OCR is disabled"):
        fulltext.parse_pdf(path)
    assert path.read_bytes() == original


def test_malformed_pdf_preserves_original(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(PDF_BYTES)
    with pytest.raises(fulltext.PdfParseError, match="could not parse"):
        fulltext.parse_pdf(path)
    assert path.read_bytes() == PDF_BYTES


def test_parser_does_not_swallow_io_failure(monkeypatch, tmp_path):
    path = pdf(tmp_path, ["content"])

    def fail(*args, **kwargs):
        raise OSError("filesystem failure")

    monkeypatch.setattr(pymupdf4llm, "to_markdown", fail)
    with pytest.raises(OSError, match="filesystem failure"):
        fulltext._parse_pdf_local(path)


def test_parser_respects_expired_run_before_reading(tmp_path):
    with execution_budget(max_llm_calls=1, timeout_seconds=0):
        with pytest.raises(DeadlineExceededError):
            fulltext.parse_pdf(tmp_path / "does-not-exist.pdf")


def test_parser_preserves_child_io_failure(tmp_path):
    with pytest.raises(FileNotFoundError):
        fulltext.parse_pdf(tmp_path / "does-not-exist.pdf")


def test_parser_invokes_only_fixed_module_with_current_python(monkeypatch, tmp_path):
    monkeypatch.setenv("RESAGENT2_PDF_PARSE_TIMEOUT_SECONDS", "900")
    path = tmp_path / "source.pdf"
    calls = []
    parsed = fulltext.PdfText("## Page 1\ntext", 1, [], "test")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "markdown": parsed.markdown, "page_count": parsed.page_count,
            "warnings": parsed.warnings, "parser_version": parsed.parser_version,
        }), "")

    monkeypatch.setattr(fulltext, "run_process", run)
    assert fulltext.parse_pdf(path) == parsed
    assert calls == [([
        sys.executable, "-m", "resagent2_components.literature.fulltext", str(path.resolve()),
    ], {"timeout": 300})]


def test_parser_child_stdout_contains_only_json():
    # Simulate both Python and native-library noise inside the real worker entry.
    code = """
import os
import sys
from resagent2_components.literature import fulltext
def extract(path):
    print('python parser notice')
    os.write(1, b'native parser notice\\n')
    return fulltext.PdfText('## Page 1\\nExact text.', 1, [], 'test')
fulltext._parse_pdf_local = extract
sys.argv = ['fulltext', '/controlled/source.pdf']
raise SystemExit(fulltext._main())
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["markdown"] == "## Page 1\nExact text."
    assert "python parser notice" in result.stderr
    assert "native parser notice" in result.stderr


@pytest.mark.parametrize("run_deadline", [False, True])
def test_parser_timeout_terminates_child_and_preserves_original(monkeypatch, tmp_path, run_deadline):
    path = pdf(tmp_path, ["original"])
    original = path.read_bytes()
    pid_path = tmp_path / "parser.pid"
    real_run = fulltext.run_process

    def slow_worker(command, **kwargs):
        assert command[:3] == [sys.executable, "-m", "resagent2_components.literature.fulltext"]
        return real_run([
            sys.executable, "-c",
            "import os, pathlib, time; pathlib.Path(os.environ['TEST_PARSER_PID']).write_text(str(os.getpid())); time.sleep(30)",
        ], env={**os.environ, "TEST_PARSER_PID": str(pid_path)}, **kwargs)

    monkeypatch.setattr(fulltext, "run_process", slow_worker)
    started = time.monotonic()
    if run_deadline:
        with execution_budget(max_llm_calls=1, timeout_seconds=0.3):
            with pytest.raises(DeadlineExceededError):
                fulltext.parse_pdf(path)
    else:
        with pytest.raises(fulltext.PdfParseError, match="exceeded 0.3 seconds"):
            fulltext.parse_pdf(path, timeout_seconds=0.3)
    assert time.monotonic() - started < 2
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert path.read_bytes() == original
