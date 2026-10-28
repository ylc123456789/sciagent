"""Bounded PDF acquisition and page-preserving text extraction, without registration."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

import httpx

from resagent2_runtime.budget import DeadlineExceededError
from resagent2_runtime.http import ResponseTooLargeError, send_request

from ..process import run_process
from ._http import USER_AGENT


DEFAULT_PDF_PARSE_TIMEOUT_SECONDS = 300


class PdfFetchError(RuntimeError):
    """A literature PDF could not be obtained within the download limits."""


class PdfParseError(RuntimeError):
    """A PDF did not yield readable text; its original bytes remain available."""


@dataclass(frozen=True)
class PdfText:
    """Text extracted from physical PDF pages and the limits of that extraction."""

    markdown: str
    page_count: int
    warnings: list[str]
    parser_version: str


def fetch_pdf(
    url: str, destination: Path, *, timeout_seconds: float = 60.0,
    max_bytes: int = 32 * 1024 * 1024,
) -> Path:
    """Download a source-provided PDF URL into a new file, without overwriting it."""
    try:
        parts = urlsplit(url)
    except ValueError as error:
        raise PdfFetchError(f"Invalid PDF URL: {error}") from error
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None):
        raise PdfFetchError("PDF URL must be an HTTP(S) URL without credentials")
    if max_bytes < 1 or timeout_seconds <= 0:
        raise ValueError("PDF byte and timeout limits must be positive")
    try:
        request = httpx.Request(
            "GET", url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"},
        )
    except httpx.InvalidURL as error:
        raise PdfFetchError(f"Invalid PDF URL: {error}") from error
    try:
        response = send_request(
            request, timeout=timeout_seconds, max_response_bytes=max_bytes,
            follow_redirects=True,
        )
    except DeadlineExceededError:
        raise
    except httpx.HTTPStatusError as error:
        raise PdfFetchError(f"PDF download failed: HTTP {error.response.status_code}") from error
    except ResponseTooLargeError as error:
        raise PdfFetchError(str(error)) from error
    except (httpx.HTTPError, TimeoutError) as error:
        raise PdfFetchError(f"PDF download failed: {type(error).__name__}: {error}") from error
    content = response.content
    if not content.startswith(b"%PDF-"):
        raise PdfFetchError("PDF download did not contain a PDF signature")
    # Opening outside the try prevents deleting a pre-existing destination.
    output = Path(destination)
    handle = output.open("xb")
    try:
        with handle:
            handle.write(content)
    except OSError:
        output.unlink(missing_ok=True)
        raise
    return output


def parse_pdf(
    path: Path, *, timeout_seconds: float = DEFAULT_PDF_PARSE_TIMEOUT_SECONDS,
) -> PdfText:
    """Parse with a bounded fixed subprocess; the parent retains the original."""
    command = [sys.executable, "-m", "resagent2_components.literature.fulltext", str(Path(path).resolve())]
    try:
        result = run_process(command, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise PdfParseError(f"PDF parsing exceeded {timeout_seconds:g} seconds") from error
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        detail = result.stderr.strip()[-2000:]
        raise PdfParseError(
            f"PDF parser returned no valid result (exit {result.returncode}): {detail}"
        ) from error
    if result.returncode != 0:
        if payload.get("error_type") == "os_error":
            raise OSError(payload.get("errno"), payload["message"], payload.get("filename"))
        raise PdfParseError(payload.get("message", f"PDF parser exited {result.returncode}"))
    try:
        return PdfText(**payload)
    except TypeError as error:
        raise PdfParseError("PDF parser returned an invalid result") from error


def _parse_pdf_local(path: Path) -> PdfText:
    """Worker implementation: extract physical pages without OCR or image writes."""
    source = Path(path)
    with source.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise PdfParseError("PDF signature is missing")
    import pymupdf4llm

    parser_version = version("pymupdf4llm")
    try:
        pages = pymupdf4llm.to_markdown(
            str(source), page_chunks=True, use_ocr=False,
            show_progress=False, write_images=False, embed_images=False,
        )
    except (RuntimeError, ValueError) as error:
        raise PdfParseError(f"PyMuPDF4LLM could not parse the PDF: {error}") from error
    if not isinstance(pages, list) or not pages:
        raise PdfParseError("PDF parser returned no pages")
    warnings = []
    sections = []
    readable_pages = 0
    for page_number, page in enumerate(pages, start=1):
        text = page.get("text") if isinstance(page, dict) else None
        if not isinstance(text, str):
            raise PdfParseError(f"PDF parser returned no text field for page {page_number}")
        if text.strip():
            readable_pages += 1
        else:
            warnings.append(f"Page {page_number} has no extracted text; OCR is disabled.")
            text = "[No text extracted from this page; OCR is disabled.]"
        sections.append(f"## Page {page_number}\n\n{text.rstrip()}")
    if not readable_pages:
        raise PdfParseError("PDF has no extractable text; OCR is disabled")
    header = f"# PDF text\n\nParser: PyMuPDF4LLM {parser_version}\n\nOCR: disabled"
    if warnings:
        header += "\n\nExtraction warnings:\n" + "\n".join(f"- {item}" for item in warnings)
    return PdfText(
        markdown=header + "\n\n" + "\n\n".join(sections) + "\n",
        page_count=len(pages), warnings=warnings, parser_version=parser_version,
    )


def _main() -> int:
    """Fixed local parser entry; reserve stdout exclusively for the JSON result."""
    if len(sys.argv) != 2:
        raise SystemExit("expected one local PDF path")
    sys.stdout.flush()
    result_fd = os.dup(sys.stdout.fileno())
    # Cover Python prints and native-library output, without changing the parent.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    exit_code = 0
    try:
        payload = asdict(_parse_pdf_local(Path(sys.argv[1])))
    except PdfParseError as error:
        payload = {"error_type": "parse_error", "message": str(error)}
        exit_code = 1
    except OSError as error:
        payload = {"error_type": "os_error", "message": error.strerror or str(error),
                   "errno": error.errno, "filename": error.filename}
        exit_code = 1
    with os.fdopen(result_fd, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False)
        output.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
