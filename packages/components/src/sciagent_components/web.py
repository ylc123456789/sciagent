"""Provider-neutral web search and bounded webpage fetching components.

These components own HTTP/provider normalization only. Model-facing tools live in
sciagent_capabilities.web.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from importlib.metadata import version
from typing import Protocol
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

from sciagent_runtime.budget import DeadlineExceededError
from sciagent_runtime.http import NonPublicAddressError, ResponseTooLargeError, send_request


USER_AGENT = "sciagent/0.1 (+https://github.com/ylc123456789/sciagent)"


class WebSearchError(RuntimeError):
    """A web search did not produce a normalized result."""

    def __init__(self, message: str, *, error_type: str = "invalid_response",
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after


class WebFetchError(RuntimeError):
    """A webpage could not be fetched or converted to text."""

    def __init__(self, message: str, *, error_type: str = "invalid_response",
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class WebSearchItem:
    title: str
    url: str
    snippet: str
    score: float | None = None
    published_at: str | None = None
    snippet_truncated: bool = False
    summary: str = ""
    summary_truncated: bool = False
    authority: int | None = None
    site_name: str | None = None


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    provider: str
    query: str
    results: list[WebSearchItem] = field(default_factory=list)
    incomplete_reason: str | None = None


class WebSearchBackend(Protocol):
    """Provider-neutral search interface used by the Web capability."""

    name: str

    def search(self, query: str) -> WebSearchResult:
        """Return all received sources within the transport response limit."""


MAX_WEB_SNIPPET_CHARS = 2000


def _retry_after(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            return max(0.0, (deadline - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0


class _HTMLClosureValidator(HTMLParser):
    """Reject only unterminated tags that can hide the rest of a page."""

    _tracked = frozenset({"script", "style", "noscript", "template", "title"})
    _foreign = frozenset({"svg", "math"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._open: list[str] = []
        self._foreign_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._foreign:
            self._foreign_depth += 1
        if tag in self._tracked and not (tag == "title" and self._foreign_depth):
            self._open.append(tag)

    def handle_startendtag(self, tag: str, attrs) -> None:
        # A self-closing foreign element does not open a depth scope.
        return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._tracked and tag in self._open:
            if self._open[-1] != tag:
                raise ValueError(f"misnested {tag} element")
            self._open.pop()
        if tag in self._foreign and self._foreign_depth:
            self._foreign_depth -= 1

    def close(self) -> None:
        super().close()
        if self._open:
            raise ValueError(f"unterminated {self._open[-1]} element")


class _WebMarkdownConverter(MarkdownConverter):
    """Project-specific presentation on top of markdownify's public API."""

    def convert_a(self, el, text, parent_tags):
        href = el.get("href")
        if not href or not text.strip():
            return text
        if "_noformat" in parent_tags:
            return f"{text} ({href})"
        if el.find("pre") is not None:
            return f"{text.rstrip(chr(10))}\n\nLink: {href}\n\n"
        return super().convert_a(el, text, parent_tags)

    def convert_img(self, el, text, parent_tags):
        alt = el.get("alt") or ""
        return alt if "_noformat" in parent_tags else self.escape(alt, parent_tags)

    def convert_pre(self, el, text, parent_tags):
        if not text:
            return ""
        longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
        fence = "`" * max(3, longest + 1)
        return f"\n\n{fence}\n{text}\n{fence}\n\n"


def _validate_html_closure(body: str) -> None:
    validator = _HTMLClosureValidator()
    validator.feed(body)
    validator.close()


def _prepare_html(body: str, final_url: str) -> tuple[BeautifulSoup, str]:
    _validate_html_closure(body)
    soup = BeautifulSoup(body, "html.parser")
    titles = [
        title.get_text(" ", strip=True)
        for title in soup.find_all("title")
        if title.find_parent(["svg", "math", "noscript", "template"]) is None
    ]
    title = " ".join(part for part in titles if part).strip()
    for node in soup.find_all(["script", "style", "noscript", "template", "title"]):
        node.decompose()
    for link in soup.find_all("a"):
        href = link.get("href")
        if not isinstance(href, str) or not href.strip():
            link.attrs.pop("href", None)
            continue
        href = href.strip()
        if any(character.isspace() or ord(character) < 32 or ord(character) == 127
               for character in href):
            link.attrs.pop("href", None)
            continue
        try:
            parsed = urlparse(href)
            if href.startswith("//") and not parsed.netloc:
                raise WebFetchError("invalid link", error_type="invalid_url")
            if parsed.scheme:
                _validate_url(href)
            target = urljoin(final_url, href)
            _validate_url(target)
        except (ValueError, WebFetchError):
            link.attrs.pop("href", None)
        else:
            link["href"] = target
    for br in soup.find_all("br"):
        if br.find_parent("pre") is not None:
            br.replace_with("\n")
    for inner in reversed(soup.find_all("pre")):
        if inner.find_parent("pre") is not None:
            inner.insert_before("\n")
            inner.insert_after("\n")
            inner.unwrap()
    return soup, title


def _render_html(body: str, final_url: str) -> tuple[str, str]:
    soup, title = _prepare_html(body, final_url)
    text = _WebMarkdownConverter(
        heading_style="ATX", code_block_style="fenced",
        strip_pre=None, strip_document="strip",
    ).convert_soup(soup)
    return text, title


@dataclass(frozen=True, slots=True)
class WebPage:
    source_url: str
    final_url: str
    title: str
    text: str
    content_type: str
    parser: str
    fetched_at: str
    parser_version: str | None = None


class WebPageFetcher:
    """Fetch one HTML/text page with bounded bytes and no browser execution."""

    def __init__(self, *, timeout_seconds: int = 30, max_response_bytes: int = 4 * 1024 * 1024) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes

    def fetch(self, url: str) -> WebPage:
        try:
            response = self._request(url)
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            retry_after = _retry_after(error.response.headers.get("Retry-After"))
            raise WebFetchError(
                f"webpage HTTP {status}; request rejected",
                error_type="rate_limited" if status == 429 else "request_rejected",
                retry_after=retry_after or None,
            ) from None
        except DeadlineExceededError:
            raise
        except (httpx.TimeoutException, TimeoutError):
            raise WebFetchError("webpage request timed out", error_type="timeout") from None
        except NonPublicAddressError:
            raise WebFetchError(
                "webpage host must resolve to a public address", error_type="invalid_url",
            ) from None
        except httpx.InvalidURL:
            raise WebFetchError("url must be an http(s) URL without credentials", error_type="invalid_url") from None
        except ResponseTooLargeError:
            raise WebFetchError(
                "webpage response exceeded the configured byte limit",
                error_type="response_too_large",
            ) from None
        except httpx.DecodingError:
            raise WebFetchError(
                "webpage response body could not be decoded", error_type="invalid_response",
            ) from None
        except httpx.TransportError:
            raise WebFetchError("webpage network request failed", error_type="network_error") from None

        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
            raise WebFetchError(
                f"unsupported webpage content type: {content_type or 'unknown'}",
                error_type="unsupported_content_type",
            )
        try:
            body = response.content.decode("utf-8")
        except (UnicodeDecodeError, LookupError):
            raise WebFetchError("webpage is not valid UTF-8 text", error_type="invalid_encoding") from None
        if "\x00" in body:
            raise WebFetchError("webpage contains NUL bytes", error_type="invalid_encoding") from None
        final_url = str(response.url)
        if content_type == "text/plain":
            title, text, parser = "", body, "plain"
        else:
            try:
                text, title = _render_html(body, final_url)
            except Exception as error:
                raise WebFetchError("webpage HTML could not be parsed", error_type="parse_failed") from error
            parser = "markdownify/html.parser"
        if not text.strip():
            raise WebFetchError(
                "webpage has no extractable text", error_type="parse_failed",
            )
        return WebPage(
            source_url=url, final_url=final_url, title=title, text=text,
            content_type=content_type, parser=parser,
            fetched_at=datetime.now(UTC).isoformat(),
            parser_version=(f"markdownify={version('markdownify')};beautifulsoup4={version('beautifulsoup4')}"
                            if parser == "markdownify/html.parser" else None),
        )

    def _request(self, url: str) -> httpx.Response:
        deadline = time.monotonic() + self.timeout_seconds
        current_url = url
        for redirect_count in range(6):
            _validate_url(current_url)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WebFetchError("webpage request timed out", error_type="timeout")
            request = httpx.Request(
                "GET", current_url,
                headers={"Accept": "text/html, text/plain", "User-Agent": USER_AGENT},
            )
            try:
                return send_request(
                    request, timeout=remaining,
                    max_response_bytes=self.max_response_bytes,
                    public_only=True,
                )
            except httpx.HTTPStatusError as error:
                location = error.response.headers.get("location")
                if error.response.status_code not in {301, 302, 303, 307, 308} or not location:
                    raise
                if redirect_count == 5:
                    raise WebFetchError("webpage exceeded 5 redirects", error_type="redirect_limit") from None
                current_url = urljoin(str(error.response.url), location)
        raise WebFetchError("webpage exceeded 5 redirects", error_type="redirect_limit")


def _validate_url(url: str) -> None:
    try:
        parsed = urlparse(url)
        parsed.port
    except ValueError:
        raise WebFetchError("url must be an http(s) URL without credentials", error_type="invalid_url") from None
    if (parsed.scheme not in {"http", "https"} or parsed.username is not None
            or parsed.password is not None
            or not parsed.netloc or not parsed.hostname):
        raise WebFetchError("url must be an http(s) URL without credentials", error_type="invalid_url")
