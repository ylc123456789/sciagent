"""Deterministic tests for provider-neutral web components."""

import socket
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from resagent2_components import (
    WebFetchError,
    WebPageFetcher,
)
from resagent2_components.web import _retry_after
from resagent2_runtime.http import NonPublicAddressError


def _response(content: bytes, *, status=200, content_type="application/json", url="https://api.test/search"):
    return httpx.Response(
        status,
        content=content,
        headers={"content-type": content_type},
        request=httpx.Request("POST", url),
    )


@pytest.fixture(autouse=True)
def resolve_test_hosts_as_public(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
        ],
    )


def test_retry_after_accepts_seconds_and_http_date():
    assert _retry_after("2") == 2.0
    assert _retry_after("not-a-date") == 0.0
    future = (datetime.now(UTC) + timedelta(seconds=30)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    assert 0 < _retry_after(future) <= 30


def test_web_page_fetcher_extracts_html_and_ignores_noncontent(monkeypatch):
    html = b"""<html><head><title>  Page title </title><style>hide</style></head>
    <body><script>bad()</script><p>Hello <b>world</b>.</p>
    <noscript>also hidden</noscript><template>hidden</template></body></html>"""
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html, content_type="text/html; charset=utf-8",
            url="https://example.test/page",
        ),
    )
    page = WebPageFetcher().fetch("https://example.test/page")

    assert page.title == "Page title"
    assert page.text == "Hello **world**."
    assert page.parser == "markdownify/html.parser"
    assert page.final_url == "https://example.test/page"


@pytest.mark.parametrize("html, expected", [
    ("<pre>    line  one\n\n\tline two  \n</pre>", "```\n    line  one\n\n\tline two  \n\n```"),
    (
        "<pre><code>for <span>x</span> in xs:\n    <b>print</b>(x)\n\n    pass\n</code></pre>",
        "```\nfor x in xs:\n    print(x)\n\n    pass\n\n```",
    ),
    (
        "<p>  Before   text </p><pre><code>  a  b\n\n  c</code></pre><p>After  text</p>",
        "Before text\n\n```\n  a  b\n\n  c\n```\n\nAfter text",
    ),
    ("<pre>  &lt;tag&gt;&amp;value<br>    next</pre>", "```\n  <tag>&value\n    next\n```"),
    ("<pre>  first\n    second", "```\n  first\n    second\n```"),
])
def test_web_page_fetcher_preserves_preformatted_whitespace(monkeypatch, html, expected):
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html.encode(), content_type="text/html", url="https://example.test/page",
        ),
    )

    assert WebPageFetcher().fetch("https://example.test/page").text == expected


def test_web_page_fetcher_preserves_links_inside_preformatted_text(monkeypatch):
    html = b'<pre>    <a href="/docs">read  docs</a>\n    next</pre><a href="/other">Other  link</a>'
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html, content_type="text/html", url="https://example.test/page",
        ),
    )

    assert WebPageFetcher().fetch("https://example.test/page").text == (
        "```\n    read  docs (https://example.test/docs)\n    next\n```\n\n"
        "[Other link](https://example.test/other)"
    )


@pytest.mark.parametrize("html, expected", [
    (
        '<pre><a href="/docs">line<br/>next</a></pre>',
        "```\nline\nnext (https://example.test/docs)\n```",
    ),
    (
        '<pre>before<a href="/docs">  \n\t </a>after</pre>',
        "```\nbefore  \n\t after\n```",
    ),
])
def test_web_page_fetcher_preserves_preformatted_link_whitespace(monkeypatch, html, expected):
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html.encode(), content_type="text/html", url="https://example.test/page",
        ),
    )

    assert WebPageFetcher().fetch("https://example.test/page").text == expected


def test_web_page_fetcher_preserves_links_and_nested_visible_labels(monkeypatch):
    calls = []
    html = b"""<title>Link page</title><p>Before</p>
    <a href="https://other.test/paper?q=1&amp;lang=en">Original <b>paper</b> &amp; <i>code</i>.</a>
    <a href="../download/file.pdf">Download <span>P</span>DF</a><p>After</p>"""

    def send(request, **kwargs):
        calls.append(str(request.url))
        return _response(html, content_type="text/html", url=str(request.url))

    monkeypatch.setattr("resagent2_components.web.send_request", send)
    page = WebPageFetcher().fetch("https://example.test/articles/index")

    assert page.text == (
        "Before\n\n[Original **paper** & *code*.](https://other.test/paper?q=1&lang=en)\n"
        "[Download PDF](https://example.test/download/file.pdf)\n\nAfter"
    )
    assert calls == ["https://example.test/articles/index"]
    assert page.title == "Link page"
    assert page.source_url == page.final_url == calls[0]
    assert page.content_type == "text/html"
    assert page.parser == "markdownify/html.parser"
    assert datetime.fromisoformat(page.fetched_at).tzinfo is not None


def test_web_page_fetcher_resolves_links_against_redirected_page(monkeypatch):
    calls = []

    def send(request, **kwargs):
        calls.append(str(request.url))
        if len(calls) == 1:
            response = httpx.Response(
                302, headers={"location": "https://other.test/docs/page"}, request=request,
            )
            response.raise_for_status()
        return httpx.Response(
            200, text='<a href="next">Next page</a>',
            headers={"content-type": "text/html"}, request=request,
        )

    monkeypatch.setattr("resagent2_components.web.send_request", send)
    page = WebPageFetcher().fetch("https://example.test/start")

    assert page.text == "[Next page](https://other.test/docs/next)"
    assert page.source_url == "https://example.test/start"
    assert page.final_url == "https://other.test/docs/page"
    assert calls == [page.source_url, page.final_url]


@pytest.mark.parametrize("href", [
    "", "   ", "mailto:author@example.test", "javascript:alert(1)",
    "data:text/plain,content", "file:///tmp/secret", "ftp://example.test/file",
    "https://user:pass@example.test/private", "//user@example.test/private",
    "http://[", "https://", "//", "https://example.test:bad/page",
    "https://example.test/has space", "https://example.test/line&#10;break",
])
def test_web_page_fetcher_keeps_labels_without_invalid_link_targets(monkeypatch, href):
    html = f'<p>Before</p><a href="{href}">Visible <b>label</b></a><p>After</p>'
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html.encode(), content_type="text/html", url="https://example.test/page",
        ),
    )
    page = WebPageFetcher().fetch("https://example.test/page")

    assert page.text == "Before\n\nVisible **label**\n\nAfter"


def test_web_page_fetcher_ignores_hidden_links_and_empty_labels(monkeypatch):
    html = b"""<title>Visible title</title><body>
    <script><a href="https://hidden.test/script">Script link</a></script>
    <style><a href="https://hidden.test/style">Style link</a></style>
    <noscript><a href="https://hidden.test/noscript">Noscript link</a></noscript>
    <template><a href="https://hidden.test/template">Template link</a></template>
    <a href="https://empty.test/"><img src="cover.png"></a>
    <a>Plain label</a><a href="//other.test/page">Open <b>link</b>"""
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html, content_type="text/html", url="https://example.test/page",
        ),
    )
    page = WebPageFetcher().fetch("https://example.test/page")

    assert page.title == "Visible title"
    assert page.text == "Plain label[Open **link**](https://other.test/page)"


def test_web_page_fetcher_supports_plain_text(monkeypatch):
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            b"plain body", content_type="text/plain; charset=utf-8",
            url="https://example.test/text",
        ),
    )
    page = WebPageFetcher().fetch("https://example.test/text")
    assert page.title == ""
    assert page.text == "plain body"
    assert page.parser == "plain"


def test_web_page_fetcher_requires_utf8(monkeypatch):
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            b"\xff", content_type="text/plain; charset=latin-1",
            url="https://example.test/text",
        ),
    )
    with pytest.raises(WebFetchError, match="valid UTF-8") as raised:
        WebPageFetcher().fetch("https://example.test/text")
    assert raised.value.error_type == "invalid_encoding"


@pytest.mark.parametrize("content_type, content", [
    ("text/html", b"<title>Only title</title><script>render()</script>"),
    ("text/html", b'<img src="/figure.png">'),
    ("text/html", b'<a href="/paper"><img src="/figure.png" alt=""></a>'),
    ("text/html", b'<a href="/paper"><img src="/figure.png" alt="  "></a>'),
    ("text/plain", b" \n\t"),
])
def test_web_page_fetcher_rejects_empty_extraction(monkeypatch, content_type, content):
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            content, content_type=content_type, url="https://example.test/page",
        ),
    )
    with pytest.raises(WebFetchError) as raised:
        WebPageFetcher().fetch("https://example.test/page")
    assert raised.value.error_type == "parse_failed"


@pytest.mark.parametrize("url", [
    "file:///tmp/secret",
    "ftp://example.test/file",
    "https://user:pass@example.test/private",
    "https://@example.test/private",
    "https://:pass@example.test/private",
    "http://[",
])
def test_web_page_fetcher_rejects_non_http_or_malformed_urls(url):
    with pytest.raises(WebFetchError) as raised:
        WebPageFetcher().fetch(url)
    assert raised.value.error_type == "invalid_url"


@pytest.mark.parametrize("url", [
    "http://localhost/private",
    "http://127.0.0.1/private",
    "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/private",
])
def test_web_page_fetcher_rejects_nonpublic_hosts(url):
    with pytest.raises(WebFetchError) as raised:
        WebPageFetcher().fetch(url)
    assert raised.value.error_type == "invalid_url"


def test_web_page_fetcher_rejects_host_resolving_to_private_address(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 0)),
        ],
    )
    with pytest.raises(WebFetchError) as raised:
        WebPageFetcher().fetch("https://internal.example/page")
    assert raised.value.error_type == "invalid_url"


def test_web_page_fetcher_revalidates_redirect_targets(monkeypatch):
    calls = []

    def send(request, **kwargs):
        calls.append(request)
        assert kwargs["public_only"] is True
        if request.url.host == "127.0.0.1":
            raise NonPublicAddressError("private address")
        response = httpx.Response(
            302, headers={"location": "http://127.0.0.1/private"}, request=request,
        )
        response.raise_for_status()
        return response

    monkeypatch.setattr("resagent2_components.web.send_request", send)
    with pytest.raises(WebFetchError) as raised:
        WebPageFetcher().fetch("https://example.test/start")
    assert raised.value.error_type == "invalid_url"
    assert len(calls) == 2


def test_web_page_fetcher_follows_relative_redirect_and_preserves_final_url(monkeypatch):
    calls = []

    def send(request, **kwargs):
        calls.append((str(request.url), kwargs))
        if len(calls) == 1:
            response = httpx.Response(
                302, headers={"location": "/article"}, request=request,
            )
            response.raise_for_status()
        return httpx.Response(
            200, text="Article", headers={"content-type": "text/plain"}, request=request,
        )

    monkeypatch.setattr("resagent2_components.web.send_request", send)
    page = WebPageFetcher().fetch("https://example.test/start")
    assert page.final_url == "https://example.test/article"
    assert [url for url, _ in calls] == ["https://example.test/start", "https://example.test/article"]
    assert all(kwargs["public_only"] for _, kwargs in calls)
    assert calls[1][1]["timeout"] <= calls[0][1]["timeout"]


def test_web_page_fetcher_bounds_redirects(monkeypatch):
    calls = []

    def send(request, **kwargs):
        calls.append(request)
        response = httpx.Response(302, headers={"location": "/next"}, request=request)
        response.raise_for_status()

    monkeypatch.setattr("resagent2_components.web.send_request", send)
    with pytest.raises(WebFetchError) as raised:
        WebPageFetcher().fetch("https://example.test/start")
    assert raised.value.error_type == "redirect_limit"
    assert len(calls) == 6


def test_web_page_fetcher_normalizes_provider_invalid_url(monkeypatch):
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: (_ for _ in ()).throw(httpx.InvalidURL("bad url")),
    )
    with pytest.raises(WebFetchError) as raised:
        WebPageFetcher().fetch("https://example.test/page")
    assert raised.value.error_type == "invalid_url"


@pytest.mark.parametrize("content_type, content, error_type", [
    ("application/pdf", b"%PDF-1.7", "unsupported_content_type"),
    ("text/plain", b"\x00binary", "invalid_encoding"),
])
def test_web_page_fetcher_rejects_unsupported_or_nontext(monkeypatch, content_type, content, error_type):
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            content, content_type=content_type, url="https://example.test/file",
        ),
    )
    with pytest.raises(WebFetchError) as raised:
        WebPageFetcher().fetch("https://example.test/file")
    assert raised.value.error_type == error_type


def test_web_page_link_card_blocks_and_nested_anchors_keep_separate_labels(monkeypatch):
    html = b"""<a href='/paper'><div>Paper title</div><div>Download PDF</div></a>
        <a href='/first'>First<a href='/second'>Second</a>After"""
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(html, content_type="text/html", url="https://example.test/page"),
    )
    page = WebPageFetcher().fetch("https://example.test/page")
    assert page.text == (
        "[Paper title\n\nDownload PDF](https://example.test/paper)"
        "\n[First[Second](https://example.test/second)After](https://example.test/first)"
    )
    assert "https://example.test/paper" in page.text
    assert "https://example.test/first" in page.text
    assert "https://example.test/second" in page.text


@pytest.mark.parametrize("html", [
    "<p>Before</p><script>unclosed",
    "<p>Before</p><style>unclosed",
    "<p>Before</p><noscript>unclosed",
    "<p>Before</p><template>unclosed",
    "<p>Before</p><title>unclosed",
])
def test_web_page_fetcher_rejects_unterminated_content_elements(monkeypatch, html):
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html.encode(), content_type="text/html", url="https://example.test/page",
        ),
    )
    with pytest.raises(WebFetchError) as raised:
        WebPageFetcher().fetch("https://example.test/page")
    assert raised.value.error_type == "parse_failed"


@pytest.mark.parametrize("tag", ["script", "style", "noscript", "template", "title"])
def test_web_page_fetcher_accepts_closed_noncontent_elements(monkeypatch, tag):
    html = f"<p>Before</p><{tag}>Noncontent</{tag}><p>After</p>"
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html.encode(), content_type="text/html", url="https://example.test/page",
        ),
    )
    assert WebPageFetcher().fetch("https://example.test/page").text == "Before\n\nAfter"


def test_web_page_fetcher_keeps_escaped_script_as_visible_text(monkeypatch):
    html = b"<p>Before</p><pre>&lt;script&gt;unclosed</pre><p>After</p>"
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html, content_type="text/html", url="https://example.test/page",
        ),
    )
    page = WebPageFetcher().fetch("https://example.test/page")
    assert page.text == "Before\n\n```\n<script>unclosed\n```\n\nAfter"


@pytest.mark.parametrize("tag", ["noscript", "template"])
def test_web_page_fetcher_ignores_titles_inside_noncontent(monkeypatch, tag):
    html = f"<title>Page title</title><{tag}><title>Hidden title</title></{tag}><p>Body</p>"
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html.encode(), content_type="text/html", url="https://example.test/page",
        ),
    )
    page = WebPageFetcher().fetch("https://example.test/page")
    assert page.title == "Page title"
    assert page.text == "Body"


def test_web_page_fetcher_keeps_image_labels_without_image_targets(monkeypatch):
    html = b'<p>Before</p><img src="/figure.png" alt="Measurement curve">' \
           b'<p><a href="/paper"><img src="javascript:bad()" alt="Original paper"></a></p>'
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html, content_type="text/html", url="https://example.test/page",
        ),
    )
    page = WebPageFetcher().fetch("https://example.test/page")
    assert page.text == "Before\n\nMeasurement curve\n\n[Original paper](https://example.test/paper)"


def test_web_page_fetcher_ignores_foreign_document_titles(monkeypatch):
    html = b"<svg><title>Icon title</title><svg/></svg><math><title>Math title</title></math>" \
           b"<title>Page title</title><p>Body</p>"
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html, content_type="text/html", url="https://example.test/page",
        ),
    )
    page = WebPageFetcher().fetch("https://example.test/page")
    assert page.title == "Page title"
    assert page.text == "Body"


def test_web_page_fetcher_drops_control_character_link_targets(monkeypatch):
    html = b'<p>Before</p><a href="https://example.test/\x01bad">Visible label</a><p>After</p>'
    monkeypatch.setattr(
        "resagent2_components.web.send_request",
        lambda request, **kwargs: _response(
            html, content_type="text/html", url="https://example.test/page",
        ),
    )
    page = WebPageFetcher().fetch("https://example.test/page")
    assert page.text == "Before\n\nVisible label\n\nAfter"


def test_corrupt_compressed_webpage_is_a_structured_failure(monkeypatch):
    from resagent2_runtime import http

    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-encoding": "gzip", "content-type": "text/html"},
            stream=httpx.ByteStream(b"PRIVATE_CORRUPT_RESPONSE"),
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(http.httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    with pytest.raises(WebFetchError, match="could not be decoded") as raised:
        WebPageFetcher().fetch("https://example.test/page")
    assert raised.value.error_type == "invalid_response"
    assert "PRIVATE_CORRUPT_RESPONSE" not in str(raised.value)
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", ["budget", "deadline"])
def test_web_page_fetcher_does_not_convert_run_limits(monkeypatch, error_type):
    from resagent2_runtime.budget import BudgetExhaustedError, DeadlineExceededError

    error = BudgetExhaustedError("exhausted") if error_type == "budget" else DeadlineExceededError("expired")

    def send(request, **kwargs):
        raise error

    monkeypatch.setattr("resagent2_components.web.send_request", send)
    with pytest.raises(type(error)) as raised:
        WebPageFetcher().fetch("https://example.test/page")
    assert raised.value is error
