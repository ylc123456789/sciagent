"""Independent Search requests keep native fields and expose safe failures."""

import pytest

from resagent2_components import DeepSeekWebSearchBackend, WebSearchError
from resagent2_runtime.budget import BudgetExhaustedError, DeadlineExceededError
from resagent2_runtime.cancellation import OperationCancelled
from resagent2_runtime.service_request import ServiceRequestError, ServiceRequestHTTPError


URL = "https://example.test/source"


class Client:
    def __init__(self, payload=None, *, error=None):
        self.payload = {"results": []} if payload is None else payload
        self.error = error
        self.requests = []
        self.trace_context = {}

    def request(self, body):
        self.requests.append(body)
        if self.error:
            raise self.error
        return self.payload

    def set_trace_context(self, **kwargs):
        self.trace_context = kwargs


def source(url=URL, **kwargs):
    return {
        "title": "Source", "url": url, "snippet": "Short query-related excerpt",
        "summary": "Provider-generated long summary", "content": "",
        "rerank_score": 2.4, "date": "2026-10-01T00:00:00Z",
        "authority": 3, "site_name": "Example", "site_icon": None, **kwargs,
    }


def test_search_uses_direct_request_and_preserves_distinct_native_fields():
    client = Client({"results": [source(content="Not requested or returned to the Agent")]})
    backend = DeepSeekWebSearchBackend(client)
    backend.set_trace_context(run_id="run_1", tool="web_search")
    result = backend.search("  a query  ")

    assert client.requests == [{"query": "a query", "count": 10, "need_content": False}]
    assert client.trace_context == {"provider": "deepseek", "run_id": "run_1", "tool": "web_search"}
    assert result.provider == "deepseek" and result.query == "a query"
    assert result.incomplete_reason is None
    item, = result.results
    assert item.title == "Source" and item.url == URL
    assert item.snippet == "Short query-related excerpt"
    assert item.summary == "Provider-generated long summary"
    assert not item.snippet_truncated and not item.summary_truncated
    assert item.score == 2.4 and item.authority == 3
    assert item.site_name == "Example" and item.published_at == "2026-10-01T00:00:00Z"
    assert not hasattr(item, "content")


def test_snippet_and_summary_are_bounded_independently():
    client = Client({"results": [
        source(snippet="x" * 2001, summary="s" * 2000),
        source(URL + "/next", snippet="", summary="y" * 2001),
    ]})
    first, second = DeepSeekWebSearchBackend(client).search("q").results
    assert first.snippet == "x" * 2000 and first.snippet_truncated
    assert first.summary == "s" * 2000 and not first.summary_truncated
    assert second.snippet == "" and not second.snippet_truncated
    assert second.summary == "y" * 2000 and second.summary_truncated


def test_dedup_is_exact_and_does_not_filter_or_reorder_provider_scores():
    client = Client({"results": [
        source(rerank_score=0), source(title="Duplicate", rerank_score=3),
        source(URL + "/", rerank_score=3), source(URL + "#section", rerank_score=1),
    ]})
    result = DeepSeekWebSearchBackend(client).search("q")
    assert [item.url for item in result.results] == [URL, URL + "/", URL + "#section"]
    assert [item.score for item in result.results] == [0, 3, 1]
    assert result.results[0].title == "Source"


@pytest.mark.parametrize("title", ["", " \t "])
def test_blank_title_preserves_the_result_and_other_valid_candidates(title):
    # The official schema permits empty titles; one must not discard the batch.
    client = Client({"results": [
        source(), source(URL + "/untitled", title=title, snippet="Untitled page excerpt"),
    ]})
    result = DeepSeekWebSearchBackend(client).search("q")
    assert [item.url for item in result.results] == [URL, URL + "/untitled"]
    assert [item.title for item in result.results] == ["Source", ""]
    assert result.results[1].snippet == "Untitled page excerpt"


def test_explicit_empty_results_are_successful_empty():
    result = DeepSeekWebSearchBackend(Client()).search("q")
    assert result.results == [] and result.incomplete_reason is None


@pytest.mark.parametrize("score,authority", [(0, -1), (3, 4), (1.5, 0)])
def test_valid_boundaries_and_nullable_metadata(score, authority):
    client = Client({"results": [source(
        rerank_score=score, authority=authority, site_name=None, site_icon=None, date=None,
        snippet="", summary="",
    )]})
    item, = DeepSeekWebSearchBackend(client).search("q").results
    assert item.score == score and item.authority == authority
    assert item.site_name is None and item.published_at is None


@pytest.mark.parametrize("payload", [
    {}, {"results": None}, {"results": {}}, {"results": "PRIVATE"},
    {"results": [None]}, {"results": [[]]},
    {"content": [{"type": "text", "text": URL}]},
])
def test_malformed_response_is_not_successful_empty(payload):
    with pytest.raises(WebSearchError) as raised:
        DeepSeekWebSearchBackend(Client(payload)).search("q")
    assert raised.value.error_type == "invalid_response"
    assert "PRIVATE" not in str(raised.value)


@pytest.mark.parametrize("field", list(source()))
def test_missing_required_field_is_rejected(field):
    item = source()
    del item[field]
    with pytest.raises(WebSearchError) as raised:
        DeepSeekWebSearchBackend(Client({"results": [item]})).search("q")
    assert raised.value.error_type == "invalid_response"


@pytest.mark.parametrize("update", [
    {"title": None}, {"title": 1},
    {"url": None}, {"url": "javascript:alert(1)"},
    {"url": "https://user:PRIVATE@example.test/a"}, {"url": "https://example.test:bad/a"},
    {"url": "https://example.test:70000/a"}, {"url": "https://[broken/a"},
    {"url": "https:///missing"}, {"url": "https://exa mple.test/a"},
    {"url": "https://example.test/line\nbreak"},
    {"snippet": None}, {"snippet": []}, {"summary": None}, {"summary": {}},
    {"content": None}, {"content": False}, {"date": 2026},
    {"site_name": []}, {"site_icon": {}},
    {"rerank_score": None}, {"rerank_score": True}, {"rerank_score": "2.4"},
    {"rerank_score": -0.01}, {"rerank_score": 3.01},
    {"rerank_score": float("nan")}, {"rerank_score": float("inf")},
    {"rerank_score": 10 ** 1000},
    {"authority": None}, {"authority": True}, {"authority": 1.0},
    {"authority": -2}, {"authority": 5},
])
def test_invalid_result_fields_fail_without_reflecting_provider_text(update):
    with pytest.raises(WebSearchError) as raised:
        DeepSeekWebSearchBackend(Client({"results": [source(**update)]})).search("q")
    assert raised.value.error_type == "invalid_response"
    assert "PRIVATE" not in str(raised.value)


def test_malformed_duplicate_is_still_rejected():
    with pytest.raises(WebSearchError):
        DeepSeekWebSearchBackend(Client({"results": [
            source(), source(summary=None),
        ]})).search("q")


@pytest.mark.parametrize("query", ["a", "x" * 128, "\u7814" * 128, " " + "x" * 128 + " "])
def test_query_character_boundaries_preserve_text(query):
    client = Client()
    result = DeepSeekWebSearchBackend(client).search(query)
    assert result.query == query.strip()
    assert client.requests[0]["query"] == query.strip()


@pytest.mark.parametrize("query", ["", "   ", None, True, "x" * 129, "\u7814" * 129])
def test_invalid_query_does_not_dispatch_or_silently_truncate(query):
    client = Client()
    with pytest.raises(WebSearchError) as raised:
        DeepSeekWebSearchBackend(client).search(query)
    assert raised.value.error_type == "invalid_query" and client.requests == []


@pytest.mark.parametrize("count", [1, 50])
def test_count_boundaries_are_sent_to_provider(count):
    client = Client()
    DeepSeekWebSearchBackend(client, count=count).search("q")
    assert client.requests[0]["count"] == count


@pytest.mark.parametrize("count", [0, 51, True, 1.0, None, "10"])
def test_invalid_count_is_rejected_before_dispatch(count):
    client = Client()
    with pytest.raises(ValueError):
        DeepSeekWebSearchBackend(client, count=count)
    assert client.requests == []


@pytest.mark.parametrize("status,expected", [
    (400, "request_rejected"), (401, "request_rejected"), (402, "request_rejected"),
    (403, "request_rejected"), (429, "rate_limited"),
    (500, "provider_error"), (503, "provider_error"),
])
def test_http_errors_are_safe_and_never_retried(status, expected):
    error = ServiceRequestHTTPError(status, "4")
    error.args = ("PRIVATE provider error body",)
    client = Client(error=error)
    with pytest.raises(WebSearchError) as raised:
        DeepSeekWebSearchBackend(client).search("q")
    assert raised.value.error_type == expected
    assert raised.value.retry_after == 4 and len(client.requests) == 1
    assert "PRIVATE" not in str(raised.value)


@pytest.mark.parametrize("error_type", [
    "network_error", "timeout", "invalid_response", "response_too_large", "missing_credentials",
])
def test_safe_transport_error_preserves_classification_without_retry(error_type):
    client = Client(error=ServiceRequestError("safe failure", error_type=error_type))
    with pytest.raises(WebSearchError) as raised:
        DeepSeekWebSearchBackend(client).search("q")
    assert raised.value.error_type == error_type and len(client.requests) == 1


@pytest.mark.parametrize("error", [
    BudgetExhaustedError("exhausted"), DeadlineExceededError("expired"), OperationCancelled("cancelled"),
])
def test_run_limits_and_cancellation_propagate(error):
    client = Client(error=error)
    with pytest.raises(type(error)):
        DeepSeekWebSearchBackend(client).search("q")
    assert len(client.requests) == 1
