"""CLI search selection is explicit while page fetching stays independent."""

import pytest

from resagent2_cli import composition
from resagent2_components import DeepSeekWebSearchBackend, WebPageFetcher
from resagent2_runtime import ServiceRequestClient


@pytest.fixture(autouse=True)
def configured_clients(monkeypatch):
    monkeypatch.setattr(composition, "_client", lambda: object())
    for name in (
        "DEEPSEEK_API_KEY", "RESAGENT2_WEB_SEARCH_PROVIDER",
        "RESAGENT2_WEB_SEARCH_MODEL", "RESAGENT2_WEB_SEARCH_TIMEOUT_SECONDS",
        "RESAGENT2_WEB_TIMEOUT_SECONDS", "RESAGENT2_WEB_MAX_RESPONSE_BYTES",
        "RESAGENT2_LLM_TRACE_LEVEL", "RESAGENT2_LLM_TRACE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def scientific_application(tmp_path):
    return composition.build_application(data_root=tmp_path / "data").controller.scientific_port


def test_default_search_without_deepseek_key_is_unavailable(tmp_path):
    scientific = scientific_application(tmp_path)

    assert scientific.web_search_backend is None
    assert isinstance(scientific.web_page_fetcher, WebPageFetcher)
    assert scientific.web_page_fetcher.timeout_seconds == 30
    assert scientific.web_page_fetcher.max_response_bytes == 4 * 1024 * 1024


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_reuses_deepseek_key_for_official_search_api(tmp_path, monkeypatch, explicit):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    if explicit:
        monkeypatch.setenv("RESAGENT2_WEB_SEARCH_PROVIDER", "deepseek")
    backend = scientific_application(tmp_path).web_search_backend

    assert isinstance(backend, DeepSeekWebSearchBackend)
    assert backend.count == 10
    assert isinstance(backend.client, ServiceRequestClient)
    assert backend.client.endpoint == "https://api.deepseek.com/v1/tools/search"
    assert backend.client.api_key_env == "DEEPSEEK_API_KEY"
    assert backend.client.timeout_seconds == 60
    assert backend.client.max_response_bytes == 4 * 1024 * 1024


def test_cli_search_bounds_are_independent_from_agent_model_and_fetch(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    monkeypatch.setenv("RESAGENT2_MODEL", "other-agent-model")
    monkeypatch.setenv("RESAGENT2_WEB_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("RESAGENT2_WEB_SEARCH_TIMEOUT_SECONDS", "11")
    monkeypatch.setenv("RESAGENT2_WEB_MAX_RESPONSE_BYTES", "1234")
    scientific = scientific_application(tmp_path)

    assert scientific.web_search_backend.count == 10
    assert scientific.web_search_backend.client.endpoint == "https://api.deepseek.com/v1/tools/search"
    assert scientific.web_search_backend.client.timeout_seconds == 11
    assert scientific.web_page_fetcher.timeout_seconds == 7
    assert scientific.web_search_backend.client.max_response_bytes == 1234
    assert scientific.web_page_fetcher.max_response_bytes == 1234


def test_cli_off_disables_search_even_when_deepseek_key_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("RESAGENT2_WEB_SEARCH_PROVIDER", "off")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    scientific = scientific_application(tmp_path)

    assert scientific.web_search_backend is None
    assert isinstance(scientific.web_page_fetcher, WebPageFetcher)


@pytest.mark.parametrize("key", [None, "", "   "])
def test_explicit_deepseek_requires_its_key_before_io_or_client_creation(
    tmp_path, monkeypatch, key,
):
    monkeypatch.setenv("RESAGENT2_WEB_SEARCH_PROVIDER", "deepseek")
    if key is not None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", key)
    monkeypatch.setattr(composition, "_client", lambda: pytest.fail("unexpected client creation"))
    monkeypatch.setattr(composition, "ServiceRequestClient", lambda **kwargs: pytest.fail("unexpected search client"))
    root = tmp_path / "data"

    with pytest.raises(ValueError, match="requires DEEPSEEK_API_KEY"):
        composition.build_application(data_root=root)
    assert not root.exists()


@pytest.mark.parametrize("provider", ["tavily", "other", "", "DeepSeek"])
def test_unknown_search_provider_fails_before_io_or_client_creation(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("RESAGENT2_WEB_SEARCH_PROVIDER", provider)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    monkeypatch.setattr(composition, "_client", lambda: pytest.fail("unexpected client creation"))
    monkeypatch.setattr(composition, "ServiceRequestClient", lambda **kwargs: pytest.fail("unexpected search client"))
    root = tmp_path / "data"

    with pytest.raises(ValueError, match="RESAGENT2_WEB_SEARCH_PROVIDER must be deepseek or off"):
        composition.build_application(data_root=root)
    assert not root.exists()


@pytest.mark.parametrize("name", [
    "RESAGENT2_WEB_TIMEOUT_SECONDS", "RESAGENT2_WEB_MAX_RESPONSE_BYTES",
    "RESAGENT2_WEB_SEARCH_TIMEOUT_SECONDS",
])
@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5", ""])
def test_invalid_web_bounds_fail_before_io_or_client_creation(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(composition, "_client", lambda: pytest.fail("unexpected client creation"))
    monkeypatch.setattr(composition, "ServiceRequestClient", lambda **kwargs: pytest.fail("unexpected search client"))
    root = tmp_path / "data"
    with pytest.raises(ValueError, match=f"{name} must be"):
        composition.build_application(data_root=root)
    assert not root.exists()


@pytest.mark.parametrize("value", ["deepseek-flash", ""])
def test_removed_search_model_setting_fails_before_io(tmp_path, monkeypatch, value):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    monkeypatch.setenv("RESAGENT2_WEB_SEARCH_MODEL", value)
    monkeypatch.setattr(composition, "_client", lambda: pytest.fail("unexpected client creation"))
    root = tmp_path / "data"

    with pytest.raises(ValueError, match="RESAGENT2_WEB_SEARCH_MODEL is unsupported"):
        composition.build_application(data_root=root)
    assert not root.exists()


@pytest.mark.parametrize("provider", [None, "off"])
def test_disabled_search_does_not_require_removing_unused_model_setting(
    tmp_path, monkeypatch, provider,
):
    monkeypatch.setenv("RESAGENT2_WEB_SEARCH_MODEL", "old-unused-setting")
    if provider is not None:
        monkeypatch.setenv("RESAGENT2_WEB_SEARCH_PROVIDER", provider)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")

    scientific = scientific_application(tmp_path)
    assert scientific.web_search_backend is None
    assert isinstance(scientific.web_page_fetcher, WebPageFetcher)


def test_cli_search_service_trace_uses_existing_trace_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    monkeypatch.setenv("RESAGENT2_LLM_TRACE_LEVEL", "metadata")
    monkeypatch.setenv("RESAGENT2_LLM_TRACE_DIR", str(tmp_path / "traces"))

    client = scientific_application(tmp_path).web_search_backend.client
    assert client.trace_level == "metadata"
    assert client.trace_dir == (tmp_path / "traces").resolve()
