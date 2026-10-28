import pytest


@pytest.fixture(autouse=True)
def isolate_default_artifact_roots(tmp_path, monkeypatch):
    """Persistent default outputs belong to this test, including reruns."""
    monkeypatch.chdir(tmp_path)
