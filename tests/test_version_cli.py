"""Tests for the `comfy-gen version` CLI verb.

Bead remote_comfy_generator-bmq.2 (A.7.6): dispatches a `health` job to the
endpoint and surfaces the worker's version as {ok, worker_version} for
BlockFlow's semver gate.
"""

from __future__ import annotations

import json

import pytest

from comfy_gen import config as _config
from comfy_gen import version_check


@pytest.fixture
def mocked(monkeypatch):
    """Patch config + the urllib + poller boundaries."""
    state: dict = {"sent": [], "health_response": {"version": "0.2.0"}}

    monkeypatch.setattr(
        _config, "load",
        lambda: {"runpod_api_key": "rpa_x", "endpoint_id": "cfg-ep"},
    )

    class _Resp:
        def __init__(self, body: bytes):
            self._body = body
        def read(self) -> bytes:
            return self._body

    def fake_urlopen(req):
        state["sent"].append({
            "url": req.full_url,
            "headers": dict(req.headers),
            "body": json.loads(req.data.decode()),
        })
        return _Resp(json.dumps({"id": "job-123"}).encode())

    monkeypatch.setattr(version_check.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        version_check.poller, "poll_job",
        lambda **kw: state["health_response"],
    )
    return state


def test_dispatches_health_command_and_reshapes_response(mocked):
    result = version_check.submit_version(endpoint_id="ep-1")
    assert result == {"ok": True, "worker_version": "0.2.0"}
    assert mocked["sent"][0]["url"].endswith("/v2/ep-1/run")
    assert mocked["sent"][0]["body"] == {"input": {"command": "health"}}


def test_uses_configured_endpoint_when_arg_omitted(mocked):
    version_check.submit_version()
    assert mocked["sent"][0]["url"].endswith("/v2/cfg-ep/run")


def test_missing_version_in_response_raises(mocked):
    mocked["health_response"] = {"ok": True}  # no version field
    with pytest.raises(RuntimeError, match="missing version"):
        version_check.submit_version(endpoint_id="ep-1")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(
        _config, "load",
        lambda: {"runpod_api_key": "", "endpoint_id": "ep-1"},
    )
    with pytest.raises(ValueError, match="API key"):
        version_check.submit_version(endpoint_id="ep-1")


def test_missing_endpoint_raises(monkeypatch):
    monkeypatch.setattr(
        _config, "load",
        lambda: {"runpod_api_key": "rpa_x", "endpoint_id": ""},
    )
    with pytest.raises(ValueError, match="endpoint"):
        version_check.submit_version()
