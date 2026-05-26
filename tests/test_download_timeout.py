"""End-to-end test that `--timeout` plumbs from CLI to worker subprocess.

Bead remote_comfy_generator-bmq.4 (A.6.3). BlockFlow computes
`300 + size_gb * 60` and passes via --timeout; the value flows:

    cli.py --timeout
      -> download.submit_download(timeout=...)
        -> payload["input"]["timeout_sec"]   (asserted here)
        -> worker download_handler reads job_input["timeout_sec"]
          -> _download_civitai / _download_url subprocess timeout
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from comfy_gen import download as download_mod


@pytest.fixture
def captured(monkeypatch):
    """Capture the payload submit_download POSTs to RunPod."""
    state = {"payload": None}

    from comfy_gen import config
    monkeypatch.setattr(config, "load", _ConfigStub.load)

    class _Resp:
        def __init__(self, body): self._body = body
        def read(self): return self._body

    def fake_urlopen(req):
        state["payload"] = json.loads(req.data.decode())
        return _Resp(json.dumps({"id": "job-1"}).encode())

    monkeypatch.setattr(download_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        download_mod.poller, "poll_job",
        lambda **kw: {"ok": True, "files": [], "elapsed_seconds": 0},
    )
    return state


class _ConfigStub:
    @staticmethod
    def load():
        return {
            "runpod_api_key": "rpa_x",
            "endpoint_id": "ep-1",
            "civitai_token": "t",
        }


def test_timeout_sec_in_payload(captured):
    download_mod.submit_download(
        downloads=[{"source": "url", "url": "https://x/y.bin", "dest": "loras"}],
        timeout=7200,
    )
    assert captured["payload"]["input"]["timeout_sec"] == 7200


def test_default_timeout_is_600(captured):
    download_mod.submit_download(
        downloads=[{"source": "url", "url": "https://x/y.bin", "dest": "loras"}],
    )
    assert captured["payload"]["input"]["timeout_sec"] == 600


def test_worker_handle_reads_timeout_sec_and_passes_to_subprocess(monkeypatch, tmp_path):
    """Worker side: timeout_sec on job_input flows into _download_url's
    subprocess timeout. We patch _download_url and assert the kwarg arrives.
    """
    import sys
    sys.path.insert(0, str(tmp_path.parent.parent / "serverless-runtime"))
    import download_handler

    seen = {}

    def fake_download_url(url, dest_dir, filename=None, **kwargs):
        seen.update(kwargs)
        out = tmp_path / "out.bin"
        out.write_bytes(b"x")
        return {"filename": "out.bin", "path": str(out), "size_mb": 0.0}

    monkeypatch.setattr(download_handler, "_download_url", fake_download_url)
    monkeypatch.setattr(download_handler, "MODELS_BASE", str(tmp_path))

    download_handler.handle({
        "id": "j-1",
        "input": {
            "command": "download",
            "timeout_sec": 4200,
            "downloads": [
                {"source": "url", "url": "https://x/out.bin", "dest": "loras"},
            ],
        },
    })

    assert seen.get("timeout_sec") == 4200


def test_worker_handle_clamps_to_min_600(monkeypatch, tmp_path):
    """Sub-600 timeout_sec is treated as 600 — a safety floor for legacy jobs
    or unintended small values that would kill aria2c mid-download."""
    import sys
    sys.path.insert(0, str(tmp_path.parent.parent / "serverless-runtime"))
    import download_handler

    seen = {}
    def fake_download_url(url, dest_dir, filename=None, **kwargs):
        seen.update(kwargs)
        out = tmp_path / "x.bin"
        out.write_bytes(b"x")
        return {"filename": "x.bin", "path": str(out), "size_mb": 0.0}

    monkeypatch.setattr(download_handler, "_download_url", fake_download_url)
    monkeypatch.setattr(download_handler, "MODELS_BASE", str(tmp_path))

    download_handler.handle({
        "id": "j-1",
        "input": {
            "command": "download",
            "timeout_sec": 30,  # absurdly small
            "downloads": [
                {"source": "url", "url": "https://x/out.bin", "dest": "loras"},
            ],
        },
    })

    assert seen["timeout_sec"] == 600


def test_worker_handle_missing_timeout_sec_defaults_to_600(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, str(tmp_path.parent.parent / "serverless-runtime"))
    import download_handler

    seen = {}
    def fake_download_url(url, dest_dir, filename=None, **kwargs):
        seen.update(kwargs)
        out = tmp_path / "x.bin"
        out.write_bytes(b"x")
        return {"filename": "x.bin", "path": str(out), "size_mb": 0.0}

    monkeypatch.setattr(download_handler, "_download_url", fake_download_url)
    monkeypatch.setattr(download_handler, "MODELS_BASE", str(tmp_path))

    download_handler.handle({
        "id": "j-1",
        "input": {
            "command": "download",
            # no timeout_sec
            "downloads": [
                {"source": "url", "url": "https://x/out.bin", "dest": "loras"},
            ],
        },
    })

    assert seen["timeout_sec"] == 600
