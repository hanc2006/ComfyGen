"""Contract tests for `error_code` field on install_error / preflight_fail events.

Bead remote_comfy_generator-bmq.3 (A.7.2). BlockFlow matches `error_code` to
decide retry vs surface. The codes are defined in `comfy_gen/_install_error_codes.py`;
both the orchestrator CLI and the worker (installer_server) emit them. Worker
duplicates the string literals because the runtime image doesn't ship
`comfy_gen` — this test enforces alignment.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from comfy_gen import _install_error_codes as ec
from comfy_gen import install_preset


# --- Spec sanity ------------------------------------------------------------


def test_all_codes_defined_and_lowercase_snake():
    assert ec.ALL_CODES, "must define at least one code"
    for code in ec.ALL_CODES:
        assert code == code.lower(), code
        assert re.fullmatch(r"[a-z_]+", code), code


def test_classify_download_exception_routes_sha_mismatch():
    exc = RuntimeError("Download 3: sha256 mismatch for x.safetensors: expected ..., got ...")
    assert ec.classify_download_exception(exc) == ec.SHA_MISMATCH


def test_classify_download_exception_falls_back_to_download_failed():
    exc = RuntimeError("aria2c failed: exit 8")
    assert ec.classify_download_exception(exc) == ec.DOWNLOAD_FAILED


# --- Orchestrator-side (comfy_gen/install_preset.py) ------------------------


@pytest.fixture
def orchestrator_mocks(monkeypatch):
    state = {"events": [], "stream_raises": None, "health_raises": None}

    def fake_spawn(api_key, image, volume_id, token, name="x", port=3000, **_):
        return {"id": "pod-abc"}

    def fake_health(pod_id, port, timeout_sec):
        if state["health_raises"]:
            raise RuntimeError(state["health_raises"])

    def fake_stream(pod_id, port, token, preset_id, civitai_token=None, hf_token=None):
        if state["stream_raises"]:
            raise RuntimeError(state["stream_raises"])
        yield from state["events"]

    monkeypatch.setattr(install_preset, "spawn_installer_pod", fake_spawn)
    monkeypatch.setattr(install_preset, "wait_for_health", fake_health)
    monkeypatch.setattr(install_preset, "stream_install", fake_stream)
    monkeypatch.setattr(install_preset, "shutdown_pod", lambda *a, **k: None)
    monkeypatch.setattr(install_preset, "delete_pod", lambda *a, **k: None)
    monkeypatch.setattr(install_preset.config, "load", lambda: {"runpod_api_key": "rpa_x"})
    return state


def _lines(buf):
    return [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]


def test_health_timeout_install_error_carries_health_timeout_code(orchestrator_mocks):
    orchestrator_mocks["health_raises"] = "not healthy"
    buf = io.StringIO()
    install_preset.run(preset_id="p", volume_id="vid", pod_id=None, token=None, out=buf)
    err = next(l for l in _lines(buf) if l["type"] == "install_error")
    assert err["error_code"] == ec.HEALTH_TIMEOUT
    assert err["error_code"] in ec.ALL_CODES
    assert err["reason"]  # reason kept for one release


def test_stream_error_install_error_carries_stream_error_code(orchestrator_mocks):
    orchestrator_mocks["stream_raises"] = "connection reset"
    buf = io.StringIO()
    install_preset.run(preset_id="p", volume_id="vid", pod_id=None, token=None, out=buf)
    err = next(l for l in _lines(buf) if l["type"] == "install_error")
    assert err["error_code"] == ec.STREAM_ERROR
    assert err["error_code"] in ec.ALL_CODES


# --- Worker-side (installer_server.py) — string-literal alignment -----------


def _read_worker_source() -> str:
    return (Path(__file__).resolve().parent.parent
            / "serverless-runtime" / "installer_server.py").read_text()


@pytest.mark.parametrize("code", [
    ec.PREFLIGHT_DISK_FULL,
    ec.PREFLIGHT_PRESET_NOT_FOUND,
    ec.DOWNLOAD_FAILED,
    ec.SHA_MISMATCH,
])
def test_worker_emits_known_error_codes(code):
    """Each worker-side code must appear as a string literal in installer_server.py.

    Drift gate — worker can't import _install_error_codes (no comfy_gen in the
    runtime image), so this test enforces the duplicated literals stay aligned.
    """
    src = _read_worker_source()
    assert f'"{code}"' in src or f"'{code}'" in src, (
        f"installer_server.py must emit error_code={code!r} (see "
        f"comfy_gen/_install_error_codes.py)"
    )
