"""Tests for env-first token resolution in cmd_install_preset / cmd_install_call.

Bead remote_comfy_generator-bmq.1 (A.7.5): tokens come from
COMFY_GEN_CIVITAI_TOKEN / COMFY_GEN_HF_TOKEN env vars; --civitai-token /
--hf-token argv flags are deprecated (one-release fallback) and emit a stderr
warning when used. BlockFlow stops passing argv after this lands.
"""

from __future__ import annotations

import argparse

import pytest

from comfy_gen import cli


def _ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        preset_id="p", volume_id="vid", image="img", port=3000,
        health_timeout_sec=180, keep_alive=False,
        civitai_token=None, hf_token=None, runtime_repo_ref=None,
        pod_id="pid", token="t",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def captured_run(monkeypatch):
    """Capture the kwargs install_preset.run was called with."""
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    from comfy_gen import install_preset
    monkeypatch.setattr(install_preset, "run", fake_run)
    return captured


def test_env_var_used_when_set(captured_run, monkeypatch):
    monkeypatch.setenv("COMFY_GEN_CIVITAI_TOKEN", "env-civitai")
    monkeypatch.setenv("COMFY_GEN_HF_TOKEN", "env-hf")
    with pytest.raises(SystemExit) as exc:
        cli.cmd_install_preset(_ns())
    assert exc.value.code == 0
    assert captured_run["civitai_token"] == "env-civitai"
    assert captured_run["hf_token"] == "env-hf"


def test_argv_fallback_when_env_unset(captured_run, monkeypatch):
    monkeypatch.delenv("COMFY_GEN_CIVITAI_TOKEN", raising=False)
    monkeypatch.delenv("COMFY_GEN_HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        cli.cmd_install_preset(_ns(civitai_token="argv-civitai", hf_token="argv-hf"))
    assert captured_run["civitai_token"] == "argv-civitai"
    assert captured_run["hf_token"] == "argv-hf"


def test_env_wins_over_argv(captured_run, monkeypatch):
    monkeypatch.setenv("COMFY_GEN_CIVITAI_TOKEN", "env-civitai")
    monkeypatch.delenv("COMFY_GEN_HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        cli.cmd_install_preset(_ns(civitai_token="argv-civitai", hf_token="argv-hf"))
    assert captured_run["civitai_token"] == "env-civitai"
    # HF env unset -> argv fallback still works per-token, not all-or-nothing.
    assert captured_run["hf_token"] == "argv-hf"


def test_argv_use_emits_stderr_deprecation_warning(captured_run, monkeypatch, capsys):
    monkeypatch.delenv("COMFY_GEN_CIVITAI_TOKEN", raising=False)
    monkeypatch.delenv("COMFY_GEN_HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        cli.cmd_install_preset(_ns(civitai_token="argv-civitai"))
    err = capsys.readouterr().err
    assert "deprecated" in err.lower()
    assert "COMFY_GEN_CIVITAI_TOKEN" in err


def test_no_warning_when_argv_unused(captured_run, monkeypatch, capsys):
    monkeypatch.setenv("COMFY_GEN_CIVITAI_TOKEN", "env-civitai")
    with pytest.raises(SystemExit):
        cli.cmd_install_preset(_ns())
    err = capsys.readouterr().err
    assert "deprecated" not in err.lower()


def test_no_warning_when_neither_set(captured_run, monkeypatch, capsys):
    monkeypatch.delenv("COMFY_GEN_CIVITAI_TOKEN", raising=False)
    monkeypatch.delenv("COMFY_GEN_HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        cli.cmd_install_preset(_ns())
    err = capsys.readouterr().err
    assert "deprecated" not in err.lower()
    assert captured_run["civitai_token"] is None
    assert captured_run["hf_token"] is None


def test_install_call_applies_same_resolution(captured_run, monkeypatch, capsys):
    monkeypatch.setenv("COMFY_GEN_CIVITAI_TOKEN", "env-civitai")
    monkeypatch.delenv("COMFY_GEN_HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        cli.cmd_install_call(_ns(civitai_token="argv-civitai", hf_token="argv-hf"))
    assert captured_run["civitai_token"] == "env-civitai"
    assert captured_run["hf_token"] == "argv-hf"
    err = capsys.readouterr().err
    assert "deprecated" in err.lower()
