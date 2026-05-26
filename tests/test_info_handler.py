"""Tests for the worker-level /info (query_info) command.

Bead remote_comfy_generator-bmq.6 / B.3.1: the response must include
`volume_root` so BlockFlow can build model paths from the worker's actual
mount point rather than hardcoding /runpod-volume/ComfyUI/models/loras.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def stubbed_handler(monkeypatch):
    import info_handler
    import delete_handler
    import list_handler

    monkeypatch.setattr(info_handler, "_get_object_info", lambda: {
        "KSampler": {"input": {"required": {
            "sampler_name": [["euler", "euler_ancestral"]],
            "scheduler": [["normal", "karras"]],
        }}}
    })
    monkeypatch.setattr(list_handler, "handle",
                        lambda job: {"ok": True, "files": [{"filename": "x.safetensors"}]})
    monkeypatch.setattr(delete_handler, "VOLUME_ROOT", "/runpod-volume")
    return info_handler


def test_info_response_includes_volume_root(stubbed_handler):
    result = stubbed_handler.handle({"input": {"command": "query_info"}})
    assert result["ok"] is True
    assert result["volume_root"] == "/runpod-volume"


def test_volume_root_tracks_delete_handler_constant(stubbed_handler, monkeypatch):
    """If ops mounts the volume somewhere else, info_handler should report
    whatever delete_handler.VOLUME_ROOT says. Single source of truth gate."""
    import delete_handler
    monkeypatch.setattr(delete_handler, "VOLUME_ROOT", "/mnt/custom-volume")
    result = stubbed_handler.handle({"input": {"command": "query_info"}})
    assert result["volume_root"] == "/mnt/custom-volume"


def test_info_still_returns_samplers_schedulers_loras(stubbed_handler):
    result = stubbed_handler.handle({"input": {"command": "query_info"}})
    assert "euler" in result["samplers"]
    assert "karras" in result["schedulers"]
    assert result["loras"][0]["filename"] == "x.safetensors"
