"""Tests for smoke_preset's schema helpers.

Covers the multi-workflow + smoke_inputs handling added in
remote_comfy_generator-kv9. The full smoke loop is integration-tested
live against a real RunPod endpoint (see automation/smoke_preset.py);
these are unit tests for the pure helpers only.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add automation/ to path so we can import smoke_preset's pure helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "automation"))

from smoke_preset import choose_workflow, fetch_smoke_inputs  # noqa: E402


def test_choose_workflow_picks_legacy_singular():
    preset = {"workflow": {"url": "https://x/wf.json", "sha256": "abc"}}
    assert choose_workflow(preset) == {"url": "https://x/wf.json", "sha256": "abc"}


def test_choose_workflow_picks_first_from_array():
    preset = {
        "workflows": [
            {"name": "Keep Background", "url": "https://x/kb.json", "sha256": "1"},
            {"name": "Replace Character", "url": "https://x/rc.json", "sha256": "2"},
        ]
    }
    chosen = choose_workflow(preset)
    assert chosen["name"] == "Keep Background"


def test_choose_workflow_empty_workflows_raises():
    with pytest.raises(RuntimeError, match="neither 'workflow' nor"):
        choose_workflow({"workflows": []})


def test_choose_workflow_missing_both_raises():
    with pytest.raises(RuntimeError):
        choose_workflow({"id": "x", "name": "y"})


def test_fetch_smoke_inputs_downloads_and_verifies(tmp_path, monkeypatch):
    """Each fixture is downloaded, sha256-verified, written, and returned."""
    payload = b"hello fixture bytes"
    sha = hashlib.sha256(payload).hexdigest()
    smoke_inputs = [
        {
            "node_id": "311",
            "field": "image",
            "url": "https://example.com/img.png",
            "sha256": sha,
            "filename": "img.png",
        }
    ]

    class FakeResp:
        def read(self_inner): return payload

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        result = fetch_smoke_inputs(smoke_inputs, preset_id="testp")

    assert len(result) == 1
    node_id, local_path = result[0]
    assert node_id == "311"
    assert local_path.endswith("img.png")
    assert Path(local_path).read_bytes() == payload


def test_fetch_smoke_inputs_sha_mismatch_raises():
    smoke_inputs = [
        {
            "node_id": "311",
            "field": "image",
            "url": "https://example.com/img.png",
            "sha256": "deadbeef" * 8,  # 64 chars but wrong
            "filename": "img.png",
        }
    ]

    class FakeResp:
        def read(self_inner): return b"some bytes"

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            fetch_smoke_inputs(smoke_inputs, preset_id="testp")


def test_fetch_smoke_inputs_multiple_fixtures_all_returned():
    inputs = [
        {"node_id": "311", "field": "image", "url": "u1",
         "sha256": hashlib.sha256(b"a").hexdigest(), "filename": "a.png"},
        {"node_id": "417", "field": "video", "url": "u2",
         "sha256": hashlib.sha256(b"bb").hexdigest(), "filename": "b.mp4"},
    ]

    class FakeResp:
        def __init__(self, data): self.data = data
        def read(self): return self.data

    fake_responses = [FakeResp(b"a"), FakeResp(b"bb")]
    with patch("urllib.request.urlopen", side_effect=fake_responses):
        result = fetch_smoke_inputs(inputs, preset_id="testp")

    assert len(result) == 2
    assert [n for n, _ in result] == ["311", "417"]
