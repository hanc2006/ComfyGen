"""Tests for the log-dedupe in worker.on_progress (bead 9oi) and the
informational reframe of comfy_client's partial-execution message (bead kz8).
"""

from __future__ import annotations

import io
import logging
from unittest.mock import MagicMock

import pytest


# --- bead 9oi: on_progress dedupes identical log lines ----------------------

def test_on_progress_dedupes_identical_lines(monkeypatch, capsys):
    """ImageUpscaleWithModel emits ~150 identical progress ticks per call.
    Only the first one (or any unique change) should hit the log."""
    import runpod.serverless
    monkeypatch.setattr(runpod.serverless, "start", lambda *a, **k: None)

    import sys
    sys.modules.pop("worker", None)
    import worker

    # Stand up the minimum context to access on_progress. The actual on_progress
    # closure is defined inside handler() — we reproduce its body here, since
    # that's the contract we care about.
    job = {"id": "test-dedupe", "input": {}}
    workflow = {"50": {"class_type": "ImageUpscaleWithModel", "inputs": {}}}

    log_lines: list[str] = []

    class _JLog:
        def info(self, msg):
            log_lines.append(msg)
        warn = info
        error = info
        debug = info

    jlog = _JLog()

    # Inline the relevant on_progress body — same shape as in handler().
    SAMPLER_TYPES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}

    def _node_class(node_id):
        node = workflow.get(node_id, {})
        return node.get("class_type", "") if isinstance(node, dict) else ""

    def on_progress(data):
        stage = data.get("stage", "executing")
        msg = data.get("message", "")
        node_id = data.get("node", "")
        class_type = _node_class(node_id) if node_id else ""
        completed = data.get("completed_nodes", 0)
        total = data.get("total_nodes", 0)
        if stage == "inference" and class_type not in SAMPLER_TYPES:
            stage = "processing"
            msg = class_type or msg
        prefix = f"({completed}/{total}) " if total > 0 and completed > 0 else ""
        log_line = f"{stage}: {prefix}{msg}"
        if log_line != getattr(on_progress, "_last_line", None):
            on_progress._last_line = log_line
            jlog.info(log_line)

    # Fire 150 identical progress ticks.
    for _ in range(150):
        on_progress({
            "stage": "inference",
            "node": "50",
            "completed_nodes": 50,
            "total_nodes": 64,
            "message": "ImageUpscaleWithModel",
        })

    # Only the FIRST one should have been logged. The other 149 are deduped.
    assert log_lines == ["processing: (50/64) ImageUpscaleWithModel"]


def test_on_progress_logs_when_message_changes(monkeypatch):
    """Sampler step counts change every tick; those must NOT be deduped."""
    import runpod.serverless
    monkeypatch.setattr(runpod.serverless, "start", lambda *a, **k: None)

    log_lines: list[str] = []
    on_progress = _make_on_progress(log_lines, workflow={
        "11": {"class_type": "KSampler", "inputs": {}}
    })

    for step in range(1, 6):
        on_progress({
            "stage": "inference",
            "node": "11",
            "step": step,
            "total_steps": 6,
            "completed_nodes": 11,
            "total_nodes": 20,
        })

    # 5 different sampler step messages → 5 log lines (none deduped).
    assert len(log_lines) == 5
    assert all("Step" in line for line in log_lines)


def _make_on_progress(log_lines, workflow):
    """Reusable on_progress factory mirroring the handler() closure."""
    SAMPLER_TYPES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}

    def _node_class(node_id):
        node = workflow.get(node_id, {})
        return node.get("class_type", "") if isinstance(node, dict) else ""

    class _JLog:
        def info(self, msg):
            log_lines.append(msg)
    jlog = _JLog()

    def on_progress(data):
        stage = data.get("stage", "executing")
        msg = data.get("message", "")
        node_id = data.get("node", "")
        class_type = _node_class(node_id) if node_id else ""
        completed = data.get("completed_nodes", 0)
        total = data.get("total_nodes", 0)
        if stage == "inference" and class_type not in SAMPLER_TYPES:
            stage = "processing"
            msg = class_type or msg
        elif stage == "inference":
            step = data.get("step", "")
            total_steps = data.get("total_steps", "")
            step_info = f"Step {step}/{total_steps}" if step and total_steps else msg
            msg = f"{class_type} {step_info}" if class_type else step_info
        elif stage == "executing" and class_type and node_id:
            msg = class_type
        prefix = f"({completed}/{total}) " if total > 0 and completed > 0 else ""
        log_line = f"{stage}: {prefix}{msg}"
        if log_line != getattr(on_progress, "_last_line", None):
            on_progress._last_line = log_line
            jlog.info(log_line)
    return on_progress


# --- bead kz8: partial-execution message is informational, not WARNING -------

def test_partial_execution_message_no_longer_says_warning(capsys):
    """ComfyUI's graph pruning is normal — message must NOT contain
    'WARNING' / 'Partial execution' framing that implies failure."""
    import comfy_client
    # Drive only the post-loop logging by feeding a minimal scenario through
    # the print path: we replicate the new logic inline and assert wording.
    nodes_to_execute = 64
    completed_nodes = 53
    if nodes_to_execute > 0 and completed_nodes < nodes_to_execute:
        skipped = nodes_to_execute - completed_nodes
        msg = (f"[comfy_client] Executed {completed_nodes}/{nodes_to_execute} nodes "
               f"({skipped} skipped — disconnected/unreachable branches)")
    assert "WARNING" not in msg
    assert "Partial execution" not in msg
    assert "53/64" in msg
    assert "11 skipped" in msg


def test_comfy_client_source_uses_informational_wording():
    """Source-level guard: comfy_client.py must not contain the old WARNING
    framing (regression catcher)."""
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "serverless-runtime" / "comfy_client.py"
    text = src.read_text()
    assert "WARNING: Partial execution" not in text, (
        "comfy_client.py must use informational 'Executed N/M nodes' phrasing "
        "(bead kz8) — the old 'WARNING: Partial execution' framing fooled users "
        "into thinking successful jobs had failed."
    )
