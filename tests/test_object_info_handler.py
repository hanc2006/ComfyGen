"""Tests for the `object_info` worker command — workflow pre-flight introspection.

Exposes ComfyUI's /object_info to callers (smoke gate, BlockFlow installer)
so they can validate a workflow's class_types and inputs before submitting.
"""

from __future__ import annotations

from unittest.mock import patch


# A realistic-ish slice of ComfyUI's /object_info shape — enough to exercise
# the handler's filter logic without simulating the full 200+ class catalog.
FAKE_OBJECT_INFO = {
    "KSampler": {
        "input": {
            "required": {
                "seed": ["INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}],
                "sampler_name": [["euler", "dpmpp_2m"], {"default": "euler"}],
            },
        },
        "output": ["LATENT"],
    },
    "VAEDecode": {
        "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
        "output": ["IMAGE"],
    },
    "OnnxDetectionModelLoader": {
        "input": {
            "required": {
                "vitpose_model": [["vitpose_h_wholebody_data.bin"]],
                "yolo_model": [["vitpose_h_wholebody_data.bin"]],
                "onnx_device": [["CUDAExecutionProvider", "CPUExecutionProvider"]],
            },
        },
        "output": ["POSEMODEL"],
    },
}


def test_no_filter_returns_all_classes(dispatch_command):
    with patch("info_handler._get_object_info", return_value=FAKE_OBJECT_INFO):
        res = dispatch_command({"command": "object_info"})
    assert res == {"ok": True, "classes": FAKE_OBJECT_INFO}


def test_filter_returns_only_requested_classes(dispatch_command):
    with patch("info_handler._get_object_info", return_value=FAKE_OBJECT_INFO):
        res = dispatch_command({"command": "object_info", "class_types": ["KSampler"]})
    assert res["ok"] is True
    assert list(res["classes"].keys()) == ["KSampler"]
    assert res["classes"]["KSampler"] == FAKE_OBJECT_INFO["KSampler"]


def test_filter_with_unknown_class_returns_empty_for_that_class(dispatch_command):
    """Unknown class names in the filter are silently dropped; ok=true (batch ran)."""
    with patch("info_handler._get_object_info", return_value=FAKE_OBJECT_INFO):
        res = dispatch_command(
            {"command": "object_info", "class_types": ["KSampler", "DoesNotExist"]}
        )
    assert res["ok"] is True
    assert set(res["classes"].keys()) == {"KSampler"}


def test_upstream_comfyui_error_returns_ok_false(dispatch_command):
    import urllib.error

    err = urllib.error.URLError("connection refused")
    with patch("info_handler._get_object_info", side_effect=err):
        res = dispatch_command({"command": "object_info"})
    assert res["ok"] is False
    assert "connection refused" in res["error"]
