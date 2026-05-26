"""Canonical `error_code` enum for `install_error` / `preflight_fail` SSE events.

Bead remote_comfy_generator-bmq.3 (A.7.2). BlockFlow matches on `error_code`
(machine-readable) instead of substring-matching `reason`. The `reason` string
stays alongside for one release to keep old BlockFlow builds working.

Codes are stable strings — adding new ones is fine; renaming/removing requires
a coordinated release with the BlockFlow side.

Both the orchestrator CLI (`comfy_gen/install_preset.py`) and the worker
(`serverless-runtime/installer_server.py`) emit these. The worker can't import
this module at runtime (it doesn't have the `comfy_gen` package installed in
its image), so the worker duplicates the string literals; this module is the
single source of truth that the contract test asserts against.
"""

from __future__ import annotations

# Spawn / scheduling
SUPPLY_CONSTRAINT = "supply_constraint"        # no CPU SKU had capacity

# Pod-lifecycle (orchestrator-side)
HEALTH_TIMEOUT = "health_timeout"              # /health never came up
STREAM_ERROR = "stream_error"                  # SSE stream broke mid-install

# Preflight (worker-side)
PREFLIGHT_DISK_FULL = "preflight_disk_full"    # not enough free bytes
PREFLIGHT_PRESET_NOT_FOUND = "preflight_preset_not_found"  # preset_id unknown

# Download (worker-side)
DOWNLOAD_FAILED = "download_failed"            # aria2c / civitai exited non-zero
SHA_MISMATCH = "sha_mismatch"                  # post-download sha256 mismatch

# Catchall
INTERNAL_ERROR = "internal_error"              # unexpected exception

ALL_CODES = frozenset({
    SUPPLY_CONSTRAINT,
    HEALTH_TIMEOUT,
    STREAM_ERROR,
    PREFLIGHT_DISK_FULL,
    PREFLIGHT_PRESET_NOT_FOUND,
    DOWNLOAD_FAILED,
    SHA_MISMATCH,
    INTERNAL_ERROR,
})


def classify_download_exception(exc: BaseException) -> str:
    """Map a download_handler exception to an error_code.

    Currently a substring sniff over the message; download_handler raises
    plain RuntimeErrors. If/when it grows typed exceptions, switch on type.
    """
    msg = str(exc).lower()
    if "sha256 mismatch" in msg:
        return SHA_MISMATCH
    return DOWNLOAD_FAILED
