"""Query a serverless endpoint for its worker version via the `health` command.

BlockFlow gates preset installs on a semver check between the preset's
`comfygen_min_version` and the live worker's reported version. Pairs with
serverless-runtime/health_handler.py, which returns {ok, version} fast (no
GPU/model work).
"""

import json
import urllib.error
import urllib.request
from typing import Any

from comfy_gen import output, poller


def submit_version(
    timeout: int = 60,
    poll_interval: int = 3,
    endpoint_id: str | None = None,
) -> dict[str, Any]:
    """Submit a `health` job and return {ok, worker_version}.

    Args:
        timeout: Max seconds to wait for completion.
        poll_interval: Seconds between status checks.
        endpoint_id: Override endpoint ID from config.

    Returns:
        {"ok": True, "worker_version": "X.Y.Z"} on success.

    Raises:
        ValueError: if api_key/endpoint_id are missing.
        RuntimeError: on HTTP error or unexpected worker response.
    """
    from comfy_gen import config

    cfg = config.load()
    api_key = cfg.get("runpod_api_key", "")
    if not endpoint_id:
        endpoint_id = cfg.get("endpoint_id", "")

    if not api_key:
        raise ValueError(
            "No RunPod API key configured. Run 'comfy-gen init' or set via:\n"
            "  comfy-gen config --set runpod_api_key=rpa_..."
        )
    if not endpoint_id:
        raise ValueError(
            "No RunPod endpoint configured. Run 'comfy-gen init' or set via:\n"
            "  comfy-gen config --set endpoint_id=<id>"
        )

    payload = {"input": {"command": "health"}}

    output.log(f"Querying worker version on endpoint {endpoint_id}...")
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"https://api.runpod.ai/v2/{endpoint_id}/run",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        resp = json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"RunPod API returned {e.code}: {body}")

    job_id = resp.get("id")
    if not job_id:
        raise RuntimeError(f"RunPod API did not return a job ID: {resp}")

    result = poller.poll_job(
        job_id=job_id,
        endpoint_id=endpoint_id,
        api_key=api_key,
        timeout=timeout,
        poll_interval=poll_interval,
    )

    # The worker's health handler returns {"ok": True, "version": "X.Y.Z"}.
    # Re-shape to BlockFlow's contract: {"ok": True, "worker_version": "..."}.
    worker_version = result.get("version")
    if not worker_version:
        raise RuntimeError(f"Worker health response missing version: {result}")

    return {"ok": True, "worker_version": worker_version}
