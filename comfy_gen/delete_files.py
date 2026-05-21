"""Delete files on the RunPod network volume via a serverless job.

Thin shim over the worker's `delete` command. The worker enforces a
realpath-based security check that rejects any path which doesn't resolve
strictly under /runpod-volume; missing files are idempotent.
"""

import json
import urllib.error
import urllib.request
from typing import Any

from comfy_gen import output, poller


def submit_delete(
    paths: list[str],
    timeout: int = 300,
    poll_interval: int = 3,
    endpoint_id: str | None = None,
) -> dict[str, Any]:
    """Submit a delete job to the serverless endpoint.

    Args:
        paths: Absolute paths under /runpod-volume to remove.
        timeout: Max seconds to wait for completion.
        poll_interval: Seconds between status checks.
        endpoint_id: Override endpoint ID from config.

    Returns:
        Result dict: {"ok": bool, "results": [{"path", "deleted", "error?"}]}.
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

    payload = {"input": {"command": "delete", "paths": paths}}

    output.log(f"Deleting {len(paths)} path(s) on network volume...")
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

    output.log(f"Job submitted: {job_id}")

    result = poller.poll_job(
        job_id=job_id,
        endpoint_id=endpoint_id,
        api_key=api_key,
        timeout=timeout,
        poll_interval=poll_interval,
    )

    results = result.get("results", [])
    deleted = sum(1 for r in results if r.get("deleted"))
    output.log(f"Deleted: {deleted}/{len(results)} ({len(results) - deleted} errors/skipped)")
    return result
