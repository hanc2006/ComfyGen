#!/usr/bin/env python3
"""Local ComfyUI worker for a dedicated RunPod GPU pod.

This worker keeps ComfyUI running once, and communicates with the local
ComfyUI server via the official SDK. It exposes a RunPod-style job API
used by server.py:
- submit(workflow_path, ...)
- status(job_id)
- cancel(job_id)

Requirements:
- ComfyUI must already be running at http://127.0.0.1:8188
- Python SDK installed: pip install comfyui
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    from comfyui import Client
except Exception:  # pragma: no cover
    try:
        from comfy_client import Client
    except Exception:  # pragma: no cover
        Client = None


COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188")
JOB_TIMEOUT_SECONDS = int(os.environ.get("COMFY_GEN_JOB_TIMEOUT", "1200"))

_JOBS: dict[str, "Job"] = {}
_JOBS_LOCK = threading.Lock()


@dataclass
class Job:
    job_id: str
    workflow_path: str
    file_inputs: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    timeout: int = JOB_TIMEOUT_SECONDS
    poll_interval: int = 3
    status: str = "IN_QUEUE"
    prompt_id: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    client: Any = None


def _register_job(job: Job) -> None:
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job


def _get_job(job_id: str) -> Job | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def _json_load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Workflow file must contain a JSON object: {path}")
    return data


def _extract_prompt_id(result: Any) -> str | None:
    if isinstance(result, dict):
        for key in ("prompt_id", "id", "job_id"):
            val = result.get(key)
            if isinstance(val, str) and val:
                return val
        if "data" in result and isinstance(result["data"], dict):
            for key in ("prompt_id", "id", "job_id"):
                val = result["data"].get(key)
                if isinstance(val, str) and val:
                    return val
    if hasattr(result, "prompt_id"):
        val = getattr(result, "prompt_id")
        if isinstance(val, str) and val:
            return val
    if hasattr(result, "id"):
        val = getattr(result, "id")
        if isinstance(val, str) and val:
            return val
    return None


def _extract_status(history_result: Any) -> str:
    if isinstance(history_result, dict):
        value = history_result.get("status")
        if isinstance(value, str):
            return value.upper()
        value = history_result.get("state")
        if isinstance(value, str):
            return value.upper()
        if "data" in history_result and isinstance(history_result["data"], dict):
            value = history_result["data"].get("status")
            if isinstance(value, str):
                return value.upper()
    if hasattr(history_result, "status"):
        return str(getattr(history_result, "status")).upper()
    if hasattr(history_result, "state"):
        return str(getattr(history_result, "state")).upper()
    return "UNKNOWN"


def _extract_outputs(history_result: Any) -> dict[str, Any]:
    if isinstance(history_result, dict):
        for key in ("outputs", "result", "data"):
            if key in history_result and isinstance(history_result[key], dict):
                return history_result[key]
        if "data" in history_result and isinstance(history_result["data"], dict):
            for key in ("outputs", "result"):
                if key in history_result["data"] and isinstance(history_result["data"][key], dict):
                    return history_result["data"][key]
    if hasattr(history_result, "outputs"):
        outputs = getattr(history_result, "outputs")
        if isinstance(outputs, dict):
            return outputs
    return {}


def _check_local_comfyui_health() -> bool:
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{COMFYUI_BASE_URL}/system_stats",
            headers={"User-Agent": "comfygen-worker/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _client() -> Any:
    if Client is None:
        raise RuntimeError(
            "The official ComfyUI Python SDK is not installed. "
            "Install it with: pip install comfyui"
        )
    return Client(base_url=COMFYUI_BASE_URL)


def _submit_to_comfyui(workflow: dict[str, Any]) -> Any:
    client = _client()
    methods = ["queue_prompt", "prompt"]
    for method_name in methods:
        if hasattr(client, method_name):
            method = getattr(client, method_name)
            try:
                return method(workflow)
            except TypeError:
                if method_name == "queue_prompt":
                    return method({"prompt": workflow})
                raise
    raise RuntimeError("ComfyUI SDK does not expose a supported prompt submission method")


def _read_history(job: Job) -> Any:
    client = _client()
    for method_name in ("get_history", "history"):
        if hasattr(client, method_name):
            method = getattr(client, method_name)
            try:
                result = method(job.prompt_id)
                return result
            except TypeError:
                try:
                    return method()
                except Exception:
                    pass
            except Exception:
                pass
    raise RuntimeError(f"ComfyUI SDK has no history method for prompt {job.prompt_id}")


def _apply_overrides(workflow: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Apply a simple nodal override map:
    {
      "7": {"seed": 42, "steps": 20},
      "10": {"clip_skip": 2}
    }
    """
    result = json.loads(json.dumps(workflow))
    for node_id, updates in overrides.items():
        if not isinstance(updates, dict):
            continue
        node = result.get(node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
        for key, value in updates.items():
            inputs[key] = value
        node["inputs"] = inputs
    return result


def _job_runner(job: Job) -> None:
    try:
        if not _check_local_comfyui_health():
            raise RuntimeError(f"ComfyUI is not reachable at {COMFYUI_BASE_URL}")

        workflow = _json_load(job.workflow_path)
        if job.overrides:
            workflow = _apply_overrides(workflow, job.overrides)

        result = _submit_to_comfyui(workflow)
        job.prompt_id = _extract_prompt_id(result)
        if not job.prompt_id:
            raise RuntimeError(f"ComfyUI submission did not return a prompt_id: {result!r}")

        job.status = "IN_PROGRESS"
        job.started_at = time.time()

        while True:
            if time.time() - job.created_at > job.timeout:
                job.status = "FAILED"
                job.error = f"Job timed out after {job.timeout} seconds"
                job.completed_at = time.time()
                return

            try:
                history_result = _read_history(job)
                status = _extract_status(history_result)
                status_upper = status.upper()

                if status_upper in ("COMPLETED", "SUCCESS", "FINISHED"):
                    outputs = _extract_outputs(history_result)
                    job.output = {
                        "ok": True,
                        "status": "COMPLETED",
                        "output": outputs,
                        "job_id": job.job_id,
                        "elapsed_seconds": int(time.time() - (job.started_at or job.created_at)),
                        "delay_seconds": 0,
                    }
                    job.status = "COMPLETED"
                    job.completed_at = time.time()
                    return

                if status_upper in ("ERROR", "FAILED", "CANCELLED", "CANCELED"):
                    job.status = "FAILED" if "FAILED" in status_upper or "ERROR" in status_upper else "CANCELLED"
                    job.error = str(history_result)
                    job.completed_at = time.time()
                    return

                time.sleep(job.poll_interval)

            except Exception as exc:
                if time.time() - job.created_at > job.timeout:
                    job.status = "FAILED"
                    job.error = f"Worker polling failed after timeout: {exc}"
                    job.completed_at = time.time()
                    return
                time.sleep(job.poll_interval)

    except Exception as exc:  # pragma: no cover
        job.status = "FAILED"
        job.error = str(exc)
        job.completed_at = time.time()


def submit(
    workflow_path: str,
    file_inputs: dict[str, str] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    timeout: int = JOB_TIMEOUT_SECONDS,
    poll_interval: int = 3,
    endpoint_id: str | None = None,
) -> dict[str, str]:
    """Submit a job to the local ComfyUI server and return RunPod-style payload."""
    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        workflow_path=workflow_path,
        file_inputs=file_inputs or {},
        overrides=overrides or {},
        timeout=int(timeout),
        poll_interval=int(poll_interval),
    )
    _register_job(job)

    thread = threading.Thread(target=_job_runner, args=(job,), daemon=True)
    thread.start()
    return {"id": job_id}


def status(job_id: str, endpoint_id: str | None = None) -> dict[str, Any]:
    """Return a RunPod-like status response."""
    job = _get_job(job_id)
    if job is None:
        return {"job_id": job_id, "status": "UNKNOWN"}

    response: dict[str, Any] = {
        "job_id": job_id,
        "status": job.status,
    }

    if job.output:
        response.update(job.output)

    if job.error is not None:
        response["error"] = job.error

    if job.status == "IN_PROGRESS":
        response["output"] = {"message": "ComfyUI is running"}

    return response


def cancel(job_id: str, endpoint_id: str | None = None) -> dict[str, Any]:
    """Cancel a job in the local worker."""
    job = _get_job(job_id)
    if job is None:
        return {"job_id": job_id, "status": "cancelled"}

    job.status = "CANCELLED"
    job.completed_at = time.time()
    job.error = "Cancelled by user"

    try:
        client = _client()
        if hasattr(client, "interrupt"):
            client.interrupt()
    except Exception:
        pass

    return {"job_id": job_id, "status": "cancelled"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local ComfyUI worker for RunPod-compatible API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    args = parser.parse_args()

    import socketserver
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path
            if path == "/health":
                self._send_json({"ok": True, "status": "healthy"})
                return
            if path.startswith("/status/"):
                job_id = path.split("/status/", 1)[1]
                self._send_json(status(job_id))
                return
            self._send_json({"error": f"Unsupported GET path: {path}"}, 404)

        def do_POST(self) -> None:
            path = self.path
            if path == "/run":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                job = payload.get("input") if isinstance(payload.get("input"), dict) else payload
                workflow = job.get("workflow") if isinstance(job, dict) else None
                if not isinstance(workflow, dict):
                    self._send_json({"error": "Missing workflow JSON"}, 400)
                    return
                with open("/tmp/comfygen_worker_workflow.json", "w", encoding="utf-8") as fh:
                    json.dump(workflow, fh)
                result = submit(
                    workflow_path="/tmp/comfygen_worker_workflow.json",
                    file_inputs=job.get("file_inputs") if isinstance(job, dict) else None,
                    overrides=job.get("overrides") if isinstance(job, dict) else None,
                    timeout=int(job.get("timeout", JOB_TIMEOUT_SECONDS)) if isinstance(job, dict) else JOB_TIMEOUT_SECONDS,
                    poll_interval=int(job.get("poll_interval", 3)) if isinstance(job, dict) else 3,
                    endpoint_id=None,
                )
                self._send_json(result)
                return
            if path.startswith("/cancel/"):
                job_id = path.split("/cancel/", 1)[1]
                self._send_json(cancel(job_id))
                return
            self._send_json({"error": f"Unsupported POST path: {path}"}, 404)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ComfyUI worker running on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down worker...")
    finally:
        server.server_close()
