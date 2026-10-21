#!/usr/bin/env python3
"""Local job worker compatible with `comfy_gen.serverless`.

This module is designed to sit behind a RunPod-compatible HTTP gateway, such as
`server.py`. It keeps the same job lifecycle as the serverless helpers in
`comfy_gen.serverless`, but executes work locally instead of talking to RunPod.

The default behavior is a safe simulation mode that does not require a ComfyUI
installation. For real local execution, set the `COMFY_GEN_WORKER_COMMAND`
environment variable to a shell command that runs your local workflow executor.

Example:
    export COMFY_GEN_WORKER_COMMAND='python /path/to/run_local_workflow.py --workflow {workflow_path}'

The placeholder `{workflow_path}` is automatically substituted before execution.
The runner can write a JSON result to stdout or to a file named `result.json` in
its working directory. If it writes JSON to stdout, the worker will parse it and
store it under the job output.

Compatibility contract:
- `submit(...)` returns a RunPod-style `{"id": "<job_id>"}` response.
- `status(job_id)` returns a RunPod-style status payload with `status` set to
  `IN_QUEUE`, `IN_PROGRESS`, `COMPLETED`, `FAILED`, `CANCELLED`, etc.
- `cancel(job_id)` returns `{"job_id": ..., "status": "cancelled"}`.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_JOBS: dict[str, "Job"] = {}
_JOBS_LOCK = threading.Lock()


@dataclass
class Job:
    job_id: str
    workflow_path: str
    file_inputs: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    timeout: int = 1200
    poll_interval: int = 3
    status: str = "IN_QUEUE"
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    process: subprocess.Popen[str] | None = None
    sim_mode: bool = False


def _register_job(job: Job) -> None:
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job


def _get_job(job_id: str) -> Job | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def _update_job(job_id: str, **kwargs: Any) -> Job | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        for key, value in kwargs.items():
            setattr(job, key, value)
        return job


def _finalize_completed(job: Job, payload: dict[str, Any]) -> None:
    job.status = "COMPLETED"
    job.completed_at = time.time()
    job.output = payload


def _finalize_failed(job: Job, message: str) -> None:
    job.status = "FAILED"
    job.completed_at = time.time()
    job.error = message


def _simulate_job(job: Job) -> None:
    """Fallback worker that completes immediately without external dependencies."""
    try:
        time.sleep(min(2, max(1, job.poll_interval)))
        payload = {
            "ok": True,
            "job_id": job.job_id,
            "output": {
                "url": f"local://simulated/{job.job_id}.png",
                "seed": 0,
                "resolution": {"width": 1024, "height": 1024},
                "message": "Simulated local worker result. No ComfyUI runner configured.",
            },
            "delay_seconds": 0,
            "elapsed_seconds": 2,
        }
        _finalize_completed(job, payload)
    except Exception as exc:  # pragma: no cover - defensive branch
        _finalize_failed(job, str(exc))


def _build_runner_command(job: Job) -> str | None:
    command = os.environ.get("COMFY_GEN_WORKER_COMMAND")
    if not command:
        return None
    return command.format(workflow_path=shlex.quote(job.workflow_path), job_id=job.job_id)


def _parse_job_output(stdout: str) -> dict[str, Any] | None:
    if not stdout:
        return None
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try the last JSON object in a larger log stream.
        start = text.rfind("{")
        if start == -1:
            return None
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            return None
    return None


def _execute_external_job(job: Job) -> None:
    """Run an external command for the job; parse stdout JSON if available."""
    command = _build_runner_command(job)
    if not command:
        _simulate_job(job)
        return

    job.status = "IN_PROGRESS"
    job.started_at = time.time()

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=os.environ.get("SHELL", "/bin/bash"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "COMFY_GEN_JOB_ID": job.job_id, "COMFY_GEN_WORKFLOW_PATH": job.workflow_path},
        )
        job.process = proc
        stdout, stderr = proc.communicate(timeout=max(1, job.timeout))
        if proc.returncode != 0:
            msg = stderr.strip() or f"Worker exited with code {proc.returncode}"
            _finalize_failed(job, msg)
            return

        payload = _parse_job_output(stdout)
        if payload is None:
            payload = {
                "ok": True,
                "job_id": job.job_id,
                "output": {"message": "Local worker completed successfully.", "url": None},
                "elapsed_seconds": 0,
                "delay_seconds": 0,
            }

        if payload.get("job_id") is None:
            payload["job_id"] = job.job_id
        if "delay_seconds" not in payload:
            payload["delay_seconds"] = 0
        if "elapsed_seconds" not in payload:
            payload["elapsed_seconds"] = max(0, int(time.time() - (job.started_at or job.created_at)))

        _finalize_completed(job, payload)
    except subprocess.TimeoutExpired:
        _finalize_failed(job, f"Worker timed out after {job.timeout} seconds")
    except Exception as exc:  # pragma: no cover - defensive branch
        _finalize_failed(job, str(exc))


def submit(
    workflow_path: str,
    file_inputs: dict[str, str] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    timeout: int = 1200,
    poll_interval: int = 3,
    endpoint_id: str | None = None,
) -> dict[str, str]:
    """Submit a local workflow job.

    Returns a RunPod-style `{"id": "..."}` payload expected by
    `comfy_gen.serverless.submit()` callers.
    """
    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        workflow_path=workflow_path,
        file_inputs=file_inputs or {},
        overrides=overrides or {},
        timeout=timeout,
        poll_interval=poll_interval,
        status="IN_QUEUE",
    )
    _register_job(job)

    worker = threading.Thread(
        target=_execute_external_job,
        args=(job,),
        daemon=True,
    )
    worker.start()
    return {"id": job_id}


def status(job_id: str, endpoint_id: str | None = None) -> dict[str, Any]:
    """Return the status for a local job.

    The response shape mirrors RunPod's status endpoint enough for
    `comfy_gen.serverless.status()` to consume it directly.
    """
    job = _get_job(job_id)
    if job is None:
        return {"job_id": job_id, "status": "UNKNOWN"}

    result: dict[str, Any] = {"job_id": job_id, "status": job.status}

    if job.status in {"COMPLETED", "FAILED", "CANCELLED"}:
        result["status"] = job.status
        if job.output:
            result.update(job.output)
        if job.error:
            result["error"] = job.error
        if job.status == "CANCELLED":
            result["cancelled"] = True
        return result

    if job.status == "IN_PROGRESS":
        result["output"] = {"message": "Local worker is running..."}
        return result

    result["status"] = "IN_QUEUE"
    return result


def cancel(job_id: str, endpoint_id: str | None = None) -> dict[str, Any]:
    """Cancel a local job."""
    job = _get_job(job_id)
    if job is None:
        return {"job_id": job_id, "status": "cancelled"}

    job.status = "CANCELLED"
    job.completed_at = time.time()
    if job.process and job.process.poll() is None:
        try:
            job.process.terminate()
        except Exception:
            pass
    return {"job_id": job_id, "status": "cancelled"}


class CompatWorkerHandler(BaseHTTPRequestHandler):
    server_version = "comfy-gen-worker/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_json(self) -> Any:
        length = self.headers.get("Content-Length")
        if length is None:
            return {}
        raw = self.rfile.read(int(length))
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_json({"ok": True, "status": "healthy"})
            return

        if path.startswith("/status/"):
            job_id = path.split("/status/", 1)[1]
            self._send_json(status(job_id))
            return

        self._send_json({"error": f"Unsupported GET path: {path}"}, 404)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path in {"/run", "/v2/run"}:
            body = self._read_json()
            payload = body.get("input") if isinstance(body.get("input"), dict) else body
            workflow = payload.get("workflow") if isinstance(payload, dict) else None
            file_inputs = payload.get("file_inputs") if isinstance(payload, dict) else None
            overrides = payload.get("overrides") if isinstance(payload, dict) else None
            timeout = int(payload.get("timeout", 1200)) if isinstance(payload, dict) else 1200
            poll_interval = int(payload.get("poll_interval", 3)) if isinstance(payload, dict) else 3

            if not isinstance(workflow, dict):
                self._send_json({"error": "Missing workflow JSON in request body"}, 400)
                return

            with open("/tmp/comfygen_workflow.json", "w", encoding="utf-8") as fh:
                json.dump(workflow, fh)

            job = submit(
                workflow_path="/tmp/comfygen_workflow.json",
                file_inputs=file_inputs or None,
                overrides=overrides or None,
                timeout=timeout,
                poll_interval=poll_interval,
                endpoint_id=None,
            )
            self._send_json(job)
            return

        if path.startswith("/cancel/"):
            job_id = path.split("/cancel/", 1)[1]
            self._send_json(cancel(job_id))
            return

        self._send_json({"error": f"Unsupported POST path: {path}"}, 404)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), CompatWorkerHandler)
    print(f"ComfyGen worker listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down worker...")
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a local ComfyGen worker backend compatible with serverless.py.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
