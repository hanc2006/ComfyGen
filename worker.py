"""Job manager that runs workflows on a long-lived local ComfyUI instance.

Used by server.py on a dedicated RunPod GPU pod (not serverless). ComfyUI is
started once and kept alive; this module talks to it over its HTTP API:

  POST /upload/image   upload input files referenced by file_inputs
  POST /prompt         queue the workflow
  GET  /queue          tell pending vs running
  GET  /history/<id>   detect completion / execution errors
  GET  /view           fetch output files (then uploaded to S3)
  POST /queue {delete} / POST /interrupt   cancel

Job state is kept in memory and exposed in the RunPod job envelope format
({"id", "status", "output", "error", "delayTime", "executionTime"}) so that
comfy_gen/serverless.py can talk to server.py unchanged.

Environment:
  COMFYUI_BASE_URL        ComfyUI address (default http://127.0.0.1:8188)
  COMFY_GEN_JOB_TIMEOUT   Default job timeout in seconds (default 1200)
  COMFY_GEN_JOB_TTL       Seconds finished jobs are kept for status (default 3600)
  AWS_* / S3_*            Output storage, see comfy_gen/storage.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
JOB_TIMEOUT_SECONDS = int(os.environ.get("COMFY_GEN_JOB_TIMEOUT", "1200"))
JOB_TTL_SECONDS = int(os.environ.get("COMFY_GEN_JOB_TTL", "3600"))

# One client_id for the whole process; ComfyUI uses it to route websocket
# progress messages, which we don't consume, but /prompt expects one.
CLIENT_ID = uuid.uuid4().hex

TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"})
OUTPUT_KINDS = ("images", "gifs", "videos", "audio")

_JOBS: dict[str, "Job"] = {}
_LOCK = threading.Lock()


@dataclass
class Job:
    job_id: str
    workflow: dict[str, Any]
    file_inputs: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    timeout: int = JOB_TIMEOUT_SECONDS
    poll_interval: float = 1.0
    status: str = "IN_QUEUE"
    prompt_id: str | None = None
    output: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    done: threading.Event = field(default_factory=threading.Event)


def _log(msg: str) -> None:
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# ComfyUI HTTP API
# ---------------------------------------------------------------------------

def _http(
    method: str,
    path: str,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 30,
) -> bytes:
    url = f"{COMFYUI_BASE_URL}/{path.lstrip('/')}"
    headers = {"User-Agent": "comfygen-worker/2.0"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        # Keep ComfyUI's JSON body: serverless._format_job_error parses it.
        detail = e.read().decode("utf-8", errors="replace")[:20000]
        route = "/" + path.lstrip("/").split("?", 1)[0]
        raise RuntimeError(f"ComfyUI {route} returned {e.code}: {detail}") from None


def _http_json(method: str, path: str, payload: Any = None, timeout: float = 30) -> Any:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    raw = _http(method, path, body, "application/json" if body is not None else None, timeout)
    return json.loads(raw) if raw else {}


def comfyui_healthy() -> bool:
    try:
        _http_json("GET", "/system_stats", timeout=5)
        return True
    except Exception:
        return False


def _upload_input(data: bytes, filename: str) -> str:
    """Upload a file into ComfyUI's input directory. Returns the name to put in the node."""
    boundary = uuid.uuid4().hex
    safe_name = Path(filename).name.replace('"', "_") or "input"
    parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{safe_name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode("utf-8"),
        data,
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        f"true\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    raw = _http("POST", "/upload/image", b"".join(parts), f"multipart/form-data; boundary={boundary}", 120)
    resp = json.loads(raw)
    name = resp.get("name")
    if not name:
        raise RuntimeError(f"ComfyUI /upload/image returned no name: {resp!r}")
    subfolder = resp.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name


def _queue_state(prompt_id: str) -> str | None:
    """Return 'running', 'pending', or None if the prompt is not in the queue."""
    queue = _http_json("GET", "/queue", timeout=10)
    for state, key in (("running", "queue_running"), ("pending", "queue_pending")):
        for item in queue.get(key, []):
            # Queue items are [number, prompt_id, prompt, extra_data, outputs_to_execute]
            if isinstance(item, list) and len(item) > 1 and item[1] == prompt_id:
                return state
    return None


def _history_entry(prompt_id: str) -> dict[str, Any] | None:
    history = _http_json("GET", f"/history/{urllib.parse.quote(prompt_id)}", timeout=30)
    entry = history.get(prompt_id) if isinstance(history, dict) else None
    return entry if isinstance(entry, dict) else None


def _execution_error(entry: dict[str, Any]) -> str:
    """Extract a readable message from a failed history entry."""
    for msg in entry.get("status", {}).get("messages", []):
        if not (isinstance(msg, list) and len(msg) == 2):
            continue
        kind, data = msg
        if kind == "execution_error" and isinstance(data, dict):
            node = data.get("node_type") or "node"
            return f"{node} (node {data.get('node_id')}): {data.get('exception_message', '').strip()}"
        if kind == "execution_interrupted":
            return "Execution was interrupted"
    return "ComfyUI execution failed"


def _cancel_in_comfyui(prompt_id: str) -> None:
    try:
        state = _queue_state(prompt_id)
        if state == "pending":
            _http_json("POST", "/queue", {"delete": [prompt_id]}, timeout=10)
        elif state == "running":
            # /interrupt stops whatever is executing; we checked it is ours just above.
            _http_json("POST", "/interrupt", {}, timeout=10)
    except Exception as exc:
        _log(f"cancel of prompt {prompt_id} failed: {exc}")


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

def _apply_overrides(workflow: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply {node_id: {input_name: value}} overrides to a copy of the workflow."""
    result = copy.deepcopy(workflow)
    for node_id, updates in overrides.items():
        node = result.get(str(node_id))
        if not isinstance(node, dict):
            raise ValueError(f"Override targets unknown node {node_id}")
        if not isinstance(updates, dict):
            raise ValueError(f"Overrides for node {node_id} must be an object")
        node.setdefault("inputs", {}).update(updates)
    return result


def _read_file_input(node_id: str, spec: Any) -> tuple[str, bytes, str]:
    """Resolve a file_inputs entry to (field, bytes, filename).

    Accepts the serverless.py format {"field", "url", "filename"}, a dict with
    "local_path"/"path" (a file on the pod), or a bare URL/path string.
    """
    if isinstance(spec, str):
        spec = {"url": spec} if spec.startswith(("http://", "https://")) else {"local_path": spec}
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported file input for node {node_id}: {spec!r}")

    field_name = spec.get("field") or "image"
    if spec.get("url"):
        url = spec["url"]
        req = urllib.request.Request(url, headers={"User-Agent": "comfygen-worker/2.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        filename = spec.get("filename") or Path(urllib.parse.urlsplit(url).path).name or f"{node_id}_input"
        return field_name, data, filename

    local = spec.get("local_path") or spec.get("path")
    if local:
        path = Path(local)
        if not path.is_file():
            raise ValueError(f"File input for node {node_id} not found on pod: {local}")
        return field_name, path.read_bytes(), spec.get("filename") or path.name

    raise ValueError(f"File input for node {node_id} needs 'url' or 'local_path'")


def _apply_file_inputs(workflow: dict[str, Any], file_inputs: dict[str, Any]) -> None:
    for node_id, spec in file_inputs.items():
        node = workflow.get(str(node_id))
        if not isinstance(node, dict):
            raise ValueError(f"File input targets unknown node {node_id}")
        field_name, data, filename = _read_file_input(node_id, spec)
        # Content-addressed name so concurrent jobs never clobber each other's inputs.
        digest = hashlib.md5(data).hexdigest()[:12]
        name = _upload_input(data, f"{digest}_{Path(filename).name}")
        node.setdefault("inputs", {})[field_name] = name


def _collect_outputs(job: Job, entry: dict[str, Any]) -> dict[str, Any]:
    """Fetch output files from ComfyUI, upload them to S3, build the handler output."""
    from comfy_gen import storage

    files: list[dict[str, Any]] = []
    for node_id, node_out in (entry.get("outputs") or {}).items():
        if not isinstance(node_out, dict):
            continue
        for kind in OUTPUT_KINDS:
            for item in node_out.get(kind) or []:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                if item.get("type") == "temp":  # PreviewImage etc.
                    continue
                query = urllib.parse.urlencode({
                    "filename": item["filename"],
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                })
                data = _http("GET", f"/view?{query}", timeout=300)
                url = storage.upload_output(data, item["filename"])
                files.append({"url": url, "filename": item["filename"], "node_id": node_id, "kind": kind})

    if not files:
        raise RuntimeError("Workflow finished but produced no saved outputs (add a SaveImage / video save node)")

    return {
        "ok": True,
        "output": {
            "url": files[0]["url"],
            "urls": [f["url"] for f in files],
            "files": files,
            "prompt_id": job.prompt_id,
        },
    }


def _finish(job: Job, status: str, output: dict[str, Any] | None = None, error: str | None = None) -> bool:
    """Move a job to a terminal state. No-op if it already is one (e.g. cancelled)."""
    with _LOCK:
        if job.status in TERMINAL_STATUSES:
            return False
        job.status = status
        job.output = output
        job.error = error
        job.completed_at = time.time()
        if job.started_at is None:
            job.started_at = job.completed_at
    job.done.set()
    _log(f"job {job.job_id} {status}" + (f": {error}" if error else ""))
    return True


def _run(job: Job) -> None:
    try:
        workflow = _apply_overrides(job.workflow, job.overrides)
        _apply_file_inputs(workflow, job.file_inputs)
        if job.status in TERMINAL_STATUSES:
            return

        resp = _http_json("POST", "/prompt", {"prompt": workflow, "client_id": CLIENT_ID}, timeout=60)
        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI /prompt did not return a prompt_id: {resp!r}")
        with _LOCK:
            job.prompt_id = prompt_id
            cancelled = job.status in TERMINAL_STATUSES
        if cancelled:  # cancel() arrived before we had a prompt_id
            _cancel_in_comfyui(prompt_id)
            return
        _log(f"job {job.job_id} queued as prompt {prompt_id}")

        deadline = job.created_at + job.timeout
        while job.status not in TERMINAL_STATUSES:
            if time.time() > deadline:
                _cancel_in_comfyui(prompt_id)
                _finish(job, "TIMED_OUT", error=f"Job timed out after {job.timeout}s")
                return
            try:
                entry = _history_entry(prompt_id)
                if entry is None:
                    if _queue_state(prompt_id) == "running":
                        with _LOCK:
                            if job.status == "IN_QUEUE":
                                job.status = "IN_PROGRESS"
                                job.started_at = time.time()
                elif entry.get("status", {}).get("status_str") == "error":
                    _finish(job, "FAILED", error=_execution_error(entry))
                    return
                elif entry.get("status", {}).get("completed", True):
                    output = _collect_outputs(job, entry)
                    _finish(job, "COMPLETED", output=output)
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                _log(f"job {job.job_id}: transient ComfyUI error, retrying: {exc}")
            time.sleep(job.poll_interval)
    except Exception as exc:
        _finish(job, "FAILED", error=str(exc))


# ---------------------------------------------------------------------------
# Public API used by server.py
# ---------------------------------------------------------------------------

def _evict_expired() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with _LOCK:
        for job_id in [j.job_id for j in _JOBS.values() if j.completed_at and j.completed_at < cutoff]:
            del _JOBS[job_id]


def submit(
    workflow: dict[str, Any],
    file_inputs: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    timeout: int = JOB_TIMEOUT_SECONDS,
    poll_interval: float = 1.0,
) -> dict[str, str]:
    """Start a job in a background thread. Returns {"id", "status"} like RunPod /run."""
    _evict_expired()
    job = Job(
        job_id=str(uuid.uuid4()),
        workflow=workflow,
        file_inputs=file_inputs or {},
        overrides=overrides or {},
        timeout=int(timeout),
        poll_interval=max(0.2, float(poll_interval)),
    )
    with _LOCK:
        _JOBS[job.job_id] = job
    threading.Thread(target=_run, args=(job,), name=f"job-{job.job_id}", daemon=True).start()
    return {"id": job.job_id, "status": job.status}


def status(job_id: str) -> dict[str, Any] | None:
    """Return the RunPod job envelope, or None if the job is unknown."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        env: dict[str, Any] = {"id": job.job_id, "status": job.status}
        if job.started_at is not None:
            env["delayTime"] = int((job.started_at - job.created_at) * 1000)
        if job.completed_at is not None and job.started_at is not None:
            env["executionTime"] = int((job.completed_at - job.started_at) * 1000)
        elif job.started_at is not None:
            env["executionTime"] = int((time.time() - job.started_at) * 1000)
        if job.status == "COMPLETED":
            env["output"] = job.output
        elif job.status == "IN_PROGRESS":
            env["output"] = {"stage": "inference", "message": "Running in ComfyUI"}
        elif job.status == "IN_QUEUE":
            env["output"] = {"stage": "queue", "message": "Waiting in ComfyUI queue"}
        if job.error is not None:
            env["error"] = job.error
        return env


def wait(job_id: str, timeout: float | None = None) -> dict[str, Any] | None:
    """Block until the job is terminal (or timeout) and return its envelope."""
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return None
    job.done.wait(timeout)
    return status(job_id)


def cancel(job_id: str) -> dict[str, Any] | None:
    """Cancel a job: drop it from ComfyUI's queue, or interrupt it if running."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        prompt_id = job.prompt_id
    if _finish(job, "CANCELLED", error="Cancelled by user") and prompt_id:
        _cancel_in_comfyui(prompt_id)
    return status(job_id)
