"""Model file management for the pod server: download, delete, disk usage.

Used by server.py so a client (BlockFlow presets) can install model files
straight into the pod's ComfyUI models folder, instead of the serverless flow
that spawns a separate installer pod against a network volume.

Paths are always relative to the models folder ("loras/x.safetensors") and
may not escape it. A download job runs its files one after another in a
background thread; only one job runs at a time.

Environment:
  COMFYUI_MODELS_DIR   ComfyUI models folder. Default: the first that exists
                       of /workspace/models, /workspace/ComfyUI/models and
                       /ComfyUI/models (start-comfygen.sh sets /workspace/models).
"""

from __future__ import annotations

import hashlib
import os
import shutil
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

DEFAULT_MODELS_DIRS = ("/workspace/models", "/workspace/ComfyUI/models", "/ComfyUI/models")
CHUNK_BYTES = 8 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 3
JOB_TTL_SECONDS = 3600
USER_AGENT = "comfygen-pod-server/2.0"

TERMINAL = frozenset({"completed", "failed", "cancelled"})


class ModelsDirError(RuntimeError):
    """The models folder is not configured or does not exist."""


class DownloadBusy(RuntimeError):
    """Another download job is still running."""


class InsufficientDisk(RuntimeError):
    """Not enough free space for the files that must be downloaded."""


def _log(msg: str) -> None:
    print(f"[models] {msg}", file=sys.stderr, flush=True)


def models_dir() -> Path:
    configured = os.environ.get("COMFYUI_MODELS_DIR", "").strip()
    if configured:
        path = Path(configured)
        if not path.is_dir():
            raise ModelsDirError(f"COMFYUI_MODELS_DIR does not exist: {configured}")
        return path.resolve()
    for candidate in DEFAULT_MODELS_DIRS:
        if Path(candidate).is_dir():
            return Path(candidate).resolve()
    raise ModelsDirError(
        "ComfyUI models folder not found; set COMFYUI_MODELS_DIR "
        f"(tried {', '.join(DEFAULT_MODELS_DIRS)})"
    )


def resolve_dest(rel: str, root: Path | None = None) -> Path:
    """Map a relative model path to an absolute path inside the models folder."""
    root = root or models_dir()
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("Model path is empty")
    clean = rel.strip().replace("\\", "/")
    if clean.startswith("/") or any(part in ("", ".", "..") for part in clean.split("/")):
        raise ValueError(f"Invalid model path (must be relative, no '..'): {rel!r}")
    if "/" not in clean:
        raise ValueError(f"Model path needs a folder, e.g. 'loras/{clean}'")
    target = (root / clean).resolve()
    if root not in target.parents:
        raise ValueError(f"Model path escapes the models folder: {rel!r}")
    return target


def disk_usage() -> dict[str, Any]:
    root = models_dir()
    usage = shutil.disk_usage(root)
    return {"models_dir": str(root), "total_bytes": usage.total, "free_bytes": usage.free}


# ---------------------------------------------------------------------------
# HTTP with per-host auth
# ---------------------------------------------------------------------------

def _auth_header_for(url: str, hf_token: str, civitai_token: str) -> str | None:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if hf_token and (host == "huggingface.co" or host.endswith(".huggingface.co")):
        return f"Bearer {hf_token}"
    if civitai_token and (host == "civitai.com" or host.endswith(".civitai.com")):
        return f"Bearer {civitai_token}"
    return None


class _StripAuthOnCrossHost(urllib.request.HTTPRedirectHandler):
    """Drop Authorization when a redirect leaves the original host.

    HF and CivitAI redirect to CDN / pre-signed storage URLs: the token must
    not leak there, and pre-signed S3/R2 URLs reject an extra Authorization.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old_host = urllib.parse.urlsplit(req.full_url).hostname
            new_host = urllib.parse.urlsplit(newurl).hostname
            if old_host != new_host:
                new.headers.pop("Authorization", None)
                new.unredirected_hdrs.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_StripAuthOnCrossHost)


def _open(url: str, hf_token: str, civitai_token: str, timeout: float = 60):
    headers = {"User-Agent": USER_AGENT}
    auth = _auth_header_for(url, hf_token, civitai_token)
    if auth:
        headers["Authorization"] = auth
    return _OPENER.open(urllib.request.Request(url, headers=headers), timeout=timeout)


# ---------------------------------------------------------------------------
# Download jobs
# ---------------------------------------------------------------------------

@dataclass
class FileTask:
    index: int
    url: str
    dest: str
    sha256: str = ""
    size_hint: int = 0
    status: str = "pending"  # pending | verifying | downloading | done | failed
    cached: bool = False
    bytes_done: int = 0
    bytes_total: int = 0
    speed_bps: float = 0.0
    error: str | None = None


@dataclass
class DownloadJob:
    job_id: str
    files: list[FileTask]
    hf_token: str = ""
    civitai_token: str = ""
    status: str = "running"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)


_JOBS: dict[str, DownloadJob] = {}
_LOCK = threading.Lock()


class _Cancelled(Exception):
    pass


def _sha256_of(path: Path, task: FileTask, job: DownloadJob) -> str:
    digest = hashlib.sha256()
    task.bytes_total = path.stat().st_size
    task.bytes_done = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            if job.cancel.is_set():
                raise _Cancelled()
            digest.update(chunk)
            task.bytes_done += len(chunk)
    return digest.hexdigest()


def _download_once(task: FileTask, job: DownloadJob, target: Path) -> None:
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    task.bytes_done = 0
    started = time.monotonic()
    try:
        with _open(task.url, job.hf_token, job.civitai_token) as resp, part.open("wb") as out:
            length = resp.headers.get("Content-Length")
            if length and length.isdigit():
                task.bytes_total = int(length)
            while chunk := resp.read(CHUNK_BYTES):
                if job.cancel.is_set():
                    raise _Cancelled()
                out.write(chunk)
                digest.update(chunk)
                task.bytes_done += len(chunk)
                elapsed = time.monotonic() - started
                if elapsed > 0:
                    task.speed_bps = task.bytes_done / elapsed
        if task.bytes_total and task.bytes_done != task.bytes_total:
            raise IOError(f"incomplete download ({task.bytes_done} of {task.bytes_total} bytes)")
        if task.sha256 and digest.hexdigest() != task.sha256:
            raise ValueError(f"sha256 mismatch (expected {task.sha256}, got {digest.hexdigest()})")
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)


def _run_file(task: FileTask, job: DownloadJob, root: Path) -> None:
    target = resolve_dest(task.dest, root)
    if target.is_file() and target.stat().st_size > 0:
        if not task.sha256:
            task.status, task.cached = "done", True
            task.bytes_done = task.bytes_total = target.stat().st_size
            return
        task.status = "verifying"
        if _sha256_of(target, task, job) == task.sha256:
            task.status, task.cached = "done", True
            return
        _log(f"{task.dest}: existing file has a different sha256, downloading again")

    task.status = "downloading"
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            _download_once(task, job, target)
            task.status = "done"
            task.bytes_total = task.bytes_total or task.bytes_done
            return
        except _Cancelled:
            raise
        except urllib.error.HTTPError as exc:
            # 4xx will not fix itself (bad URL, missing/invalid token).
            if 400 <= exc.code < 500:
                hint = " (check the HF / CivitAI token)" if exc.code in (401, 403) else ""
                raise RuntimeError(f"{task.dest}: HTTP {exc.code} from {urllib.parse.urlsplit(task.url).hostname}{hint}") from None
            last_error = exc
        except ValueError:
            raise  # sha256 mismatch: the source file differs, retrying will not help
        except Exception as exc:  # network errors, truncated bodies
            last_error = exc
        _log(f"{task.dest}: attempt {attempt} failed: {last_error}")
        time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"{task.dest}: download failed after {DOWNLOAD_ATTEMPTS} attempts: {last_error}")


def _run_job(job: DownloadJob, root: Path) -> None:
    current: FileTask | None = None
    try:
        for task in job.files:
            current = task
            if job.cancel.is_set():
                raise _Cancelled()
            _run_file(task, job, root)
            _log(f"{task.dest}: {'cached' if task.cached else 'downloaded'}")
        _finish(job, "completed")
    except _Cancelled:
        _finish(job, "cancelled", "Cancelled by user")
    except Exception as exc:
        if current is not None:
            current.status, current.error = "failed", str(exc)
        _finish(job, "failed", str(exc))


def _finish(job: DownloadJob, status: str, error: str | None = None) -> None:
    with _LOCK:
        job.status = status
        job.error = error
        job.completed_at = time.time()
    _log(f"job {job.job_id} {status}" + (f": {error}" if error else ""))


def _evict_expired() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    for job_id in [j.job_id for j in _JOBS.values() if j.completed_at and j.completed_at < cutoff]:
        del _JOBS[job_id]


def _parse_size_hint(spec: dict[str, Any]) -> int:
    for key, unit in (("size_bytes", 1), ("size_gb", 1024 ** 3)):
        value = spec.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value * unit)
    return 0


def start_download(
    models: list[dict[str, Any]],
    hf_token: str = "",
    civitai_token: str = "",
) -> dict[str, Any]:
    """Validate the request and start a background download job."""
    if not isinstance(models, list) or not models:
        raise ValueError("'models' must be a non-empty list")
    root = models_dir()

    tasks: list[FileTask] = []
    seen: set[str] = set()
    for spec in models:
        if not isinstance(spec, dict):
            raise ValueError("Each model must be an object with 'url' and 'dest'")
        url = str(spec.get("url") or "").strip()
        dest = str(spec.get("dest") or "").strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Model {dest or '?'} needs an http(s) 'url'")
        resolve_dest(dest, root)  # validates
        if dest in seen:
            continue
        seen.add(dest)
        sha = str(spec.get("sha256") or "").strip().lower()
        if sha and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError(f"Model {dest} has an invalid sha256")
        hint = _parse_size_hint(spec)
        tasks.append(FileTask(index=len(tasks), url=url, dest=dest, sha256=sha, size_hint=hint, bytes_total=hint))

    # Disk preflight for files not already present (existing ones are reused
    # or overwritten in place, so they need no extra space).
    needed = sum(t.size_hint for t in tasks if not resolve_dest(t.dest, root).is_file())
    free = shutil.disk_usage(root).free
    if needed and needed > free:
        raise InsufficientDisk(
            f"Not enough disk space: need ~{needed / 1024 ** 3:.1f} GB, "
            f"{free / 1024 ** 3:.1f} GB free in {root}"
        )

    with _LOCK:
        _evict_expired()
        busy = next((j for j in _JOBS.values() if j.status not in TERMINAL), None)
        if busy is not None:
            raise DownloadBusy(f"Another download is running: {busy.job_id}")
        job = DownloadJob(job_id=str(uuid.uuid4()), files=tasks,
                          hf_token=hf_token.strip(), civitai_token=civitai_token.strip())
        _JOBS[job.job_id] = job
    threading.Thread(target=_run_job, args=(job, root), name=f"download-{job.job_id}", daemon=True).start()
    _log(f"job {job.job_id}: {len(tasks)} file(s) into {root}")
    return download_status(job.job_id)  # type: ignore[return-value]


def download_status(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        files = [{
            "index": t.index,
            "dest": t.dest,
            "status": t.status,
            "cached": t.cached,
            "bytes_done": t.bytes_done,
            "bytes_total": t.bytes_total,
            "speed_bps": round(t.speed_bps),
            **({"error": t.error} if t.error else {}),
        } for t in job.files]
        result: dict[str, Any] = {
            "id": job.job_id,
            "status": job.status,
            "files": files,
            "files_total": len(files),
            "files_done": sum(1 for f in files if f["status"] == "done"),
            "bytes_total": sum(f["bytes_total"] for f in files),
            "bytes_done": sum(
                f["bytes_total"] if f["status"] == "done" else (f["bytes_done"] if f["status"] == "downloading" else 0)
                for f in files
            ),
        }
        if job.error is not None:
            result["message"] = job.error  # not "error": BlockFlow's client treats that key as a failed request
        return result


def cancel_download(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return None
    job.cancel.set()
    return download_status(job_id)


def delete_models(paths: list[str]) -> dict[str, Any]:
    """Delete model files by relative path. Missing files count as deleted."""
    if not isinstance(paths, list) or not paths:
        raise ValueError("'paths' must be a non-empty list")
    root = models_dir()
    results: list[dict[str, Any]] = []
    for rel in paths:
        try:
            target = resolve_dest(str(rel), root)
            if target.is_file():
                target.unlink()
                results.append({"path": rel, "deleted": True})
            elif target.exists():
                results.append({"path": rel, "deleted": False, "reason": "not a file"})
            else:
                results.append({"path": rel, "deleted": True, "reason": "not found"})
        except (ValueError, OSError) as exc:
            results.append({"path": rel, "deleted": False, "reason": str(exc)})
    return {"ok": all(r["deleted"] for r in results), "results": results}
