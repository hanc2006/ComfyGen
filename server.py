#!/usr/bin/env python3
"""RunPod-compatible HTTP server for a dedicated ComfyUI pod.

Runs next to a long-lived ComfyUI instance and exposes the same job API as a
RunPod serverless endpoint, so `comfy-gen submit` works against the pod by
setting `api_base_url` (see README). Jobs are executed by worker.py.

Routes (the /v2/<endpoint_id> prefix is optional and the ID is ignored):
  POST /run                 -> {"id", "status"}
  POST /runsync             -> job envelope once finished (or still running after --runsync-timeout)
  GET  /status/<job_id>     -> {"id", "status", "output", "error", "delayTime", "executionTime"}
  POST /cancel/<job_id>     -> job envelope
  GET  /health              -> unauthenticated; {"ok", "status": "healthy"|"unhealthy", "comfyui"}, 503 if ComfyUI is down

BlockFlow "persistent ComfyGen server" routes:
  POST /cancel              {"job_id"} -> {"ok", "id", "status"}
  PUT  /upload/<image|video>  raw bytes, X-Filename header -> {"name", "subfolder", "field"}
  GET  /outputs/<job_id>/<index>  -> the output file's bytes (index from output.files[].index)
  GET  /inventory           -> {"ok", "samplers", "schedulers", "models": {"loras": [...], ...}}

Model files (see pod_models.py; paths are relative to the ComfyUI models folder):
  POST /models/download     {"models": [{"url", "dest", "sha256"?, "size_gb"?}], "hf_token"?, "civitai_token"?}
                            -> 202 download job status; 409 if one is running, 507 if the disk is too small
  GET  /models/download/<id>          -> {"id", "status", "files": [...], "bytes_done", "bytes_total", ...}
  POST /models/download/<id>/cancel
  POST /models/delete       {"paths": ["loras/x.safetensors", ...]} -> {"ok", "results": [...]}
  GET  /models/disk         -> {"models_dir", "total_bytes", "free_bytes"}
  GET  /models/list?folders=loras,checkpoints
                            -> {"models_dir", "folders": {"loras": [{"filename", "path", "size_bytes", "modified"}]}}

Every route except /health requires `Authorization: Bearer $COMFYGEN_API_TOKEN`.

Environment:
  COMFYGEN_MAX_UPLOAD_MB    Largest accepted /upload body in MB (default 1024)
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pod_models
import worker

MAX_BODY_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_BYTES = int(os.environ.get("COMFYGEN_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
RUNSYNC_TIMEOUT_SECONDS = 90


def _route(path: str) -> list[str]:
    """Split a path into segments, dropping an optional /v2/<endpoint_id> prefix."""
    parts = [p for p in urlsplit(path).path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "v2":
        parts = parts[2:]
    return parts


def _parse_submit(body: Any) -> dict[str, Any]:
    """Validate a RunPod-style {"input": {...}} body and return worker.submit kwargs."""
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    payload = body.get("input", body)
    if not isinstance(payload, dict):
        raise ValueError("'input' must be a JSON object")

    workflow = payload.get("workflow")
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Missing 'workflow' object in input")
    if not any(isinstance(n, dict) and "class_type" in n for n in workflow.values()):
        raise ValueError("Workflow is not in ComfyUI API format (no class_type found)")

    file_inputs = payload.get("file_inputs") or {}
    overrides = payload.get("overrides") or {}
    if not isinstance(file_inputs, dict):
        raise ValueError("'file_inputs' must be an object")
    if not isinstance(overrides, dict):
        raise ValueError("'overrides' must be an object")

    try:
        timeout = int(payload.get("timeout") or worker.JOB_TIMEOUT_SECONDS)
        poll_interval = float(payload.get("poll_interval") or 1.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timeout/poll_interval: {exc}") from None

    return {
        "workflow": workflow,
        "file_inputs": file_inputs,
        "overrides": overrides,
        "timeout": timeout,
        "poll_interval": poll_interval,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "comfygen-pod-server/2.0"
    api_token: str | None = None
    runsync_timeout: float = RUNSYNC_TIMEOUT_SECONDS

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        safe_name = filename.replace('"', "_").replace("\r", "").replace("\n", "")
        self.send_header("Content-Disposition", f'inline; filename="{safe_name}"')
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int) -> None:
        self._send_json({"error": message}, status)

    def _authorized(self) -> bool:
        if not self.api_token:
            return True
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), self.api_token):
            return True
        self._send_error("Unauthorized", 401)
        return False

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw) if raw else {}

    def _read_upload(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            raise ValueError(f"Upload too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
        return self.rfile.read(length) if length else b""

    def _handle_upload(self, field_name: str) -> None:
        filename = (self.headers.get("X-Filename") or "").strip() or f"upload.{'mp4' if field_name == 'video' else 'png'}"
        data = self._read_upload()
        self._send_json(worker.upload_file(field_name, data, filename))

    def _handle_output(self, job_id: str, index_text: str) -> None:
        try:
            index = int(index_text)
        except ValueError:
            self._send_error(f"Invalid output index: {index_text}", 400)
            return
        try:
            result = worker.output_file(job_id, index)
        except worker.OutputNotReady as exc:
            self._send_error(str(exc), 409)
            return
        if result is None:
            self._send_error(f"Unknown job or output: {job_id}/{index}", 404)
            return
        data, content_type, filename = result
        self._send_bytes(data, content_type, filename)

    def _send_job(self, env: dict[str, Any] | None, job_id: str) -> None:
        if env is None:
            self._send_error(f"Unknown job: {job_id}", 404)
        else:
            self._send_json(env)

    def do_GET(self) -> None:  # noqa: N802
        parts = _route(self.path)
        if parts == ["health"]:
            healthy = worker.comfyui_healthy()
            # BlockFlow's attach check requires ok=True and status="healthy".
            self._send_json(
                {
                    "ok": healthy,
                    "status": "healthy" if healthy else "unhealthy",
                    "comfyui": worker.COMFYUI_BASE_URL,
                },
                200 if healthy else 503,
            )
            return
        if not self._authorized():
            return
        try:
            if len(parts) == 2 and parts[0] == "status":
                self._send_job(worker.status(parts[1]), parts[1])
                return
            if len(parts) == 3 and parts[0] == "outputs":
                self._handle_output(parts[1], parts[2])
                return
            if parts == ["inventory"]:
                self._send_json(worker.inventory())
                return
            if parts == ["models", "disk"]:
                self._send_json(pod_models.disk_usage())
                return
            if parts == ["models", "list"]:
                query = parse_qs(urlsplit(self.path).query)
                folders = [f for value in query.get("folders", []) for f in value.split(",") if f.strip()]
                try:
                    self._send_json(pod_models.list_models(folders))
                except ValueError as exc:
                    self._send_error(str(exc), 400)
                return
            if len(parts) == 3 and parts[:2] == ["models", "download"]:
                status = pod_models.download_status(parts[2])
                if status is None:
                    self._send_error(f"Unknown download: {parts[2]}", 404)
                else:
                    self._send_json(status)
                return
        except pod_models.ModelsDirError as exc:
            self._send_error(str(exc), 503)
            return
        except Exception as exc:
            self.log_error("unhandled error on %s: %r", self.path, exc)
            self._send_error(f"Internal error: {exc}", 500)
            return
        self._send_error(f"Unsupported GET path: {self.path}", 404)

    def do_PUT(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        parts = _route(self.path)
        try:
            if len(parts) == 2 and parts[0] == "upload":
                self._handle_upload(parts[1])
                return
        except ValueError as exc:
            self._send_error(str(exc), 400)
            return
        except Exception as exc:
            self.log_error("unhandled error on %s: %r", self.path, exc)
            self._send_error(f"Internal error: {exc}", 500)
            return
        self._send_error(f"Unsupported PUT path: {self.path}", 404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        parts = _route(self.path)
        try:
            if parts in (["run"], ["runsync"]):
                kwargs = _parse_submit(self._read_json())
                result = worker.submit(**kwargs)
                if parts == ["runsync"]:
                    result = worker.wait(result["id"], timeout=self.runsync_timeout)
                self._send_json(result)
                return
            if len(parts) == 2 and parts[0] == "cancel":
                self._send_job(worker.cancel(parts[1]), parts[1])
                return
            if parts == ["cancel"]:
                # BlockFlow form: job ID in the body. No "error" key in the reply,
                # because BlockFlow treats any "error" key as a failed request.
                body = self._read_json()
                job_id = str(body.get("job_id") or body.get("id") or "") if isinstance(body, dict) else ""
                if not job_id:
                    raise ValueError("Missing 'job_id'")
                env = worker.cancel(job_id)
                if env is None:
                    self._send_error(f"Unknown job: {job_id}", 404)
                else:
                    self._send_json({"ok": True, "id": job_id, "status": env["status"]})
                return
            if len(parts) == 2 and parts[0] == "upload":
                self._handle_upload(parts[1])
                return
            if parts == ["models", "download"]:
                body = self._read_json()
                if not isinstance(body, dict):
                    raise ValueError("Request body must be a JSON object")
                self._send_json(pod_models.start_download(
                    body.get("models"),
                    hf_token=str(body.get("hf_token") or ""),
                    civitai_token=str(body.get("civitai_token") or ""),
                ), 202)
                return
            if len(parts) == 4 and parts[:2] == ["models", "download"] and parts[3] == "cancel":
                status = pod_models.cancel_download(parts[2])
                if status is None:
                    self._send_error(f"Unknown download: {parts[2]}", 404)
                else:
                    self._send_json(status)
                return
            if parts == ["models", "delete"]:
                body = self._read_json()
                self._send_json(pod_models.delete_models(body.get("paths") if isinstance(body, dict) else None))
                return
        except pod_models.DownloadBusy as exc:
            self._send_error(str(exc), 409)
            return
        except pod_models.InsufficientDisk as exc:
            self._send_error(str(exc), 507)
            return
        except pod_models.ModelsDirError as exc:
            self._send_error(str(exc), 503)
            return
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_error(str(exc), 400)
            return
        except Exception as exc:
            self.log_error("unhandled error on %s: %r", self.path, exc)
            self._send_error(f"Internal error: {exc}", 500)
            return
        self._send_error(f"Unsupported POST path: {self.path}", 404)


def run_server(host: str, port: int, api_token: str | None) -> None:
    Handler.api_token = api_token
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"ComfyGen pod server on http://{host}:{port} -> ComfyUI at {worker.COMFYUI_BASE_URL}", file=sys.stderr)
    if not api_token:
        print("WARNING: running without authentication (--insecure)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod-compatible API server for a dedicated ComfyUI pod.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9010, help="Listen port (default: 9010)")
    parser.add_argument(
        "--insecure", action="store_true",
        help="Allow running without COMFYGEN_API_TOKEN (anyone with the URL can use the GPU)",
    )
    args = parser.parse_args()

    token = os.environ.get("COMFYGEN_API_TOKEN", "").strip() or None
    if token is None and not args.insecure:
        parser.error("COMFYGEN_API_TOKEN is not set. Set it, or pass --insecure for local testing.")
    run_server(args.host, args.port, token)


if __name__ == "__main__":
    main()
