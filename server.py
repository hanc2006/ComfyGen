#!/usr/bin/env python3
"""RunPod-compatible HTTP server for a dedicated ComfyUI pod.

Runs next to a long-lived ComfyUI instance and exposes the same job API as a
RunPod serverless endpoint, so `comfy-gen submit` works against the pod by
setting `api_base_url` (see README). Jobs are executed by worker.py.

Routes (the /v2/<endpoint_id> prefix is optional and the ID is ignored):
  POST /run                 -> {"id", "status"}
  POST /runsync             -> job envelope once finished (or still running after --runsync-timeout)
  GET  /status/<job_id>     -> {"id", "status", "output", "error", "delayTime", "executionTime"}
  POST /cancel/<job_id>
  GET  /health              -> unauthenticated; 503 if ComfyUI is unreachable

Every route except /health requires `Authorization: Bearer $COMFYGEN_API_TOKEN`.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import worker

MAX_BODY_BYTES = 50 * 1024 * 1024
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

    def _send_job(self, env: dict[str, Any] | None, job_id: str) -> None:
        if env is None:
            self._send_error(f"Unknown job: {job_id}", 404)
        else:
            self._send_json(env)

    def do_GET(self) -> None:  # noqa: N802
        parts = _route(self.path)
        if parts == ["health"]:
            healthy = worker.comfyui_healthy()
            self._send_json({"ok": healthy, "comfyui": worker.COMFYUI_BASE_URL}, 200 if healthy else 503)
            return
        if not self._authorized():
            return
        if len(parts) == 2 and parts[0] == "status":
            self._send_job(worker.status(parts[1]), parts[1])
            return
        self._send_error(f"Unsupported GET path: {self.path}", 404)

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
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
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
