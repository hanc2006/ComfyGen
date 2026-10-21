#!/usr/bin/env python3
"""RunPod-style compatibility server for ComfyGen.

This file exposes the same HTTP surface that the RunPod serverless API expects:
- POST /run
- POST /v2/<endpoint_id>/run
- GET /status/<job_id>
- GET /v2/<endpoint_id>/status/<job_id>
- POST /cancel/<job_id>
- POST /v2/<endpoint_id>/cancel/<job_id>
- GET /health

Internally it delegates to the existing functions in `comfy_gen.serverless`:
- `serverless.submit(...)`
- `serverless.status(...)`
- `serverless.cancel(...)`

This is a compatibility shim: it preserves the RunPod request/response shape,
while reusing the project's existing job submission logic.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from comfy_gen import serverless


def _json_response(payload: dict[str, Any], status: int = 200) -> tuple[bytes, int]:
    body = json.dumps(payload).encode("utf-8")
    return body, status


def _parse_endpoint_id(path: str) -> str | None:
    """Extract endpoint ID from /v2/<endpoint_id>/... routes."""
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "v2":
        return parts[1]
    return None


def _parse_job_id(path: str) -> str | None:
    """Extract job ID from /status/<job_id> or /v2/.../status/<job_id>."""
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "status":
        return parts[1]
    if len(parts) >= 4 and parts[0] == "v2" and parts[2] == "status":
        return parts[3]
    if len(parts) >= 4 and parts[0] == "v2" and parts[2] == "cancel":
        return parts[3]
    return None


def _normalize_workflow_payload(body: Any) -> dict[str, Any]:
    """Normalize RunPod-like payload into a dict expected by `serverless.submit`."""
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")

    payload = body.get("input") if isinstance(body.get("input"), dict) else body
    if not isinstance(payload, dict):
        raise ValueError("Request must contain an 'input' object or equivalent JSON payload")

    workflow = payload.get("workflow")
    if workflow is None:
        workflow = body.get("workflow")
    if workflow is None:
        raise ValueError("Missing workflow JSON in request body")
    if not isinstance(workflow, dict):
        raise ValueError("workflow must be a JSON object")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(workflow, fh)
        workflow_path = fh.name

    file_inputs = payload.get("file_inputs")
    if file_inputs is None:
        file_inputs = body.get("file_inputs")
    if file_inputs is None:
        file_inputs = {}

    resolved_inputs: dict[str, str] = {}
    for node_id, value in file_inputs.items():
        if isinstance(value, str):
            resolved_inputs[node_id] = value
            continue
        if isinstance(value, dict):
            if "local_path" in value and isinstance(value["local_path"], str):
                resolved_inputs[node_id] = value["local_path"]
                continue
            if "path" in value and isinstance(value["path"], str):
                resolved_inputs[node_id] = value["path"]
                continue
            if "url" in value and isinstance(value["url"], str):
                remote_url = value["url"]
                filename = value.get("filename") or Path(remote_url).name or f"{node_id}_input"
                with tempfile.NamedTemporaryFile("wb", suffix="_" + filename, delete=False) as tmp:
                    req = Request(remote_url, headers={"User-Agent": "comfy-gen-compat-server/1.0"})
                    with urlopen(req, timeout=120) as resp:
                        tmp.write(resp.read())
                    resolved_inputs[node_id] = tmp.name
                continue
        raise ValueError(f"Unsupported file input payload for node {node_id}: {value!r}")

    overrides = payload.get("overrides")
    if overrides is None:
        overrides = body.get("overrides")
    if overrides is None:
        overrides = {}

    timeout = payload.get("timeout")
    if timeout is None:
        timeout = body.get("timeout")
    if timeout is None:
        timeout = 1200

    poll_interval = payload.get("poll_interval")
    if poll_interval is None:
        poll_interval = body.get("poll_interval")
    if poll_interval is None:
        poll_interval = 3

    endpoint_id = payload.get("endpoint_id")
    if endpoint_id is None:
        endpoint_id = body.get("endpoint_id")

    return {
        "workflow_path": workflow_path,
        "file_inputs": resolved_inputs,
        "overrides": overrides,
        "timeout": timeout,
        "poll_interval": poll_interval,
        "endpoint_id": endpoint_id,
    }


class CompatRequestHandler(BaseHTTPRequestHandler):
    server_version = "comfy-gen-compat-server/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Silence noisy default logging to keep service output clean.
        return

    def _read_json_body(self) -> Any:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            return {}
        try:
            length = int(length_header)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc

        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body, code = _json_response(payload, status)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path

        if path == "/health":
            self._send_json({"ok": True, "status": "healthy"})
            return

        job_id = _parse_job_id(path)
        if job_id is None:
            self._send_error(f"Unsupported GET path: {path}", 404)
            return

        endpoint_id = _parse_endpoint_id(path)
        try:
            result = serverless.status(job_id, endpoint_id=endpoint_id)
            self._send_json(result)
        except Exception as exc:
            self._send_error(str(exc), 500)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path

        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_error(f"Invalid JSON body: {exc}", 400)
            return

        if path in {"/run", "/v2/run"}:
            self._handle_submit(body, endpoint_id=None)
            return

        if path.startswith("/v2/") and path.endswith("/run"):
            self._handle_submit(body, endpoint_id=_parse_endpoint_id(path))
            return

        if path in {"/cancel", "/v2/cancel"}:
            self._handle_cancel(body, endpoint_id=None)
            return

        if path.startswith("/v2/") and path.endswith("/cancel"):
            self._handle_cancel(body, endpoint_id=_parse_endpoint_id(path))
            return

        self._send_error(f"Unsupported POST path: {path}", 404)

    def _handle_submit(self, body: Any, endpoint_id: str | None) -> None:
        try:
            prepared = _normalize_workflow_payload(body)
            if endpoint_id is not None:
                prepared["endpoint_id"] = endpoint_id

            result = serverless.submit(
                workflow_path=prepared["workflow_path"],
                file_inputs=prepared["file_inputs"] or None,
                overrides=prepared["overrides"] or None,
                timeout=int(prepared["timeout"]),
                poll_interval=int(prepared["poll_interval"]),
                endpoint_id=prepared["endpoint_id"],
            )
            self._send_json(result)
        except Exception as exc:
            self._send_error(str(exc), 500)

    def _handle_cancel(self, body: Any, endpoint_id: str | None) -> None:
        job_id = None
        if isinstance(body, dict):
            job_id = body.get("job_id") or body.get("id")
        if job_id is None:
            job_id = _parse_job_id(self.path)
        if job_id is None:
            self._send_error("Missing job_id", 400)
            return

        try:
            result = serverless.cancel(job_id, endpoint_id=endpoint_id)
            self._send_json(result)
        except Exception as exc:
            self._send_error(str(exc), 500)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), CompatRequestHandler)
    print(f"ComfyGen compatibility server listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a RunPod-compatible compatibility server for ComfyGen. "
            "It exposes the same routes as the RunPod API and delegates to "
            "`comfy_gen.serverless` for submit/status/cancel."
        )
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
