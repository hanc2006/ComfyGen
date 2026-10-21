"""Tests for the pod-mode worker (worker.py) and server (server.py).

A small fake ComfyUI HTTP server stands in for the real one, implementing just
the routes worker.py uses: /prompt, /queue, /history/<id>, /view,
/upload/image, /interrupt, /system_stats.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import server
import worker
from comfy_gen import serverless

SAVE_WORKFLOW = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "out"}},
}


class FakeComfy:
    """State for the fake ComfyUI. Behaviour is chosen by class_types in the prompt."""

    def __init__(self) -> None:
        self.prompts: dict[str, dict] = {}
        self.polls: dict[str, int] = {}
        self.uploads: list[tuple[str, bytes]] = []
        self.deleted: list[str] = []
        self.interrupts = 0
        self.lock = threading.Lock()

    def kinds(self, pid: str) -> set[str]:
        return {n.get("class_type") for n in self.prompts[pid].values()}

    def queue(self) -> dict:
        running, pending = [], []
        for pid in self.prompts:
            if pid in self.deleted or self.polls.get(pid, 0) >= 2 and "Hang" not in self.kinds(pid):
                continue
            item = [0, pid, {}, {}, []]
            (pending if "Hang" in self.kinds(pid) else running).append(item)
        return {"queue_running": running, "queue_pending": pending}

    def history(self, pid: str) -> dict:
        with self.lock:
            self.polls[pid] = self.polls.get(pid, 0) + 1
            polls = self.polls[pid]
        kinds = self.kinds(pid)
        if "Hang" in kinds or polls < 2:
            return {}
        if "Explode" in kinds:
            status = {
                "status_str": "error",
                "completed": False,
                "messages": [["execution_error", {
                    "node_id": "9", "node_type": "Explode", "exception_message": "CUDA out of memory\n",
                }]],
            }
            return {pid: {"outputs": {}, "status": status}}
        outputs = {
            "2": {"images": [{"filename": "out_00001_.png", "subfolder": "", "type": "output"}]},
            "3": {"images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}]},
        }
        return {pid: {"outputs": outputs, "status": {"status_str": "success", "completed": True, "messages": []}}}


@pytest.fixture
def fake_comfy(monkeypatch):
    state = FakeComfy()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def do_GET(self):
            if self.path == "/system_stats":
                return self._json({"system": {}})
            if self.path == "/queue":
                return self._json(state.queue())
            if self.path.startswith("/history/"):
                return self._json(state.history(self.path.rsplit("/", 1)[1]))
            if self.path.startswith("/view?"):
                body = b"PNGDATA"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._json({}, 404)

        def do_POST(self):
            body = self._body()
            if self.path == "/prompt":
                prompt = json.loads(body)["prompt"]
                if any(n.get("class_type") == "Invalid" for n in prompt.values()):
                    return self._json({
                        "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                        "node_errors": {"5": {"class_type": "CheckpointLoaderSimple", "errors": [{
                            "message": "Value not in list",
                            "details": "ckpt_name: 'missing.safetensors' not in []",
                            "extra_info": {"input_name": "ckpt_name", "received_value": "missing.safetensors"},
                        }]}},
                    }, 400)
                pid = str(uuid.uuid4())
                state.prompts[pid] = prompt
                return self._json({"prompt_id": pid, "number": 0, "node_errors": {}})
            if self.path == "/upload/image":
                filename = re.search(rb'filename="([^"]+)"', body).group(1).decode()
                state.uploads.append((filename, body))
                return self._json({"name": filename, "subfolder": "", "type": "input"})
            if self.path == "/queue":
                state.deleted.extend(json.loads(body).get("delete", []))
                return self._json({})
            if self.path == "/interrupt":
                state.interrupts += 1
                return self._json({})
            self._json({}, 404)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(worker, "COMFYUI_BASE_URL", f"http://127.0.0.1:{httpd.server_address[1]}")

    from comfy_gen import storage
    monkeypatch.setattr(storage, "upload_output", lambda data, filename, **kw: f"https://s3.example/{filename}")
    yield state
    httpd.shutdown()
    httpd.server_close()


def _wait(job_id: str, timeout: float = 10) -> dict:
    env = worker.wait(job_id, timeout=timeout)
    assert env is not None
    return env


def test_job_completes_with_s3_url(fake_comfy):
    job = worker.submit(SAVE_WORKFLOW, poll_interval=0.05)
    env = _wait(job["id"])
    assert env["status"] == "COMPLETED", env
    out = env["output"]
    assert out["ok"] is True
    assert out["output"]["url"] == "https://s3.example/out_00001_.png"
    # temp (preview) outputs are skipped
    assert out["output"]["urls"] == ["https://s3.example/out_00001_.png"]
    assert "delayTime" in env and "executionTime" in env


def test_overrides_and_file_inputs_rewrite_nodes(fake_comfy, tmp_path):
    img = tmp_path / "ref.png"
    img.write_bytes(b"REFIMAGE")
    job = worker.submit(
        SAVE_WORKFLOW,
        file_inputs={"1": {"field": "image", "local_path": str(img)}},
        overrides={"2": {"filename_prefix": "custom"}},
        poll_interval=0.05,
    )
    assert _wait(job["id"])["status"] == "COMPLETED"
    (prompt,) = fake_comfy.prompts.values()
    uploaded_name, upload_body = fake_comfy.uploads[0]
    assert b"REFIMAGE" in upload_body
    assert uploaded_name.endswith("_ref.png")
    assert prompt["1"]["inputs"]["image"] == uploaded_name
    assert prompt["2"]["inputs"]["filename_prefix"] == "custom"
    # the caller's workflow is not mutated
    assert SAVE_WORKFLOW["1"]["inputs"]["image"] == "placeholder.png"


def test_unknown_override_node_fails(fake_comfy):
    env = _wait(worker.submit(SAVE_WORKFLOW, overrides={"99": {"seed": 1}})["id"])
    assert env["status"] == "FAILED"
    assert "unknown node 99" in env["error"]


def test_validation_error_is_formatted_by_client(fake_comfy):
    wf = {"5": {"class_type": "Invalid", "inputs": {}}}
    env = _wait(worker.submit(wf)["id"])
    assert env["status"] == "FAILED"
    msg = serverless._format_job_error(env["error"])
    assert "Workflow validation failed" in msg
    assert "'missing.safetensors' not found for input 'ckpt_name'" in msg


def test_execution_error(fake_comfy):
    wf = {"9": {"class_type": "Explode", "inputs": {}}}
    env = _wait(worker.submit(wf, poll_interval=0.05)["id"])
    assert env["status"] == "FAILED"
    assert env["error"] == "Explode (node 9): CUDA out of memory"


def test_cancel_pending_prompt_deletes_from_queue(fake_comfy):
    wf = {"1": {"class_type": "Hang", "inputs": {}}}
    job_id = worker.submit(wf, poll_interval=0.05)["id"]
    for _ in range(100):
        if fake_comfy.prompts:
            break
        time.sleep(0.02)
    time.sleep(0.1)  # let the runner record the prompt_id
    env = worker.cancel(job_id)
    assert env["status"] == "CANCELLED"
    assert list(fake_comfy.prompts) == fake_comfy.deleted
    assert fake_comfy.interrupts == 0
    time.sleep(0.2)
    assert worker.status(job_id)["status"] == "CANCELLED"


def test_timeout(fake_comfy):
    wf = {"1": {"class_type": "Hang", "inputs": {}}}
    env = _wait(worker.submit(wf, timeout=1, poll_interval=0.05)["id"], timeout=5)
    assert env["status"] == "TIMED_OUT"
    assert fake_comfy.deleted


def test_unknown_job():
    assert worker.status("nope") is None
    assert worker.cancel("nope") is None


# ---------------------------------------------------------------------------
# server.py
# ---------------------------------------------------------------------------

@pytest.fixture
def pod_server(fake_comfy, monkeypatch):
    monkeypatch.setattr(server.Handler, "api_token", "secret")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _call(url: str, method: str = "GET", body: dict | None = None, token: str | None = "secret"):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_routes(pod_server):
    assert _call(f"{pod_server}/health", token=None) == (200, {"ok": True, "comfyui": worker.COMFYUI_BASE_URL})
    assert _call(f"{pod_server}/status/x", token=None)[0] == 401
    assert _call(f"{pod_server}/status/x", token="wrong")[0] == 401
    assert _call(f"{pod_server}/status/x")[0] == 404
    assert _call(f"{pod_server}/run", "POST", {"input": {}})[0] == 400
    assert _call(f"{pod_server}/run", "POST", {"input": {"workflow": {"1": {}}}})[0] == 400

    code, resp = _call(f"{pod_server}/v2/any/run", "POST", {"input": {"workflow": SAVE_WORKFLOW, "poll_interval": 0.05}})
    assert code == 200 and resp["status"] == "IN_QUEUE"
    env = _wait(resp["id"])
    assert env["status"] == "COMPLETED"
    assert _call(f"{pod_server}/v2/any/status/{resp['id']}") == (200, env)
    # cancelling a finished job is a no-op that reports its state
    assert _call(f"{pod_server}/cancel/{resp['id']}", "POST")[1]["status"] == "COMPLETED"


def test_server_runsync(pod_server):
    code, env = _call(f"{pod_server}/runsync", "POST", {"input": {"workflow": SAVE_WORKFLOW, "poll_interval": 0.05}})
    assert code == 200 and env["status"] == "COMPLETED"


def test_cli_client_against_pod_server(pod_server, monkeypatch, tmp_path):
    """comfy_gen.serverless.submit/status/cancel work unchanged against server.py."""
    from comfy_gen import config

    monkeypatch.setattr(config, "load", lambda: {
        "api_base_url": pod_server, "runpod_api_key": "secret", "endpoint_id": "",
    })
    wf_path = tmp_path / "wf.json"
    wf_path.write_text(json.dumps(SAVE_WORKFLOW))
    monkeypatch.setattr(serverless.Path, "home", lambda: tmp_path)

    result = serverless.submit(str(wf_path), timeout=20, poll_interval=1)
    assert result["ok"] is True
    assert result["output"]["url"] == "https://s3.example/out_00001_.png"

    st = serverless.status(result["job_id"])
    assert st["status"] == "completed"
    assert serverless.cancel(result["job_id"])["status"] == "cancelled"
