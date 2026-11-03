"""Tests for pod_models.py (model download/delete) and its server.py routes.

A local HTTP server stands in for HuggingFace / CivitAI. It can redirect to a
different host name (127.0.0.1 -> localhost) to check that the auth header is
not forwarded across hosts.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import pod_models
import server

FILES = {
    "/a.bin": b"A" * 300_000,
    "/b.bin": b"B" * 120_000,
    "/slow.bin": b"S" * 2_000_000,
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def file_server():
    seen_auth: dict[str, str | None] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            seen_auth[f"{self.headers.get('Host', '').split(':')[0]}{path}"] = self.headers.get("Authorization")
            if path == "/private.bin" and self.headers.get("Authorization") != "Bearer tok":
                self.send_response(401)
                self.end_headers()
                return
            if path.startswith("/redirect/"):
                port = self.server.server_address[1]
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{port}/{path.split('/', 2)[2]}")
                self.end_headers()
                return
            data = FILES.get(path) or (b"P" * 1000 if path == "/private.bin" else None)
            if data is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if path == "/slow.bin":
                for i in range(0, len(data), 100_000):
                    self.wfile.write(data[i:i + 100_000])
                    time.sleep(0.05)
            else:
                self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen_auth
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def models_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("COMFYUI_MODELS_DIR", str(root))
    monkeypatch.setattr(pod_models, "CHUNK_BYTES", 64 * 1024)
    pod_models._JOBS.clear()
    yield root
    for job in list(pod_models._JOBS.values()):
        job.cancel.set()


def _wait(job_id: str, timeout: float = 10) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = pod_models.download_status(job_id)
        if st["status"] in pod_models.TERMINAL:
            return st
        time.sleep(0.05)
    raise AssertionError(f"download did not finish: {pod_models.download_status(job_id)}")


def test_resolve_dest_rejects_escapes(models_root):
    for bad in ("", "x.bin", "/etc/passwd", "../x.bin", "loras/../../x", "loras//x", "loras/./x"):
        with pytest.raises(ValueError):
            pod_models.resolve_dest(bad)
    assert pod_models.resolve_dest("loras/sub/x.safetensors") == models_root.resolve() / "loras/sub/x.safetensors"


def test_models_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("COMFYUI_MODELS_DIR", str(tmp_path / "nope"))
    with pytest.raises(pod_models.ModelsDirError):
        pod_models.models_dir()


def test_download_verify_and_cache(file_server, models_root):
    base, _ = file_server
    models = [
        {"url": f"{base}/a.bin", "dest": "checkpoints/a.bin", "sha256": _sha(FILES["/a.bin"])},
        {"url": f"{base}/b.bin", "dest": "loras/b.bin"},
    ]
    st = _wait(pod_models.start_download(models)["id"])
    assert st["status"] == "completed", st
    assert (models_root / "checkpoints/a.bin").read_bytes() == FILES["/a.bin"]
    assert (models_root / "loras/b.bin").read_bytes() == FILES["/b.bin"]
    assert [f["cached"] for f in st["files"]] == [False, False]
    assert st["files_done"] == 2 and st["bytes_done"] == st["bytes_total"] == 420_000
    assert not list(models_root.rglob("*.part"))

    # second run: both files are reused (hash verified for the one with sha256)
    st2 = _wait(pod_models.start_download(models)["id"])
    assert st2["status"] == "completed"
    assert [f["cached"] for f in st2["files"]] == [True, True]


def test_existing_file_with_wrong_hash_is_replaced(file_server, models_root):
    base, _ = file_server
    (models_root / "checkpoints").mkdir()
    (models_root / "checkpoints/a.bin").write_bytes(b"stale")
    st = _wait(pod_models.start_download(
        [{"url": f"{base}/a.bin", "dest": "checkpoints/a.bin", "sha256": _sha(FILES["/a.bin"])}],
    )["id"])
    assert st["status"] == "completed" and st["files"][0]["cached"] is False
    assert (models_root / "checkpoints/a.bin").read_bytes() == FILES["/a.bin"]


def test_sha_mismatch_fails_without_leaving_files(file_server, models_root):
    base, _ = file_server
    st = _wait(pod_models.start_download(
        [{"url": f"{base}/a.bin", "dest": "checkpoints/a.bin", "sha256": "0" * 64}],
    )["id"])
    assert st["status"] == "failed"
    assert "sha256 mismatch" in st["message"] and "error" not in st
    assert st["files"][0]["status"] == "failed"
    assert not (models_root / "checkpoints/a.bin").exists()
    assert not list(models_root.rglob("*.part"))


def test_http_404_fails_fast(file_server, models_root):
    base, _ = file_server
    st = _wait(pod_models.start_download([{"url": f"{base}/missing.bin", "dest": "loras/m.bin"}])["id"])
    assert st["status"] == "failed" and "HTTP 404" in st["message"]


def test_token_sent_only_to_matching_host_and_dropped_on_redirect(file_server, models_root, monkeypatch):
    base, seen = file_server
    # Treat 127.0.0.1 as "huggingface" for this test.
    monkeypatch.setattr(
        pod_models, "_auth_header_for",
        lambda url, hf, cv: f"Bearer {hf}" if hf and "127.0.0.1" in url else None,
    )
    st = _wait(pod_models.start_download([
        {"url": f"{base}/private.bin", "dest": "loras/p.bin"},
        {"url": f"{base}/redirect/b.bin", "dest": "loras/b.bin"},
    ], hf_token="tok")["id"])
    assert st["status"] == "completed", st
    assert seen["127.0.0.1/private.bin"] == "Bearer tok"
    assert seen["127.0.0.1/redirect/b.bin"] == "Bearer tok"
    assert seen["localhost/b.bin"] is None  # stripped on the cross-host redirect


def test_real_auth_host_matching():
    assert pod_models._auth_header_for("https://huggingface.co/x", "h", "c") == "Bearer h"
    assert pod_models._auth_header_for("https://civitai.com/api/download/models/1", "h", "c") == "Bearer c"
    assert pod_models._auth_header_for("https://evil.example/huggingface.co", "h", "c") is None
    assert pod_models._auth_header_for("https://huggingface.co/x", "", "c") is None


def test_cancel_and_busy(file_server, models_root):
    base, _ = file_server
    job = pod_models.start_download([{"url": f"{base}/slow.bin", "dest": "checkpoints/slow.bin"}])
    with pytest.raises(pod_models.DownloadBusy):
        pod_models.start_download([{"url": f"{base}/a.bin", "dest": "checkpoints/a.bin"}])
    time.sleep(0.15)
    pod_models.cancel_download(job["id"])
    st = _wait(job["id"])
    assert st["status"] == "cancelled"
    assert not (models_root / "checkpoints/slow.bin").exists()
    assert not list(models_root.rglob("*.part"))


def test_insufficient_disk(file_server, models_root):
    base, _ = file_server
    with pytest.raises(pod_models.InsufficientDisk):
        pod_models.start_download([{"url": f"{base}/a.bin", "dest": "checkpoints/a.bin", "size_gb": 10 ** 6}])


def test_invalid_requests(models_root):
    for bad in ([], None, [{"url": "ftp://x", "dest": "loras/x"}], [{"url": "http://x", "dest": "../x"}],
                [{"url": "http://x", "dest": "loras/x", "sha256": "abc"}]):
        with pytest.raises(ValueError):
            pod_models.start_download(bad)


def test_delete(models_root):
    (models_root / "loras").mkdir()
    (models_root / "loras/x.bin").write_bytes(b"x")
    res = pod_models.delete_models(["loras/x.bin", "loras/gone.bin", "../outside"])
    assert not (models_root / "loras/x.bin").exists()
    assert res["ok"] is False
    assert [r["deleted"] for r in res["results"]] == [True, True, False]


# ---------------------------------------------------------------------------
# server.py routes
# ---------------------------------------------------------------------------

@pytest.fixture
def pod_server(models_root, monkeypatch):
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


def test_server_model_routes(pod_server, file_server, models_root):
    base, _ = file_server
    assert _call(f"{pod_server}/models/disk", token=None)[0] == 401
    code, disk = _call(f"{pod_server}/models/disk")
    assert code == 200 and disk["models_dir"] == str(models_root.resolve()) and disk["free_bytes"] > 0

    code, job = _call(f"{pod_server}/models/download", "POST", {"models": [
        {"url": f"{base}/a.bin", "dest": "checkpoints/a.bin", "sha256": _sha(FILES["/a.bin"])},
    ]})
    assert code == 202 and job["status"] == "running"
    for _ in range(100):
        code, st = _call(f"{pod_server}/models/download/{job['id']}")
        if st["status"] in pod_models.TERMINAL:
            break
        time.sleep(0.05)
    assert code == 200 and st["status"] == "completed"
    assert _call(f"{pod_server}/models/download/nope")[0] == 404

    assert _call(f"{pod_server}/models/download", "POST", {"models": [{"url": "x", "dest": "y"}]})[0] == 400
    assert _call(f"{pod_server}/models/download", "POST", {"models": [
        {"url": f"{base}/a.bin", "dest": "checkpoints/big.bin", "size_gb": 10 ** 6},
    ]})[0] == 507

    code, slow = _call(f"{pod_server}/models/download", "POST", {"models": [
        {"url": f"{base}/slow.bin", "dest": "checkpoints/slow.bin"},
    ]})
    assert code == 202
    assert _call(f"{pod_server}/models/download", "POST", {"models": [
        {"url": f"{base}/b.bin", "dest": "loras/b.bin"},
    ]})[0] == 409
    code, st = _call(f"{pod_server}/models/download/{slow['id']}/cancel", "POST")
    assert code == 200
    _wait(slow["id"])

    code, res = _call(f"{pod_server}/models/delete", "POST", {"paths": ["checkpoints/a.bin"]})
    assert code == 200 and res == {"ok": True, "results": [{"path": "checkpoints/a.bin", "deleted": True}]}
    assert _call(f"{pod_server}/models/delete", "POST", {})[0] == 400


def test_server_models_dir_missing(pod_server, monkeypatch, tmp_path):
    monkeypatch.setenv("COMFYUI_MODELS_DIR", str(tmp_path / "nope"))
    assert _call(f"{pod_server}/models/disk")[0] == 503
    assert _call(f"{pod_server}/models/download", "POST", {"models": [{"url": "http://x/a", "dest": "loras/a"}]})[0] == 503


def test_models_dir_default_prefers_first_existing(monkeypatch, tmp_path):
    monkeypatch.delenv("COMFYUI_MODELS_DIR", raising=False)
    first, second = tmp_path / "workspace_models", tmp_path / "comfy_models"
    second.mkdir()
    monkeypatch.setattr(pod_models, "DEFAULT_MODELS_DIRS", (str(first), str(second)))
    assert pod_models.models_dir() == second.resolve()
    first.mkdir()
    assert pod_models.models_dir() == first.resolve()


def test_default_models_dirs_start_with_workspace_models():
    assert pod_models.DEFAULT_MODELS_DIRS[0] == "/workspace/models"


def test_list_models(models_root):
    (models_root / "loras/sub").mkdir(parents=True)
    (models_root / "loras/a.safetensors").write_bytes(b"x" * 10)
    (models_root / "loras/sub/b.safetensors").write_bytes(b"y" * 3)
    (models_root / "loras/c.safetensors.part").write_bytes(b"partial")
    (models_root / "loras/.hidden").write_bytes(b"h")
    res = pod_models.list_models(["loras", "vae"])
    assert res["models_dir"] == str(models_root.resolve())
    assert [(e["filename"], e["path"], e["size_bytes"]) for e in res["folders"]["loras"]] == [
        ("a.safetensors", "loras/a.safetensors", 10),
        ("sub/b.safetensors", "loras/sub/b.safetensors", 3),
    ]
    assert res["folders"]["vae"] == []
    for bad in ([], ["../x"], ["a/b"], [""]):
        with pytest.raises(ValueError):
            pod_models.list_models(bad)


def test_server_list_route(pod_server, models_root):
    (models_root / "checkpoints").mkdir()
    (models_root / "checkpoints/m.safetensors").write_bytes(b"m")
    code, res = _call(f"{pod_server}/models/list?folders=checkpoints,loras")
    assert code == 200
    assert res["folders"]["checkpoints"][0]["path"] == "checkpoints/m.safetensors"
    assert res["folders"]["loras"] == []
    assert _call(f"{pod_server}/models/list")[0] == 400
    assert _call(f"{pod_server}/models/list?folders=../etc")[0] == 400
    assert _call(f"{pod_server}/models/list?folders=loras", token=None)[0] == 401
