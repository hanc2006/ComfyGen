#!/usr/bin/env bash
# Start the ComfyGen pod server against an already running ComfyUI.
#
# ComfyUI is NOT started here: the pod template launches it. This script only
# waits until it answers, then (re)starts the ComfyGen server.
#
# Usage (on the pod, from the cloned repo):
#   ./start-comfygen.sh
#
# Optional environment:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET
#                      S3 storage for outputs. Without it outputs are not uploaded
#                      and are only available via GET /outputs/<job_id>/<index>
#                      (enough for BlockFlow; `comfy-gen submit` needs S3 URLs).
#   COMFYGEN_API_TOKEN Shared secret; clients send it as "Authorization: Bearer <token>".
#                      If unset, a random token is generated and printed on startup.
#   COMFYUI_PORT       ComfyUI port (default: 8188)
#   COMFYUI_WAIT       Seconds to wait for ComfyUI to answer (default: 300)
#   COMFYUI_MODELS_DIR ComfyUI models folder used by /models/* (preset installs)
#                      (default: /workspace/models)
#   SERVER_PORT        Pod server port to expose via RunPod (default: 8787)
#   PYTHON             Interpreter (default: /usr/local/bin/python)
#   PYTHON_LIBS        Persistent package folder on the volume (default: /workspace/.python).
#                      Used as PYTHONUSERBASE: packages go to its lib/pythonX.Y/site-packages,
#                      console scripts to its bin/.
#   S3_ENDPOINT_URL, S3_REGION   for R2/B2/MinIO etc.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
COMFYUI_WAIT="${COMFYUI_WAIT:-300}"
SERVER_PORT="${SERVER_PORT:-8787}"
PYTHON="${PYTHON:-/usr/local/bin/python}"
PYTHON_LIBS="${PYTHON_LIBS:-/workspace/.python}"
LOG_DIR="${LOG_DIR:-/tmp}"
export COMFYUI_MODELS_DIR="${COMFYUI_MODELS_DIR:-/workspace/models}"

missing_s3=()
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY S3_BUCKET; do
  [ -n "${!var:-}" ] || missing_s3+=("$var")
done
if [ "${#missing_s3[@]}" -gt 0 ]; then
  echo "==> S3 output storage not configured (missing: ${missing_s3[*]}); outputs are served via /outputs only."
fi

# Model installs (/models/*) write here. Generation works without it, so only warn.
if [ -d "$COMFYUI_MODELS_DIR" ]; then
  echo "==> Models folder: ${COMFYUI_MODELS_DIR}"
else
  echo "WARNING: models folder ${COMFYUI_MODELS_DIR} does not exist; preset installs will fail." >&2
  echo "         Set COMFYUI_MODELS_DIR to the folder your ComfyUI loads models from." >&2
fi

export COMFYUI_BASE_URL="http://127.0.0.1:${COMFYUI_PORT}"

# The interpreter lives in the image; packages live on the volume. Pointing
# PYTHONUSERBASE at the volume folder makes it Python's "user site": pip --user
# installs there, and Python imports from it (including the .pth file an
# editable install relies on). Exported, so server.py sees the same packages.
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: Python not found at ${PYTHON} (set PYTHON)" >&2
  exit 1
fi
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "ERROR: pip is not available for ${PYTHON}" >&2
  exit 1
fi
mkdir -p "$PYTHON_LIBS"
export PYTHONUSERBASE="$PYTHON_LIBS"
USER_SITE="$("$PYTHON" -c 'import site; print(site.getusersitepackages() if site.ENABLE_USER_SITE else "")')"
if [ -z "$USER_SITE" ]; then
  echo "ERROR: ${PYTHON} has user site-packages disabled, so ${PYTHON_LIBS} would not be importable." >&2
  exit 1
fi
echo "==> Python: ${PYTHON} ($("$PYTHON" --version 2>&1))"
echo "==> Packages: ${USER_SITE}"

# Generate an API token when none was provided. It is exported so server.py
# (started below) inherits it, and printed so clients can be configured.
if [ -z "${COMFYGEN_API_TOKEN:-}" ]; then
  COMFYGEN_API_TOKEN="$("$PYTHON" -c 'import secrets; print(secrets.token_urlsafe(32))')"
  echo "==> COMFYGEN_API_TOKEN was not set; generated a new one:"
  echo "    COMFYGEN_API_TOKEN=${COMFYGEN_API_TOKEN}"
fi
export COMFYGEN_API_TOKEN

echo "==> Installing ComfyGen from ${REPO_DIR}"
"$PYTHON" -m pip install -q --user -e "$REPO_DIR" boto3

# Only check that ComfyUI is reachable; the pod template is responsible for
# starting it (and may still be booting, so wait for a while).
echo -n "==> Waiting for ComfyUI on port ${COMFYUI_PORT}"
deadline=$((SECONDS + COMFYUI_WAIT))
until curl -fsS "${COMFYUI_BASE_URL}/system_stats" >/dev/null 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo
    echo "ERROR: ComfyUI did not answer on ${COMFYUI_BASE_URL} within ${COMFYUI_WAIT}s." >&2
    echo "       This script does not start ComfyUI; check the pod's ComfyUI process and its log." >&2
    exit 1
  fi
  echo -n "."
  sleep 2
done
echo " ready"

# --- Make sure the port is free and ends up served by the process we start ---
# `ss`/`lsof` are often missing on pod images, so read /proc directly.
# Prints "pid<TAB>cwd<TAB>cmdline" for each process listening on TCP port $1.
port_owners() {
  "$PYTHON" - "$1" <<'PY'
import glob, os, sys
port = int(sys.argv[1])
inodes = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = open(table).read().splitlines()[1:]
    except OSError:
        continue
    for line in lines:
        f = line.split()
        if len(f) > 9 and f[3] == "0A" and int(f[1].rsplit(":", 1)[1], 16) == port:
            inodes.add(f[9])
for fd in glob.glob("/proc/[0-9]*/fd/*"):
    try:
        link = os.readlink(fd)
    except OSError:
        continue
    if link.startswith("socket:[") and link[8:-1] in inodes:
        pid = fd.split("/")[2]
        inodes.discard(link[8:-1])
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cwd, cmd = "?", "?"
        print(f"{pid}\t{cwd}\t{cmd}")
PY
}

stop_pid() {  # stop_pid <pid>: TERM, then KILL after 10s
  kill "$1" 2>/dev/null || return 0
  for _ in $(seq 1 10); do
    kill -0 "$1" 2>/dev/null || return 0
    sleep 1
  done
  kill -9 "$1" 2>/dev/null || true
}

# 1. The server this script started last time.
PID_FILE="${LOG_DIR}/comfygen-server.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "==> Stopping previous server (pid $(cat "$PID_FILE"))"
  stop_pid "$(cat "$PID_FILE")"
fi
rm -f "$PID_FILE"

# 2. Anything else still holding the port: an older ComfyGen server (started by
#    hand, by an older script, or with another LOG_DIR) is replaced; any other
#    program is left alone and we refuse to start.
while IFS=$'\t' read -r opid ocwd ocmd; do
  [ -n "$opid" ] || continue
  if [[ "$ocmd" == *server.py* || "$ocwd" == "$REPO_DIR"* ]]; then
    echo "==> Stopping a ComfyGen server not started by this script (pid ${opid}: ${ocmd})"
    stop_pid "$opid"
  else
    echo "ERROR: port ${SERVER_PORT} is used by another program (pid ${opid}: ${ocmd})." >&2
    echo "       Stop it or choose another port with SERVER_PORT=..." >&2
    exit 1
  fi
done < <(port_owners "$SERVER_PORT")
if [ -n "$(port_owners "$SERVER_PORT")" ]; then
  echo "ERROR: port ${SERVER_PORT} is still in use:" >&2
  port_owners "$SERVER_PORT" >&2
  exit 1
fi

echo "==> Starting ComfyGen server on port ${SERVER_PORT} (log: ${LOG_DIR}/comfygen-server.log)"
cd "$REPO_DIR"
nohup "$PYTHON" server.py --host 0.0.0.0 --port "$SERVER_PORT" >"${LOG_DIR}/comfygen-server.log" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

# Ready only when OUR process owns the port and answers /health.
for _ in $(seq 1 15); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: the server exited during startup. Log:" >&2
    tail -n 30 "${LOG_DIR}/comfygen-server.log" >&2 || true
    rm -f "$PID_FILE"
    exit 1
  fi
  if [ "$(port_owners "$SERVER_PORT" | cut -f1)" = "$pid" ] \
     && curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
    echo "==> Ready (pid ${pid}): http://127.0.0.1:${SERVER_PORT}/health"
    [ -n "${RUNPOD_POD_ID:-}" ] && echo "    Public URL: https://${RUNPOD_POD_ID}-${SERVER_PORT}.proxy.runpod.net"
    exit 0
  fi
  sleep 1
done

echo "ERROR: server (pid ${pid}) did not answer on port ${SERVER_PORT}. Log:" >&2
tail -n 30 "${LOG_DIR}/comfygen-server.log" >&2 || true
exit 1
