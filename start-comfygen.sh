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

# Restart the pod server if one started by this script is still running.
PID_FILE="${LOG_DIR}/comfygen-server.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "==> Stopping previous server (pid $(cat "$PID_FILE"))"
  kill "$(cat "$PID_FILE")" || true
  sleep 1
fi

echo "==> Starting ComfyGen server on port ${SERVER_PORT} (log: ${LOG_DIR}/comfygen-server.log)"
cd "$REPO_DIR"
nohup "$PYTHON" server.py --host 0.0.0.0 --port "$SERVER_PORT" >"${LOG_DIR}/comfygen-server.log" 2>&1 &
echo $! >"$PID_FILE"

for _ in $(seq 1 15); do
  if curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
    echo "==> Ready: http://127.0.0.1:${SERVER_PORT}/health"
    [ -n "${RUNPOD_POD_ID:-}" ] && echo "    Public URL: https://${RUNPOD_POD_ID}-${SERVER_PORT}.proxy.runpod.net"
    exit 0
  fi
  sleep 1
done

echo "ERROR: server did not respond on port ${SERVER_PORT}. Log:" >&2
tail -n 30 "${LOG_DIR}/comfygen-server.log" >&2 || true
exit 1
