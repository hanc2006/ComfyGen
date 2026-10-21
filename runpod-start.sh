#!/usr/bin/env bash
# Start ComfyUI (if not already running) and the ComfyGen pod server.
#
# Usage (on the pod, from the cloned repo):
#   export COMFYGEN_API_TOKEN=...          # clients send it as "Authorization: Bearer <token>"
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... S3_BUCKET=...
#   ./runpod-start.sh
#
# Optional environment:
#   COMFYUI_DIR      ComfyUI checkout (default: /workspace/ComfyUI)
#   COMFYUI_PORT     ComfyUI port (default: 8188)
#   SERVER_PORT      Pod server port to expose via RunPod (default: 8000)
#   PYTHON           Python interpreter (default: python3)
#   S3_ENDPOINT_URL, S3_REGION   for R2/B2/MinIO etc.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
SERVER_PORT="${SERVER_PORT:-8000}"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${LOG_DIR:-/tmp}"

: "${COMFYGEN_API_TOKEN:?Set COMFYGEN_API_TOKEN (shared secret for the API)}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID (S3 storage for outputs)}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"
: "${S3_BUCKET:?Set S3_BUCKET}"

export COMFYUI_BASE_URL="http://127.0.0.1:${COMFYUI_PORT}"

echo "==> Installing ComfyGen from ${REPO_DIR}"
"$PYTHON" -m pip install -q -e "$REPO_DIR" boto3

# Start ComfyUI unless something (e.g. the pod template) already runs it.
if curl -fsS "${COMFYUI_BASE_URL}/system_stats" >/dev/null 2>&1; then
  echo "==> ComfyUI already running on port ${COMFYUI_PORT}"
else
  if [ ! -f "${COMFYUI_DIR}/main.py" ]; then
    echo "ERROR: ComfyUI not found at ${COMFYUI_DIR} (set COMFYUI_DIR)" >&2
    exit 1
  fi
  echo "==> Starting ComfyUI from ${COMFYUI_DIR} (log: ${LOG_DIR}/comfyui.log)"
  (cd "$COMFYUI_DIR" && nohup "$PYTHON" main.py --listen 127.0.0.1 --port "$COMFYUI_PORT" \
    >"${LOG_DIR}/comfyui.log" 2>&1 &)
fi

echo -n "==> Waiting for ComfyUI"
for _ in $(seq 1 150); do
  if curl -fsS "${COMFYUI_BASE_URL}/system_stats" >/dev/null 2>&1; then
    echo " ready"
    break
  fi
  echo -n "."
  sleep 2
done
if ! curl -fsS "${COMFYUI_BASE_URL}/system_stats" >/dev/null 2>&1; then
  echo
  echo "ERROR: ComfyUI did not come up within 5 minutes. Last log lines:" >&2
  tail -n 30 "${LOG_DIR}/comfyui.log" >&2 || true
  exit 1
fi

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
