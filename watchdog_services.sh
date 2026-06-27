#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
STREAMLIT_BIN="$VENV_DIR/bin/streamlit"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-5000}"
STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-5}"

API_LOG="${API_LOG:-$ROOT_DIR/api.log}"
STREAMLIT_LOG="${STREAMLIT_LOG:-$ROOT_DIR/streamlit.log}"
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:${API_PORT}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at $PYTHON_BIN"
  echo "Create and activate the virtual environment first."
  exit 1
fi

if [[ ! -x "$STREAMLIT_BIN" ]]; then
  echo "Streamlit executable not found at $STREAMLIT_BIN"
  echo "Install dependencies: pip install -r requirements.txt"
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for health checks but was not found on PATH."
  exit 1
fi

API_PID=""
STREAMLIT_PID=""

is_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

wait_for_api() {
  local retries=30
  for _ in $(seq 1 "$retries"); do
    if curl -sS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_api() {
  echo "[watchdog] Starting API on $API_HOST:$API_PORT"
  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" app.py >>"$API_LOG" 2>&1
  ) &
  API_PID=$!

  if wait_for_api; then
    echo "[watchdog] API ready (pid=$API_PID)"
  else
    echo "[watchdog] API did not become healthy in time. Check $API_LOG"
  fi
}

start_streamlit() {
  echo "[watchdog] Starting Streamlit on $STREAMLIT_HOST:$STREAMLIT_PORT"
  (
    cd "$ROOT_DIR"
    API_BASE_URL="$API_BASE_URL" \
      "$STREAMLIT_BIN" run streamlit_app.py \
      --server.address="$STREAMLIT_HOST" \
      --server.port="$STREAMLIT_PORT" >>"$STREAMLIT_LOG" 2>&1
  ) &
  STREAMLIT_PID=$!
  echo "[watchdog] Streamlit started (pid=$STREAMLIT_PID)"
}

cleanup() {
  echo "[watchdog] Stopping services..."

  if is_running "$STREAMLIT_PID"; then
    kill "$STREAMLIT_PID" 2>/dev/null || true
  fi

  if is_running "$API_PID"; then
    kill "$API_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "[watchdog] Logs:"
echo "[watchdog]   API       -> $API_LOG"
echo "[watchdog]   Streamlit -> $STREAMLIT_LOG"
echo "[watchdog] Check interval: ${WATCHDOG_INTERVAL}s"

start_api
start_streamlit

while true; do
  if ! is_running "$API_PID"; then
    echo "[watchdog] API process exited. Restarting..."
    start_api
  fi

  if ! is_running "$STREAMLIT_PID"; then
    echo "[watchdog] Streamlit process exited. Restarting..."
    start_streamlit
  fi

  sleep "$WATCHDOG_INTERVAL"
done
