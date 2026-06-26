#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
STREAMLIT_BIN="$VENV_DIR/bin/streamlit"

API_HOST="0.0.0.0"
API_PORT="5000"
STREAMLIT_HOST="0.0.0.0"
STREAMLIT_PORT="8501"

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

cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

cd "$ROOT_DIR"

echo "Starting Flask API on $API_HOST:$API_PORT ..."
"$PYTHON_BIN" app.py > "$ROOT_DIR/api.log" 2>&1 &
API_PID=$!

echo "Waiting for API to become ready ..."
for _ in {1..30}; do
  if curl -sS "http://127.0.0.1:${API_PORT}/monitoring/summary" >/dev/null 2>&1; then
    echo "API is ready at http://127.0.0.1:${API_PORT}"
    break
  fi
  sleep 1
done

if ! curl -sS "http://127.0.0.1:${API_PORT}/monitoring/summary" >/dev/null 2>&1; then
  echo "API did not become ready in time. Check api.log for details."
  exit 1
fi

echo "Starting Streamlit dashboard on $STREAMLIT_HOST:$STREAMLIT_PORT ..."
API_BASE_URL="http://127.0.0.1:${API_PORT}" \
  "$STREAMLIT_BIN" run streamlit_app.py --server.address="$STREAMLIT_HOST" --server.port="$STREAMLIT_PORT"
