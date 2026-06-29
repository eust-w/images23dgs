#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${IMAGES23DGS_APP_ROOT:-/opt/images23dgs_app}"
CONFIG="${IMAGES23DGS_CONFIG:-$APP_ROOT/config.toml}"
PYTHON_BIN="${APP_ROOT}/venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

if [[ "${1:-}" == "--check-only" ]]; then
  "$PYTHON_BIN" -m images23dgs product doctor --config "$CONFIG"
  exit 0
fi

exec "$PYTHON_BIN" -m images23dgs product serve --config "$CONFIG"
