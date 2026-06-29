#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${IMAGES23DGS_APP_ROOT:-/opt/images23dgs_app}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  echo "uv not found; installing uv with ${PYTHON_BOOTSTRAP} -m pip --user"
  if [[ "$DRY_RUN" == "1" ]]; then
    run "$PYTHON_BOOTSTRAP" -m pip install --user -U uv
  else
    "$PYTHON_BOOTSTRAP" -m pip install --user -U uv
  fi

  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation finished but uv is not on PATH. Tried: $HOME/.local/bin and $HOME/.cargo/bin" >&2
    exit 1
  fi
}

run mkdir -p "$APP_ROOT"
run mkdir -p "$APP_ROOT/workspace"
run mkdir -p "$APP_ROOT/DISCOVERSE/discoverse"

ensure_uv

run uv venv "$APP_ROOT/venv"
if [[ "$DRY_RUN" == "1" ]]; then
  run uv pip install --python "$APP_ROOT/venv/bin/python" -U pip
  run uv pip install --python "$APP_ROOT/venv/bin/python" ".[web]"
elif [[ -x "$APP_ROOT/venv/bin/pip" ]]; then
  run "$APP_ROOT/venv/bin/pip" install -U pip
  run "$APP_ROOT/venv/bin/pip" install ".[web]"
else
  run uv pip install --python "$APP_ROOT/venv/bin/python" -U pip
  run uv pip install --python "$APP_ROOT/venv/bin/python" ".[web]"
fi
run "$APP_ROOT/venv/bin/python" -m images23dgs product write-config --path "$APP_ROOT/config.toml"
run "$APP_ROOT/venv/bin/python" -m images23dgs product doctor --config "$APP_ROOT/config.toml"

echo "install complete: $APP_ROOT"
