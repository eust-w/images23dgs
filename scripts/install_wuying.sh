#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${IMAGES23DGS_APP_ROOT:-/opt/images23dgs_app}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
NODE_VERSION="${IMAGES23DGS_NODE_VERSION:-22.22.1}"
AHOLO_SPLAT_TRANSFORM_VERSION="${IMAGES23DGS_AHOLO_SPLAT_TRANSFORM_VERSION:-1.5.1}"
SKIP_AHOLO_NODE="${IMAGES23DGS_SKIP_AHOLO_NODE:-0}"

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

install_aholo_transform() {
  if [[ "$SKIP_AHOLO_NODE" == "1" ]]; then
    echo "skip Aholo Node/splat-transform installation: IMAGES23DGS_SKIP_AHOLO_NODE=1"
    return 0
  fi

  local node_bin="$APP_ROOT/node/bin/node"
  local npm_bin="$APP_ROOT/node/bin/npm"
  if [[ ! -x "$node_bin" ]]; then
    local os arch package url tmp
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"
    case "$arch" in
      x86_64|amd64) arch="x64" ;;
      aarch64|arm64) arch="arm64" ;;
      *) echo "skip Aholo Node install: unsupported arch $arch" >&2; return 0 ;;
    esac
    case "$os" in
      linux|darwin) ;;
      *) echo "skip Aholo Node install: unsupported os $os" >&2; return 0 ;;
    esac
    package="node-v${NODE_VERSION}-${os}-${arch}"
    url="https://nodejs.org/dist/v${NODE_VERSION}/${package}.tar.gz"
    tmp="$(mktemp -d)"
    echo "+ install Node.js ${NODE_VERSION} for Aholo transform"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "+ curl -fL $url -o $tmp/node.tar.gz"
      echo "+ tar -xzf $tmp/node.tar.gz -C $tmp"
      echo "+ mkdir -p $APP_ROOT/node"
    else
      if command -v curl >/dev/null 2>&1; then
        curl -fL "$url" -o "$tmp/node.tar.gz"
      elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp/node.tar.gz" "$url"
      else
        echo "skip Aholo Node install: missing curl/wget" >&2
        return 0
      fi
      tar -xzf "$tmp/node.tar.gz" -C "$tmp"
      rm -rf "$APP_ROOT/node"
      mkdir -p "$APP_ROOT/node"
      tar -cf - -C "$tmp/$package" . | tar -xf - -C "$APP_ROOT/node"
    fi
  fi

  export PATH="$APP_ROOT/node/bin:$PATH"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ $npm_bin install -g @manycore/aholo-splat-transform@${AHOLO_SPLAT_TRANSFORM_VERSION}"
  elif [[ -x "$npm_bin" ]]; then
    run "$npm_bin" install -g "@manycore/aholo-splat-transform@${AHOLO_SPLAT_TRANSFORM_VERSION}"
  else
    echo "skip Aholo splat-transform install: npm not available at $npm_bin" >&2
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
install_aholo_transform
run "$APP_ROOT/venv/bin/python" -m images23dgs product write-config --path "$APP_ROOT/config.toml"
run "$APP_ROOT/venv/bin/python" -m images23dgs product doctor --config "$APP_ROOT/config.toml"

echo "install complete: $APP_ROOT"
