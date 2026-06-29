#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${IMAGES23DGS_APP_ROOT:-/opt/images23dgs_app}"
SRC_DIR="${IMAGES23DGS_SRC_DIR:-$APP_ROOT/src}"
SOURCE_URL="${IMAGES23DGS_SOURCE_URL:-}"
SOURCE_REF="${IMAGES23DGS_SOURCE_REF:-}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
START_AFTER_INSTALL=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bootstrap_wuying.sh [options]

Options:
  --source URL          Source archive or git URL. Supports .tar.gz, .tgz, .zip, and .git.
  --ref REF             Git branch/tag/commit when --source is a git URL.
  --app-root PATH       Product app root. Default: /opt/images23dgs_app
  --src-dir PATH        Source checkout directory. Default: $APP_ROOT/src
  --start               Start the web service after installation.
  -h, --help            Show this help.

Environment:
  IMAGES23DGS_SOURCE_URL, IMAGES23DGS_SOURCE_REF, IMAGES23DGS_APP_ROOT,
  IMAGES23DGS_SRC_DIR, PYTHON_BOOTSTRAP
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_URL="${2:?missing value for --source}"
      shift 2
      ;;
    --ref)
      SOURCE_REF="${2:?missing value for --ref}"
      shift 2
      ;;
    --app-root)
      APP_ROOT="${2:?missing value for --app-root}"
      SRC_DIR="${IMAGES23DGS_SRC_DIR:-$APP_ROOT/src}"
      shift 2
      ;;
    --src-dir)
      SRC_DIR="${2:?missing value for --src-dir}"
      shift 2
      ;;
    --start)
      START_AFTER_INSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

fetch_file() {
  local url="$1"
  local dst="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$dst"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dst" "$url"
  else
    echo "missing required command: curl or wget" >&2
    exit 1
  fi
}

install_unzip_if_needed() {
  if command -v unzip >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y unzip
  elif command -v yum >/dev/null 2>&1; then
    yum install -y unzip
  else
    echo "missing unzip and no apt-get/yum available to install it" >&2
    exit 1
  fi
}

copy_current_source_if_available() {
  if [[ -f "pyproject.toml" && -d "images23dgs" && -f "scripts/install_wuying.sh" ]]; then
    mkdir -p "$SRC_DIR"
    tar --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' -cf - . | tar -xf - -C "$SRC_DIR"
    return 0
  fi
  return 1
}

download_source() {
  if [[ -z "$SOURCE_URL" ]]; then
    if copy_current_source_if_available; then
      return 0
    fi
    cat >&2 <<EOF
No source found.

For a completely clean machine, pass a source URL:
  bash scripts/bootstrap_wuying.sh --source https://example.com/images23dgs.tar.gz

Or run from an existing images23dgs source directory:
  bash scripts/bootstrap_wuying.sh
EOF
    exit 1
  fi

  rm -rf "$SRC_DIR"
  mkdir -p "$SRC_DIR"

  case "$SOURCE_URL" in
    *.git|git@*|ssh://*)
      need_cmd git
      if [[ -n "$SOURCE_REF" ]]; then
        git clone --depth 1 --branch "$SOURCE_REF" "$SOURCE_URL" "$SRC_DIR"
      else
        git clone --depth 1 "$SOURCE_URL" "$SRC_DIR"
      fi
      ;;
    *.zip)
      install_unzip_if_needed
      local_tmp="$(mktemp -d)"
      fetch_file "$SOURCE_URL" "$local_tmp/source.zip"
      unzip -q "$local_tmp/source.zip" -d "$local_tmp/unpacked"
      first_dir="$(find "$local_tmp/unpacked" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
      if [[ -n "$first_dir" && -f "$first_dir/pyproject.toml" ]]; then
        tar -cf - -C "$first_dir" . | tar -xf - -C "$SRC_DIR"
      else
        tar -cf - -C "$local_tmp/unpacked" . | tar -xf - -C "$SRC_DIR"
      fi
      ;;
    *.tar.gz|*.tgz)
      need_cmd tar
      local_tmp="$(mktemp -d)"
      fetch_file "$SOURCE_URL" "$local_tmp/source.tar.gz"
      mkdir -p "$local_tmp/unpacked"
      tar -xzf "$local_tmp/source.tar.gz" -C "$local_tmp/unpacked"
      first_dir="$(find "$local_tmp/unpacked" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
      if [[ -n "$first_dir" && -f "$first_dir/pyproject.toml" ]]; then
        tar -cf - -C "$first_dir" . | tar -xf - -C "$SRC_DIR"
      else
        tar -cf - -C "$local_tmp/unpacked" . | tar -xf - -C "$SRC_DIR"
      fi
      ;;
    *)
      echo "unsupported source URL: $SOURCE_URL" >&2
      echo "supported: .git, .tar.gz, .tgz, .zip" >&2
      exit 1
      ;;
  esac
}

need_cmd "$PYTHON_BOOTSTRAP"
mkdir -p "$APP_ROOT"
download_source

if [[ ! -f "$SRC_DIR/scripts/install_wuying.sh" ]]; then
  echo "source is missing scripts/install_wuying.sh: $SRC_DIR" >&2
  exit 1
fi

cd "$SRC_DIR"
IMAGES23DGS_APP_ROOT="$APP_ROOT" PYTHON_BOOTSTRAP="$PYTHON_BOOTSTRAP" bash scripts/install_wuying.sh

echo "bootstrap complete"
echo "app root: $APP_ROOT"
echo "source: $SRC_DIR"
echo "start command: IMAGES23DGS_APP_ROOT=$APP_ROOT bash $SRC_DIR/scripts/start_wuying.sh"

if [[ "$START_AFTER_INSTALL" == "1" ]]; then
  exec env IMAGES23DGS_APP_ROOT="$APP_ROOT" bash "$SRC_DIR/scripts/start_wuying.sh"
fi
