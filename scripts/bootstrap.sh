#!/usr/bin/env bash
set -Eeuo pipefail

# --- config / flags -----------------------------------------------------------
WITH_DOCS=auto   # auto | yes | no
PRECOMMIT=yes
VENV_DIR=".venv"

usage() {
  cat <<EOF
Bootstrap local dev environment (uv).

Usage: $0 [--with-docs {auto|yes|no}] [--no-pre-commit] [--venv DIR]

Options:
  --with-docs auto|yes|no   Build docs deps if mkdocs is configured (auto),
                            always (yes), or never (no). Default: auto
  --no-pre-commit           Skip installing/enabling pre-commit hooks
  --venv DIR                Virtualenv directory (default: .venv)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-docs) WITH_DOCS="${2:-auto}"; shift 2 ;;
    --no-pre-commit) PRECOMMIT=no; shift ;;
    --venv) VENV_DIR="${2:-.venv}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

log() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }

# --- prerequisites ------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install: https://github.com/astral-sh/uv" >&2
  echo "macOS (Homebrew): brew install uv" >&2
  exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# --- venv ---------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating venv in $VENV_DIR"
  uv venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
# shellcheck source=/dev/null
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # bash/zsh
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # Windows Git Bash
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
fi

# --- install dev + test -------------------------------------------------------
log "Installing dev + test deps (editable)"
uv pip install -e '.[dev,test]'

# --- install config + dataio -------------------------------------------------------
log "Installing config + dataio deps (editable)"
uv pip install -e '.[config,dataio]'

# --- run tests ----------------------------------------------------------------
log "Running tests"
uv run pytest -q || true

# --- pre-commit ---------------------------------------------------------------
if [[ "$PRECOMMIT" == "yes" ]]; then
  log "Installing and running pre-commit hooks"
  uv pip install pre-commit
  uv run pre-commit install
  uv run pre-commit run --all-files || true
fi

# --- docs (MkDocs) ------------------------------------------------------------
if [[ "$WITH_DOCS" == "yes" ]] || { [[ "$WITH_DOCS" == "auto" ]] && [[ -f "mkdocs.yml" ]]; }; then
  log "Installing docs extras"
  uv pip install -e '.[docs]'
  log "Docs installed. Preview locally with: mkdocs serve"
fi

log "Done. Activate your env with: source $VENV_DIR/bin/activate"
# --- end of script ------------------------------------------------------------
