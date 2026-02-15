#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() {
  printf '[validate] %s\n' "$*"
}

run_step() {
  local label="$1"
  shift
  log "$label"
  "$@"
}

UV_BIN=""
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif command -v uv.exe >/dev/null 2>&1; then
  UV_BIN="$(command -v uv.exe)"
else
  for candidate in "$HOME/.local/bin/uv" "$HOME/.local/bin/uv.exe"; do
    if [ -x "$candidate" ]; then
      UV_BIN="$candidate"
      break
    fi
  done
fi

if [ -z "$UV_BIN" ]; then
  echo "[validate] ERROR: uv is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[validate] ERROR: git is required but was not found in PATH." >&2
  exit 1
fi

log "Starting lightweight validation"
log "Using uv at: $UV_BIN"

mapfile -t PY_FILES < <(git ls-files '*.py')
if ((${#PY_FILES[@]})); then
  run_step "Compiling Python files" "$UV_BIN" run python -m py_compile "${PY_FILES[@]}"
else
  log "No Python files found; skipping compile check"
fi

if [ -d tests ]; then
  run_step "Running pytest" "$UV_BIN" run python -m pytest -q tests
else
  log "No tests directory found; skipping pytest"
fi

if "$UV_BIN" run ruff --version >/dev/null 2>&1; then
  run_step "Running ruff check" "$UV_BIN" run ruff check rapidshot tests diagnostic_script.py
else
  log "ruff not installed; skipping lint"
fi

if [ "${RAPIDSHOT_VALIDATE_DIAGNOSTIC:-0}" = "1" ]; then
  if [ -f diagnostic_script.py ]; then
    run_step "Running DirectX diagnostic script" "$UV_BIN" run ./diagnostic_script.py
  else
    log "diagnostic_script.py not found; skipping diagnostic pass"
  fi
fi

log "Validation finished successfully"
