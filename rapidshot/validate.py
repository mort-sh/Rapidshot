"""Validation entrypoint for `uv run validate`."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _log(message: str) -> None:
    print(f"[validate] {message}")


def _run_step(label: str, command: list[str]) -> None:
    _log(label)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def _tracked_python_files() -> list[str]:
    if not shutil.which("git"):
        _log("git not found; skipping compile check")
        return []

    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _log("Unable to list tracked Python files; skipping compile check")
        return []

    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    _log("Starting lightweight validation")

    py_files = _tracked_python_files()
    if py_files:
        _run_step("Compiling Python files", [sys.executable, "-m", "py_compile", *py_files])
    else:
        _log("No Python files found; skipping compile check")

    tests_dir = ROOT_DIR / "tests"
    if tests_dir.is_dir():
        _run_step("Running pytest", [sys.executable, "-m", "pytest", "-q", "tests"])
    else:
        _log("No tests directory found; skipping pytest")

    if shutil.which("ruff"):
        _run_step("Running ruff check", ["ruff", "check", "rapidshot", "tests", "diagnostic_script.py"])
    else:
        _log("ruff not installed; skipping lint")

    if os.environ.get("RAPIDSHOT_VALIDATE_DIAGNOSTIC", "0") == "1":
        diagnostic_script = ROOT_DIR / "diagnostic_script.py"
        if diagnostic_script.is_file():
            _run_step("Running DirectX diagnostic script", [sys.executable, str(diagnostic_script)])
        else:
            _log("diagnostic_script.py not found; skipping diagnostic pass")

    _log("Validation finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
