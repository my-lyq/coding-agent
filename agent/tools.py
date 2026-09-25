"""Small, deliberately constrained tools exposed to the coding agent."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolResult:
    ok: bool
    output: str


class CodingTools:
    """File and test tools whose paths are confined to one workspace."""

    def __init__(self, workspace: Path, timeout: int = 30) -> None:
        self.workspace = workspace.resolve()
        self.timeout = timeout

    def _safe_path(self, path: str) -> Path:
        candidate = (self.workspace / path).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return candidate

    def read_file(self, path: str) -> ToolResult:
        try:
            target = self._safe_path(path)
            return ToolResult(True, target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return ToolResult(False, f"read_file failed: {exc}")

    def write_file(self, path: str, content: str) -> ToolResult:
        try:
            target = self._safe_path(path)
            if not target.is_file():
                return ToolResult(False, f"write_file refused: file does not exist: {path}")
            target.write_text(content, encoding="utf-8")
            return ToolResult(True, f"wrote {len(content)} characters to {path}")
        except (OSError, ValueError) as exc:
            return ToolResult(False, f"write_file failed: {exc}")

    def run_test(self, command: str) -> ToolResult:
        """Run tests without a shell; this avoids shell operators and interpolation."""
        allowed = {
            "python -m unittest discover -s examples -p test_*.py",
            "python -m pytest -q examples",
        }
        if command not in allowed:
            return ToolResult(False, f"run_test refused command: {command!r}")
        try:
            completed = subprocess.run(
                command.split(),
                cwd=self.workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            output = (completed.stdout + completed.stderr).strip()
            return ToolResult(completed.returncode == 0, output or "tests produced no output")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(False, f"run_test failed: {exc}")

