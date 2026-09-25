"""Small, constrained tools exposed to the coding agent."""
from __future__ import annotations
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

@dataclass
class ToolResult:
    ok: bool
    output: str

class CodingTools:
    """File and test tools whose paths are confined to one workspace."""
    DEFAULT_TEST_COMMANDS = {
        "python -m unittest discover -s examples -p test_*.py",
        "python -m pytest -q examples",
    }
    def __init__(self, workspace: Path, timeout: int = 30, allowed_test_commands: Iterable[str] | None = None, max_observation_chars: int = 100_000) -> None:
        self.workspace = workspace.resolve()
        self.timeout = timeout
        self.allowed_test_commands = set(allowed_test_commands if allowed_test_commands is not None else self.DEFAULT_TEST_COMMANDS)
        self.max_observation_chars = max_observation_chars

    def _safe_path(self, path: str) -> Path:
        candidate = (self.workspace / path).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return candidate

    def _truncate(self, output: str) -> str:
        if len(output) <= self.max_observation_chars:
            return output
        removed = len(output) - self.max_observation_chars
        return output[:self.max_observation_chars] + f"\n...[truncated {removed} chars]"

    def snapshot_file(self, path: str) -> ToolResult:
        """Read the complete file for internal diff bookkeeping."""
        try:
            target = self._safe_path(path)
            if not target.is_file():
                return ToolResult(False, f"read_file failed: not a file: {path}")
            return ToolResult(True, target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolResult(False, f"read_file failed: {exc}")

    def read_file(self, path: str) -> ToolResult:
        result = self.snapshot_file(path)
        return ToolResult(result.ok, self._truncate(result.output))

    def write_file(self, path: str, content: str) -> ToolResult:
        try:
            target = self._safe_path(path)
            if not target.is_file():
                return ToolResult(False, f"write_file refused: file does not exist: {path}")
            target.write_text(content, encoding="utf-8")
            return ToolResult(True, f"wrote {len(content)} characters to {path}")
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolResult(False, f"write_file failed: {exc}")

    def run_test(self, command: str) -> ToolResult:
        """Execute one exact allowlisted command without invoking a shell."""
        if command not in self.allowed_test_commands:
            return ToolResult(False, f"run_test refused command: {command!r}")
        try:
            arguments = shlex.split(command)
            if not arguments:
                return ToolResult(False, "run_test failed: empty command")
            if arguments[0] == "python":
                arguments[0] = sys.executable
            completed = subprocess.run(arguments, cwd=self.workspace, text=True, capture_output=True, timeout=self.timeout, check=False)
            output = self._truncate((completed.stdout + completed.stderr).strip())
            return ToolResult(completed.returncode == 0, output or "tests produced no output")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return ToolResult(False, f"run_test failed: {exc}")
