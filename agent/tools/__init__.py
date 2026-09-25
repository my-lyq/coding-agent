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

    def __init__(
        self,
        workspace: Path,
        timeout: int = 30,
        allowed_test_commands: Iterable[str] | None = None,
        max_observation_chars: int = 100_000,
    ) -> None:
        self.workspace = workspace.resolve()
        self.timeout = timeout
        self.allowed_test_commands = set(
            allowed_test_commands
            if allowed_test_commands is not None
            else self.DEFAULT_TEST_COMMANDS
        )
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
        return output[: self.max_observation_chars] + f"\n...[truncated {removed} chars]"

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix()

    def list_files(self, path: str = ".", max_entries: int = 200) -> ToolResult:
        try:
            if not 1 <= max_entries <= 1000:
                raise ValueError("max_entries must be between 1 and 1000")
            target = self._safe_path(path)
            if not target.exists():
                return ToolResult(False, f"list_files failed: path does not exist: {path}")
            candidates = [target] if target.is_file() else target.rglob("*")
            files = []
            for candidate in candidates:
                if not candidate.is_file() or ".git" in candidate.parts:
                    continue
                resolved = candidate.resolve()
                if self.workspace not in resolved.parents:
                    continue
                files.append(self._relative(resolved))
            files = sorted(set(files))
            shown = files[:max_entries]
            suffix = f"\n... {len(files) - len(shown)} more files" if len(files) > len(shown) else ""
            return ToolResult(True, "\n".join(shown) + suffix)
        except (OSError, TypeError, ValueError) as exc:
            return ToolResult(False, f"list_files failed: {exc}")

    def search_code(
        self,
        query: str,
        path: str = ".",
        max_results: int = 50,
    ) -> ToolResult:
        try:
            if not query:
                raise ValueError("query must not be empty")
            if not 1 <= max_results <= 200:
                raise ValueError("max_results must be between 1 and 200")
            target = self._safe_path(path)
            if not target.exists():
                return ToolResult(False, f"search_code failed: path does not exist: {path}")
            candidates = [target] if target.is_file() else target.rglob("*")
            matches = []
            for candidate in candidates:
                if not candidate.is_file() or ".git" in candidate.parts:
                    continue
                resolved = candidate.resolve()
                if self.workspace not in resolved.parents or resolved.stat().st_size > 2_000_000:
                    continue
                try:
                    lines = resolved.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeError):
                    continue
                for line_number, line in enumerate(lines, 1):
                    if query in line:
                        matches.append(f"{self._relative(resolved)}:{line_number}:{line}")
                        if len(matches) >= max_results:
                            return ToolResult(True, self._truncate("\n".join(matches)))
            return ToolResult(True, self._truncate("\n".join(matches) or "no matches"))
        except (OSError, TypeError, ValueError) as exc:
            return ToolResult(False, f"search_code failed: {exc}")

    def snapshot_file(self, path: str) -> ToolResult:
        """Read the complete file for internal diff bookkeeping."""
        try:
            target = self._safe_path(path)
            if not target.is_file():
                return ToolResult(False, f"read_file failed: not a file: {path}")
            return ToolResult(True, target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolResult(False, f"read_file failed: {exc}")

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ToolResult:
        result = self.snapshot_file(path)
        if not result.ok or (start_line is None and end_line is None):
            return ToolResult(result.ok, self._truncate(result.output))
        try:
            start = 1 if start_line is None else start_line
            lines = result.output.splitlines(keepends=True)
            end = len(lines) if end_line is None else end_line
            if start < 1 or end < start:
                raise ValueError("line range must satisfy 1 <= start_line <= end_line")
            return ToolResult(True, self._truncate("".join(lines[start - 1 : end])))
        except (TypeError, ValueError) as exc:
            return ToolResult(False, f"read_file failed: {exc}")

    def write_file(self, path: str, content: str) -> ToolResult:
        """Legacy calculator-only complete-file writer."""
        try:
            target = self._safe_path(path)
            if not target.is_file():
                return ToolResult(False, f"write_file refused: file does not exist: {path}")
            target.write_text(content, encoding="utf-8")
            return ToolResult(True, f"wrote {len(content)} characters to {path}")
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolResult(False, f"write_file failed: {exc}")

    def patch_paths(self, patch: str) -> list[str]:
        paths = []
        for line in patch.splitlines():
            if not (line.startswith("--- ") or line.startswith("+++ ")):
                continue
            value = line[4:].split("\t", 1)[0].strip()
            if value == "/dev/null":
                raise ValueError("creating or deleting files is not allowed")
            if value.startswith(("a/", "b/")):
                value = value[2:]
            target = self._safe_path(value)
            if not target.is_file():
                raise ValueError(f"patch target is not an existing file: {value}")
            if value not in paths:
                paths.append(value)
        if not paths:
            raise ValueError("patch has no valid file headers")
        return paths

    def apply_patch(self, patch: str) -> ToolResult:
        try:
            paths = self.patch_paths(patch)
            command = ["git", "apply", "--recount", "--whitespace=nowarn", "-"]
            checked = subprocess.run(
                command[:2] + ["--check", *command[2:]],
                cwd=self.workspace,
                input=patch,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            if checked.returncode != 0:
                return ToolResult(False, f"apply_patch failed: {(checked.stderr or checked.stdout).strip()}")
            applied = subprocess.run(
                command,
                cwd=self.workspace,
                input=patch,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            if applied.returncode != 0:
                return ToolResult(False, f"apply_patch failed: {(applied.stderr or applied.stdout).strip()}")
            return ToolResult(True, f"applied patch to {', '.join(paths)}")
        except (OSError, TypeError, ValueError, subprocess.TimeoutExpired) as exc:
            return ToolResult(False, f"apply_patch failed: {exc}")

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
            completed = subprocess.run(
                arguments,
                cwd=self.workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            output = self._truncate((completed.stdout + completed.stderr).strip())
            return ToolResult(completed.returncode == 0, output or "tests produced no output")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return ToolResult(False, f"run_test failed: {exc}")
