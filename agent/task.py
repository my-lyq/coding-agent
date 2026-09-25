"""Typed task contract for repository-level Coding Agent rollouts."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TRAJECTORY_DIR = Path("/data_local/lyq/data_coding_agent/trajectories/raw")
DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

@dataclass(frozen=True)
class Task:
    task_id: str
    repo_path: Path
    problem_statement: str
    test_command: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Task":
        required = {"task_id", "repo_path", "problem_statement", "test_command"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"task is missing fields: {sorted(missing)}")
        task = cls(
            task_id=str(value["task_id"]).strip(),
            repo_path=Path(str(value["repo_path"])).expanduser().resolve(),
            problem_statement=str(value["problem_statement"]).strip(),
            test_command=str(value["test_command"]).strip(),
        )
        task.validate()
        return task

    @classmethod
    def from_json(cls, path: str | Path) -> "Task":
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load task {source}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"task JSON must be an object: {source}")
        return cls.from_dict(value)

    def validate(self) -> None:
        if not self.task_id or not TASK_ID_PATTERN.fullmatch(self.task_id):
            raise ValueError(
                "task_id must contain only letters, numbers, dot, underscore, or hyphen"
            )
        if not self.problem_statement:
            raise ValueError("problem_statement must not be empty")
        if not self.test_command:
            raise ValueError("test_command must not be empty")
        if not self.repo_path.is_dir():
            raise ValueError(f"repo_path is not a directory: {self.repo_path}")
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"repo_path is not a Git checkout: {self.repo_path}")

def validate_trajectory_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed = DEFAULT_TRAJECTORY_DIR.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"trajectory directory must be under {allowed}; got {resolved}")
    return resolved
