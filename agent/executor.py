"""Parse model actions, execute tools, and retain a training-ready trajectory."""
from __future__ import annotations
import difflib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from tools import CodingTools, ToolResult

@dataclass
class Step:
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str
    success: bool

class AgentExecutor:
    def __init__(
        self,
        workspace: Path,
        *,
        allowed_test_commands: Iterable[str] | None = None,
        test_timeout: int = 30,
    ) -> None:
        self.tools = CodingTools(
            workspace,
            timeout=test_timeout,
            allowed_test_commands=allowed_test_commands,
        )
        self.steps: list[Step] = []
        self.original_files: dict[str, str] = {}

    @staticmethod
    def _decode_object(text: str) -> dict[str, Any]:
        start = text.find("{")
        if start < 0:
            raise ValueError("action arguments must contain a JSON object")
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        if not isinstance(value, dict):
            raise ValueError("action arguments must be a JSON object")
        return value

    @classmethod
    def parse(cls, response: str) -> tuple[str, str, dict[str, Any]]:
        thought = re.search(r"Thought:\s*(.*?)(?=\nAction:)", response, re.S)
        action = re.search(r"Action:\s*(\w+)", response)
        if not (thought and action):
            raise ValueError("model output must contain Thought and Action")
        separated = re.search(r"Action Input:\s*", response)
        argument_text = (
            response[separated.end() :]
            if separated
            else response[action.end() :]
        )
        return (
            thought.group(1).strip(),
            action.group(1),
            cls._decode_object(argument_text),
        )

    def execute(self, response: str) -> Step:
        thought, action, arguments = self.parse(response)
        if action == "read_file":
            result = self.tools.read_file(**arguments)
        elif action == "write_file":
            path = arguments.get("path", "")
            if path not in self.original_files:
                before = self.tools.snapshot_file(path)
                if before.ok:
                    self.original_files[path] = before.output
            result = self.tools.write_file(**arguments)
        elif action == "run_test":
            result = self.tools.run_test(**arguments)
        else:
            result = ToolResult(False, f"unknown action: {action}")
        step = Step(thought, action, arguments, result.output, result.ok)
        self.steps.append(step)
        return step

    def final_patch(self) -> str:
        patches: list[str] = []
        for path, before in self.original_files.items():
            after_result = self.tools.snapshot_file(path)
            if not after_result.ok:
                continue
            patches.extend(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after_result.output.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
        return "".join(patches) or "(no changes)"

    def trajectory(self) -> dict[str, Any]:
        return {
            "steps": [asdict(step) for step in self.steps],
            "final_patch": self.final_patch(),
        }
