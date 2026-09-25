"""Parse model actions, execute tools, and retain a training-ready trajectory."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools import CodingTools


@dataclass
class Step:
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str
    success: bool


class AgentExecutor:
    def __init__(self, workspace: Path) -> None:
        self.tools = CodingTools(workspace)
        self.steps: list[Step] = []
        self.original_files: dict[str, str] = {}

    @staticmethod
    def parse(response: str) -> tuple[str, str, dict[str, Any]]:
        thought = re.search(r"Thought:\s*(.*?)(?=\nAction:)", response, re.S)
        action = re.search(r"Action:\s*(\w+)", response)
        action_input = re.search(r"Action Input:\s*(\{.*\})", response, re.S)
        if not (thought and action and action_input):
            raise ValueError("model output must contain Thought, Action, and Action Input")
        return thought.group(1).strip(), action.group(1), json.loads(action_input.group(1))

    def execute(self, response: str) -> Step:
        thought, action, arguments = self.parse(response)
        if action == "read_file":
            result = self.tools.read_file(**arguments)
        elif action == "write_file":
            path = arguments.get("path", "")
            if path not in self.original_files:
                before = self.tools.read_file(path)
                if before.ok:
                    self.original_files[path] = before.output
            result = self.tools.write_file(**arguments)
        elif action == "run_test":
            result = self.tools.run_test(**arguments)
        else:
            result = type("Result", (), {"ok": False, "output": f"unknown action: {action}"})()
        step = Step(thought, action, arguments, result.output, result.ok)
        self.steps.append(step)
        return step

    def final_patch(self) -> str:
        patches: list[str] = []
        for path, before in self.original_files.items():
            after_result = self.tools.read_file(path)
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
        return {"steps": [asdict(step) for step in self.steps], "final_patch": self.final_patch()}

