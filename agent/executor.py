"""Parse model actions, execute tools, and retain a training-ready trajectory."""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .tools import CodingTools, ToolResult
    from .tools.schema import ACTION_ALIASES, normalize_action_name, parse_structured_tool_call
except ImportError:  # Support `python agent/main.py`.
    from tools import CodingTools, ToolResult
    from tools.schema import ACTION_ALIASES, normalize_action_name, parse_structured_tool_call

@dataclass
class Step:
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str
    success: bool
    original_action: str | None = None
    repaired_action: str | None = None
    invalid_action: bool = False
    repair_applied: bool = False
    phase: str | None = None
    previous_phase: str | None = None
    model_proposed_tool: str | None = None
    executed_tool: str | None = None
    controller_intervened: bool = False
    intervention_reason: str | None = None
    tool_success: bool | None = None

@dataclass(frozen=True)
class ActionRequest:
    thought: str
    action: str
    arguments: dict[str, Any]
    original_action: str
    repaired_action: str | None
    invalid_action: bool
    repair_applied: bool

class AgentExecutor:
    LEGACY_ACTIONS = {"read_file", "write_file", "run_test"}

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
    def parse(cls, response: str) -> ActionRequest:
        """Parse structured JSON, while preserving the calculator's legacy ReAct form."""
        thought = re.search(r"Thought:\s*(.*?)(?=\nAction:)", response, re.S)
        action = re.search(r"Action:\s*(\w+)", response)
        if thought and action:
            separated = re.search(r"Action Input:\s*", response)
            argument_text = response[separated.end() :] if separated else response[action.end() :]
            name = action.group(1)
            normalized = normalize_action_name(name)
            repaired = ACTION_ALIASES.get(normalized) if normalized != "write_file" else None
            canonical = repaired or normalized
            return ActionRequest(
                thought=thought.group(1).strip(),
                action=canonical,
                arguments=cls._decode_object(argument_text),
                original_action=name,
                repaired_action=canonical if repaired else None,
                invalid_action=bool(repaired) or canonical not in cls.LEGACY_ACTIONS,
                repair_applied=bool(repaired),
            )

        call = parse_structured_tool_call(response)
        return ActionRequest(
            thought=f"Call structured tool {call.tool}.",
            action=call.tool,
            arguments=call.arguments,
            original_action=call.original_action,
            repaired_action=call.repaired_action,
            invalid_action=call.invalid_action,
            repair_applied=call.repair_applied,
        )

    def _remember_file(self, path: str) -> None:
        if path in self.original_files:
            return
        before = self.tools.snapshot_file(path)
        if before.ok:
            self.original_files[path] = before.output

    def _repair_patch_arguments(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if isinstance(arguments.get("patch"), str):
            return {"patch": arguments["patch"]}
        if isinstance(arguments.get("diff"), str):
            return {"patch": arguments["diff"]}
        path, content = arguments.get("path"), arguments.get("content")
        if isinstance(path, str) and isinstance(content, str):
            before = self.tools.snapshot_file(path)
            if before.ok:
                patch = "".join(
                    difflib.unified_diff(
                        before.output.splitlines(keepends=True),
                        content.splitlines(keepends=True),
                        fromfile=f"a/{path}",
                        tofile=f"b/{path}",
                    )
                )
                return {"patch": patch}
        return arguments

    def execute(self, response: str) -> Step:
        request = self.parse(response)
        action = request.action
        arguments = request.arguments
        if request.repair_applied and action == "apply_patch":
            arguments = self._repair_patch_arguments(arguments)
        try:
            if action == "list_files":
                result = self.tools.list_files(**arguments)
            elif action == "search_code":
                result = self.tools.search_code(**arguments)
            elif action == "read_file":
                result = self.tools.read_file(**arguments)
            elif action == "apply_patch":
                try:
                    for path in self.tools.patch_paths(str(arguments.get("patch", ""))):
                        self._remember_file(path)
                except (TypeError, ValueError):
                    pass
                result = self.tools.apply_patch(**arguments)
            elif action == "write_file":  # Legacy calculator path.
                path = str(arguments.get("path", ""))
                self._remember_file(path)
                result = self.tools.write_file(**arguments)
            elif action == "run_test":
                result = self.tools.run_test(**arguments)
            else:
                result = ToolResult(False, f"unknown action: {request.original_action}")
        except TypeError as exc:
            result = ToolResult(False, f"{action} failed: invalid arguments: {exc}")

        step = Step(
            thought=request.thought,
            action=action,
            action_input=arguments,
            observation=result.output,
            success=result.ok,
            original_action=request.original_action,
            repaired_action=request.repaired_action,
            invalid_action=request.invalid_action,
            repair_applied=request.repair_applied,
            model_proposed_tool=request.original_action,
            executed_tool=action,
            controller_intervened=request.repair_applied,
            intervention_reason=(
                "action_repair" if request.repair_applied else None
            ),
            tool_success=result.ok,
        )
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
