"""Strict trajectory schema validation and atomic persistence."""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from agent.executor import Step

class TrajectoryRecorder:
    REQUIRED_FIELDS = {
        "instance_id",
        "problem_statement",
        "steps",
        "modified_files",
        "patch",
        "test_result",
        "success",
        "rollout_test_passed",
        "host_test_result",
        "benchmark_resolved",
        "token_usage",
    }

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path.resolve()

    @staticmethod
    def build_record(
        *,
        instance_id: str,
        problem_statement: str,
        steps: list[Step],
        modified_files: list[str],
        patch: str,
        test_result: dict[str, Any],
        success: bool,
        rollout_test_passed: bool | None = None,
        benchmark_resolved: bool | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        record = {
            "instance_id": instance_id,
            "problem_statement": problem_statement,
            "steps": [
                {
                    "thought": step.thought,
                    "action": (
                        f"{step.action}("
                        f"{json.dumps(step.action_input, ensure_ascii=False, sort_keys=True)})"
                    ),
                    "observation": step.observation,
                    "original_action": step.original_action or step.action,
                    "repaired_action": step.repaired_action,
                    "invalid_action": step.invalid_action,
                    "repair_applied": step.repair_applied,
                    "phase": step.phase,
                    "previous_phase": step.previous_phase,
                    "model_proposed_tool": step.model_proposed_tool,
                    "executed_tool": step.executed_tool,
                    "controller_intervened": step.controller_intervened,
                    "intervention_reason": step.intervention_reason,
                    "tool_success": step.tool_success,
                    "prompt_tokens": step.prompt_tokens,
                    "generated_tokens": step.generated_tokens,
                    "generation_truncated": step.generation_truncated,
                }
                for step in steps
            ],
            "modified_files": sorted(modified_files),
            "patch": patch,
            "test_result": test_result,
            "host_test_result": test_result,
            "rollout_test_passed": (
                success if rollout_test_passed is None else rollout_test_passed
            ),
            "benchmark_resolved": benchmark_resolved,
            "token_usage": token_usage or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "success": success,
        }
        TrajectoryRecorder.validate(record)
        return record

    @classmethod
    def validate(cls, record: dict[str, Any]) -> None:
        if set(record) != cls.REQUIRED_FIELDS:
            raise ValueError(
                f"trajectory fields must be {sorted(cls.REQUIRED_FIELDS)}; "
                f"got {sorted(record)}"
            )
        if not isinstance(record["steps"], list):
            raise ValueError("trajectory steps must be a list")
        expected_step = {
            "thought", "action", "observation", "original_action",
            "repaired_action", "invalid_action", "repair_applied",
            "phase", "previous_phase", "model_proposed_tool",
            "executed_tool", "controller_intervened",
            "intervention_reason", "tool_success", "prompt_tokens",
            "generated_tokens", "generation_truncated",
        }
        if any(set(step) != expected_step for step in record["steps"]):
            raise ValueError("trajectory step fields do not match structured schema")
        if not isinstance(record["modified_files"], list):
            raise ValueError("modified_files must be a list")
        if not isinstance(record["success"], bool):
            raise ValueError("success must be boolean")
        if not isinstance(record["rollout_test_passed"], bool):
            raise ValueError("rollout_test_passed must be boolean")
        if record["benchmark_resolved"] is not None:
            raise ValueError("benchmark_resolved must remain null without official evaluation")
        if not isinstance(record["token_usage"], dict):
            raise ValueError("token_usage must be an object")

    def save(self, record: dict[str, Any], overwrite: bool = False) -> Path:
        self.validate(record)
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"trajectory already exists: {self.output_path}; use --overwrite"
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.output_path.name}.",
            suffix=".tmp",
            dir=self.output_path.parent,
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return self.output_path
