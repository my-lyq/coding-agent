#!/usr/bin/env python3
"""Aggregate rollout metrics and classify SWE-bench Agent failure modes."""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DEFAULT_INPUT = DATA_ROOT / "trajectories" / "raw"
DEFAULT_OUTPUT = DATA_ROOT / "logs" / "rollout_analysis.json"
TOOL_ACTIONS = {
    "list_files", "search_code", "read_file", "apply_patch", "run_test",
    "write_file",  # Backward-compatible analysis of pre-structured trajectories.
}
FAILURE_REASONS = (
    "invalid_action",
    "tool_error",
    "no_patch",
    "patch_failed",
    "test_failed",
    "timeout",
    "environment_error",
)
PHASE_TRANSITIONS = (
    "LOCATE -> UNDERSTAND",
    "UNDERSTAND -> MODIFY",
    "MODIFY -> VERIFY",
    "VERIFY -> UNDERSTAND",
)
TIMEOUT_RE = re.compile(r"\b(?:time[ -]?out|timed out|timeoutexpired|connecttimeout)\b", re.I)
ENVIRONMENT_RE = re.compile(
    r"(?:modulenotfounderror|importerror|filenotfounderror|connectionerror|"
    r"connecterror|connection refused|network is unreachable|"
    r"no space left|out of memory|cuda out of memory|"
    r"prepared repository not found|infrastructure failed|policy_error)",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_data_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be under {root}; got {resolved}")
    return resolved


def action_name(step: dict[str, Any]) -> str:
    action = str(step.get("action", "")).strip()
    return action.split("(", 1)[0].strip().lower()


def is_invalid_action(name: str) -> bool:
    return name not in TOOL_ACTIONS and name != "error"


def observation(step: dict[str, Any]) -> str:
    return str(step.get("observation", ""))


def has_patch(record: dict[str, Any]) -> bool:
    patch = str(record.get("patch", "")).strip()
    return bool(patch and patch != "(no changes)")


def is_tool_failure(step: dict[str, Any]) -> bool:
    action = action_name(step)
    if action not in TOOL_ACTIONS:
        return False
    text = observation(step).lower()
    if action == "read_file":
        return "read_file failed:" in text or "not a file:" in text
    if action in {"write_file", "apply_patch"}:
        return f"{action} failed:" in text or f"{action} refused:" in text
    if action == "run_test":
        return "run_test failed:" in text or "run_test refused" in text
    return f"{action} failed:" in text


def phase_transition_counts(steps: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for step in steps:
        previous = step.get("previous_phase")
        current = step.get("phase")
        if (
            isinstance(previous, str)
            and isinstance(current, str)
            and previous != current
        ):
            counts[f"{previous} -> {current}"] += 1
    return dict(counts)


def classify_failure(record: dict[str, Any]) -> str | None:
    if record.get("success") is True:
        return None
    steps = record.get("steps") if isinstance(record.get("steps"), list) else []
    names = [action_name(step) for step in steps if isinstance(step, dict)]
    all_text = "\n".join(
        observation(step) for step in steps if isinstance(step, dict)
    )
    test_result = record.get("test_result")
    if isinstance(test_result, dict):
        all_text += "\n" + str(test_result.get("output", ""))
        all_text += "\n" + str(test_result.get("policy_error", ""))

    if any(
        step.get("invalid_action") is True
        or ("invalid_action" not in step and is_invalid_action(action_name(step)))
        for step in steps if isinstance(step, dict)
    ):
        return "invalid_action"
    if TIMEOUT_RE.search(all_text):
        return "timeout"
    if "error" in names or ENVIRONMENT_RE.search(all_text):
        return "environment_error"

    write_steps = [
        step for step in steps
        if isinstance(step, dict) and action_name(step) in {"write_file", "apply_patch"}
    ]
    if any(is_tool_failure(step) for step in write_steps):
        return "patch_failed"
    if any(is_tool_failure(step) for step in steps if isinstance(step, dict)):
        return "tool_error"
    if not has_patch(record):
        return "no_patch"
    if write_steps and not record.get("modified_files"):
        return "patch_failed"
    if not isinstance(test_result, dict):
        return "environment_error"
    if test_result.get("passed") is not True:
        return "test_failed"
    return "patch_failed"


def analyze_record(record: dict[str, Any], source: Path) -> dict[str, Any]:
    steps = record.get("steps") if isinstance(record.get("steps"), list) else []
    typed_steps = [step for step in steps if isinstance(step, dict)]
    names = [action_name(step) for step in typed_steps]
    invalid_count = sum(
        step.get("invalid_action") is True
        or ("invalid_action" not in step and is_invalid_action(action_name(step)))
        for step in typed_steps
    )
    tool_calls = sum(name in TOOL_ACTIONS for name in names)
    tool_failures = sum(is_tool_failure(step) for step in typed_steps)
    test_result = record.get("test_result")
    if isinstance(test_result, dict):
        final_test = (
            "passed" if test_result.get("passed") is True
            else "failed" if test_result.get("passed") is False
            else "missing"
        )
    else:
        final_test = "missing"
    success = record.get("success") is True
    return {
        "instance_id": str(
            record.get("instance_id") or record.get("task_id") or source.stem
        ),
        "source_file": source.name,
        "success": success,
        "steps": len(typed_steps),
        "tool_calls": tool_calls,
        "invalid_actions": invalid_count,
        "tool_failures": tool_failures,
        "final_test_result": final_test,
        "failure_reason": classify_failure(record),
        "phase_transition": phase_transition_counts(typed_steps),
    }


def rounded_average(total: int, count: int) -> float:
    return round(total / count, 4) if count else 0.0


def analyze_directory(input_dir: Path) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    input_errors: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("task_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("top-level JSON value must be an object")
            details.append(analyze_record(value, path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            input_errors.append(
                {"source_file": path.name, "error": f"{type(exc).__name__}: {exc}"}
            )

    total = len(details)
    successes = sum(item["success"] for item in details)
    test_counts = Counter(item["final_test_result"] for item in details)
    failure_counts = Counter(
        item["failure_reason"] for item in details if item["failure_reason"]
    )
    phase_counts: Counter[str] = Counter()
    for item in details:
        phase_counts.update(item["phase_transition"])
    phase_transition = {name: phase_counts[name] for name in PHASE_TRANSITIONS}
    for name in sorted(set(phase_counts) - set(PHASE_TRANSITIONS)):
        phase_transition[name] = phase_counts[name]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "total_rollouts": total,
        "success_count": successes,
        "success_rate": rounded_average(successes, total),
        "average_steps": rounded_average(
            sum(item["steps"] for item in details), total
        ),
        "average_tool_calls": rounded_average(
            sum(item["tool_calls"] for item in details), total
        ),
        "invalid_action_count": sum(
            item["invalid_actions"] for item in details
        ),
        "tool_failure_count": sum(item["tool_failures"] for item in details),
        "final_test_result": {
            "passed": test_counts["passed"],
            "failed": test_counts["failed"],
            "missing": test_counts["missing"],
            "pass_rate": rounded_average(test_counts["passed"], total),
        },
        "failure_reasons": {
            reason: failure_counts[reason] for reason in FAILURE_REASONS
        },
        "phase_transition": phase_transition,
        "tasks": details,
        "input_error_count": len(input_errors),
        "input_errors": input_errors,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    input_dir = require_data_path(args.input_dir, "--input-dir")
    output = require_data_path(args.output, "--output")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"trajectory directory not found: {input_dir}")
    analysis = analyze_directory(input_dir)
    write_json_atomic(output, analysis)
    print(json.dumps({
        "total_rollouts": analysis["total_rollouts"],
        "success_rate": analysis["success_rate"],
        "average_steps": analysis["average_steps"],
        "average_tool_calls": analysis["average_tool_calls"],
        "invalid_action_count": analysis["invalid_action_count"],
        "tool_failure_count": analysis["tool_failure_count"],
        "final_test_result": analysis["final_test_result"],
        "failure_reasons": analysis["failure_reasons"],
        "phase_transition": analysis["phase_transition"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
