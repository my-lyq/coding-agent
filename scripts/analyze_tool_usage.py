#!/usr/bin/env python3
"""Analyze structured tool usage and compare it with the free-action baseline."""
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
DEFAULT_STRUCTURED = DATA_ROOT / "trajectories" / "structured"
DEFAULT_OLD = DATA_ROOT / "trajectories" / "raw"
DEFAULT_OUTPUT = DATA_ROOT / "logs" / "structured_baseline_analysis.json"
STRUCTURED_TOOLS = (
    "list_files",
    "search_code",
    "read_file",
    "apply_patch",
    "run_test",
)
OLD_TOOLS = {"read_file", "write_file", "run_test"}
FAILURE_TAXONOMY = (
    "no_patch",
    "patch_failed",
    "test_failed",
    "environment_error",
    "planning_error",
)
ENVIRONMENT_RE = re.compile(
    r"(?:modulenotfounderror|importerror|filenotfounderror|connectionerror|"
    r"connecttimeout|connection refused|network is unreachable|"
    r"no space left|out of memory|cuda out of memory|"
    r"setuptools_scm|prepared repository not found|infrastructure failed|"
    r"error while parsing the following warning configuration)",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-dir", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_data_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be under {root}; got {resolved}")
    return resolved


def action_name(step: dict[str, Any]) -> str:
    return str(step.get("action", "")).split("(", 1)[0].strip().lower()


def patch_generated(record: dict[str, Any]) -> bool:
    patch = str(record.get("patch", "")).strip()
    return bool(patch and patch != "(no changes)")


def explicit_invalid(step: dict[str, Any], structured: bool) -> bool:
    if step.get("invalid_action") is True:
        return True
    name = action_name(step)
    allowed = set(STRUCTURED_TOOLS) if structured else OLD_TOOLS
    return name not in allowed


def observation(step: dict[str, Any]) -> str:
    return str(step.get("observation", ""))


def call_key(step: dict[str, Any]) -> str:
    return str(step.get("action", "")).strip()


def has_planning_error(steps: list[dict[str, Any]], structured: bool) -> bool:
    if any(explicit_invalid(step, structured) for step in steps):
        return True
    if any("invalid arguments:" in observation(step).lower() for step in steps):
        return True
    repeated = Counter(call_key(step) for step in steps if action_name(step) != "run_test")
    return any(count >= 3 for count in repeated.values())


def is_environment_error(record: dict[str, Any]) -> bool:
    test_result = record.get("test_result")
    if not isinstance(test_result, dict):
        return True
    output = str(test_result.get("output", ""))
    return bool(ENVIRONMENT_RE.search(output))


def patch_attempt_failed(steps: list[dict[str, Any]], structured: bool) -> bool:
    patch_actions = {"apply_patch"} if structured else {"write_file"}
    attempts = [step for step in steps if action_name(step) in patch_actions]
    if not attempts:
        return False
    return all(
        "failed:" in observation(step).lower()
        or "refused:" in observation(step).lower()
        or "unknown action:" in observation(step).lower()
        for step in attempts
    )


def failure_reason(
    record: dict[str, Any], steps: list[dict[str, Any]], structured: bool
) -> str | None:
    if record.get("success") is True:
        return None
    if patch_attempt_failed(steps, structured):
        return "patch_failed"
    if is_environment_error(record):
        return "environment_error"
    if patch_generated(record):
        return "test_failed"
    if has_planning_error(steps, structured):
        return "planning_error"
    return "no_patch"


def protocol_valid(steps: list[dict[str, Any]], record: dict[str, Any], structured: bool) -> bool:
    if not steps or not isinstance(record.get("test_result"), dict):
        return False
    allowed = set(STRUCTURED_TOOLS) if structured else OLD_TOOLS
    if any(action_name(step) not in allowed for step in steps):
        return False
    if any(explicit_invalid(step, structured) for step in steps):
        return False
    return any(action_name(step) == "run_test" for step in steps)


def effective_trajectory(steps: list[dict[str, Any]], record: dict[str, Any], structured: bool) -> bool:
    if not protocol_valid(steps, record, structured):
        return False
    executed_tests = [
        step for step in steps
        if action_name(step) == "run_test"
        and "run_test failed: invalid arguments:" not in observation(step).lower()
        and "run_test refused" not in observation(step).lower()
    ]
    return bool(executed_tests)


def analyze_record(record: dict[str, Any], source: Path, structured: bool) -> dict[str, Any]:
    raw_steps = record.get("steps")
    steps = [step for step in raw_steps if isinstance(step, dict)] if isinstance(raw_steps, list) else []
    frequency = Counter(action_name(step) for step in steps)
    invalid_count = sum(explicit_invalid(step, structured) for step in steps)
    repair_count = sum(step.get("repair_applied") is True for step in steps)
    generated = patch_generated(record)
    reason = failure_reason(record, steps, structured)
    return {
        "instance_id": str(record.get("instance_id") or record.get("task_id") or source.stem),
        "source_file": source.name,
        "success": record.get("success") is True,
        "patch_generated": generated,
        "trajectory_length": len(steps),
        "tool_frequency": {tool: frequency[tool] for tool in STRUCTURED_TOOLS},
        "invalid_action_count": invalid_count,
        "repair_applied_count": repair_count,
        "protocol_valid": protocol_valid(steps, record, structured),
        "effective_trajectory": effective_trajectory(steps, record, structured),
        "failure_reason": reason,
    }


def rounded_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def analyze_directory(directory: Path, structured: bool) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    input_errors: list[dict[str, str]] = []
    for path in sorted(directory.glob("task_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("top-level JSON value must be an object")
            tasks.append(analyze_record(value, path, structured))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            input_errors.append({"source_file": path.name, "error": f"{type(exc).__name__}: {exc}"})

    total = len(tasks)
    total_steps = sum(task["trajectory_length"] for task in tasks)
    successes = sum(task["success"] for task in tasks)
    patches = sum(task["patch_generated"] for task in tasks)
    valid = sum(task["protocol_valid"] for task in tasks)
    effective = sum(task["effective_trajectory"] for task in tasks)
    invalid_actions = sum(task["invalid_action_count"] for task in tasks)
    repairs = sum(task["repair_applied_count"] for task in tasks)
    frequencies = Counter()
    reasons = Counter()
    for task in tasks:
        frequencies.update(task["tool_frequency"])
        if task["failure_reason"]:
            reasons[task["failure_reason"]] += 1
    return {
        "trajectory_count": total,
        "success_count": successes,
        "success_rate": rounded_ratio(successes, total),
        "patch_count": patches,
        "patch_generation_rate": rounded_ratio(patches, total),
        "tool_frequency": {tool: frequencies[tool] for tool in STRUCTURED_TOOLS},
        "average_trajectory_length": round(total_steps / total, 4) if total else 0.0,
        "invalid_action_count": invalid_actions,
        "invalid_action_rate": rounded_ratio(invalid_actions, total_steps),
        "repair_applied_count": repairs,
        "protocol_valid_trajectory_count": valid,
        "protocol_valid_trajectory_rate": rounded_ratio(valid, total),
        "effective_trajectory_count": effective,
        "effective_trajectory_rate": rounded_ratio(effective, total),
        "failure_taxonomy": {reason: reasons[reason] for reason in FAILURE_TAXONOMY},
        "tasks": tasks,
        "input_error_count": len(input_errors),
        "input_errors": input_errors,
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def comparison_rows(old: dict[str, Any], structured: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"metric": "success_rate", "old_free_action": old["success_rate"], "structured_tool": structured["success_rate"]},
        {"metric": "patch_generation_rate", "old_free_action": old["patch_generation_rate"], "structured_tool": structured["patch_generation_rate"]},
        {"metric": "invalid_action_rate", "old_free_action": old["invalid_action_rate"], "structured_tool": structured["invalid_action_rate"]},
        {"metric": "protocol_valid_trajectory_rate", "old_free_action": old["protocol_valid_trajectory_rate"], "structured_tool": structured["protocol_valid_trajectory_rate"]},
        {"metric": "effective_trajectory_rate", "old_free_action": old["effective_trajectory_rate"], "structured_tool": structured["effective_trajectory_rate"]},
        {"metric": "average_trajectory_length", "old_free_action": old["average_trajectory_length"], "structured_tool": structured["average_trajectory_length"]},
    ]


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def print_comparison(rows: list[dict[str, Any]]) -> None:
    print("\n| Metric | Old free-action Agent | Structured Tool Agent |")
    print("|---|---:|---:|")
    rate_metrics = {"success_rate", "patch_generation_rate", "invalid_action_rate", "protocol_valid_trajectory_rate", "effective_trajectory_rate"}
    for row in rows:
        old = pct(row["old_free_action"]) if row["metric"] in rate_metrics else str(row["old_free_action"])
        new = pct(row["structured_tool"]) if row["metric"] in rate_metrics else str(row["structured_tool"])
        print(f"| {row['metric']} | {old} | {new} |")


def main() -> int:
    args = parse_args()
    structured_dir = require_data_path(args.structured_dir, "--structured-dir")
    old_dir = require_data_path(args.old_dir, "--old-dir")
    output = require_data_path(args.output, "--output")
    for directory in (structured_dir, old_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"trajectory directory not found: {directory}")

    structured = analyze_directory(structured_dir, structured=True)
    old = analyze_directory(old_dir, structured=False)
    rows = comparison_rows(old, structured)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structured_dir": str(structured_dir),
        "old_free_action_dir": str(old_dir),
        "structured_baseline": structured,
        "old_free_action_baseline": old,
        "comparison_table": rows,
        "step8_validation": {
            "invalid_action_rate_reduction": round(old["invalid_action_rate"] - structured["invalid_action_rate"], 4),
            "protocol_valid_trajectory_rate_gain": round(structured["protocol_valid_trajectory_rate"] - old["protocol_valid_trajectory_rate"], 4),
            "effective_trajectory_rate_gain": round(structured["effective_trajectory_rate"] - old["effective_trajectory_rate"], 4),
            "patch_generation_rate_gain": round(structured["patch_generation_rate"] - old["patch_generation_rate"], 4),
            "success_rate_gain": round(structured["success_rate"] - old["success_rate"], 4),
        },
        "metric_definitions": {
            "patch_generated": "patch is non-empty and not '(no changes)'",
            "invalid_action_rate": "invalid or repaired action steps divided by all steps",
            "protocol_valid_trajectory": "non-empty trajectory with only version-registered actions, no invalid markers, a test_result, and at least one run_test call",
            "effective_trajectory": "protocol-valid trajectory with at least one run_test invocation that reached the test process rather than failing argument validation or command allowlisting",
            "failure_taxonomy": "one primary reason per failed trajectory, prioritized as patch_failed, environment_error, test_failed, planning_error, no_patch",
        },
    }
    write_json_atomic(output, report)
    print_comparison(rows)
    print("\nStructured tool frequency:")
    print(json.dumps(structured["tool_frequency"], indent=2))
    print("Structured failure taxonomy:")
    print(json.dumps(structured["failure_taxonomy"], indent=2))
    print(f"output={output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
