#!/usr/bin/env python3
"""Audit and analyze Base-vs-SFT held-out SWE-bench Lite rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools.schema import ACTION_ALIASES, TOOL_NAMES, TOOL_SCHEMAS, normalize_action_name
from evaluation.patch_proxy import evaluate_patch


DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
SUBSET_DIR = DATA_ROOT / "swebench" / "subset"
DEV_MANIFEST = DATA_ROOT / "swebench_dev" / "split_manifest.json"
DATASET_DIR = DATA_ROOT / "datasets" / "sft_v2"
REPOS_DIR = DATA_ROOT / "swebench" / "repos"
BASE_DIR = DATA_ROOT / "trajectories" / "heldout_base"
SFT_DIR = DATA_ROOT / "trajectories" / "heldout_sft"
LOGS_DIR = DATA_ROOT / "logs"
PROXY_WORKSPACES = DATA_ROOT / "rollouts" / "patch_proxy"
TOOLS = ("list_files", "search_code", "read_file", "apply_patch", "run_test")
PHASES = ("LOCATE", "UNDERSTAND", "MODIFY", "VERIFY")
TRANSITIONS = (
    "LOCATE -> UNDERSTAND",
    "UNDERSTAND -> MODIFY",
    "MODIFY -> VERIFY",
    "VERIFY -> UNDERSTAND",
)
ACTION_FAILURES = (
    "invalid_json",
    "unknown_tool",
    "schema_error",
    "missing_argument",
    "wrong_path",
    "file_not_found",
    "search_no_result",
    "patch_parse_error",
    "git_apply_check_failed",
    "controller_blocked",
    "generation_truncated",
    "test_environment_error",
    "other_tool_error",
)
TASK_FAILURES = (
    "protocol_error",
    "planning_error",
    "patch_generation_error",
    "patch_application_error",
    "environment_error",
    "generation_truncation",
    "no_patch",
    "unknown",
)
ENVIRONMENT_RE = re.compile(
    r"(modulenotfounderror|no module named|importerror|cannot import name|"
    r"missing dependency|hypothesis|setuptools_scm|pkg_resources|"
    r"error loading conftest|error while loading|django.*import|"
    r"module 'collections' has no attribute|python 3\.1[3-9]|"
    r"command not found|no such file or directory.*python|"
    r"broken installation|infrastructure failed|model backend failed)",
    re.I,
)
SCHEMA_BY_TOOL = {str(item["name"]): item["parameters"] for item in TOOL_SCHEMAS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--subset-dir", type=Path, default=SUBSET_DIR)
    parser.add_argument("--dev-manifest", type=Path, default=DEV_MANIFEST)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--repos-dir", type=Path, default=REPOS_DIR)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--sft-dir", type=Path, default=SFT_DIR)
    parser.add_argument("--proxy-workspaces", type=Path, default=PROXY_WORKSPACES)
    parser.add_argument("--split-audit-output", type=Path, default=LOGS_DIR / "heldout_split_audit.json")
    parser.add_argument("--output-json", type=Path, default=LOGS_DIR / "heldout_agent_ablation.json")
    parser.add_argument("--output-markdown", type=Path, default=LOGS_DIR / "heldout_agent_ablation.md")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def require_data_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be under {root}; got {resolved}")
    return resolved


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def selected_tasks(subset_dir: Path, count: int) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(subset_dir.glob("task_*.json"))[:count]
    if len(paths) != count:
        raise RuntimeError(f"expected {count} task files, found {len(paths)}")
    tasks = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    required = {"instance_id", "repo", "base_commit", "problem_statement"}
    for path, task in tasks:
        missing = required - task.keys()
        if missing:
            raise RuntimeError(f"{path} is missing {sorted(missing)}")
    return tasks


def build_split_audit(
    tasks: Sequence[tuple[Path, dict[str, Any]]],
    dataset_dir: Path,
    dev_manifest_path: Path,
) -> dict[str, Any]:
    train = load_jsonl(dataset_dir / "train.jsonl")
    validation = load_jsonl(dataset_dir / "validation.jsonl")
    sft_ids = {str(item["instance_id"]) for item in [*train, *validation]}
    sft_repos = {str(item["repo"]) for item in [*train, *validation]}
    heldout_ids = [str(task["instance_id"]) for _, task in tasks]
    heldout_repos = sorted({str(task["repo"]) for _, task in tasks})
    overlap = sorted(set(heldout_ids) & sft_ids)
    repo_overlap = sorted(set(heldout_repos) & sft_repos)
    dev_manifest = json.loads(dev_manifest_path.read_text(encoding="utf-8"))
    audit = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "heldout_source": "SWE-bench Lite fixed first 10 subset tasks",
        "sft_source": {
            "dataset": dev_manifest.get("dataset_name"),
            "split": dev_manifest.get("dataset_split"),
            "revision": dev_manifest.get("dataset_revision"),
        },
        "heldout_task_count": len(tasks),
        "sft_instance_count": len(sft_ids),
        "task_order": heldout_ids,
        "tasks": [
            {
                "position": index,
                "task_file": path.name,
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "base_commit": task["base_commit"],
            }
            for index, (path, task) in enumerate(tasks, 1)
        ],
        "instance_overlap": overlap,
        "instance_overlap_count": len(overlap),
        "repo_overlap": repo_overlap,
        "repo_overlap_count": len(repo_overlap),
        "heldout_repositories": heldout_repos,
        "sft_repositories": sorted(sft_repos),
        "passed": not overlap,
    }
    if overlap:
        raise RuntimeError(f"held-out/SFT instance leakage: {overlap}")
    return audit


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_action_arguments(step: Mapping[str, Any]) -> dict[str, Any] | None:
    action = step.get("action")
    if not isinstance(action, str):
        return None
    start = action.find("(")
    if start < 0 or not action.endswith(")"):
        return None
    try:
        value = json.loads(action[start + 1 : -1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def schema_status(tool: str | None, arguments: dict[str, Any] | None) -> tuple[bool, bool]:
    if tool not in SCHEMA_BY_TOOL or arguments is None:
        return False, False
    schema = SCHEMA_BY_TOOL[tool]
    required = set(schema.get("required", []))
    missing = bool(required - arguments.keys())
    if missing:
        return False, True
    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False and arguments.keys() - properties.keys():
        return False, False
    for name, value in arguments.items():
        spec = properties.get(name)
        if not isinstance(spec, dict):
            return False, False
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            return False, False
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return False, False
        if isinstance(value, int):
            if "minimum" in spec and value < spec["minimum"]:
                return False, False
            if "maximum" in spec and value > spec["maximum"]:
                return False, False
    return True, False


def proposed_tool(step: Mapping[str, Any]) -> str | None:
    value = step.get("model_proposed_tool")
    if not isinstance(value, str) or not value.strip():
        return None
    return normalize_action_name(value)


def executed_tool(step: Mapping[str, Any]) -> str | None:
    value = step.get("executed_tool")
    return str(value) if isinstance(value, str) and value in TOOL_NAMES else None


def is_model_step(step: Mapping[str, Any]) -> bool:
    if step.get("intervention_reason") == "final_verification":
        return False
    return proposed_tool(step) is not None or step.get("action", "").startswith("invalid(")


def is_nonempty_patch(record: Mapping[str, Any]) -> bool:
    patch = str(record.get("patch", "")).strip()
    return bool(patch and patch != "(no changes)")


def classify_action_failure(step: Mapping[str, Any]) -> str | None:
    observation = str(step.get("observation", ""))
    lowered = observation.lower()
    proposed = proposed_tool(step)
    executed = executed_tool(step)
    arguments = parse_action_arguments(step)
    canonical = ACTION_ALIASES.get(proposed or "", proposed)
    valid_schema, missing = schema_status(canonical, arguments)

    if step.get("generation_truncated") is True:
        return "generation_truncated"
    if step.get("action", "").startswith("invalid("):
        if proposed and proposed not in TOOL_NAMES and proposed not in ACTION_ALIASES:
            return "unknown_tool"
        return "invalid_json"
    if proposed and proposed not in TOOL_NAMES and proposed not in ACTION_ALIASES:
        return "unknown_tool"
    if missing:
        return "missing_argument"
    if canonical in TOOL_NAMES and not valid_schema:
        return "schema_error"
    if "planning controller blocked" in lowered:
        return "controller_blocked"
    if executed == "search_code" and lowered.strip() == "no matches":
        return "search_no_result"
    if executed == "run_test" and step.get("tool_success") is False and ENVIRONMENT_RE.search(observation):
        return "test_environment_error"
    if step.get("tool_success") is not False:
        return None
    if "path escapes workspace" in lowered:
        return "wrong_path"
    if "path does not exist" in lowered or "not a file" in lowered or "target is not an existing file" in lowered:
        return "file_not_found"
    if executed == "apply_patch":
        if "no valid file headers" in lowered or "creating or deleting files is not allowed" in lowered:
            return "patch_parse_error"
        if "apply_patch failed:" in lowered:
            return "git_apply_check_failed"
    return "other_tool_error"


def repeated_exploration(steps: Sequence[Mapping[str, Any]]) -> int:
    tools = [
        executed_tool(step) for step in steps if executed_tool(step) is not None
    ]
    return sum(
        current == previous and current in {"list_files", "search_code"}
        for previous, current in zip(tools, tools[1:])
    )


def phase_summary(steps: Sequence[Mapping[str, Any]]) -> tuple[dict[str, bool], Counter[str]]:
    autonomous = [
        step for step in steps if step.get("intervention_reason") != "final_verification"
    ]
    seen = {"LOCATE"} if autonomous else set()
    transitions: Counter[str] = Counter()
    for step in autonomous:
        previous = step.get("previous_phase")
        current = step.get("phase")
        if isinstance(previous, str):
            seen.add(previous)
        if isinstance(current, str):
            seen.add(current)
        if isinstance(previous, str) and isinstance(current, str) and previous != current:
            transitions[f"{previous} -> {current}"] += 1
    return {phase: phase in seen for phase in PHASES}, transitions


def host_test_classification(record: Mapping[str, Any]) -> str:
    result = record.get("host_test_result") or record.get("test_result")
    if not isinstance(result, dict):
        return "not_executed"
    if result.get("passed") is True:
        return "passed"
    output = str(result.get("output", ""))
    return "environment_error" if ENVIRONMENT_RE.search(output) else "failed_non_environment"


def primary_failure(
    *,
    action_failures: Counter[str],
    reached: Mapping[str, bool],
    patch_attempt: bool,
    valid_apply_patch: bool,
    final_patch: bool,
    host_test: str,
    interventions: int,
    repetitions: int,
) -> tuple[str, list[str]]:
    secondary: set[str] = set()
    protocol = sum(action_failures[name] for name in ("invalid_json", "unknown_tool", "schema_error", "missing_argument"))
    if protocol:
        secondary.add("protocol_error")
    if action_failures["generation_truncated"]:
        secondary.add("generation_truncation")
    if action_failures["patch_parse_error"]:
        secondary.add("patch_generation_error")
    if action_failures["git_apply_check_failed"] or (patch_attempt and not valid_apply_patch):
        secondary.add("patch_application_error")
    if host_test == "environment_error":
        secondary.add("environment_error")
    if interventions or repetitions or not reached["MODIFY"]:
        secondary.add("planning_error")
    if not final_patch:
        secondary.add("no_patch")

    precedence = (
        "protocol_error",
        "generation_truncation",
        "patch_generation_error",
        "patch_application_error",
        "environment_error",
        "planning_error",
        "no_patch",
    )
    primary = next((name for name in precedence if name in secondary), "unknown")
    return primary, sorted(secondary)


def analyze_task(
    task: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    repos_dir: Path,
    proxy_workspaces: Path,
) -> dict[str, Any]:
    steps = [item for item in record.get("steps", []) if isinstance(item, dict)]
    model_steps = [step for step in steps if is_model_step(step)]
    json_valid = [
        not str(step.get("action", "")).startswith("invalid(") for step in model_steps
    ]
    registered = [
        proposed_tool(step) in TOOL_NAMES for step in model_steps
    ]
    schema_valid: list[bool] = []
    action_failures: Counter[str] = Counter()
    failure_details: list[dict[str, Any]] = []
    for step in model_steps:
        proposed = proposed_tool(step)
        canonical = ACTION_ALIASES.get(proposed or "", proposed)
        valid, _ = schema_status(canonical, parse_action_arguments(step))
        schema_valid.append(valid)
    for index, step in enumerate(steps):
        failure = classify_action_failure(step)
        if failure:
            action_failures[failure] += 1
            failure_details.append(
                {
                    "trajectory_step_index": index,
                    "category": failure,
                    "model_proposed_tool": step.get("model_proposed_tool"),
                    "executed_tool": step.get("executed_tool"),
                    "tool_success": step.get("tool_success"),
                    "observation": str(step.get("observation", ""))[:1000],
                }
            )

    patch_steps = [step for step in steps if executed_tool(step) == "apply_patch"]
    successful_patch_steps = [
        step for step in patch_steps if step.get("tool_success") is True
    ]
    reached, transitions = phase_summary(steps)
    interventions = sum(step.get("controller_intervened") is True for step in steps)
    proposed_steps = [step for step in model_steps if proposed_tool(step) is not None]
    agreements = sum(proposed_tool(step) == executed_tool(step) for step in proposed_steps)
    repetitions = repeated_exploration(steps)
    patch = str(record.get("patch", ""))
    proxy = evaluate_patch(
        instance_id=str(task["instance_id"]),
        repo=str(task["repo"]),
        base_commit=str(task["base_commit"]),
        patch=patch,
        repos_dir=repos_dir,
        workspaces_dir=proxy_workspaces,
        tokenizer=tokenizer,
    )
    host_test = host_test_classification(record)
    primary, secondary = primary_failure(
        action_failures=action_failures,
        reached=reached,
        patch_attempt=bool(patch_steps),
        valid_apply_patch=bool(successful_patch_steps),
        final_patch=is_nonempty_patch(record),
        host_test=host_test,
        interventions=interventions,
        repetitions=repetitions,
    )
    generated = [
        int(step["generated_tokens"])
        for step in model_steps
        if isinstance(step.get("generated_tokens"), int)
    ]
    return {
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "base_commit": task["base_commit"],
        "trajectory_steps": len(steps),
        "model_action_count": len(model_steps),
        "tool_invocation_count": sum(executed_tool(step) is not None for step in steps),
        "tool_invocation_success_count": sum(
            executed_tool(step) is not None and step.get("tool_success") is True
            for step in steps
        ),
        "json_valid_count": sum(json_valid),
        "registered_tool_count": sum(registered),
        "argument_schema_valid_count": sum(schema_valid),
        "controller_intervention_count": interventions,
        "controller_agreement_count": agreements,
        "controller_agreement_denominator": len(proposed_steps),
        "repeated_exploration_count": repetitions,
        "generated_tokens": sum(generated),
        "generation_count": len(generated),
        "generation_truncation_count": sum(
            step.get("generation_truncated") is True for step in model_steps
        ),
        "phase_reached": reached,
        "phase_transitions": {name: transitions[name] for name in TRANSITIONS},
        "patch_attempt": bool(patch_steps),
        "apply_patch_attempts": len(patch_steps),
        "apply_patch_successes": len(successful_patch_steps),
        "apply_patch_failures": len(patch_steps) - len(successful_patch_steps),
        "valid_apply_patch": bool(successful_patch_steps),
        "final_patch": is_nonempty_patch(record),
        "patch_proxy": proxy,
        "host_test_result": host_test,
        "rollout_test_passed": record.get("rollout_test_passed"),
        "benchmark_resolved": None,
        "action_failure_taxonomy": {name: action_failures[name] for name in ACTION_FAILURES},
        "action_failure_details": failure_details,
        "primary_failure_reason": primary,
        "secondary_failures": secondary,
    }


def ratio(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def aggregate(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(tasks)
    actions = sum(int(task["model_action_count"]) for task in tasks)
    steps = sum(int(task["trajectory_steps"]) for task in tasks)
    proposed = sum(int(task["controller_agreement_denominator"]) for task in tasks)
    exploration_repeats = sum(int(task["repeated_exploration_count"]) for task in tasks)
    generations = sum(int(task["generation_count"]) for task in tasks)
    generated_tokens = sum(int(task["generated_tokens"]) for task in tasks)
    tool_invocations = sum(int(task["tool_invocation_count"]) for task in tasks)
    successful_tool_invocations = sum(
        int(task["tool_invocation_success_count"]) for task in tasks
    )
    patch_attempts = sum(int(task["apply_patch_attempts"]) for task in tasks)
    patch_successes = sum(int(task["apply_patch_successes"]) for task in tasks)
    action_failures: Counter[str] = Counter()
    task_failures: Counter[str] = Counter()
    host_tests: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    for task in tasks:
        action_failures.update(task["action_failure_taxonomy"])
        task_failures[str(task["primary_failure_reason"])] += 1
        host_tests[str(task["host_test_result"])] += 1
        transitions.update(task["phase_transitions"])
    interventions = sum(int(task["controller_intervention_count"]) for task in tasks)
    agreements = sum(int(task["controller_agreement_count"]) for task in tasks)
    truncations = sum(int(task["generation_truncation_count"]) for task in tasks)
    final_patches = sum(bool(task["final_patch"]) for task in tasks)
    applicable = sum(bool(task["patch_proxy"]["proxy_patch_applicable"]) for task in tasks)
    valid_tasks = sum(bool(task["valid_apply_patch"]) for task in tasks)
    attempted_tasks = sum(bool(task["patch_attempt"]) for task in tasks)
    return {
        "trajectory_count": count,
        "model_action_count": actions,
        "json_valid_action_rate": ratio(sum(int(task["json_valid_count"]) for task in tasks), actions),
        "registered_tool_rate": ratio(sum(int(task["registered_tool_count"]) for task in tasks), actions),
        "argument_schema_valid_rate": ratio(sum(int(task["argument_schema_valid_count"]) for task in tasks), actions),
        "tool_invocation_success_rate": ratio(successful_tool_invocations, tool_invocations),
        "tool_invocation_count": tool_invocations,
        "tool_invocation_success_count": successful_tool_invocations,
        "controller_intervention_count": interventions,
        "controller_intervention_rate": ratio(interventions, steps),
        "controller_agreement_rate": ratio(agreements, proposed),
        "controller_agreement_denominator": proposed,
        "repeated_exploration_count": exploration_repeats,
        "repeated_exploration_rate": ratio(exploration_repeats, actions),
        "repeated_exploration_task_rate": ratio(
            sum(int(task["repeated_exploration_count"]) > 0 for task in tasks), count
        ),
        "average_steps": round(steps / count, 4) if count else 0.0,
        "average_generated_tokens": round(generated_tokens / generations, 4) if generations else 0.0,
        "average_generated_tokens_per_trajectory": round(generated_tokens / count, 4) if count else 0.0,
        "generation_truncation_count": truncations,
        "generation_truncation_rate": ratio(truncations, generations),
        "phase_progression": {
            f"reached_{phase}": {
                "count": sum(bool(task["phase_reached"][phase]) for task in tasks),
                "rate": ratio(sum(bool(task["phase_reached"][phase]) for task in tasks), count),
            }
            for phase in PHASES
        },
        "phase_transitions": {name: transitions[name] for name in TRANSITIONS},
        "patch_metrics": {
            "patch_attempt_rate": ratio(attempted_tasks, count),
            "valid_apply_patch_execution_rate": ratio(valid_tasks, count),
            "apply_patch_attempts": patch_attempts,
            "apply_patch_successful_executions": patch_successes,
            "apply_patch_failed_executions": patch_attempts - patch_successes,
            "apply_patch_success_rate": ratio(patch_successes, patch_attempts),
            "final_patch_generation_rate": ratio(final_patches, count),
            "proxy_patch_applicable_rate": ratio(applicable, count),
            "proxy_patch_applicable_rate_given_patch": ratio(applicable, final_patches),
            "average_modified_files": round(
                sum(int(task["patch_proxy"]["modified_file_count"]) for task in tasks) / count, 4
            ) if count else 0.0,
            "average_patch_size_tokens": round(
                sum(int(task["patch_proxy"]["patch_size_tokens"]) for task in tasks) / count, 4
            ) if count else 0.0,
            "average_patch_size_lines": round(
                sum(int(task["patch_proxy"]["patch_size_lines"]) for task in tasks) / count, 4
            ) if count else 0.0,
        },
        "action_failure_taxonomy": {name: action_failures[name] for name in ACTION_FAILURES},
        "failure_taxonomy": {name: task_failures[name] for name in TASK_FAILURES},
        "host_test_results": dict(sorted(host_tests.items())),
        "benchmark_resolved": None,
        "tasks": list(tasks),
    }


def load_condition(
    directory: Path,
    tasks: Sequence[tuple[Path, dict[str, Any]]],
    *,
    tokenizer: Any,
    repos_dir: Path,
    proxy_workspaces: Path,
) -> dict[str, Any]:
    analyzed: list[dict[str, Any]] = []
    for task_path, task in tasks:
        trajectory_path = directory / task_path.name
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"missing trajectory: {trajectory_path}")
        record = json.loads(trajectory_path.read_text(encoding="utf-8"))
        if record.get("instance_id") != task["instance_id"]:
            raise RuntimeError(f"task/trajectory mismatch: {trajectory_path}")
        if record.get("benchmark_resolved") is not None:
            raise RuntimeError(f"benchmark_resolved must be null: {trajectory_path}")
        analyzed.append(
            analyze_task(
                task,
                record,
                tokenizer=tokenizer,
                repos_dir=repos_dir,
                proxy_workspaces=proxy_workspaces,
            )
        )
    return aggregate(analyzed)


def boolean_outcome(base: bool, sft: bool) -> str:
    if not base and sft:
        return "improved"
    if base and not sft:
        return "regressed"
    return "same"


def pairwise(base: Mapping[str, Any], sft: Mapping[str, Any]) -> list[dict[str, Any]]:
    base_tasks = {str(item["instance_id"]): item for item in base["tasks"]}
    sft_tasks = {str(item["instance_id"]): item for item in sft["tasks"]}
    rows: list[dict[str, Any]] = []
    for instance_id in base_tasks:
        left = base_tasks[instance_id]
        right = sft_tasks[instance_id]
        left_rate = ratio(left["controller_intervention_count"], left["trajectory_steps"])
        right_rate = ratio(right["controller_intervention_count"], right["trajectory_steps"])
        controller_outcome = (
            "improved" if right_rate < left_rate else "regressed" if right_rate > left_rate else "same"
        )
        objective = {
            "reached_modify": boolean_outcome(
                left["phase_reached"]["MODIFY"], right["phase_reached"]["MODIFY"]
            ),
            "patch_attempt": boolean_outcome(left["patch_attempt"], right["patch_attempt"]),
            "valid_apply_patch": boolean_outcome(
                left["valid_apply_patch"], right["valid_apply_patch"]
            ),
            "final_patch": boolean_outcome(left["final_patch"], right["final_patch"]),
            "proxy_patch_applicable": boolean_outcome(
                left["patch_proxy"]["proxy_patch_applicable"],
                right["patch_proxy"]["proxy_patch_applicable"],
            ),
            "controller_intervention_rate": controller_outcome,
        }
        rows.append(
            {
                "instance_id": instance_id,
                "base": {
                    "reached_modify": left["phase_reached"]["MODIFY"],
                    "patch_attempt": left["patch_attempt"],
                    "valid_apply_patch": left["valid_apply_patch"],
                    "final_patch": left["final_patch"],
                    "proxy_patch_applicable": left["patch_proxy"]["proxy_patch_applicable"],
                    "controller_intervention_rate": left_rate,
                },
                "sft": {
                    "reached_modify": right["phase_reached"]["MODIFY"],
                    "patch_attempt": right["patch_attempt"],
                    "valid_apply_patch": right["valid_apply_patch"],
                    "final_patch": right["final_patch"],
                    "proxy_patch_applicable": right["patch_proxy"]["proxy_patch_applicable"],
                    "controller_intervention_rate": right_rate,
                },
                "objective_metric_outcomes": objective,
            }
        )
    return rows


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def delta(base: float, sft: float) -> str:
    return f"{100 * (sft - base):+.2f} pp"


def next_stage(base: Mapping[str, Any], sft: Mapping[str, Any]) -> tuple[str, list[bool]]:
    gates = [
        sft["controller_intervention_rate"] < base["controller_intervention_rate"],
        sft["patch_metrics"]["final_patch_generation_rate"] > base["patch_metrics"]["final_patch_generation_rate"],
        sft["patch_metrics"]["proxy_patch_applicable_rate"] > base["patch_metrics"]["proxy_patch_applicable_rate"],
    ]
    if sum(gates) >= 2:
        return "PROJECT_WRAP_UP", gates
    patch_failures = (
        sft["action_failure_taxonomy"]["patch_parse_error"]
        + sft["action_failure_taxonomy"]["git_apply_check_failed"]
        + sft["failure_taxonomy"]["patch_generation_error"]
        + sft["failure_taxonomy"]["patch_application_error"]
    )
    policy_failures = (
        sft["action_failure_taxonomy"]["controller_blocked"]
        + sft["failure_taxonomy"]["planning_error"]
        + sft["failure_taxonomy"]["protocol_error"]
        + sft["failure_taxonomy"]["no_patch"]
    )
    return (
        "PATCH_ARGUMENT_FAILURE_ANALYSIS"
        if patch_failures > policy_failures
        else "POLICY_FAILURE_ANALYSIS",
        gates,
    )


def experiment_conclusion(base: Mapping[str, Any], sft: Mapping[str, Any]) -> str:
    controller_better = sft["controller_intervention_rate"] < base["controller_intervention_rate"]
    modify_better = (
        sft["phase_progression"]["reached_MODIFY"]["rate"]
        > base["phase_progression"]["reached_MODIFY"]["rate"]
    )
    patch_better = (
        sft["patch_metrics"]["valid_apply_patch_execution_rate"]
        > base["patch_metrics"]["valid_apply_patch_execution_rate"]
    )
    if controller_better and modify_better and patch_better:
        return (
            "Structured SFT improved autonomous agent policy behavior on the measured "
            "held-out SWE-bench Lite tasks."
        )
    if patch_better:
        return (
            "Structured SFT improved protocol compliance and produced a small positive "
            "executable-patch signal, but did not broadly improve autonomous policy: "
            "controller dependency increased and MODIFY reach decreased."
        )
    return (
        "Structured SFT did not improve the measured autonomous held-out rollout behavior; "
        "this is a negative transfer result despite static-policy gains."
    )


def top_failure_delta(base: Mapping[str, Any], sft: Mapping[str, Any]) -> str:
    differences = {
        name: int(sft["action_failure_taxonomy"][name]) - int(base["action_failure_taxonomy"][name])
        for name in ACTION_FAILURES
    }
    name = max(differences, key=lambda item: (abs(differences[item]), item))
    value = differences[name]
    return f"{name} ({value:+d} actions versus Base)"


def render_markdown(report: Mapping[str, Any]) -> str:
    base = report["conditions"]["base"]
    sft = report["conditions"]["sft"]
    keys = [
        ("JSON valid action", "json_valid_action_rate"),
        ("Registered tool", "registered_tool_rate"),
        ("Argument schema valid", "argument_schema_valid_rate"),
        ("Controller intervention", "controller_intervention_rate"),
        ("Controller agreement", "controller_agreement_rate"),
        ("Repeated exploration", "repeated_exploration_rate"),
        ("Generation truncation", "generation_truncation_rate"),
    ]
    rows = ["| Metric | Base | SFT | Delta |", "|---|---:|---:|---:|"]
    for label, key in keys:
        rows.append(f"| {label} | {pct(base[key])} | {pct(sft[key])} | {delta(base[key], sft[key])} |")
    for label, key in (
        ("Reached MODIFY", "MODIFY"),
        ("Reached VERIFY", "VERIFY"),
    ):
        left = base["phase_progression"][f"reached_{key}"]["rate"]
        right = sft["phase_progression"][f"reached_{key}"]["rate"]
        rows.append(f"| {label} | {pct(left)} | {pct(right)} | {delta(left, right)} |")
    for label, key in (
        ("Patch attempt", "patch_attempt_rate"),
        ("Valid apply_patch task", "valid_apply_patch_execution_rate"),
        ("Final patch generation", "final_patch_generation_rate"),
        ("Proxy patch applicable", "proxy_patch_applicable_rate"),
        ("apply_patch execution success", "apply_patch_success_rate"),
    ):
        left = base["patch_metrics"][key]
        right = sft["patch_metrics"][key]
        rows.append(f"| {label} | {pct(left)} | {pct(right)} | {delta(left, right)} |")

    rq1 = (
        f"Controller intervention changed from {pct(base['controller_intervention_rate'])} "
        f"to {pct(sft['controller_intervention_rate'])}; agreement changed from "
        f"{pct(base['controller_agreement_rate'])} to {pct(sft['controller_agreement_rate'])}."
    )
    rq2 = (
        f"MODIFY reach changed from {pct(base['phase_progression']['reached_MODIFY']['rate'])} "
        f"to {pct(sft['phase_progression']['reached_MODIFY']['rate'])}; patch attempts changed "
        f"from {pct(base['patch_metrics']['patch_attempt_rate'])} to "
        f"{pct(sft['patch_metrics']['patch_attempt_rate'])}."
    )
    rq3 = (
        f"apply_patch execution success changed from "
        f"{pct(base['patch_metrics']['apply_patch_success_rate'])} to "
        f"{pct(sft['patch_metrics']['apply_patch_success_rate'])}; final patch generation changed "
        f"from {pct(base['patch_metrics']['final_patch_generation_rate'])} to "
        f"{pct(sft['patch_metrics']['final_patch_generation_rate'])}; proxy applicability changed "
        f"from {pct(base['patch_metrics']['proxy_patch_applicable_rate'])} to "
        f"{pct(sft['patch_metrics']['proxy_patch_applicable_rate'])}."
    )
    transfer = (
        "Static gains transferred to at least one full-rollout patch metric."
        if (
            sft["patch_metrics"]["patch_attempt_rate"] > base["patch_metrics"]["patch_attempt_rate"]
            or sft["patch_metrics"]["valid_apply_patch_execution_rate"] > base["patch_metrics"]["valid_apply_patch_execution_rate"]
            or sft["patch_metrics"]["final_patch_generation_rate"] > base["patch_metrics"]["final_patch_generation_rate"]
        )
        else "No positive transfer to the measured full-rollout patch metrics was observed."
    )
    return (
        "# Held-out Agent Rollout Ablation\n\n"
        "The only model difference is whether the frozen Structured SFT v2.1 LoRA adapter is loaded. "
        "All benchmark_resolved values are null; host tests and patch applicability are not official SWE-bench resolution.\n\n"
        "## Core comparison\n\n"
        + "\n".join(rows)
        + "\n\n## Research questions\n\n"
        + f"- RQ1 — Planning Controller dependency: {rq1}\n"
        + f"- RQ2 — MODIFY progression: {rq2}\n"
        + f"- RQ3 — Executable patch reliability: {rq3}\n"
        + f"- RQ4 — Static-to-rollout transfer: {transfer}\n"
        + f"- RQ5 — Static execution decline diagnosis: the largest observed action-failure change was "
        + f"{top_failure_delta(base, sft)}. See action_failure_taxonomy and per-action details in the JSON report.\n\n"
        + "## Conclusion\n\n"
        + experiment_conclusion(base, sft) + "\n\n"
        + "## Environment limitation\n\n"
        + f"- Base host test results: {json.dumps(base['host_test_results'], sort_keys=True)}\n"
        + f"- SFT host test results: {json.dumps(sft['host_test_results'], sort_keys=True)}\n"
        + "- proxy_patch_applicable means only that git apply --check accepted the diff on a fresh clean base commit.\n"
        + "- No gold patch, test patch, FAIL_TO_PASS, or PASS_TO_PASS metadata was used by the held-out model policy.\n\n"
        + f"NEXT_STAGE = {report['next_stage']}\n"
    )


def main() -> int:
    args = parse_args()
    subset_dir = require_data_path(args.subset_dir, "--subset-dir")
    dataset_dir = require_data_path(args.dataset_dir, "--dataset-dir")
    dev_manifest = require_data_path(args.dev_manifest, "--dev-manifest")
    repos_dir = require_data_path(args.repos_dir, "--repos-dir")
    proxy_workspaces = require_data_path(args.proxy_workspaces, "--proxy-workspaces")
    split_output = require_data_path(args.split_audit_output, "--split-audit-output")
    tasks = selected_tasks(subset_dir, args.num_tasks)
    split_audit = build_split_audit(tasks, dataset_dir, dev_manifest)
    write_json_atomic(split_output, split_audit)
    print(f"split_audit={split_output} instance_overlap=0 repo_overlap={split_audit['repo_overlap']}")
    if args.audit_only:
        return 0

    base_dir = require_data_path(args.base_dir, "--base-dir")
    sft_dir = require_data_path(args.sft_dir, "--sft-dir")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        cache_dir=str(DATA_ROOT / "hf_cache" / "hub"),
        local_files_only=True,
        trust_remote_code=False,
    )
    base = load_condition(
        base_dir, tasks, tokenizer=tokenizer, repos_dir=repos_dir,
        proxy_workspaces=proxy_workspaces / "base",
    )
    sft = load_condition(
        sft_dir, tasks, tokenizer=tokenizer, repos_dir=repos_dir,
        proxy_workspaces=proxy_workspaces / "sft",
    )
    stage, gates = next_stage(base, sft)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "heldout_agent_rollout_ablation",
        "split_audit": split_audit,
        "control": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "tokenizer": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "same_tokenizer": True,
            "base_adapter": None,
            "sft_adapter": str(DATA_ROOT / "checkpoints" / "sft_v2" / "final_adapter"),
            "only_model_difference": "LoRA adapter loaded for SFT condition",
            "task_order_equal": True,
            "base_commits_equal": True,
            "max_steps": 12,
            "max_action_tokens": 1024,
            "do_sample": False,
            "test_command_policy": "metadata_free_repository_level",
            "gold_test_metadata_exposed": False,
            "code_fingerprints": {
                path: sha256_file(PROJECT_ROOT / path)
                for path in (
                    "agent/prompting.py",
                    "agent/planner.py",
                    "agent/tools/schema.py",
                    "rollout/policy.py",
                    "rollout/runner.py",
                )
            },
        },
        "conditions": {"base": base, "sft": sft},
        "pairwise_task_comparison": pairwise(base, sft),
        "stage_gate": {
            "controller_intervention_rate_decreased": gates[0],
            "final_patch_generation_rate_increased": gates[1],
            "proxy_patch_applicable_rate_increased": gates[2],
            "passed_count": sum(gates),
            "required_count": 2,
        },
        "benchmark_resolved": None,
        "conclusion": experiment_conclusion(base, sft),
        "next_stage": stage,
    }
    output_json = require_data_path(args.output_json, "--output-json")
    output_markdown = require_data_path(args.output_markdown, "--output-markdown")
    write_json_atomic(output_json, report)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"json={output_json}")
    print(f"markdown={output_markdown}")
    print(f"NEXT_STAGE={stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
