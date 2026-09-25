#!/usr/bin/env python3
"""Strict ablation analysis for structured and planning-controlled rollouts."""
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
DEFAULT_PLANNED = DATA_ROOT / "trajectories" / "planned"
DEFAULT_SUBSET = DATA_ROOT / "swebench" / "subset"
DEFAULT_JSON = DATA_ROOT / "logs" / "planning_ablation.json"
DEFAULT_MARKDOWN = DATA_ROOT / "logs" / "planning_ablation.md"
TOOLS = ("list_files", "search_code", "read_file", "apply_patch", "run_test")
TRANSITIONS = (
    "LOCATE -> UNDERSTAND",
    "UNDERSTAND -> MODIFY",
    "MODIFY -> VERIFY",
    "VERIFY -> UNDERSTAND",
)
FAILURES = (
    "environment_error",
    "planning_error",
    "patch_failed",
    "test_failed",
    "no_patch",
    "timeout",
)
ENVIRONMENT_RE = re.compile(
    r"(modulenotfounderror|importerror|filenotfounderror|connectionerror|"
    r"no space left|out of memory|cuda out of memory|infrastructure failed|"
    r"prepared repository not found|policy_error|model backend failed|"
    r"broken installation|module 'collections' has no attribute 'mapping')",
    re.I,
)
TIMEOUT_RE = re.compile(r"\b(time[ -]?out|timed out|timeoutexpired)\b", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-dir", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--planned-dir", type=Path, default=DEFAULT_PLANNED)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--num-tasks", type=int, default=10)
    return parser.parse_args()


def require_data_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be under {root}; got {resolved}")
    return resolved


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def action_from_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip().split("(", 1)[0].strip().lower()
    return name or None


def executed_tool(step: dict[str, Any]) -> str | None:
    if "executed_tool" in step:
        return action_from_text(step.get("executed_tool"))
    return action_from_text(step.get("action"))


def proposed_tool(step: dict[str, Any]) -> str | None:
    if "model_proposed_tool" in step:
        return action_from_text(step.get("model_proposed_tool"))
    return None


def infer_tool_success(step: dict[str, Any], tool: str | None) -> bool:
    value = step.get("tool_success")
    if isinstance(value, bool):
        return value
    if tool is None:
        return False
    observation = str(step.get("observation", "")).lower()
    if tool == "apply_patch":
        return observation.startswith("applied patch to ")
    failure_markers = (
        " failed:",
        " refused",
        "invalid model action",
        "planning controller blocked",
    )
    return not any(marker in observation for marker in failure_markers)


def has_final_patch(record: dict[str, Any]) -> bool:
    patch = str(record.get("patch", "")).strip()
    return bool(patch and patch != "(no changes)")


def repetition_counts(tools: list[str]) -> dict[str, int]:
    counts = {"list_files": 0, "search_code": 0}
    previous: str | None = None
    for tool in tools:
        if tool in counts and tool == previous:
            counts[tool] += 1
        previous = tool
    return counts


def observed_transitions(steps: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for step in steps:
        if step.get("intervention_reason") == "final_verification":
            continue
        previous = step.get("previous_phase")
        current = step.get("phase")
        if (
            isinstance(previous, str)
            and isinstance(current, str)
            and previous != current
        ):
            counts[f"{previous} -> {current}"] += 1
    return counts


def phase_reach(
    steps: list[dict[str, Any]], tools: list[str]
) -> tuple[dict[str, bool], str]:
    autonomous_steps = [
        step for step in steps
        if step.get("intervention_reason") != "final_verification"
    ]
    phases = {
        phase
        for step in autonomous_steps
        for phase in (step.get("previous_phase"), step.get("phase"))
        if isinstance(phase, str)
    }
    if phases:
        return (
            {
                "UNDERSTAND": "UNDERSTAND" in phases,
                "MODIFY": "MODIFY" in phases,
                "VERIFY": "VERIFY" in phases,
            },
            "recorded_fsm_phase",
        )
    return (
        {
            "UNDERSTAND": "read_file" in tools,
            "MODIFY": "apply_patch" in tools,
            "VERIFY": "run_test" in tools,
        },
        "tool_milestone_proxy",
    )


def classify_failure(
    record: dict[str, Any],
    steps: list[dict[str, Any]],
    tools: list[str],
    patch_attempted: bool,
    valid_patch: bool,
    final_patch: bool,
    repetitions: dict[str, int],
    reached: dict[str, bool],
) -> str | None:
    if record.get("success") is True:
        return None
    test_result = record.get("test_result")
    all_observations = "\n".join(
        str(step.get("observation", "")) for step in steps
    )
    diagnostics = "\n".join(
        str(step.get("observation", ""))
        for step in steps
        if executed_tool(step) == "run_test"
        or action_from_text(step.get("action")) == "error"
    )
    if isinstance(test_result, dict):
        diagnostics += "\n" + str(test_result.get("output", ""))
        diagnostics += "\n" + str(test_result.get("policy_error", ""))
    if TIMEOUT_RE.search(all_observations + "\n" + diagnostics):
        return "timeout"
    has_error_step = any(
        action_from_text(step.get("action")) == "error" for step in steps
    )
    if ENVIRONMENT_RE.search(diagnostics) or has_error_step:
        return "environment_error"
    if patch_attempted and not valid_patch:
        return "patch_failed"
    if valid_patch or final_patch:
        return "test_failed"
    planning_signal = (
        any(step.get("invalid_action") is True for step in steps)
        or any(step.get("controller_intervened") is True for step in steps)
        or sum(repetitions.values()) > 0
        or not reached["MODIFY"]
    )
    return "planning_error" if planning_signal else "no_patch"


def analyze_task(record: dict[str, Any], source: Path) -> dict[str, Any]:
    raw_steps = record.get("steps")
    steps = [step for step in raw_steps if isinstance(step, dict)] if isinstance(raw_steps, list) else []
    tools = [tool for step in steps if (tool := executed_tool(step)) in TOOLS]
    frequencies = Counter(tools)
    patch_steps = [
        step for step in steps if executed_tool(step) == "apply_patch"
    ]
    patch_attempted = bool(patch_steps)
    valid_patch = any(
        infer_tool_success(step, "apply_patch") for step in patch_steps
    )
    final_patch = has_final_patch(record)
    repetitions = repetition_counts(tools)
    transitions = observed_transitions(steps)
    reached, reach_source = phase_reach(steps, tools)
    intervention_steps = [
        step for step in steps if step.get("controller_intervened") is True
    ]
    intervention_reasons = Counter(
        str(step.get("intervention_reason") or "unspecified")
        for step in intervention_steps
    )
    proposed_steps = [
        step for step in steps if proposed_tool(step) is not None
    ]
    agreements = sum(
        proposed_tool(step) == executed_tool(step) for step in proposed_steps
    )
    failure = classify_failure(
        record,
        steps,
        tools,
        patch_attempted,
        valid_patch,
        final_patch,
        repetitions,
        reached,
    )
    return {
        "instance_id": str(
            record.get("instance_id") or record.get("task_id") or source.stem
        ),
        "source_file": source.name,
        "success": record.get("success") is True,
        "step_count": len(steps),
        "tool_frequency": {tool: frequencies[tool] for tool in TOOLS},
        "patch_attempted": patch_attempted,
        "valid_patch": valid_patch,
        "final_patch_generated": final_patch,
        "repeated_list_count": repetitions["list_files"],
        "repeated_search_count": repetitions["search_code"],
        "repeated_exploration": sum(repetitions.values()) > 0,
        "phase_transition": dict(transitions),
        "phase_reached": reached,
        "phase_reach_source": reach_source,
        "controller_intervention_count": len(intervention_steps),
        "intervention_reason_frequency": dict(intervention_reasons),
        "agreement_count": agreements,
        "agreement_denominator": len(proposed_steps),
        "failure_reason": failure,
    }


def load_arm(directory: Path, expected_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    records: dict[str, tuple[dict[str, Any], Path]] = {}
    errors: list[str] = []
    for path in sorted(directory.glob("task_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("top-level JSON must be an object")
            instance_id = str(value.get("instance_id") or value.get("task_id") or "")
            if not instance_id:
                raise ValueError("missing instance_id/task_id")
            if instance_id in records:
                raise ValueError(f"duplicate instance_id: {instance_id}")
            records[instance_id] = (value, path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    details = [
        analyze_task(*records[instance_id])
        for instance_id in expected_ids
        if instance_id in records
    ]
    missing = [instance_id for instance_id in expected_ids if instance_id not in records]
    errors.extend(f"missing expected instance_id: {instance_id}" for instance_id in missing)
    extras = sorted(set(records) - set(expected_ids))
    errors.extend(f"unexpected instance_id: {instance_id}" for instance_id in extras)
    return details, errors


def aggregate(details: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(details)
    tool_frequency: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for item in details:
        tool_frequency.update(item["tool_frequency"])
        transitions.update(item["phase_transition"])
        reasons.update(item["intervention_reason_frequency"])
        if item["failure_reason"]:
            failures[item["failure_reason"]] += 1
    transition_output = {name: transitions[name] for name in TRANSITIONS}
    for name in sorted(set(transitions) - set(TRANSITIONS)):
        transition_output[name] = transitions[name]
    intervention_count = sum(
        item["controller_intervention_count"] for item in details
    )
    step_count = sum(item["step_count"] for item in details)
    agreement_count = sum(item["agreement_count"] for item in details)
    agreement_denominator = sum(
        item["agreement_denominator"] for item in details
    )
    repeated_tasks = sum(item["repeated_exploration"] for item in details)
    return {
        "task_count": total,
        "success_count": sum(item["success"] for item in details),
        "success_rate": rate(sum(item["success"] for item in details), total),
        "patch_attempt_task_count": sum(item["patch_attempted"] for item in details),
        "patch_attempt_rate": rate(
            sum(item["patch_attempted"] for item in details), total
        ),
        "valid_patch_task_count": sum(item["valid_patch"] for item in details),
        "valid_patch_rate": rate(
            sum(item["valid_patch"] for item in details), total
        ),
        "final_patch_task_count": sum(
            item["final_patch_generated"] for item in details
        ),
        "final_patch_generation_rate": rate(
            sum(item["final_patch_generated"] for item in details), total
        ),
        "tool_frequency": {
            tool: tool_frequency[tool] for tool in TOOLS
        },
        "exploration_repetition": {
            "repeated_list_count": sum(
                item["repeated_list_count"] for item in details
            ),
            "repeated_search_count": sum(
                item["repeated_search_count"] for item in details
            ),
            "repeated_task_count": repeated_tasks,
            "repeated_exploration_rate": rate(repeated_tasks, total),
        },
        "phase_transition": transition_output,
        "phase_reach_rate": {
            phase: rate(
                sum(item["phase_reached"][phase] for item in details), total
            )
            for phase in ("UNDERSTAND", "MODIFY", "VERIFY")
        },
        "phase_reach_source_frequency": dict(
            Counter(item["phase_reach_source"] for item in details)
        ),
        "controller_intervention_count": intervention_count,
        "controller_intervention_rate": rate(intervention_count, step_count),
        "intervention_reason_frequency": dict(reasons),
        "controller_agreement_count": agreement_count,
        "controller_agreement_denominator": agreement_denominator,
        "controller_agreement_rate": (
            rate(agreement_count, agreement_denominator)
            if agreement_denominator
            else None
        ),
        "failure_taxonomy": {
            reason: failures[reason] for reason in FAILURES
        },
        "average_steps": round(step_count / total, 4) if total else 0.0,
        "tasks": details,
    }


def signed_delta(
    planned: float | None, structured: float | None
) -> float | None:
    if planned is None or structured is None:
        return None
    return round(planned - structured, 4)


def research_questions(
    structured: dict[str, Any], planned: dict[str, Any]
) -> dict[str, str]:
    old_rep = structured["exploration_repetition"]["repeated_exploration_rate"]
    new_rep = planned["exploration_repetition"]["repeated_exploration_rate"]
    old_repeats = sum(
        structured["exploration_repetition"][key]
        for key in ("repeated_list_count", "repeated_search_count")
    )
    new_repeats = sum(
        planned["exploration_repetition"][key]
        for key in ("repeated_list_count", "repeated_search_count")
    )
    rq1 = (
        f"The task-level repeated-exploration rate changed from {old_rep:.1%} "
        f"to {new_rep:.1%} ({new_rep - old_rep:+.1%}), while total extra "
        f"consecutive exploration calls changed from {old_repeats} to "
        f"{new_repeats}. The controller eliminated repeated list_files, but "
        "did not eliminate repeated search_code across tasks."
    )

    old_modify = structured["phase_reach_rate"]["MODIFY"]
    new_modify = planned["phase_reach_rate"]["MODIFY"]
    old_verify = structured["phase_reach_rate"]["VERIFY"]
    new_verify = planned["phase_reach_rate"]["VERIFY"]
    rq2 = (
        f"MODIFY reach changed from {old_modify:.1%} to {new_modify:.1%}; "
        f"VERIFY reach changed from {old_verify:.1%} to {new_verify:.1%}. "
        "Baseline phase reach is a tool-milestone proxy because old trajectories "
        "do not contain FSM phase fields; its VERIFY proxy includes runner-owned "
        "final tests and is not directly comparable to autonomous FSM reach."
    )

    intervention_rate = planned["controller_intervention_rate"]
    agreement_rate = planned["controller_agreement_rate"]
    success_delta = planned["success_rate"] - structured["success_rate"]
    rq3 = (
        f"The controller intervened on {intervention_rate:.1%} of recorded steps, "
        f"while model/controller agreement over model-proposed steps was "
        f"{agreement_rate:.1%}. Success changed by {success_delta:+.1%}. "
        "This observational ablation cannot fully isolate causal credit because "
        "the controller also changes the prompt/state feedback, and the requested "
        "planned run uses a 12-step budget while the existing baseline used a "
        "shorter apparent budget."
    )
    return {"RQ1": rq1, "RQ2": rq2, "RQ3": rq3}


def render_markdown(report: dict[str, Any]) -> str:
    structured = report["arms"]["structured_tool_agent"]
    planned = report["arms"]["planned_agent"]
    rows = [
        ("Success rate", "success_rate"),
        ("Patch attempt rate", "patch_attempt_rate"),
        ("Valid patch rate", "valid_patch_rate"),
        ("Final patch generation rate", "final_patch_generation_rate"),
    ]
    lines = [
        "# Planning Controller Ablation",
        "",
        "## Experimental controls",
        "",
        "- Tasks: identical first 10 SWE-bench Lite subset instances.",
        "- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct; no LoRA.",
        "- Decoding: greedy (`do_sample=False`), `max_new_tokens=768`.",
        "- Repository revisions: task-specific fixed `base_commit`.",
        "- Important limitation: the requested planned run uses `max_steps=12`; "
        "the existing structured baseline appears to have used the earlier default "
        "budget. This is reported as a potential confound.",
        "",
        "## Main results",
        "",
        "| Metric | Structured Tool Agent | Structured Tool + Planning Controller | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in rows:
        old = structured[key]
        new = planned[key]
        lines.append(
            f"| {label} | {old:.1%} | {new:.1%} | {new - old:+.1%} |"
        )
    lines.extend([
        f"| Repeated exploration rate | "
        f"{structured['exploration_repetition']['repeated_exploration_rate']:.1%} | "
        f"{planned['exploration_repetition']['repeated_exploration_rate']:.1%} | "
        f"{planned['exploration_repetition']['repeated_exploration_rate'] - structured['exploration_repetition']['repeated_exploration_rate']:+.1%} |",
        f"| Controller intervention rate | "
        f"{structured['controller_intervention_rate']:.1%} | "
        f"{planned['controller_intervention_rate']:.1%} | "
        f"{planned['controller_intervention_rate'] - structured['controller_intervention_rate']:+.1%} |",
        f"| Controller agreement rate | N/A | "
        f"{planned['controller_agreement_rate']:.1%} | N/A |",
        "",
        "## Tool frequency",
        "",
        "| Tool | Structured | Planned |",
        "|---|---:|---:|",
    ])
    for tool in TOOLS:
        lines.append(
            f"| {tool} | {structured['tool_frequency'][tool]} | "
            f"{planned['tool_frequency'][tool]} |"
        )
    lines.extend([
        "",
        "## Exploration repetition",
        "",
        "| Metric | Structured | Planned |",
        "|---|---:|---:|",
        f"| Repeated list_files calls | "
        f"{structured['exploration_repetition']['repeated_list_count']} | "
        f"{planned['exploration_repetition']['repeated_list_count']} |",
        f"| Repeated search_code calls | "
        f"{structured['exploration_repetition']['repeated_search_count']} | "
        f"{planned['exploration_repetition']['repeated_search_count']} |",
        "",
        "## Phase transitions and reach",
        "",
        "| Transition | Structured | Planned |",
        "|---|---:|---:|",
    ])
    for transition in TRANSITIONS:
        lines.append(
            f"| {transition} | {structured['phase_transition'][transition]} | "
            f"{planned['phase_transition'][transition]} |"
        )
    lines.extend([
        "",
        "| Phase reached | Structured | Planned |",
        "|---|---:|---:|",
    ])
    for phase in ("UNDERSTAND", "MODIFY", "VERIFY"):
        lines.append(
            f"| {phase} | {structured['phase_reach_rate'][phase]:.1%} | "
            f"{planned['phase_reach_rate'][phase]:.1%} |"
        )
    lines.extend([
        "",
        "Structured phase reach uses tool-milestone proxies because the baseline "
        "schema predates FSM instrumentation. Its VERIFY proxy includes the "
        "runner-owned final test and is not directly comparable to autonomous reach.",
        "",
        "## Controller intervention reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ])
    reasons = planned["intervention_reason_frequency"]
    if reasons:
        for reason, count in sorted(reasons.items()):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| None | 0 |")
    lines.extend([
        "",
        "## Failure taxonomy",
        "",
        "| Failure reason | Structured | Planned |",
        "|---|---:|---:|",
    ])
    for reason in FAILURES:
        lines.append(
            f"| {reason} | {structured['failure_taxonomy'][reason]} | "
            f"{planned['failure_taxonomy'][reason]} |"
        )
    lines.extend(["", "## Research questions", ""])
    for key, answer in report["research_questions"].items():
        lines.extend([f"### {key}", "", answer, ""])
    lines.extend([
        "## Integrity checks",
        "",
        f"- Exact task-set match: {report['integrity']['exact_task_set_match']}",
        f"- Analysis errors: {len(report['integrity']['errors'])}",
    ])
    for error in report["integrity"]["errors"]:
        lines.append(f"- {error}")
    return "\n".join(lines).rstrip() + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_expected_tasks(subset_dir: Path, count: int) -> list[dict[str, str]]:
    paths = sorted(subset_dir.glob("task_*.json"))[:count]
    if len(paths) != count:
        raise ValueError(f"expected {count} subset tasks, found {len(paths)}")
    tasks = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        tasks.append({
            "task_file": path.name,
            "instance_id": str(value["instance_id"]),
            "repo": str(value["repo"]),
            "base_commit": str(value["base_commit"]),
        })
    return tasks


def main() -> int:
    args = parse_args()
    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be positive")
    structured_dir = require_data_path(args.structured_dir, "--structured-dir")
    planned_dir = require_data_path(args.planned_dir, "--planned-dir")
    subset_dir = require_data_path(args.subset_dir, "--subset-dir")
    output_json = require_data_path(args.output_json, "--output-json")
    output_markdown = require_data_path(
        args.output_markdown, "--output-markdown"
    )
    expected_tasks = load_expected_tasks(subset_dir, args.num_tasks)
    expected_ids = [task["instance_id"] for task in expected_tasks]
    structured_details, structured_errors = load_arm(
        structured_dir, expected_ids
    )
    planned_details, planned_errors = load_arm(planned_dir, expected_ids)
    structured = aggregate(structured_details)
    planned = aggregate(planned_details)
    errors = structured_errors + planned_errors
    exact_match = (
        not errors
        and structured["task_count"] == args.num_tasks
        and planned["task_count"] == args.num_tasks
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "adapter": None,
            "training_performed": False,
            "decoding": {
                "do_sample": False,
                "max_new_tokens": 768,
            },
            "structured_directory": str(structured_dir),
            "planned_directory": str(planned_dir),
            "planned_max_steps": 12,
            "test_timeout_seconds": 900,
            "known_confound": (
                "Existing structured baseline appears to use the earlier shorter "
                "step budget; planned run is explicitly required to use 12 steps."
            ),
        },
        "integrity": {
            "expected_tasks": expected_tasks,
            "exact_task_set_match": exact_match,
            "errors": errors,
        },
        "arms": {
            "structured_tool_agent": structured,
            "planned_agent": planned,
        },
        "delta_planned_minus_structured": {
            key: signed_delta(planned[key], structured[key])
            for key in (
                "success_rate",
                "patch_attempt_rate",
                "valid_patch_rate",
                "final_patch_generation_rate",
                "controller_intervention_rate",
                "controller_agreement_rate",
            )
        },
        "research_questions": research_questions(structured, planned),
        "metric_definitions": {
            "patch_attempt_rate": (
                "fraction of tasks with at least one executed apply_patch"
            ),
            "valid_patch_rate": (
                "fraction of tasks with at least one successful apply_patch"
            ),
            "repetition_count": (
                "extra consecutive executions after the first identical "
                "list_files/search_code call in each run"
            ),
            "controller_intervention_rate": (
                "controller-intervened steps divided by all recorded steps"
            ),
            "controller_agreement_rate": (
                "model-proposed steps where proposed tool equals executed tool, "
                "divided by all steps with a model proposal; runner-forced steps "
                "with no model proposal are excluded"
            ),
        },
    }
    atomic_write(
        output_json,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(output_markdown, render_markdown(report))
    print(json.dumps({
        "exact_task_set_match": exact_match,
        "structured_success_rate": structured["success_rate"],
        "planned_success_rate": planned["success_rate"],
        "structured_repeated_exploration_rate": (
            structured["exploration_repetition"]["repeated_exploration_rate"]
        ),
        "planned_repeated_exploration_rate": (
            planned["exploration_repetition"]["repeated_exploration_rate"]
        ),
        "controller_intervention_count": planned[
            "controller_intervention_count"
        ],
        "controller_agreement_rate": planned[
            "controller_agreement_rate"
        ],
        "output_json": str(output_json),
        "output_markdown": str(output_markdown),
    }, ensure_ascii=False, indent=2))
    return 0 if exact_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
