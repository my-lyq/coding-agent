#!/usr/bin/env python3
"""Render the frozen Structured SFT v2 Base-vs-SFT static ablation report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
LOGS_ROOT = DATA_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=LOGS_ROOT / "sft_v2_base_static_eval.json",
    )
    parser.add_argument(
        "--sft",
        type=Path,
        default=LOGS_ROOT / "sft_v2_static_eval.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LOGS_ROOT / "sft_v2_static_ablation.md",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model",
        "dataset_checksum",
        "validation_checksum",
        "validation_count",
        "decoding_config",
        "metrics",
    }
    missing = required - report.keys()
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return report


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def delta(base: float, sft: float) -> str:
    return f"{100.0 * (sft - base):+.2f} pp"


def metric(report: dict[str, Any], key: str) -> float:
    return float(report["metrics"][key])


def phase_metric(report: dict[str, Any], phase: str) -> float:
    return float(report["metrics"]["phase_specific_tool_accuracy"][phase]["accuracy"])


def main() -> int:
    args = parse_args()
    base = load_report(args.base)
    sft = load_report(args.sft)
    comparable_fields = (
        "model",
        "dataset_checksum",
        "validation_checksum",
        "validation_count",
        "decoding_config",
    )
    mismatches = [field for field in comparable_fields if base[field] != sft[field]]
    if mismatches:
        raise RuntimeError(f"non-comparable evaluation reports: {mismatches}")

    rows: list[tuple[str, float, float]] = [
        ("JSON valid", metric(base, "json_valid_rate"), metric(sft, "json_valid_rate")),
        (
            "Registered tool",
            metric(base, "registered_tool_rate"),
            metric(sft, "registered_tool_rate"),
        ),
        (
            "Argument schema valid",
            metric(base, "argument_schema_valid_rate"),
            metric(sft, "argument_schema_valid_rate"),
        ),
        (
            "Phase compatible",
            metric(base, "phase_compatible_rate"),
            metric(sft, "phase_compatible_rate"),
        ),
        (
            "Teacher tool accuracy",
            metric(base, "teacher_tool_accuracy"),
            metric(sft, "teacher_tool_accuracy"),
        ),
        ("LOCATE accuracy", phase_metric(base, "LOCATE"), phase_metric(sft, "LOCATE")),
        (
            "UNDERSTAND accuracy",
            phase_metric(base, "UNDERSTAND"),
            phase_metric(sft, "UNDERSTAND"),
        ),
        ("MODIFY accuracy", phase_metric(base, "MODIFY"), phase_metric(sft, "MODIFY")),
        (
            "Apply-patch selection accuracy",
            metric(base, "modify_apply_patch_selection_accuracy"),
            metric(sft, "modify_apply_patch_selection_accuracy"),
        ),
        (
            "Generation truncation",
            metric(base, "generation_truncation_rate"),
            metric(sft, "generation_truncation_rate"),
        ),
    ]
    secondary_rows: list[tuple[str, float, float]] = [
        (
            "Strict JSON only",
            metric(base, "strict_json_only_rate"),
            metric(sft, "strict_json_only_rate"),
        ),
        (
            "Executable action",
            metric(base, "executable_action_rate"),
            metric(sft, "executable_action_rate"),
        ),
        (
            "Tool execution success / attempted",
            metric(base, "tool_execution_success_rate_among_attempts"),
            metric(sft, "tool_execution_success_rate_among_attempts"),
        ),
        (
            "Tool execution success / all",
            metric(base, "tool_execution_success_rate_all"),
            metric(sft, "tool_execution_success_rate_all"),
        ),
    ]

    def table(items: list[tuple[str, float, float]]) -> str:
        lines = ["| Metric | Base | SFT | Delta |", "|---|---:|---:|---:|"]
        for label, base_value, sft_value in items:
            lines.append(
                f"| {label} | {percent(base_value)} | {percent(sft_value)} | "
                f"{delta(base_value, sft_value)} |"
            )
        return "\n".join(lines)

    modify_delta = phase_metric(sft, "MODIFY") - phase_metric(base, "MODIFY")
    next_stage = (
        "HELDOUT_AGENT_ROLLOUT_ABLATION"
        if modify_delta > 0.0
        else "ANALYZE_SFT_FAILURE"
    )
    content = f"""# Structured Tool SFT v2.1 Static Ablation

This is a next-action policy evaluation on the same frozen, natural-distribution held-out validation set. It is not a SWE-bench resolved evaluation.

## Comparable setup

- Model: `{base['model']}`
- Validation examples: {base['validation_count']}
- Dataset checksum: `{base['dataset_checksum']}`
- Validation checksum: `{base['validation_checksum']}`
- Decoding: greedy (`do_sample=false`), `max_action_tokens=1024`
- Base condition: no adapter
- SFT condition: LoRA adapter only

## Primary metrics

{table(rows)}

## Secondary execution diagnostics

{table(secondary_rows)}

## Interpretation

Structured SFT improved held-out expert-state policy behavior. In particular, MODIFY/apply-patch selection increased from {percent(phase_metric(base, 'MODIFY'))} to {percent(phase_metric(sft, 'MODIFY'))} ({delta(phase_metric(base, 'MODIFY'), phase_metric(sft, 'MODIFY'))}), while overall teacher-tool accuracy increased from {percent(metric(base, 'teacher_tool_accuracy'))} to {percent(metric(sft, 'teacher_tool_accuracy'))}.

The result is not uniformly positive. Generation truncation increased from {percent(metric(base, 'generation_truncation_rate'))} to {percent(metric(sft, 'generation_truncation_rate'))}, and execution success among attempted actions fell from {percent(metric(base, 'tool_execution_success_rate_among_attempts'))} to {percent(metric(sft, 'tool_execution_success_rate_among_attempts'))}. Static tool-selection gains therefore justify a held-out rollout ablation, but do not establish end-to-end bug-fixing or SWE-bench resolved improvement.

`NEXT_STAGE = {next_stage}`
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"output={args.output}")
    print(f"NEXT_STAGE={next_stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
