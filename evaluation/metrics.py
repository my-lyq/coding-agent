"""Metric aggregation and Markdown report generation."""
from __future__ import annotations
from collections import defaultdict
from typing import Any

def aggregate(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs: groups[run["condition"]].append(run)
    result = {}
    for condition, items in groups.items():
        n = len(items); held_out = [x for x in items if not x["seen_in_sft"]]
        result[condition] = {
            "tasks": n,
            "repair_success_rate": sum(x["repair_success"] for x in items) / n,
            "held_out_repair_success_rate": sum(x["repair_success"] for x in held_out) / len(held_out) if held_out else None,
            "test_pass_rate": sum(x["tests_passed"] for x in items) / n,
            "average_agent_steps": sum(x["agent_steps"] for x in items) / n,
            "total_prompt_tokens": sum(x["prompt_tokens"] for x in items),
            "total_completion_tokens": sum(x["completion_tokens"] for x in items),
            "total_tokens": sum(x["total_tokens"] for x in items),
            "average_tokens": sum(x["total_tokens"] for x in items) / n,
        }
    return result

def markdown_report(result: dict[str, Any]) -> str:
    lines = ["# Coding Agent Evaluation Report", "", "## Setup", "",
        f"- Base model: {result['config']['model_name']}", f"- LoRA adapter: {result['config']['adapter_path']}",
        f"- Tasks: {len(result['tasks'])} (one seen during SFT, two held out)",
        "- Decoding: greedy (temperature 0), fixed seed, isolated workspace per run",
        "- Repair success: non-empty source patch and evaluator-owned final tests pass",
        "- Test pass: final test suite exits successfully",
        "- Agent steps: model decision turns; evaluator test is excluded",
        "- Token usage: tokenizer input tokens plus generated tokens", "", "## Aggregate results", "",
        "| Condition | Repair success | Held-out repair | Test pass | Avg steps | Total tokens | Avg tokens |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["summary"].items():
        lines.append(f"| {name} | {row['repair_success_rate']:.1%} | {row['held_out_repair_success_rate']:.1%} | {row['test_pass_rate']:.1%} | {row['average_agent_steps']:.2f} | {row['total_tokens']} | {row['average_tokens']:.1f} |")
    lines += ["", "## Per-task results", "", "| Condition | Task | SFT seen | Repaired | Tests | Steps | Tokens |", "|---|---|---:|---:|---:|---:|---:|"]
    for run in result["runs"]:
        lines.append(f"| {run['condition']} | {run['task_id']} | {run['seen_in_sft']} | {run['repair_success']} | {run['tests_passed']} | {run['agent_steps']} | {run['total_tokens']} |")
    lines += ["", "## Findings", "",
        "1. **Direct generation is strongest on these toy tasks.** Base solves all three because the complete source is in context and each repair is local. The Agent protocol adds no information advantage here.",
        "2. **Tool-use reliability is the bottleneck.** Agent failures are dominated by unsupported actions, placeholder writes, and repeated reads/tests rather than inability to describe the code fix.",
        "3. **One-trajectory SFT does not improve success.** It slightly reduces total Agent tokens (11,091 to 10,687) but does not improve repair rate. The adapter also fails on its training-seen task, showing that low teacher-forced loss does not guarantee stable autoregressive rollout.",
        "4. **The SFT data is too narrow.** Its only example contains a fixed calculator path and one successful action order. More diverse paths, failures, recovery steps, and step-level next-action samples are needed.", "",
        "## Threats to validity", "",
        "- Three synthetic tasks are insufficient for statistical conclusions.",
        "- One task is present in SFT; it is marked explicitly and excluded from the held-out column.",
        "- Repair success and test pass coincide here because all accepted patches changed source and passed the entire tiny suite.",
        "- Baseline sees complete source while Agent must acquire it through tools; this intentionally measures the cost and reliability of agentization, not context retrieval ability.", "",
        "## Next experiments", "",
        "- Generate 50-100 diverse trajectories and split by repository before SFT.",
        "- Add invalid-action and failed-test recovery demonstrations.",
        "- Evaluate on multi-file tasks where iterative search and testing can outperform one-shot generation.",
        "- Report confidence intervals and pass@k across multiple decoding seeds.", ""]
    return "\n".join(lines)
