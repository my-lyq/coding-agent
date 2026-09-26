#!/usr/bin/env python3
"""Upgrade oracle trajectories to sequential <=1024-token patch actions."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.prompting import canonical_tool_target  # noqa: E402
from oracle.action_splitter import GoldPatchActionSplitter  # noqa: E402
from oracle.expert_builder import OracleBuildError  # noqa: E402
from scripts.build_expert_trajectories import isolated_worktree  # noqa: E402
from scripts.load_swebench import write_json_atomic  # noqa: E402
from scripts.prepare_repo import TaskRef  # noqa: E402


DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-action-tokens", type=int, default=1024)
    parser.add_argument("--repo-timeout", type=int, default=1800)
    parser.add_argument("--tool-timeout", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_index(directory: Path, pattern: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    result = {}
    for path in sorted(directory.glob(pattern)):
        value = json.loads(path.read_text(encoding="utf-8"))
        result[str(value["instance_id"])] = (path, value)
    return result


def modify_step(patch: str, observation: str, target_tokens: int) -> dict[str, Any]:
    return {
        "phase": "MODIFY",
        "tool": "apply_patch",
        "arguments": {"patch": patch},
        "observation": observation,
        "tool_success": True,
        "observation_source": "repository_tool_execution",
        "controller_intervened": False,
        "repair_applied": False,
        "invalid_action": False,
        "target_tokens": target_tokens,
    }


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if data_root != DATA_ROOT and DATA_ROOT not in data_root.parents:
        raise ValueError(f"--data-root must be under {DATA_ROOT}")
    if args.max_action_tokens != 1024:
        raise ValueError("SFT v2.1 action budget is fixed to 1024")
    dev_root = data_root / "swebench_dev"
    trajectories_dir = dev_root / "expert_trajectories"
    backup_dir = dev_root / "expert_trajectories_pre_action_split"
    if not backup_dir.exists():
        shutil.copytree(trajectories_dir, backup_dir)
    trajectories = load_index(trajectories_dir, "*.json")
    sources = load_index(dev_root / "subset", "task_*.json")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=str(data_root / "hf_cache" / "hub"),
        local_files_only=True,
        trust_remote_code=False,
    )
    splitter = GoldPatchActionSplitter(
        tokenizer,
        max_action_tokens=args.max_action_tokens,
        timeout=args.tool_timeout,
    )
    results: list[dict[str, Any]] = []
    failures = 0
    for position, (instance_id, (path, trajectory)) in enumerate(trajectories.items(), 1):
        alignment = trajectory.get("action_alignment", {})
        if (
            not args.force
            and alignment.get("version") == "2.1"
            and alignment.get("max_action_tokens") == args.max_action_tokens
        ):
            results.append(
                {
                    "instance_id": instance_id,
                    "status": "resume",
                    "action_count": alignment.get("action_count", 0),
                    "max_target_tokens": alignment.get("max_target_tokens", 0),
                }
            )
            print(f"[{position:02d}/{len(trajectories)}] {instance_id}: resume")
            continue
        source = sources[instance_id][1]
        task = TaskRef(
            instance_id=instance_id,
            repo=str(trajectory["repo"]),
            base_commit=str(trajectory["base_commit"]),
            source_file=str(sources[instance_id][0]),
        )
        try:
            with isolated_worktree(
                task,
                repos_dir=dev_root / "repos",
                workspaces_dir=dev_root / "workspaces_action_split",
                refresh=False,
                timeout=args.repo_timeout,
                keep=False,
            ) as (workspace, resolved):
                split = splitter.split_and_apply(
                    workspace,
                    str(source["patch"]),
                    resolved_commit=resolved,
                )
            navigation = [
                step for step in trajectory["steps"] if step.get("tool") != "apply_patch"
            ]
            modifications = [
                modify_step(action.patch, action.observation, action.target_tokens)
                for action in split.actions
            ]
            trajectory["steps"] = [*navigation, *modifications]
            trajectory["model_response"] = {
                "tool": "apply_patch",
                "arguments": {"patch": split.actions[-1].patch},
            }
            trajectory["action_sequence"] = [
                {
                    "tool": "apply_patch",
                    "arguments": {"patch": action.patch},
                    "target_tokens": action.target_tokens,
                }
                for action in split.actions
            ]
            trajectory["git_diff"] = split.final_diff
            trajectory["action_alignment"] = {
                "version": "2.1",
                "strategy": "file_then_hunk_then_bounded_edit",
                "max_action_tokens": args.max_action_tokens,
                "action_count": len(split.actions),
                "max_target_tokens": max(action.target_tokens for action in split.actions),
                "sequential_git_apply_check": True,
                "modified_files_match": True,
                "normalized_gold_diff_match": True,
            }
            write_json_atomic(path, trajectory)
            results.append(
                {
                    "instance_id": instance_id,
                    "status": "aligned",
                    "action_count": len(split.actions),
                    "max_target_tokens": max(
                        action.target_tokens for action in split.actions
                    ),
                }
            )
            print(
                f"[{position:02d}/{len(trajectories)}] {instance_id}: "
                f"actions={len(split.actions)} max_target={results[-1]['max_target_tokens']}"
            )
        except Exception as exc:
            failures += 1
            reason = exc.reason if isinstance(exc, OracleBuildError) else type(exc).__name__
            detail = exc.detail if isinstance(exc, OracleBuildError) else str(exc)
            results.append(
                {
                    "instance_id": instance_id,
                    "status": "rejected",
                    "rejection_reason": reason,
                    "rejection_detail": detail,
                }
            )
            print(f"[{position:02d}/{len(trajectories)}] {instance_id}: rejected ({reason})")

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "sft_v2.1",
        "model_tokenizer": args.model_name,
        "max_action_tokens": args.max_action_tokens,
        "requested_trajectories": len(trajectories),
        "aligned_trajectories": sum(item["status"] in {"aligned", "resume"} for item in results),
        "rejected_trajectories": failures,
        "total_patch_actions": sum(int(item.get("action_count", 0)) for item in results),
        "maximum_target_tokens": max(
            (int(item.get("max_target_tokens", 0)) for item in results), default=0
        ),
        "status_distribution": dict(sorted(Counter(item["status"] for item in results).items())),
        "results": results,
    }
    report_path = data_root / "logs" / "sft_v21_action_alignment.json"
    write_json_atomic(report_path, report)
    print(f"report={report_path}")
    print(f"aligned={report['aligned_trajectories']} rejected={failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
