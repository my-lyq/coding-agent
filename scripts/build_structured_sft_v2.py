#!/usr/bin/env python3
"""Audit repository splits and materialize Structured Tool SFT v2 JSONL.

No model weights are loaded and no training is started.  The only Hugging Face
artifact used is the Qwen tokenizer, cached under the external data root.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.prompting import canonical_tool_target, render_chat_prompt  # noqa: E402
from oracle.expert_builder import detect_prompt_leakage, percentile  # noqa: E402
from scripts.load_swebench import write_json_atomic  # noqa: E402
from training.structured_dataset import (  # noqa: E402
    DEFAULT_MAX_LENGTH,
    SUPPORTED_PHASES,
    UNSUPPORTED_TRAINING_PHASES,
    ContextBudgetManager,
    StructuredDatasetError,
    StructuredSFTDataset,
    build_step_example,
    tokenize_supervised,
    validate_structured_target,
)


DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DEV_ROOT = DATA_ROOT / "swebench_dev"
OUTPUT_ROOT = DATA_ROOT / "datasets" / "sft_v2"
LOGS_ROOT = DATA_ROOT / "logs"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
EXPECTED_REVISION = "c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--sampling-strategy",
        choices=("natural", "phase-balanced"),
        default="phase-balanced",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def choose_repository_split(
    accepted_counts: Mapping[str, int],
    *,
    seed: int,
    min_train: int = 40,
    min_validation: int = 8,
    validation_fraction: float = 0.2,
) -> tuple[set[str], set[str]]:
    """Find a deterministic repo partition closest to the desired ratio."""
    repositories = sorted(accepted_counts)
    random.Random(seed).shuffle(repositories)
    total = sum(accepted_counts.values())
    target = total * validation_fraction
    candidates: list[tuple[tuple[float, int, int], set[str]]] = []
    for width in range(1, len(repositories)):
        for indices in itertools.combinations(range(len(repositories)), width):
            validation = {repositories[index] for index in indices}
            count = sum(accepted_counts[repo] for repo in validation)
            if count < min_validation or total - count < min_train:
                continue
            bitmask = sum(1 << index for index in indices)
            candidates.append(((abs(count - target), width, bitmask), validation))
    if not candidates:
        raise RuntimeError(
            "no repository-level split satisfies minimum accepted trajectory counts"
        )
    validation = min(candidates, key=lambda item: item[0])[1]
    return set(repositories) - validation, validation


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def duplicate_groups(
    trajectories: Sequence[Mapping[str, Any]],
    extractor: Callable[[Mapping[str, Any]], str],
) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for trajectory in trajectories:
        groups[_digest(extractor(trajectory))].append(str(trajectory["instance_id"]))
    return sorted(sorted(group) for group in groups.values() if len(group) > 1)


def overlap_audit(
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train = [item for item in trajectories if item["split"] == "train"]
    validation = [item for item in trajectories if item["split"] == "validation"]
    return {
        "instance_id_overlap": sorted(
            {item["instance_id"] for item in train}
            & {item["instance_id"] for item in validation}
        ),
        "repository_overlap": sorted(
            {item["repo"] for item in train} & {item["repo"] for item in validation}
        ),
        "base_commit_overlap": sorted(
            {item["base_commit"] for item in train}
            & {item["base_commit"] for item in validation}
        ),
    }


def update_repository_split(
    manifest_path: Path,
    accepted_paths: Mapping[str, Path],
    rejected_paths: Mapping[str, Path],
    subset_paths: Mapping[str, Path],
    *,
    train_repos: set[str],
    validation_repos: set[str],
    seed: int,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    assignments: dict[str, str] = {}
    for record in manifest["records"]:
        repo = str(record["repo"])
        if repo in train_repos:
            split = "train"
        elif repo in validation_repos:
            split = "validation"
        else:
            raise RuntimeError(f"repository missing split assignment: {repo}")
        record["split"] = split
        assignments[str(record["instance_id"])] = split

    for path_map in (accepted_paths, rejected_paths, subset_paths):
        for instance_id, path in path_map.items():
            if instance_id not in assignments:
                continue
            payload = read_json(path)
            payload["split"] = assignments[instance_id]
            write_json_atomic(path, payload)

    manifest["train_instance_ids"] = sorted(
        instance_id for instance_id, split in assignments.items() if split == "train"
    )
    manifest["validation_instance_ids"] = sorted(
        instance_id for instance_id, split in assignments.items() if split == "validation"
    )
    manifest["train_count"] = len(manifest["train_instance_ids"])
    manifest["validation_count"] = len(manifest["validation_instance_ids"])
    manifest["repository_split"] = {
        "method": "deterministic_repository_level_subset_search",
        "seed": seed,
        "target_train_fraction": 0.8,
        "train_repositories": sorted(train_repos),
        "validation_repositories": sorted(validation_repos),
    }
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(manifest_path, manifest)
    return manifest


def distribution(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def length_stats(values: Sequence[int], percentiles: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "median": 0.0, **{f"p{p}": 0.0 for p in percentiles}, "max": 0}
    result: dict[str, float | int] = {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }
    for value in percentiles:
        result[f"p{value}"] = percentile(values, value / 100)
    result["max"] = max(values)
    return result


def build_examples_for_trajectory(
    trajectory: Mapping[str, Any],
    source: Mapping[str, Any],
    tokenizer: Any,
    budget: ContextBudgetManager,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    steps = trajectory["steps"]
    future_targets = [
        canonical_tool_target(str(step["tool"]), dict(step["arguments"]))
        for step in steps
    ]
    gold_patch = str(source.get("patch", ""))
    test_patch = str(source.get("test_patch", ""))
    for index in range(len(steps)):
        example = build_step_example(trajectory, index, tokenizer, budget)
        prompt = str(example["prompt"])
        leaks = detect_prompt_leakage(
            prompt,
            gold_patch=gold_patch,
            test_patch=test_patch,
            fail_to_pass=source.get("FAIL_TO_PASS"),
            pass_to_pass=source.get("PASS_TO_PASS"),
        )
        if leaks:
            raise StructuredDatasetError(
                "prompt_leakage", f"step {index} contains {', '.join(leaks)}"
            )
        leaked_future = [
            future_index
            for future_index, target in enumerate(future_targets[index:], index)
            if target in prompt
        ]
        if leaked_future:
            raise StructuredDatasetError(
                "future_action_leakage",
                f"step {index} prompt contains future targets {leaked_future}",
            )
        if render_chat_prompt(tokenizer, example["messages"]) != prompt:
            raise StructuredDatasetError(
                "prompt_inference_mismatch", f"step {index} prompt rendering drift"
            )
        validate_structured_target(example["target"], example["phase"])
        examples.append(example)
    return examples


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if data_root != DATA_ROOT and DATA_ROOT not in data_root.parents:
        raise ValueError(f"--data-root must be under {DATA_ROOT}")
    if args.seed != 42:
        raise ValueError("repository-level split seed is fixed to 42")
    if args.max_length != 8192:
        raise ValueError("Structured SFT v2 audit uses the required max_length=8192")
    dev_root = data_root / "swebench_dev"
    output_root = data_root / "datasets" / "sft_v2"
    logs_root = data_root / "logs"
    manifest_path = dev_root / "split_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("dataset_name") != "SWE-bench/SWE-bench":
        raise RuntimeError("unexpected dataset; SWE-bench Lite/test is forbidden")
    if manifest.get("dataset_split") != "dev":
        raise RuntimeError("source dataset split must be dev")
    if manifest.get("dataset_revision") != EXPECTED_REVISION:
        raise RuntimeError("unexpected SWE-bench dev dataset revision")

    accepted_paths: dict[str, Path] = {}
    trajectories: list[dict[str, Any]] = []
    for path in sorted((dev_root / "expert_trajectories").glob("*.json")):
        item = read_json(path)
        accepted_paths[str(item["instance_id"])] = path
        trajectories.append(item)
    rejected_paths = {
        str((item := read_json(path))["instance_id"]): path
        for path in sorted((dev_root / "rejected").glob("*.json"))
    }
    subset_paths = {
        str((item := read_json(path))["instance_id"]): path
        for path in sorted((dev_root / "subset").glob("task_*.json"))
    }
    selected_ids = {str(record["instance_id"]) for record in manifest["records"]}
    trajectories = [item for item in trajectories if item["instance_id"] in selected_ids]
    counts = Counter(str(item["repo"]) for item in trajectories)
    initial = overlap_audit(trajectories)
    trajectory_duplicates = duplicate_groups(
        trajectories,
        lambda item: json.dumps(item["steps"], ensure_ascii=False, sort_keys=True),
    )
    patch_duplicates = duplicate_groups(
        trajectories, lambda item: str(item["model_response"]["arguments"]["patch"])
    )

    train_repos, validation_repos = choose_repository_split(counts, seed=args.seed)
    manifest = update_repository_split(
        manifest_path,
        accepted_paths,
        rejected_paths,
        subset_paths,
        train_repos=train_repos,
        validation_repos=validation_repos,
        seed=args.seed,
    )
    trajectories = [read_json(accepted_paths[item["instance_id"]]) for item in trajectories]
    final = overlap_audit(trajectories)
    accepted_train = sum(item["split"] == "train" for item in trajectories)
    accepted_validation = sum(item["split"] == "validation" for item in trajectories)
    split_audit = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "initial": initial,
        "final": final,
        "duplicate_trajectory_groups": trajectory_duplicates,
        "duplicate_patch_groups": patch_duplicates,
        "train_repositories": sorted(train_repos),
        "validation_repositories": sorted(validation_repos),
        "accepted_train_trajectories": accepted_train,
        "accepted_validation_trajectories": accepted_validation,
        "accepted_train_fraction": accepted_train / len(trajectories),
        "manifest": str(manifest_path),
    }
    logs_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(logs_root / "sft_v2_split_audit.json", split_audit)
    write_json_atomic(output_root / "split_audit.json", split_audit)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=str(data_root / "hf_cache" / "hub"),
        local_files_only=True,
        trust_remote_code=False,
    )
    budget = ContextBudgetManager(tokenizer, max_length=args.max_length)
    train_examples: list[dict[str, Any]] = []
    validation_examples: list[dict[str, Any]] = []
    dataset_rejections: list[dict[str, Any]] = []
    included_trajectory_ids: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
    }
    for index, trajectory in enumerate(trajectories, 1):
        instance_id = str(trajectory["instance_id"])
        source = read_json(subset_paths[instance_id])
        try:
            examples = build_examples_for_trajectory(
                trajectory, source, tokenizer, budget
            )
        except StructuredDatasetError as exc:
            dataset_rejections.append(
                {
                    "instance_id": instance_id,
                    "repo": trajectory["repo"],
                    "split": trajectory["split"],
                    "rejection_reason": exc.reason,
                    "rejection_detail": exc.detail,
                }
            )
            print(f"[{index:02d}/{len(trajectories)}] {instance_id}: rejected ({exc.reason})")
            continue
        destination = (
            train_examples if trajectory["split"] == "train" else validation_examples
        )
        destination.extend(examples)
        included_trajectory_ids[str(trajectory["split"])].add(instance_id)
        print(f"[{index:02d}/{len(trajectories)}] {instance_id}: {len(examples)} examples")

    train_path = output_root / "train.jsonl"
    validation_path = output_root / "validation.jsonl"
    write_jsonl_atomic(train_path, train_examples)
    write_jsonl_atomic(validation_path, validation_examples)
    write_jsonl_atomic(output_root / "rejected_examples.jsonl", dataset_rejections)

    train_dataset = StructuredSFTDataset(
        train_path,
        tokenizer,
        max_length=args.max_length,
        sampling_strategy=args.sampling_strategy,
        split="train",
    )
    validation_dataset = StructuredSFTDataset(
        validation_path,
        tokenizer,
        max_length=args.max_length,
        sampling_strategy=args.sampling_strategy,
        split="validation",
    )
    all_examples = [*train_examples, *validation_examples]
    prompt_lengths = [int(item["prompt_tokens"]) for item in all_examples]
    target_lengths = [int(item["target_tokens"]) for item in all_examples]
    total_lengths = [int(item["total_tokens"]) for item in all_examples]
    original_lengths = [int(item["original_tokens"]) for item in all_examples]
    within_budget_before = sum(value <= args.max_length for value in original_lengths)
    within_budget_after = sum(value <= args.max_length for value in total_lengths)
    phase_distribution = {
        "train": distribution(train_examples, "phase"),
        "validation": distribution(validation_examples, "phase"),
        "all": distribution(all_examples, "phase"),
    }
    tool_distribution = {
        split: dict(
            sorted(Counter(json.loads(item["target"])["tool"] for item in records).items())
        )
        for split, records in (
            ("train", train_examples),
            ("validation", validation_examples),
            ("all", all_examples),
        )
    }
    train_ready = len(included_trajectory_ids["train"])
    validation_ready = len(included_trajectory_ids["validation"])
    stage_checks = {
        "train_trajectories_at_least_40": train_ready >= 40,
        "validation_trajectories_at_least_8": validation_ready >= 8,
        "repository_disjoint": not final["repository_overlap"],
        "all_targets_valid_json_and_registered": True,
        "gold_leakage_zero": True,
        "test_patch_leakage_zero": True,
        "p95_total_tokens_reported": bool(total_lengths),
        "budget_preserves_at_least_95_percent": (
            within_budget_after / max(1, sum(len(item["steps"]) for item in trajectories))
            >= 0.95
        ),
        "modify_examples_present": tool_distribution["train"].get("apply_patch", 0) > 0,
        "validation_sampling_is_natural": validation_dataset.effective_sampling_strategy
        == "natural",
    }
    next_stage = (
        "TRAIN_STRUCTURED_SFT_V2" if all(stage_checks.values()) else "BLOCKED"
    )
    stats = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "navigation + understanding + patch generation bootstrap",
        "model_tokenizer": args.model_name,
        "max_length": args.max_length,
        "sampling_strategy": args.sampling_strategy,
        "validation_sampling_strategy": "natural",
        "supported_phases": list(SUPPORTED_PHASES),
        "unsupported_training_phase": list(UNSUPPORTED_TRAINING_PHASES),
        "trajectory_counts": {
            "train": train_ready,
            "validation": validation_ready,
            "rejected_during_step_expansion": len(dataset_rejections),
        },
        "example_counts": {
            "train": len(train_examples),
            "validation": len(validation_examples),
            "total": len(all_examples),
        },
        "natural_phase_distribution": phase_distribution,
        "natural_tool_distribution": tool_distribution,
        "phase_balanced_expected_train_distribution": train_dataset.sampled_phase_distribution(),
        "prompt_tokens": length_stats(prompt_lengths, (90, 95, 99)),
        "target_tokens": length_stats(target_lengths, (90, 95, 99)),
        "total_tokens": length_stats(total_lengths, (90, 95, 99)),
        "context_budget": {
            "truncated_examples": sum(bool(item["truncated"]) for item in all_examples),
            "removed_history_steps": sum(int(item["removed_history_steps"]) for item in all_examples),
            "truncated_observations": sum(len(item["truncated_observations"]) for item in all_examples),
            "within_budget_before": within_budget_before,
            "within_budget_after": within_budget_after,
            "within_budget_after_rate": within_budget_after / max(1, len(all_examples)),
        },
        "leakage_audit": {
            "gold_leakage_count": 0,
            "test_patch_leakage_count": 0,
            "future_action_leakage_count": 0,
        },
        "split_audit": str(output_root / "split_audit.json"),
        "stage_checks": stage_checks,
        "NEXT_STAGE": next_stage,
    }
    write_json_atomic(output_root / "dataset_stats.json", stats)
    print(f"train={train_path} examples={len(train_examples)} trajectories={train_ready}")
    print(
        f"validation={validation_path} examples={len(validation_examples)} "
        f"trajectories={validation_ready}"
    )
    print(f"stats={output_root / 'dataset_stats.json'}")
    print(f"NEXT_STAGE={next_stage}")
    return 0 if next_stage == "TRAIN_STRUCTURED_SFT_V2" else 2


if __name__ == "__main__":
    raise SystemExit(main())
