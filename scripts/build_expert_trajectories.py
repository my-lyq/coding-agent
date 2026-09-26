#!/usr/bin/env python3
"""Build deterministic SWE-bench dev oracle trajectories for SFT bootstrap.

This command never runs project tests. Its strongest claim is that the official
gold patch applies at the declared base commit and produces the expected diff.
SWE-bench Lite test data is not read by this command.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle.expert_builder import (  # noqa: E402
    OracleBuildError,
    build_expert_trajectory,
    numeric_summary,
    percentile,
    tool_frequency,
    validate_expert_trajectory,
)
from scripts.load_swebench import (  # noqa: E402
    configure_huggingface_cache,
    validate_data_root,
    write_json_atomic,
)
from scripts.prepare_repo import (  # noqa: E402
    TaskRef,
    ensure_mirror,
    resolve_commit,
    run_command,
)


DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DEFAULT_DATASET = "SWE-bench/SWE-bench"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SOURCE_SPLIT = "dev"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--num-instances", type=int, default=60)
    parser.add_argument("--train-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-timeout", type=int, default=1800)
    parser.add_argument("--tool-timeout", type=int, default=120)
    parser.add_argument("--refresh-mirrors", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild accepted samples too.")
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--tokenizer", default=DEFAULT_MODEL)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def safe_instance_name(instance_id: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    cleaned = "".join(character if character in allowed else "_" for character in instance_id)
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"unsafe instance id: {instance_id!r}")
    return cleaned


def discover_dataset_revision(dataset_name: str, endpoint: str | None) -> str:
    from huggingface_hub import HfApi

    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    info = api.dataset_info(dataset_name)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a revision for {dataset_name}")
    return str(info.sha)


def load_dev_records(
    dataset_name: str,
    revision: str,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        split=SOURCE_SPLIT,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    records = [dict(row) for row in dataset]
    required = {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "patch",
        "test_patch",
    }
    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"dev row {index} is missing fields: {sorted(missing)}")
    ids = [str(record["instance_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("dev split contains duplicate instance_id values")
    return records


def deterministic_split(
    records: Sequence[dict[str, Any]],
    *,
    num_instances: int,
    train_count: int,
    seed: int,
) -> list[tuple[dict[str, Any], str]]:
    if num_instances <= 0 or num_instances > len(records):
        raise ValueError(
            f"--num-instances must be in [1, {len(records)}], got {num_instances}"
        )
    if train_count < 0 or train_count > num_instances:
        raise ValueError("--train-count must be between 0 and --num-instances")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    chosen = [records[index] for index in indices[:num_instances]]
    return [
        (record, "train" if index < train_count else "validation")
        for index, record in enumerate(chosen)
    ]


def materialize_source_data(
    root: Path,
    records: Sequence[dict[str, Any]],
    selected: Sequence[tuple[dict[str, Any], str]],
    *,
    dataset_name: str,
    revision: str,
    seed: int,
) -> tuple[str, Path]:
    raw_path = root / "raw" / "swebench_dev.json"
    write_json_atomic(raw_path, list(records))
    raw_sha256 = canonical_sha256(records)

    subset_dir = root / "subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, Any]] = []
    for index, (record, split) in enumerate(selected):
        payload = dict(record)
        payload.update(
            {
                "split": split,
                "dataset_name": dataset_name,
                "dataset_split": SOURCE_SPLIT,
                "dataset_revision": revision,
                "dataset_checksum": raw_sha256,
                "teacher_only_fields": [
                    "patch",
                    "test_patch",
                    "FAIL_TO_PASS",
                    "PASS_TO_PASS",
                ],
            }
        )
        write_json_atomic(subset_dir / f"task_{index:05d}.json", payload)
        manifest_records.append(
            {
                "instance_id": str(record["instance_id"]),
                "repo": str(record["repo"]),
                "base_commit": str(record["base_commit"]),
                "split": split,
                "dataset_name": dataset_name,
                "dataset_split": SOURCE_SPLIT,
                "dataset_revision": revision,
                "dataset_checksum": raw_sha256,
            }
        )
    train_ids = [item["instance_id"] for item in manifest_records if item["split"] == "train"]
    validation_ids = [
        item["instance_id"] for item in manifest_records if item["split"] == "validation"
    ]
    if not set(train_ids).isdisjoint(validation_ids):
        raise RuntimeError("train/validation instance leakage detected")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "dataset_split": SOURCE_SPLIT,
        "dataset_revision": revision,
        "dataset_checksum": raw_sha256,
        "selection": {"method": "seeded_shuffle", "seed": seed},
        "requested_instances": len(selected),
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "train_instance_ids": train_ids,
        "validation_instance_ids": validation_ids,
        "records": manifest_records,
    }
    manifest_path = root / "split_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return raw_sha256, manifest_path


@contextlib.contextmanager
def isolated_worktree(
    task: TaskRef,
    *,
    repos_dir: Path,
    workspaces_dir: Path,
    refresh: bool,
    timeout: int,
    keep: bool,
) -> Iterator[tuple[Path, str]]:
    mirror, _ = ensure_mirror(task, repos_dir, refresh=refresh, timeout=timeout)
    resolved = resolve_commit(mirror, task, timeout)
    workspaces_dir.mkdir(parents=True, exist_ok=True)
    workspace = workspaces_dir / safe_instance_name(task.instance_id)

    # Replace only this generated, instance-specific worktree. Mirrors and data
    # records are never deleted here.
    run_command(
        ["git", "-C", str(mirror), "worktree", "remove", "--force", str(workspace)],
        timeout=timeout,
        check=False,
    )
    if workspace.exists():
        shutil.rmtree(workspace)
    run_command(["git", "-C", str(mirror), "worktree", "prune"], timeout=timeout)
    result = run_command(
        [
            "git",
            "-C",
            str(mirror),
            "worktree",
            "add",
            "--detach",
            str(workspace),
            resolved,
        ],
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"worktree checkout failed: {detail[-4000:]}")
    try:
        yield workspace, resolved
    finally:
        if not keep:
            run_command(
                [
                    "git",
                    "-C",
                    str(mirror),
                    "worktree",
                    "remove",
                    "--force",
                    str(workspace),
                ],
                timeout=timeout,
                check=False,
            )
            if workspace.exists():
                shutil.rmtree(workspace)


def accepted_sample_is_current(
    path: Path,
    *,
    instance_id: str,
    split: str,
    revision: str,
    raw_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        validate_expert_trajectory(trajectory)
        dataset = trajectory["dataset"]
        if (
            trajectory["instance_id"] != instance_id
            or trajectory["split"] != split
            or dataset["split"] != SOURCE_SPLIT
            or dataset["revision"] != revision
            or dataset["raw_sha256"] != raw_sha256
        ):
            return None
        return trajectory
    except (OSError, json.JSONDecodeError, KeyError, TypeError, OracleBuildError):
        return None


def rejection_payload(
    record: dict[str, Any],
    split: str,
    reason: str,
    detail: str,
    *,
    dataset_name: str,
    revision: str,
    raw_sha256: str,
) -> dict[str, Any]:
    leakage_types: list[str] = []
    if reason == "prompt_leakage" and "contains:" in detail:
        leakage_types = [part.strip() for part in detail.split("contains:", 1)[1].split(",")]
    return {
        "schema_version": 1,
        "instance_id": str(record.get("instance_id", "")),
        "repo": str(record.get("repo", "")),
        "base_commit": str(record.get("base_commit", "")),
        "split": split,
        "dataset": {
            "name": dataset_name,
            "split": SOURCE_SPLIT,
            "revision": revision,
            "raw_sha256": raw_sha256,
        },
        "teacher_source": "swebench_gold_oracle",
        "accepted": False,
        "rejection_reason": reason,
        "rejection_detail": detail,
        "leakage_types": leakage_types,
        "official_resolved": None,
    }


def estimate_prompt_tokens(
    prompts: Sequence[str], model_name: str, cache_root: Path
) -> tuple[list[int], str]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_root / "hub"),
            local_files_only=True,
            trust_remote_code=False,
        )
        return [len(tokenizer.encode(prompt, add_special_tokens=True)) for prompt in prompts], f"tokenizer:{model_name}"
    except Exception as exc:
        # This remains explicitly labelled as an estimate if no local tokenizer
        # is present; report generation never triggers a model download.
        return [max(1, (len(prompt) + 3) // 4) for prompt in prompts], f"chars_div_4_fallback:{type(exc).__name__}"


def create_report(
    selected: Sequence[tuple[dict[str, Any], str]],
    accepted: Sequence[dict[str, Any]],
    rejected: Sequence[dict[str, Any]],
    *,
    tokenizer_name: str,
    cache_root: Path,
    manifest_path: Path,
    dataset_name: str,
    dataset_revision: str,
) -> dict[str, Any]:
    requested = len(selected)
    train_count = sum(item["split"] == "train" for item in accepted)
    validation_count = sum(item["split"] == "validation" for item in accepted)
    patch_sizes = [int(item["patch_stats"]["total_changed_lines"]) for item in accepted]
    modified_counts = [int(item["patch_stats"]["modified_file_count"]) for item in accepted]
    steps = [len(item["steps"]) for item in accepted]
    prompts = [str(item["model_prompt"]) for item in accepted]
    prompt_chars = [len(prompt) for prompt in prompts]
    prompt_tokens, token_method = estimate_prompt_tokens(prompts, tokenizer_name, cache_root)
    rejection_counts = Counter(item["rejection_reason"] for item in rejected)
    gold_leaks = sum("gold_patch" in item.get("leakage_types", []) for item in rejected)
    test_leaks = sum("test_patch" in item.get("leakage_types", []) for item in rejected)
    next_stage = (
        "STRUCTURED_SFT_V2"
        if train_count >= 30 and validation_count >= 5
        else "BLOCKED_INSUFFICIENT_EXPERT_DATA"
    )
    modified_summary = numeric_summary(modified_counts)
    modified_summary["max"] = max(modified_counts, default=0)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset_name,
            "split": SOURCE_SPLIT,
            "revision": dataset_revision,
            "manifest": str(manifest_path),
        },
        "requested_instances": requested,
        "accepted_instances": len(accepted),
        "rejected_instances": len(rejected),
        "train_count": train_count,
        "validation_count": validation_count,
        "git_apply_success_rate": len(accepted) / requested if requested else 0.0,
        "repository_distribution": dict(
            sorted(Counter(item["repo"] for item in accepted).items())
        ),
        "average_steps": statistics.fmean(steps) if steps else 0.0,
        "tool_frequency": tool_frequency(accepted),
        "patch_size": numeric_summary(patch_sizes, include_p95=True),
        "modified_files": modified_summary,
        "prompt_length_estimate": {
            "characters": {
                **numeric_summary(prompt_chars),
                "p95": percentile(prompt_chars, 0.95),
            },
            "tokens": {
                **numeric_summary(prompt_tokens),
                "p95": percentile(prompt_tokens, 0.95),
            },
            "method": token_method,
        },
        "quality_tier_distribution": dict(
            sorted(Counter(item["quality_tier"] for item in accepted).items())
        ),
        "gold_leakage_count": gold_leaks,
        "test_patch_leakage_count": test_leaks,
        "rejection_reason_frequency": dict(sorted(rejection_counts.items())),
        "official_resolved": None,
        "NEXT_STAGE": next_stage,
    }


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    validate_data_root(data_root)
    if args.dataset_name != DEFAULT_DATASET:
        raise ValueError(f"dataset must be {DEFAULT_DATASET}; Lite/test data is forbidden")
    if args.num_instances != 60 or args.train_count != 50 or args.seed != 42:
        raise ValueError("this bootstrap is fixed to 60 instances, 50 train, 10 validation, seed=42")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    cache_dir = configure_huggingface_cache(data_root)
    root = data_root / "swebench_dev"
    repos_dir = root / "repos"
    trajectories_dir = root / "expert_trajectories"
    rejected_dir = root / "rejected"
    workspaces_dir = root / "workspaces"
    logs_dir = data_root / "logs"
    for directory in (repos_dir, trajectories_dir, rejected_dir, workspaces_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    revision = args.dataset_revision or discover_dataset_revision(
        args.dataset_name, args.hf_endpoint
    )
    records = load_dev_records(args.dataset_name, revision, cache_dir)
    selected = deterministic_split(
        records,
        num_instances=args.num_instances,
        train_count=args.train_count,
        seed=args.seed,
    )
    raw_sha256, manifest_path = materialize_source_data(
        root,
        records,
        selected,
        dataset_name=args.dataset_name,
        revision=revision,
        seed=args.seed,
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, (record, split) in enumerate(selected, 1):
        instance_id = str(record["instance_id"])
        name = safe_instance_name(instance_id)
        accepted_path = trajectories_dir / f"{name}.json"
        rejected_path = rejected_dir / f"{name}.json"
        existing = None if args.force else accepted_sample_is_current(
            accepted_path,
            instance_id=instance_id,
            split=split,
            revision=revision,
            raw_sha256=raw_sha256,
        )
        if existing is not None:
            accepted.append(existing)
            print(f"[{index:02d}/{len(selected)}] {instance_id}: accepted (resume)", flush=True)
            continue

        task = TaskRef(
            instance_id=instance_id,
            repo=str(record["repo"]),
            base_commit=str(record["base_commit"]),
            source_file=str(root / "subset" / f"task_{index - 1:05d}.json"),
        )
        try:
            with isolated_worktree(
                task,
                repos_dir=repos_dir,
                workspaces_dir=workspaces_dir,
                refresh=args.refresh_mirrors,
                timeout=args.repo_timeout,
                keep=args.keep_workspaces,
            ) as (workspace, resolved):
                trajectory = build_expert_trajectory(
                    record,
                    workspace,
                    split=split,
                    dataset_name=args.dataset_name,
                    dataset_split=SOURCE_SPLIT,
                    dataset_revision=revision,
                    raw_sha256=raw_sha256,
                    resolved_commit=resolved,
                    timeout=args.tool_timeout,
                )
            write_json_atomic(accepted_path, trajectory)
            rejected_path.unlink(missing_ok=True)
            accepted.append(trajectory)
            print(f"[{index:02d}/{len(selected)}] {instance_id}: accepted", flush=True)
        except Exception as exc:
            reason = exc.reason if isinstance(exc, OracleBuildError) else type(exc).__name__
            detail = exc.detail if isinstance(exc, OracleBuildError) else str(exc)
            rejection = rejection_payload(
                record,
                split,
                reason,
                detail,
                dataset_name=args.dataset_name,
                revision=revision,
                raw_sha256=raw_sha256,
            )
            write_json_atomic(rejected_path, rejection)
            accepted_path.unlink(missing_ok=True)
            rejected.append(rejection)
            print(
                f"[{index:02d}/{len(selected)}] {instance_id}: rejected ({reason})",
                flush=True,
            )
            if args.fail_fast:
                break

    report = create_report(
        selected,
        accepted,
        rejected,
        tokenizer_name=args.tokenizer,
        cache_root=data_root / "hf_cache",
        manifest_path=manifest_path,
        dataset_name=args.dataset_name,
        dataset_revision=revision,
    )
    report_path = logs_dir / "expert_bootstrap_report.json"
    write_json_atomic(report_path, report)
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")
    print(
        f"accepted={report['accepted_instances']} rejected={report['rejected_instances']} "
        f"train={report['train_count']} validation={report['validation_count']}"
    )
    print(f"NEXT_STAGE={report['NEXT_STAGE']}")
    return 0 if report["NEXT_STAGE"] == "STRUCTURED_SFT_V2" else 2


if __name__ == "__main__":
    raise SystemExit(main())
