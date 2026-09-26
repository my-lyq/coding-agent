#!/usr/bin/env python3
"""Validate action budgets and write the immutable SFT v2.1 dataset manifest."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle.expert_builder import percentile
from scripts.load_swebench import write_json_atomic


DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DATASET_DIR = DATA_ROOT / "datasets" / "sft_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    train_path = DATASET_DIR / "train.jsonl"
    validation_path = DATASET_DIR / "validation.jsonl"
    stats = json.loads((DATASET_DIR / "dataset_stats.json").read_text(encoding="utf-8"))
    split = json.loads((DATASET_DIR / "split_audit.json").read_text(encoding="utf-8"))
    source_manifest = json.loads(
        (DATA_ROOT / "swebench_dev" / "split_manifest.json").read_text(encoding="utf-8")
    )
    train = load_jsonl(train_path)
    validation = load_jsonl(validation_path)
    max_action_tokens = 1024
    violations = [
        (item["instance_id"], item["step_index"], item["target_tokens"])
        for item in [*train, *validation]
        if int(item["target_tokens"]) > max_action_tokens
    ]
    if violations:
        raise RuntimeError(f"action target budget violations: {violations[:10]}")
    train_checksum = sha256_file(train_path)
    validation_checksum = sha256_file(validation_path)
    checksum_payload = json.dumps(
        {
            "version": "2.1",
            "train_sha256": train_checksum,
            "validation_sha256": validation_checksum,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    dataset_checksum = hashlib.sha256(checksum_payload).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_version": "2.1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": source_manifest["dataset_name"],
        "source_dataset_split": source_manifest["dataset_split"],
        "source_dataset_revision": source_manifest["dataset_revision"],
        "seed": 42,
        "train_trajectory_count": stats["trajectory_counts"]["train"],
        "validation_trajectory_count": stats["trajectory_counts"]["validation"],
        "train_example_count": len(train),
        "validation_example_count": len(validation),
        "repo_split": {
            "train": split["train_repositories"],
            "validation": split["validation_repositories"],
        },
        "max_length": 8192,
        "max_action_tokens": max_action_tokens,
        "sampling_strategy": "phase-balanced",
        "validation_sampling_strategy": "natural",
        "dataset_checksum": dataset_checksum,
        "train_checksum": train_checksum,
        "validation_checksum": validation_checksum,
        "target_tokens": {
            **stats["target_tokens"],
            "p90": percentile(
                [int(item["target_tokens"]) for item in [*train, *validation]], 0.90
            ),
            "p99": percentile(
                [int(item["target_tokens"]) for item in [*train, *validation]], 0.99
            ),
        },
        "action_budget_violations": 0,
    }
    target = DATASET_DIR / "dataset_manifest.json"
    write_json_atomic(target, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"manifest={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
