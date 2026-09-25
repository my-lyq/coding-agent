#!/usr/bin/env python3
"""Download SWE-bench Lite and materialize deterministic Agent task files.

All dataset artifacts and Hugging Face caches live outside the Git repository
under /data_local/lyq/data_coding_agent by default. Repository cloning is
intentionally out of scope for this loader.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


DEFAULT_DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
REQUIRED_SOURCE_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "test_patch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Deterministically shuffle with --seed before taking the subset.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore the Hugging Face download cache and fetch the dataset again.",
    )
    return parser.parse_args()


def configure_huggingface_cache(data_root: Path) -> Path:
    """Redirect every Hugging Face cache used by this process to the data disk."""
    cache_root = data_root / "hf_cache"
    os.environ["HF_HOME"] = str(cache_root / "home")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    return cache_root / "datasets"


def validate_data_root(data_root: Path) -> None:
    """Prevent accidental dataset writes to the repository or a home directory."""
    allowed_root = DEFAULT_DATA_ROOT.resolve()
    if data_root != allowed_root and allowed_root not in data_root.parents:
        raise ValueError(
            f"--data-root must be {allowed_root} or one of its subdirectories; "
            f"got {data_root}"
        )


def validate_record(record: dict[str, Any], index: int) -> None:
    missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"dataset record {index} is missing fields: {missing}")
    empty = [field for field in REQUIRED_SOURCE_FIELDS if not str(record[field]).strip()]
    if empty:
        raise ValueError(f"dataset record {index} has empty fields: {empty}")


def to_agent_task(record: dict[str, Any], index: int) -> dict[str, str]:
    """Project a raw SWE-bench row onto the stable Agent task contract."""
    validate_record(record, index)
    hints = record.get("hints", record.get("hints_text", ""))
    return {
        "instance_id": str(record["instance_id"]),
        "repo": str(record["repo"]),
        "base_commit": str(record["base_commit"]),
        "problem_statement": str(record["problem_statement"]),
        "test_patch": str(record["test_patch"]),
        "hints": "" if hints is None else str(hints),
    }


def select_records(
    records: Sequence[dict[str, Any]],
    num_tasks: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    if num_tasks <= 0:
        raise ValueError("--num-tasks must be positive")
    if num_tasks > len(records):
        raise ValueError(
            f"requested {num_tasks} tasks, but split contains only {len(records)}"
        )
    indices = list(range(len(records)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return [records[index] for index in indices[:num_tasks]]


def write_json_atomic(path: Path, value: Any) -> None:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_subset(
    subset_dir: Path,
    records: Sequence[dict[str, Any]],
    *,
    dataset_name: str,
    split: str,
    seed: int,
    shuffled: bool,
    raw_sha256: str,
) -> None:
    """Build in a staging directory, then atomically replace the active subset."""
    staging = Path(tempfile.mkdtemp(prefix=".subset-staging-", dir=subset_dir.parent))
    backup = subset_dir.parent / ".subset-backup"
    try:
        instance_ids: list[str] = []
        for index, record in enumerate(records):
            task = to_agent_task(record, index)
            instance_ids.append(task["instance_id"])
            write_json_atomic(staging / f"task_{index:05d}.json", task)
        manifest = {
            "schema_version": 1,
            "dataset": dataset_name,
            "split": split,
            "num_tasks": len(records),
            "selection": "seeded_shuffle" if shuffled else "dataset_order",
            "seed": seed if shuffled else None,
            "raw_sha256": raw_sha256,
            "instance_ids": instance_ids,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(staging / "manifest.json", manifest)

        if backup.exists():
            shutil.rmtree(backup)
        if subset_dir.exists():
            os.replace(subset_dir, backup)
        os.replace(staging, subset_dir)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if not subset_dir.exists() and backup.exists():
            os.replace(backup, subset_dir)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@contextlib.contextmanager
def exclusive_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_records(
    dataset_name: str,
    split: str,
    cache_dir: Path,
    force_download: bool,
) -> list[dict[str, Any]]:
    # Import after configuring environment variables so datasets cannot write to
    # ~/.cache before our explicit cache_dir is applied.
    from datasets import DownloadMode, load_dataset

    mode = DownloadMode.FORCE_REDOWNLOAD if force_download else DownloadMode.REUSE_DATASET_IF_EXISTS
    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=str(cache_dir),
        download_mode=mode,
    )
    return [dict(record) for record in dataset]


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    validate_data_root(data_root)
    swebench_root = data_root / "swebench"
    raw_dir = swebench_root / "raw"
    subset_dir = swebench_root / "subset"
    repos_dir = swebench_root / "repos"
    cache_dir = configure_huggingface_cache(data_root)

    raw_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)  # Deliberately left empty.

    with exclusive_lock(swebench_root / ".loader.lock"):
        records = load_records(
            args.dataset_name, args.split, cache_dir, args.force_download
        )
        raw_path = raw_dir / "swebench_lite.json"
        write_json_atomic(raw_path, records)
        chosen = select_records(
            records,
            args.num_tasks,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        materialize_subset(
            subset_dir,
            chosen,
            dataset_name=args.dataset_name,
            split=args.split,
            seed=args.seed,
            shuffled=args.shuffle,
            raw_sha256=sha256_file(raw_path),
        )

    print(f"dataset={args.dataset_name} split={args.split} raw_records={len(records)}")
    print(f"raw={raw_path}")
    print(f"subset={subset_dir} tasks={len(chosen)}")
    print(f"repos={repos_dir} (no repositories cloned)")
    print(f"hf_cache={cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
