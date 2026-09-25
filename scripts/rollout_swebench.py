#!/usr/bin/env python3
"""Run resumable SWE-bench Agent rollouts with one shared model instance."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "agent"))
from main import HFPolicy, SWEbenchPolicy, run_repository_task  # noqa: E402
from task import Task  # noqa: E402

DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DEFAULT_SUBSET = DATA_ROOT / "swebench" / "subset"
DEFAULT_REPOS = DATA_ROOT / "swebench" / "repos"
DEFAULT_OUTPUT = DATA_ROOT / "trajectories" / "raw"
DEFAULT_WORKSPACES = DATA_ROOT / "rollouts" / "workspaces"
TERMINAL_STATUSES = {"success", "fail", "error"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--repos-dir", type=Path, default=DEFAULT_REPOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workspaces-dir", type=Path, default=DEFAULT_WORKSPACES)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--test-timeout", type=int, default=900)
    parser.add_argument(
        "--test-command",
        default="python -m pytest -q",
        help="Fallback exact test command when no per-task mapping exists.",
    )
    parser.add_argument(
        "--test-command-map",
        type=Path,
        help="JSON object mapping instance_id or repo to an exact command.",
    )
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def under_data_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be under {root}; got {resolved}")
    return resolved

def load_command_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(command, str)
        for key, command in value.items()
    ):
        raise ValueError("--test-command-map must contain a JSON string-to-string object")
    return value

def load_subset_tasks(
    subset_dir: Path, start_index: int, num_tasks: int
) -> list[tuple[Path, dict[str, Any]]]:
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if num_tasks <= 0:
        raise ValueError("--num-tasks must be positive")
    files = sorted(subset_dir.glob("task_*.json"))
    selected = files[start_index : start_index + num_tasks]
    if not selected:
        raise ValueError(
            f"no task files selected from {subset_dir} at index {start_index}"
        )
    tasks = []
    for path in selected:
        value = json.loads(path.read_text(encoding="utf-8"))
        required = {"instance_id", "repo", "base_commit", "problem_statement"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        tasks.append((path, value))
    return tasks

def repo_slug(repo: str) -> str:
    return repo.replace("/", "__")

def resolve_test_command(
    raw: dict[str, Any], mapping: dict[str, str], fallback: str
) -> str:
    command = (
        mapping.get(str(raw["instance_id"]))
        or mapping.get(str(raw["repo"]))
        or raw.get("test_command")
        or fallback
    )
    command = str(command).strip()
    if not command:
        raise ValueError(f"no test command for {raw['instance_id']}")
    return command

@contextlib.contextmanager
def task_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def completed_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = value.get("status")
    return status if status in TERMINAL_STATUSES else None

def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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

def run_git(arguments: list[str], timeout: int = 300) -> str:
    result = subprocess.run(
        ["git", *arguments],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()

def prepare_isolated_workspace(
    base_repo: Path, workspace: Path, base_commit: str
) -> None:
    base_head = run_git(["-C", str(base_repo), "rev-parse", "HEAD"])
    expected = run_git(
        ["-C", str(base_repo), "rev-parse", f"{base_commit}^{{commit}}"]
    )
    if base_head != expected:
        raise RuntimeError(
            f"prepared repository is at {base_head}, expected {expected}"
        )
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_git(
            [
                "clone",
                "--shared",
                "--no-checkout",
                str(base_repo),
                str(workspace),
            ],
            timeout=600,
        )
        run_git(["-C", str(workspace), "checkout", "--detach", expected])
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise

def error_record(
    raw: dict[str, Any],
    test_command: str,
    exc: Exception,
    token_usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": str(raw["instance_id"]),
        "problem": str(raw.get("problem_statement", "")),
        "steps": [],
        "patch": "(no changes)",
        "test_result": {
            "command": test_command,
            "passed": False,
            "output": "",
        },
        "status": "error",
        "success": False,
        "fail": False,
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "token_usage": token_usage
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "num_steps": 0,
        "repo": str(raw.get("repo", "")),
        "base_commit": str(raw.get("base_commit", "")),
    }

def finalize_record(
    generated: dict[str, Any],
    raw: dict[str, Any],
    status: str,
    token_usage: dict[str, int],
) -> dict[str, Any]:
    generated.update(
        {
            "status": status,
            "success": status == "success",
            "fail": status == "fail",
            "error": None,
            "token_usage": token_usage,
            "num_steps": len(generated.get("steps", [])),
            "repo": str(raw["repo"]),
            "base_commit": str(raw["base_commit"]),
        }
    )
    return generated

def main() -> int:
    args = parse_args()
    subset_dir = under_data_root(args.subset_dir, "--subset-dir")
    repos_dir = under_data_root(args.repos_dir, "--repos-dir")
    output_dir = under_data_root(args.output_dir, "--output-dir")
    workspaces_dir = under_data_root(args.workspaces_dir, "--workspaces-dir")
    mapping = load_command_map(args.test_command_map)
    selected = load_subset_tasks(subset_dir, args.start_index, args.num_tasks)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".locks").mkdir(exist_ok=True)
    shared_backend: HFPolicy | None = None
    counts = {"success": 0, "fail": 0, "error": 0, "skipped": 0, "dry_run": 0}

    for position, (task_file, raw) in enumerate(selected, 1):
        output_path = output_dir / task_file.name
        lock_path = output_dir / ".locks" / f"{task_file.stem}.lock"
        with task_lock(lock_path):
            previous = completed_status(output_path)
            if previous and not (previous == "error" and args.retry_errors):
                counts["skipped"] += 1
                print(
                    f"[{position}/{len(selected)}] {task_file.stem} "
                    f"status=skipped previous={previous}"
                )
                continue
            test_command = resolve_test_command(raw, mapping, args.test_command)
            base_repo = (
                repos_dir / repo_slug(str(raw["repo"])) / str(raw["base_commit"])
            )
            workspace = workspaces_dir / task_file.stem
            if args.dry_run:
                counts["dry_run"] += 1
                print(
                    f"[{position}/{len(selected)}] {task_file.stem} dry_run "
                    f"repo_exists={base_repo.is_dir()} output={output_path}"
                )
                continue

            if previous == "error" and args.retry_errors:
                history_dir = output_dir / ".history"
                history_dir.mkdir(exist_ok=True)
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                os.replace(
                    output_path,
                    history_dir / f"{task_file.stem}__{timestamp}.json",
                )
            policy: SWEbenchPolicy | None = None
            staging = output_dir / ".staging" / task_file.stem
            try:
                if not base_repo.is_dir():
                    raise FileNotFoundError(
                        f"prepared repository not found: {base_repo}; "
                        "run scripts/prepare_repo.py first"
                    )
                prepare_isolated_workspace(
                    base_repo, workspace, str(raw["base_commit"])
                )
                task = Task.from_dict(
                    {
                        "task_id": str(raw["instance_id"]),
                        "repo_path": str(workspace),
                        "problem_statement": str(raw["problem_statement"]),
                        "test_command": test_command,
                    }
                )
                if shared_backend is None:
                    shared_backend = HFPolicy(args.model_name, args.adapter_path)
                policy = SWEbenchPolicy.from_shared_backend(task, shared_backend)
                if staging.exists():
                    shutil.rmtree(staging)
                staging.mkdir(parents=True)
                exit_code = run_repository_task(
                    task,
                    policy,
                    max_steps=args.max_steps,
                    trajectory_dir=staging,
                    test_timeout=args.test_timeout,
                )
                generated_files = list(staging.glob("*.json"))
                if len(generated_files) != 1:
                    raise RuntimeError(
                        f"expected one staged trajectory, found {len(generated_files)}"
                    )
                generated = json.loads(
                    generated_files[0].read_text(encoding="utf-8")
                )
                status = "success" if exit_code == 0 else "fail"
                record = finalize_record(
                    generated, raw, status, policy.token_usage()
                )
                write_json_atomic(output_path, record)
                counts[status] += 1
                print(
                    f"[{position}/{len(selected)}] {task_file.stem} "
                    f"status={status} steps={record['num_steps']} "
                    f"tokens={record['token_usage']['total_tokens']}"
                )
            except KeyboardInterrupt:
                print(f"interrupted at {task_file.stem}; no completion marker written")
                raise
            except Exception as exc:
                usage = policy.token_usage() if policy else None
                record = error_record(raw, test_command, exc, usage)
                write_json_atomic(output_path, record)
                counts["error"] += 1
                print(
                    f"[{position}/{len(selected)}] {task_file.stem} "
                    f"status=error {type(exc).__name__}: {exc}"
                )
            finally:
                shutil.rmtree(staging, ignore_errors=True)
                if not args.keep_workspaces:
                    shutil.rmtree(workspace, ignore_errors=True)

    print("summary=" + json.dumps(counts, sort_keys=True))
    return 1 if counts["error"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
