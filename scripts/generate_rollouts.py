#!/usr/bin/env python3
"""Generate one or more SWE-bench Coding Agent trajectories."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.executor import Step  # noqa: E402
from agent.task import Task  # noqa: E402
from rollout.policy import HFBackend, ReActPolicy  # noqa: E402
from rollout.recorder import TrajectoryRecorder  # noqa: E402
from rollout.runner import AgentRolloutRunner  # noqa: E402

DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DEFAULT_SUBSET = DATA_ROOT / "swebench" / "subset"
DEFAULT_RAW = DATA_ROOT / "swebench" / "raw" / "swebench_lite.json"
DEFAULT_REPOS = DATA_ROOT / "swebench" / "repos"
DEFAULT_OUTPUT = DATA_ROOT / "trajectories" / "raw"
DEFAULT_WORKSPACES = DATA_ROOT / "rollouts" / "single"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--instance-id")
    selection.add_argument("--num-tasks", type=int)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--raw-file", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--repos-dir", type=Path, default=DEFAULT_REPOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workspaces-dir", type=Path, default=DEFAULT_WORKSPACES)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--test-command")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-action-tokens", type=int, default=1024)
    parser.add_argument("--test-timeout", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--heldout", action="store_true")
    return parser.parse_args()

def require_data_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be under {root}; got {resolved}")
    return resolved

def find_subset_task(
    subset_dir: Path, instance_id: str
) -> tuple[Path, dict[str, Any]]:
    for path in sorted(subset_dir.glob("task_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("instance_id") == instance_id:
            return path, value
    raise ValueError(f"instance_id not found in subset: {instance_id}")

def find_raw_task(raw_file: Path, instance_id: str) -> dict[str, Any] | None:
    records = json.loads(raw_file.read_text(encoding="utf-8"))
    return next(
        (record for record in records if record.get("instance_id") == instance_id),
        None,
    )

def parse_test_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]

def derive_test_command(raw: dict[str, Any] | None, base_repo: Path) -> str:
    """Use FAIL_TO_PASS file paths without exposing or applying the gold test patch."""
    test_ids = parse_test_ids(raw.get("FAIL_TO_PASS") if raw else None)
    if raw and raw.get("repo") == "django/django":
        labels = []
        for test_id in test_ids:
            match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*) \(([^)]+)\)", test_id)
            if match:
                label = f"{match.group(2)}.{match.group(1)}"
                if label not in labels:
                    labels.append(label)
        if labels:
            return "python tests/runtests.py --verbosity 1 " + " ".join(
                shlex.quote(label) for label in labels
            )
    files: list[str] = []
    for test_id in test_ids:
        candidate = test_id.split("::", 1)[0]
        if candidate not in files and (base_repo / candidate).is_file():
            files.append(candidate)
    if files:
        return "python -m pytest -q " + " ".join(shlex.quote(path) for path in files)
    return "python -m pytest -q"

def derive_metadata_free_test_command(task: dict[str, Any]) -> str:
    """Choose a repository-level command without gold/test evaluation metadata."""
    if task.get("repo") == "django/django":
        return "python tests/runtests.py --verbosity 1"
    return "python -m pytest -q"

def repo_slug(repo: str) -> str:
    return repo.replace("/", "__")

def run_git(arguments: list[str], timeout: int = 600) -> str:
    result = subprocess.run(
        ["git", *arguments],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()

def create_workspace(base_repo: Path, workspace: Path, base_commit: str) -> None:
    expected = run_git(
        ["-C", str(base_repo), "rev-parse", f"{base_commit}^{{commit}}"]
    )
    current = run_git(["-C", str(base_repo), "rev-parse", "HEAD"])
    if current != expected:
        raise RuntimeError(
            f"base repository is at {current}, expected {expected}"
        )
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_git(
            ["clone", "--shared", "--no-checkout", str(base_repo), str(workspace)]
        )
        run_git(["-C", str(workspace), "checkout", "--detach", expected])
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise

@contextlib.contextmanager
def output_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def save_infrastructure_error(
    recorder: TrajectoryRecorder,
    raw: dict[str, Any],
    command: str,
    exc: Exception,
    overwrite: bool,
) -> None:
    step = Step(
        thought="The rollout infrastructure failed before the Agent could continue.",
        action="error",
        action_input={},
        observation=f"{type(exc).__name__}: {exc}",
        success=False,
    )
    record = recorder.build_record(
        instance_id=str(raw["instance_id"]),
        problem_statement=str(raw["problem_statement"]),
        steps=[step],
        modified_files=[],
        patch="(no changes)",
        test_result={
            "command": command,
            "passed": False,
            "output": f"{type(exc).__name__}: {exc}",
        },
        success=False,
        rollout_test_passed=False,
        benchmark_resolved=None,
        token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    recorder.save(record, overwrite=overwrite)

def select_tasks(args: argparse.Namespace, subset_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    if args.instance_id:
        return [find_subset_task(subset_dir, args.instance_id)]
    if args.num_tasks is None or args.num_tasks <= 0:
        raise ValueError("--num-tasks must be positive")
    paths = sorted(subset_dir.glob("task_*.json"))[: args.num_tasks]
    if len(paths) < args.num_tasks:
        raise ValueError(f"requested {args.num_tasks} tasks, found {len(paths)} in {subset_dir}")
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]

def main() -> int:
    args = parse_args()
    subset_dir = require_data_path(args.subset_dir, "--subset-dir")
    raw_file = require_data_path(args.raw_file, "--raw-file")
    repos_dir = require_data_path(args.repos_dir, "--repos-dir")
    output_dir = require_data_path(args.output_dir, "--output-dir")
    workspaces_dir = require_data_path(args.workspaces_dir, "--workspaces-dir")
    selected = select_tasks(args, subset_dir)
    raw_index: dict[str, dict[str, Any]] = {}
    if not args.heldout:
        raw_records = json.loads(raw_file.read_text(encoding="utf-8"))
        raw_index = {str(item.get("instance_id")): item for item in raw_records}
    backend: HFBackend | None = None
    counts = {"success": 0, "fail": 0, "error": 0, "skipped": 0, "dry_run": 0}
    last_code = 0

    for position, (task_file, raw) in enumerate(selected, 1):
        instance_id = str(raw["instance_id"])
        base_repo = repos_dir / repo_slug(str(raw["repo"])) / str(raw["base_commit"])
        raw_task = raw_index.get(instance_id)
        test_command = args.test_command or (
            derive_metadata_free_test_command(raw)
            if args.heldout
            else derive_test_command(raw_task, base_repo)
        )
        output_path = output_dir / task_file.name
        workspace = workspaces_dir / task_file.stem
        lock_path = output_dir / ".locks" / f"{task_file.stem}.lock"
        prefix = f"[{position}/{len(selected)}] " if len(selected) > 1 else ""
        print(f"{prefix}instance_id={instance_id}")
        print(f"{prefix}base_repo={base_repo}")
        print(f"{prefix}test_command={test_command}")
        print(f"{prefix}output={output_path}")

        if args.dry_run:
            counts["dry_run"] += 1
            print(f"{prefix}repo_exists={base_repo.is_dir()} status=dry_run")
            continue

        with output_lock(lock_path):
            if output_path.exists() and not args.overwrite:
                counts["skipped"] += 1
                print(f"{prefix}status=skipped trajectory_exists={output_path}")
                continue
            recorder = TrajectoryRecorder(output_path)
            try:
                if not base_repo.is_dir():
                    raise FileNotFoundError(
                        f"prepared repository not found: {base_repo}; "
                        "run scripts/prepare_repo.py first"
                    )
                create_workspace(base_repo, workspace, str(raw["base_commit"]))
                task = Task.from_dict({
                    "task_id": instance_id,
                    "repo_path": str(workspace),
                    "problem_statement": str(raw["problem_statement"]),
                    "test_command": test_command,
                })
                if backend is None:
                    backend = HFBackend(args.model_name, args.adapter_path)
                policy = ReActPolicy(
                    task, args.model_name, args.adapter_path, backend=backend,
                    max_action_tokens=args.max_action_tokens,
                )
                runner = AgentRolloutRunner(
                    task, policy, recorder,
                    max_steps=args.max_steps, test_timeout=args.test_timeout,
                )
                result = runner.run(overwrite=args.overwrite)
                status = "success" if result.success else "fail"
                counts[status] += 1
                last_code = max(last_code, 0 if result.success else 1)
                print(
                    f"{prefix}status={status} steps={result.num_steps} "
                    f"modified_files={len(result.modified_files)}"
                )
            except KeyboardInterrupt:
                print(f"{prefix}interrupted; no incomplete trajectory was committed")
                raise
            except Exception as exc:
                counts["error"] += 1
                last_code = 2
                print(f"{prefix}status=error {type(exc).__name__}: {exc}")
                save_infrastructure_error(
                    recorder, raw, test_command, exc, args.overwrite
                )
            finally:
                if not args.keep_workspace:
                    shutil.rmtree(workspace, ignore_errors=True)

    print("summary=" + json.dumps(counts, sort_keys=True))
    return last_code

if __name__ == "__main__":
    raise SystemExit(main())
