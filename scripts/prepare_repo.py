#!/usr/bin/env python3
"""Prepare immutable SWE-bench repository worktrees at their base commits."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DEFAULT_SUBSET = DATA_ROOT / "swebench" / "subset"
DEFAULT_REPOS = DATA_ROOT / "swebench" / "repos"
REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")


class RepositoryError(RuntimeError):
    """Base class for repository preparation failures."""


class RepositoryNotFound(RepositoryError):
    pass


class CloneFailed(RepositoryError):
    pass


class CommitNotFound(RepositoryError):
    pass


class WorkspaceConflict(RepositoryError):
    pass


@dataclass(frozen=True)
class TaskRef:
    instance_id: str
    repo: str
    base_commit: str
    source_file: str


@dataclass
class PrepareResult:
    instance_id: str
    repo: str
    base_commit: str
    workspace: str
    status: str
    resolved_commit: str = ""
    requirement_files: list[str] | None = None
    error_type: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--repos-dir", type=Path, default=DEFAULT_REPOS)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Prepare only this instance; may be specified multiple times.",
    )
    parser.add_argument("--refresh-mirrors", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_under_data_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be under {root}; got {resolved}")
    return resolved


def run_command(
    command: Sequence[str],
    *,
    timeout: int,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RepositoryError(f"command failed: {command[0]}: {exc}") from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RepositoryError(f"{' '.join(command[:3])} failed: {detail[-2000:]}")
    return result


def check_runtime_requirements(
    subset_dir: Path, repos_dir: Path, min_free_gb: float, timeout: int
) -> str:
    git = shutil.which("git")
    if not git:
        raise RuntimeError("git is required but was not found on PATH")
    version_result = run_command([git, "--version"], timeout=timeout)
    version = version_result.stdout.strip()
    match = re.search(r"(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (2, 25):
        raise RuntimeError(f"git >= 2.25 is required for partial clone/worktree; got {version}")
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"subset directory does not exist: {subset_dir}")
    repos_dir.mkdir(parents=True, exist_ok=True)
    probe = repos_dir / ".write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
    finally:
        probe.unlink(missing_ok=True)
    free_gb = shutil.disk_usage(repos_dir).free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"insufficient disk: {free_gb:.1f} GiB free, require {min_free_gb:.1f} GiB"
        )
    return f"{version}; free_disk={free_gb:.1f}GiB"


def load_tasks(
    subset_dir: Path, instance_ids: Sequence[str], num_tasks: int | None
) -> list[TaskRef]:
    requested = set(instance_ids)
    tasks: list[TaskRef] = []
    for path in sorted(subset_dir.glob("task_*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid task file {path}: {exc}") from exc
        missing = {"instance_id", "repo", "base_commit"} - raw.keys()
        if missing:
            raise ValueError(f"{path} missing required fields: {sorted(missing)}")
        task = TaskRef(
            instance_id=str(raw["instance_id"]),
            repo=str(raw["repo"]),
            base_commit=str(raw["base_commit"]),
            source_file=str(path),
        )
        if not REPO_PATTERN.fullmatch(task.repo):
            raise ValueError(f"{path}: invalid GitHub repo: {task.repo!r}")
        if not COMMIT_PATTERN.fullmatch(task.base_commit):
            raise ValueError(f"{path}: invalid base_commit: {task.base_commit!r}")
        if not requested or task.instance_id in requested:
            tasks.append(task)
    if requested:
        found = {task.instance_id for task in tasks}
        missing_ids = requested - found
        if missing_ids:
            raise ValueError(f"instance ids not found in subset: {sorted(missing_ids)}")
    if num_tasks is not None:
        if num_tasks <= 0:
            raise ValueError("--num-tasks must be positive")
        tasks = tasks[:num_tasks]
    if not tasks:
        raise ValueError(f"no tasks selected from {subset_dir}")
    return tasks


def repo_slug(repo: str) -> str:
    return repo.replace("/", "__")


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def classify_clone_error(repo: str, detail: str) -> RepositoryError:
    lowered = detail.lower()
    if ("repository not found" in lowered or ("not found" in lowered and "repository" in lowered) or "could not read username for 'https://github.com'" in lowered):
        return RepositoryNotFound(f"GitHub repository does not exist or is inaccessible: {repo}")
    return CloneFailed(f"clone failed for {repo}: {detail[-2000:]}")


def ensure_mirror(
    task: TaskRef,
    repos_dir: Path,
    *,
    refresh: bool,
    timeout: int,
) -> tuple[Path, bool]:
    slug = repo_slug(task.repo)
    mirror = repos_dir / ".mirrors" / f"{slug}.git"
    lock = repos_dir / ".locks" / f"{slug}.lock"
    clone_url = f"https://github.com/{task.repo}.git"
    with exclusive_lock(lock):
        if mirror.exists():
            check = run_command(
                ["git", "-C", str(mirror), "rev-parse", "--is-bare-repository"],
                timeout=timeout,
                check=False,
            )
            if check.returncode != 0 or check.stdout.strip() != "true":
                raise WorkspaceConflict(f"mirror path exists but is not a bare repo: {mirror}")
            if refresh:
                run_command(
                    ["git", "-C", str(mirror), "remote", "update", "--prune"],
                    timeout=timeout,
                )
            return mirror, False

        mirror.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{slug}.clone-", dir=mirror.parent)
        )
        shutil.rmtree(temporary)
        command = [
            "git",
            "clone",
            "--mirror",
            "--filter=blob:none",
            clone_url,
            str(temporary),
        ]
        result = run_command(command, timeout=timeout, check=False)
        if result.returncode != 0:
            shutil.rmtree(temporary, ignore_errors=True)
            raise classify_clone_error(task.repo, (result.stderr or result.stdout).strip())
        try:
            os.replace(temporary, mirror)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return mirror, True


def resolve_commit(mirror: Path, task: TaskRef, timeout: int) -> str:
    revision = f"{task.base_commit}^{{commit}}"
    check = run_command(
        ["git", "-C", str(mirror), "cat-file", "-e", revision],
        timeout=timeout,
        check=False,
    )
    if check.returncode != 0:
        fetch = run_command(
            ["git", "-C", str(mirror), "fetch", "--no-tags", "origin", task.base_commit],
            timeout=timeout,
            check=False,
        )
        check = run_command(
            ["git", "-C", str(mirror), "cat-file", "-e", revision],
            timeout=timeout,
            check=False,
        )
        if check.returncode != 0:
            detail = (fetch.stderr or fetch.stdout).strip()
            raise CommitNotFound(
                f"commit {task.base_commit} does not exist in {task.repo}: {detail[-1000:]}"
            )
    resolved = run_command(
        ["git", "-C", str(mirror), "rev-parse", revision], timeout=timeout
    )
    return resolved.stdout.strip()


def discover_requirement_files(workspace: Path) -> list[str]:
    names = (
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "environment.yml",
        "Pipfile",
        "poetry.lock",
    )
    found = [path for name in names if (path := workspace / name).is_file()]
    requirements_dir = workspace / "requirements"
    if requirements_dir.is_dir():
        found.extend(sorted(requirements_dir.glob("*.txt")))
    return sorted({str(path.relative_to(workspace)) for path in found})


def existing_workspace_commit(workspace: Path, timeout: int) -> str | None:
    if not workspace.exists():
        return None
    result = run_command(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise WorkspaceConflict(f"path exists but is not a Git checkout: {workspace}")
    return result.stdout.strip()


def write_metadata(
    repos_dir: Path,
    task: TaskRef,
    workspace: Path,
    resolved: str,
    requirement_files: list[str],
) -> None:
    target = (
        repos_dir
        / ".metadata"
        / repo_slug(task.repo)
        / f"{task.base_commit}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "instance_id": task.instance_id,
        "repo": task.repo,
        "base_commit": task.base_commit,
        "resolved_commit": resolved,
        "workspace": str(workspace),
        "requirement_files": requirement_files,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_task(
    task: TaskRef,
    repos_dir: Path,
    *,
    refresh: bool,
    timeout: int,
    dry_run: bool,
) -> PrepareResult:
    workspace = repos_dir / repo_slug(task.repo) / task.base_commit
    if dry_run:
        return PrepareResult(
            task.instance_id,
            task.repo,
            task.base_commit,
            str(workspace),
            "dry_run",
        )
    mirror, cloned = ensure_mirror(
        task, repos_dir, refresh=refresh, timeout=timeout
    )
    resolved = resolve_commit(mirror, task, timeout)
    current = existing_workspace_commit(workspace, timeout)
    if current:
        if current != resolved:
            raise WorkspaceConflict(
                f"{workspace} is at {current}, expected {resolved}; refusing to overwrite"
            )
        requirement_files = discover_requirement_files(workspace)
        write_metadata(repos_dir, task, workspace, resolved, requirement_files)
        return PrepareResult(
            task.instance_id,
            task.repo,
            task.base_commit,
            str(workspace),
            "skipped_existing",
            resolved,
            requirement_files,
        )

    workspace.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        ["git", "-C", str(mirror), "worktree", "prune"], timeout=timeout
    )
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
        run_command(
            ["git", "-C", str(mirror), "worktree", "remove", "--force", str(workspace)],
            timeout=timeout,
            check=False,
        )
        if workspace.exists():
            shutil.rmtree(workspace)
        raise RepositoryError(
            f"checkout failed for {task.repo}@{resolved}: "
            f"{(result.stderr or result.stdout).strip()[-2000:]}"
        )
    requirement_files = discover_requirement_files(workspace)
    write_metadata(repos_dir, task, workspace, resolved, requirement_files)
    return PrepareResult(
        task.instance_id,
        task.repo,
        task.base_commit,
        str(workspace),
        "prepared_from_new_mirror" if cloned else "prepared_from_cache",
        resolved,
        requirement_files,
    )


def write_report(repos_dir: Path, prerequisite: str, results: list[PrepareResult]) -> Path:
    target = repos_dir / "prepare_report.json"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prerequisites": prerequisite,
        "counts": {
            status: sum(result.status == status for result in results)
            for status in sorted({result.status for result in results})
        },
        "results": [asdict(result) for result in results],
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def main() -> int:
    args = parse_args()
    subset_dir = ensure_under_data_root(args.subset_dir, "--subset-dir")
    repos_dir = ensure_under_data_root(args.repos_dir, "--repos-dir")
    prerequisite = check_runtime_requirements(
        subset_dir, repos_dir, args.min_free_gb, args.timeout
    )
    tasks = load_tasks(subset_dir, args.instance_id, args.num_tasks)
    results: list[PrepareResult] = []
    failures = 0
    for index, task in enumerate(tasks, 1):
        try:
            result = prepare_task(
                task,
                repos_dir,
                refresh=args.refresh_mirrors,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            failures += 1
            result = PrepareResult(
                task.instance_id,
                task.repo,
                task.base_commit,
                str(repos_dir / repo_slug(task.repo) / task.base_commit),
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        results.append(result)
        requirement_count = len(result.requirement_files or [])
        print(
            f"[{index}/{len(tasks)}] {task.instance_id} status={result.status} "
            f"requirements={requirement_count}"
        )
        if result.error:
            print(f"  {result.error_type}: {result.error}")
            if args.fail_fast:
                break
    report = write_report(repos_dir, prerequisite, results)
    print(f"report={report} successes={len(results)-failures} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
