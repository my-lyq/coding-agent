"""Fresh-base applicability proxy for model-generated repository patches.

This module never runs tests and never reports benchmark resolution. A positive
result means only that git apply --check accepts the final diff at the task's
clean base commit.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DEFAULT_REPOS = DATA_ROOT / "swebench" / "repos"
DEFAULT_WORKSPACES = DATA_ROOT / "rollouts" / "patch_proxy"


def repo_slug(repo: str) -> str:
    return repo.replace("/", "__")


def _git(
    repo: Path | None,
    arguments: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    command = ["git"]
    if repo is not None:
        command.extend(["-C", str(repo)])
    command.extend(arguments)
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def patch_statistics(patch: str, tokenizer: Any | None = None) -> dict[str, Any]:
    files: list[str] = []
    added = 0
    deleted = 0
    for line in patch.splitlines():
        if line.startswith("+++ "):
            value = line[4:].split("\t", 1)[0]
            if value != "/dev/null":
                if value.startswith("b/"):
                    value = value[2:]
                if value not in files:
                    files.append(value)
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
    patch_tokens = (
        len(tokenizer.encode(patch, add_special_tokens=False))
        if tokenizer is not None and patch.strip() and patch.strip() != "(no changes)"
        else 0
    )
    return {
        "modified_files": sorted(files),
        "modified_file_count": len(files),
        "patch_lines_added": added,
        "patch_lines_deleted": deleted,
        "patch_size_lines": added + deleted,
        "patch_size_tokens": patch_tokens,
    }


def evaluate_patch(
    *,
    instance_id: str,
    repo: str,
    base_commit: str,
    patch: str,
    repos_dir: Path = DEFAULT_REPOS,
    workspaces_dir: Path = DEFAULT_WORKSPACES,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Check one final patch in a newly cloned clean base-commit workspace."""
    stats = patch_statistics(patch, tokenizer)
    result: dict[str, Any] = {
        "instance_id": instance_id,
        "base_commit": base_commit,
        "proxy_patch_applicable": False,
        "clean_base_verified": False,
        "check_command": "git apply --check --recount --whitespace=nowarn -",
        "error": None,
        **stats,
    }
    if not patch.strip() or patch.strip() == "(no changes)":
        result["error"] = "empty_patch"
        return result

    base_repo = repos_dir / repo_slug(repo) / base_commit
    workspaces_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(prefix=f"{instance_id.replace('/', '__')}--", dir=workspaces_dir)
    )
    try:
        if not base_repo.is_dir():
            raise FileNotFoundError(f"prepared repository not found: {base_repo}")
        clone = _git(None, ["clone", "--shared", "--no-checkout", str(base_repo), str(workspace)])
        if clone.returncode != 0:
            raise RuntimeError((clone.stderr or clone.stdout).strip())
        checkout = _git(workspace, ["checkout", "--detach", base_commit])
        if checkout.returncode != 0:
            raise RuntimeError((checkout.stderr or checkout.stdout).strip())
        head = _git(workspace, ["rev-parse", "HEAD"])
        if head.returncode != 0 or head.stdout.strip() != base_commit:
            raise RuntimeError("fresh workspace did not resolve to requested base_commit")
        status = _git(workspace, ["status", "--porcelain", "--untracked-files=all"])
        if status.returncode != 0:
            raise RuntimeError((status.stderr or status.stdout).strip())
        if status.stdout.strip():
            raise RuntimeError("fresh base workspace is not clean")
        result["clean_base_verified"] = True
        checked = _git(
            workspace,
            ["apply", "--check", "--recount", "--whitespace=nowarn", "-"],
            input_text=patch,
        )
        result["proxy_patch_applicable"] = checked.returncode == 0
        if checked.returncode != 0:
            result["error"] = (checked.stderr or checked.stdout).strip() or "git_apply_check_failed"
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--repos-dir", type=Path, default=DEFAULT_REPOS)
    parser.add_argument("--workspaces-dir", type=Path, default=DEFAULT_WORKSPACES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    task = json.loads(args.task.read_text(encoding="utf-8"))
    result = evaluate_patch(
        instance_id=str(task["instance_id"]),
        repo=str(task["repo"]),
        base_commit=str(task["base_commit"]),
        patch=str(trajectory.get("patch", "")),
        repos_dir=args.repos_dir,
        workspaces_dir=args.workspaces_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["proxy_patch_applicable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
