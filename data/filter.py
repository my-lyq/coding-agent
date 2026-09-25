"""Quality gates for raw coding-agent trajectories."""
from __future__ import annotations
import re
from typing import Any

EMPTY_PATCH_MARKERS = {"", "(no changes)", "no changes", "null", "none"}

def _steps(sample: dict[str, Any]) -> list[dict[str, Any]]:
    steps = sample.get("steps", sample.get("trajectory", []))
    return steps if isinstance(steps, list) else []

def _patch(sample: dict[str, Any]) -> str:
    value = sample.get("final_patch", sample.get("answer", ""))
    return value if isinstance(value, str) else ""

def modified_files(patch: str) -> set[str]:
    """Return files changed by a unified diff, including deleted files."""
    files: set[str] = set()
    for pattern in (r"^\+\+\+\s+(?:b/)?(.+)$", r"^---\s+(?:a/)?(.+)$"):
        for match in re.finditer(pattern, patch, re.MULTILINE):
            path = match.group(1).strip()
            if path != "/dev/null":
                files.add(path)
    return files

def quality_issues(sample: dict[str, Any], max_modified_files: int = 3) -> list[str]:
    """Explain rejection reasons; only the final test determines task success."""
    issues: list[str] = []
    tests = [step for step in _steps(sample) if step.get("action") == "run_test"]
    if not tests:
        issues.append("missing_test")
    elif tests[-1].get("success") is not True:
        issues.append("final_test_failed")
    patch = _patch(sample).strip()
    if patch.lower() in EMPTY_PATCH_MARKERS or not re.search(r"^[+-](?![+-])", patch, re.MULTILINE):
        issues.append("empty_patch")
    file_count = len(modified_files(patch))
    if file_count > max_modified_files:
        issues.append(f"too_many_modified_files:{file_count}>{max_modified_files}")
    return issues

def quality_filter(sample: dict[str, Any], max_modified_files: int = 3) -> bool:
    """Accept only a tested, non-empty, reasonably scoped patch."""
    return not quality_issues(sample, max_modified_files)
