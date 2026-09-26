"""Build verifiable, thought-free expert trajectories from SWE-bench gold patches.

The gold patch is teacher-side supervision.  It is never included in the model
prompt: only the issue and observations produced by real repository tools are.
This module deliberately does not execute tests or claim benchmark resolution.
"""
from __future__ import annotations

import io
import json
import re
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from agent.tools import CodingTools, ToolResult
from agent.tools.schema import TOOL_SCHEMAS


class OracleBuildError(RuntimeError):
    """A sample failed a required oracle-data quality gate."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class AffectedRange:
    start: int
    end: int


@dataclass(frozen=True)
class FileChange:
    path: str
    ranges: tuple[AffectedRange, ...]
    search_candidates: tuple[str, ...]
    is_added: bool
    is_removed: bool


@dataclass(frozen=True)
class ParsedGoldPatch:
    files: tuple[FileChange, ...]
    added_lines: int
    removed_lines: int

    @property
    def modified_files(self) -> tuple[str, ...]:
        return tuple(change.path for change in self.files)

    @property
    def patch_size(self) -> int:
        return self.added_lines + self.removed_lines


def _normalise_diff_path(value: str) -> str:
    value = value.strip()
    if value in {"", "/dev/null"}:
        return value
    if value.startswith(("a/", "b/")):
        value = value[2:]
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise OracleBuildError("unsafe_patch_path", f"unsafe diff path: {value!r}")
    return path.as_posix()


def _merge_ranges(ranges: Iterable[AffectedRange]) -> tuple[AffectedRange, ...]:
    merged: list[AffectedRange] = []
    for current in sorted(ranges, key=lambda item: (item.start, item.end)):
        if merged and current.start <= merged[-1].end + 20:
            previous = merged[-1]
            merged[-1] = AffectedRange(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    return tuple(merged)


def _useful_search_line(lines: Sequence[str]) -> tuple[str, ...]:
    """Prefer stable definitions, then sufficiently distinctive source lines."""
    cleaned: list[str] = []
    for value in lines:
        candidate = value.strip()
        if not candidate or candidate.startswith(("#", "//", "/*", "*")):
            continue
        if len(candidate) < 4 or len(candidate) > 180:
            continue
        if candidate not in cleaned:
            cleaned.append(candidate)
    definitions = [
        value
        for value in cleaned
        if re.match(r"^(?:async\s+def|def|class|function|func|public|private|protected)\b", value)
    ]
    return tuple((definitions + cleaned)[:3])


def parse_gold_patch(patch: str) -> ParsedGoldPatch:
    """Parse unified diff metadata without applying it."""
    if not patch or not patch.strip():
        raise OracleBuildError("empty_patch", "gold patch is empty")
    try:
        from unidiff import PatchSet

        patch_set = PatchSet(io.StringIO(patch))
    except Exception as exc:
        raise OracleBuildError("patch_parse_error", f"cannot parse gold patch: {exc}") from exc
    if not patch_set:
        raise OracleBuildError("patch_parse_error", "gold patch has no file diffs")

    files: list[FileChange] = []
    added = 0
    removed = 0
    for patched_file in patch_set:
        source = _normalise_diff_path(str(patched_file.source_file))
        target = _normalise_diff_path(str(patched_file.target_file))
        is_added = bool(patched_file.is_added_file)
        is_removed = bool(patched_file.is_removed_file)
        path = target if target != "/dev/null" else source
        if not path:
            raise OracleBuildError("patch_parse_error", "diff contains an empty target path")

        ranges: list[AffectedRange] = []
        candidates: list[str] = []
        for hunk in patched_file:
            start = max(1, int(hunk.source_start or 1))
            source_length = max(1, int(hunk.source_length or 1))
            ranges.append(AffectedRange(max(1, start - 30), start + source_length + 30))
            source_lines: list[str] = []
            for line in hunk:
                if line.is_added:
                    added += 1
                elif line.is_removed:
                    removed += 1
                    source_lines.append(str(line.value).rstrip("\n"))
                elif line.is_context:
                    source_lines.append(str(line.value).rstrip("\n"))
            candidates.extend(_useful_search_line(source_lines))
        files.append(
            FileChange(
                path=path,
                ranges=_merge_ranges(ranges),
                search_candidates=tuple(dict.fromkeys(candidates)),
                is_added=is_added,
                is_removed=is_removed,
            )
        )
    return ParsedGoldPatch(tuple(files), added, removed)


def _git(
    workspace: Path,
    arguments: Sequence[str],
    *,
    input_text: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OracleBuildError("git_error", f"git {' '.join(arguments)} failed: {exc}") from exc


def _require_git_success(
    result: subprocess.CompletedProcess[str], reason: str, operation: str
) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise OracleBuildError(reason, f"{operation} failed: {detail[-4000:]}")
    return result.stdout


def verify_clean_base(workspace: Path, expected_commit: str) -> str:
    status = _require_git_success(
        _git(workspace, ["status", "--porcelain"]), "git_error", "git status"
    )
    if status.strip():
        raise OracleBuildError("dirty_base", f"base workspace is dirty: {status[:2000]}")
    head = _require_git_success(
        _git(workspace, ["rev-parse", "HEAD"]), "base_commit_mismatch", "git rev-parse"
    ).strip()
    if head != expected_commit:
        raise OracleBuildError(
            "base_commit_mismatch", f"workspace HEAD={head}, expected={expected_commit}"
        )
    return head


def _step(
    phase: str, tool: str, arguments: dict[str, Any], result: ToolResult
) -> dict[str, Any]:
    return {
        "phase": phase,
        "tool": tool,
        "arguments": arguments,
        "observation": result.output,
        "tool_success": result.ok,
        "observation_source": "repository_tool_execution",
        "controller_intervened": False,
        "repair_applied": False,
        "invalid_action": False,
    }


def _require_tool(step: dict[str, Any]) -> None:
    if not step["tool_success"]:
        raise OracleBuildError(
            "tool_error", f"{step['tool']} failed: {step['observation'][:2000]}"
        )


def _select_search_query(tools: CodingTools, change: FileChange) -> tuple[str, ToolResult]:
    for query in change.search_candidates:
        result = tools.search_code(query=query, path=change.path, max_results=20)
        if result.ok and result.output != "no matches":
            return query, result
    # The observation is still real, but an expert navigation trace must locate
    # something.  Reject instead of inventing a successful search result.
    raise OracleBuildError(
        "search_target_not_found", f"no source-context query matched {change.path}"
    )


def render_model_prompt(problem_statement: str, navigation_steps: Sequence[dict[str, Any]]) -> str:
    transcript = [
        {
            "phase": step["phase"],
            "action": {"tool": step["tool"], "arguments": step["arguments"]},
            "observation": step["observation"],
            "tool_success": step["tool_success"],
        }
        for step in navigation_steps
    ]
    return (
        "You are a coding agent. Use one structured tool call to make the required change.\n"
        "Issue:\n"
        f"{problem_statement.strip()}\n\n"
        "Repository navigation transcript:\n"
        f"{json.dumps(transcript, ensure_ascii=False, indent=2)}\n\n"
        "Return exactly one JSON object with fields tool and arguments."
    )


def detect_prompt_leakage(
    prompt: str,
    *,
    gold_patch: str,
    test_patch: str,
    fail_to_pass: Any = None,
    pass_to_pass: Any = None,
) -> list[str]:
    """Return explicit teacher/evaluation artifacts found in a model prompt."""
    leaks: list[str] = []
    if gold_patch.strip() and gold_patch.strip() in prompt:
        leaks.append("gold_patch")
    if test_patch.strip() and test_patch.strip() in prompt:
        leaks.append("test_patch")
    upper = prompt.upper()
    if "FAIL_TO_PASS" in upper:
        leaks.append("FAIL_TO_PASS")
    if "PASS_TO_PASS" in upper:
        leaks.append("PASS_TO_PASS")
    # Values can be checked when they are informative; very short tokens (for
    # example "[]") would create meaningless false positives.
    for label, value in (("FAIL_TO_PASS_values", fail_to_pass), ("PASS_TO_PASS_values", pass_to_pass)):
        if value in (None, "", [], ()):  # type: ignore[comparison-overlap]
            continue
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if len(serialized) >= 16 and serialized in prompt:
            leaks.append(label)
    return sorted(set(leaks))


def _validate_arguments(tool: str, arguments: dict[str, Any]) -> None:
    schemas = {schema["name"]: schema["parameters"] for schema in TOOL_SCHEMAS}
    if tool not in schemas:
        raise OracleBuildError("schema_invalid", f"unknown tool: {tool}")
    schema = schemas[tool]
    required = set(schema.get("required", []))
    missing = required - arguments.keys()
    if missing:
        raise OracleBuildError("schema_invalid", f"{tool} missing arguments: {sorted(missing)}")
    if schema.get("additionalProperties") is False:
        unexpected = arguments.keys() - schema.get("properties", {}).keys()
        if unexpected:
            raise OracleBuildError(
                "schema_invalid", f"{tool} has unexpected arguments: {sorted(unexpected)}"
            )
    for name, value in arguments.items():
        expected = schema["properties"][name].get("type")
        valid = (
            (expected == "string" and isinstance(value, str))
            or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or expected not in {"string", "integer"}
        )
        if not valid:
            raise OracleBuildError("schema_invalid", f"{tool}.{name} must be {expected}")


def validate_expert_trajectory(trajectory: dict[str, Any]) -> None:
    required = {
        "instance_id",
        "repo",
        "base_commit",
        "split",
        "problem_statement",
        "teacher_source",
        "quality_tier",
        "steps",
        "model_prompt",
        "model_response",
        "verification",
    }
    missing = required - trajectory.keys()
    if missing:
        raise OracleBuildError("schema_invalid", f"trajectory missing: {sorted(missing)}")
    if trajectory["teacher_source"] != "swebench_gold_oracle":
        raise OracleBuildError("schema_invalid", "unexpected teacher_source")
    if trajectory["quality_tier"] != "B":
        raise OracleBuildError("schema_invalid", "oracle sample must be Tier B")
    if trajectory["split"] not in {"train", "validation"}:
        raise OracleBuildError("split_mismatch", "split must be train or validation")
    steps = trajectory["steps"]
    if not isinstance(steps, list) or not steps:
        raise OracleBuildError("schema_invalid", "steps must be a non-empty list")
    allowed_phases = {
        "LOCATE": {"list_files", "search_code"},
        "UNDERSTAND": {"read_file", "search_code"},
        "MODIFY": {"apply_patch"},
    }
    for step in steps:
        if not isinstance(step, dict):
            raise OracleBuildError("schema_invalid", "each step must be an object")
        phase, tool = step.get("phase"), step.get("tool")
        if tool not in allowed_phases.get(phase, set()):
            raise OracleBuildError("schema_invalid", f"tool {tool!r} is illegal in {phase!r}")
        arguments = step.get("arguments")
        if not isinstance(arguments, dict):
            raise OracleBuildError("schema_invalid", "step arguments must be an object")
        _validate_arguments(tool, arguments)
        if step.get("tool_success") is not True:
            raise OracleBuildError("tool_error", f"unsuccessful expert tool step: {tool}")
        if step.get("observation_source") != "repository_tool_execution":
            raise OracleBuildError("observation_not_real", "invalid observation source")
        for field in ("controller_intervened", "repair_applied", "invalid_action"):
            if step.get(field) is not False:
                raise OracleBuildError("schema_invalid", f"{field} must be false")
    verification = trajectory["verification"]
    expected_verification = {
        "level": "patch_applicable_only",
        "git_apply_check": True,
        "official_resolved": None,
    }
    if verification != expected_verification:
        raise OracleBuildError("schema_invalid", "invalid verification claim")


def build_expert_trajectory(
    record: dict[str, Any],
    workspace: Path,
    *,
    split: str,
    dataset_name: str,
    dataset_split: str,
    dataset_revision: str,
    raw_sha256: str,
    resolved_commit: str,
    timeout: int = 120,
) -> dict[str, Any]:
    if dataset_split != "dev":
        raise OracleBuildError("source_split_invalid", "oracle data must come from dev")
    gold_patch = str(record.get("patch", ""))
    parsed = parse_gold_patch(gold_patch)
    if any(change.is_added or change.is_removed for change in parsed.files):
        raise OracleBuildError(
            "unsupported_file_operation", "current apply_patch tool only modifies existing files"
        )
    verify_clean_base(workspace, resolved_commit)
    for change in parsed.files:
        if not (workspace / change.path).is_file():
            raise OracleBuildError("target_file_missing", f"target does not exist: {change.path}")

    checked = _git(workspace, ["apply", "--check", "--recount", "--whitespace=nowarn", "-"], input_text=gold_patch, timeout=timeout)
    _require_git_success(checked, "git_apply_check_failed", "git apply --check")

    tools = CodingTools(workspace, timeout=timeout, max_observation_chars=100_000)
    steps: list[dict[str, Any]] = []
    listed: set[str] = set()
    for change in parsed.files:
        parent = str(PurePosixPath(change.path).parent)
        if parent == "":
            parent = "."
        if parent not in listed:
            arguments = {"path": parent, "max_entries": 500}
            step = _step("LOCATE", "list_files", arguments, tools.list_files(**arguments))
            _require_tool(step)
            steps.append(step)
            listed.add(parent)

        query, result = _select_search_query(tools, change)
        search_arguments = {"query": query, "path": change.path, "max_results": 20}
        step = _step("LOCATE", "search_code", search_arguments, result)
        _require_tool(step)
        steps.append(step)

    for change in parsed.files:
        for affected in change.ranges:
            arguments = {
                "path": change.path,
                "start_line": affected.start,
                "end_line": affected.end,
            }
            step = _step("UNDERSTAND", "read_file", arguments, tools.read_file(**arguments))
            _require_tool(step)
            steps.append(step)

    prompt = render_model_prompt(str(record["problem_statement"]), steps)
    leaks = detect_prompt_leakage(
        prompt,
        gold_patch=gold_patch,
        test_patch=str(record.get("test_patch", "")),
        fail_to_pass=record.get("FAIL_TO_PASS"),
        pass_to_pass=record.get("PASS_TO_PASS"),
    )
    if leaks:
        raise OracleBuildError("prompt_leakage", f"prompt contains: {', '.join(leaks)}")

    response = {"tool": "apply_patch", "arguments": {"patch": gold_patch}}
    applied = tools.apply_patch(gold_patch)
    modify_step = _step("MODIFY", "apply_patch", response["arguments"], applied)
    _require_tool(modify_step)
    steps.append(modify_step)

    diff = _require_git_success(_git(workspace, ["diff", "--no-ext-diff"]), "git_error", "git diff")
    if not diff.strip():
        raise OracleBuildError("empty_git_diff", "patch applied but git diff is empty")
    changed_output = _require_git_success(
        _git(workspace, ["diff", "--name-only"]), "git_error", "git diff --name-only"
    )
    changed = sorted(line for line in changed_output.splitlines() if line)
    expected = sorted(set(parsed.modified_files))
    if changed != expected:
        raise OracleBuildError(
            "modified_files_mismatch", f"expected modified files {expected}, observed {changed}"
        )

    trajectory: dict[str, Any] = {
        "schema_version": 2,
        "instance_id": str(record["instance_id"]),
        "repo": str(record["repo"]),
        "base_commit": str(record["base_commit"]),
        "resolved_commit": resolved_commit,
        "split": split,
        "dataset": {
            "name": dataset_name,
            "split": dataset_split,
            "revision": dataset_revision,
            "raw_sha256": raw_sha256,
        },
        "problem_statement": str(record["problem_statement"]),
        "teacher_source": "swebench_gold_oracle",
        "quality_tier": "B",
        "steps": steps,
        "model_prompt": prompt,
        "model_response": response,
        "modified_files": expected,
        "patch_stats": {
            "added_lines": parsed.added_lines,
            "removed_lines": parsed.removed_lines,
            "total_changed_lines": parsed.patch_size,
            "modified_file_count": len(expected),
        },
        "git_diff": diff,
        "verification": {
            "level": "patch_applicable_only",
            "git_apply_check": True,
            "official_resolved": None,
        },
    }
    validate_expert_trajectory(trajectory)
    return trajectory


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentage
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def numeric_summary(values: Sequence[int], *, include_p95: bool = False) -> dict[str, float]:
    result = {
        "mean": statistics.fmean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
    }
    if include_p95:
        result["p95"] = percentile(values, 0.95)
    return result


def tool_frequency(trajectories: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        step["tool"] for trajectory in trajectories for step in trajectory.get("steps", [])
    )
    return dict(sorted(counts.items()))
