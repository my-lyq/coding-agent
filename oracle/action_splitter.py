"""Split a gold diff into bounded, sequentially executable patch actions."""
from __future__ import annotations

import difflib
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from unidiff import PatchSet

from agent.prompting import canonical_tool_target
from agent.tools import CodingTools
from oracle.expert_builder import OracleBuildError, parse_gold_patch, verify_clean_base


@dataclass(frozen=True)
class PatchAction:
    patch: str
    observation: str
    target_tokens: int
    modified_files: tuple[str, ...]


@dataclass(frozen=True)
class SplitPatchResult:
    actions: tuple[PatchAction, ...]
    final_diff: str
    reference_diff: str
    modified_files: tuple[str, ...]


def _git(
    workspace: Path,
    arguments: Sequence[str],
    *,
    input_text: str | None = None,
    timeout: int = 180,
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
        raise OracleBuildError("git_error", str(exc)) from exc


def _require(result: subprocess.CompletedProcess[str], operation: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise OracleBuildError("patch_split_apply_failed", f"{operation}: {detail[-4000:]}")
    return result.stdout


def _normalise_diff(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").strip().splitlines())


def _file_diff(path: str, before: str, after: str, context: int = 3) -> str:
    body = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=context,
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}" if body else ""


def _first_hunk_patch(diff: str) -> str:
    patch_set = PatchSet(io.StringIO(diff))
    if len(patch_set) != 1 or not patch_set[0]:
        raise OracleBuildError("patch_split_error", "expected one file with at least one hunk")
    patched_file = patch_set[0]
    return (
        f"diff --git {patched_file.source_file} {patched_file.target_file}\n"
        f"--- {patched_file.source_file}\n"
        f"+++ {patched_file.target_file}\n"
        f"{patched_file[0]}"
    )


def _first_edit_candidate(before: str, final: str, take: int) -> str:
    current_lines = before.splitlines(keepends=True)
    final_lines = final.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, current_lines, final_lines, autojunk=False)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_count = old_end - old_start
        new_count = new_end - new_start
        old_take = min(old_count, take) if old_count else 0
        new_take = min(new_count, take) if new_count else 0
        if old_take == 0 and new_take == 0:
            raise OracleBuildError("patch_split_error", "empty edit chunk")
        return "".join(
            current_lines[:old_start]
            + final_lines[new_start : new_start + new_take]
            + current_lines[old_start + old_take :]
        )
    raise OracleBuildError("patch_split_error", "files differ but no edit opcode was found")


class GoldPatchActionSplitter:
    """Produce bounded patch calls and execute each against one workspace state."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_action_tokens: int = 1024,
        timeout: int = 180,
        max_actions: int = 500,
    ) -> None:
        if max_action_tokens <= 0:
            raise ValueError("max_action_tokens must be positive")
        self.tokenizer = tokenizer
        self.max_action_tokens = max_action_tokens
        self.timeout = timeout
        self.max_actions = max_actions

    def target_tokens(self, patch: str) -> int:
        target = canonical_tool_target("apply_patch", {"patch": patch})
        # Qwen adds three assistant/end control tokens around this compact JSON.
        # The final dataset builder recomputes and gates the exact chat-template
        # count; this conservative counter is used to choose chunks.
        return len(self.tokenizer.encode(target, add_special_tokens=False)) + 3

    def _bounded_edit_patch(self, path: str, before: str, final: str) -> str:
        current_lines = before.splitlines(keepends=True)
        final_lines = final.splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(None, current_lines, final_lines, autojunk=False)
        first = next((opcode for opcode in matcher.get_opcodes() if opcode[0] != "equal"), None)
        if first is None:
            raise OracleBuildError("patch_split_error", f"no remaining edit for {path}")
        _, old_start, old_end, new_start, new_end = first
        maximum = max(old_end - old_start, new_end - new_start, 1)
        take = min(maximum, 128)
        while take >= 1:
            candidate = _first_edit_candidate(before, final, take)
            patch = _file_diff(path, before, candidate)
            if patch and self.target_tokens(patch) <= self.max_action_tokens:
                return patch
            take //= 2
        raise OracleBuildError(
            "action_target_too_long",
            f"one indivisible edit in {path} exceeds {self.max_action_tokens} tokens",
        )

    def _apply_action(
        self,
        tools: CodingTools,
        patch: str,
    ) -> PatchAction:
        tokens = self.target_tokens(patch)
        if tokens > self.max_action_tokens:
            raise OracleBuildError(
                "action_target_too_long",
                f"patch target has {tokens} tokens; budget={self.max_action_tokens}",
            )
        try:
            paths = tuple(tools.patch_paths(patch))
        except ValueError as exc:
            raise OracleBuildError("patch_split_error", str(exc)) from exc
        result = tools.apply_patch(patch)
        if not result.ok:
            raise OracleBuildError("patch_split_apply_failed", result.output)
        return PatchAction(patch, result.output, tokens, paths)

    def split_and_apply(
        self,
        workspace: Path,
        gold_patch: str,
        *,
        resolved_commit: str,
    ) -> SplitPatchResult:
        parsed = parse_gold_patch(gold_patch)
        verify_clean_base(workspace, resolved_commit)
        expected_files = tuple(sorted(set(parsed.modified_files)))
        tools = CodingTools(workspace, timeout=self.timeout, max_observation_chars=100_000)

        # Build the gold reference in this isolated workspace, capture exact
        # target contents/diff, then reverse the official patch back to base.
        checked = _git(
            workspace,
            ["apply", "--check", "--recount", "--whitespace=nowarn", "-"],
            input_text=gold_patch,
            timeout=self.timeout,
        )
        _require(checked, "gold git apply --check")
        applied = tools.apply_patch(gold_patch)
        if not applied.ok:
            raise OracleBuildError("git_apply_check_failed", applied.output)
        reference_diff = _require(
            _git(workspace, ["diff", "--no-ext-diff"], timeout=self.timeout),
            "gold git diff",
        )
        final_contents = {
            path: (workspace / path).read_text(encoding="utf-8") for path in expected_files
        }
        reverse_check = _git(
            workspace,
            ["apply", "-R", "--check", "--recount", "--whitespace=nowarn", "-"],
            input_text=gold_patch,
            timeout=self.timeout,
        )
        _require(reverse_check, "gold reverse apply --check")
        reverse = _git(
            workspace,
            ["apply", "-R", "--recount", "--whitespace=nowarn", "-"],
            input_text=gold_patch,
            timeout=self.timeout,
        )
        _require(reverse, "gold reverse apply")
        verify_clean_base(workspace, resolved_commit)

        actions: list[PatchAction] = []
        for path in parsed.modified_files:
            final = final_contents[path]
            while True:
                before = (workspace / path).read_text(encoding="utf-8")
                if before == final:
                    break
                if len(actions) >= self.max_actions:
                    raise OracleBuildError(
                        "patch_split_error", f"exceeded {self.max_actions} actions"
                    )
                whole_file_patch = _file_diff(path, before, final)
                if self.target_tokens(whole_file_patch) <= self.max_action_tokens:
                    patch = whole_file_patch
                else:
                    first_hunk = _first_hunk_patch(whole_file_patch)
                    if self.target_tokens(first_hunk) <= self.max_action_tokens:
                        patch = first_hunk
                    else:
                        patch = self._bounded_edit_patch(path, before, final)
                actions.append(self._apply_action(tools, patch))

        final_diff = _require(
            _git(workspace, ["diff", "--no-ext-diff"], timeout=self.timeout),
            "reconstructed git diff",
        )
        changed = tuple(
            sorted(
                line
                for line in _require(
                    _git(workspace, ["diff", "--name-only"], timeout=self.timeout),
                    "reconstructed changed files",
                ).splitlines()
                if line
            )
        )
        if changed != expected_files:
            raise OracleBuildError(
                "modified_files_mismatch",
                f"expected={expected_files}, reconstructed={changed}",
            )
        if _normalise_diff(final_diff) != _normalise_diff(reference_diff):
            raise OracleBuildError(
                "reconstructed_gold_diff_mismatch",
                "sequential patch actions do not reconstruct the gold repository diff",
            )
        return SplitPatchResult(tuple(actions), final_diff, reference_diff, expected_files)
