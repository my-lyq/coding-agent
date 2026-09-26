from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from oracle.expert_builder import (
    OracleBuildError,
    build_expert_trajectory,
    detect_prompt_leakage,
    parse_gold_patch,
)


GOLD = """diff --git a/example.py b/example.py
index 89e6c98..2f49857 100644
--- a/example.py
+++ b/example.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


def _git(path: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=path, text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def _record(commit: str) -> dict[str, object]:
    return {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": commit,
        "problem_statement": "The add helper subtracts instead of adding.",
        "patch": GOLD,
        "test_patch": "diff --git a/test_example.py b/test_example.py\n+hidden test\n",
        "FAIL_TO_PASS": ["test_example.py::test_add"],
        "PASS_TO_PASS": [],
    }


def test_gold_leakage_test() -> None:
    prompt = f"Issue only\n{GOLD}"
    assert "gold_patch" in detect_prompt_leakage(prompt, gold_patch=GOLD, test_patch="")


def test_test_patch_leakage_test() -> None:
    test_patch = "diff --git a/test_x.py b/test_x.py\n+secret assertion\n"
    leaks = detect_prompt_leakage(
        f"Issue\n{test_patch}", gold_patch=GOLD, test_patch=test_patch
    )
    assert "test_patch" in leaks


def test_split_leakage_test() -> None:
    manifest = {
        "train_instance_ids": ["a", "b"],
        "validation_instance_ids": ["c", "d"],
    }
    assert set(manifest["train_instance_ids"]).isdisjoint(
        manifest["validation_instance_ids"]
    )
    manifest["validation_instance_ids"].append("a")
    assert not set(manifest["train_instance_ids"]).isdisjoint(
        manifest["validation_instance_ids"]
    )


def test_build_expert_trajectory_uses_real_observations(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "example.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    _git(tmp_path, "add", "example.py")
    _git(tmp_path, "commit", "-qm", "base")
    commit = _git(tmp_path, "rev-parse", "HEAD")

    trajectory = build_expert_trajectory(
        _record(commit),
        tmp_path,
        split="train",
        dataset_name="SWE-bench/SWE-bench",
        dataset_split="dev",
        dataset_revision="revision",
        raw_sha256="checksum",
        resolved_commit=commit,
    )

    assert trajectory["verification"]["official_resolved"] is None
    assert trajectory["quality_tier"] == "B"
    assert all(step["observation_source"] == "repository_tool_execution" for step in trajectory["steps"])
    assert GOLD not in trajectory["model_prompt"]
    assert trajectory["model_response"]["arguments"]["patch"] == GOLD
    assert _git(tmp_path, "diff", "--name-only") == "example.py"


def test_non_dev_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OracleBuildError, match="must come from dev"):
        build_expert_trajectory(
            _record("0" * 40),
            tmp_path,
            split="train",
            dataset_name="SWE-bench/SWE-bench",
            dataset_split="test",
            dataset_revision="revision",
            raw_sha256="checksum",
            resolved_commit="0" * 40,
        )


def test_parse_patch_metadata() -> None:
    parsed = parse_gold_patch(GOLD)
    assert parsed.modified_files == ("example.py",)
    assert parsed.added_lines == 1
    assert parsed.removed_lines == 1
