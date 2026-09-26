from __future__ import annotations

import difflib
import subprocess
from pathlib import Path

import pytest

from oracle.action_splitter import GoldPatchActionSplitter


class ApproxTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [0] * max(1, (len(text) + 3) // 4)


def _git(path: Path, *arguments: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def _init_repo(path: Path) -> tuple[str, str, dict[str, str]]:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    base = {
        "a.py": "".join(f"value_{index} = {index}\n" for index in range(120)),
        "b.py": "start = 1\n" + "middle = 2\n" * 20 + "end = 3\n",
    }
    final = {
        "a.py": "".join(f"value_{index} = {index + 10}\n" for index in range(120)),
        "b.py": "start = 10\n" + "middle = 2\n" * 20 + "end = 30\n",
    }
    for name, content in base.items():
        (path / name).write_text(content, encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "base")
    commit = _git(path, "rev-parse", "HEAD").strip()
    patch = "".join(
        f"diff --git a/{name} b/{name}\n" + "".join(
            difflib.unified_diff(
                base[name].splitlines(keepends=True),
                final[name].splitlines(keepends=True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
        for name in base
    )
    return commit, patch, final


@pytest.fixture(scope="module")
def split_result(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("action-split")
    commit, patch, final = _init_repo(workspace)
    splitter = GoldPatchActionSplitter(
        ApproxTokenizer(), max_action_tokens=180, timeout=30
    )
    result = splitter.split_and_apply(workspace, patch, resolved_commit=commit)
    return workspace, commit, patch, final, splitter, result


def test_action_target_budget(split_result) -> None:
    _, _, _, _, splitter, result = split_result
    assert len(result.actions) > 2
    assert max(action.target_tokens for action in result.actions) <= 180


def test_split_patch_apply_sequence(tmp_path: Path, split_result) -> None:
    _, _, _, _, _, result = split_result
    commit, _, final = _init_repo(tmp_path)
    assert _git(tmp_path, "rev-parse", "HEAD").strip() == commit
    for action in result.actions:
        _git(tmp_path, "apply", "--check", "-", input_text=action.patch)
        _git(tmp_path, "apply", "-", input_text=action.patch)
    for name, expected in final.items():
        assert (tmp_path / name).read_text(encoding="utf-8") == expected


def test_reconstructed_gold_diff(split_result) -> None:
    workspace, _, _, final, _, result = split_result
    assert result.final_diff == result.reference_diff
    assert result.modified_files == ("a.py", "b.py")
    for name, expected in final.items():
        assert (workspace / name).read_text(encoding="utf-8") == expected
