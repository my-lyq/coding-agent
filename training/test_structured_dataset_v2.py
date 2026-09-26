from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pytest

from agent.planner import ALLOWED_TOOLS, Phase
from agent.prompting import build_agent_messages, canonical_tool_target, render_chat_prompt
from agent.task import Task
from agent.tools.schema import TOOL_SCHEMAS
from oracle.expert_builder import detect_prompt_leakage
from rollout.policy import ReActPolicy
from scripts.build_structured_sft_v2 import choose_repository_split, overlap_audit
from training.structured_dataset import (
    ContextBudgetManager,
    StructuredDatasetError,
    StructuredSFTDataset,
    build_step_example,
    tokenize_supervised,
    validate_structured_target,
)


class TinyChatTokenizer:
    pad_token_id = 0

    @staticmethod
    def _render(messages, add_generation_prompt):
        text = "".join(
            f"<{message['role']}>\n{message['content']}\n" for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>\n"
        else:
            text += "<eos>"
        return text

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = self._render(messages, add_generation_prompt)
        return [ord(character) + 1 for character in text] if tokenize else text


@pytest.fixture
def tokenizer():
    return TinyChatTokenizer()


def _trajectory() -> dict:
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
    return {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "problem_statement": "Change old to new in x.py.",
        "steps": [
            {
                "phase": "LOCATE",
                "tool": "search_code",
                "arguments": {"query": "old", "path": "x.py"},
                "observation": "x.py:1:old",
                "tool_success": True,
            },
            {
                "phase": "UNDERSTAND",
                "tool": "read_file",
                "arguments": {"path": "x.py", "start_line": 1, "end_line": 1},
                "observation": "old\n",
                "tool_success": True,
            },
            {
                "phase": "MODIFY",
                "tool": "apply_patch",
                "arguments": {"patch": patch},
                "observation": "applied patch to x.py",
                "tool_success": True,
            },
        ],
    }


def test_repo_and_instance_level_split_leakage() -> None:
    counts = {"repo/a": 17, "repo/b": 15, "repo/c": 9, "repo/d": 6, "repo/e": 4, "repo/f": 4}
    train, validation = choose_repository_split(counts, seed=42)
    assert train.isdisjoint(validation)
    assert sum(counts[repo] for repo in train) >= 40
    assert sum(counts[repo] for repo in validation) >= 8

    records = [
        {"instance_id": "same", "repo": "repo/a", "base_commit": "a", "split": "train"},
        {"instance_id": "same", "repo": "repo/b", "base_commit": "b", "split": "validation"},
    ]
    assert overlap_audit(records)["instance_id_overlap"] == ["same"]


def test_structured_json_target_and_tool_schema_validation() -> None:
    target = canonical_tool_target("read_file", {"path": "x.py"})
    assert validate_structured_target(target, "UNDERSTAND")["tool"] == "read_file"
    with pytest.raises(StructuredDatasetError, match="arguments"):
        validate_structured_target('{"tool":"read_file","arguments":[]}', "UNDERSTAND")
    with pytest.raises(StructuredDatasetError, match="unexpected"):
        validate_structured_target(
            '{"tool":"read_file","arguments":{"path":"x.py","secret":1}}',
            "UNDERSTAND",
        )


def test_phase_tool_compatibility() -> None:
    with pytest.raises(StructuredDatasetError) as error:
        validate_structured_target(
            canonical_tool_target("apply_patch", {"patch": "diff"}), "LOCATE"
        )
    assert error.value.reason == "phase_tool_mismatch"


def test_prompt_inference_consistency(tmp_path: Path, tokenizer) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "x.py").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "x.py"], check=True)
    task = Task("owner__repo-1", tmp_path, "Change old to new.", "python -m pytest")

    class UnusedBackend:
        def generate(self, messages):  # pragma: no cover
            raise AssertionError("generation must not run")

    policy = ReActPolicy(task, backend=UnusedBackend())
    inference_messages = policy.build_messages([])
    training_messages = build_agent_messages(
        task.problem_statement,
        "LOCATE",
        sorted(ALLOWED_TOOLS[Phase.LOCATE]),
        TOOL_SCHEMAS,
        [],
        repository_index=policy.repository_index,
        test_command=task.test_command,
    )
    assert training_messages == inference_messages
    assert render_chat_prompt(tokenizer, training_messages) == render_chat_prompt(
        tokenizer, inference_messages
    )


def test_target_and_future_action_not_in_prompt(tokenizer) -> None:
    trajectory = _trajectory()
    manager = ContextBudgetManager(tokenizer, max_length=20_000)
    for index in range(len(trajectory["steps"])):
        example = build_step_example(trajectory, index, tokenizer, manager)
        assert example["target"] not in example["prompt"]
        for future in trajectory["steps"][index:]:
            future_target = canonical_tool_target(future["tool"], future["arguments"])
            assert future_target not in example["prompt"]


def test_gold_and_testpatch_not_in_prompt() -> None:
    gold = "diff --git a/x.py b/x.py\n+gold secret\n"
    test_patch = "diff --git a/test_x.py b/test_x.py\n+test secret\n"
    assert detect_prompt_leakage(gold, gold_patch=gold, test_patch=test_patch) == ["gold_patch"]
    assert detect_prompt_leakage(test_patch, gold_patch=gold, test_patch=test_patch) == ["test_patch"]
    assert not detect_prompt_leakage("ordinary issue", gold_patch=gold, test_patch=test_patch)


def test_context_truncation_preserves_target(tokenizer) -> None:
    manager = ContextBudgetManager(tokenizer, max_length=8_000)
    target = canonical_tool_target("read_file", {"path": "important.py"})
    history = [
        {
            "phase": "LOCATE",
            "tool": "list_files",
            "arguments": {"path": "."},
            "observation": "\n".join(f"file_{index}.py" for index in range(3000)),
            "tool_success": True,
        }
    ]
    result = manager.fit("Inspect important.py.", "UNDERSTAND", history, target)
    assert result.truncated
    assert result.final_tokens <= 8_000
    assert result.truncated_observations or result.removed_history_steps
    encoded = tokenize_supervised(tokenizer, result.messages, target)
    target_text = "".join(chr(value - 1) for value in encoded["target_ids"])
    assert target in target_text


def _write_examples(path: Path, tokenizer, count: int = 25) -> None:
    manager = ContextBudgetManager(tokenizer, max_length=20_000)
    examples = []
    trajectory = _trajectory()
    for index in range(count):
        example = build_step_example(
            trajectory, index % len(trajectory["steps"]), tokenizer, manager
        )
        example["instance_id"] = f"instance-{index}"
        examples.append(example)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in examples),
        encoding="utf-8",
    )


def test_label_masking_on_at_least_20_random_examples(tmp_path: Path, tokenizer) -> None:
    path = tmp_path / "train.jsonl"
    _write_examples(path, tokenizer, 25)
    dataset = StructuredSFTDataset(path, tokenizer, max_length=20_000)
    for index in random.Random(42).sample(range(len(dataset)), 20):
        encoded = dataset[index]
        example = dataset.examples[index]
        prompt_tokens = int(example["prompt_tokens"])
        assert all(label == -100 for label in encoded["labels"][:prompt_tokens])
        assert all(label != -100 for label in encoded["labels"][prompt_tokens:])


def test_padding_labels_and_validation_natural_distribution(tmp_path: Path, tokenizer) -> None:
    path = tmp_path / "validation.jsonl"
    _write_examples(path, tokenizer, 25)
    dataset = StructuredSFTDataset(
        path,
        tokenizer,
        max_length=20_000,
        sampling_strategy="phase-balanced",
        split="validation",
    )
    assert dataset.effective_sampling_strategy == "natural"
    assert dataset.make_sampler() is None
    assert set(dataset.sample_weights) == {1.0}
