"""Structured Tool SFT v2 examples, context budgeting, and label masking.

Legacy ``training/dataset.py`` remains untouched for v1 reproducibility.  This
module supervises only the next canonical JSON tool call; observations are
prompt context and never labels.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from agent.planner import ALLOWED_TOOLS, Phase
from agent.prompting import (
    build_agent_messages,
    canonical_tool_target,
    render_chat_prompt,
)
from agent.tools.schema import TOOL_NAMES, TOOL_SCHEMAS


SUPPORTED_PHASES = ("LOCATE", "UNDERSTAND", "MODIFY")
UNSUPPORTED_TRAINING_PHASES = ("VERIFY",)
DEFAULT_MAX_LENGTH = 8192


class StructuredDatasetError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def tokenize_supervised(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    target: str,
) -> dict[str, list[int]]:
    """Apply one chat template and mask every token before the assistant target."""
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True
        )
    )
    full_messages = [*list(messages), {"role": "assistant", "content": target}]
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise StructuredDatasetError(
            "chat_template_prefix_mismatch",
            "assistant conversation does not preserve the generation prompt prefix",
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise StructuredDatasetError("empty_target", "assistant target has no tokens")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "prompt_ids": prompt_ids,
        "target_ids": full_ids[len(prompt_ids) :],
    }


def validate_structured_target(target: str, phase: str) -> dict[str, Any]:
    try:
        value = json.loads(target)
    except json.JSONDecodeError as exc:
        raise StructuredDatasetError("target_invalid_json", str(exc)) from exc
    if not isinstance(value, dict) or set(value) != {"tool", "arguments"}:
        raise StructuredDatasetError(
            "target_invalid_schema", "target must contain only tool and arguments"
        )
    tool, arguments = value["tool"], value["arguments"]
    if tool not in TOOL_NAMES or not isinstance(arguments, dict):
        raise StructuredDatasetError(
            "target_invalid_schema", "target tool must be registered and arguments an object"
        )
    try:
        allowed = ALLOWED_TOOLS[Phase(phase)]
    except (KeyError, ValueError) as exc:
        raise StructuredDatasetError("unsupported_phase", f"unsupported phase: {phase}") from exc
    if phase not in SUPPORTED_PHASES or tool not in allowed:
        raise StructuredDatasetError(
            "phase_tool_mismatch", f"{tool} is not allowed for SFT phase {phase}"
        )
    schema = next(schema for schema in TOOL_SCHEMAS if schema["name"] == tool)["parameters"]
    missing = set(schema.get("required", [])) - arguments.keys()
    unexpected = arguments.keys() - schema.get("properties", {}).keys()
    if missing or (schema.get("additionalProperties") is False and unexpected):
        raise StructuredDatasetError(
            "target_invalid_schema",
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}",
        )
    for name, argument in arguments.items():
        expected = schema["properties"][name].get("type")
        if expected == "string" and not isinstance(argument, str):
            raise StructuredDatasetError("target_invalid_schema", f"{name} must be string")
        if expected == "integer" and (
            not isinstance(argument, int) or isinstance(argument, bool)
        ):
            raise StructuredDatasetError("target_invalid_schema", f"{name} must be integer")
    return value


@dataclass(frozen=True)
class BudgetResult:
    messages: list[dict[str, str]]
    history: list[dict[str, Any]]
    truncated: bool
    original_tokens: int
    final_tokens: int
    prompt_tokens: int
    target_tokens: int
    removed_history_steps: int
    truncated_observations: list[dict[str, Any]]


class ContextBudgetManager:
    """Fit history into a fixed chat budget without truncating protected fields."""

    def __init__(self, tokenizer: Any, max_length: int = DEFAULT_MAX_LENGTH) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _messages(
        self,
        problem_statement: str,
        phase: str,
        history: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        return build_agent_messages(
            problem_statement,
            phase,
            sorted(ALLOWED_TOOLS[Phase(phase)]),
            TOOL_SCHEMAS,
            history,
        )

    def _measure(
        self,
        problem_statement: str,
        phase: str,
        history: Sequence[Mapping[str, Any]],
        target: str,
    ) -> tuple[list[dict[str, str]], dict[str, list[int]]]:
        messages = self._messages(problem_statement, phase, history)
        return messages, tokenize_supervised(self.tokenizer, messages, target)

    @staticmethod
    def _truncate_observation(text: str, max_chars: int) -> str:
        marker = "\n[TRUNCATED]"
        if len(text) <= max_chars:
            return text
        limit = max(0, max_chars - len(marker))
        prefix = text[:limit]
        if "\n" in prefix:
            prefix = prefix.rsplit("\n", 1)[0]
        return prefix + marker

    def fit(
        self,
        problem_statement: str,
        phase: str,
        history: Sequence[Mapping[str, Any]],
        target: str,
    ) -> BudgetResult:
        if phase not in SUPPORTED_PHASES:
            raise StructuredDatasetError("unsupported_phase", phase)
        target_value = validate_structured_target(target, phase)
        if canonical_tool_target(target_value["tool"], target_value["arguments"]) != target:
            raise StructuredDatasetError("target_not_canonical", "target JSON is not canonical")

        working = [deepcopy(dict(item)) for item in history]
        original_messages, original = self._measure(
            problem_statement, phase, working, target
        )
        original_tokens = len(original["input_ids"])
        protected_messages, protected = self._measure(
            problem_statement, phase, [], target
        )
        if len(protected["input_ids"]) > self.max_length:
            raise StructuredDatasetError(
                "target_too_long",
                "protected system/schema/problem/phase plus target exceeds max_length",
            )
        if original_tokens <= self.max_length:
            return BudgetResult(
                original_messages,
                working,
                False,
                original_tokens,
                original_tokens,
                len(original["prompt_ids"]),
                len(original["target_ids"]),
                0,
                [],
            )

        removed = 0
        truncated_observations: list[dict[str, Any]] = []
        # Preserve several most-recent calls, while dropping genuinely old state
        # first. This also removes early, huge list_files outputs in most traces.
        while len(working) > 4:
            working.pop(0)
            removed += 1
            _, measured = self._measure(problem_statement, phase, working, target)
            if len(measured["input_ids"]) <= self.max_length:
                break

        messages, measured = self._measure(problem_statement, phase, working, target)
        if len(measured["input_ids"]) > self.max_length:
            priorities = {"list_files": 0, "search_code": 1, "read_file": 2}
            candidates = sorted(
                range(len(working)),
                key=lambda index: (
                    priorities.get(str(working[index].get("tool")), 3),
                    index,
                ),
            )
            for index in candidates:
                observation = str(working[index].get("observation", ""))
                if len(observation) <= 256:
                    continue
                original_chars = len(observation)
                limit = max(256, original_chars // 2)
                while limit >= 256:
                    working[index]["observation"] = self._truncate_observation(
                        observation, limit
                    )
                    messages, measured = self._measure(
                        problem_statement, phase, working, target
                    )
                    if len(measured["input_ids"]) <= self.max_length or limit == 256:
                        break
                    limit = max(256, limit // 2)
                final_chars = len(str(working[index]["observation"]))
                truncated_observations.append(
                    {
                        "history_index": index + removed,
                        "tool": str(working[index].get("tool", "")),
                        "original_chars": original_chars,
                        "final_chars": final_chars,
                    }
                )
                if len(measured["input_ids"]) <= self.max_length:
                    break

        while len(measured["input_ids"]) > self.max_length and working:
            working.pop(0)
            removed += 1
            messages, measured = self._measure(
                problem_statement, phase, working, target
            )
        if len(measured["input_ids"]) > self.max_length:
            # The no-history check above normally catches this; retain a distinct
            # defensive error for custom tokenizers with unstable templates.
            raise StructuredDatasetError(
                "context_budget_failed", "unable to fit protected prompt and target"
            )
        return BudgetResult(
            messages,
            working,
            True,
            original_tokens,
            len(measured["input_ids"]),
            len(measured["prompt_ids"]),
            len(measured["target_ids"]),
            removed,
            truncated_observations,
        )


def build_step_example(
    trajectory: Mapping[str, Any],
    step_index: int,
    tokenizer: Any,
    budget_manager: ContextBudgetManager,
) -> dict[str, Any]:
    steps = trajectory.get("steps", [])
    if not 0 <= step_index < len(steps):
        raise IndexError(step_index)
    step = steps[step_index]
    phase = str(step["phase"])
    target = canonical_tool_target(str(step["tool"]), dict(step["arguments"]))
    validate_structured_target(target, phase)
    history = [
        {
            "phase": previous["phase"],
            "tool": previous["tool"],
            "arguments": previous["arguments"],
            "observation": previous["observation"],
            "tool_success": previous["tool_success"],
        }
        for previous in steps[:step_index]
    ]
    budget = budget_manager.fit(
        str(trajectory["problem_statement"]), phase, history, target
    )
    prompt = render_chat_prompt(tokenizer, budget.messages)
    if target in prompt:
        raise StructuredDatasetError(
            "future_action_leakage", "supervised target appears in input prompt"
        )
    return {
        "instance_id": str(trajectory["instance_id"]),
        "repo": str(trajectory["repo"]),
        "base_commit": str(trajectory["base_commit"]),
        "trajectory_id": str(trajectory["instance_id"]),
        "step_index": step_index,
        "phase": phase,
        "prompt": prompt,
        "messages": budget.messages,
        "target": target,
        "teacher_source": "swebench_gold_oracle",
        "quality_tier": "B",
        "controller_intervened": False,
        "invalid_action": False,
        "repair_applied": False,
        "prompt_tokens": budget.prompt_tokens,
        "target_tokens": budget.target_tokens,
        "total_tokens": budget.final_tokens,
        "truncated": budget.truncated,
        "original_tokens": budget.original_tokens,
        "final_tokens": budget.final_tokens,
        "removed_history_steps": budget.removed_history_steps,
        "truncated_observations": budget.truncated_observations,
    }


def load_structured_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            validate_structured_target(str(record.get("target", "")), str(record.get("phase", "")))
            records.append(record)
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


class StructuredSFTDataset(Dataset):
    """Tokenized next-tool dataset with optional train-only phase balancing."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_length: int = DEFAULT_MAX_LENGTH,
        sampling_strategy: str = "phase-balanced",
        split: str | None = None,
    ) -> None:
        if sampling_strategy not in {"natural", "phase-balanced"}:
            raise ValueError("sampling_strategy must be natural or phase-balanced")
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = load_structured_jsonl(self.path)
        self.split = split or ("validation" if "validation" in self.path.stem else "train")
        self.sampling_strategy = sampling_strategy
        self.effective_sampling_strategy = (
            "natural" if self.split == "validation" else sampling_strategy
        )
        counts = Counter(str(item["phase"]) for item in self.examples)
        if self.effective_sampling_strategy == "phase-balanced":
            self.sample_weights = [1.0 / counts[str(item["phase"])] for item in self.examples]
        else:
            self.sample_weights = [1.0 for _ in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        encoded = tokenize_supervised(
            self.tokenizer, example["messages"], str(example["target"])
        )
        if len(encoded["input_ids"]) > self.max_length:
            raise StructuredDatasetError(
                "example_exceeds_budget",
                f"{example['instance_id']} step {example['step_index']} exceeds {self.max_length}",
            )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["labels"],
        }

    def make_sampler(self, seed: int = 42) -> WeightedRandomSampler | None:
        if self.effective_sampling_strategy == "natural":
            return None
        generator = torch.Generator()
        generator.manual_seed(seed)
        return WeightedRandomSampler(
            self.sample_weights,
            num_samples=len(self.examples),
            replacement=True,
            generator=generator,
        )

    def sampled_phase_distribution(self, draws: int = 10_000, seed: int = 42) -> dict[str, float]:
        """Deterministic audit helper; does not mutate or duplicate examples."""
        if draws <= 0:
            raise ValueError("draws must be positive")
        rng = random.Random(seed)
        indices = rng.choices(
            range(len(self.examples)), weights=self.sample_weights, k=draws
        )
        counts = Counter(str(self.examples[index]["phase"]) for index in indices)
        return {phase: counts[phase] / draws for phase in sorted(counts)}


class StructuredCausalLMCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = width - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {
            name: torch.tensor(values, dtype=torch.long)
            for name, values in batch.items()
        }
