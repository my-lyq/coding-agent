"""Trajectory-to-policy examples, tokenization, and causal-LM collation."""
from __future__ import annotations
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset

SYSTEM_PROMPT = """You are a coding agent. Solve the task with tools. Emit one next step as:\nThought: <brief reasoning>\nAction: <tool call>\nUse observations from the environment; never invent them. When done, emit Final Answer with a unified diff."""

def expand_trajectory(record: dict[str, Any]) -> list[dict[str, str]]:
    """Turn one trajectory into state-to-next-action policy examples."""
    instruction, repository = str(record.get("instruction", "")), str(record.get("input", ""))
    base = f"{SYSTEM_PROMPT}\n\nTask:\n{instruction}\n\nRepository context:\n{repository}"
    history, examples = [], []
    for step in record.get("trajectory", []):
        prompt = base + ("\n\nCompleted steps:\n" + "\n\n".join(history) if history else "")
        target = f"Thought: {step.get('thought', '')}\nAction: {step.get('action', '')}"
        examples.append({"prompt": prompt, "target": target})
        history.append(target + f"\nObservation: {step.get('observation', '')}")
    answer = str(record.get("answer", "")).strip()
    if answer:
        prompt = base + ("\n\nCompleted steps:\n" + "\n\n".join(history) if history else "")
        examples.append({"prompt": prompt, "target": f"Final Answer:\n{answer}"})
    return examples

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip(): continue
            try: record = json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            missing = {"instruction", "input", "trajectory", "answer"} - record.keys()
            if missing: raise ValueError(f"{path}:{line_number}: missing fields: {sorted(missing)}")
            records.append(record)
    if not records: raise ValueError(f"no records found in {path}")
    return records

def _token_ids(encoded: Any) -> list[int]:
    """Normalize Transformers 4.x list and 5.x BatchEncoding outputs."""
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)

class TrajectorySFTDataset(Dataset):
    """Loss is computed only on the next Agent action or final patch."""
    def __init__(self, path: str | Path, tokenizer: Any, max_length: int = 2048) -> None:
        self.tokenizer, self.max_length = tokenizer, max_length
        self.examples = [item for record in load_jsonl(path) for item in expand_trajectory(record)]
        if not self.examples: raise ValueError("records contain no trajectory steps or final answers")
    def __len__(self) -> int: return len(self.examples)
    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        user = {"role": "user", "content": example["prompt"]}
        prompt_ids = _token_ids(self.tokenizer.apply_chat_template([user], tokenize=True, add_generation_prompt=True))
        full_ids = _token_ids(self.tokenizer.apply_chat_template([user, {"role": "assistant", "content": example["target"]}], tokenize=True, add_generation_prompt=False))
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("chat template does not preserve the prompt prefix")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if len(full_ids) > self.max_length:
            overflow = len(full_ids) - self.max_length
            full_ids, labels = full_ids[overflow:], labels[overflow:]
        if not any(label != -100 for label in labels):
            raise ValueError("max_length removed the complete supervised target")
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

class CausalLMCollator:
    def __init__(self, pad_token_id: int) -> None: self.pad_token_id = pad_token_id
    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            n = width - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * n)
            batch["attention_mask"].append(item["attention_mask"] + [0] * n)
            batch["labels"].append(item["labels"] + [-100] * n)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
