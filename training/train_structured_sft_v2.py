#!/usr/bin/env python3
"""Non-quantized LoRA training for frozen Structured Tool SFT v2.1 data."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from torch.utils.data import Dataset, WeightedRandomSampler  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from scripts.load_swebench import write_json_atomic  # noqa: E402
from training.structured_dataset import (  # noqa: E402
    StructuredCausalLMCollator,
    StructuredSFTDataset,
)


DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DATASET_DIR = DATA_ROOT / "datasets" / "sft_v2"
CHECKPOINT_ROOT = DATA_ROOT / "checkpoints"
LOGS_ROOT = DATA_ROOT / "logs"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--train-file", type=Path, default=DATASET_DIR / "train.jsonl")
    parser.add_argument("--validation-file", type=Path, default=DATASET_DIR / "validation.jsonl")
    parser.add_argument("--dataset-manifest", type=Path, default=DATASET_DIR / "dataset_manifest.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-action-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-examples", type=int, default=12)
    parser.add_argument("--smoke-max-steps", type=int, default=2)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


class PhaseTrackingDataset(Dataset):
    def __init__(
        self,
        base: StructuredSFTDataset,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.base = base
        self.indices = list(indices) if indices is not None else list(range(len(base)))
        self.examples = [base.examples[index] for index in self.indices]
        self.sample_weights = [base.sample_weights[index] for index in self.indices]
        self.phase_accesses: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        base_index = self.indices[index]
        self.phase_accesses[str(self.base.examples[base_index]["phase"])] += 1
        return self.base[base_index]


class PhaseBalancedTrainer(Trainer):
    def __init__(self, *args: Any, sampler_seed: int = 42, **kwargs: Any) -> None:
        self.sampler_seed = sampler_seed
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset: Dataset | None = None):
        dataset = train_dataset or self.train_dataset
        if not isinstance(dataset, PhaseTrackingDataset):
            return super()._get_train_sampler(train_dataset)
        generator = torch.Generator()
        generator.manual_seed(self.sampler_seed)
        return WeightedRandomSampler(
            dataset.sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )


class PhaseCountCallback(TrainerCallback):
    def __init__(self, dataset: PhaseTrackingDataset) -> None:
        self.dataset = dataset
        self.history: list[dict[str, Any]] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        self.history.append(
            {
                "epoch": float(state.epoch or 0.0),
                "global_step": int(state.global_step),
                "phase_counts": dict(sorted(self.dataset.phase_accesses.items())),
            }
        )
        self.dataset.phase_accesses.clear()
        return control


def select_smoke_indices(dataset: StructuredSFTDataset, count: int, seed: int) -> list[int]:
    if not 8 <= count <= 16:
        raise ValueError("smoke examples must be between 8 and 16")
    count = min(count, len(dataset))
    longest = sorted(
        range(len(dataset)),
        key=lambda index: int(dataset.examples[index]["total_tokens"]),
        reverse=True,
    )[:4]
    selected = list(longest)
    rng = __import__("random").Random(seed)
    for phase in ("LOCATE", "UNDERSTAND", "MODIFY"):
        candidates = [
            index
            for index, item in enumerate(dataset.examples)
            if item["phase"] == phase and index not in selected
        ]
        rng.shuffle(candidates)
        selected.extend(candidates[:2])
    remaining = [index for index in range(len(dataset)) if index not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "name": None, "vram_bytes": 0, "bf16_supported": False}
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": properties.name,
        "vram_bytes": int(properties.total_memory),
        "vram_gib": properties.total_memory / 1024**3,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def finite_training_logs(log_history: Sequence[dict[str, Any]]) -> tuple[bool, bool]:
    losses = [float(item["loss"]) for item in log_history if "loss" in item]
    gradients = [float(item["grad_norm"]) for item in log_history if "grad_norm" in item]
    return bool(losses) and all(math.isfinite(value) for value in losses), bool(gradients) and all(
        math.isfinite(value) for value in gradients
    )


def main() -> int:
    args = parse_args()
    if args.max_length != 8192 or args.max_action_tokens != 1024:
        raise ValueError("SFT v2.1 requires max_length=8192 and max_action_tokens=1024")
    if args.batch_size != 1 or args.eval_batch_size != 1:
        raise ValueError("first experiment uses per-device train/eval batch size 1")
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if manifest["max_length"] != 8192 or manifest["max_action_tokens"] != 1024:
        raise RuntimeError("frozen dataset manifest does not match training budgets")
    output_dir = args.output_dir or (
        CHECKPOINT_ROOT / ("sft_v2_smoke" if args.mode == "smoke" else "sft_v2")
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output:
        raise FileExistsError(f"non-empty output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_path = LOGS_ROOT / (
        "sft_v2_smoke.json" if args.mode == "smoke" else "sft_v2_training_summary.json"
    )
    set_seed(args.seed)
    hardware = gpu_info()
    if not hardware["available"]:
        raise RuntimeError("CUDA GPU is required; refusing CPU training fallback")
    precision = "bf16" if hardware["bf16_supported"] else "fp16"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    cache_dir = DATA_ROOT / "hf_cache" / "hub"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_base = StructuredSFTDataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        sampling_strategy="phase-balanced",
        split="train",
    )
    validation_dataset = StructuredSFTDataset(
        args.validation_file,
        tokenizer,
        max_length=args.max_length,
        sampling_strategy="natural",
        split="validation",
    )
    indices = (
        select_smoke_indices(train_base, args.smoke_examples, args.seed)
        if args.mode == "smoke"
        else None
    )
    train_dataset = PhaseTrackingDataset(train_base, indices)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        local_files_only=True,
        dtype=dtype,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    base_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    )
    if base_trainable:
        raise RuntimeError(f"base model is not frozen: {base_trainable} trainable parameters")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    is_smoke = args.mode == "smoke"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.smoke_max_steps if is_smoke else -1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=1.0,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="no" if is_smoke else "epoch",
        save_strategy="steps" if is_smoke else "epoch",
        save_steps=1 if is_smoke else 500,
        save_total_limit=2,
        load_best_model_at_end=not is_smoke,
        metric_for_best_model="eval_loss" if not is_smoke else None,
        greater_is_better=False if not is_smoke else None,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        use_cache=False,
    )
    phase_callback = PhaseCountCallback(train_dataset)
    trainer = PhaseBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None if is_smoke else validation_dataset,
        data_collator=StructuredCausalLMCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[phase_callback],
        sampler_seed=args.seed,
    )
    started = time.monotonic()
    initial_eval_loss = None
    status = "running"
    oom_stage = None
    try:
        if not is_smoke:
            initial_eval_loss = float(trainer.evaluate()["eval_loss"])
        train_output = trainer.train()
        status = "trained"
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        oom_stage = "forward_or_backward"
        result = {
            "status": status,
            "error": str(exc),
            "oom_stage": oom_stage,
            "gpu": hardware,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
        }
        write_json_atomic(logs_path, result)
        print(json.dumps(result, indent=2))
        return 2

    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)
    loss_finite, gradient_finite = finite_training_logs(trainer.state.log_history)
    final_eval_loss = None
    if not is_smoke:
        final_eval_loss = float(trainer.evaluate()["eval_loss"])
    peak_memory = int(torch.cuda.max_memory_allocated())
    training_time = time.monotonic() - started
    best_checkpoint = trainer.state.best_model_checkpoint
    last_checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    last_checkpoint = str(last_checkpoints[-1]) if last_checkpoints else str(final_adapter)
    adapter_reload_success = False
    reload_error = None
    if is_smoke:
        try:
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
            reload_base = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                cache_dir=str(cache_dir),
                local_files_only=True,
                dtype=dtype,
            )
            reloaded = PeftModel.from_pretrained(reload_base, str(final_adapter), local_files_only=True)
            adapter_reload_success = any("lora_" in name for name, _ in reloaded.named_parameters())
            del reloaded, reload_base
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            reload_error = f"{type(exc).__name__}: {exc}"

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "status": status,
        "base_model": args.model_name,
        "dataset_checksum": manifest["dataset_checksum"],
        "train_trajectories": manifest["train_trajectory_count"],
        "validation_trajectories": manifest["validation_trajectory_count"],
        "train_examples": len(train_dataset),
        "full_train_examples": len(train_base),
        "validation_examples": len(validation_dataset),
        "smoke_indices": indices,
        "phase_distribution_per_epoch": phase_callback.history,
        "max_length": args.max_length,
        "max_action_tokens": args.max_action_tokens,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percentage": 100 * trainable / total,
        "base_trainable_params": base_trainable,
        "precision": precision,
        "gpu": hardware,
        "peak_allocated_vram_bytes": peak_memory,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "epochs": args.epochs,
        "seed": args.seed,
        "training_steps": int(train_output.global_step),
        "initial_eval_loss": initial_eval_loss,
        "best_eval_loss": trainer.state.best_metric if not is_smoke else None,
        "final_eval_loss": final_eval_loss,
        "final_train_loss": float(train_output.training_loss),
        "loss_finite": loss_finite,
        "gradient_finite": gradient_finite,
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
        "final_adapter": str(final_adapter),
        "adapter_reload_success": adapter_reload_success if is_smoke else None,
        "adapter_reload_error": reload_error,
        "no_oom": True,
        "training_time_seconds": training_time,
        "log_history": trainer.state.log_history if not is_smoke else [],
    }
    # In smoke mode trainer was deleted before this line; best/eval fields above
    # are not accessed after deletion because their conditional branches skip.
    smoke_success = (
        status == "trained"
        and loss_finite
        and gradient_finite
        and adapter_reload_success
        and int(train_output.global_step) <= 10
    )
    if is_smoke:
        summary["smoke_success"] = smoke_success
    write_json_atomic(logs_path, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "log_history"}, ensure_ascii=False, indent=2))
    print(f"summary={logs_path}")
    return 0 if (smoke_success if is_smoke else status == "trained") else 2


if __name__ == "__main__":
    raise SystemExit(main())
