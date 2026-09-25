"""LoRA SFT for Qwen2.5-Coder on coding-agent trajectories."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from dataset import CausalLMCollator, TrajectorySFTDataset

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=root / "data" / "train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "qwen2.5-coder-agent-lora")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    if not args.train_file.is_file():
        raise FileNotFoundError(f"training data not found: {args.train_file}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = TrajectorySFTDataset(args.train_file, tokenizer, args.max_length)
    print(f"trajectory policy examples: {len(train_dataset)}")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        weight_decay=0.01,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    summary = {"model": args.model_name, "examples": len(train_dataset), **result.metrics}
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"LoRA adapter saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
