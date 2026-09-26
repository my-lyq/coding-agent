#!/usr/bin/env python3
"""Static next-action evaluation for Base or Structured SFT v2 policy."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # noqa: E402

from agent.tools import CodingTools  # noqa: E402
from agent.tools.schema import TOOL_NAMES, TOOL_SCHEMAS  # noqa: E402
from scripts.build_expert_trajectories import isolated_worktree  # noqa: E402
from scripts.load_swebench import write_json_atomic  # noqa: E402
from scripts.prepare_repo import TaskRef  # noqa: E402


DATA_ROOT = Path("/data_local/lyq/data_coding_agent")
DATASET_DIR = DATA_ROOT / "datasets" / "sft_v2"
DEFAULT_OUTPUT = DATA_ROOT / "logs" / "sft_v2_base_static_eval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--validation-file", type=Path, default=DATASET_DIR / "validation.jsonl")
    parser.add_argument("--dataset-manifest", type=Path, default=DATASET_DIR / "dataset_manifest.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-action-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def strict_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.I | re.S)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def arguments_valid(tool: Any, arguments: Any) -> bool:
    if tool not in TOOL_NAMES or not isinstance(arguments, dict):
        return False
    schema = next(item["parameters"] for item in TOOL_SCHEMAS if item["name"] == tool)
    if set(schema.get("required", [])) - arguments.keys():
        return False
    if schema.get("additionalProperties") is False and (
        arguments.keys() - schema.get("properties", {}).keys()
    ):
        return False
    for name, value in arguments.items():
        spec = schema["properties"][name]
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            return False
        if expected == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            return False
        if isinstance(value, int):
            if "minimum" in spec and value < spec["minimum"]:
                return False
            if "maximum" in spec and value > spec["maximum"]:
                return False
    return True


PHASE_TOOLS = {
    "LOCATE": {"list_files", "search_code"},
    "UNDERSTAND": {"read_file", "search_code"},
    "MODIFY": {"apply_patch"},
}


def execute_action(
    example: Mapping[str, Any],
    value: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    data_root: Path,
) -> tuple[bool, str]:
    instance_id = str(example["instance_id"])
    task = TaskRef(
        instance_id=instance_id,
        repo=str(example["repo"]),
        base_commit=str(example["base_commit"]),
        source_file="static_policy_eval",
    )
    with isolated_worktree(
        task,
        repos_dir=data_root / "swebench_dev" / "repos",
        workspaces_dir=data_root / "evaluation_workspaces" / "static_policy",
        refresh=False,
        timeout=1800,
        keep=False,
    ) as (workspace, _):
        tools = CodingTools(workspace, timeout=180, max_observation_chars=100_000)
        for previous in trajectory["steps"][: int(example["step_index"])]:
            if previous["tool"] != "apply_patch":
                continue
            replay = tools.apply_patch(str(previous["arguments"]["patch"]))
            if not replay.ok:
                return False, f"teacher-state replay failed: {replay.output}"
        tool = str(value["tool"])
        arguments = dict(value["arguments"])
        try:
            result = getattr(tools, tool)(**arguments)
        except (AttributeError, TypeError, ValueError) as exc:
            return False, f"execution failed: {type(exc).__name__}: {exc}"
        return result.ok, result.output


@torch.inference_mode()
def generate_action(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
    max_action_tokens: int,
) -> tuple[str, int, int, bool]:
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=max_action_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    generated = output[0, prompt_tokens:]
    generated_tokens = int(generated.numel())
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    hit_limit = generated_tokens >= max_action_tokens
    return response, prompt_tokens, generated_tokens, hit_limit


def aggregate(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(predictions)

    def total(field: str) -> int:
        return sum(bool(item.get(field)) for item in predictions)

    def rate(field: str) -> float:
        return total(field) / count if count else 0.0

    phase_accuracy: dict[str, dict[str, Any]] = {}
    for phase in ("LOCATE", "UNDERSTAND", "MODIFY"):
        items = [item for item in predictions if item["phase"] == phase]
        correct = sum(bool(item["teacher_tool_correct"]) for item in items)
        phase_accuracy[phase] = {
            "count": len(items),
            "correct": correct,
            "accuracy": correct / len(items) if items else 0.0,
        }
    attempts = [item for item in predictions if item.get("execution_attempted")]
    execution_successes = sum(bool(item.get("execution_success")) for item in attempts)
    return {
        "validation_count": count,
        "strict_json_only_count": total("strict_json_only"),
        "strict_json_only_rate": rate("strict_json_only"),
        "json_valid_count": total("json_valid"),
        "json_valid_rate": rate("json_valid"),
        "registered_tool_count": total("registered_tool"),
        "registered_tool_rate": rate("registered_tool"),
        "argument_schema_valid_count": total("argument_schema_valid"),
        "argument_schema_valid_rate": rate("argument_schema_valid"),
        "phase_compatible_count": total("phase_compatible"),
        "phase_compatible_rate": rate("phase_compatible"),
        "teacher_tool_correct_count": total("teacher_tool_correct"),
        "teacher_tool_accuracy": rate("teacher_tool_correct"),
        "phase_specific_tool_accuracy": phase_accuracy,
        "modify_apply_patch_selection_accuracy": phase_accuracy["MODIFY"]["accuracy"],
        "teacher_argument_exact_match_count": total("teacher_argument_exact_match"),
        "teacher_argument_exact_match_rate": rate("teacher_argument_exact_match"),
        "executable_action_count": len(attempts),
        "executable_action_rate": len(attempts) / count if count else 0.0,
        "tool_execution_success_count": execution_successes,
        "tool_execution_success_rate_among_attempts": (
            execution_successes / len(attempts) if attempts else 0.0
        ),
        "tool_execution_success_rate_all": execution_successes / count if count else 0.0,
        "generation_truncation_count": total("generation_truncated"),
        "generation_truncation_rate": rate("generation_truncated"),
        "generation_error_count": sum(bool(item.get("generation_error")) for item in predictions),
    }


def main() -> int:
    args = parse_args()
    if args.max_action_tokens != 1024:
        raise ValueError("static v2.1 evaluation requires max_action_tokens=1024")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite")
    set_seed(args.seed)
    data_root = DATA_ROOT
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if manifest["max_action_tokens"] != args.max_action_tokens:
        raise RuntimeError("dataset/evaluation action budgets differ")
    examples = load_jsonl(args.validation_file)
    trajectories = {
        item["instance_id"]: item
        for path in (data_root / "swebench_dev" / "expert_trajectories").glob("*.json")
        for item in [json.loads(path.read_text(encoding="utf-8"))]
    }
    cache_dir = data_root / "hf_cache" / "hub"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        local_files_only=args.local_files_only,
        dtype=dtype,
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter_path), local_files_only=True)
    model.to(args.device).eval()
    predictions: list[dict[str, Any]] = []
    started = time.monotonic()
    for position, example in enumerate(examples, 1):
        teacher = json.loads(example["target"])
        prediction: dict[str, Any] = {
            "instance_id": example["instance_id"],
            "step_index": example["step_index"],
            "phase": example["phase"],
            "teacher_tool": teacher["tool"],
        }
        try:
            response, prompt_tokens, generated_tokens, truncated = generate_action(
                model,
                tokenizer,
                example["prompt"],
                device=args.device,
                max_action_tokens=args.max_action_tokens,
            )
            prediction.update(
                {
                    "response": response,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "generation_truncated": truncated,
                    "generation_error": None,
                }
            )
            try:
                strict_json_only = isinstance(json.loads(response.strip()), dict)
            except json.JSONDecodeError:
                strict_json_only = False
            value = strict_json_object(response)
            json_valid = value is not None
            tool = value.get("tool") if value else None
            arguments = value.get("arguments") if value else None
            registered = tool in TOOL_NAMES
            schema_valid = arguments_valid(tool, arguments)
            phase_compatible = bool(
                registered and tool in PHASE_TOOLS.get(str(example["phase"]), set())
            )
            teacher_correct = tool == teacher["tool"]
            prediction.update(
                {
                    "parsed_action": value,
                    "strict_json_only": strict_json_only,
                    "json_valid": json_valid,
                    "registered_tool": registered,
                    "argument_schema_valid": schema_valid,
                    "phase_compatible": phase_compatible,
                    "teacher_tool_correct": teacher_correct,
                    "teacher_argument_exact_match": bool(
                        teacher_correct and arguments == teacher["arguments"]
                    ),
                    "execution_attempted": False,
                    "execution_success": False,
                    "execution_observation": None,
                }
            )
            if args.execute_actions and schema_valid and phase_compatible and value:
                prediction["execution_attempted"] = True
                success, observation = execute_action(
                    example,
                    value,
                    trajectories[str(example["instance_id"])],
                    data_root=data_root,
                )
                prediction["execution_success"] = success
                prediction["execution_observation"] = observation[:4000]
        except Exception as exc:
            prediction.update(
                {
                    "response": "",
                    "prompt_tokens": int(example["prompt_tokens"]),
                    "generated_tokens": 0,
                    "generation_truncated": False,
                    "generation_error": f"{type(exc).__name__}: {exc}",
                    "parsed_action": None,
                    "strict_json_only": False,
                    "json_valid": False,
                    "registered_tool": False,
                    "argument_schema_valid": False,
                    "phase_compatible": False,
                    "teacher_tool_correct": False,
                    "teacher_argument_exact_match": False,
                    "execution_attempted": False,
                    "execution_success": False,
                    "execution_observation": None,
                }
            )
        predictions.append(prediction)
        print(
            f"[{position:02d}/{len(examples)}] {example['instance_id']} "
            f"phase={example['phase']} json={prediction['json_valid']} "
            f"tool_ok={prediction['teacher_tool_correct']} tokens={prediction['generated_tokens']}",
            flush=True,
        )

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "condition": "base" if args.adapter_path is None else "sft_v2",
        "model": args.model_name,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "dataset_checksum": manifest["dataset_checksum"],
        "validation_checksum": manifest["validation_checksum"],
        "validation_count": len(examples),
        "decoding_config": {
            "do_sample": False,
            "max_action_tokens": args.max_action_tokens,
            "temperature": None,
            "seed": args.seed,
        },
        "execution_enabled": args.execute_actions,
        "metrics": aggregate(predictions),
        "elapsed_seconds": time.monotonic() - started,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
