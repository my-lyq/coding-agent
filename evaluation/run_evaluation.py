"""Evaluate Base, prompted Agent, and LoRA-SFT Agent on isolated bug tasks."""
from __future__ import annotations
import argparse
import difflib
import gc
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from metrics import aggregate, markdown_report

AGENT_PROMPT = """You are a coding agent. Solve the task with tools. Emit exactly one next step:\nThought: <brief reasoning>\nAction: read_file({\"path\": \"...\"})\nor Action: write_file({\"path\": \"...\", \"content\": \"...\"})\nor Action: run_test({\"command\": \"python -m unittest discover -s . -p test_*.py\"})\nUse observations from the environment; never invent them. Make the smallest fix and run tests."""

class ModelRunner:
    def __init__(self, model: Any, tokenizer: Any, device: str) -> None:
        self.model, self.tokenizer, self.device = model, tokenizer, device
    @torch.inference_mode()
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> tuple[str, int, int]:
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        prompt_tokens = encoded["input_ids"].shape[-1]
        generated = output[0, prompt_tokens:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip(), prompt_tokens, generated.numel()

def load_runner(model_name: str, adapter: Path | None, local_only: bool) -> ModelRunner:
    tokenizer_source = str(adapter) if adapter else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_only)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, local_files_only=local_only)
    if adapter: model = PeftModel.from_pretrained(model, str(adapter), local_files_only=local_only)
    model.to(device).eval()
    return ModelRunner(model, tokenizer, device)

def run_tests(workspace: Path, command: list[str], timeout: int = 30) -> tuple[bool, str]:
    actual = [sys.executable, *command[1:]] if command and command[0] == "python" else command
    try:
        proc = subprocess.run(actual, cwd=workspace, text=True, capture_output=True, timeout=timeout, check=False)
        return proc.returncode == 0, (proc.stdout + proc.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"test execution failed: {exc}"

def snapshot(workspace: Path, editable: list[str]) -> dict[str, str]:
    return {path: (workspace / path).read_text(encoding="utf-8") for path in editable}

def make_patch(before: dict[str, str], after: dict[str, str]) -> str:
    parts = []
    for path in before:
        parts.extend(difflib.unified_diff(before[path].splitlines(True), after[path].splitlines(True), fromfile=f"a/{path}", tofile=f"b/{path}"))
    return "".join(parts)

def extract_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{": continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict): return value
        except json.JSONDecodeError: pass
    return None

def safe_path(workspace: Path, path: str) -> Path | None:
    target = (workspace / path).resolve()
    return target if target == workspace or workspace in target.parents else None

def baseline_run(runner: ModelRunner, task: dict[str, Any], workspace: Path, max_tokens: int) -> dict[str, Any]:
    files = snapshot(workspace, task["editable_files"])
    context = "\n\n".join(f"### {path}\n{content}" for path, content in files.items())
    prompt = f"""Fix the bug described below. Return only the complete corrected source file inside one Python code block. Do not return a diff or partial snippet.\n\nBug: {task['instruction']}\n\nRepository:\n{context}"""
    response, prompt_tokens, completion_tokens = runner.generate([{"role":"user","content":prompt}], max_tokens)
    value = extract_json(response)
    action_ok = False
    if value and value.get("path") in task["editable_files"] and isinstance(value.get("content"), str):
        (workspace / value["path"]).write_text(value["content"], encoding="utf-8"); action_ok = True
    else:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
        if len(task["editable_files"]) == 1 and blocks:
            (workspace / task["editable_files"][0]).write_text(blocks[-1], encoding="utf-8"); action_ok = True
    return {"steps":[{"response":response,"action_valid":action_ok}],"prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens}

def parse_action(text: str) -> tuple[str, dict[str, Any]] | None:
    match = re.search(r"Action:\s*(read_file|write_file|run_test)", text)
    if not match: return None
    tail = text[match.end():].lstrip()
    if tail.startswith("("): tail = tail[1:].lstrip()
    if tail.startswith(":"): tail = tail[1:].lstrip()
    input_match = re.search(r"Action Input:\s*", tail)
    if input_match: tail = tail[input_match.end():].lstrip()
    value = extract_json(tail)
    return (match.group(1), value) if value is not None else None

def agent_run(runner: ModelRunner, task: dict[str, Any], workspace: Path, max_steps: int, max_tokens: int) -> dict[str, Any]:
    base = f"{AGENT_PROMPT}\n\nTask:\n{task['instruction']}\n\nRepository files:\n" + "\n".join(task["editable_files"])
    history, logs = [], []
    prompt_total = completion_total = 0
    for _ in range(max_steps):
        prompt = base + ("\n\nCompleted steps:\n" + "\n\n".join(history) if history else "")
        response, pt, ct = runner.generate([{"role":"user","content":prompt}], max_tokens)
        prompt_total += pt; completion_total += ct
        parsed = parse_action(response)
        if not parsed:
            logs.append({"response":response,"action":"invalid","success":False,"observation":"invalid action format"})
            history.append(response + "\nObservation: invalid action format"); continue
        action, args = parsed
        success = False
        if action == "read_file":
            target = safe_path(workspace, str(args.get("path","")))
            if target and target.is_file(): observation = target.read_text(encoding="utf-8"); success = True
            else: observation = "read_file failed"
        elif action == "write_file":
            path, content = args.get("path"), args.get("content")
            if path in task["editable_files"] and isinstance(content, str) and content.strip() != "...":
                (workspace / path).write_text(content, encoding="utf-8"); observation = f"wrote {len(content)} characters to {path}"; success = True
            else: observation = "write_file refused: only editable source files are allowed"
        else:
            success, observation = run_tests(workspace, task["test_command"])
        logs.append({"response":response,"action":action,"arguments":args,"success":success,"observation":observation})
        history.append(response + f"\nObservation: {observation}")
        if action == "run_test" and success: break
    return {"steps":logs,"prompt_tokens":prompt_total,"completion_tokens":completion_total}

def evaluate_condition(name: str, runner: ModelRunner, tasks: list[dict[str, Any]], tasks_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    runs = []
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix=f"coding-agent-{task['id']}-") as tmp:
            workspace = Path(tmp)
            shutil.copytree(tasks_dir / task["id"], workspace, dirs_exist_ok=True)
            initial_ok, initial_output = run_tests(workspace, task["test_command"])
            if initial_ok: raise RuntimeError(f"task {task['id']} is invalid: tests pass before repair")
            before = snapshot(workspace, task["editable_files"])
            trace = baseline_run(runner, task, workspace, args.baseline_max_tokens) if name == "base" else agent_run(runner, task, workspace, args.max_steps, args.agent_max_tokens)
            after = snapshot(workspace, task["editable_files"])
            patch = make_patch(before, after)
            tests_passed, test_output = run_tests(workspace, task["test_command"])
            total = trace["prompt_tokens"] + trace["completion_tokens"]
            run = {"condition":name,"task_id":task["id"],"seen_in_sft":task["seen_in_sft"],"repair_success":bool(patch.strip()) and tests_passed,"tests_passed":tests_passed,"agent_steps":len(trace["steps"]),"prompt_tokens":trace["prompt_tokens"],"completion_tokens":trace["completion_tokens"],"total_tokens":total,"patch":patch,"final_test_output":test_output,"trace":trace["steps"]}
            runs.append(run)
            print(f"{name:10s} {task['id']:24s} repaired={run['repair_success']} steps={run['agent_steps']} tokens={total}")
    return runs

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=root/"outputs"/"qwen2.5-coder-agent-lora")
    parser.add_argument("--output", type=Path, default=root/"evaluation"/"result.json")
    parser.add_argument("--report", type=Path, default=root/"evaluation"/"REPORT.md")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--baseline-max-tokens", type=int, default=512)
    parser.add_argument("--agent-max-tokens", type=int, default=384)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); set_seed(args.seed)
    tasks_dir = root/"evaluation"/"tasks"
    tasks = json.loads((root/"evaluation"/"tasks.json").read_text(encoding="utf-8"))
    runs = []
    base = load_runner(args.model_name, None, args.local_files_only)
    runs += evaluate_condition("base", base, tasks, tasks_dir, args)
    runs += evaluate_condition("agent", base, tasks, tasks_dir, args)
    del base; gc.collect(); torch.cuda.empty_cache()
    sft = load_runner(args.model_name, args.adapter_path, args.local_files_only)
    runs += evaluate_condition("agent_sft", sft, tasks, tasks_dir, args)
    result = {"created_at":datetime.now(timezone.utc).isoformat(),"config":{"model_name":args.model_name,"adapter_path":str(args.adapter_path),"seed":args.seed,"max_steps":args.max_steps},"tasks":tasks,"summary":aggregate(runs),"runs":runs}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    args.report.write_text(markdown_report(result),encoding="utf-8")
    print(f"results: {args.output}\nreport: {args.report}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
