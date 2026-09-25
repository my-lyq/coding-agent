"""Model policy that maps repository state and observations to structured tools."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from agent.executor import Step
from agent.task import Task
from agent.tools.schema import render_tool_schemas

class HFBackend:
    """One reusable Hugging Face model backend for one or many rollouts."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        adapter_path: Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        tokenizer_source = str(adapter_path) if adapter_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 768) -> tuple[str, int, int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0][inputs.input_ids.shape[1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer, int(inputs.input_ids.shape[1]), int(generated.numel())

class ReActPolicy:
    """Qwen-compatible repository policy with a strict JSON tool protocol."""

    def __init__(
        self,
        task: Task,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        adapter_path: Path | None = None,
        max_index_files: int = 400,
        backend: HFBackend | None = None,
    ) -> None:
        self.task = task
        self.backend = backend or HFBackend(model_name, adapter_path)
        self.repository_index = build_repository_index(
            task.repo_path, task.problem_statement, max_index_files
        )
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def next_action(self, steps: list[Step]) -> str | None:
        if steps and steps[-1].action == "run_test" and steps[-1].success:
            return None
        current_phase = steps[-1].phase if steps and steps[-1].phase else "LOCATE"
        phase_tools = {
            "LOCATE": ["list_files", "search_code"],
            "UNDERSTAND": ["read_file", "search_code"],
            "MODIFY": ["apply_patch"],
            "VERIFY": ["run_test"],
        }
        allowed_tools = phase_tools.get(current_phase, phase_tools["LOCATE"])
        history = []
        for step in steps:
            observation = step.observation
            if len(observation) > 12_000:
                observation = observation[:12_000] + "\n...[truncated in prompt]"
            metadata = ""
            if step.repair_applied:
                metadata = (
                    f"\nRepair: {step.original_action} -> {step.repaired_action}"
                )
            transition = (
                f"Phase: {step.previous_phase or 'UNKNOWN'} -> "
                f"{step.phase or 'UNKNOWN'}\n"
            )
            proposal = step.model_proposed_tool or step.original_action or step.action
            execution = step.executed_tool or "(not executed)"
            history.append(
                transition
                + f"Model proposed: {proposal}\n"
                + f"Executed tool: {execution}\n"
                + "Arguments: "
                + json.dumps(step.action_input, ensure_ascii=False)
                + metadata
                + f"\nObservation: {observation}"
            )
        prompt = f"""You are a repository-level coding agent.

Problem statement:
{self.task.problem_statement}

Ranked repository file index:
{self.repository_index}

Current planning phase: {current_phase}
Allowed tools in this phase: {", ".join(allowed_tools)}

Available tools (JSON Schema):
{render_tool_schemas()}

Return exactly one JSON object and no markdown, prose, Thought, or code fence:
{{"tool":"<one registered tool name>","arguments":{{}}}}

Rules:
- Call only a tool allowed in the current planning phase.
- Use only list_files, search_code, read_file, apply_patch, or run_test.
- After three consecutive search_code calls, call read_file on a concrete candidate.
- A successful apply_patch automatically advances to VERIFY.
- If run_test fails, inspect the failure in UNDERSTAND before modifying again.
- Never invent aliases such as edit_file, update_file, modify_file, or write_file.
- Paths are relative to the repository root.
- Read relevant source before editing it.
- apply_patch requires a unified diff with --- a/path and +++ b/path headers.
- Make the smallest relevant source-code change and never modify tests.
- run_test must use exactly: {self.task.test_command}
- Use observations from completed calls; never invent tool results.
"""
        if history:
            prompt += "\nCompleted calls:\n" + "\n\n".join(history)
        prompt += "\n\nReturn the next JSON tool call only."
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        answer, prompt_tokens, completion_tokens = self.backend.generate(prompt)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        return answer

    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

def build_repository_index(
    repo_path: Path, problem_statement: str, max_files: int = 400
) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "ls-files"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {result.stderr.strip()}")
    paths = [line for line in result.stdout.splitlines() if line]
    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", problem_statement)
    }
    source_suffixes = {".py", ".js", ".ts", ".java", ".cpp", ".cc", ".rs", ".go"}

    def rank(path: str) -> tuple[int, str]:
        lowered = path.lower()
        overlap = sum(term in lowered for term in terms)
        source = 2 if Path(path).suffix in source_suffixes else 0
        test = 1 if "test" in lowered else 0
        return -(overlap * 10 + source + test), path

    selected = sorted(paths, key=rank)[:max_files]
    omitted = len(paths) - len(selected)
    suffix = f"\n... {omitted} more tracked files omitted" if omitted else ""
    return "\n".join(selected) + suffix
