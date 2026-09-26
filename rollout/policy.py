"""Model policy that maps repository state and observations to structured tools."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.executor import Step
from agent.planner import ALLOWED_TOOLS, Phase
from agent.prompting import build_agent_messages
from agent.task import Task
from agent.tools.schema import TOOL_SCHEMAS

DEFAULT_HF_CACHE = Path("/data_local/lyq/data_coding_agent/hf_cache/hub")


class HFBackend:
    """One reusable Hugging Face model backend for one or many rollouts."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        adapter_path: Path | None = None,
        cache_dir: Path = DEFAULT_HF_CACHE,
        local_files_only: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        tokenizer_source = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
            trust_remote_code=False,
            torch_dtype="auto",
            device_map="auto",
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
                local_files_only=local_files_only,
            )
        self.model.eval()

    def generate(
        self,
        messages: str | Sequence[Mapping[str, str]],
        max_new_tokens: int = 1024,
    ) -> tuple[str, int, int]:
        """Render the shared chat state and generate one structured tool call."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        text = self.tokenizer.apply_chat_template(
            list(messages),
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
        max_action_tokens: int = 1024,
    ) -> None:
        if max_action_tokens <= 0:
            raise ValueError("max_action_tokens must be positive")
        self.task = task
        self.max_action_tokens = max_action_tokens
        self.backend = backend or HFBackend(model_name, adapter_path)
        self.repository_index = build_repository_index(
            task.repo_path, task.problem_statement, max_index_files
        )
        self.last_generation: dict[str, int | bool] | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def next_action(self, steps: list[Step]) -> str | None:
        if steps and steps[-1].action == "run_test" and steps[-1].success:
            return None
        return self._generate(self.build_messages(steps))

    def build_messages(self, steps: list[Step]) -> list[dict[str, str]]:
        """Expose the exact shared inference state used by SFT v2."""
        current_phase = (
            steps[-1].phase if steps and steps[-1].phase else Phase.LOCATE.value
        )
        try:
            phase = Phase(current_phase)
        except ValueError:
            phase = Phase.LOCATE
            current_phase = phase.value
        history: list[dict[str, Any]] = []
        for step in steps:
            history.append(
                {
                    "phase": step.phase,
                    "previous_phase": step.previous_phase,
                    "tool": step.executed_tool or step.action,
                    "arguments": step.action_input,
                    "observation": step.observation,
                    "tool_success": step.tool_success,
                    "model_proposed_tool": step.model_proposed_tool,
                    "executed_tool": step.executed_tool,
                    "intervention_reason": step.intervention_reason,
                }
            )
        return build_agent_messages(
            self.task.problem_statement,
            current_phase,
            sorted(ALLOWED_TOOLS[phase]),
            TOOL_SCHEMAS,
            history,
            repository_index=self.repository_index,
            test_command=self.task.test_command,
        )

    def _generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        answer, prompt_tokens, completion_tokens = self.backend.generate(messages, max_new_tokens=self.max_action_tokens)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.last_generation = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": completion_tokens,
            "generation_truncated": completion_tokens >= self.max_action_tokens,
        }
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
