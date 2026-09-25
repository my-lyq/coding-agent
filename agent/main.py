"""Day 1: a minimal coding agent with a ReAct tool loop.

Run from the repository root with: python agent/main.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Protocol

from executor import AgentExecutor, Step


SYSTEM_PROMPT = """You are a coding agent. Solve the task by repeatedly emitting exactly:
Thought: <brief reasoning>
Action: <read_file|write_file|run_test>
Action Input: <valid JSON object>
Read relevant code, make the smallest justified edit, run tests, and use failures to iterate.
The only permitted test command is: python -m unittest discover -s examples -p test_*.py
"""


class Policy(Protocol):
    def next_action(self, task: str, steps: list[Step]) -> str | None: ...


class DemoPolicy:
    """Deterministic local policy so the repository demos without a GPU or download."""

    target = "examples/calculator.py"

    def next_action(self, task: str, steps: list[Step]) -> str | None:
        if not steps:
            return self._action(
                "I should inspect the file named in the bug report before editing it.",
                "read_file",
                {"path": self.target},
            )
        if len(steps) == 1:
            source = steps[-1].observation
            fixed = source.replace("return a + b  # BUG: should subtract", "return a - b")
            return self._action(
                "subtract incorrectly adds its operands; I will make the minimal operator fix.",
                "write_file",
                {"path": self.target, "content": fixed},
            )
        if steps[-1].action == "write_file":
            return self._action(
                "The edit succeeded; tests must validate behavior and catch regressions.",
                "run_test",
                {"command": "python -m unittest discover -s examples -p test_*.py"},
            )
        if steps[-1].action == "run_test" and steps[-1].success:
            return None
        # A real model would diagnose the failure and propose another edit. The demo
        # rereads the target, leaving the same feedback loop visible and bounded.
        return self._action(
            "Tests failed, so I should reread the current code before revising the patch.",
            "read_file",
            {"path": self.target},
        )

    @staticmethod
    def _action(thought: str, action: str, arguments: dict[str, str]) -> str:
        return f"Thought: {thought}\nAction: {action}\nAction Input: {json.dumps(arguments)}"


class QwenPolicy:
    """Optional Hugging Face backend for Qwen2.5-Coder."""

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    def next_action(self, task: str, steps: list[Step]) -> str | None:
        history = []
        for step in steps:
            history.append(
                f"Thought: {step.thought}\nAction: {step.action}\n"
                f"Action Input: {json.dumps(step.action_input)}\nObservation: {step.observation}"
            )
        prompt = SYSTEM_PROMPT + f"\nTask: {task}\n" + "\n".join(history) + "\n"
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = output[0][inputs.input_ids.shape[1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        if re.search(r"Action:\s*finish", answer, re.I):
            return None
        return answer


def run(task: str, backend: str, max_steps: int = 8) -> int:
    root = Path(__file__).resolve().parents[1]
    executor = AgentExecutor(root)
    policy: Policy = DemoPolicy() if backend == "demo" else QwenPolicy(backend)

    print(f"Task: {task}\nBackend: {backend}\n")
    solved = False
    for number in range(1, max_steps + 1):
        response = policy.next_action(task, executor.steps)
        if response is None:
            solved = bool(executor.steps and executor.steps[-1].action == "run_test" and executor.steps[-1].success)
            break
        try:
            step = executor.execute(response)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"Step {number}\n{response}\nObservation: invalid model action: {exc}\n")
            continue
        print(f"Step {number}\nThought: {step.thought}\nAction: {step.action}")
        print(f"Observation: {step.observation}\n")

    print("Final patch:\n" + executor.final_patch())
    out = root / "trajectory.json"
    out.write_text(json.dumps(executor.trajectory(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Trajectory saved to {out.name}")
    print("Status:", "SOLVED" if solved else "NOT SOLVED")
    return 0 if solved else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="Fix bug in examples/calculator.py")
    parser.add_argument(
        "--backend",
        default=os.getenv("CODING_AGENT_MODEL", "demo"),
        help="demo or a Hugging Face model id, e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    return run(args.task, args.backend, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())

