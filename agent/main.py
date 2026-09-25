"""Minimal ReAct Agent plus a repository-task backend for SWE-bench."""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from executor import AgentExecutor, Step
from task import DEFAULT_TRAJECTORY_DIR, Task, validate_trajectory_dir

SYSTEM_PROMPT = """You are a coding agent. Solve the task by repeatedly emitting exactly:
Thought: <brief reasoning>
Action: <read_file|write_file|run_test>
Action Input: <valid JSON object>
Read relevant code, make the smallest justified edit, run tests, and use failures to iterate.
The only permitted test command is: python -m unittest discover -s examples -p test_*.py
"""

class Policy(Protocol):
    def next_action(self, problem: str, steps: list[Step]) -> str | None: ...

class DemoPolicy:
    """Deterministic local policy so the calculator demo needs no model."""
    target = "examples/calculator.py"
    def next_action(self, problem: str, steps: list[Step]) -> str | None:
        if not steps:
            return self._action(
                "I should inspect the file named in the bug report before editing it.",
                "read_file", {"path": self.target}
            )
        if len(steps) == 1:
            fixed = steps[-1].observation.replace(
                "return a + b  # BUG: should subtract", "return a - b"
            )
            return self._action(
                "subtract incorrectly adds its operands; I will make the minimal operator fix.",
                "write_file", {"path": self.target, "content": fixed}
            )
        if steps[-1].action == "write_file":
            return self._action(
                "The edit succeeded; tests must validate behavior and catch regressions.",
                "run_test",
                {"command": "python -m unittest discover -s examples -p test_*.py"},
            )
        if steps[-1].action == "run_test" and steps[-1].success:
            return None
        return self._action(
            "Tests failed, so I should reread the current code before revising the patch.",
            "read_file", {"path": self.target}
        )

    @staticmethod
    def _action(thought: str, action: str, arguments: dict[str, str]) -> str:
        return (
            f"Thought: {thought}\nAction: {action}\n"
            f"Action Input: {json.dumps(arguments)}"
        )

class HFPolicy:
    """Shared deterministic Hugging Face causal-LM inference backend."""
    def __init__(self, model_name: str, adapter_path: Path | None = None) -> None:
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
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        self.prompt_tokens += int(inputs.input_ids.shape[1])
        self.completion_tokens += int(generated.numel())
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def token_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

class QwenPolicy(HFPolicy):
    """Original optional model backend used by the calculator Agent."""
    def next_action(self, problem: str, steps: list[Step]) -> str | None:
        history = []
        for step in steps:
            history.append(
                f"Thought: {step.thought}\nAction: {step.action}\n"
                f"Action Input: {json.dumps(step.action_input)}\n"
                f"Observation: {step.observation}"
            )
        prompt = SYSTEM_PROMPT + f"\nTask: {problem}\n" + "\n".join(history) + "\n"
        answer = self.generate(prompt, 256)
        if re.search(r"Action:\s*finish", answer, re.I):
            return None
        return answer

def build_repository_index(
    repo_path: Path, problem_statement: str, max_files: int = 400
) -> str:
    """Rank tracked files by lexical overlap with the issue and return a bounded index."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "ls-files"],
        text=True, capture_output=True, check=False, timeout=30
    )
    if result.returncode == 0:
        paths = [line for line in result.stdout.splitlines() if line]
    else:
        paths = [
            str(path.relative_to(repo_path))
            for path in repo_path.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]
    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", problem_statement)
    }
    def score(path: str) -> tuple[int, str]:
        lowered = path.lower()
        overlap = sum(term in lowered for term in terms)
        source_bonus = 2 if Path(path).suffix in {".py", ".js", ".ts", ".java", ".cpp", ".rs"} else 0
        test_bonus = 1 if "test" in lowered else 0
        return overlap * 10 + source_bonus + test_bonus, path
    ranked = sorted(paths, key=lambda path: (-score(path)[0], score(path)[1]))
    chosen = ranked[:max_files]
    suffix = f"\n... {len(paths)-len(chosen)} more tracked files omitted" if len(paths) > len(chosen) else ""
    return "\n".join(chosen) + suffix

class SWEbenchPolicy(HFPolicy):
    """Repository-aware policy for the task-file backend."""
    def __init__(
        self, task: Task, model_name: str, adapter_path: Path | None = None
    ) -> None:
        super().__init__(model_name, adapter_path)
        self.task = task
        self.repo_index = build_repository_index(
            task.repo_path, task.problem_statement
        )


    @classmethod
    def from_shared_backend(cls, task: Task, backend: HFPolicy) -> "SWEbenchPolicy":
        policy = cls.__new__(cls)
        policy.torch = backend.torch
        policy.tokenizer = backend.tokenizer
        policy.model = backend.model
        policy.prompt_tokens = 0
        policy.completion_tokens = 0
        policy.task = task
        policy.repo_index = build_repository_index(
            task.repo_path, task.problem_statement
        )
        return policy

    def next_action(self, problem: str, steps: list[Step]) -> str | None:
        if steps and steps[-1].action == "run_test" and steps[-1].success:
            return None
        system = f"""You are a repository-level coding agent.
Solve the issue with only these tools and emit exactly one action per turn:
Thought: <brief repository-grounded reasoning>
Action: read_file
Action Input: {{"path": "<relative path>"}}
or:
Thought: <brief reasoning>
Action: write_file
Action Input: {{"path": "<existing relative path>", "content": "<complete file>"}}
or:
Thought: <brief reasoning>
Action: run_test
Action Input: {{"command": {json.dumps(self.task.test_command)}}}

Rules:
- Paths are relative to the repository root.
- Read a file before changing it.
- Make the smallest relevant source change.
- Never modify tests.
- The only allowed test command is exactly: {self.task.test_command}
- Use test failures as observations and iterate.
"""
        history = []
        for step in steps:
            observation = step.observation
            if len(observation) > 12_000:
                observation = observation[:12_000] + "\n...[observation truncated in prompt]"
            history.append(
                f"Thought: {step.thought}\nAction: {step.action}\n"
                f"Action Input: {json.dumps(step.action_input, ensure_ascii=False)}\n"
                f"Observation: {observation}"
            )
        prompt = (
            system
            + f"\nProblem statement:\n{self.task.problem_statement}\n"
            + f"\nRanked repository file index:\n{self.repo_index}\n"
            + ("\nCompleted steps:\n" + "\n\n".join(history) if history else "")
            + "\n\nProduce the next action only."
        )
        answer = self.generate(prompt, 768)
        if re.search(r"Action:\s*finish", answer, re.I):
            return None
        return answer

def run(problem: str, backend: str, max_steps: int = 8) -> int:
    """Preserved Day 1 calculator entry point."""
    root = Path(__file__).resolve().parents[1]
    executor = AgentExecutor(root)
    policy: Policy = DemoPolicy() if backend == "demo" else QwenPolicy(backend)
    print(f"Task: {problem}\nBackend: {backend}\n")
    solved = False
    for number in range(1, max_steps + 1):
        response = policy.next_action(problem, executor.steps)
        if response is None:
            solved = bool(
                executor.steps
                and executor.steps[-1].action == "run_test"
                and executor.steps[-1].success
            )
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
    out.write_text(
        json.dumps(executor.trajectory(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Trajectory saved to {out.name}")
    print("Status:", "SOLVED" if solved else "NOT SOLVED")
    return 0 if solved else 1

def _save_task_trajectory(
    task: Task,
    executor: AgentExecutor,
    trajectory_dir: Path,
) -> Path:
    test_step = next(
        (step for step in reversed(executor.steps) if step.action == "run_test"),
        None,
    )
    record: dict[str, Any] = {
        "task_id": task.task_id,
        "problem": task.problem_statement,
        "steps": [
            {
                "thought": step.thought,
                "action": (
                    f"{step.action}("
                    f"{json.dumps(step.action_input, ensure_ascii=False, sort_keys=True)})"
                ),
                "observation": step.observation,
            }
            for step in executor.steps
        ],
        "patch": executor.final_patch(),
        "test_result": {
            "command": task.test_command,
            "passed": bool(test_step and test_step.success),
            "output": test_step.observation if test_step else "test was not executed",
        },
    }
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = trajectory_dir / f"{task.task_id}__{timestamp}__{uuid.uuid4().hex[:8]}.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=trajectory_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target

def run_repository_task(
    task: Task,
    policy: Policy,
    *,
    max_steps: int,
    trajectory_dir: Path,
    test_timeout: int,
) -> int:
    executor = AgentExecutor(
        task.repo_path,
        allowed_test_commands={task.test_command},
        test_timeout=test_timeout,
    )
    print(f"Task ID: {task.task_id}\nRepository: {task.repo_path}")
    print(f"Problem:\n{task.problem_statement}\n")
    for number in range(1, max_steps + 1):
        response = policy.next_action(task.problem_statement, executor.steps)
        if response is None:
            break
        try:
            step = executor.execute(response)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            step = Step(
                thought=response[:2000],
                action="invalid",
                action_input={},
                observation=f"invalid model action: {exc}",
                success=False,
            )
            executor.steps.append(step)
        print(
            f"Step {number}: action={step.action} success={step.success}\n"
            f"Observation: {step.observation[:2000]}\n"
        )
        if step.action == "run_test" and step.success:
            break

    if not executor.steps or executor.steps[-1].action != "run_test":
        forced = (
            "Thought: I must execute the task's allowed test command before finishing.\n"
            "Action: run_test\n"
            f"Action Input: {json.dumps({'command': task.test_command})}"
        )
        step = executor.execute(forced)
        print(f"Final validation: success={step.success}\n{step.observation[:2000]}")

    output = _save_task_trajectory(task, executor, trajectory_dir)
    test_step = next(
        (step for step in reversed(executor.steps) if step.action == "run_test"),
        None,
    )
    passed = bool(test_step and test_step.success)
    print(f"Patch:\n{executor.final_patch()}\nTrajectory: {output}")
    print("Status:", "SOLVED" if passed else "NOT SOLVED")
    return 0 if passed else 1

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="Fix bug in examples/calculator.py")
    parser.add_argument(
        "--backend",
        default=os.getenv("CODING_AGENT_MODEL", "demo"),
        help="demo, swebench, or a Hugging Face model id",
    )
    parser.add_argument("--task-file", type=Path)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model used by the swebench backend.",
    )
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--test-timeout", type=int, default=600)
    args = parser.parse_args()

    if args.backend == "swebench":
        if not args.task_file:
            parser.error("--task-file is required for --backend swebench")
        task = Task.from_json(args.task_file)
        trajectory_dir = validate_trajectory_dir(args.trajectory_dir)
        policy = SWEbenchPolicy(task, args.model_name, args.adapter_path)
        return run_repository_task(
            task,
            policy,
            max_steps=args.max_steps,
            trajectory_dir=trajectory_dir,
            test_timeout=args.test_timeout,
        )
    if args.task_file:
        parser.error("--task-file is only supported by --backend swebench")
    return run(args.task, args.backend, args.max_steps)

if __name__ == "__main__":
    raise SystemExit(main())
