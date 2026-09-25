"""Single-task structured rollout state machine with planning control."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent.executor import AgentExecutor, ActionRequest, Step
from agent.planner import PlanningController
from agent.task import Task
from agent.tools.schema import guess_original_action
from .recorder import TrajectoryRecorder

class Policy(Protocol):
    def next_action(self, steps: list[Step]) -> str | None: ...

@dataclass
class RolloutResult:
    output_path: Path
    success: bool
    num_steps: int
    modified_files: list[str]
    patch: str

class AgentRolloutRunner:
    def __init__(
        self,
        task: Task,
        policy: Policy,
        recorder: TrajectoryRecorder,
        *,
        max_steps: int = 8,
        test_timeout: int = 900,
    ) -> None:
        self.task = task
        self.policy = policy
        self.recorder = recorder
        self.max_steps = max_steps
        self.executor = AgentExecutor(
            task.repo_path,
            allowed_test_commands={task.test_command},
            test_timeout=test_timeout,
        )
        self.planner = PlanningController()
        self.policy_error: str | None = None

    def _blocked_step(
        self, request: ActionRequest, observation: str, reason: str
    ) -> Step:
        phase = self.planner.phase.value
        step = Step(
            thought=request.thought,
            action=request.action,
            action_input=request.arguments,
            observation=observation,
            success=False,
            original_action=request.original_action,
            repaired_action=request.repaired_action,
            invalid_action=request.invalid_action,
            repair_applied=request.repair_applied,
            phase=phase,
            previous_phase=phase,
            model_proposed_tool=request.original_action,
            executed_tool=None,
            controller_intervened=True,
            intervention_reason=reason,
            tool_success=False,
        )
        self.executor.steps.append(step)
        return step

    @staticmethod
    def _test_reached_process(step: Step) -> bool:
        if step.executed_tool != "run_test":
            return False
        lowered = step.observation.lower()
        return not (
            lowered.startswith("planning controller blocked")
            or "run_test failed: invalid arguments:" in lowered
            or "run_test refused" in lowered
        )

    def run(self, overwrite: bool = False) -> RolloutResult:
        for _ in range(self.max_steps):
            try:
                response = self.policy.next_action(self.executor.steps)
            except Exception as exc:
                self.policy_error = f"{type(exc).__name__}: {exc}"
                phase = self.planner.phase.value
                self.executor.steps.append(
                    Step(
                        thought="The model backend failed during rollout.",
                        action="error",
                        action_input={},
                        observation=self.policy_error,
                        success=False,
                        phase=phase,
                        previous_phase=phase,
                        model_proposed_tool=None,
                        executed_tool=None,
                        controller_intervened=False,
                        intervention_reason=None,
                        tool_success=False,
                    )
                )
                break
            if response is None:
                break
            try:
                request = self.executor.parse(response)
                decision = self.planner.validate(request.action)
                if decision.allowed:
                    step = self.executor.execute(response)
                    previous, current = self.planner.observe(step)
                    step.previous_phase = previous.value
                    step.phase = current.value
                else:
                    step = self._blocked_step(
                        request, decision.observation,
                        decision.reason or "illegal_phase_action",
                    )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                phase = self.planner.phase.value
                step = Step(
                    thought=response[:2000],
                    action="invalid",
                    action_input={},
                    observation=f"invalid model action: {exc}",
                    success=False,
                    original_action=guess_original_action(response),
                    repaired_action=None,
                    invalid_action=True,
                    repair_applied=False,
                    phase=phase,
                    previous_phase=phase,
                    model_proposed_tool=guess_original_action(response),
                    executed_tool=None,
                    controller_intervened=False,
                    intervention_reason=None,
                    tool_success=False,
                )
                self.executor.steps.append(step)
            if step.executed_tool == "run_test" and step.success:
                break

        executed_tests = [
            step for step in self.executor.steps if self._test_reached_process(step)
        ]
        if not executed_tests:
            self.planner.begin_final_verification()
            forced = json.dumps(
                {
                    "tool": "run_test",
                    "arguments": {"command": self.task.test_command},
                }
            )
            step = self.executor.execute(forced)
            step.model_proposed_tool = None
            step.executed_tool = "run_test"
            step.controller_intervened = True
            step.intervention_reason = "final_verification"
            step.tool_success = step.success
            previous, current = self.planner.observe(step)
            step.previous_phase = previous.value
            step.phase = current.value
            executed_tests.append(step)

        test_step = executed_tests[-1] if executed_tests else None
        modified_files = []
        for path, before in self.executor.original_files.items():
            after = self.executor.tools.snapshot_file(path)
            if after.ok and after.output != before:
                modified_files.append(path)
        patch = self.executor.final_patch()
        passed = bool(test_step and test_step.success)
        success = passed and self.policy_error is None
        test_result = {
            "command": self.task.test_command,
            "passed": passed,
            "output": test_step.observation if test_step else "test not executed",
        }
        if self.policy_error:
            test_result["policy_error"] = self.policy_error
        record = self.recorder.build_record(
            instance_id=self.task.task_id,
            problem_statement=self.task.problem_statement,
            steps=self.executor.steps,
            modified_files=modified_files,
            patch=patch,
            test_result=test_result,
            success=success,
        )
        output_path = self.recorder.save(record, overwrite=overwrite)
        return RolloutResult(
            output_path=output_path,
            success=success,
            num_steps=len(self.executor.steps),
            modified_files=sorted(modified_files),
            patch=patch,
        )
