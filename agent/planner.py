"""Finite-state planning controller for repository-level coding rollouts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .executor import Step

class Phase(str, Enum):
    LOCATE = "LOCATE"
    UNDERSTAND = "UNDERSTAND"
    MODIFY = "MODIFY"
    VERIFY = "VERIFY"

ALLOWED_TOOLS: dict[Phase, frozenset[str]] = {
    Phase.LOCATE: frozenset({"list_files", "search_code"}),
    Phase.UNDERSTAND: frozenset({"read_file", "search_code"}),
    Phase.MODIFY: frozenset({"apply_patch"}),
    Phase.VERIFY: frozenset({"run_test"}),
}

@dataclass(frozen=True)
class PlannerDecision:
    allowed: bool
    phase: Phase
    allowed_tools: frozenset[str]
    observation: str
    reason: str | None

class PlanningController:
    """Guard tool calls and advance a generic locate-understand-modify-verify FSM."""

    def __init__(self) -> None:
        self.phase = Phase.LOCATE
        self.has_read_file = False
        self.search_code_streak = 0
        self.list_files_streak = 0

    @property
    def allowed_tools(self) -> frozenset[str]:
        return ALLOWED_TOOLS[self.phase]

    def validate(self, tool: str) -> PlannerDecision:
        allowed = self.allowed_tools
        if self.phase is Phase.LOCATE and self.list_files_streak >= 2 and tool != "search_code":
            return self._blocked(
                "repeated_list",
                "list_files was called repeatedly; use search_code to narrow the candidate files",
            )
        if self.search_code_streak >= 3 and tool != "read_file":
            return self._blocked(
                "repeated_search",
                "search_code reached the limit of three consecutive calls; read a concrete candidate file",
            )
        if tool == "apply_patch" and not self.has_read_file:
            return self._blocked(
                "modify_without_read",
                "apply_patch is forbidden until at least one read_file succeeds",
            )
        if tool not in allowed:
            return self._blocked(
                "illegal_phase_action",
                f"tool {tool!r} is not allowed in phase {self.phase.value}",
            )
        return PlannerDecision(
            allowed=True,
            phase=self.phase,
            allowed_tools=allowed,
            observation="",
            reason=None,
        )

    def _blocked(self, reason: str, message: str) -> PlannerDecision:
        names = ", ".join(sorted(self.allowed_tools))
        return PlannerDecision(
            allowed=False,
            phase=self.phase,
            allowed_tools=self.allowed_tools,
            observation=(
                f"Planning Controller blocked this action: {message}. "
                f"Current phase={self.phase.value}; allowed tools={names}."
            ),
            reason=reason,
        )

    def observe(self, step: Step) -> tuple[Phase, Phase]:
        """Update counters and state after an executed tool call."""
        previous = self.phase
        if step.action == "search_code":
            self.search_code_streak += 1
        else:
            self.search_code_streak = 0
        if step.action == "list_files":
            self.list_files_streak += 1
        else:
            self.list_files_streak = 0

        if self.phase is Phase.LOCATE and step.action == "search_code" and step.success:
            self.phase = Phase.UNDERSTAND
        elif self.phase is Phase.UNDERSTAND and step.action == "read_file" and step.success:
            self.has_read_file = True
            self.phase = Phase.MODIFY
        elif self.phase is Phase.MODIFY and step.action == "apply_patch" and step.success:
            self.phase = Phase.VERIFY
        elif self.phase is Phase.VERIFY and step.action == "run_test" and not step.success:
            self.phase = Phase.UNDERSTAND
        return previous, self.phase

    def begin_final_verification(self) -> Phase:
        """Enter VERIFY for the runner-owned final validation at budget exhaustion."""
        previous = self.phase
        self.phase = Phase.VERIFY
        self.search_code_streak = 0
        self.list_files_streak = 0
        return previous
