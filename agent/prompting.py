"""Shared structured-agent prompt protocol for inference and SFT v2.

This module is intentionally tokenizer-independent.  It produces canonical
chat messages; both rollout inference and dataset construction must render
those messages with the same tokenizer ``apply_chat_template`` method.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


SYSTEM_INSTRUCTION = """You are a repository-level coding agent.
Return exactly one JSON object and no markdown, prose, Thought, Final Answer, or code fence:
{"tool":"<one registered tool name>","arguments":{}}

The JSON object is the next tool call. Use real observations from completed
calls and never invent repository contents or tool results."""


def _canonical_json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def normalise_history_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Project rollout/oracle history onto the shared prompt contract."""
    tool = item.get("tool") or item.get("executed_tool") or item.get("action")
    arguments = item.get("arguments", item.get("action_input", {}))
    projected: dict[str, Any] = {
        "phase": str(item.get("phase") or "UNKNOWN"),
        "tool_call": {
            "tool": str(tool or "unknown"),
            "arguments": deepcopy(arguments) if isinstance(arguments, Mapping) else {},
        },
        "observation": str(item.get("observation", "")),
        "tool_success": bool(item.get("tool_success", item.get("success", False))),
    }
    previous = item.get("previous_phase")
    if previous is not None:
        projected["previous_phase"] = str(previous)
    proposed = item.get("model_proposed_tool")
    executed = item.get("executed_tool")
    if proposed is not None and proposed != tool:
        projected["model_proposed_tool"] = str(proposed)
    if executed is None and "executed_tool" in item:
        projected["executed_tool"] = None
    reason = item.get("intervention_reason")
    if reason:
        projected["intervention_reason"] = str(reason)
    return projected


def build_agent_messages(
    problem_statement: str,
    phase: str,
    allowed_tools: Sequence[str],
    tool_schemas: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    *,
    repository_index: str = "",
    test_command: str = "",
) -> list[dict[str, str]]:
    """Build the one canonical structured-tool chat state.

    Gold patches, test patches, benchmark test lists, and future actions are not
    accepted as parameters, which keeps teacher targets outside the prompt API.
    """
    allowed = tuple(sorted(dict.fromkeys(str(name) for name in allowed_tools)))
    schemas = [deepcopy(dict(schema)) for schema in tool_schemas]
    completed = [normalise_history_item(item) for item in history]

    sections = [
        "Problem statement:\n" + problem_statement.strip(),
        "Current planning phase:\n" + str(phase),
        "Allowed tools in this phase:\n" + ", ".join(allowed),
        "Available tools (JSON Schema):\n" + _canonical_json(schemas),
    ]
    if repository_index.strip():
        sections.insert(1, "Ranked repository file index:\n" + repository_index.strip())
    rules = [
        "Call only a registered tool allowed in the current planning phase.",
        "Use only paths relative to the repository root.",
        "Read relevant source before editing it.",
        "apply_patch requires a unified diff with --- a/path and +++ b/path headers.",
        "Make the smallest relevant source-code change and never modify tests.",
        "After three consecutive search_code calls, read a concrete candidate.",
        "Use observations from completed calls; never invent tool results.",
    ]
    if test_command:
        rules.append(f"run_test must use exactly: {test_command}")
    sections.append("Rules:\n- " + "\n- ".join(rules))
    if completed:
        sections.append("Completed calls:\n" + _canonical_json(completed))
    sections.append("Return the next JSON tool call only.")
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def render_chat_prompt(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    """Render the exact Qwen inference prefix used before generation."""
    return str(
        tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
    )


def canonical_tool_target(tool: str, arguments: Mapping[str, Any]) -> str:
    """Canonical compact JSON supervised by Structured SFT v2."""
    return _canonical_json(
        {"tool": str(tool), "arguments": deepcopy(dict(arguments))}, indent=None
    )
