"""Canonical tool schemas and structured-call repair utilities."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_files",
        "description": "List repository files below a relative directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory; defaults to ."},
                "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "search_code",
        "description": "Search source text and return matching files and lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Literal text to search for."},
                "path": {"type": "string", "description": "Relative file or directory; defaults to ."},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read an existing UTF-8 repository file, optionally by line range.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path of an existing file."},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_patch",
        "description": "Apply a unified diff to existing repository files.",
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Unified diff with a/ and b/ headers."},
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_test",
        "description": "Run the exact allowlisted test command for this task.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Exact allowlisted command."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
)

TOOL_NAMES = frozenset(schema["name"] for schema in TOOL_SCHEMAS)
ACTION_ALIASES = {
    "edit_file": "apply_patch",
    "update_file": "apply_patch",
    "modify_file": "apply_patch",
    "update_code": "apply_patch",
    "modify": "apply_patch",
    "write_file": "apply_patch",
    "ls": "list_files",
    "list_directory": "list_files",
    "find_files": "list_files",
    "grep": "search_code",
    "search": "search_code",
    "find_code": "search_code",
    "test": "run_test",
    "pytest": "run_test",
}

@dataclass(frozen=True)
class StructuredToolCall:
    tool: str
    arguments: dict[str, Any]
    original_action: str
    repaired_action: str | None
    invalid_action: bool
    repair_applied: bool


def render_tool_schemas() -> str:
    return json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2)


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.I | re.S)
    if fenced:
        stripped = fenced.group(1).strip()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("tool call must contain a JSON object")
    value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError("tool call must be a JSON object")
    return value


def normalize_action_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parse_structured_tool_call(response: str) -> StructuredToolCall:
    """Parse one JSON call and repair known action aliases."""
    value = _json_object(response)
    raw_tool = value.get("tool")
    if not isinstance(raw_tool, str) or not raw_tool.strip():
        raise ValueError("tool call requires a non-empty string field 'tool'")
    arguments = value.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("tool call field 'arguments' must be an object")
    original = raw_tool.strip()
    normalized = normalize_action_name(original)
    repaired = ACTION_ALIASES.get(normalized)
    tool = repaired or normalized
    repair_applied = repaired is not None
    invalid_action = repair_applied or tool not in TOOL_NAMES
    return StructuredToolCall(
        tool=tool,
        arguments=arguments,
        original_action=original,
        repaired_action=tool if repair_applied else None,
        invalid_action=invalid_action,
        repair_applied=repair_applied,
    )


def guess_original_action(response: str) -> str:
    """Best-effort action extraction for malformed model output."""
    for pattern in (
        r'"tool"\s*:\s*"([^"]+)"',
        r"Action:\s*([A-Za-z_][A-Za-z0-9_-]*)",
    ):
        match = re.search(pattern, response, re.I)
        if match:
            return match.group(1)
    return response.strip()[:200]
