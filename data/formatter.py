"""Format executor output as an SFT record."""
from __future__ import annotations
import json
from typing import Any

def repository_context(raw: dict[str, Any]) -> str:
    files: dict[str, str] = {}
    for step in raw.get("steps", []):
        if step.get("action") == "read_file" and step.get("success") is True:
            path = step.get("action_input", {}).get("path")
            content = step.get("observation")
            if isinstance(path, str) and isinstance(content, str) and path not in files:
                files[path] = content
    return "\n\n".join(f"### File: {path}\n{content}" for path, content in files.items())

def format_trajectory(raw: dict[str, Any], instruction: str, repository: str | None = None) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "input": repository if repository is not None else repository_context(raw),
        "trajectory": [{
            "thought": str(step.get("thought", "")),
            "action": f"{step.get('action', '')}({json.dumps(step.get('action_input', {}), ensure_ascii=False, sort_keys=True)})",
            "observation": str(step.get("observation", "")),
        } for step in raw.get("steps", [])],
        "answer": str(raw.get("final_patch", "")),
    }
