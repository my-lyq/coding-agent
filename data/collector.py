"""Collect, filter, and format Agent trajectories as JSONL."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable
from filter import quality_issues
from formatter import format_trajectory

def collect_trajectories(paths: Iterable[Path], instruction: str, max_modified_files: int = 3) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rejected.append({"source": str(path), "reasons": [f"invalid_json:{exc}"]})
            continue
        issues = quality_issues(raw, max_modified_files)
        if issues:
            rejected.append({"source": str(path), "reasons": issues})
        else:
            accepted.append(format_trajectory(raw, instruction))
    return accepted, rejected

def write_jsonl(records: Iterable[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[root / "trajectory.json"])
    parser.add_argument("--instruction", default="Fix bug in examples/calculator.py")
    parser.add_argument("--output", type=Path, default=root / "data" / "train.jsonl")
    parser.add_argument("--max-modified-files", type=int, default=3)
    args = parser.parse_args()
    accepted, rejected = collect_trajectories(args.inputs, args.instruction, args.max_modified_files)
    write_jsonl(accepted, args.output)
    print(f"accepted={len(accepted)} rejected={len(rejected)} output={args.output}")
    for item in rejected:
        print(f"REJECT {item['source']}: {', '.join(item['reasons'])}")
    return 0 if accepted else 1

if __name__ == "__main__":
    raise SystemExit(main())
