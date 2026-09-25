import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent.task import Task
from rollout.recorder import TrajectoryRecorder
from rollout.runner import AgentRolloutRunner


class SequencePolicy:
    def __init__(self, actions: list[dict[str, object]]) -> None:
        self.actions = list(actions)

    def next_action(self, steps):
        if not self.actions:
            return None
        return json.dumps(self.actions.pop(0))


class AgentRolloutPlannerTest(unittest.TestCase):
    def test_controller_blocks_then_recovers_through_full_flow(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            (root / "test_module.py").write_text(
                "import unittest\n"
                "import module\n\n"
                "class ValueTest(unittest.TestCase):\n"
                "    def test_value(self):\n"
                "        self.assertEqual(module.value, 2)\n",
                encoding="utf-8",
            )
            command = "python -m unittest -q"
            task = Task(
                task_id="synthetic__planner",
                repo_path=root,
                problem_statement="Make module.value equal two.",
                test_command=command,
            )
            patch_text = (
                "--- a/module.py\n"
                "+++ b/module.py\n"
                "@@ -1 +1 @@\n"
                "-value = 1\n"
                "+value = 2\n"
            )
            policy = SequencePolicy([
                {"tool": "read_file", "arguments": {"path": "module.py"}},
                {
                    "tool": "search_code",
                    "arguments": {"query": "value", "path": "."},
                },
                {"tool": "read_file", "arguments": {"path": "module.py"}},
                {"tool": "apply_patch", "arguments": {"patch": patch_text}},
                {"tool": "run_test", "arguments": {"command": command}},
            ])
            output = root / "trajectory.json"
            runner = AgentRolloutRunner(
                task,
                policy,
                TrajectoryRecorder(output),
                max_steps=5,
                test_timeout=30,
            )

            result = runner.run()
            record = json.loads(output.read_text(encoding="utf-8"))

            self.assertTrue(result.success)
            self.assertEqual(record["steps"][0]["previous_phase"], "LOCATE")
            self.assertEqual(record["steps"][0]["phase"], "LOCATE")
            self.assertEqual(record["steps"][0]["model_proposed_tool"], "read_file")
            self.assertIsNone(record["steps"][0]["executed_tool"])
            self.assertTrue(record["steps"][0]["controller_intervened"])
            self.assertEqual(
                record["steps"][0]["intervention_reason"],
                "illegal_phase_action",
            )
            self.assertFalse(record["steps"][0]["tool_success"])
            self.assertTrue(
                record["steps"][0]["observation"].startswith(
                    "Planning Controller blocked"
                )
            )
            transitions = [
                (step["previous_phase"], step["phase"])
                for step in record["steps"]
                if step["previous_phase"] != step["phase"]
            ]
            self.assertEqual(
                transitions,
                [
                    ("LOCATE", "UNDERSTAND"),
                    ("UNDERSTAND", "MODIFY"),
                    ("MODIFY", "VERIFY"),
                ],
            )
            self.assertIn("+value = 2", record["patch"])


if __name__ == "__main__":
    unittest.main()
