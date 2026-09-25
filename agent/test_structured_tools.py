import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent.executor import AgentExecutor
from agent.tools.schema import TOOL_SCHEMAS, parse_structured_tool_call

class StructuredToolTest(unittest.TestCase):
    def test_canonical_schema_names(self):
        self.assertEqual(
            [schema["name"] for schema in TOOL_SCHEMAS],
            ["list_files", "search_code", "read_file", "apply_patch", "run_test"],
        )
        for schema in TOOL_SCHEMAS:
            self.assertEqual(set(schema), {"name", "description", "parameters"})
            self.assertEqual(schema["parameters"]["type"], "object")

    def test_alias_is_repaired(self):
        call = parse_structured_tool_call(
            json.dumps({"tool": "update_file", "arguments": {"patch": "diff"}})
        )
        self.assertEqual(call.tool, "apply_patch")
        self.assertEqual(call.original_action, "update_file")
        self.assertEqual(call.repaired_action, "apply_patch")
        self.assertTrue(call.invalid_action)
        self.assertTrue(call.repair_applied)

    def test_repaired_full_content_is_applied(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            target = root / "module.py"
            target.write_text("value = 1\n", encoding="utf-8")
            executor = AgentExecutor(root)
            step = executor.execute(json.dumps({
                "tool": "edit_file",
                "arguments": {"path": "module.py", "content": "value = 2\n"},
            }))
            self.assertTrue(step.success)
            self.assertTrue(step.repair_applied)
            self.assertEqual(step.repaired_action, "apply_patch")
            self.assertEqual(step.model_proposed_tool, "edit_file")
            self.assertEqual(step.executed_tool, "apply_patch")
            self.assertTrue(step.controller_intervened)
            self.assertEqual(step.intervention_reason, "action_repair")
            self.assertTrue(step.tool_success)
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\n")
            self.assertIn("+value = 2", executor.final_patch())

    def test_legacy_calculator_action_is_preserved(self):
        request = AgentExecutor.parse(
            'Thought: fix\nAction: write_file\n'
            'Action Input: {"path":"x.py","content":"fixed"}'
        )
        self.assertEqual(request.action, "write_file")
        self.assertFalse(request.invalid_action)
        self.assertFalse(request.repair_applied)

if __name__ == "__main__":
    unittest.main()
