import unittest

from agent.executor import Step
from agent.planner import Phase, PlanningController


def completed(action: str, success: bool = True) -> Step:
    return Step(
        thought="test",
        action=action,
        action_input={},
        observation="ok" if success else "failed",
        success=success,
    )


class PlanningControllerTest(unittest.TestCase):
    def test_initial_phase_and_allowed_tools(self):
        planner = PlanningController()
        self.assertEqual(planner.phase, Phase.LOCATE)
        self.assertEqual(planner.allowed_tools, {"list_files", "search_code"})
        for action in ("read_file", "apply_patch", "run_test"):
            with self.subTest(action=action):
                self.assertFalse(planner.validate(action).allowed)

    def test_main_phase_progression_and_failed_test_loop(self):
        planner = PlanningController()
        previous, current = planner.observe(completed("search_code"))
        self.assertEqual((previous, current), (Phase.LOCATE, Phase.UNDERSTAND))
        self.assertTrue(planner.validate("read_file").allowed)

        previous, current = planner.observe(completed("read_file"))
        self.assertEqual((previous, current), (Phase.UNDERSTAND, Phase.MODIFY))
        self.assertTrue(planner.has_read_file)
        self.assertTrue(planner.validate("apply_patch").allowed)

        previous, current = planner.observe(completed("apply_patch"))
        self.assertEqual((previous, current), (Phase.MODIFY, Phase.VERIFY))
        self.assertTrue(planner.validate("run_test").allowed)

        previous, current = planner.observe(completed("run_test", success=False))
        self.assertEqual((previous, current), (Phase.VERIFY, Phase.UNDERSTAND))

    def test_search_limit_requires_read_file(self):
        planner = PlanningController()
        for _ in range(3):
            planner.observe(completed("search_code"))

        self.assertFalse(planner.validate("search_code").allowed)
        self.assertFalse(planner.validate("apply_patch").allowed)
        self.assertTrue(planner.validate("read_file").allowed)
        decision = planner.validate("search_code")
        self.assertIn("three consecutive", decision.observation)
        self.assertEqual(decision.reason, "repeated_search")

    def test_apply_patch_requires_a_successful_read(self):
        planner = PlanningController()
        planner.phase = Phase.MODIFY
        decision = planner.validate("apply_patch")
        self.assertFalse(decision.allowed)
        self.assertIn("until at least one read_file succeeds", decision.observation)
        self.assertEqual(decision.reason, "modify_without_read")

    def test_failed_read_does_not_unlock_modify(self):
        planner = PlanningController()
        planner.observe(completed("search_code"))
        previous, current = planner.observe(completed("read_file", success=False))
        self.assertEqual((previous, current), (Phase.UNDERSTAND, Phase.UNDERSTAND))
        self.assertFalse(planner.has_read_file)
        self.assertFalse(planner.validate("apply_patch").allowed)

    def test_repeated_list_files_requires_search(self):
        planner = PlanningController()
        planner.observe(completed("list_files"))
        planner.observe(completed("list_files"))
        self.assertFalse(planner.validate("list_files").allowed)
        self.assertTrue(planner.validate("search_code").allowed)


if __name__ == "__main__":
    unittest.main()
