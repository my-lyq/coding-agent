import unittest
from filter import modified_files, quality_filter, quality_issues
from formatter import format_trajectory

def sample(success=True, patch=None):
    patch = patch if patch is not None else "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-bad\n+good\n"
    return {"steps": [
        {"thought":"inspect","action":"read_file","action_input":{"path":"a.py"},"observation":"bad\n","success":True},
        {"thought":"verify","action":"run_test","action_input":{"command":"test"},"observation":"OK" if success else "FAILED","success":success}], "final_patch":patch}

class PipelineTest(unittest.TestCase):
    def test_valid_schema(self):
        raw=sample(); self.assertTrue(quality_filter(raw)); result=format_trajectory(raw,"Fix bug")
        self.assertEqual(set(result),{"instruction","input","trajectory","answer"})
        self.assertIn("### File: a.py",result["input"])
    def test_failed_test(self): self.assertIn("final_test_failed",quality_issues(sample(False)))
    def test_empty_patch(self): self.assertIn("empty_patch",quality_issues(sample(patch="(no changes)")))
    def test_too_many_files(self):
        patch="".join(f"--- a/{i}.py\n+++ b/{i}.py\n-x\n+y\n" for i in range(4))
        self.assertFalse(quality_filter(sample(patch=patch))); self.assertEqual(len(modified_files(patch)),4)
    def test_recovery_is_kept(self):
        raw=sample(); raw["steps"].insert(1,{"action":"run_test","success":False})
        self.assertTrue(quality_filter(raw))

if __name__=="__main__": unittest.main()
