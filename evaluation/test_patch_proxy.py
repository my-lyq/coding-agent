import subprocess
import tempfile
import unittest
from pathlib import Path

from evaluation.patch_proxy import evaluate_patch


class PatchProxyTest(unittest.TestCase):
    def test_clean_base_patch_is_applicable(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            base = root / "repos" / "owner__repo" / "commit"
            base.mkdir(parents=True)
            subprocess.run(["git", "init", "-q", str(base)], check=True)
            subprocess.run(["git", "-C", str(base), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(base), "config", "user.name", "Test"], check=True)
            (base / "module.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(base), "add", "module.py"], check=True)
            subprocess.run(["git", "-C", str(base), "commit", "-qm", "base"], check=True)
            commit = subprocess.run(
                ["git", "-C", str(base), "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            target = root / "repos" / "owner__repo" / commit
            base.rename(target)
            patch = (
                "--- a/module.py\n"
                "+++ b/module.py\n"
                "@@ -1 +1 @@\n"
                "-value = 1\n"
                "+value = 2\n"
            )
            result = evaluate_patch(
                instance_id="owner__repo-1",
                repo="owner/repo",
                base_commit=commit,
                patch=patch,
                repos_dir=root / "repos",
                workspaces_dir=root / "workspaces",
            )
            self.assertTrue(result["clean_base_verified"])
            self.assertTrue(result["proxy_patch_applicable"])
            self.assertEqual(result["modified_files"], ["module.py"])
            self.assertEqual(result["patch_lines_added"], 1)
            self.assertEqual(result["patch_lines_deleted"], 1)

    def test_empty_patch_is_not_applicable(self):
        result = evaluate_patch(
            instance_id="owner__repo-1",
            repo="owner/repo",
            base_commit="deadbeef",
            patch="(no changes)",
        )
        self.assertFalse(result["proxy_patch_applicable"])
        self.assertEqual(result["error"], "empty_patch")


if __name__ == "__main__":
    unittest.main()
