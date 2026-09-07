"""Exercise the real pre-push hook without network or changes to project branches."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HOOKS = Path(__file__).resolve().parents[2] / ".githooks"


class PushGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="maxtext-push-guard-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.remote = self.root / "remote.git"
        self.env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_TERMINAL_PROMPT="0")
        self.run_git("init", "--bare", str(self.remote), cwd=self.root)
        self.run_git("init", "-b", "refactor-bam", str(self.repo), cwd=self.root)
        self.run_git("config", "user.name", "Push Guard Test")
        self.run_git("config", "user.email", "push-guard@example.invalid")
        self.run_git("config", "core.hooksPath", str(HOOKS))
        self.run_git("remote", "add", "origin", str(self.remote))
        self.run_git("commit", "--allow-empty", "-m", "base")
        self.run_git("push", "origin", "HEAD:refs/heads/refactor-bam")
        self.base = self.run_git("rev-parse", "HEAD").stdout.strip()

    def run_git(self, *args, cwd=None, ok=True):
        result = subprocess.run(["git", *args], cwd=cwd or self.repo, env=self.env,
                                capture_output=True, text=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def feature(self):
        self.run_git("switch", "-c", "codex/experiment")
        self.run_git("commit", "--allow-empty", "-m", "experiment")

    def rejected(self, *refspecs, cwd=None, remote="origin"):
        result = self.run_git("push", remote, *refspecs, cwd=cwd, ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("branch-target guard rejected", result.stderr)
        actual = self.run_git("rev-parse", "refs/heads/refactor-bam", cwd=self.remote).stdout.strip()
        self.assertEqual(actual, self.base)

    def test_main_same_name(self):
        self.run_git("commit", "--allow-empty", "-m", "main update")
        self.run_git("push", "origin", "HEAD:refactor-bam")

    def test_experiment_same_name(self):
        self.feature()
        self.run_git("push", "origin", "HEAD:refs/heads/codex/experiment")

    def test_head_wrong_target_and_direct_url(self):
        self.feature()
        self.rejected("HEAD:refactor-bam")
        self.rejected("HEAD:refactor-bam", remote=str(self.remote))

    def test_explicit_branch_and_sha_wrong_target(self):
        self.feature()
        self.rejected("refs/heads/codex/experiment:refs/heads/refactor-bam")
        tip = self.run_git("rev-parse", "HEAD").stdout.strip()
        self.rejected(f"{tip}:refs/heads/refactor-bam")
        self.run_git("push", "origin", f"{tip}:refs/heads/codex/experiment")

    def test_detached_head(self):
        self.run_git("checkout", "--detach")
        self.run_git("commit", "--allow-empty", "-m", "detached update")
        self.rejected("HEAD:refactor-bam")

    def test_protected_branch_requires_main_worktree_context(self):
        self.run_git("commit", "--allow-empty", "-m", "main update")
        self.feature()
        self.rejected("refs/heads/refactor-bam:refs/heads/refactor-bam")

    def test_linked_historical_worktree_without_hook_files(self):
        linked = self.root / "linked"
        self.run_git("worktree", "add", "-b", "codex/linked", str(linked))
        self.assertFalse((linked / ".githooks").exists())
        self.run_git("commit", "--allow-empty", "-m", "linked update", cwd=linked)
        self.rejected("HEAD:refactor-bam", cwd=linked)
        self.run_git("push", "origin", "HEAD:codex/linked", cwd=linked)

    def test_mixed_push_is_rejected_before_remote_update(self):
        self.feature()
        self.rejected("HEAD:codex/experiment", "HEAD:refactor-bam")
        result = self.run_git("show-ref", "--verify", "refs/heads/codex/experiment",
                              cwd=self.remote, ok=False)
        self.assertNotEqual(result.returncode, 0)

    def test_tag_and_experiment_deletion_but_protected_deletion_rejected(self):
        self.run_git("tag", "test-tag")
        self.run_git("push", "origin", "refs/tags/test-tag")
        self.rejected(":refs/heads/refactor-bam")
        self.feature()
        self.run_git("push", "origin", "HEAD:codex/experiment")
        self.run_git("push", "origin", ":refs/heads/codex/experiment")


if __name__ == "__main__":
    unittest.main()
