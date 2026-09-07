# Branch-target guard

Install from the main worktree:

```sh
git config --local core.hooksPath "$(git rev-parse --show-toplevel)/.githooks"
```

The absolute path and shared repository configuration cover existing linked worktrees,
including historical commits without `.githooks`. Independent clones need their own installation.

Branch updates require matching source/destination names; `HEAD` and explicit commit IDs must
resolve to the current branch tip. Pushes to `refactor-bam`, `main`, or `master` also require
checking out that branch. Deleting those protected branches is rejected. Tag pushes and
deleting experiment branches remain available. A mixed push is rejected as a whole on any violation.

This is a local accident guard, not server-side access control: `--no-verify` bypasses Git hooks.
After a push, fetch the corresponding remote branch and verify its commit matches the intended tip.

Test actual pushes against temporary local bare repositories:

```sh
python3 -m unittest discover -s scripts/tests -p test_git_push_guard.py
```
