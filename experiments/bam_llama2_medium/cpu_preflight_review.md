# CPU training preflight review — 2026-09-26

Source scope: 135 saved candidate logs under `/data0/xd` and local parallel
launch state. Identify full BAM suite sessions by at least20 named
`BamReadKeyTransformTest` checks; deduplicate identical session content.
This is the available-log sample, not every historical launch. Raw records:
`/data0/xd/rmt-cpu-history-audit-dedup.json`.

56 distinct full-suite sessions:47 passed,9 failed. The suite varied over time;
84 distinct method names appear across the saved versions. Failures involve9
distinct methods (10.7% of the historical union);6 remain among the current47
methods (12.8%). Subtest failures are grouped under their enclosing method.
Diagnostic and development runs are included; this is not a launch-only failure rate.

At least5 failed sessions caught runtime code defects: four shared MHA-path
initialization omissions (LocalO static, concatenation write mix, LocalV replace,
LocalV mode), and one configuration-class MRO conflict. Other failures included
test-fixture parameter unboxing/config deepcopy, invoking a Flax helper outside
its production compact forward, and asserting a nonzero gradient at dormant
zero-initialized read keys. These should not all be counted as training bugs.

The local parallel-launch sample has10 full BAM sessions, all passed. Two earlier
SingleOuterWrite preparation attempts failed their RMT-specific equivalence check
before the full BAM suite began. This favors change-based coverage for local RMT
edits, not deletion of general regression tests. In particular, retain a shared
MHA initialization smoke check when editing shared BAM/MHA setup paths.

## Measured acceleration

Same RMT worktree BAM suite (47 checks):359.146s serial versus140.74s with4
processes, each pinned to8 disjoint physical CPU cores (2.55x). All47 passed,
group durations93.30/106.91/128.72/140.74s. Physical host:64 cores/128 hardware
threads; shared-host load around40–42 at the start. This is one timing comparison,
not a controlled repeated benchmark. Source/logs:
`/data0/xd/rmt-cpu-bam-parallel4/manifest.json`, group logs and
`/data0/xd/rmt-cpu-bam-parallel4.log`.

Direct32 unembedding targeted checks: full-size parent/target parameter audits,
target full-model finite gradients/zero-init equivalence/boundary health,
nonzero direct32 read value/gradient reference, and C8 shared/independent fetch
key regression. Three CPU groups passed in56.04s, plus a short parameter audit
(about1 minute total). Logs: `/data0/xd/rmt-cpu-targeted-direct32.log`.
The earlier broad preflight used142.728s for4 RMT tests and359.146s for47 BAM
tests; AOT/queue were already ready before this CPU gate finished.

## Workflow

`launch_train_parallel.py` defaults to full regression, now CPU-parallel. Local
changes on a verified parent can explicitly use `--cpu-test-scope targeted`
with nonempty `--extra-test-script` entries. Scope follows affected code and
dependencies, rather than selecting only historically failing tests. Full
regression remains appropriate for shared runtime changes and merge checks.
Both scopes preserve runtime sealing, failure gating, AOT/queue parallelism,
actual training first-step confirmation and owned-training-TPU cleanup.

`preparation.json` records the exact scope/commands. The CPU runner stores every
selected test/group/affinity and propagates any group failure. Regression checks
verify empty targeted scope rejection, check short-circuiting, default full-suite
selection, and termination of a concurrent live group after another fails.
