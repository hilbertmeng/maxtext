---
name: tpu-training
description: Manage preemptible TPU lifecycle and MaxText training on GCP for xd's BAM/Llama2Medium experiments. Use when creating, launching, monitoring, stopping, resuming, hot-retraining, or closing out TPU training runs and comparing same-step loss gaps.
---

# TPU Training

Run Codex locally. Run GCP lifecycle commands on `tpu-ag` through `$tpu-ag`'s persistent
SSH socket `/tmp/ssh-tpu-ag-xd.sock`. Use `$tpu-diagnostics` for checkpoint probes/ablations.

Defaults: repo `/home/xd/projects/maxtext` (`refactor-bam`); tpu-ag scripts
`/home/lishengping/xd/projects`; TPU VM repo `/home/lishengping/xd/projects/maxtext`;
project `newproject-1-451205`; TPU `v5p-16`; formal v5p region policy
`PRIMARY_ZONE=europe-west4-b`, `BACKUP_ZONES=`; output
`gs://newproject-1-llm_base_models_us-central1/log/`.
Authoritative orchestration sources are `/home/xd/projects/xd_tpu_scripts`; deploy only those
exact files to tpu-ag and verify matching hashes.

Classify by intent: a short speed/profile arm is `$tpu-diagnostics` even at full layer count;
use `run_exp_xd.sh` only for a RUN intended to train through its registered plan.

## Start Training

Uses `run_exp_xd.sh` → `auto_train_xd_maxtext.sh` and `run_registry.py`.

Default validation is the pinned local BAM test suite followed by the target RUN's
`FIRST_STEP`. Add a standalone v6e check only for TPU-specific uncertainty those two gates do
not cover; it is not a routine prerequisite for training.

1. Choose `EXP`, TPU `ID`, `MODE`, and direct experimental baselines in `COMPARE_RUNS`.
   For formal spot `v5p`, set `PRIMARY_ZONE` and the user-directed `BACKUP_ZONES`; `ZONE` defaults
   to `PRIMARY_ZONE` for the active assignment.
   Use `install+train` for a new/reprovisioned VM and `train` for an installed READY VM.
   Formal training defaults to `scan_layers=True`; verify the resolved class value before launch.
   Use another setting only when the user explicitly requests it.
2. Before a parameter-tree change, use a new run name/GCS prefix. Commit the prepared runtime
   code, push it to `origin/refactor-bam`, and use its full hash. Commit/push first-step fixes and
   update the RUN hash before relaunch.

```bash
git status --short --branch && git push origin refactor-bam
CODE_COMMIT=$(git rev-parse HEAD)
```

3. Launch on tpu-ag in tmux. `run_exp_xd.sh` rejects unpushed hashes; registry records
   `code_commit`; every initial/retry/preemption launch checks out that exact detached commit.

For a hot switch on an allocated preemptible TPU, first make the new RUN launch-ready: push its
commit and prepare its registry, environment and AOT artifact. At handoff, stop the old auto-trainer
and worker processes while retaining the READY TPU/queued resource, then launch the new RUN
immediately with `MODE=train`. Mark the old RUN paused and handle its TensorBoard bookkeeping while
waiting; ownership transfers only after the new RUN reaches `FIRST_STEP`.

```bash
EXP=BamLlama2Medium ID=0 MODE=install+train
PRIMARY_ZONE=europe-west4-b BACKUP_ZONES=
CODE_COMMIT=$(git rev-parse HEAD)
BASES=Llama2Medium
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "tmux new-session -d -s ${EXP}-TPU${ID}-xd \
   'env EXP=$EXP ID=$ID MODE=$MODE PRIMARY_ZONE=$PRIMARY_ZONE BACKUP_ZONES=$BACKUP_ZONES \
   BRANCH=bam CODE_COMMIT=$CODE_COMMIT \
   COMPARE_RUNS=$BASES bash /home/lishengping/xd/projects/run_exp_xd.sh'"
```

4. Verify registration immediately:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  '/home/lishengping/xd/projects/run_registry.py status'
# Registry and auto_pid environment must agree on CODE_COMMIT and COMPARE_RUNS.
```

5. Gate launch with the registry's one-shot waiter. It polls the worker log internally and exits
   on `FIRST_STEP:`, confirmed failure, or timeout. Launch succeeds only after this command
   returns `FIRST_STEP:`.

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py wait-step '$EXP' 0"
```

After `FIRST_STEP`, copy the RUN registry's seven-character `code_commit` prefix into its
`exp.py` class as `# code_commit: HASH`. If a first-step fix changes the runtime commit, replace
the comment after the successful relaunch. The registry remains authoritative for the full hash;
the later metadata commit is not the RUN's runtime hash.

For every sealed full-layer RUN, cross-topology AOT-compile the exact target topology before its
target launch (in parallel with the resource queue when useful). Key the executable by commit,
environment, topology, training shapes and schedule;
stage the matching environment, detached commit and executable during TPU installation, then
require `Loaded compiled function!` plus an actual first step. Recompile when any key changes.
Launch with `COMPILED_TRAINSTEP_GCS=gs://...`; auto-train stages that artifact on every recovery.
For checkpoint resume, compile the original total schedule, never the remaining-step count; the
restored optimizer `state.step` selects the resumed learning rate. Require the first resumed step
and logged LR to match the checkpoint and original schedule; stop immediately on mismatch.
Generate formal AOT artifacts only in a validated TPU-VM environment. If the target TPU becomes
READY first, start its native compile immediately and reserve any later AOT artifact for recovery;
do not replace a healthy pre-first-step compile merely because the artifact finishes.

6. Use the same one-shot gate for the step 10–14 speed check. Compare `~steps/s` with direct
   `compare_runs` and the expected architectural delta, then record it tersely in the `exp.py`
   class. Immediately report and investigate a material unexplained speed deviation; mark the
   class comment `!?` or `!!` until resolved.

Once training is steady, ignore isolated/short-lived `steps/s` changes; preemptible TPU throughput
is otherwise stable and these are normally checkpoint or I/O scheduling effects.

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py wait-step '$EXP' 14"
```

A successful launch is not task completion. Unless the user requested launch only, transition
immediately to **Monitor Training** and keep the current turn/goal active until the RUN stops or
completes.

## Monitor Training

Use `run_registry.py status` for liveness and `loss-report` for loss. Reserve TensorBoard sync
for run closeout.

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  '/home/lishengping/xd/projects/run_registry.py status'
```

`status` persists and displays queue age, no-progress age, preemption count, recent READY lease
durations, and `ACTION`. Treat one long queue and repeated short leases as separate recovery
signals; either may require a passive candidate in another zone.

At each wake, compare observed step gain over elapsed time with logged steps/s. A large shortfall
plus stale per-worker train-log mtimes means a hang even if TPU is READY and processes are alive;
auto-train owns this machine-level liveness check. It arms after three progress samples, waits
`max(600s, 20 / steps_per_second)` (or per-RUN `STALE_TIMEOUT_SECONDS`), requires two all-worker
SSH confirmations with valid speed, then recreates. Codex verifies watchdog health and owns
loss/trend decisions.

Each `run_registry/<RUN>.json` contains run/TPU/launch data, current `base_output_directory`,
report interval/window/cursor, and direct `compare_runs`. `loss-report` refreshes live worker-0
logs, merges repeated steps
(latest launch wins) into persistent `run_registry/loss_cache/`, then prints every
`gap = RUN loss - BASE loss` (negative favors RUN) at exact common steps:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  '/home/lishengping/xd/projects/run_registry.py loss-report RUN --through-step STEP'
```

It samples `step % 5 == 0` inside each ±25-step window, preserving the historical 11-sample
gap definition even though future TensorBoard files record every 10 steps. For each RUN−BASE,
print one cumulative horizontal table with only `step`, `gap`, and `r200`; omit absolute losses
and split after about 20 steps into another horizontal block. Report every direct `compare_runs`
entry, including completed BASEs with no new common steps; omit one only after the user explicitly
removes it or the registry is updated. Here
`r200 = (abs(gap[s]) - abs(gap[s-200])) / abs(gap[s-200])`: negative means the gap
magnitude shrank from the preceding window, positive means it grew. Summarize the current gap
level with the mean and range of the latest 5–8 reported points. Judge direction from successive
signed-gap window means and sign changes; use `r200` only for magnitude change, especially near zero.

Investigate every anomalous monitoring metric immediately and restore 200-step reporting until
its root cause is resolved.
Treat a gap sign crossing or trend reversal as material even when its magnitude is small; report
the transition explicitly and monitor it closely until its direction is clear.

At every due milestone:

1. Run one shared `status`, verify each due RUN's latest checkpoint has
   `commit_success.txt`, then run `loss-report`; summarize healthy checkpoints tersely and
   investigate pending or rollback immediately.
2. Report the cumulative horizontal rows; judge stability from signed-gap trends plus `r200`.
3. Mark the cursor:

```bash
RUN=BamLlama2Medium STEP=200
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py mark-reported $RUN $STEP"
```

Auto-train caches logs before clean/crash deletion. Before manually deleting a TPU, run
`run_registry.py collect-loss RUN`. If an old completed BASE has no cache, export its already
synced TensorBoard once with `scripts/export_tensorboard_loss.py`, copy the small `STEP LOSS`
file to tpu-ag, and run `run_registry.py import-loss BASE --file FILE`.

Require explicit user permission for any stop before step 2,800. At/after 2,800, use
`MHA advantage = MHA loss - RUN loss`: an advantage below 0.08 and still shrinking rapidly
likely finishes below 0.05 and may stop; an advantage above 0.05 with curves becoming parallel
may merit continuing. Train a configuration with a credible loss, speed, or parameter-efficiency
gain longer—possibly to completion—to verify that the gain persists; material parameter reduction
with near-baseline loss counts even when wall-clock speed is unchanged. A configuration still
unlikely to beat its direct baseline and offering no other gain may stop at 2,800. Also stop a run
clearly dominated by a prior failed configuration.

For multiple runs, use one shared wake-up and batch-check all runs; use per-run wake-ups only
for anomalies or imminent completion/decisions. Stable runs may accumulate about five 200-step
windows per loss report. Independently, when preemptions are frequent, wake for a shared health
check within ~10–12 minutes; an unchanged health check need not trigger a loss report. Estimate
from steps/s and stay silent between wakes; modest overshoot is fine.

## Stop Training

Uses the auto-train PID/tmux session, an exact train-process match, `delete_tpu_xd.sh`,
`run_registry.py stop`, and the closeout procedure below.

1. Stop the creator first, or it can recreate the TPU:

```bash
RUN=BamLlama2Medium SESSION=${RUN}-TPU0-xd
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "kill \$(cat /home/lishengping/xd/projects/logs/${RUN}.pid) 2>/dev/null; \
   tmux kill-session -t $SESSION 2>/dev/null"
```

2. Record the stop UTC, then stop only the intended train process. Make the pattern unable to
   match its own shell; pass that UTC to the later registry `stop` so the final READY lease ends at
   SIGTERM rather than after checkpointing and teardown:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "gcloud compute tpus tpu-vm ssh --internal-ip xd-v5p-16-0-maxtext \
   --zone=us-central1-a --project=newproject-1-451205 --worker=all \
   --command=\"pkill -TERM -f '[M]axText/train.py.*run_name=$RUN'\""
```

Wait for any Orbax SIGTERM checkpoint to finish. Preserve the final log, then use the shared
helper as the sole resource-release path:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py collect-loss $RUN"
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  'bash /home/lishengping/xd/projects/delete_tpu_xd.sh \
   xd-v5p-16-0-maxtext us-central1-a newproject-1-451205'
```

The helper uses `gcloud compute tpus tpu-vm delete`, deletes the queued-resource, retries
transient failures, and succeeds only after both describes return `NOT_FOUND`.

Record the stop only after teardown is verified, then run Close Out:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py stop $RUN \
   --status stopped --step STEP --reason REASON --end-utc STOP_UTC"
```

## Close Out a Completed Run

Uses auto-train's verified cleanup and automatic TensorBoard sync, final `run_registry` state,
and `exp.py` result comments.

On clean exit, auto-train must verify node and queued-resource deletion through
`delete_tpu_xd.sh`, then mark the registry complete.

For every stopped/completed run:

1. Verify final step/status and resource teardown with `run_registry.py status --all` and GCP.
2. Report final same-step/window gaps, cumulative trajectory, and whether prior extrapolation
   matched.
3. Run `run_registry.py lease-report RUN`; report the TPU type, full active-zone path and switch
   points, preemption count, and **every** chronological READY lease with UTC start/end and
   duration—not only `status`'s recent three. Keep unknown starts visible and distinguish the final
   manual stop.
4. Append the same evidence to `experiments/tpu_region_preemption_history.md`: one assignment row
   per active-zone stint grouped by RUN, plus one event row per READY lease globally sorted by UTC
   end time. Record passive candidates separately; they are not region switches. This ordering
   exposes correlated preemptions and supplies evidence for region/time-of-day choices.
5. Replace the experiment class's running comment with one terse line containing speed, final
   step, and the main conclusion against its registered direct `compare_runs`; retain every
   decision-relevant direct baseline. Express loss, speed, parameters, cache, and compute as
   deltas or ratios versus that baseline; use absolute values only as supplemental context, e.g.:

```python
# ~0.280 steps/s; completed 13,500. dloss -0.0678 (-2.77%) vs MHA @13,400
```

## Hot Retrain

Commit/push every fix and restart the launcher with the new `CODE_COMMIT`; registry registration
updates the RUN hash. Reuse a RUN only when its checkpoint parameter tree remains compatible.
For a no-stop migration, restart only auto-train: it adopts the existing `train.py`, while the
new hash takes effect on the next relaunch. Changing the live training code itself requires a
training restart. Record `OLD_STEP` before relaunch, then bind the launch gate to a strictly newer
step:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py wait-step '$RUN' 0 --after-step '$OLD_STEP'"
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py wait-step '$RUN' '$((OLD_STEP + 14))' \
   --after-step '$OLD_STEP'"
```

When replacing a train process, wait for both the process and any Orbax save to finish, verify the
latest checkpoint is committed and its data cursor agrees, then require the exact `pgrep` to be
empty before relaunch. Stop a standalone creator before deleting its TPU and queued resource.

## Create Standalone v6e-1

For an unregistered diagnostic TPU, use the bounded standalone launcher. It uses absolute paths
for backgrounding/logging, repairs/validates xd's named gcloud configuration, checks tpu-ag
disk/log health, submits once, polls the accepted queue, records its exact PID, and exits after
installation:

```bash
NAME=xd-v6e-1-bamdiag ZONE=us-east5-a
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/start_standalone_tpu.sh \
   '$NAME' v6e-1 '$ZONE' install_xd_maxtext_jax081.sh"
```

Inspect `logs/${NAME}-create.log` on tpu-ag. Release it only through `delete_tpu_xd.sh`, which
stops the recorded creator before deleting and verifying both resources.

## Recover Preemption

Uses `auto_train_xd_maxtext.sh`, the RUN's registered commit, and `delete_tpu_xd.sh`.

- For spot `v6e-1`, query
  `serviceusage.googleapis.com/v1beta1/projects/$NUM/services/tpu.googleapis.com/consumerQuotaMetrics?view=FULL`
  (`effectiveLimit>0` or override `-1`; missing means zero), then intersect with `gcloud alpha compute tpus
  accelerator-types list --zone=ZONE --filter=type=v6e-1`; treat quota and current capacity as
  separate conditions. Prefer proven zones `us-central1-a`, `europe-west4-a`, then `us-east5-a`.
- Preserve WAITING_FOR_RESOURCES/PROVISIONING queues; deleting resets queue position.
- If best-effort pods are repeatedly reclaimed before useful progress, first validate the exact
  topology/zone with a passive FLEX_START queued-resource. After creation succeeds, freeze the RUN
  and activate exactly one flex trainer with a duration longer than its ETA.
- For formal spot `v5p`, queue and recover in `PRIMARY_ZONE`; passive candidates come only from the
  user-directed `BACKUP_ZONES`. Revisit both from the shared region/preemption history.
- In xd's v5p experience, maintenance warning + refused SSH is almost always preemption. Start
  reclaim immediately.
- A queued-resource `SUSPENDED; stateInitiator=SERVICE` is terminal even if the TPU node has
  already disappeared (empty/NOT_FOUND node state).
  Auto-train must release both resources through `delete_tpu_xd.sh`, recreate, reinstall, apply
  `CODE_COMMIT`, and resume the same RUN from its latest GCS checkpoint.
- Storage is explicit RUN state; never let a replacement zone select an empty prefix. Prefer
  same-zone recovery. If the source TPU is terminal and recovery must change zones, stop its reclaim
  launcher, then run `run_registry.py migrate-storage RUN --to-base B_BUCKET`. This copies the latest
  committed checkpoint, verifies it, and atomically updates `base_output_directory`; launch reads that
  registry value. Accept recovery only after `FIRST_STEP` exceeds the migrated step **and** the next
  periodic checkpoint commits. A step directory without `commit_success.txt` is incomplete.
- Treat a post-maintenance SSH timeout as `alive=unknown`.

## TensorBoard Service

Auto-train publishes `log/tensorboard_complete/RUN`; local `maxtext-tensorboard-sync.timer`
retries the full sync independently of Codex. Use manual sync only to repair a reported failure:

```bash
RUN=Llama2Medium
mkdir -p ~/tensorboard_logs/$RUN
~/google-cloud-sdk/bin/gsutil -m rsync -r \
  gs://newproject-1-llm_base_models_us-central1/log/summaries/train/$RUN/ \
  ~/tensorboard_logs/$RUN/
/home/xd/miniconda3/envs/tune/bin/tensorboard \
  --logdir ~/tensorboard_logs --port=6007 --bind_all
```

Open `http://localhost:6007` (or the configured host alias). Checkpoints are under
`.../log/<RUN>/checkpoints/`.

## Guardrails

- Keep BAM `full` runs as capability-ceiling experiments; test speed variants as separate runs.
- Report architectural parameter overhead in per-layer `W_Q = d_model^2` units; include
  biases/gates only when comparison-relevant.
- Test zero-initialized BAM read updates at layer 1+ after step 0; step 0 has zero LR and layer 0
  has zero `M_in`.
- Use a new RUN after adding/removing conditional parameters; resume only compatible trees.
- For resume-only retention use `max_to_keep=2`, `keep_period=None` (normalize non-positive
  config values to `None`). `keep_period=1000` permanently accumulates large checkpoints.
