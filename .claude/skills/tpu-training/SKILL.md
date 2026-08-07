---
name: tpu-training
description: Manage preemptible TPU lifecycle and MaxText training on GCP for xd's BAM/Llama2Medium experiments. Use when creating, launching, monitoring, stopping, resuming, hot-retraining, or closing out TPU training runs and comparing same-step loss gaps.
---

# TPU Training

Run Codex locally. Run GCP lifecycle commands on `tpu-ag` through `$tpu-ag`'s persistent
SSH socket `/tmp/ssh-tpu-ag-xd.sock`. Use `$tpu-diagnostics` for checkpoint probes/ablations.

Defaults: repo `/home/xd/projects/maxtext` (`refactor-bam`); tpu-ag scripts
`/home/lishengping/xd/projects`; TPU VM repo `/home/lishengping/xd/projects/maxtext`;
project `newproject-1-451205`; zone `us-central1-a`; TPU `v5p-16`; output
`gs://newproject-1-llm_base_models_us-central1/log/`.

## Start Training

Uses `run_exp_xd.sh` → `auto_train_xd_maxtext.sh`, `run_registry.py`, and the mandatory
`watch_train_xd.sh` watcher.

1. Choose `EXP`, TPU `ID`, `MODE`, and direct experimental baselines in `COMPARE_RUNS`.
   Use `install+train` for a new/reprovisioned VM and `train` for an installed READY VM.
2. Before a parameter-tree change, use a new run name/GCS prefix. Commit the prepared runtime
   code, push it to `origin/refactor-bam`, and use its full hash. If first-step debugging changes
   code, make/push another commit and update the RUN hash; do not wait for `FIRST_STEP` to commit.

```bash
git status --short --branch && git push origin refactor-bam
CODE_COMMIT=$(git rev-parse HEAD)
```

3. Launch on tpu-ag in tmux. `run_exp_xd.sh` rejects unpushed hashes; registry records
   `code_commit`; every initial/retry/preemption launch checks out that exact detached commit.

```bash
EXP=BamLlama2Medium ID=0 MODE=install+train
CODE_COMMIT=$(git rev-parse HEAD)
BASES=Llama2Medium
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "tmux new-session -d -s ${EXP}-TPU${ID}-xd \
   'env EXP=$EXP ID=$ID MODE=$MODE BRANCH=bam CODE_COMMIT=$CODE_COMMIT \
   COMPARE_RUNS=$BASES bash /home/lishengping/xd/projects/run_exp_xd.sh'"
```

4. Verify registration immediately:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  '/home/lishengping/xd/projects/run_registry.py status'
# Registry and auto_pid environment must agree on CODE_COMMIT and COMPARE_RUNS.
```

5. **Always attach the watcher as part of launch. Launch is not handed off until it is
   watching `FIRST_STEP:|ERR:`.** Copy it to worker 0, then run it in a backgrounded local
   shell/exec session with output notification:

```bash
TPU=xd-v5p-16-0-maxtext
LOG=/home/lishengping/train_${EXP}_xd.log
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "gcloud compute tpus tpu-vm scp --internal-ip \
   /home/lishengping/xd/projects/watch_train_xd.sh $TPU:~/ \
   --zone=us-central1-a --project=newproject-1-451205 --worker=0"
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "gcloud compute tpus tpu-vm ssh --internal-ip $TPU \
   --zone=us-central1-a --project=newproject-1-451205 --worker=0 \
   --command='bash ~/watch_train_xd.sh $LOG'"
```

Never use a watcher `pkill` pattern containing the train-log name; it can match itself. Kill
old watchers only by watcher PID or `pkill -f watch_train_xd.sh`.

6. After 30–40 stable steps, record `~steps/s` tersely in the experiment class in `exp.py`.

## Monitor Training

Use `run_registry.py status` for liveness and `loss-report` for loss. Do not use timers,
TensorBoard sync, plots, `_status.json`, or alert files during training.

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  '/home/lishengping/xd/projects/run_registry.py status'
```

At each wake, compare observed step gain over elapsed time with logged steps/s. A large shortfall
plus stale per-worker train-log mtimes means a hang even if TPU is READY and processes are alive;
auto-train owns this machine-level liveness check. It arms after three progress samples, waits
`max(600s, 20 / steps_per_second)` (or per-RUN `STALE_TIMEOUT_SECONDS`), requires two all-worker
SSH confirmations, then recreates. Any worker SSH failure or missing/invalid speed vetoes
deletion. Codex owns
loss/trend decisions and verifies the watchdog rather than duplicating its normal work.

Each `run_registry/<RUN>.json` contains run/TPU/launch data, report interval/window/cursor,
and direct `compare_runs`. `loss-report` refreshes live worker-0 logs, merges repeated steps
(latest launch wins) into persistent `run_registry/loss_cache/`, then prints every
`gap = RUN loss - BASE loss` (negative favors RUN) at exact common steps:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  '/home/lishengping/xd/projects/run_registry.py loss-report RUN --through-step STEP'
```

It samples `step % 5 == 0` inside each ±25-step window, preserving the historical 11-sample
gap definition even though future TensorBoard files record every 10 steps. Print cumulative
`step`, `run`, `base`, `gap`, and `r200` as horizontal rows; split into more row blocks when
long, never transpose milestones into vertical table rows. Here
`r200 = (abs(gap[s]) - abs(gap[s-200])) / abs(gap[s-200])`: negative means the gap
magnitude shrank from the preceding window, positive means it grew. Summarize the current gap
level with the mean of the latest 5–8 reported points (and its range), not one point; use recent
`r200` values for direction.

At every due milestone:

1. Run one shared `status`, then `loss-report` once per due RUN.
2. Report the cumulative horizontal rows; use `r200`, not visual flatness, for stability.
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

Never stop before step 2,800 without explicit user permission. At/after 2,800, use
`MHA advantage = MHA loss - RUN loss`: an advantage below 0.08 and still shrinking rapidly
likely finishes below 0.05 and may stop; an advantage above 0.05 with curves becoming parallel
may merit continuing. Train a configuration with a credible loss or speed gain longer—possibly
to completion—to verify that the gain persists. A configuration still unlikely to beat its direct
baseline may stop at 2,800. Also stop a run clearly dominated by a prior failed configuration.

For multiple runs, use one shared wake-up and batch-check all runs; use per-run wake-ups only
for anomalies or imminent completion/decisions. Independently, lengthen the shared sleep for
stable runs—normally enough to collect ~5 report intervals. Estimate from steps/s; modest
overshoot is fine. Stay silent, and do not repeatedly sync TensorBoard or reread this skill.

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

2. Stop only the intended train process. Make the pattern unable to match its own shell:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "gcloud compute tpus tpu-vm ssh --internal-ip xd-v5p-16-0-maxtext \
   --zone=us-central1-a --project=newproject-1-451205 --worker=all \
   --command=\"pkill -TERM -f '[M]axText/train.py.*run_name=$RUN'\""
```

Wait for any Orbax SIGTERM checkpoint to finish. Then release resources through the shared
helper; first preserve the final log, then do not duplicate raw delete commands:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py collect-loss $RUN"
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  'bash /home/lishengping/xd/projects/delete_tpu_xd.sh \
   xd-v5p-16-0-maxtext us-central1-a newproject-1-451205'
```

The helper uses the V2 command `gcloud compute tpus tpu-vm delete`, then deletes the
queued-resource, retries transient failures, and succeeds only after both describes return
`NOT_FOUND`. Never use legacy `gcloud alpha compute tpus delete` for v5p.

Record the stop only after teardown is verified, then run Close Out:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py stop $RUN \
   --status stopped --step STEP --reason REASON"
```

## Close Out a Completed Run

Uses auto-train's verified cleanup, final `run_registry` state, gsutil/TensorBoard, and
`exp.py` result comments.

On clean exit, auto-train must run `delete_tpu_xd.sh` until deletion is verified **before**
marking the registry complete. Treat a success message without absent node+queued-resource as
a bug; investigate immediately.

For every stopped/completed run:

1. Verify final step/status and resource teardown with `run_registry.py status --all` and GCP.
2. Sync the full TensorBoard directory once; do not routinely parse it when the loss cache exists.
3. Report final same-step/window gaps, cumulative trajectory, and whether prior extrapolation
   matched.
4. Replace the experiment class's running comment with one terse line containing speed, final
   step, and the main conclusion against its registered direct `compare_runs`; retain every
   decision-relevant direct baseline, e.g.:

```python
# ~0.280 steps/s; completed 13,500. dloss -0.0678 (-2.77%) vs MHA @13,400
```

## Hot Retrain

Commit/push every fix and restart the launcher with the new `CODE_COMMIT`; registry registration
updates the RUN hash. Reuse a RUN only when its checkpoint parameter tree remains compatible.
For a no-stop migration, restart only auto-train: it adopts the existing `train.py`, while the
new hash takes effect on the next relaunch. Changing the live training code itself requires a
training restart.

## Recover Preemption

Uses `auto_train_xd_maxtext.sh`, the RUN's registered commit, and `delete_tpu_xd.sh`.

- Preserve WAITING_FOR_RESOURCES/PROVISIONING queues; deleting resets queue position.
- In xd's v5p experience, maintenance warning + refused SSH is almost always preemption. Start
  reclaim early rather than waiting for recovery.
- `PREEMPTED|TERMINATED` plus queued-resource `SUSPENDED; stateInitiator=SERVICE` is terminal.
  Auto-train must release both resources through `delete_tpu_xd.sh`, recreate, reinstall, apply
  `CODE_COMMIT`, and resume the same RUN from its latest GCS checkpoint.
- A post-maintenance SSH timeout (`alive=unknown`) is not evidence that training is alive.

## TensorBoard Service

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

- Keep BAM `full` runs as capability-ceiling experiments; never silently replace them for speed.
- Report architectural parameter overhead in per-layer `W_Q = d_model^2` units, not raw counts;
  omit negligible biases/gates unless they matter to the comparison.
- A step-0 loss cannot prove zero-initialized BAM reads update: LR is zero and layer 0 has zero
  `M_in`; inspect layer 1+ after a later step.
- Use a new RUN after adding/removing conditional parameters; never resume an incompatible tree.
- For resume-only retention use `max_to_keep=2`, `keep_period=None` (normalize non-positive
  config values to `None`). `keep_period=1000` permanently accumulates large checkpoints.
