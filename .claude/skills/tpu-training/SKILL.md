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

Uses `run_exp_xd.sh` → `auto_train_xd_maxtext.sh` and `run_registry.py`.

1. Choose `EXP`, TPU `ID`, `MODE`, and direct experimental baselines in `COMPARE_RUNS`.
   Use `install+train` for a new/reprovisioned VM and `train` for an installed READY VM.
2. Before a parameter-tree change, use a new run name/GCS prefix. Commit the prepared runtime
   code, push it to `origin/refactor-bam`, and use its full hash. Commit/push first-step fixes and
   update the RUN hash before relaunch.

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

5. Gate launch with the registry's one-shot waiter. It polls the worker log internally and exits
   on `FIRST_STEP:`, confirmed failure, or timeout. Launch succeeds only after this command
   returns `FIRST_STEP:`.

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py wait-step '$EXP' 0"
```

6. Use the same one-shot gate for the 30–40-step speed check. Compare `~steps/s` with direct
   `compare_runs` and the expected architectural delta, then record it tersely in the `exp.py`
   class. Immediately report and investigate a material unexplained speed deviation; mark the
   class comment `!?` or `!!` until resolved.

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py wait-step '$EXP' 40"
```

## Monitor Training

Use `run_registry.py status` for liveness and `loss-report` for loss. Reserve TensorBoard sync
for run closeout.

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  '/home/lishengping/xd/projects/run_registry.py status'
```

At each wake, compare observed step gain over elapsed time with logged steps/s. A large shortfall
plus stale per-worker train-log mtimes means a hang even if TPU is READY and processes are alive;
auto-train owns this machine-level liveness check. It arms after three progress samples, waits
`max(600s, 20 / steps_per_second)` (or per-RUN `STALE_TIMEOUT_SECONDS`), requires two all-worker
SSH confirmations with valid speed, then recreates. Codex verifies watchdog health and owns
loss/trend decisions.

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
`step`, `run`, `base`, `gap`, and `r200` as horizontal rows; split long sequences into additional
horizontal row blocks. Here
`r200 = (abs(gap[s]) - abs(gap[s-200])) / abs(gap[s-200])`: negative means the gap
magnitude shrank from the preceding window, positive means it grew. Summarize the current gap
level with the mean and range of the latest 5–8 reported points; use recent `r200` values for
direction.

At every due milestone:

1. Run one shared `status`, then `loss-report` once per due RUN.
2. Report the cumulative horizontal rows; judge stability with `r200`.
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
may merit continuing. Train a configuration with a credible loss or speed gain longer—possibly
to completion—to verify that the gain persists. A configuration still unlikely to beat its direct
baseline may stop at 2,800. Also stop a run clearly dominated by a prior failed configuration.

For multiple runs, use one shared wake-up and batch-check all runs; use per-run wake-ups only
for anomalies or imminent completion/decisions. Independently, lengthen the shared sleep for
stable runs—normally enough to collect ~5 report intervals. Estimate from steps/s; modest
overshoot is fine. Stay silent between wakes.

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
   --status stopped --step STEP --reason REASON"
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
3. Replace the experiment class's running comment with one terse line containing speed, final
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
training restart. Record `OLD_STEP` before relaunch, then bind the launch gate to a strictly newer
step:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py wait-step '$RUN' 0 --after-step '$OLD_STEP'"
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py wait-step '$RUN' '$((OLD_STEP + 40))' \
   --after-step '$OLD_STEP'"
```

## Create Standalone v6e-1

For an unregistered diagnostic TPU, call tpu-ag's creator directly; the installer is a basename
resolved from the working directory:

```bash
NAME=xd-v6e-1-bamdiag ZONE=us-east5-a
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "cd /home/lishengping/xd/projects && mkdir -p logs && \
   nohup python3 /home/lishengping/lsp/create_tpu.py \
   --project newproject-1-451205 --tpu_name '$NAME' --type v6e-1 --zone '$ZONE' -p \
   -inf install_xd_maxtext_jax081.sh \
   > 'logs/${NAME}-create.log' 2>&1 < /dev/null &"
```

`-p` creates a best-effort spot queued resource; the creator waits for `READY`, then installs
the environment. Inspect `logs/${NAME}-create.log` on tpu-ag.

## Recover Preemption

Uses `auto_train_xd_maxtext.sh`, the RUN's registered commit, and `delete_tpu_xd.sh`.

- For spot `v6e-1`, query
  `serviceusage.googleapis.com/v1beta1/projects/$NUM/services/tpu.googleapis.com/consumerQuotaMetrics?view=FULL`
  (`effectiveLimit>0` or override `-1`; missing means zero), then intersect with `gcloud alpha compute tpus
  accelerator-types list --zone=ZONE --filter=type=v6e-1`; treat quota and current capacity as
  separate conditions. Prefer proven zones `us-central1-a`, `europe-west4-a`, then `us-east5-a`.
- Preserve WAITING_FOR_RESOURCES/PROVISIONING queues; deleting resets queue position.
- In xd's v5p experience, maintenance warning + refused SSH is almost always preemption. Start
  reclaim immediately.
- `PREEMPTED|TERMINATED` plus queued-resource `SUSPENDED; stateInitiator=SERVICE` is terminal.
  Auto-train must release both resources through `delete_tpu_xd.sh`, recreate, reinstall, apply
  `CODE_COMMIT`, and resume the same RUN from its latest GCS checkpoint.
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
