---
name: tpu-training
description: Manage preemptible TPU lifecycle and MaxText training on GCP for xd's BAM/Llama2Medium experiments. Use when creating, launching, monitoring, stopping, resuming, hot-retraining, or closing out TPU training runs and comparing same-step TensorBoard loss gaps.
---

# TPU Training

Run Codex locally. Run GCP lifecycle commands on `tpu-ag` through `$tpu-ag`'s persistent
SSH socket `/tmp/ssh-tpu-ag-xd.sock`. Use `$tpu-diagnostics` for checkpoint probes/ablations.

Defaults: repo `/home/xd/projects/maxtext` (`refactor-bam`); tpu-ag scripts
`/home/lishengping/xd/projects`; TPU VM repo `/home/lishengping/xd/projects/maxtext`;
project `newproject-1-451205`; zone `us-central1-a`; TPU `v5p-16`; output
`gs://newproject-1-llm_base_models_us-central1/log/`.

## Start Training

Uses `run_exp_xd.sh` → `auto_train_xd_maxtext.sh`, `run_registry.py`, a persistent code
overlay when needed, and the mandatory `watch_train_xd.sh` watcher.

1. Choose `EXP`, TPU `ID`, `MODE`, and direct experimental baselines in `COMPARE_RUNS`.
   Use `install+train` for a new/reprovisioned VM and `train` for an installed READY VM.
2. Before a parameter-tree change, use a new run name/GCS prefix. Check both dirty files and
   commits ahead of origin:

```bash
git status --short --branch
git diff --name-status origin/refactor-bam
```

3. For dirty/unpushed code, make a tar overlay on tpu-ag containing every runtime file and set
   `CODE_OVERLAY`; auto-train reapplies it after reprovision. `sync_to_vm.sh` alone omits local
   commits ahead of origin.
4. Launch on tpu-ag in tmux:

```bash
EXP=BamLlama2Medium ID=0 MODE=install+train
OVERLAY=/home/lishengping/xd/projects/maxtext_overlay.tar.gz
BASES=Llama2Medium
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "tmux new-session -d -s ${EXP}-TPU${ID}-xd \
   'env EXP=$EXP ID=$ID MODE=$MODE BRANCH=local CODE_OVERLAY=$OVERLAY \
   COMPARE_RUNS=$BASES bash /home/lishengping/xd/projects/run_exp_xd.sh'"
```

5. Verify registration and overlay persistence immediately:

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  '/home/lishengping/xd/projects/run_registry.py status'
# Inspect the registered auto_pid environment; CODE_OVERLAY and COMPARE_RUNS must be present.
```

6. **Always attach the watcher as part of launch. Launch is not handed off until it is
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

7. After 30–40 stable steps, record `~steps/s` tersely in the experiment class in `exp.py`.

## Monitor Training

Uses `run_registry.py status`, TPU train logs, TensorBoard/gsutil, and
`scripts/compare_train_loss.py`. Do not use timers, `_status.json`, or alert files.

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  '/home/lishengping/xd/projects/run_registry.py status'
```

Each `/home/lishengping/xd/projects/run_registry/<RUN>.json` contains run/TPU/launch data,
planned steps, report interval/window, cursor, and `compare_runs`. For every listed `BASE`,
report `gap = RUN loss - BASE loss` (negative favors RUN). Keep `compare_runs` to direct
controls; do not add transitive baselines.

At every due milestone:

1. Sync only the active RUN; reuse completed BASE logs already local.
2. Compare exact same steps plus the configured ±window; never substitute a nearby step.
3. Report cumulative history for all prior milestones and assess level plus trend (first/second
   derivative qualitatively), not a single gap.
4. Mark the cursor:

```bash
RUN=BamLlama2Medium BASE=Llama2Medium STEP=200
mkdir -p ~/tensorboard_logs/$RUN
~/google-cloud-sdk/bin/gsutil -m rsync -r \
  gs://newproject-1-llm_base_models_us-central1/log/summaries/train/$RUN/ \
  ~/tensorboard_logs/$RUN/
/home/xd/miniconda3/envs/tune/bin/python \
  .agents/skills/tpu-training/scripts/compare_train_loss.py \
  --experiment-dir ~/tensorboard_logs/$RUN --baseline-dir ~/tensorboard_logs/$BASE \
  --experiment-name "$RUN" --baseline-name "$BASE" --step "$STEP" --window-radius 25
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
  "/home/lishengping/xd/projects/run_registry.py mark-reported $RUN $STEP"
```

Normally evaluate early stopping at about 2,800/13,500 steps. Use
`MHA advantage = MHA loss - RUN loss`: an advantage below 0.08 and still shrinking rapidly
likely finishes below 0.05 and may stop; an advantage above 0.05 with curves becoming parallel
may merit continuing. Also stop a run clearly dominated by a prior failed configuration.

Between milestones, estimate sleep from `(next_ready_step-step)/steps_per_second`; overshooting
by up to 400 steps is fine. Stay silent. Do not repeatedly sync TensorBoard or reread this skill.

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
helper; do not duplicate raw delete commands:

```bash
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
2. Sync the full TensorBoard directory once.
3. Report final same-step/window gaps, cumulative trajectory, and whether prior extrapolation
   matched.
4. Replace the experiment class's running comment with one terse line containing speed, final
   step, and one key windowed gap, e.g.:

```python
# ~0.280 steps/s; completed 13,500. dloss -0.0678 (-2.77%) vs MHA @13,400
```

## Hot Retrain

Uses local `retrain_xd.sh` → `sync_to_vm.sh` → TPU-VM `retrain_on_vm.sh`. Use it for code
iteration on an installed TPU without committing/pushing:

```bash
bash /home/xd/projects/xd_tpu_scripts/retrain_xd.sh
```

`sync_to_vm.sh` transfers modified/untracked/deleted files via `git ls-files`. The VM relaunch
must kill the prior `train.py`, kill holders of `/dev/vfio/0`, remove
`/tmp/libtpu_lockfile`, and recreate writable `/tmp/tpu_logs`.

If validated files were later committed/pushed, an old VM may still show identical dirty files
and reject `git pull`. Stop only auto-train, verify those files equal `origin/refactor-bam`,
stash them, then `git pull --ff-only`; do not stop the live training Python process.

## Recover Preemption

Uses `auto_train_xd_maxtext.sh`, the persistent overlay, and `delete_tpu_xd.sh`.

- Preserve WAITING_FOR_RESOURCES/PROVISIONING queues; deleting resets queue position.
- In xd's v5p experience, maintenance warning + refused SSH is almost always preemption. Start
  reclaim early rather than waiting for recovery.
- `PREEMPTED|TERMINATED` plus queued-resource `SUSPENDED; stateInitiator=SERVICE` is terminal.
  Auto-train must release both resources through `delete_tpu_xd.sh`, recreate, reinstall, apply
  `CODE_OVERLAY`, and resume the same RUN from its latest GCS checkpoint.
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
- A step-0 loss cannot prove zero-initialized BAM reads update: LR is zero and layer 0 has zero
  `M_in`; inspect layer 1+ after a later step.
- Use a new RUN after adding/removing conditional parameters; never resume an incompatible tree.
- For resume-only retention use `max_to_keep=2`, `keep_period=None` (normalize non-positive
  config values to `None`). `keep_period=1000` permanently accumulates large checkpoints.
