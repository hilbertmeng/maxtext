---
name: tpu-diagnostics
description: Run reproducible, read-only BAM checkpoint diagnostics on GCP TPUs. Use for runtime tensor health checks, randomized Pile eval probes, same-batch parameter ablations, diagnostic profiling, and artifact collection; use tpu-training for TPU lifecycle and training runs.
---

# TPU Diagnostics

Repo: `/home/xd/projects/maxtext`; TPU VM copy: `/home/lishengping/xd/projects/maxtext`.
Use `$tpu-ag` for VM commands and `$tpu-training` only for TPU lifecycle.
Source uses Git/HTTPS at the exact pushed commit; environment packages and AOT use GCS.

## Rules

- Keep this skill procedural. Put checkpoint-specific measurements and conclusions in
  `experiments/`.
- Run local BAM unit tests with `.claude/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh`;
  it uses the pinned CPU
  environment instead of whichever conda environment happens to be active.
- Use a spot non-pod `v6e-1` for inference probes; choose a larger TPU when memory requires it.
- Use only Codex-owned diagnostic resources whose names start with `xd-`; never borrow an idle
  TPU owned by another workflow.
- Acquire v6e candidates with this region policy and command:

  ```bash
  PRIMARY_ZONE=europe-west4-a
  BACKUP_ZONES=(us-central1-a us-east5-a)
  NAME_PREFIX=xd-v6e-1-bamdiag
  COMMIT=FULL_40_CHAR_HASH
  for ZONE in "$PRIMARY_ZONE" "${BACKUP_ZONES[@]}"; do
    NAME="$NAME_PREFIX-$ZONE"
    ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag \
      "/home/lishengping/xd/projects/start_standalone_tpu.sh \
       '$NAME' v6e-1 '$ZONE' install_xd_maxtext_jax081.sh '$COMMIT'"
  done
  ```

  Keep candidates until one reaches `FIRST_STEP`, then delete their exact resources. If it is
  preempted first, continue with the next queue. For parallel profile arms, request multiple TPUs
  in `PRIMARY_ZONE`.
- Restore the source checkpoint read-only; use `only_eval=True` and a local output dir.
- Add only necessary raw `sow` values to `attentions.py`; keep statistics in standalone runners.
- After the artifact workflow below completes, delete every diagnostic TPU; keep the reusable
  diagnostic runner.
- At closeout, audit every delay/failure as repeated or new. Root-fix recurring causes in a
  script or concise general skill rule; do not preserve incident-specific narrative here.

## Design a probe

Define the question and intervention scope first: layer, row/column side, token positions,
and downstream paths. Distinguish correlation, direct residual attribution, whole-network causal
necessity, and retraining benefit; report conclusions within the tested scope.

## Runtime health probe

Use `MaxText/bam_diagnostics.py` with `exp_class=BamLlama2MediumDiagnostics` and
`load_parameters_path=.../checkpoints/STEP/items`.

- Reuse the cohort registered in the relevant report for follow-up/model comparisons (including
  the existing 128-sequence Pile cohort). For a new probe, start with 32 fixed-seed,
  pre-batch-shuffled sequences; retain per-sequence measurements and hashes so aggregation can change.
- Slice diagnostic collections inside the jitted forward before device-to-host transfer.
- Run host statistics on the TPU VM. Use `BAM_DIAG_RAW_LAYERS` to retain selected layers or
  `BAM_DIAG_SAVE_RAW=0` when JSON is sufficient.
- For CPU-heavy host statistics, inspect worker CPU/memory availability and measure utilization
  and sample throughput early. Parallelize independent layer/side/sample tasks with bounded
  concurrency, coordinating numerical-library threads and memory to avoid oversubscription.
  Verify serial/parallel numerical agreement and measured speedup on a small workload before
  scaling to the full cohort; keep TPU inference and CPU statistics pipelined where practical.
- Check finiteness, adjacent-layer `M` continuity, write gates, `dM/M`, rank concentration,
  read-key scale, row/column read balance, BAM/MHA readout norm ratio, and route-logit
  delta/base RMS.

Raw capture must be gated by `cfg.bam_diagnostics and not self.is_initializing()`; otherwise
Flax adds diagnostic collections to the restore tree.

## Parameter ablation

Use `MaxText/bam_wr_ablation.py` with `exp_class=BamLlama2MediumReadAblation`.

- Disable raw diagnostics.
- Restore once, draw the cohort once, and reuse one compiled forward for all same-shape variants.
- Construct a new parameter pytree for each variant; leave restored parameters unchanged.
- Compare paired per-sequence loss deltas and verify identical sequence hashes.

For CPU-only Orbax inspection, override saved TPU sharding with single-device CPU sharding and
use partial restore for only the required leaves.

## Paired train-step profile

For compiler acquisition through `prepare_train_aot.py`, pass `--primary-zone "$PRIMARY_ZONE"`
and `--backup-zones "${BACKUP_ZONES[@]}"`. The script submits primary first and adds backups
after its 300-second default timeout, retaining primary preference.

Use `TrainStepProfile` (`xplane`, skip 10, trace steps 10–14, no checkpoints). Keep TPU type,
VM, commit, model/batch/data, and trace steps identical; prefer 6 layers for operator/scope
comparisons, then verify the winning combination with full layers.

Match both generic training-health and BAM-specific `sow` settings across throughput
comparisons, including historical controls. Record their resolved values with timings;
use an explicit speed-only class for all-health-OFF measurements while preserving the
formal RUN's health settings and schedule.

- Launch direct TPU smoke/profile runs with `scripts/run_train_smoke.sh EXP RUN [STEPS]` from the
  checked-out commit; do not reconstruct its dataset/output/checkpoint CLI by hand.
- Run paired matrices through `scripts/run_profile_matrix.sh TPU ZONE COMMIT LABEL EXP...`; it
  executes an immutable runner snapshot, performs tpu-ag/gcloud preflight, stages all AOT objects,
  derives the collector's trace count from the schedule, and closes each exact train process.
  Release raced backup resources with `scripts/release_profile_backups.sh MANIFEST TPU ZONE...`;
  it requires a verified target trace. Pull only primary traces with
  `scripts/pull_primary_xplanes.sh GCS_PREFIX LOCAL_DIR`.
- Keep profile TPU lifecycle separate from `auto-train`: create/install it standalone, launch
  paired arms directly, collect the complete profile set, then delete it. Keep `auto-train`
  detached from profile TPUs. For a large matrix, distribute arms across cheap spot `v6e-1`s;
  one shared control per TPU type is normally sufficient. Re-pair on one VM only for marginal or
  anomalous results. Keep every arm used for a cross-configuration conclusion on one TPU type
  (normally `v6e-1`); use the target training TPU only as a final confirmation.
- Keep every full-layer arm in a paired throughput matrix in one zone as well as one TPU type;
  if the zone changes, rerun the whole matrix rather than filling missing rows across zones.
- Precompile every sealed full-layer target configuration for its exact topology on a cheap TPU/CPU with
  `scripts/compile_aot_matrix.sh GCS_ROOT TARGET_TOPOLOGY STEPS EXP...`. Stage the sealed commit,
  environment and executables during installation, then load them with
  `scripts/run_train_smoke_compiled.sh`; require `Loaded compiled function!` and an actual first
  step. Model/batch shapes, total steps/learning-rate schedule, JAX/libtpu and compiler flags must
  match the target run.
- If a spot `v5p-16` remains `WAITING_FOR_RESOURCES` in `us-central1-a` for 5 minutes, also queue
  one in `europe-west4-b`. If EW4b wins, retain the UC1a queue and first verify the identical Pile
  config at the same commit and steps on both regions; if stable step/s differs, use UC1a timing.
  Otherwise keep either validated TPU and immediately stop/delete the other exact resource.
- For short `v5p` profiles, UC1a/EW4b/UE5a may be queued concurrently when acquisition latency
  matters; run only the winner and delete the exact losers after its required trace/step succeeds.
- Write XPlane on the TPU worker and upload it directly to a unique GCS prefix as soon as
  `*.xplane.pb` appears; never route profile bytes through `tpu-ag`. For a critical spot arm, race
  two zones and never use `us-east5-a` as its sole copy. The matrix runner and collector derive
  the required object count from the configured trace schedule and auto-detect the owning worker.
  After GCS verification, pull the
  artifacts directly to `/data0/xd/bam_diagnostics/` on the local workstation.
- Use the watcher as a `FIRST_STEP` hint, but declare failure only when the exact train process
  exits or its main thread fails; a background uploader traceback is not sufficient. Control
  lifecycle from the **actual train-log step**; after step 14, wait for the collector to verify
  the nonempty primary XPlane in GCS,
  then `SIGKILL` the exact no-checkpoint RUN and require `pgrep` empty before the next arm.
  Set the RUN length beyond the trace window (for example 100 steps); collector verification,
  rather than configured-step completion, ends it and keeps the TPU alive through artifact copy.
- Compare stable log speed and all-device XPlane step time; split read-key projection, gate,
  transform, M contraction, and routing scopes. Report theoretical cost in `W_Q` units.
- Inspect HLO/XPlane lowering, layout/copies, fusion type, kernel count, and whether conceptual
  broadcast/zero tensors materialize.
- Keep `tpu-ag` for orchestration and object verification only; store no profile artifacts and
  parse XPlane traces only on the local workstation.

## Artifacts

Upload artifacts from the TPU worker to a unique GCS prefix with `gsutil`, verify object count and
sizes, then `gsutil rsync` that prefix to `/data0/xd/bam_diagnostics/` and verify complete local
files before deleting the TPU. Do not route bytes through `tpu-ag` or rely on recursive SCP.

Record configuration full names, training and diagnostic commits (plus implementation branch/worktree),
checkpoint URI/actual step, cohort seed/hashes, runner path and invocation, overrides, timings,
results, and artifact paths under `experiments/`. Commit reusable diagnostic runners;
keep large per-sequence data outside the repo.
Name every configuration class in important result tables; update the canonical table in place
instead of appending overlapping snapshots.
