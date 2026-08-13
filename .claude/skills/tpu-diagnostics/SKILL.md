---
name: tpu-diagnostics
description: Run reproducible, read-only BAM checkpoint diagnostics on GCP TPUs. Use for runtime tensor health checks, randomized Pile eval probes, same-batch parameter ablations, diagnostic profiling, and artifact collection; use tpu-training for TPU lifecycle and training runs.
---

# TPU Diagnostics

Repo: `/home/xd/projects/maxtext`; TPU VM copy: `/home/lishengping/xd/projects/maxtext`.
Use `$tpu-ag` for VM commands and `$tpu-training` only for TPU lifecycle.

## Rules

- Keep this skill procedural. Put checkpoint-specific measurements and conclusions in
  `experiments/`.
- Use a spot non-pod `v6e-1` for inference probes; choose a larger TPU when memory requires it.
- Queue `v6e-1` concurrently in `us-central1-a`, `europe-west4-a`, and `us-east5-a`;
  keep the first READY TPU and immediately stop creators and delete the exact remaining resources.
- Create it with `$tpu-training`'s **Create Standalone v6e-1** command.
- Restore the source checkpoint read-only; use `only_eval=True` and a local output dir.
- Add only necessary raw `sow` values to `attentions.py`; keep statistics in standalone runners.

## Runtime health probe

Use `MaxText/bam_diagnostics.py` with `exp_class=BamLlama2MediumDiagnostics` and
`load_parameters_path=.../checkpoints/STEP/items`.

- Prefer one fixed-seed, pre-batch-shuffled cohort of 32 sequences; retain all 32 and report
  per-sequence distributions plus hashes.
- Slice diagnostic collections inside the jitted forward before device-to-host transfer.
- Run host statistics on the TPU VM. Use `BAM_DIAG_RAW_LAYERS` to retain selected layers or
  `BAM_DIAG_SAVE_RAW=0` when JSON is sufficient.
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

Use `TrainStepProfile` (`xplane`, skip 10, trace steps 10–14, no checkpoints). Keep TPU type,
VM, commit, model/batch/data, and trace steps identical; prefer 6 layers for operator/scope
comparisons, then verify the winning combination with full layers.

- Keep profile TPU lifecycle separate from `auto-train`: create/install it standalone, launch
  paired arms directly, collect the complete profile set, then delete it. Keep `auto-train`
  detached from profile TPUs. For a large matrix, distribute arms across cheap spot `v6e-1`s;
  one shared control per TPU type is normally sufficient. Re-pair on one VM only for marginal or
  anomalous results. Keep every arm used for a cross-configuration conclusion on one TPU type
  (normally `v6e-1`); use the target training TPU only as a final confirmation.
- Write XPlane locally and copy it to `tpu-ag` as soon as `*.xplane.pb` appears. For a critical
  spot arm, race two zones and never use `us-east5-a` as its sole copy. Also record an insurance
  trace at steps 2–6 and the primary trace at 10–14 with
  `skip_first_n_steps_for_profiler=2 profile_periodically_period=8 profiler_steps=5`; analyze
  `step_10`, using `step_2` only if preempted first. Before launch, run
  `scripts/collect_xplane.sh TPU ZONE REMOTE_PROFILE_DIR DEST_DIR PROJECT 2` on `tpu-ag`.
  For a pod, append `WORKER` and run one collector per worker into separate destinations.
- Use the watcher as a `FIRST_STEP`/error gate. Control lifecycle from the **actual train-log
  step**; after step 14, wait for the collector to verify the nonempty primary XPlane on `tpu-ag`,
  then `SIGKILL` the exact no-checkpoint RUN and require `pgrep` empty before the next arm.
- Compare stable log speed and all-device XPlane step time; split read-key projection, gate,
  transform, M contraction, and routing scopes. Report theoretical cost in `W_Q` units.
- Inspect HLO/XPlane lowering, layout/copies, fusion type, kernel count, and whether conceptual
  broadcast/zero tensors materialize.
- Keep `tpu-ag` for orchestration/artifact storage only; parse XPlane traces on the local
  workstation, never on `tpu-ag`.

## Artifacts

Record checkpoint URI, step, code state, cohort seed/hashes, overrides, timings, results, and
artifact paths in a new file under `experiments/`. Keep large raw arrays outside the repo.
