---
name: tpu-diagnostics
description: Run reproducible, read-only BAM checkpoint diagnostics on GCP TPUs. Use for runtime tensor health checks, randomized Pile eval probes, same-batch parameter ablations, diagnostic profiling, and artifact collection; use tpu-training for TPU lifecycle and training runs.
---

# TPU Diagnostics

Repo: `/home/xd/projects/maxtext`; TPU VM copy: `/home/lishengping/xd/projects/maxtext`.
Use `$tpu-ag` for VM commands and `$tpu-training` only for TPU lifecycle.

## Rules

- Keep this skill procedural. Put checkpoint-specific measurements and conclusions in
  `experiments/`, never here.
- Use a spot non-pod `v6e-1` for inference-only probes unless memory requires more.
- Reclaim `v6e-1` only in known viable zones, in order: `us-central1-a`,
  `europe-west4-a`, `europe-west4-b`, `us-east5-a`; follow `$tpu-training` lifecycle rules.
- Never mutate or save over the source checkpoint. Use `only_eval=True` and a local output dir.
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

- Run paired arms on matched TPU types and VM/software environments. If a TPU must remain
  allocated, avoid launch paths whose clean-completion policy deletes it.
- The watcher proves `FIRST_STEP`/errors only. Stop by parsing the **actual train-log step**; at
  step 30–40, `SIGKILL` the exact no-checkpoint profile RUN on all workers, then require `pgrep`
  empty before launching the next arm. Never time stopping from watcher delivery.
- Compare stable log speed and all-device XPlane step time; split read-key projection, gate,
  transform, M contraction, and routing scopes. Report theoretical cost in `W_Q` units.
- Inspect HLO/XPlane lowering, layout/copies, fusion type, and whether conceptual broadcast/zero
  tensors materialize. A single JAX expression does not imply one device kernel.

## Artifacts

Record checkpoint URI, step, code state, cohort seed/hashes, overrides, timings, results, and
artifact paths in a new file under `experiments/`. Keep large raw arrays outside the repo.
