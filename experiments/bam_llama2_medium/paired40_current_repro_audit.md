# Paired40 Rank2 current-code reproduction audit

## Scope and status

Historical control: `BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl`,
runtime `0038e21`. Current probe: `BamMediumPaired40Rank2CurrentControlRepro`,
runtime `28aefca`, branch `codex/local-read-gram`, worktree
`/data0/xd/local-read-gram`. Both use scan+AOT and the full 13,500-step schedule.
The probe explicitly sets `wd_mults=[]` to reproduce the historical AOT all-decay
optimizer. This is a 100-step reproduction check, not a new baseline result.

Reproduction is established to printed-loss precision through100 steps after
restoring the historical packed module name. The proposed
`BamMediumPaired40Rank2CFp32` now inherits that initialization path; its old AOT
artifact is superseded and must be recompiled.

## Evidence collected on 2026-09-11

TB `learning/loss`, exact steps, latest event wins; these are not window averages:

| Step | Probe | Historical | Probe − historical |
|---:|---:|---:|---:|
| 0 | 10.84615517 | 10.846155 | +0.00000017 |
| 10 | 10.30989933 | 10.309916 | −0.00001667 |
| 20 | 9.95136547 | 9.951451 | −0.00008553 |
| 30 | 9.38162518 | 9.384468 | −0.00284282 |
| 40 | 8.66864491 | 8.664366 | +0.00427891 |
| 50 | 8.07022381 | 8.061051 | +0.00917281 |
| 60 | 7.657539 | 7.686422 | −0.028883 |
| 70 | 7.391550 | 7.391161 | +0.000389 |
| 80 | 7.115118 | 7.099284 | +0.015834 |
| 90 | 6.797756 | 6.794589 | +0.003167 |
| 100 | 6.610661 | 6.617669 | −0.007008 |

The EW4b and UE5a recoveries from checkpoint26 emitted identical step30 loss.
This checks that one replayed point, not entire resume equivalence. After completing
the100-step check, the probe was paused at committed124. The prior EW4b lease
reached50 without preserving all those steps in a committed checkpoint.

## Configuration-name audit

Executing each runtime's `exp.py` and inspecting the complete inherited class
attributes identified these renames; current setup consumption was checked:

- `bam_local_qk_rank` → `bam_local_q_rank`: 2; K falls back to Q.
- `bam_local_qk_rank_routing` → `bam_local_q_rank_routing`: legacy for Q/K.
- `bam_local_qk_pre_rms_bias` → `bam_local_q_pre_rms_bias`: True for Q/K.
- `bam_local_qk_second_implementation` → `bam_local_second_implementation`: mul_reduce.
- `bam_record_local_qk_routing_metrics` → `bam_record_local_routing_metrics`: True.
- Optional per-arm key scales fall back to the unchanged `bam_read_key_scale`.

No inactive old attribute was found in these settings. This is not yet a full
resolved-pyconfig/parameter-tree equivalence proof.

## Separate candidate causes

1. Packed module name changed from `W_local_qk_packed` to `W_local_packed`.
   Flax folds module scope into initialization RNG. A same-seed isolated probe
   confirms changed random parameters (unequal fraction1.0); zero-initialized
   segments remain zero. Full-model leaf-by-leaf initialization comparison remains
   necessary to quantify this candidate and identify other RNG changes.
2. Historical head mix splits row/col before RMS over `(N,R)`; current code RMSes
   `[B,T,N,side,R]` over `(N,R)` before splitting. On CPU JAX0.8.1, tested fp32
   and bf16 forward/VJP are bitwise equal. TPU layout/fusion in the full train
   graph may differ, so this candidate remains open.

Keep initialization and normalization layout separate in a 2×2 verification;
do not treat a CPU equality result as a TPU trajectory result.

## Reproduction artifacts

Primitive runner: `experiments/bam_llama2_medium/probe_paired40_repro_primitives.py`
(introduced in `c5166aa`). Run from the implementation worktree:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=MaxText /data0/xd/conda/envs/maxtext-cpu/bin/python \
  experiments/bam_llama2_medium/probe_paired40_repro_primitives.py
```

Probe TB: `/data0/xd/tensorboard_logs/BamMediumPaired40Rank2CurrentControlRepro`.
Export: `/data0/xd/bam_diagnostics/paired40-repro-0911-loss.txt`.
Historical raw loss cache: tpu-ag
`/home/lishengping/xd/projects/run_registry/loss_cache/BamLlama2MediumV2C256Paired40LocalQKRank2CurrentControl.json`.
Final checkpoint124: `gs://newproject-1-llm_projects_us-east5/log/BamMediumPaired40Rank2CurrentControlRepro/checkpoints/124/`.

Next isolated control: `BamMediumPaired40Rank2HistoricalInitRepro`, runtime `ecae35c`.
Only the packed module name is restored to `W_local_qk_packed`; current normalization
layout is retained. This intervention tests the renamed RNG path, not a claim that
all historical parameter initializations have already been restored.

## Isolated initialization result

`BamMediumPaired40Rank2HistoricalInitRepro` ran on the retained UE5a v5p-16.
Raw gaps versus historical CurrentControl:

```text
step  0 10 20 30       40 50 60 70 80 90 100
gap   0  0  0  0 +.000001  0  0  0  0  0   0
```

Thus the renamed packed-module RNG path explains essentially all observed
0–100 divergence in this reproduction. The current merged RMS layout is retained;
there is no evidence here of a material trajectory penalty from that layout.
This does not establish full-horizon bitwise equivalence on all inputs/models.
The pinned CPU BAM suite also passed all53 tests (246.8s).
