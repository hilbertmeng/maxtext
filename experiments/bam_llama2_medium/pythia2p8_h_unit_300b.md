# Unit H: 300B-token preparation

Configuration: `BamPythia2p8BHUnit300B`.
Implementation worktree `/home/xd/projects/maxtext`, branch `refactor-bam`.
This is configuration preparation only; no RUN, TPU or AOT job has been created.
The durable orchestration settings are in
[the launch/report plan](pythia2p8_h_unit_300b_plan.json).

## Model identity

Direct architecture parent:
`BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne`.
Retain its D2560/L32/N32, M80x64, all R640 projections, independent V/O addresses,
negative attention diagonal, full 100-dimensional QKNorm before RoPE20, serial
GELU/LayerNorm and single read-only M-shift .5. Keep ordinary/private MLP widths
7808/2944, terminal 9984, scan 31 plus one explicit tail, Pallas v8t and minimal
remat. Keep the learned V/local O common scale initialized to 1, raw private-MLP
content, fixed embedding divisor .0020825863 and normalized attention-write
content. No dynamic-O gamma, extra Q/K read scalar or private-MLP bias.

Embedding/BAM kernels retain .006 initialization; standard Transformer kernels
retain Pythia initialization. Semantic zeros, gate biases and norm scales remain
unchanged. Total parameters remain 2775190912, exactly equal to the parent;
MLP budgets and inference M-cache are unchanged.

The sole numerical model change is the attention logit multiplier:
`bam_qk_logit_scale_dim=100` selects 1/sqrt(100)=0.1 instead of 1/sqrt(80).
The V/output `head_dim` remains 80; concatenated Q/K width remains 100.
The multiplier is applied after full-head QKNorm, once, in both native JAX
and Pallas paths. Legacy classes keep `None`, preserving their existing scale.
The existing score-health observer uses the same denominator when enabled.

## Long-training plan

Restore the old long RUN's training plan, rather than overriding only `steps`:

| Setting | Value |
|---|---|
| Topology | v5p-256: 128 JAX devices, 32 hosts |
| Global batch / sequence length | 1024 / 2048; accumulation 1 |
| Effective per-device batch | 8 |
| Optimizer updates / LR schedule | 143000 / 143000 |
| Training tokens | 299892736000 (about 300B) |
| Peak LR / warmup | 1.6e-4 / 1430 updates (1%) |
| Decay | Existing affine cosine to 0.1 times peak, exactly as old BAM |
| Optimizer | AdamPax, betas .9/.95, epsilon 1e-8, WD .1; inherited clipping |
| Checkpoints | Every 1000; latest two; keep_period 0; force final |
| Permanent checkpoints | 3000, 13000, ..., 143000 |
| TensorBoard / evaluation | Rotate every 1000; periodic evaluation disabled |
| Health | Generic training ON; internal NN and BAM activation health OFF |

This is a fresh-initialization experiment, not continuation of the 28k short
schedule. Preserve the actual full 143k schedule on any later restart.

FSDP128 audit: 22968832 parameters/device versus ideal 21681179; overhead
5.93904%, caused by 1297792 intentionally replicated small parameters.
Large matrices are correctly sharded and their sharded dimensions divide 128.
Use `sharding_tolerance=.06`, rather than the short-run .05 guard. This is a
validation threshold, with no change to model computation or initialization.
The CPU shape audit shortens sequence length and disables fused kernels only
for metadata inspection; it does not claim target-TPU memory/AOT verification.

## Baselines, coordinates and reporting

Keep exactly these four comparisons:

- `Pythia_2.8B_2022`: common reference for all gain multipliers.
- `Pythia_6.9B_2022_Decay`.
- `PileMUDDPythia3BPlusOcdbtv6e`: historical parallel-MLP Mudd, WD .01.
- `BamPythia2p8BSotaIndependentR768Pallas`: old 300B-plan BAM, runtime 1ac1183,
  paused at checkpoint 37042. Never extend its observed curve past its data.

Use completed-update comparison coordinates. Both the new RUN and the old
MaxText BAM baseline need offset +1; external Pythia/PaxML histories stay at 0.
Applying the offset only to the new RUN would recreate the old one-step error
in the new-versus-old BAM comparison. Keep native caches/checkpoint labels
unchanged. Register all three gain triples from the JSON plan at launch;
use the shared registry calculations, not a separate gain formula.

Retain the old normal long-run cadence: reports at 8k, 13k, 18k, ... (every 5k).
Check the first permanent 3k checkpoint separately. Routine reports show the
latest 5k, excluding the initial <2k transient, and retain complete raw caches
for longer-window investigation. Permanent-checkpoint and midpoint reports
alternate. Report final 143k without waiting for nonexistent 143025 samples.
Reporting lives in the registry/group, not in model math; the plan is prepared
but is not registered as an active RUN at this stage.

## Checkpoint rescue audit

Current main includes `CoordinatedCheckpointSaver` in the actual training loop,
with `checkpoint_save_timeout_seconds=1200` passed to Orbax AsyncOptions.
The ordinary persistent-manager path is active for this configuration; emergency
checkpointing is disabled. Previous-save and preemption-save failures coordinate
across ranks, attempt one fresh-directory save of the live state/cursor and exit
resumably after verification. Persistent failure does not report success.

The deployed tpu-ag auto-train, `checkpoint_rescue.py` and incomplete-checkpoint
repair helper match canonical `/home/xd/projects/xd_tpu_scripts` SHA256 hashes.
The outer pending-save trigger is 1300 seconds; rescue has a 600-second upper
bound, after which worker exit must be verified before repair/relaunch.
These are enabled defaults, not flags waiting for an additional patch.

The production recovery module SHA256 is
`b2460321665a54d9252241051df372dce0c525bdc5ce51696f3ee82f27ed9d64`, identical
to the previously validated two-rank real-Orbax/GCS production probe. Those probes
used shortened 1-second local and 10-second GCS timeouts and checked transient,
early-signal, persistent-storage and cursor failures, including restored state
and cursor consistency. Evidence:
`/data0/xd/bam_diagnostics/checkpoint-recovery-implementation-20261031/validation-audit.json`.
The current preparation reruns protocol, retention, rotation and outer-controller
tests; it does not launch a new fault-injection job or claim that this unstarted
143k RUN has already exercised a timeout in production.

Preparation artifacts:
`/data0/xd/bam_diagnostics/h-unit-300b-prep-20261101/`.

## Checks completed

Effective configuration was resolved with the v5p-256 topology hint: 128 target
devices, batch 8/device, global batch 1024, T2048, checkpointing/async saves ON
and emergency checkpointing OFF. No backend compilation was requested. Actual
LR values at warmup, middle, late and final steps exactly match the old BAM
long-plan class. Configuration differences from H are limited to the requested
training plan, baselines, sharding guard and Q/K logit denominator.

The new tests verify the complete parameter count/sharding budget, all intended
configuration overrides and report offsets, validation of the scale option,
bitwise-identical initial parameter trees, and actual attention-boundary scaling
in native JAX and fused Pallas. The full scanned-model native/Pallas forward and
all-parameter gradient check has worst relative leaf error 1.949e-6. The 47 shared
BAM regressions, checkpoint recovery/retention and TB rotation tests, and 22
outer-controller/step-offset tests pass.

The preparation artifact directory retains the resolved configuration, exact
class diff, partition specs, LR samples, rescue audit, and individual test logs.
TPU memory fit, actual loaded AOT, first-step execution and throughput remain
launch-time checks; they are not claimed by this CPU-only preparation.
