# Unit H with a shared local Q/K scale

Requested 2026-11-01. Implementation worktree `/home/xd/projects/maxtext`, branch
`refactor-bam`. This task owns only the new RUN below and its training TPU;
it borrows the retained FLEX compiler without lifecycle ownership.

| Role | Full configuration / RUN |
|---|---|
| Experiment | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneCommonQKScale` |
| Direct parent | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne` |
| Gain reference | `Pythia2p8BShortTrain` |

The direct parent is unit-initialized H without dynamic-O gamma. It ended at
checkpoint 21549; complete native comparisons extend through 21.5k. Do not
synthesize later parent values or silently change the gain reference to Mudd.
The new RUN starts from scratch with the same 28000-step LR plan.

## Change and prior

Each layer gets one directly learned signed `common_qk_read_scale`, initially 1,
shared by the complete local Q/K reads (static plus dynamic). Scale these reads
before concatenating the 20-dimensional standard RoPE arm and applying full-head
QKNorm over all 100 coordinates. Preserve the independent learned V/local O
scale, initialized to 1. The standard Q/K projection, RoPE, QKNorm parameters,
read gates, M-shift, writers and initialization remain unchanged.

In the native path, multiply local Q/K before concatenation. In the Pallas
path, the existing QKNorm epilogue uses `s^2 * local_sumsq + standard_sumsq`
in its denominator and `s * local_norm_scale` in its numerator. This implements
pre-normalization scaling without editing the M-read kernel, adding VMEM buffers
or materializing a separate full-head scaling pass. The scale includes both
static and dynamic local reads; it is not an additional dynamic-gate multiplier.

Add 32 parameters (0.00000488 W_Q total, about 0.00000115% of the model), exempt
from MLP repayment. Total parameters: 2775190944, parent +32. Keep ordinary/private
GELU widths 7808/2944, tail 9984 and M-cache unchanged. Log signed scale values
using existing generic parameter health: 32 additional signed scalars, without
new activation/gate dumps. Existing scale WD exemption applies.

Prior at the common 21.5k horizon: MHA-reference gain ratio versus parent 1.01,
range 0.99–1.03. Throughput within 1% of parent (startup 0.316382 step/s,
late about 0.315 step/s), with matched generic health ON, internal/BAM health OFF
and minimal remat. The hypothesis is easier learning of the M/RoPE addressing
balance, not a large expansion in representational capacity.

## Launch plan and infrastructure audit

Use spot v5p-128, global batch 1024 (16/device), T2048, scan 31 plus explicit tail,
28000 total steps, checkpoint period 1000, fixed checkpoints every 2k, latest two
rolling checkpoints and GCS TB rotation 1000. Primary `us-east5-a` (UE5a): recent
comparable H leases lasted 14h 20m and 4h 58m without preemption; backup EW4b.
Training identity: `xd-v5p-128-310496-maxtext`, logical ID 310496. Compilation:
retained `llm-jax-v6e-1-0` in `europe-west4-a`, worker 0, verified FLEX_START,
guaranteed and idle. Do not delete the compiler.

Audit base: parent runtime `6c5dbfe9df446b1c6f6421616f7af97156c7f9a5` against
main `95183d39`. New training/checkpoint infrastructure is retained: Orbax save
timeout 1200 seconds and coordinated in-process rescue; the deployed outer
controller uses a 1300-second trigger and 600-second rescue budget. Canonical
and deployed auto-train, rescue, compiler wrapper and AOT scripts have matching
SHA256 hashes. The default memory-MLP bias option is False for this RUN. Optimizer,
AOT entrypoint and summary-writer implementations have no additional parent-to-main
changes; the rotating writer/begin-step path is retained. The model lineage is
unchanged apart from the explicit Q/K scalar.

Artifacts: `/data0/xd/bam_diagnostics/h-unit-qk-scale-20261101/` (infrastructure
audit, CPU gates, AOT state/manifest, registry and startup evidence). Reports use
native 500±25 windows, every 2k steps, with cumulative gap/gain plots starting at
2k. Use the direct parent and original MHA for the gain triple. No automatic
16k or 21.5k stop is installed; those are potential reviews, not schedule changes.

## Validation

Runtime `bcdf29931b5949a0244fd822a5431d585c088e6e`, exact v5p-128/s28000
AOT and manifest verified. The 47 shared BAM regressions, five summary-writer
tests and 15 checkpoint-recovery tests pass. Targeted checks cover the full
parameter budget, signed scalar health and WD exemption, initial equivalence
to the parent (including every old parameter gradient), configuration guards,
and native/Pallas logits and gradients with zero, fractional and negative Q/K
scales. The FP32 worst relative leaf-gradient error is 6.124e-6.

BF16 global gradient L2 errors against FP32 are 1.483% native and 2.421% Pallas.
The original native-relative bound rejected the Pallas value. A matched control
that removes only the Q/K scalar, preserving every common parameter, sample and
target, gives 1.507% native and 2.542% Pallas. The new path does not worsen the
parent's fused-normalization numerical error. Retain the 5% absolute bound and
compare against the matched parent's error as well as the native path; no
runtime code, precision or compiled executable changed during this check.

## Startup verified

All 16 workers run the sealed runtime `bcdf29931b5949a0244fd822a5431d585c088e6e`.
`Loaded compiled function!`, actual step 0 and finite subsequent losses are
verified. Checkpoint 0 has its GCS commit marker and data cursor: seed 9876,
global batch 1024, 16 infeed hosts, checkpoint step 0. Each update contains
2097152 token weights (1024 x 2048); on 64 v5p devices, the resolved per-device
batch is 16. The early `[EXP] per_device_batch_size: 1.0` line is printed before
the target-global-batch resolution and is not the effective training batch.

The 20–39 window averages 0.3164245 step/s (harmonic mean of all 20 logged
samples, including the slower step 28), versus the parent's 0.316382: +0.0134%,
effectively tied. Both use generic training health ON, internal/BAM health OFF
and minimal remat; these are separate leases, not a paired hardware benchmark.
There is no observed speed regression. Startup proof, all-worker hashes,
checkpoint cursor and timing samples are retained in the artifact directory.

Registry group `pythia28-h-unit-qk-scale-1101` owns this RUN only, with 2k public
cadence and native 500±25 windows. The gain triple is experiment/direct
parent/original MHA. The 4k, 8k, 16k and 21.5k reviews do not automatically stop
the run. No existing group's cursor or baseline training state was changed.
