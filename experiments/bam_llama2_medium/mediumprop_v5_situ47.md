# MediumProp V5 ordinary-MLP SiTU 4/7

RUN: `BamMediumPropV5SiTU47`; branch `codex/mediumprop-v5-situ47`;
worktree `/home/xd/.codex/worktrees/medium-v5-situ47/maxtext`.
Training ownership: `xd-v5p-16-311013-maxtext`, UE5a primary / EW4b and UC1a backups.
On 2026-11-12 the user authorized UC1a for Medium only; XL remains excluded.
Borrow retained FLEX `llm-jax-v6e-1-0` only for AOT.

## Change and controls

Derive from current-main `BamMediumPropV5`, not XL and not Pythia.
Only ordinary MLPs, including the terminal ordinary MLP, replace SwiGLU by
`(4*tanh(u/4)*sigmoid(u)) * (7*tanh(v/7))`.
Private memory-writing MLPs retain SwiGLU. No parameters, matrix projections,
Pallas interfaces, initialization or optimizer rules change.
Ordinary/private/terminal widths remain 2510/1128/3904; verified total 432,112,228 parameters.
The existing Pallas read/write kernels are reused: activation is in XLA's MLP path.

The trained direct V5 control is the pre-rename RUN
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormQKScaleTruePile`
(runtime `f8d07cd`, completed 13,500, ~.677145 step/s).
Use `BamMHAMediumPropC256TruePile` only as the native gain-multiplier reference.
Do not plot a separate MHA loss-gap curve.

## Budget, runtime and workflow audit

From-scratch 13,500 steps and LR endpoint 13,500; T4096, per-device batch 16 on
v5p-16 (8 devices), global batch 128. Total 7,077,888,000 tokens, unchanged
from the same-size V5 control. Original Prop initialization/AdamPax recipe;
full remat and single-layer scan. Generic training health ON, internal
activation/BAM sow statistics OFF; TB rotation 500, checkpoint 200,
resume-only retention 2 and forced final checkpoint.

Forked latest main `b855d5f0`; this already includes the current logging,
checkpoint, optimizer/AOT, parameter-budget and common V5 infrastructure.
Copied only the validated SiTU activation from XL runtime `bd181bc`; XL
model/config and its historical infrastructure were not copied.
CPU gates cover analytic derivatives, fused/unfused MLP full gradients,
formal parameter counts/config, scanned ordinary/private isolation and
initial parameter identity, plus summary-writer rotation. Run the current
full BAM V5 regression suite as well. Seal exact effective configuration
before AOT; launch only after all preparation gates and require FIRST_STEP.

## Prior and monitoring

Prior: final MHA-referenced gain multiplier versus V5 .995, range .985--1.005;
speed within -2% of matched V5. Medium V5 has much less sustained loss-spike
damage to recover than XL, so a net quality gain is not the central prior.
Evaluate actual loss-spike amplitude/duration together with normal-phase CE.
No inference-cache change; parameter delta 0 W_Q.

Fixed public cadence 2,000; reviews 4,000/8,000 give recommendations only,
and require user confirmation before stopping. Otherwise finish 13,500.
Draw cumulative native loss-gap/gain and loss/raw/clipped health series.
Keep XL SiTU recovery and final 50k closeout active independently.

Sealed runtime: `ff2fe4b66be25389bd53a9218efa71f406a9b9e5`.
Main contains the configuration ledger only; executable SiTU code is in the runtime worktree.

## Verified launch

The 20 CPU gates passed (4 SiTU, 5 summary-writer, 11 current V5 regressions).
AOT/manifest: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/ff2fe4b/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamMediumPropV5SiTU47.pickle`.
The obsolete two-option fixture in the first attempt was rejected before training;
the corrected fixture leaves model code unchanged and is sealed in `ff2fe4b`.

UC1a acquired 2026-11-12T04:41:19.662838154Z; actual fresh steps 0--14 captured,
first recovery gate observed step 12 with resume_step=-1. Formal waiter then
verified steps 220/223, ~.685--.688 step/s (recorded parent ~.677145, +1--1.6%).
Worker log confirms `Loaded compiled function!`.
Checkpoint 200 is committed: seed 9876, global batch 128, 2 infeed hosts, UC1a
TruePile b0/b500 cursor at step_in_file=201. Generic health ON/internal OFF.
Existing UE/EW queues were retained during policy change and released by the
native recovery controller only after actual UC1a first-step confirmation.

Artifacts: `/data0/xd/bam_diagnostics/mediumprop-v5-situ47-20261112/`.
Group `mediumprop-v5-situ47` owns 2k reporting and 4k/8k confirmation-only reviews;
native gain triple is configured against the trained V5 control with MHA reference.
