# MediumProp V5 ordinary-MLP SiTU 4/7

RUN: `BamMediumPropV5SiTU47`; branch `codex/mediumprop-v5-situ47`;
worktree `/home/xd/.codex/worktrees/medium-v5-situ47/maxtext`.
Training ownership: `xd-v5p-16-311013-maxtext`, UE5a primary / EW4b backup;
UC1a remains excluded. Borrow retained FLEX `llm-jax-v6e-1-0` only for AOT.

## Change and controls

Derive from current-main `BamMediumPropV5`, not XL and not Pythia.
Only ordinary MLPs, including the terminal ordinary MLP, replace SwiGLU by
`(4*tanh(u/4)*sigmoid(u)) * (7*tanh(v/7))`.
Private memory-writing MLPs retain SwiGLU. No parameters, matrix projections,
Pallas interfaces, initialization or optimizer rules change.
Widths remain 2510/1128/tail3904; expected total 432,112,228 parameters.
The existing Pallas read/write kernels are reused: activation is in XLA's MLP path.

The trained direct V5 control is the pre-rename RUN
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormQKScaleTruePile`
(runtime `f8d07cd`, completed13,500, ~.677145step/s).
Use `BamMHAMediumPropC256TruePile` only as the native gain-multiplier reference.
Do not plot a separate MHA loss-gap curve.

## Budget, runtime and workflow audit

From-scratch13,500 steps and LR endpoint13,500; T4096, per-device batch16 on
v5p-16 (8 devices), global batch128. Total7,077,888,000 tokens, unchanged
from the same-size V5 control. Original Prop initialization/AdamPax recipe;
full remat and single-layer scan. Generic training health ON, internal
activation/BAM sow statistics OFF; TB rotation500, checkpoint200,
resume-only retention2 and forced final checkpoint.

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
No inference-cache change; parameter delta0 W_Q.

Fixed public cadence2,000; reviews4,000/8,000 give recommendations only,
and require user confirmation before stopping. Otherwise finish13,500.
Draw cumulative native loss-gap/gain and loss/raw/clipped health series.
Keep XL SiTU recovery and final50k closeout active independently.
