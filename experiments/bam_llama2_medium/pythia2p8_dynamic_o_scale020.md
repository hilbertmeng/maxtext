# Pythia2.8B strongest combination: dynamic local-O gamma .2

User confirmed configuration/code only; no2.8B training or AOT resource request.
Parent: BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80
New: BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80LocalODynamicNormScale020
Development: /data0/xd/local-o-dynamic-norm-init, branch codex/local-o-dynamic-norm-init.
Artifacts: /data0/xd/bam_diagnostics/pythia2p8-dynamic-o-scale020.

The parent is the trained R640, independent-VO, strict-history negative-diagonal,
full80 NoPE+20 RoPE combination. Full100-D QKNorm before RoPE, GELU serial MLP,
embedding.006/BAM.006 initialization and ordinary transformer Pythia initialization
all inherit. Body31 layers single-layer scan plus explicitL31, ordinary7808/private2944,
terminal9984, minimal remat, tunedSplash, Pallasv8t read/write and60MiB scopedVMEM.
Original28000-step LR/optimizer/batch/data/checkpoint plan inherits without a restore.

Change only learned dynamic-O address scale initialization to.2. Each layer has
32heads*64coordinates, after address RMSNorm/dynamic gating and before adding staticO.
StaticO, Q/K/V, existing dynamic/static sigmoid gates and M shifts remain unchanged.
The scale is unconstrained, not sigmoid-reparameterized and not a1+gamma offset.

Added32*32*64=65536 parameters (.01 W_Q total, .00236% of parent), exempt from MLP
repayment. Exact total2775256416 versus parent2775190880. All MLP widths unchanged
and128-aligned; no per-layer width override. Existing scale WD exclusion applies.

Health producer is the existing record_training_health_metrics=True path, with
record_internal_nn_metrics=False and broad BAM statisticsOFF. It produces32 gamma
parameter L2 norms plus32 raw-gradient L2 norms. Body31 and explicit tail are covered.
For readable layer RMS divide bysqrt(2048)=45.254834, not Medium sqrt(512).
Gamma-init.2 therefore has initial layer L2 norm9.050967 and RMS.2. Parameter RMS
is not a signed mean, actual O/context output ratio or causal attribution.

Pallas support reuses the trained881f991 implementation and its2655622 O-mask
prerequisite, keeping parent behavior when scale/mask are disabled. This ports the
existing dynamic-O forward and dgamma VJP, not a new kernel algorithm. Main's old
Medium reference configurations retain their native flags; running Medium single-gate
experiments remain sealed at2f36e3a and are not hot-switched.

Validation: exact full-model abstract parameter budget; actual32-layer generic
parameter/raw-gradient health export and WD exclusion;32-head M80x64/QK100 attention
native versus Pallas outputs and all gradients with packed segments, M-shift and
nonzero staticO/perturbed gamma (max relative gradient error2.53e-6); full47 BAM
regressions. Extra finite-BF16/validation checks are saved alongside these receipts.
The geometry test respects native[K,V] versus fused[V,K] carry layout and the fused
attention's deferred-writer return. Formal configuration/health checks passed directly;
geometry fixture setup was corrected before its successful result.

A future training launch must compile this new class's target-topology AOT; the
parent AOT has a different parameter tree and is not reused. No target-TPU AOT or
hardware timing claim is made by these CPU interpret checks.
