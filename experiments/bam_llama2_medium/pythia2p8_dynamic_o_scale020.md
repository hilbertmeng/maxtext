# Pythia2.8B strongest combination: dynamic local-O gamma .2

Initial delivery covered configuration/code only. On2026-10-28 the user authorized
training by hot-replacing the unmask continuation on its existing v5p128.
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

## Authorized training launch

Main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`; runtime
`5c3a995444897bd8a148a2e888a8c0e1d6721c6b`, EXP/RUN equal the new class above.
Fresh28000-step training with the parent's original schedule, global1024/T2048,
64 devices/B16, normal healthON/internal+BAMOFF and minimal remat. No checkpoint
or optimizer state is inherited from the unmask experiment: its TPU is reused.
Task TPU `xd-v5p-128-unmask500-open-1028`, UE5a primary/EW4b backup, logicalID310421.
The user-owned EW4a FLEX `llm-jax-v6e-1-0` worker0 is borrowed for AOT only;
it remains outside this task's resource ownership.

The existing52-test receipt matches the exact sealed model-source hashes;
full47 shared regressions, full32-head QK100 forward/all-gradient parity,
parameter budget, health and BF16 checks are reused. Effective-config seal passes
despite an unrelated unstaged parent-VMEM edit. Infrastructure audit44141a7..66f2d96:
optimizer/checkpointing/train_compile/max_utils/maxtext_utils unchanged; optional
no-dropout loss probe and O regularizers added since the parent are disabled.
TensorBoard rotation1000 and begin_step remain active. ScopedVMEM60MiB retained.

The first target AOT at66f2d96 failed:read reverse required67.05MiB versus
63.94MiB physical VMEM, including17.05MiB register spills. Runtime5c3a995 ports
the already-tested7161995 MXU gamma-gradient row sum and O-only multiplication.
No experimental/numerical configuration or parameter shape changed. Fresh47
shared regressions and the full32-head QK100 all-gradient parity check passed
(maxrelative2.52106e-6); target AOT then compiled with unchanged128-token tiles
and60MiB scoped setting. The borrowed FLEX remains outside cleanup ownership.

Direct comparisons: the combination parent and `MuddPythia3BSerialMLPShortTrain`;
gain ratios use `Pythia2p8BShortTrain` as the common MHA reference. Native500-step
centers/±25/sample10; public reporting every2000 steps, plots from2000.
Before-launch bet: at common21.5k, gain multiplier versus parent1.02
(plausible1.00–1.04), corresponding to roughly1.56 versus Mudd if the historical
parent1.532 holds; steady speed about.316step/s. No measured28k parent terminal
is available, so21.5k is the declared comparison horizon.

Task artifacts and ownership/monitor state:
`/data0/xd/bam_diagnostics/pythia2p8b-dynamic-o020-train-20261028/`.

Handoff completed from unmaskCP23453 with zero observed lost steps, all16 old
workers confirmed gone and the existing node/queue retained. New worker's actual
HEAD5c3a995, Loaded compiled function, fresh step0, gamma init.2, strict mask,
QK100/NoPE80+RoPE20, health/remat/global1024 and committedCP0 were verified.
Startup20..39 mean.3175step/s versus parent.316482 (about+.3%, no material
regression). The new READY assignment began04:15:03UTC2026-10-28. Old RUN is
stopped with TB SYNC_OK; its closeout reports already_closed and skips every
resource operation so the new trainer remains protected.

## User pause at16k

Paused at committedCP16040 after collecting the full16k comparison window;
the25-step guard slack preserves the native ±25/sample10 measurements.
BothCP16000 andCP16040 commit markers and the data cursor were verified.
The28k run/LR plan and exact5c3a995 AOT remain unchanged for a future resume.
Node and queued resource independently verified absent; localTB sync returned
SYNC_OK. No other task resource or borrowed FLEX was released.

| Same16k comparator | Loss gap (new minus comparator) | Gain multiplier, common originalMHA |
|---|---:|---:|
| Combination parent | -.0033906 | 1.031252996 |
| Serial Mudd | -.0425976 | 1.614845457 |
| Original MHA | -.1118794 | — |

The early deficit disappeared around2.5k; subsequent absolute gain over the
combination parent remained around.003. Final five windows14k..16k average
gap-.00319208, with gain multipliers1.02493..1.03429. This is a persistent small
improvement of the direction SOTA with negligible parameter cost and matched
speed. It has not removed the overall decline of BAM/Mudd gain multipliers.
These are provisional16k results; the declared21.5k R1.02 bet and28k terminal
were not tested because the user requested this pause.

One v5p128 READY lease, UE5a, zero preemptions:
2026-10-28T04:15:03Z–2026-10-28T18:31:01Z,14h15m58s. The end is the pause
boundary; resource release was verified afterward. Assignment started04:15:00Z
on the hot-transferred node. No zone switch or passive candidate.

Final native/window data:`native-final-16000.json` and `report-16000.json`;
the two required figures and their sources:`figures-16000.json`;
lifecycle/lease evidence:`final-pause-evidence.json`;
independent node/queue/checkpoint/cursor proof:`final-independent-verification.json`;
localTB receipt:`final-tb-sync.log`, all under the training artifact directory.
