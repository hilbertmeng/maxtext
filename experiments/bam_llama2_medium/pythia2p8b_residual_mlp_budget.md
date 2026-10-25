# Pythia2.8B residual-MLP capacity controls

User-authorized2026-10-25: launch the narrow-MHA and wide-residual-BAM
mechanism controls together, plus an equal-budget rank-refund BAM. The final
rank instruction is **all four ranks640=D/4**: shared read, attention write,
MLP write and embedding write. Earlier512/write-only designs were superseded
before sealing or training. Use main `/home/xd/projects/maxtext`, branch
`refactor-bam`; no new worktree. Task artifacts:
`/data0/xd/bam_diagnostics/pythia2p8b-residual-mlp-budget-20261025/`.

The question is whether reducing direct vector-residual FFN capacity explains
the late erosion of BAM's MHA advantage. Historical diagnosis found sparse
M-writing FFNs contribute little through their direct vector addition; their
mere residual connectivity does not justify counting the whole FFN as ordinary
residual capacity. The2:1 parent separates these roles explicitly. Its ordinary
FFNs total1,235,728,010 versus MHA/Mudd1,678,131,200 (-26.36%).

## Arms and budgets

Parent RUN `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1`
is paused21545 with complete21500 comparison windows; runtime6c60bfd.
MHA `Pythia2p8BShortTrain` is paused21726; Mudd
`MuddPythia3BSerialMLPShortTrain` paused21666. Preserve their exact caches and
original28000-step LR schedules; do not resume or delete their resources.

| New configuration | Ordinary GELU widths | Memory width | Rank read/attention/MLP/embedding | Total parameters |
|---|---|---|---|---:|
| `Pythia2p8BShortTrainResidualMLP7552` | 31x7552, final7178 | none | standard MHA |2,332,810,890|
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1ResidualMLP10240` | all10240 |3072|768/768/768/768|3,217,324,032|
| `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640` | [8064,7936,7936]x10, tails8064/8192 |3072|640/640/640/640|2,774,883,968|

The first two have the user's explicit diagnostic exception to equal total
parameters: respectively MHA-442,403,190 and2:1+442,403,190 (67.50537 W_Q,
W_Q=2560^2). The wide BAM does **not** use the existing `MemoryMLPMHAWidth`
configuration, which shrinks memory3072->256 and cannot isolate FFN capacity.

R640 saves73,400,320=11.2 W_Q: attention+MLP writes37,158,912, shared readers
35,651,584, embedding589,824. Integer widths refund73,363,446 into ordinary
FFNs; total residue-36,874 (-.001329%), the closest128-aligned total. Ordinary
FFNs become1,309,091,456=78.0089% of MHA, recovering16.5829% of the parent's
ordinary-FFN deficit. Read/write up dimensions,32 heads,M80x64, independent
write/read downs, memory3072 and all write placements remain unchanged.

All arms use T2048/global1024, v5p-128/64 devices/16 sequences per device,
no accumulation, BF16/minimal remat, LR2e-4/warmup560/cosine28000/WD.1,
complete-head QKNorm, generic health ON/internal and BAM health OFF. MHA
retains parallel residual and Pythia embedding initialization; BAM retains
serial residual and.006 embedding/BAM roles. Each diagnostic changes only
the intended family's FFN widths; common baseline comparisons are unchanged.
Initialization **rules** remain fixed, not identical forward functions:
initial FFN-output RMS scales approximately sqrt(H), predicting.8588x for
narrow MHA and1.1644x for wide BAM body FFNs. Treat this as a known width/init
coupling, not a falsely claimed initialization-equivalent experiment.

Narrow MHA needs31 scanned layers plus one explicit final Pythia layer to
match7178 exactly. The opt-in `pythia_last_mlp_dim` path preserves full model
depth in initialization and LR; default0 leaves other models unchanged. The
two BAMs retain10 three-layer scan blocks plus two explicit tails.

## Validation and infrastructure

The historical2:1 effective model settings and runtime6c60bfd are the parent.
Main-only train/train_compile/optimizer/max_utils/maxtext_utils/checkpointing/
summary-writer changes through main15d676be were audited: no differences.
Existing host AsyncOptions checkpoint fix and TB rotation are present. New
code only adds the optional plain-Pythia final-width path; test coverage checks
full abstract parameter trees, all ranks and writer placements, live prefix/
tail gradients and causality, ordinary Pythia and BAM regressions. Summary
writer's five tests passed in `cpu-targeted`; the complete gate is `cpu-final`.
The old2.8B parameter-count regression was corrected for its already-enabled
complete-head QKNorm (+3840 vs the historical20-wide scales), not by changing
model initialization or projections.

Before training, seal/push, check all inherited runtime configurations, verify
exact v5p-128/s28000 AOT manifests and actual `Loaded compiled function!` plus
FIRST_STEP. Require the first1000-step checkpoint and permanent2000 retention
to commit; rotation1000/begin_step and fixed2k/.../26k retention inherit.

## Ownership, reporting and prior

Three task-owned training prequeues submitted UE5a at2026-10-25T02:23:55–59Z:

| Tag / ID | RUN suffix | Training TPU |
|---|---|---|
|narrow /310360|`Pythia2p8BShortTrainResidualMLP7552`|`xd-v5p-128-mlp-small-1025`|
|wide /310361|`MemoryMLPRatio2to1ResidualMLP10240`|`xd-v5p-128-mlp-wide-1025`|
|rank /310362|`MemoryMLPRatio2to1AllR640`|`xd-v5p-128-allr512-1025`|

The rank arm's physical TPU was named during the superseded512 draft; it is
only an allocation identity. The RUN/config/AOT must all say640. Do not reset
a provisioning queue to rename it. PrimaryUE5a, backupEW4b; staged passive
backup policy applies after5min. Borrow verified READY FLEX_START compiler
`llm-jax-v6e-1-0`, EW4a/worker0; exact QR provisioningModel verified and no
active trainer seen. Serialize AOTs under the shared worker lock; own neither
the compiler nor unrelated resources. Start each prepared trainer as soon as
its own gates pass, while the remaining AOTs continue.

Keep automatic/LR horizon28000, with task-owned guards at21530 for the common
21500 comparison window unless the user changes the horizon. This mechanism
question needs late training; early underperformance alone is not a stop
criterion. Group every2000 steps with native500/±25/sample10 windows. All
plots omit the first2000. Keep original MHA as gain-ratio reference; retain
2:1 direction SOTA and Mudd. Narrow MHA supplies the explicit auxiliary
width-penalty curve, not a silent replacement of the common reference.
No routine raw-gradient reports or new BAM/shift health statistics.

Pre-run judgment ranges, not confidence intervals, at common21.5k:
- Narrow MHA loss gap vs MHA +.025 [.010,.045]; the width penalty should
  grow substantially in the second half if it explains the erosion.
  Expected~.415step/s [.395,.435] versus MHA~.359.
- Wide BAM gap vs2:1 -.015 [-.025,-.007], MHA-reference gain~1.16
  [1.08,1.27], BAM/Mudd~1.59. Expected~.285 [.270,.300]step/s vs2:1~.325.
- AllR640 gap vs2:1 -.003 [-.009,+.004], gain~1.03 [.96,1.10],
  expected~.330 [.320,.340]step/s. Smaller rank reduction buys only16.6%
  of the ordinary-FFN deficit, so this is an engineering tradeoff test.

Judge full trajectories: does narrowing MHA create a growing penalty, and
does widening BAM with fixed memory weaken late erosion rather than simply
shift the whole loss curve? Do not call an equal-budget tradeoff alone proof
of the mechanism, or call21.5k a completed28k terminal result. Result comments
belong to closeout; keep ordinary monitoring in task artifacts.

## Sealed launch

Runtime **7b6aecbc39184ea4edbb32c6a40e513f6c44c69b** on all three trainers;
the later startup-metadata commit does not change this runtime. All65 CPU
checks and sealed inherited-config checks passed. Three exact v5p-128/s28000
AOTs passed manifest/size/CRC verification. The retained FLEX compiler was
borrowed under its worker lock and remains user-owned. All trainers loaded
their compiled function and passed FIRST_STEP and20-step speed gates.

| Arm | First READY observation UTC | Formal registration UTC | Startup throughput |
|---|---|---|---:|
| narrow MHA |2026-10-25 02:31:04|2026-10-25 02:37:21|~.400step/s|
| wide residual BAM |2026-10-25 02:31:05|2026-10-25 02:37:40|~.294step/s|
| allR640 BAM |2026-10-25 02:31:07|2026-10-25 02:39:11|~.330step/s|

All timings use minimal remat, generic health ON and internal/BAM health OFF.
The parent2:1 startup control was.3294step/s; its later resumed speed~.325 is
a different timing phase, not evidence of a measured2% rank-refund speedup.
Ordinary MHA's corresponding minimal-remat control was.3603step/s.
Checkpoint0 was observed committed; require1000/2000 at the first due reports
to verify ongoing checkpointing and fixed-step preservation. Exact launch
commands, AOT identities, worker HEAD/loading/step evidence, report cursors
and task guards are in the task artifact directory, not interim loss diaries
in `exp.py`. Runtime disables `save_config_to_gcs`; use sealed-source config
verification plus the actual worker HEAD/AOT/batch-token/LR evidence instead
of claiming a nonexistent GCS `config.yml` was inspected.
