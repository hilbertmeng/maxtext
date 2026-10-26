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

## Capacity-control closeout and R640 comparison horizon

The user hot-replaced narrow MHA at committed16668 and wide BAM at
committed12308. Both stopped registries were checked separately after the
replacement's FIRST_STEP. Their TPUs transferred to the new owners; closeout
reported `already_closed=true` and skipped all resource operations. Both local
TB syncs returned SYNC_OK. No preemptions or rollback occurred.

Narrow MHA's original-MHA gap was +.022331 at2k, +.015511 at3k,
+.017079 at8k and +.019705 at16.5k; final5 mean+.019436. Its final gap
against2:1 BAM was+.116139 (final5+.117534). Later throughput~.397step/s
versus matched original MHA.3603. The narrowing penalty persists, but does
not reproduce BAM's large declining advantage through the observed interval.
Its21.5k +.025 prior and full28k result remain untested after the user stop.

Wide BAM's gap against2:1 improved from-.004937 at2k to about-.0094 at6k,
then held near-.009..-.010 through12k. Final gap-.009671/R1.093015;
final5 gap-.009272/R1.086569. Final gaps against MHA/Mudd were
-.113646/-.043346, with MHA-reference BAM/Mudd R1.616594. Later speed
~.291 versus2:1~.325step/s, with matched health settings. Its added ordinary
FFNs lower the loss level but do not materially change6k..10k erosion:
the original2:1 MHA advantage shrinks.0208886 and wide BAM.0211590.
That .0002704 endpoint difference changes sign when endpoints use five-window
means (wide's lead grows.0002302); it is not evidence that widening worsens
decay. The21.5k R1.16 prior remains untested.

AllR640 reached its planned common comparison horizon and paused at21551,
with committed checkpoint21551 and both node/queued resource verified absent.
The local incremental TB sync completed successfully. Versus originalR768
2:1, early lag crosses at6.5k and becomes a persistent small lead. At21.5k:
gap-.0014486/R1.0157622; final5 mean-.00119368/R1.012852. Versus Mudd:
gap-.0262238/R1.3906537, compared with2:1 R1.3690740. The rank refund gives
a small net gain, without a material late widening; original R1.03 prior was
optimistic at this horizon. This is a provisional21.5k comparison, not a
completed28k run; the original LR/AOT schedule remains unchanged.

Together these controls weaken ordinary-MLP shortage as the primary observed
decay mechanism. They establish a capacity cost and loss-level benefit;
the two unequal-budget arms did not run to the full endpoint, and the
fixed-budget R640 tradeoff alone does not isolate its cause.

## Independent VO addresses and hybrid-V followups

Two user-supplied, checked configurations run on the retained training TPUs:

| RUN suffix | ID / retained TPU | Runtime | Ordinary / memory GELU / final | Total parameters |
|---|---|---|---|---:|
| `MemoryMLPRatio2to1AllR640SeparateVOKeys` |310380 / `xd-v5p-128-mlp-small-1025`|b79aae76|7808 /2944 /9984|2,775,189,600|
| `MemoryMLPRatio2to1AllR640HybridV8DropLocalOSeparateVOKeys` |310381 / `xd-v5p-128-mlp-wide-1025`|b79aae76|7808 /2816 /8320|2,775,141,856|

Both names have the prefix `BamPythia2p8BSotaIndependentR768PallasShortTrain`.
The full runtime is `b79aae76d98bcf2afb467de82639e93908428ceb`, on main
`refactor-bam`. All four ranks remain640. L0..L30 use single-layer scan;
L31 is an explicit writer-free tail. SeparateVO gives V/O independent
dynamic addresses while retaining their shared read down. Hybrid adds eight
ordinary W_V heads plus24 M-V heads, with M Q/K reads on all32 heads;
it removes local O only for the ordinary heads and retains standard W_O.
The hybrid cost is funded from private memory MLP, preserving ordinary7808;
do not substitute the superseded ordinary7680/memory2944 draft.
Parameter residues against R640 are respectively+305632/.011014% and
+257888/.009294%, with every hidden width128-aligned.

The original56MiB VO AOT failed because reverse scoped VMEM needed58.13MiB.
Runtimeb79aae76 increases only the scoped quota to60MiB (physical64MiB),
keeping the128 tiles. Both exact v5p-128/s28000 AOTs and real training
verified60MiB, `Loaded compiled function!`, FIRST_STEP and steady speed:
SeparateVO~.318..320, Hybrid~.307..310step/s. Health/minimal remat match
R640; no new internal/shift statistics. Later metadata commits do not change
either registered runtime. Replay VMEM from the sealed runtime rather than inferring its
quota from a later mutable working copy.

The verified parent's targeted VO forward/backward equivalence check passed;
existing main integration had67 CPU checks. Per user instruction, broad CPU
gates were not repeated for these checked configurations. The borrowed
FLEX_START compiler remained user-owned; exact AOT identity/startup/handoff
evidence is in `separate-vo-hotswitch/` and `hybrid-v8-hotswitch/` under the
task artifact root. Direct baselines retain R640, original2:1, MHA and Mudd;
Hybrid additionally retains SeparateVO. Persistent native gain-ratio triples
use the original MHA reference. Keep500-step windows and grouped2k reports,
omit initial2k, and preserve the common21.5k pause guards and28k LR plan.

Pre-run bets at21.5k: SeparateVO/R640 R~1.04; Hybrid/SeparateVO~1.02,
equivalently Hybrid/R640~1.06. These are hypotheses, not measured endpoints.
Independent VO is motivated by separating value selection from output routing;
Hybrid tests a retained ordinary value path, but its memory-budget and tail
changes mean it is not an isolated test of ordinary W_V causality.


### Hybrid V8 closeout

User-stopped at committed11550/28000, before the common21.5k horizon.
Last complete11.5k window versus SeparateVO: gap+.0008142/R.9927631;
last5 gap+.0007022/R.9938612. Its early deficit briefly closed around8k,
then became positive again from9.5k and widened modestly. Final gaps against
R640/original2:1/MHA/Mudd were-.0036496/-.0053618/-.1116922/-.0398166;
MHA-reference R versus R640/Mudd was1.0337793/1.5539655.
At this allocation the ordinary-V addition did not improve the direction SOTA;
the21.5k R1.02 prior remains untested after the user stop. Medium's completed
matching hybrid+VO arm likewise finished near pure VO (final5 R.99915727),
supporting weak net additivity across the two scales, not proof about an
isolated W_V mechanism. SeparateVO continues with its registered comparisons.

Native closeout completed2026-10-26 01:09:16 UTC: checkpoint11550 committed,
all16 worker processes gone, node/queued resource independently verified
absent, zero lost steps, TB SYNC_OK. One UE5a v5p-128 READY lease:
2026-10-25 14:33:10 ->2026-10-26 01:06:29 UTC,10h33m19s, no preemptions.
Authoritative summary: tpu-ag `/home/lishengping/xd/projects/logs/closeout-20261026T010916Z.json`.


### SeparateVO common-horizon pause

Paused at committed21554/28000, finishing the matched21500 window. The durable
guard released the node and queued resource; both absence checks were repeated
independently. Fixed checkpoints2000 and20000 remain committed. Local TB sync
completed successfully. Zero preemptions; sole READY lease2026-10-25 14:23:32
through2026-10-26 09:26:09 UTC (19h02m37s), followed by verified resource release.

Versus R640: early gap-.009926 at2k shrinks to-.004459 at8k, then remains
near-.004..-.0046 through21.5k. Final gap-.0040188/R1.0430501;
final5 mean-.0041680/R1.0442794. This supports the original21.5k R1.04 prior.
Versus originalR7682:1: final-.0054674/R1.0594909; final5-.00536168/R1.0576997.
Final gaps versus MHA/Mudd are-.0973706/-.0302426, MHA-reference R versus Mudd
1.4505214 (final5 1.4552445). The direction improvement holds while the overall
BAM/Mudd ratio continues its slowing decline. Later speed~.317 versus matched
R640~.328step/s (~-3.4%). Added total parameters are only.0110%; memory/cache
shapes and health settings remain matched. This is the strongest measured parameter-matched
candidate in this local allocation family; restoring ordinary V heads did not
add a net gain over it at the tested Hybrid allocation.

Original28k LR/AOT remains resumable. The separate calculated28k R640/Mudd
forecast1.38 (model/window range1.35..1.40) is not validated by this21.5k pause.
Its reproducible fits/backtests are task-local `R640-terminal-forecast/`;
no interim loss diary or forecast is added to the configuration ledger.


## Full NoPE QK80 / total QK100 followup

User-authorized2026-10-26: run
`BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysQK80`
on main, no new worktree. The class retains full80-D M-derived NoPE plus20-D
RoPE, versus the SeparateVO parent's60+20. Value head80 and M80x64 stay fixed;
complete-head QKNorm covers100 coordinates, attention scale remains1/sqrt80.
Only1280 norm scalars are added, total2,775,190,880 (.0001953125W_Q, .0000461%
above direct parent); the negligible cost leaves all128-aligned MLP widths intact.
All ranks640, body ordinary7808/private2944, tail9984 and single31+tail1 scan remain.

Runtime59ef09e04368599b143776f4f707ae2c86e2162f, refactor-bam. The verified optimized
full-NoPE kernel was merged at96a57fc2; user-directed minimal CPU launch gate reuses
its60 passed checks (including exact Pythia trees, full-QKNorm native/Pallas gradients
and summary rotation) and reruns the exact effective-config seal. No differences in
train/train_compile/optimizer/max_utils/maxtext_utils/checkpoint/summary/pyconfig/base
infrastructure were found versus parent runtimeb79aae7. Other merged model features
are dormant for this configuration. Source at83579c8 used the working-copy inherited
56MiB budget and AOT failed at58.11MiB in read backward; candidate-only60MiB at59ef09e
passes with physical64MiB and the same math/128 tiles. Keep the parent60MiB default
runnable in committed source; preserve the separate unstaged working-copy removal.

Borrowed verified guaranteed FLEX_START llm-jax-v6e-1-0/EW4a/worker0, under the shared
worker lock; its lifecycle remains user-owned. Exact v5p-128/s28000 artifact:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/59ef09e/jax081-i0ae3f58-c17f538a/v5p-128/s28000/BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysQK80.pickle`
plus.manifest.json. Exact AOT manifest/CRC verified before requesting training hardware.
Formal UE5a spot trainer xd-v5p-128-qk80-1026/ID310382 registered09:51:25UTC, no passive
backup submitted; EW4b remains staged recovery policy. Actual worker HEAD59ef09e,
Loaded compiled function, FIRST_STEP and checkpoint0 committed all verified. Window
123..142,20 logged steps, harmonic mean.316706step/s/median.318 including a routine
flush; matched parent~.317, approximately unchanged. Health ON/internal+BAM/shift OFF.
Global batch1024,64 devices,16/device, T2048, minimal remat, no accumulation.

Prerun common21.5k bet: candidate/SeparateVO MHA-reference R1.03 [1.00,1.06],
expected speed.312 [.300,.320]. Prior evidence: cached Medium QK75 gain1.02717
at10k; optimized paired same-TPU speed-1.359%. New model restores the previously
truncated20 M-content coordinates for Q/K with negligible parameter cost.
Direct baselines are SeparateVO, original MHA and Mudd; native triples use original
MHA reference. Keep public2k/native500 cadence and skip first2k. Preserve requested
original sparse-BAM/2:1/R640 references in cumulative MHA-gap/Mudd-ratio overlays.
Original28000 LR/AOT retained, common21500 comparison checkpoint pause guard21530.
Task state, commands, AOT failure, worker/speed/CP evidence and figure sources:
`/data0/xd/bam_diagnostics/pythia2p8b-residual-mlp-budget-20261025/qk80-launch/`.
