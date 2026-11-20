# Medium V5 SwiGLU ablation study

Owner codex-xl-v5-situ47. Runtime worktree `/data0/xd/mediumprop-v5-ablation`,
branch `codex/mediumprop-v5-ablation`, forked from main1871c962. Infrastructure
review uses that same main revision; no historical training infrastructure is
copied. TB rotation500/begin_step, checkpoint200/latest2/final checkpoint,
WD rules and original13500-step LR schedule are inherited unchanged.

Parent configuration `BamMediumPropV5`; trained direct control is
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormQKScaleTruePile`.
MHA reference `BamMHAMediumPropC256TruePile`. T4096/global128, v5p-16,
seed9876, UE5a primary/UC1a backup, EW4b excluded. Both MLPs use SwiGLU.
Generic training health ON, internal/BAM activation statistics OFF.

The user authorizes two sequential groups: eight read/write ablations first,
then the other five only after all first-group RUNs finish their full plans.
Each group reports jointly every4000 steps, without performance stopping/reviews.
The implementation never modifies a Pallas kernel. Unsupported address-norm
ablations run native JAX matrix reads/writes, retaining tuned Splash attention.
Other ablations reuse the existing v8t read/write kernels with altered inputs.
The separate `...AblationControlTruePile` class is only a parity fixture, not a
fourteenth training RUN.

Read/write ablation semantics:
- Static write addresses are learned per-layer/per-head [N,V] parameters, normal
  BAM initialization, broadcast over tokens. Contents and write gates remain dynamic.
  Static-all includes embedding, attention and private-MLP writers.
- Static reads remove all dynamic address projections and their gates, as well as
  every static read gate. Static addresses remain learned; M pre-norm, QK/VO scales,
  M-shift and the standard/RoPE QK branch remain. No residual token-conditioned
  gate is secretly retained on the static-read path.
- Gate removal sets the dynamic read/write multipliers exactly to1, including the
  embedding write gate. Static read gates and M-shift are separate and retained.
  The inherited .1 read-key coefficient is retained; no learned compensation is added.
- Address-norm removal covers dynamic QKVO read keys and embedding/attention/MLP
  write addresses. It retains attention write-content RMSNorm and whole-M pre-norm.
- The other group removes M-shift, removes the entire private-MLP writer, removes
  localO and restores self attention, removes attention write-content RMSNorm, or
  deletes standard Q/K projections while applying partial RoPE to the last18 of
  the75 M-read coordinates. The other57 remain unrotated; logit temperature stays
  at the parent's1/sqrt(75).

Every non-negligible saving is refunded to the ordinary MLP at its layer, using
one uniform prefix width and the existing explicit last-layer width. Embedding
savings and integer residuals use the last MLP. No per-layer width mechanism.
Exact full-size counts and widths: `v5_ablation_budgets.json`; target432112228,
maximum absolute integer residual1376. Small-MLP width1128 is retained except
when that entire branch is ablated. Initial static addresses use ordinary BAM
random initialization, without an amplitude-fitting coefficient.

Pre-run priors, terminal gain relative to the parent using MHA as reference:
static MLP write .96, static attention write .95, all-static write .90,
static reads .90, static reads+writes .80, no dynamic gates .97,
no address norms .80, no gates+address norms .85. These are fallible mechanism
predictions, not stopping rules. Static addressing still has token-varying M and
write content, so none of these arms is equivalent to deleting memory.
Pallas speed prior approximately .62-.69 step/s versus historical parent .677;
two native arms approximately .40-.50. The wrapper changes fusion opportunity,
so startup speed must be measured rather than inferred from removed FLOPs.

Second-group priors are to be reviewed before its actual launch, using the first
completed group as evidence. `LoomFormer实验结果.xlsx` ablation tab is updated
in the main worktree's `loomformer_paper` submodule: columnsD/E/F contain the
configuration, batch and backend. An original workbook backup is in the task
artifact directory `/data0/xd/bam_diagnostics/medium-v5-ablation-20261120/`.

Pre-first-step corrections: the two pure-static-read arms cannot inherit both
zero static Q and zero static K: their NoPE dot-product gradients would stay
zero on both sides. Static K therefore uses BAM normal(.006), while static Q
keeps zero; a cold-start model-gradient check requires nonzero static-Q gradient.
Only these two pending controllers were held, retaining their resource queues,
before any training step. Their new AOT/runtime is recorded per RUN; the other
six first-group RUNs retain a253b50. This is a trainability fix, not amplitude
fitting. Group2's M-read partial-RoPE wrapper supplies a constant-zero standard
slab to the unchanged read kernel, drops that output tail, and rotates18 of the
75 M-read coordinates; this avoids a zero-width Pallas input without restoring
any standard Q/K parameter.

## First-group final results

All eight completed 13,500. The final five matched windows are 12,600--13,400;
loss gaps compare against the trained V5 SwiGLU parent, and retained-gain ratios
use MHA as the common reference. Generic health ON; internal/BAM activation
statistics OFF. The final checkpoint/cursor, local TB copies, and UE5a/UC1a node/queue
teardown were verified. Source receipts are in
`/data0/xd/bam_diagnostics/medium-v5-ablation-20261120/`.

| RUN | Final-five loss gap | Retained gain |
|---|---:|---:|
| `BamMediumPropV5AblationStaticMLPWriteTruePile` | +0.001774 | 0.99011 |
| `BamMediumPropV5AblationStaticAttentionWriteTruePile` | +0.000737 | 0.99589 |
| `BamMediumPropV5AblationStaticAllWritesTruePile` | +0.008332 | 0.95347 |
| `BamMediumPropV5AblationStaticReadsTruePile` | +0.044680 | 0.75048 |
| `BamMediumPropV5AblationStaticReadsWritesTruePile` | +0.060870 | 0.66004 |
| `BamMediumPropV5AblationNoDynamicGatesTruePile` | +0.011452 | 0.93606 |
| `BamMediumPropV5AblationNoAddressNormTruePile` | +0.023699 | 0.86764 |
| `BamMediumPropV5AblationNoDynamicGatesAddressNormTruePile` | +0.017694 | 0.90119 |

Interpretation boundary: static-write arms retain the original token-dependent
write gates. They test fixed address directions, not fully static write controls.
Pure-static read arms, in contrast, remove both dynamic read branches and static
read-branch gates at the user's explicit direction. Near-parity static-write
results cannot establish that conditional write strength is unnecessary.
All-static-write also changes embedding addressing, so its larger deficit cannot
be attributed solely to jointly fixing the attention and memory-MLP addresses.
Savings are refunded to ordinary MLPs; these are matched-budget ablations.

The new `BamMediumPropV5AblationStaticAttentionWriteNoGateTruePile` and
`BamMediumPropV5AblationStaticMLPWriteNoGateTruePile`
controls remove only the relevant existing gate, fixing its multiplier to 1;
content remains token dependent and all other gates remain unchanged. Compare
each against its own completed static-address parent. Widths 2636/1128/tail 3922,
total 432,110,932 (target -1,296); runtime 2733de9, same isolated family, existing
Pallas read/write kernels. The earlier token-independent learned-gate proposal
`b1ab4395` was cancelled before training and reverted; it is not a study arm.

The source note's initial priors (.94 attention/.97 MLP) were revised before
startup to .97/.99 after checking H's full no-MLP-write-gate result: final-five
retained gain .993012, runtime ed83fad. New controls run 13,500, cadence 4,000,
no performance reviews/stopping. Their CPU gate checks selected gate and bias
omission, unaffected gates, full formal budgets, original parent controls and
native/Pallas full-model gradients. The updated workbook contains both classes.

A controller on the completed no-gates/no-address-norm arm hung in an unbounded
GCS metadata read. Native closeout finalized its already committed endpoint;
orchestration 6769691 now bounds those queries. Its missing console-cache samples
were backfilled from actual synced TB, preserving existing points: 1,350 TB samples were available at 10-step
intervals; 1,311 overlapped the cache with a maximum difference of 5e-7,
and 39 missing samples were imported. No interpolation was used.

## Fully static addressing without read/write branch gates

`BamMediumPropV5AblationStaticReadsWritesNoGateTruePile` directly compares with
StaticReadsWrites. It removes all three write-gate projections and biases, using
exact multiplier 1; the parent's read projections/gates were already absent.
Write contents, M-shift, matrix pre-norm, read scales and content normalization
stay unchanged. M-shift remains its own factor; "static" refers to address and
extra read/write-branch gate controls, not token-independent contents.

Ordinary width 3037/tail 4329, memory width 1128; total 432,111,796, residual -432.
It removes 672,560 write-gate parameters, refunding 673,200 to ordinary MLPs.
No Pallas changes are needed. Same 13,500 schedule and 4,000 cadence, no stopping.
Pre-run retained-gain prior vs the gated static-address parent: .95 (MHA ref),
with plausible .88--1.00. That corresponds to about .627 of the V5 gain if the
parent's .660 persists. Speed should be similar or slightly faster than its
~.636 step/s parent. This arm joins the same ungated-write follow-up group.
