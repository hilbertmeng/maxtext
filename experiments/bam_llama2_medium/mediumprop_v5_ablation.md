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

## Static-address, ungated-write controls

Two follow-ups remove the existing input-dependent write gate from each trained
static-address parent, one writer kind at a time. Attention static-address / no-gate
compares only with StaticAttentionWrite; memory-MLP static-address / no-gate only
with StaticMLPWrite. The selected gate is exactly1, with its projection and bias
removed. There is no new gate, no learned token-independent multiplier and no
amplitude-matching compensation. All other write/read gates remain unchanged.
Attention content RMSNorm and raw memory-MLP content retain the parents' recipes.
This distinguishes static addressing supported by conditional write strength
from an entirely fixed write-control vector.

Both arms use ordinary width2636/tail3922 and memory width1128. Removing
326672 gate parameters (17 x (1200 x16 +16)) and refunding to ordinary MLPs
leaves total432110932, target432112228, residual-1296. Saved projection per
active layer is .013333 W_Q; no Pallas changes are needed (gate logit +inf).
Runtime infrastructure is unchanged from mainb9a4a6f9; deployed orchestration
6769691 bounds GCS metadata queries. Each arm runs13500 with joint4000-step
reports and no performance stopping/reviews. Pre-run retained-gain priors using
MHA as reference: attention .94, MLP .97 relative to each static-address parent;
throughput near their .608/.610 step/s, possibly slightly faster.

The cancelled b1ab4395 proposal mistakenly made the gate token-independent.
It never launched training and was reverted by1380735b (tree identical160c9c8).

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

## Fully static ungated addressing without M-shift

`BamMediumPropV5AblationStaticReadsWritesNoGateNoMShiftTruePile` directly derives
from `BamMediumPropV5AblationStaticReadsWritesNoGateTruePile`. It sets only the
M-shift mechanism to `none`; the same static read/write addresses, unit branch
gates, token-dependent contents, full-M pre-norm and learned V/O and Q/K read
scales remain. This tests the remaining neighboring-token memory mixture after
conditional address/gate controls have been removed, rather than interpreting
static addressing as token-independent memory.

Removing 18 x (1200 x 32 + 32) = 691,776 M-shift gate/bias parameters refunds
691,200 to ordinary MLPs: prefix width 3047, final width 4351, memory width 1128.
Full total 432,111,220; target 432,112,228; residual -1008 (576 fewer than parent).
Reuse the existing read/write Pallas kernels with `shift=None`; no kernel change.
Infrastructure audit: main 5672136e adds only ledger/documentation changes after
the previous audit; the runtime and deployed bounded-GCS controller fixes remain.
RUN 311216, same worktree/branch, UE5a primary and UC1a backup, 13,500 steps,
4,000-step joint reports, no performance stopping. Pre-run retained-gain prior
relative to its direct ungated-static parent, MHA reference: .90 (plausible
.80--.98). Static readers still benefit from a neighboring token's content even
without token-conditioned addressing. Speed should be similar or slightly faster
than the parent; verify actual startup rather than inferring it from saved FLOPs.
