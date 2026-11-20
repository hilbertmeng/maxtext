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

## Remaining groups: final results

All nine remaining RUNs completed 13,500 without performance stopping. Final-five
means below use the native registry's 12,600--13,400 windows, not single-step loss.
Every gain multiplier uses `BamMHAMediumPropC256TruePile` as reference.
Final checkpoints/cursors (seed 9876, global batch 128, two hosts), all nine local
TB syncs, and node/queue absence in UE5a, UC1a and the scoped EW4b candidates
were independently verified. Group 2/3 source JSON and plot provenance are under
`/data0/xd/bam_diagnostics/medium-v5-ablation-20261120/`.

| RUN | Comparator | Final-five gap | Gain multiplier |
|---|---|---:|---:|
| `BamMediumPropV5AblationNoMShiftTruePile` | V5 SwiGLU | +0.010732 | 0.94008 |
| `BamMediumPropV5AblationNoMemoryMLPTruePile` | V5 SwiGLU | +0.034947 | 0.80482 |
| `BamMediumPropV5AblationNoLocalOTruePile` | V5 SwiGLU | +0.032842 | 0.81659 |
| `BamMediumPropV5AblationNoAttentionWriteContentNormTruePile` | V5 SwiGLU | +0.002726 | 0.98479 |
| `BamMediumPropV5AblationMReadQKPartialRoPETruePile` | V5 SwiGLU | +0.020772 | 0.88400 |
| `BamMediumPropV5AblationStaticAttentionWriteNoGateTruePile` | static-attention-address parent | +0.016494 | 0.90750 |
| `BamMediumPropV5AblationStaticMLPWriteNoGateTruePile` | static-MLP-address parent | +0.005019 | 0.97168 |
| `BamMediumPropV5AblationStaticReadsWritesNoGateTruePile` | gated static-read/write parent | +0.017872 | 0.84878 |
| `BamMediumPropV5AblationStaticReadsWritesNoGateTruePile` | V5 SwiGLU | +0.078742 | 0.56023 |
| `BamMediumPropV5AblationStaticReadsWritesNoGateTruePile` | Mudd18 post-norm, fixed router | -0.008636 | 1.09424 |
| `BamMediumPropV5AblationStaticReadsWritesNoGateNoMShiftTruePile` | ungated static-read/write parent | +0.003927 | 0.96084 |
| `BamMediumPropV5AblationStaticReadsWritesNoGateNoMShiftTruePile` | V5 SwiGLU | +0.082669 | 0.53828 |
| `BamMediumPropV5AblationStaticReadsWritesNoGateNoMShiftTruePile` | Mudd18 post-norm, fixed router | -0.004708 | 1.05138 |

The pure-static controls retain token-dependent write contents, 17 nonlinear
private M-MLP writers, whole-M read pre-norm, read scales and the residual-derived
standard Q/K branch. They are not input-independent memory. Removing M-shift from
this circuit incurs a small additional loss, rather than destroying its benefit.
Both controls remain ahead of the stronger fixed-router/post-norm Mudd baseline
at completion, although their advantage narrowed over training. Mudd retains its
historical extra router/norm parameters without an MLP deduction.

Static write directions alone nearly preserve V5's benefit, but deleting the
remaining token-dependent write-strength gate is measurably harmful. The static
attention writer loses more than the static private-MLP writer. A plausible
mechanism is that normalized attention content has lost its amplitude channel,
whereas unnormalized private-MLP content can encode conditional write strength;
these observations do not establish that mechanism without a matched interaction
control. The apparent attention no-gate recovery around 7k is a parent loss spike,
not sustained child catch-up (actual point evidence archived separately).

Removing attention content RMSNorm on V5 costs much less than on UnitH: the older
UnitH control retained only 0.77733 of its parent gain through 7.6k, while V5 stays
near 0.985 at completion. The main new stabilization candidate is whole-M read
pre-norm, which removes direct read-amplitude feedback from M magnitude. Q/K read
scaling was also added, so its contribution is not isolated. The tiny standalone
pre-norm gain does not exclude a large interaction with raw attention writes.

Prior checks: raw attention writes substantially outperform the 0.85 retained-gain
prior; the ungated static-read/write control underperforms its 0.95 prior versus
gated static writes; removing M-shift is less costly than its 0.90 prior. No prior
was used as a stopping rule. Ablations refund parameters, so loss deltas include
the specified capacity reallocation and are not fixed-width surgical interventions.

| RUN | Ordinary / private / final widths | Total parameter delta vs 432,112,228 | Startup step/s |
|---|---|---:|---:|
| `BamMediumPropV5AblationNoMShiftTruePile` | 2520 / 1128 / 3926 | -576 | 0.639 |
| `BamMediumPropV5AblationNoMemoryMLPTruePile` | 3765 / 0 / 3908 | +800 | 0.641 |
| `BamMediumPropV5AblationNoLocalOTruePile` | 2557 / 1128 / 3958 | +1,296 | 0.619 |
| `BamMediumPropV5AblationNoAttentionWriteContentNormTruePile` | 2510 / 1128 / 3904 | +0 | 0.608 |
| `BamMediumPropV5AblationMReadQKPartialRoPETruePile` | 2702 / 1128 / 4096 | +0 | 0.595 |
| `BamMediumPropV5AblationStaticAttentionWriteNoGateTruePile` | 2636 / 1128 / 3922 | -1,296 | 0.616 |
| `BamMediumPropV5AblationStaticMLPWriteNoGateTruePile` | 2636 / 1128 / 3922 | -1,296 | 0.613 |
| `BamMediumPropV5AblationStaticReadsWritesNoGateTruePile` | 3037 / 1128 / 4329 | -432 | 0.645 |
| `BamMediumPropV5AblationStaticReadsWritesNoGateNoMShiftTruePile` | 3047 / 1128 / 4351 | -1,008 | 0.677 |

Speeds are startup observations on different leases, not paired kernel benchmarks.
Existing Pallas kernels were reused; all formal source hashes are recorded on the
main configuration classes. No kernel hot switch was needed for these groups.

The two fully static controls had small missing console-cache regions. Only
missing steps were imported from actual synced TensorBoard values: 28/59 points,
with 1,322/1,290 overlapping points agreeing within 5e-7. Existing console points
were preserved and nothing was interpolated. Final comparisons were recomputed
through the native loss-report path after both child and parent completed, so a
child finishing first cannot leave its final parent comparison stale at 13.2k.

## Historical all-static comparison

The old `BamMediumPropK75AllLocalStaticAlibiRMTBudget` runtime `a520e4a` had no
residual-derived standard Q/K projection, no M pre-norm and no private-MLP content
feedback into M. Its static-attention implementation uses `inputs_q` only for
shape; at fixed M, later ordinary-MLP residual features cannot update the memory.
The new controls keep that nonlinear content path even after dynamic addresses
and branch gates are deleted. This is the strongest circuit-level distinction.
The old control also used a roughly 328.6M RMT-matched budget instead of 432.1M,
ALiBi, different static-address initialization and legacy padded data. Its dynamic
BAM sibling won at the same smaller budget, so budget alone cannot explain the
old static failure. Cross-cohort raw loss is not a matched architecture delta.
The immutable source audit and native curves are archived in
`old-static-v5-comparison.md` in the task artifact directory. No additional
checkpoint diagnosis or new control RUN was launched as part of that audit.

## Resource closeout

Each remaining RUN started in UE5a and finished in UC1a; individual same-zone
preemptions are retained in the full lease history. The user approved EW4b only
as a temporary extra candidate for 311210 and 311214. Neither had an active EW4b
training lease; this exception never changed the global defaults or other RUNs.
Every READY interval and active-zone assignment is recorded in
`experiments/tpu_region_preemption_history.md`; queue candidates are not counted
as active leases. The retained FLEX compiler `llm-jax-v6e-1-0` was not deleted.
