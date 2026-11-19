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
