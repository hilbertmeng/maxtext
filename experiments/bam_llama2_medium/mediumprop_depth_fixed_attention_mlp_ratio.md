# Width-scaled Medium depth controls

Control the MHA attention/MLP allocation when changing depth. Target311084400
non-embedding parameters, matching the original L18/D1200/MLP3200 MHA. This
excludes only the token embedding and unembedding tables; BAM's embedding-to-M
writer projections remain in the model budget. Hold16query/KV heads and use
D=16*head_dim. MHA weights satisfy P=L*(4D²+3DH+2D)+D. Aim at
attention:MLP=1:2, i.e. H=8D/3; integer heads prevent exact simultaneous matching.
Choose the nearest admissible D and uniform integer H, with no terminal exception.

| Depth | D/head_dim | MHA H | MHA non-embedding | MHA attention:MLP |
|---|---|---:|---:|---:|
|12|1472/92|3907|311082176|1:1.99065897|
|18 reference|1200/75|3200|311084400|1:2|
|27|976/61|2633|311086336|1:2.02330943|

MHA classes: `Llama2MediumPropL12AttentionMLP1to2TruePile` and
`Llama2MediumPropL27AttentionMLP1to2TruePile`. They retain the original Prop
RMSNorm/SwiGLU/.006 recipe, TruePile4096, fixed13500-step schedule, tuned Splash,
full remat and single-layer scan. At fixed non-embedding budget the existing
training-token budget remains matched; all-in parameter counts differ with D.

## Mudd counterparts

`MuddLlama2MediumPropL12AttentionMLP1to2TruePile` and
`MuddLlama2MediumPropL27AttentionMLP1to2TruePile` preserve the matched MHA's
complete MLP weight budget. Keep the existing rounded .5--1.5 depth profile,
correct its rounding at the last MLP, and retain full growing history, unscanned
layers and the trained router expansion rule. Native pre/post norm stays OFF.
As in earlier Mudd controls, router overhead is additional and is not repaid
from MLP; no new repayment convention is introduced.

| Depth | MLP base / terminal | Non-embedding | Extra versus matched MHA |
|---|---|---:|---:|
|12|3907 /5924|312459969|1377793|
|27|2633 /4019|314084672|2998336|

## BAM V5 SiTU counterparts

`BamMediumPropV5SiTU47L12AttentionMLP1to2TruePile` and
`BamMediumPropV5SiTU47L27AttentionMLP1to2TruePile` retain ordinary-MLP SiTU4/7,
private-writer SwiGLU, independent VO addresses, M pre-norm, unit-initialized
common QK/VO scales, read-only .5 M-shift and negative attention diagonal.
The1:2 condition defines the matched MHA geometry; BAM then reallocates its
saved attention parameters, preserving ordinary:complete-private-branch ~2:1.
Count private write addresses and gates as part of that branch.

| Depth | M / standard RoPE | Ordinary / private / terminal | Branch ratio | Non-embedding / delta MHA |
|---|---|---|---:|---|
|12|92x32 /24|3129 /1444 /4811|2.00004169|311080776 /-1400|
|27|61x32 /16|1978 /853 /3098|2.00078489|311085334 /-1002|

Use nearest-even quarter-head RoPE widths24/16 and full NoPE92/61. All read,
attention-write, private-write and embedding-write ranks remain256. Embedding
content divisor follows the existing cross-width rule .0016052833*sqrt(D/1200).
Scan the uniform L-1 prefix singly; the explicit terminal has neither attention
nor small-MLP M write. Keep v8t Pallas read/write, packed keys, fused inputs and
48MiB VMEM. K92 and concatenated QK116 stay inside the parent's128-wide padded
blocks; K61 fits64 and QK77 fits128. No new kernel interface is required.

Current main initially contained only the SiTU experiment ledger: its runtime
implementation remained in the completed medium-v5-situ47 worktree. Restore its
bounded gate/up activation, ordinary-only wiring, paired-limit validation and
zero-default fallback, plus the proven SiTU tests. This makes these configurations
executable rather than silently using SwiGLU under a SiTU class name. The private
writer and all default-SwiGLU models retain their old formulas and parameters.

Validation artifacts: `/data0/xd/bam_diagnostics/mediumprop-depth-ratio-20261114/`.
Use full abstract parameter trees for counts/MLP profiles/terminal write omission;
validate both new head geometries against native/Pallas outputs and all parameter
gradients with actual16heads, M shapes and RoPE widths (smaller sequence/model
MLP/ranks for CPU interpretation). Completed:11/11 retained-V5 regressions and4/4 restored SiTU tests passed.
Both new geometry probes covered all104parameter leaves: maximum per-leaf
gradient max-absolute error normalized by the leaf peak was1.843e-4 (L12)
and4.465e-6 (L27), below the existing kernel-control tolerance. SiTU checks
include independent fused/unfused MLP forward/all gradients and explicit
ordinary/private-writer isolation. Full parameter audits passed for all four
new classes, including the Mudd depth profile and BAM terminal write omission.
On2026-11-14 the user authorized all six classes for training. Launch runtime
958cbc017a10b49c18f7904fcf125329a8b66b6a on main/refactor-bam. No additional
model changes: reuse the passed parameter audits,11V5 and4SiTU controls; add
5summary-writer rotation checks and seal all six exact configurations. Current
train/compile/logger/checkpoint/optimizer infrastructure is unchanged since the
reviewed83fa2ad2 main (and the old Mudd control'sc7a6730); no historical files
are copied into the runtime. All six use TBrotate500, checkpoint200/latest2,
forced final checkpoint, generic health ON/internal OFF, init/data seed9876,
global128xT4096,13500steps=7077888000tokens and LR endpoint13500.

| TPU suffix | Architecture | Depth |
|---|---|---:|
|311023|MHA|12|
|311024|Mudd|12|
|311025|BAM V5 SiTU|12|
|311026|MHA|27|
|311027|Mudd|27|
|311028|BAM V5 SiTU|27|

All are xd-v5p-16-SUFFIX-maxtext. UE5a primary, UC1a/EW4b backups after300s.
No idle retained FLEX/non-preemptible compiler was available at launch; each
AOT uses the automatic EW4a/UC1a/UE5a v6e-1 compiler workflow. Launch each trainer
only after its own artifact/manifest passes. The journaled launch and first-step
gates live on tpu-ag under logs/depth-ratio-{launch,first}*; local receipts and
launch-plan.json live with the validation artifacts. Borrow no other workflow's
spot compiler; temporary AOT resources are automatically released.

Group mediumprop-depth-fixed-ratio reports all six together at fixed2000-step
milestones, with Mudd27 as its final clock. No performance-based stopping or
review points: full budgets measure scaling rather than race for SOTA. Native
comparison windows stay200steps. MHA compares its original18-layer and old
same-depth MHA; Mudd compares its original18-layer; BAM compares original SiTU18
and new same-depth Mudd. Matched MHA references are registered for gains without
adding routine MHA loss-gap lines to BAM reports.

Primary metric R_d=(MHA_d-BAM_d)/(MHA_d-Mudd_d), with R_d/R18 quantifying whether
BAM's relative Mudd advantage grows with depth. Also retain both architectures'
independently matched-MHA depth-gain ratios and actual12/18/27 depth-axis loss
curves; the direct absolute depth effect and proportional advantage differ.
Prior: K12=R12/R18=.96; K27=1.00 (range .94--1.07). Corrected MHA/MLP allocation
should help MHA/Mudd27 more than before, potentially narrowing the old BAM depth
advantage. Startup speed priors(step/s):12-layer MHA/Mudd/BAM .95/.84/.82;
27-layer .68/.60/.61. These are unpaired predictions, not measured results.
