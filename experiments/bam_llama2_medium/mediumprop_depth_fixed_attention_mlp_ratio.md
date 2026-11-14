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
No AOT, TPU acquisition or training launch is requested for these classes.
