# All-in-budget BAM24 ordinary SwiGLU control

The user corrected the27-layer wording: intended arm belongs to the active
all-in-total-budget24-layer study. Do not reuse the mistaken D976/BAM27
configuration, checkpoint or conclusions. Reuse only its retained v5p-16
resource after its old worker is quiescent.

Class `BamMediumPropV5L24TotalBudgetSwiGLUTruePile` inherits directly from
`BamMediumPropV5SiTU47L24TotalBudgetTruePile` (actual parent runtime46e26e7).
Only ordinary SiTU limits change4/7 to0/0; private writers already use
SwiGLU. D1056/16x66, M66x32, all independent R256; ordinary/private/tail
2201/968/3435. Total432120064, unchanged and1136 below the MHA18/24 target.
No parameter reallocation or kernel interface change.

Current main shared layers/training sources remain identical to the actual
parent runtime; keep v8t packed fused read/write,48MiB VMEM, full remat,
tuned Splash and single-layer scan of L0–22 plus explicit L23. Terminal
attention/private M writes stay disabled. Seed9876, global128xT4096,
TruePile4097 records,13500 steps =7077888000 tokens; LR endpoint13500.
Generic health ON/internal OFF, TBrotate500, checkpoint200/latest2/final.

Direct controls: SiTU24 parent, current strongest post-only fixed-router
Mudd24, matched MHA24. Native MHA-referenced gain triples; paired depth
comparison retains SiTU18/MHA18 to isolate the changed24-layer activation.
Full budget, no elimination; fixed2k reports. UE5a primary and UC1a backup,
EW4b excluded. Start from scratch under a distinct RUN after handoff.

Prior: final gap-.001 versus SiTU24, gain1.005 (plausible .985–1.025);
speed~.55–.56 steps/s versus parent's~.550 (unpaired). The point is to
remove the activation confound in the depth study, not to declare that
SiTU already explains its results.

Main/refactor-bam. Intended retained TPU xd-v5p-16-311042-maxtext; use new
launcher ID311043 with explicit retained TPU override. Preserve the old
RUN's actual checkpoint/TB/leases and mark it cancelled due to scope
correction after successful new FIRST_STEP. Artifact root:
`/data0/xd/bam_diagnostics/mediumprop-bam24-swiglu-20261115`.
