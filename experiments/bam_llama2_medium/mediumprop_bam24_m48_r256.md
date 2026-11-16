# L24 fixed-total SwiGLU: fixed-R256 matrix capacity pair

Both RUNs use main `/home/xd/projects/maxtext`, branch `refactor-bam`.
Parent is the completed `BamMediumPropV5L24TotalBudgetSwiGLUTruePile`,
runtime5a5dd84. The user explicitly requests a fresh R256 parent control;
do not substitute an18-layer control. Both arms preserve D1056,24 layers,
16x66 heads, all four independent address ranks256 and ordinary/private
SwiGLU. User confirmed66x48 rather than75x48 for the expanded arm.

| RUN | M | Ordinary/private/terminal widths | Total parameters | Delta MHA432121200 | Complete-branch ratio | Owned training TPU |
|---|---|---|---:|---:|---:|---|
|`BamMediumPropV5L24TotalBudgetSwiGLUR256ControlTruePile`|66x32|2201/968/3435|432120064|-1136|2.000532|`xd-v5p-16-311048-maxtext`|
|`BamMediumPropV5L24TotalBudgetSwiGLUM48R256TruePile`|66x48|2128/911/3319|432118976|-2224|2.000047|`xd-v5p-16-311047-maxtext`|

Expanded minus parent is-1088 parameters. Unrepaid enlargement adds9838720
parameters (8.823W_Q, D1056); charge each body layer locally while retaining
approximately2:1 ordinary/complete private-branch parameters. Body layers
are parent+848 each; terminal ordinary MLP closes embedding/terminal and
integer residue. No new per-layer-width mechanism. Single-layer scan23+tail;
terminal attention/private M writes remain off. Reuse packed fused v8t reads
and writes,48MiB VMEM, full remat and tuned Splash. No kernel source changes.

Same confirmed13500-step/7077888000-token budget, T4096/global128,
TruePile4097 records, seed9876, original Prop optimizer and LR endpoint.
Generic health ON/internal activation OFF, checkpoint200/latest2/final,
no permanent keep-period, TBrotate500. UE5a primary and UC1a backup after
300s, EW4b excluded. Full budget, no elimination, both arms reported
together on a fixed2000-step calendar. The independent XL capacity RUN
continues under its own calendar.

Expanded arm compares to the fresh32-column control, using the matched
`Llama2MediumPropL24TotalBudgetTruePile` MHA as native gain reference.
The fresh control compares to its completed identical parent: this checks
whether current runtime reproduces the historical depth point before
attributing differences to capacity. Keep that reproduction comparison
separate from the primary capacity effect.

Completed18-layer75x48/R384 (`BamMediumPropV5SiTU47M48R384`, runtime2458bb3)
yielded final-five gap-.00473824/gain1.026518. It used ordinary SiTU and
R384, so it is historical context rather than a clean fixed-R cross-depth
control. The new pair isolates32-to48 columns at24 layers and fixedR256,
testing whether matrix capacity partly limits the deeper model's return.
Prior: final capacity gain1.025 (1.00--1.06), speed about-4% versus parent;
parent replay should agree within BF16/compiler noise. These are scaling
controls, not a SOTA elimination campaign.

Audit current main75099911 against actual parent5a5dd84: model, kernel,
training/AOT/checkpoint/optimizer/logger source files are unchanged; only
experiment definitions/documentation differ. Focused CPU gates cover both
full-sized parameter trees and complete branch budgets, actual M66x48
native/Pallas feedback plus all parameter gradients, and logger rotation.
Exact target AOT gates TPU layout/VMEM. Require both worker source hashes,
AOT load beacons and FIRST_STEP. Borrow ACTIVE FLEX `llm-jax-v6e-1-0` in
EW4a for compilation only; its lifecycle is outside these training RUNs.
Artifacts: `/data0/xd/bam_diagnostics/mediumprop-bam24-m48-r256-20261116`.
