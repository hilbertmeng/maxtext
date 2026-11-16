# L24 fixed-total SwiGLU: M66x48 at R256 and R384

Both RUNs use main `/home/xd/projects/maxtext`, branch `refactor-bam`.
Parent is the completed `BamMediumPropV5L24TotalBudgetSwiGLUTruePile`,
runtime5a5dd84. The initially submitted fresh parent control was redundant: its historical
parent already uses R256. It was cancelled before accepted FIRST_STEP; use
the completed parent directly. No new18-layer RUN is needed. Both active
capacity arms preserve D1056,24 layers,16x66 heads and ordinary/private
SwiGLU. User confirmed66x48 rather than75x48 for the expanded arm.

| RUN | M | Ordinary/private/terminal widths | Total parameters | Delta MHA432121200 | Complete-branch ratio | Owned training TPU |
|---|---|---|---:|---:|---:|---|
|Cancelled `BamMediumPropV5L24TotalBudgetSwiGLUR256ControlTruePile`|66x32|2201/968/3435|432120064|-1136|2.000532|`xd-v5p-16-311048-maxtext`|
|`BamMediumPropV5L24TotalBudgetSwiGLUM48R256TruePile`|66x48|2128/911/3319|432118976|-2224|2.000047|`xd-v5p-16-311047-maxtext`|
|`BamMediumPropV5L24TotalBudgetSwiGLUM48R384TruePile`|66x48|1968/757/3075|432121088|-112|2.000667|retained `xd-v5p-16-311048-maxtext`, launcher311049|

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
300s, EW4b excluded. Full budget, no elimination, active arms reported together
on a fixed2000-step calendar. The independent XL capacity RUN
continues under its own calendar.

Expanded arm compares to its completed32-column parent, using the matched
`Llama2MediumPropL24TotalBudgetTruePile` MHA as native gain reference.
There is no model, seed, schedule or numerical-code difference that justifies
retraining the cancelled parent control. Its brief preparation and cleanup
remain recorded separately for resource accounting.

Both active arms also compare to completed18-layer75x48/R384
(`BamMediumPropV5SiTU47M48R384`, runtime2458bb3), final-five parent
gap-.00473824/gain1.026518. It used ordinary SiTU; retain this qualification
in the depth interpretation. R384 holds address rank fixed across depths;
R256 tests a cheaper address budget at the same enlarged matrix capacity.
Native paired gains24/18 use each depth's own MHA reference. Keep these
alongside the native gain against the completed24-layer32-column parent;
R384 also directly compares to R256. Overlay both active arms for shared
comparisons. This is a depth/capacity scaling study, not SOTA elimination.
Prior: final capacity gain1.025 (1.00--1.06) for R256,1.04 (1.01--1.09)
for R384; speed approximately-4%/-5% versus the32-column parent.
R384 repays the33,493,120 unrepaid added parameters (30.035W_Q), with body
layers parent+1424 each and terminal MLP closing the residual budget.

Audit current main75099911 against actual parent5a5dd84: model, kernel,
training/AOT/checkpoint/optimizer/logger source files are unchanged; only
experiment definitions/documentation differ. Focused CPU gates cover both
full-sized parameter trees and complete branch budgets, actual M66x48
native/Pallas feedback plus all parameter gradients, and logger rotation.
Exact target AOT gates TPU layout/VMEM. Require both worker source hashes,
AOT load beacons and FIRST_STEP. Borrow ACTIVE FLEX `llm-jax-v6e-1-0` in
EW4a for compilation only; its lifecycle is outside these training RUNs.
Artifacts: `/data0/xd/bam_diagnostics/mediumprop-bam24-m48-r256-20261116`.

## Accepted startup and retained-resource handoff

R256 runtimea03c7ea, TPU311047/UE5a: both workers load the exact AOT/source,
FIRST_STEP at2 and later233-step timing proof. Median .526 step/s over
133--233 (200 worker samples), -3.7% versus historical parent .546.
R384 runtime1a3b1d0, launcher311049 on retained TPU311048/UE5a: both workers
load the exact AOT/source, FIRST_STEP and step14 verified, later step104.
Median .524 over77--176 (200 worker samples), -4.0% versus parent and
approximately-.4% versus R256. Timings are unpaired with matched health.

R384's M66x48 all-parameter native/Pallas gradient check maxrel5.31e-6;
full-budget recheck and exact v5p-16 AOT pass with48MiB VMEM. The initial
abandoned test wrapper mistakenly invoked absent pytest after its numerical
unittests passed; logger regressions subsequently ran correctly as unittest
and passed5/5 in the R256 preparation. No model/kernel source changed.

The cancelled duplicate parent never produced an accepted FIRST_STEP or
checkpoint. Both workers were verified without trainer/installer processes,
its controller was dead, and registry ownership was paused at10:59:04Z.
Its already-READY UE5a TPU was retained, then reassigned to the fresh R384
RUN under its distinct GCS prefix. Only after R384 FIRST_STEP was confirmed
was the old logical RUN finalized stopped, without inventing a checkpoint
or deleting the transferred TPU. Its observed prequeue READY start10:42:07Z
and cancellation boundary are in the regional history. The physical lease
continues under R384; neither this handoff nor the preparation interval is
a preemption. The retained user FLEX compiler is preserved.
