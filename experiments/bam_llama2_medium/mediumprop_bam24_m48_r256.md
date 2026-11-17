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

## Primary depth-scaling interpretation

The scientific question is whether BAM uses depth better than Mudd, and whether
matrix capacity explains BAM's earlier depth-gain flattening. BAM32 is a necessary
internal depth control alongside Mudd. Compare full12/18/24 curves, not only the
new24-layer configurations' local ranking. Existing18-layer M48/R384 is the
large-matrix depth anchor; no new18-layer RUN is submitted.

Use matched same-depth MHA gains G_X(d)=loss_MHA(d)-loss_X(d), from native
200-step windows. Report each architecture's absolute G(24)-G(18), its differences
against BAM32 and Mudd, and both normalized depth effects:

- Capacity: K_capacity=[G_B48(24)/G_B48(18)]/[G_B32(24)/G_B32(18)].
- Relative to Mudd: K_mudd=[G_B(24)/G_Mudd(24)]/[G_B(18)/G_Mudd(18)].

Values above1 indicate an improvement in the corresponding relative depth
response; absolute and relative effects answer complementary questions. Gains
at both depths do not by themselves imply a better depth response. R384 is the
fixed-rank48-column depth comparison; R256 at24 layers additionally tests a
cheaper rank. Keep BAM18 SiTU versus BAM24 SwiGLU visible in figure notes.

The context plot anchors all losses to MHA18 so MHA's own depth behavior remains
visible; its static panel uses the same five mature windows for every curve.
Other panels retain cumulative training-stage sequences. Mudd controls are
router-fixed plain12, post-only18 and post-only24, following the completed norm
controls; their router/norm overhead retains the authorized unrepaid convention.

Native monitoring keeps all existing capacity/depth comparisons and adds
same-depth post-only Mudd24 for both new RUNs. This changes monitoring only;
runtime commits, AOT artifacts and model parameters remain unchanged.
The task's `collect_depth_family.py`, `plot_depth_family.py` and `report_group.py`
under its artifact root reproduce this full context automatically at subsequent
shared reports. Loss/gain windows come from run_registry; derived depth effects
compose native paired-gain reports rather than reimplementing gain arithmetic.

## R384 closeout and depth-specific rank selection

User-stopped R384 at committed checkpoint5579/P0. Both candidate regions'
node/queue identities are independently absent; seed9876/global128/two-host
cursor and regional data filenames match. TensorBoard returned SYNC_OK.
One UE5a READY lease11:01:31--14:03:07 UTC on2026-11-16 (3h01m36s), assigned
11:01:23. The user stop is not a preemption. The physical TPU was retained
from the cancelled pre-start control; that earlier preparation boundary is
recorded separately in the regional history. The FLEX compiler is preserved.

Final five native200-step windows4600--5400: R384 versus R256 gap+.00567972,
MHA24-referenced gain.9767886. Absolute deficit narrowed from the early
transient, but relative gain retention stayed near.98, then weakened; there
was no sustained relative catch-up. Versus original24-layer M32 parent,
gap-.00213384/gain1.008968; versus18-layer M48 gap-.00059840/paired gain1.026055;
versus post-only Mudd24 gap-.10872420/gain1.831352. Thus a small absolute
capacity benefit did not offset the larger rank's budget cost against R256.
The terminal1.04 prior was not observed to its13.5k horizon; the partial
result at stop was substantially smaller. Startup speed.524 versus parent
.546 (-4.0%, unpaired), matching the approximate speed prior.

Depth scaling need not force identical rank across depths, just as the Mudd
comparison uses depth-appropriate norm choices. Continue the stronger24-layer
M48/R256 against the existing18-layer M48/R384 and the BAM32/Mudd controls.
Do not retain a weaker training arm solely to maintain identical rank.
The stopped R384 data remain archived; exclude it from future shared-window
intersections so its5.4k horizon cannot freeze the continuing scaling plot.
R256 remains on the original13500-step plan, fixed2k reports, no scheduled
elimination reviews. Reproduction: `r384-final-proof.json`,
`r384-closeout-local.log`, `r384-direct-review/`, and `k6000/` in the task root.


## R256 completed budget and depth result

Completed 13,500 steps at runtime a03c7ea. Checkpoint 13,500 has its commit
marker and matching seed 9876, global batch 128, two-host cursor and UE5a
TruePile files. Both registered regions' node and queue identities are absent;
TensorBoard returned SYNC_OK. One preemption, recovered in UE5a from committed
checkpoint 11,379 with unchanged source/AOT and verified next checkpoint 11,400.
The user-owned FLEX compiler remains outside training cleanup.

Final five native windows, 12.6k--13.4k:

| Direct comparison | Loss gap | Gain multiplier |
|---|---:|---:|
| L24 M66x32 / R256 parent | -0.00704888 | 1.03857059 |
| L18 M75x48 / R384 | -0.00810436 | 1.03564416, matched MHA24/MHA18 |
| L24 post-only Mudd | -0.08433152 | 1.79913314 |

After the initial transient, the parent gap stayed near -0.007 while relative
capacity gain gently rose to approximately 1.04. The 1.025 terminal prior was
conservative; startup speed .526 versus the historical parent's .546 (-3.7%,
unpaired) matched the approximate -4% prior. M48 has 1.5 times the parent's
matrix-cache elements at the same head dimension.

The same five windows give G(24)-G(18) = 0.006531 for BAM48, versus 0.01388144
for Mudd. Enlargement adds 0.00231064 to BAM32's depth response. Mean
K_capacity = 1.01174697 and K_mudd = 0.89944813: capacity helps the absolute
loss and modestly improves depth response, but does not explain or resolve
BAM's weaker additional depth gain. BAM remains substantially better in
absolute loss than Mudd; this is distinct from which architecture benefits
more from 18 to 24 layers. Retain the SiTU18/SwiGLU24 qualification.

All READY leases, UTC on 2026-11-16, for v5p-16 /311047:

| Zone | Start | End | Duration | Exit |
|---|---|---|---|---|
| UE5a | 10:42:23 | 16:52:41 | 6h10m18s | preempted |
| UE5a | 16:59:01 | 18:13:33 | 1h14m32s | scheduled completion |

The single active-zone assignment spans 10:42:16--18:13:33 UTC; UC1a was a
passive backup, with no zone switch. Final proof and full native series are
`r256-final-proof.json` and `final13500/` under the task artifact root.
The completed Medium group is archived; XL capacity training continues.
