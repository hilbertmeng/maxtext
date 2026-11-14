# MediumProp depth controls including token tables

Prepared on main `refactor-bam`, `/home/xd/projects/maxtext`. Configuration-only
delivery: no new RUN, TPU allocation or AOT has been submitted.

## Budget and geometry

Match the original L18/D1200 MHA's **432121200 total parameters**, including
both untied vocabulary 50432 tables. Retain 16 query/KV heads, and choose D in
multiples of 16 to keep MHA attention:MLP as close as possible to 1:2. Choose a
uniform integer MHA MLP width at that D; no per-layer width mechanism is added.

The standing Mudd convention still applies: match the complete MHA MLP weight
budget, retain the rounded depth-varying MLP profile, and leave router/norm
overhead additional. The user's total-budget instruction does not revoke this
exception. BAM includes every embedding-to-M projection, read/write control,
norm and learned scale in its budget; ordinary:complete memory branch remains
approximately 2:1. Integer-width residuals are recorded, not hidden.

| Depth | D / head | Token tables | MHA uniform MLP | MHA total / residual vs432121200 | BAM ordinary / memory / tail | BAM total / residual |
|---|---|---:|---:|---:|---|---:|
|12|1424 /89|143630336|3728|432111376 /-9824|2990 /1373 /4600|432120776 /-424|
|18 reference|1200 /75|121036800|3200|432121200 /0|2510 /1128 /3904|432112228 /-8972|
|24|1056 /66|106512384|2874|432133152 /+11952|2201 /968 /3435|432120064 /-1136|

MHA attention:MLP is 1:1.963483 at 12 and 1:2.041193 at 24. Maximum base-budget
residual is .00277%. BAM complete ordinary:memory ratios are 2.000707 and 2.000532.
Tables count both input and output matrices, not one shared matrix.

| Mudd depth | Norm flags | MLP base / corrected tail | Actual total | Router/norm overhead over same-depth MHA |
|---|---|---|---:|---:|
|12|both off|3728 /5568|433257201|1145825|
|18 repaired reference|both off|3200 /4864 (original profile unchanged)|433911307|1790107|
|24|both on|2874 /4208|434683717|2550565|

## Router correction

The legacy fallback applies expand 4 when history length is L-1. With full
history this is the QKVM router **before layer L-2**, not final readout. For L18,
it produces hidden 320 before L16 and hidden 64 at final readout.

Use the existing `mudd_router_hidden_expand = [1] * (L-1) + [4]`, whose index is
`history_length - 2`. Every in-layer router uses expand1 and final readout uses
expand 4. Keep native strict next-64 rounding and the final router's actual C=1.
Thus final hidden widths are 64/128/128 at L12/L18/L24, respectively. Expand 4
does not mean artificially giving the final router four output branches.

The repaired L18 has hidden 128 before L16 and hidden 128 at final readout;
its only changed parameter shapes are the down/up kernels at those two sites.
Its total is 165440 below the old 434076747, with no MLP compensation because
router overhead has always been additional. Do not change the fallback globally:
historical classes and checkpoint shapes keep their recorded semantics.

## Classes and comparisons

MHA:

- `Llama2MediumPropL12TotalBudgetTruePile`
- `Llama2MediumPropL24TotalBudgetTruePile`

Mudd:

- `MuddLlama2MediumPropL12TotalBudgetRouterFixedTruePile`
- `MuddLlama2MediumPropRouterFixedTruePile`
- `MuddLlama2MediumPropL24TotalBudgetRouterFixedPrePostNormTruePile`

BAM:

- `BamMediumPropV5SiTU47L12TotalBudgetTruePile`
- `BamMediumPropV5SiTU47L24TotalBudgetTruePile`

All inherit the Prop recipe, T4096, seed 9876, global batch 128 on v5p-16,
a 13500-step schedule, full remat, generic health enabled and internal health
disabled. Tuned fused-backward Splash is inherited. This is a fixed-total-model,
fixed-data-exposure depth study, not a new Chinchilla budget calculation.

BAM retains ordinary SiTU 4/7, memory SwiGLU, M pre-norm, independent VO,
unit-initialized QK/VO scales, read-only M-shift .5, and negative attention
diagonal. M89x32/66x32 and full NoPE 89/66 combine with standard RoPE 22/16.
All four low ranks remain 256. Each scans its uniform L-1 prefix one layer at a
time; neither attention nor memory MLP writes M at the explicit terminal.
The embedding fixed divisor retains the cross-width sqrt(D/1200) rule.

Existing v8t packed/fused read/write kernels are enabled, VMEM 48MiB. Both new
K widths and concatenated QK111/82 fit the existing 128-wide padded blocks.
CPU native/Pallas full-model gradient checks cover the new K/RoPE widths;
formal TPU AOT compilation and startup validation are deferred until a launch
is requested. No kernel implementation change is introduced here.

Use matched-depth MHA loss gaps, their depth differences and BAM/Mudd gain
ratios. Mudd12/18 are plain, 24 is pre/post norm, per the user's requested strong
depth-specific controls. Keep the norm choice explicit when interpreting depth
effects. The repaired Mudd18 compares against the historical router-bug parent
and MHA18, permitting direct measurement of the router correction.

## Validation and paused predecessors

Full-size abstract parameter trees and the focused CPU regression are recorded
under `/data0/xd/bam_diagnostics/mediumprop-depth-total-20261115/`.
Regression entry: `MaxText/tests/mediumprop_total_budget_depth_test.py`.
All four focused tests passed. The two native/Pallas full-model probes checked
all 104 gradient leaves, including both learned read scales: worst per-leaf
relative errors were 3.90e-6 (K89/RoPE22) and 5.60e-5 (K66/RoPE16).
Full-size abstract trees verified all seven counts and the corrected router
placement. Neither model code nor kernel code changed.

User-paused old norm controls (provisional, not full-budget conclusions):

- `MuddLlama2MediumPropL12AttentionMLP1to2PrePostNormTruePile`: CP13071,
  P0, UE5a v5p-16 /311030, about .91 step/s. Early -.05874 at 600 versus plain
  parent reversed; last five 12200--13000 average +.017155.
- `MuddLlama2MediumPropPrePostNormTruePile`: CP11019, P0, UE5a v5p-16 /311031,
  about .69 step/s. Early -.06254 at 600 reversed; last five 10000--10800
  average +.007820 versus plain parent.

Both committed checkpoint markers/cursors, local TB sync and absence of all
regional nodes/queues were independently verified. Registries are `paused`,
not eliminated. Each had one READY lease: 12-layer 12:14:54--16:22:18 UTC
(4h07m24s); 18-layer 11:49:35--16:22:21 UTC (4h32m46s), 2026-11-14.
Evidence: `pause-norm12-proof.json`, `pause-norm18-proof.json`, `pause-receipt.json`.
