# Per-layer dedicated memory MLPs

Implementation worktree `/data0/xd/mediumprop-dedicated-m-mlp`, branch
`codex/mediumprop-dedicated-m-mlp`; forked current `refactor-bam` at781a86a6.
Parent: `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile`.

Separate ordinary residual SwiGLU and M-content SwiGLU, both consuming the same
post-attention normalized residual. The memory branch never enters the vector residual;
its output is reshaped to16x75. Preserve the parent's independent R256 GELU addresses,
per-head write gates (initial .1), attention writes, generalized reads and read-only M-shift .5.
No added write-amplitude scaling. M-cache75x32 is unchanged.
L0–L16 write memory; L17 has no memory branch/address/gate because it has no downstream reader.
The parent's otherwise unused L17 attention writer is preserved for a controlled comparison.

| RUN | Ordinary widths | Memory widths | Ordinary params | Memory params | MLP write controls | Total |
|---|---|---|---:|---:|---:|---:|
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile` | all3200 | L0–15:511, L16:505 |207360000|31251600|7786000|432113280|
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile` | L0–16:2535, L17:3806 | L0–16:1140 |168843600|69768000|7786000|432113280|

Both totals are parent+1600, MHA−7920; respectively +.001111/−.0055 W_Q (D²=1440000).
Ratio2:1 includes address/gate parameters: a body ordinary branch9126000 versus
memory4104000+controls458000=4562000, ratio2.000438. L17's freed update budget goes to
its ordinary MLP, so the aggregate ratio need not equal2. Exact integer width residuals
are included in the totals above. Body16 layers scan as8 identical two-layer blocks;
L16 and L17 compile separately for the last writer/terminal budget exceptions.

Both classes use the original T4096/global128/per-device16/13500 schedule, BF16,
full remat, scan, tuned Splash1024/fused backward and Pallas v8t/48MiB/fused inputs.
Generic training health ON; internal activations and BAM-specific health OFF;
checkpoint200/latest2/keep0, TensorBoard rotation500.
Infrastructure audit: source is the latest main781a86a6, preserving current train/AOT/WD,
checkpoint and complete rotating-writer path; validate summary-writer regression with CPU gate.
No model/numerical initialization changes beyond introducing a separately initialized branch.

Direct comparison: original single .5 parent only; matched MHA is the gain-ratio reference.
Report early each1000 steps with200-step loss windows; review2800/5000, plan13500.
Pre-run terminal gain-ratio bets: MHAWidth1.01, Ratio2to1 1.04 versus parent;
throughput bets −3%/−5%. Historical parent timings have different health/kernel settings,
so formal speed claims require a matched timing control.

Resource ownership: new v5p-16 IDs310317/310318, us-east5-a; no passive candidates.
Retained FLEX compiler llm-jax-v6e-1-0/europe-west4-a worker0 is borrowed, never released.
Independent launch preparations share its worker-level AOT lock (serial compiles).
Preparation artifacts: `/data0/xd/bam_diagnostics/mediumprop-dedicated-m-mlp`.

Sealed runtime18a9d9cf5e912ffbc5549e65746f261b722c28ec; both FIRST_STEP and step14 gates passed.
Startup throughput~.707step/s, with the declared metric/kernel settings. CPU gate:47 shared
BAM regressions +4 dedicated-path checks +5 summary-writer tests, all passed.
Both AOTs compiled serially on the retained worker and launched on their assigned TPUs.

## Shared hidden follow-up

RUN `BamMediumPropMShiftReadG050SharedHiddenMemoryMLPTruePile`, TPU
`xd-v5p-16-310319-maxtext`, UE5a. Same family worktree and retained compiler.
Share W1/Wg and the actual SwiGLU hidden activation; independent vector `wo` and
memory `memory_wo` output heads. L0–16 hidden2801, L17 hidden2794 with no memory head,
address, or gate. Both outputs use the same post-attention normalized input and hidden;
only vector `wo` contributes to the vector residual, memory `memory_wo` writes M.
Pallas v8t read/write interfaces and M-cache remain unchanged.

Exact parameter budget: 181479600 for shared inputs + vector output heads,
57140400 for memory output heads, 7786000 write controls. Total432121680:
MHA+480 (+.000333W_Q), original single.5 parent+10000 (+.006944W_Q),
both independent-branch arms+8400 (+.005833W_Q). Attribution of common W1/Wg to the
vector column here is bookkeeping, not a claim that they serve only that stream.

Main-only commits after fork781a86a6 through current maincb8ba616 contain ledger/startup
metadata only: no train/AOT/summary-writer/checkpoint/optimizer/default changes to port.
Same schedule, initialization recipe, per-write gate opening.1, kernel and health settings;
no new normalization or amplitude coefficient. CPU gate adds exact shared budget/tree,
explicit hidden/head reference and cross-head gradients, scanned causality and live memory
head gradients to the existing family checks and full shared BAM suite.

Compare direct independent2:1 arm plus original single.5 direction anchor; MHA common
reference for both gain ratios. Terminal bets:1.05 versus original parent, approximately
1.01 versus independent2:1 (consistent with that arm's1.04 prior); throughput+2% versus
independent2:1. Full13500 plan, reviews2800/5000, grouped reports every1000 early.

Shared-hidden sealed runtime89d6bec543615acbae3ea5115235e90b2e7b2c4b;59 CPU checks
(47 shared +7 family +5 summary writer) passed. AOT load, FIRST_STEP12 and step14
gates verified; startup~.71step/s under the declared health/kernel settings.

## Matched3:1 budget arm

RUN `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1TruePile`, TPU
`xd-v5p-16-310320-maxtext`, UE5a. Derived from independent2:1, with exactly the same
per-body-layer total and terminal3806: L0–16 ordinary2852/memory823; all other knobs
unchanged. Ordinary188244000, memory50367600, controls7786000, total432113280,
identical to independent2:1. Actual per-body-layer branch ratio10267200/3420800
=3.00140318. New arm compares independent2:1 and original single.5; MHA common
reference for both ratios. Terminal bet1.01 vs2:1 (~1.05 vs original parent, consistent
with the previous priors), throughput flat. Same13500 plan,2800/5000 reviews.

This is a configuration-only change on fully verified runtime89d6bec5 (59 checks).
Main updates through7c02f49c are ledger metadata only; no infrastructure fixes missing.
Use sealed inherited-config checks, the family's exact budget/settings CPU check and
full target AOT/FIRST_STEP gates; no repeat of unrelated shared regressions.

Matched3:1 runtimefaefbada: sealed inherited-config, exact family budget/kernel-settings
CPU check, target AOT load, FIRST_STEP8 and step14 gates verified. Startup~.700step/s
versus independent2:1 startup.707 (about−1%, same health/kernel settings).

## Half memory MLP

RUN `BamMediumPropMShiftReadG050HalfMemoryMLPTruePile`, TPU
`xd-v5p-16-310321-maxtext`, UE5a; same family worktree and retained FLEX compiler.
Independent W1/Wg, SwiGLU hidden1200=16x75 goes directly to the existing memory
content RMS/gate/outer-product path. No memory `wo` is created. Ordinary widths
L0–16:2875, L17:3806; no last-layer memory branch. Addresses/gates, initial gate.1,
attention writes, read-only M-shift .5, read recipe, Pallas/tuned Splash and health
settings unchanged. No new amplitude scaling.

Body budget: ordinary10350000 + two memory inputs2880000 + write controls458000
=13688000, identical to2:1 and3:1. Ordinary189651600, memory48960000,
controls7786000, model432113280 (MHA−7920, same as all independent full-memory arms).
Complete branch ratio3.100659:1, kernel-only3.59375:1. Compared with full3:1,
ordinary width2852→2875, memory feature width823→1200, no postactivation mixing.
Leading dense MAC count is unchanged because all saved projection weights fund
the ordinary MLP; total hidden activations grow, so fewer output GEMMs do not prove
whole-step acceleration. M-cache and the Pallas input/output shapes are unchanged.

Primary comparator full3:1; retain full2:1 and original single.5 as references,
with matched MHA as common gain-ratio reference. Bet terminal1.01 vsfull3:1,
throughput flat. Full13500 plan; reviews2800/5000 and grouped1000-step early reports.
Main-only updates throughca0a179f are ledger metadata, with no infrastructure fixes
missing. Run full shared regressions plus family half-tree/reference/gradient and
causality checks, summary-writer regression, sealed config, target AOT and FIRST_STEP.

Half-memory runtime884962320e2aa9b757a839924a6615950f7b98ec:62 checks passed
(47 shared +10 family +5 summary writer), AOT loaded, step14 gate and CP400 verified.
Stable20–99 median.702step/s,100–399 about.701, versus full3:1 startup.700 (flat).


## Compact rank8 shared-hidden writes

Two arms in the same implementation worktree/branch, new v5p-16 IDs310322/310323,
us-east5-a, no passive candidates. Retained FLEX compiler unchanged; independent
preparation jobs serialize AOT on its worker lock. Full13500 schedule, normal1000-step
group reports and2800/5000 reviews; all inherited Pallas/tuned Splash, health,
checkpoint, scan, dataset and initialization settings unchanged.

Both replace the shared-hidden parent's16x75 memory output and independent R256
GELU address/gate network with8 memory write heads, independent of16 attention heads.
Ordinary SwiGLU hidden h is shared; U has8x75 entries. RMS content/address normalization,
per-head sigmoid gates with bias logit(.1), additive write and original attention writes
remain. Direct controls consume x = the same post-attention normalized MLP input,
not raw residual or attention input. L17 has no memory branch/control parameters.
No new amplitude scaling compensates16→8 write heads. Per-token update rank is at most8
for this branch, unrelated to LocalQK read rank. Cache remains75x32. AOT uses16 scanned
body layers as8two-layer blocks and separately compiled L16/L17.

| RUN | Body hidden | Terminal | Body write projections | Total | MHA delta |
|---|---:|---:|---|---:|---:|
| `BamMediumPropMShiftReadG050SharedHiddenRank8WriteTruePile` |3114|2802|h→864 U/A/g factors|432122600|+1400|
| `BamMediumPropMShiftReadG050SharedHiddenRank8InputControlsTruePile` |3234|2808|h→600 U; x→264 A/g|432122168|+968|

Body budgets respectively4464*3114+264=13901160 and
4200*3234+1200*264+264=13899864. Bias264 is256 address plus8 gate entries;
content has no bias. Relative to shared-hidden432121680, integer residuals+920/+488
(+.000639/+.000339 W_Q); versus MHA+.000972/+.000672 W_Q. Terminal widths differ
from shared-hidden2794 only by8/14 to absorb accumulated integer rounding. Do not
silently redistribute its terminal budget to3806; that would introduce a large confound.
At equal H3114, x controls save(3114-1200)*264=505296 per body layer (.3509 W_Q),
which funds the wider ordinary hidden in the matched-total arm.

Joint arm compares shared-hidden parent and the independent2:1 direction leader;
x-control arm additionally compares the joint arm. MHA remains the gain-ratio reference.
Pre-run terminal bets vs shared-hidden1.02/1.03, x-control vsjoint1.01; speed flat.
Main-only changes from fork781a86a6 through2b6fb63a are ledger metadata; no missing
train/AOT/optimizer/checkpoint/writer infrastructure fixes. CPU gate covers shared suite,
formal full-tree budgets/no redundant controls, explicit projection references and gradient
sources, unequal-head scan causality/gradients, v8t unequal-group fwd/back and writer rotation.


Compact-write runtime47b0e1b0032d27851291ea7885243a0acda4af18:65 CPU checks
(47 shared +13 family +5 rotating-writer) passed; both sealed configurations match.
Both AOTs loaded and FIRST_STEP/step14 gates passed on the assigned UE5a v5p-16s.
Startup joint~.723step/s, x controls~.725, versus shared-hidden~.710 (about+1.8%/+2.1%,
same declared health/kernel settings). Registry gain triples use MHA as reference for
every comparator. Borrowed FLEX compiler remains intact and idle after both compiles.
