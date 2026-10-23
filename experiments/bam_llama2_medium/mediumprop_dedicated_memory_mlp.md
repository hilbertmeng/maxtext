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


## Half memory MLP closeout

`BamMediumPropMShiftReadG050HalfMemoryMLPTruePile`, runtime
884962320e2aa9b757a839924a6615950f7b98ec, stopped2998 after its2800 review.
It trailed full3:1 from400 onward; last five common windows2000/2200/2400/2600/2800
average gap+.0088078, MHA-reference gain.96927780 (latest2800 +.0087062/.96637240).
Over those same windows, vs independent2:1 +.01227432/.95772495;
vs original single.5 +.00965984/.96625229. Its early advantage over original parent
vanished around1200 and turned into a growing deficit. Pre-run terminal prior1.01
versus full3:1 was unsupported through the review stop.

Steady~.701step/s versus full3:1~.700 (flat), with matching training health ON,
internal/BAM health OFF and Pallas/tuned Splash. Same432113280 parameters and75x32
M-cache as full3:1. Memory nonlinear hidden823→1200 (+45.8%), ordinary2852→2875
(+.81%), but removing the content output projection worsened quality. This supports
retaining learned postactivation content mixing in the next compact-write designs.

v5p-16 xd-v5p-16-310321-maxtext stayed in us-east5-a, no passive candidate or
preemption. Assignment07:13:51–08:29:47 UTC on2026-10-23. One full READY lease:
07:13:43–08:29:47,1h16m04s, ending by review stop. The first timestamp is the
parallel prequeue's READY node observation; registry covers07:13:54–08:29:47,
1h15m53s,11s shorter. CP2998 commit_success and independent node/queue NOT_FOUND
verified; local TB SYNC_OK after one bounded retry. Borrowed FLEX compiler retained.

Reproduction/proof: `/data0/xd/bam_diagnostics/mediumprop-dedicated-m-mlp/`
`half-final-report.json`, `half-final-proof.json`, `half-lease-final.json`,
`closeout-half.json`, `review-half2800-shared5k.png/.csv`. Parameter/configuration
and startup details above remain the runtime specification.


## MHA-width and shared-hidden closeout

Both stopped by user. MHA-width runtime18a9d9cf5e912ffbc5549e65746f261b722c28ec
stopped7849: its initial parent advantage vanished, and5k–7.8k showed a persistent
loss deficit without late catch-up. Last five7000/7200/7400/7600/7800 windows average
+.00285040, MHA-reference gain.98381510 vs original single.5. Giving the ordinary
MLP the MHA width while reducing the dedicated M network to511/505 did not improve
the SOTA parent. Total parent+1600 (.001111W_Q), unchanged75x32 M-cache;
startup~.707step/s, with parent timings incomparable because of health/kernel settings.
Its terminal1.01 prior was unsupported through the stop.

Shared-hidden runtime89d6bec543615acbae3ea5115235e90b2e7b2c4b stopped6264.
It initially beat independent2:1, lost that advantage around2.8k, then persistently
trailed original single.5 from about4k. Last five5400/5600/5800/6000/6200:
vs2:1 +.00357544/gain.98130088; vs original +.00270456/gain.98578882.
Its early learning benefit did not persist. Startup~.710 vs2:1~.707 (+.4%),
parameters+8400 (.005833W_Q), same cache and matched metric/kernel settings;
those small deltas do not justify the quality deficit. Initial priors1.05 vs
original/1.01 vs2:1 were unsupported through the stop. Cached trajectory remains
available to the rank8 descendants; independent2:1 continues as the full-run control.

Both stayed on v5p-16 in us-east5-a, no passive candidates or zone switches,
one preemption each. CP7849/CP6264 committed, independent TPU+queue NOT_FOUND
verified, local TB SYNC_OK. Retained FLEX compiler untouched.

| RUN suffix | READY start UTC | End UTC | Full duration | End reason |
|---|---|---|---|---|
|MHAWidth lease1|2026-10-23 05:28:56|2026-10-23 06:31:51|1h02m55s|preempted; first prequeue READY, registry starts9s later|
|SharedHidden lease1|2026-10-23 06:12:03|2026-10-23 06:31:53|19m50s|preempted; first prequeue READY, registry starts11s later|
|MHAWidth lease2|2026-10-23 06:37:36|2026-10-23 08:54:47|2h17m11s|user stop7849|
|SharedHidden lease2|2026-10-23 06:37:29|2026-10-23 08:56:44|2h19m15s|user stop6264|

Assignments: MHAWidth05:29:03–08:54:47, SharedHidden06:12:11–08:56:44 UTC.
Artifacts under `/data0/xd/bam_diagnostics/mediumprop-dedicated-m-mlp/`:
`stop-pair-proof.json`, `stop-pair-registry-final.json`, `stop-pair-final-report.json`,
`stop-pair-final.png/.csv`, `closeout-mha-width.json`, `closeout-shared-hidden.json`.
Metrics use final native canonical loss caches, not hand-computed gain ratios.


## Rank8 write closeout

Both runtime47b0e1b0032d27851291ea7885243a0acda4af18, stopped after2800 review:
joint-h3005, direct-x-controls2983. Last complete mature window remains2800.

| Arm | Comparator | Final5 windows | Mean gap | MHA-reference gain |
|---|---|---|---:|---:|
|Joint h|independent2:1|2000..2800|+.00218924|.99240501|
|Joint h|shared-hidden|2000..2800|+.00376248|.98737491|
|Input controls|independent2:1|2000..2800|+.00386316|.98642332|
|Input controls|shared-hidden|2000..2800|+.00543640|.98140964|
|Input controls|joint h|2000..2800|+.00167392|.99397106|

Joint-h's early advantage over independent2:1 disappeared around1800, then did not
recover. Input controls had a transient win over joint-h at2000, lost the next four
windows and fell further behind2:1. Both also trail the already failed shared-hidden
parent at matched steps. Slight startup acceleration (.723/.725 vs~.710, about+2%),
negligible integer parameter residuals (+920/+488 vs shared-hidden), and unchanged
75x32 cache do not compensate for the quality deficits. Initial terminal bets1.02/1.03
vs shared-hidden and input/joint1.01 were unsupported through the review stop.

The input-control arm uses direct linear x→address/gate projections; the parent's
address controller was independent R256+GELU. Parent comparisons change both the
number of write records16→8 and the controller, so they do not isolate head count.
Under the matched-total budgets tested, joint nonlinear h controls were better than
direct x controls despite the latter's wider ordinary MLP. Next candidates should
preserve16 records and proven nonlinear address control while considering a cheap,
M-specific nonlinear decoder after shared features; preserving learned content mixing
is also supported by the half-MLP failure.

Both v5p-16s stayed in us-east5-a, zero preemptions, no passive candidates or zone
switches. CP3005/CP2983 committed, independent node+queue NOT_FOUND verified,
local TB SYNC_OK. Assignments joint07:59:41–09:13:44, input08:00:02–09:13:48 UTC.
One full READY lease each on2026-10-23:

| Arm | Start UTC | End UTC | Duration | End reason |
|---|---|---|---|---|
|Joint h|07:59:32|09:13:44|1h14m12s|review stop3005; first prequeue READY, registry starts12s later|
|Input controls|07:59:37|09:13:48|1h14m11s|review stop2983; first prequeue READY, registry starts28s later|

Evidence under `/data0/xd/bam_diagnostics/mediumprop-dedicated-m-mlp/`:
`rank8-final-proof.json`, `rank8-registry-final.json`, `rank8-final-report.json`,
`closeout-rank8.json`, `review-rank8-2800.json`, and final charts. Retained FLEX compiler
remains intact. Independent2:1 and3:1 continue; failed parents are cached comparisons.


## Independent3:1 closeout

`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1TruePile`, runtime
faefbadaf669ce19744430a525a4c9b9f2d8b1a9, stopped7795 after the post5k extension.
Its early deficit to2:1 narrowed, but from about4k its MHA-reference gain plateaued
near.992. It trailed2:1 at every mature400..7600 window. Final five
6800/7000/7200/7400/7600 average+.00150608/gain.99159309 vs2:1;
vs original single.5 +.00039436/gain.99777083. Ordinary MLP parameters+11.5%
vs2:1 did not yield a later relative catch-up. Initial terminal prior1.01 vs2:1
was unsupported through the review stop. No speed/parameter/cache benefit:
startup~.700 vs.707 (about−1%), identical432113280 parameters and75x32 cache.
Same training-health ON, internal/BAM metrics OFF, Pallas and tuned Splash.

v5p-16 xd-v5p-16-310320-maxtext stayed in us-east5-a, no passive candidates,
one preemption. Recovered from committed4747 and verified next periodic4800.
Assignment2026-10-23 06:27:52–09:52:46 UTC. Full chronological READY leases:

| Lease | Start UTC | End UTC | Duration | End reason |
|---|---|---|---|---|
|1|2026-10-23 06:27:44|2026-10-23 08:28:16|2h00m32s|preempted; first prequeue READY, registry starts11s later|
|2|2026-10-23 08:35:31|2026-10-23 09:52:46|1h17m15s|later review stop7795|

CP7795 committed, independent node+queue NOT_FOUND verified, local TB SYNC_OK;
retained FLEX compiler untouched. Artifacts under
`/data0/xd/bam_diagnostics/mediumprop-dedicated-m-mlp/`:
`closeout-ratio3.json`, `ratio3-final-proof.json`, `ratio3-registry-final.json`,
`ratio3-final-report.json`. Independent2:1 is the remaining live arm.


## Independent rank8 / R128 / hidden768

RUN `BamMediumPropMShiftReadG050DedicatedMemoryMLPRank8R128H768TruePile`,
new v5p-16 ID310324, us-east5-a, same isolated family worktree/branch and retained
FLEX compiler. Direct parent/control independent2:1 only; matched MHA is the
native gain-ratio reference. Compare at common steps, report1000 early, review2800/5000,
full13500 endpoint. Initial terminal prior1.02 versus independent2:1, throughput flat.

Preserve independent M-only SwiGLU input/gate matrices, both taking the same
post-attention normalized x as the ordinary MLP. Memory hidden768 outputs600=8x75
through learned W2. Addresses retain the independent GELU bottleneck, now R128,
producing8x32; write gates remain x→8 plus bias. Reader, attention and embedding
projections retain their parent settings. MLP writes L0–16 only; no final memory
branch. Ordinary widths3108 at L0–16,3801 at L17. Existing scan8two-layer blocks
plus separately compiled L16/L17, Pallas v8t/48MiB/fused inputs and tuned Splash
are retained. M-cache75x32 unchanged; no Pallas operator changes.

Exact counts: ordinary203893200, M-MLP39168000, write controls3335944,
model432112824 (parent−456=−.000317W_Q, MHA−8376=−.005817W_Q).
Per writing layer ordinary11188800; memory kernels2304000 + address186624 +
gates9608 =2500232. Complete branch ratio4.47510471:1; aggregate4.79704189:1
including the terminal vector MLP. Content nonlinear width/output ratio768/600=1.28,
versus parent1140/1200=.95; the higher vector/M parameter ratio accompanies fewer
write records and does not by itself measure features available per record.

Only the M-MLP write-gate bias opens at.1*sqrt(2)=.1414213562, approximately
matching16-head parent write-increment RMS under independent outer products.
Attention/embedding gate initialization remains.1; BAM kernels retain their.006
recipe and existing factor RMS epsilon. Dense MAC budget is effectively constant;
MLP write outer products halve, nonlinear hidden count changes, so no guaranteed
whole-step speedup. Generic training health ON; internal/BAM health OFF,
checkpoint200/latest2/keep0, TensorBoard rotation500, full remat/T4096/global128.

Infrastructure audit through main7f9c373b: main-only changes since fork781a86a6
are ledger/closeout metadata, no missing train/AOT/optimizer/checkpoint/writer fixes.
CPU gates cover full shared regressions, private8-head full-tree budgets and controls,
content projection reference/live gradients, scanned causal gradients/gate initialization,
legacy family behavior, unequal-head v8t forward/back and rotating-writer regressions.

Startup verified2026-10-23: sealed runtime`eb76af1`, full CPU68 checks and retained
FLEX AOT passed; registry/auto environment agree on commit, parent,13500 schedule
and zone-local truepile4096 dataset. Worker loaded compiled executable, FIRST_STEP
and step32 passed; startup~.729step/s versus parent~.707 (+3.1%), matched health.
Preparation proof: `/home/xd/.local/state/maxtext-parallel-launch/`
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRank8R128H768TruePile-20261023T103551Z/`.


## Independent2:1 completed13500

`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile` completed its
13500-step schedule on runtime18a9d9cf5e912ffbc5549e65746f261b722c28ec.
Against original single.5, the large initial convergence lead faded to a small
positive platform:4.2k–6k mean gain1.005809,6.2k–8k1.005447,
8.2k–10k1.007384,10.2k–12k1.007235. The last five12600..13400 windows
average gap−.00143288, MHA-reference gain1.00921392. The12.2k–13k mean
1.008724 and terminal1.009214 support a slight late strengthening; they do not
recover the original1.04 prior or2k working1.02 forecast. Direction was right,
magnitude optimistic. Last point13400 gain1.014751 is not the five-window endpoint.

Total432113280 (single.5+1600=.001111W_Q), unchanged75x32 M-cache;
ordinary L0–16 width2535, private M-MLP1140, terminal ordinary3806.
Startup~.707step/s, later~.704; Pallas v8t/tuned Splash, generic health ON,
internal/BAM OFF. Historical single.5 runtime has different health/kernel
settings, so its speed is not a matched architectural timing comparison.

v5p-16 xd-v5p-16-310318-maxtext stayed in us-east5-a, no passive candidates,
zero preemptions. Assignment2026-10-23 05:29:47–11:02:11 UTC.
Full READY lease:2026-10-23 05:28:35–11:02:11 UTC,5h33m36s, ending at
planned completion and verified resource release. First READY is the prequeue
log; registry observation begins05:29:49,74seconds later. Training itself
exited cleanly11:00:03, followed by automatic resource deletion.

CP13500 commit_success and node/queue NOT_FOUND were independently verified;
local TB SYNC_OK. Follow-up closeout returned already_closed=true and made no
resource mutations. Retained FLEX compiler preserved. Evidence under
`/data0/xd/bam_diagnostics/mediumprop-dedicated-m-mlp/`: ratio2-final-proof.json,
ratio2-final-report.json, ratio2-registry-final.json, closeout-ratio2.json,
ratio2-auto-completion.log. Independent rank8/R128/768 remains in training.
