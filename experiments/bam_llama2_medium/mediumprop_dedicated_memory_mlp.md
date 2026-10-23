# Per-layer dedicated memory MLPs

Implementation is available on `refactor-bam` (merged2026-10-24 JST).
Original experiment worktree `/data0/xd/mediumprop-dedicated-m-mlp`, branch
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


## Dense2:1 plus native Mudd combination race

Three new RUNs extend this isolated family with the existing native shared Mudd
implementation from `codex/mediumprop-mudd-tail` (4db9b15 model lineage),
retaining one shared D→64 down per site, identity initialized joint up/bias,
L2 residual anchors and final C1/expand4 readout. Original Mudd rounding gives64
for both internal and final controllers. The ordinary MLP and dedicated M-MLP
consume the same post-attention normalized residual, including the Mudd m route
at selective sites. Private writers retain16 records, addressR256 and gate.1.
No rank8/R128/gate-rescaling changes are included in this race.

Prefix `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1Mudd`:

| Suffix | TPU ID / zone | Ordinary L14–17 | Private M-MLP L14–17 | Total / exempt routers / non-router rounding vs2:1 | Old tail direct control | Terminal gain bet |
|---|---|---|---|---|---|---|
|QKVML2TruePile|310325 / us-east5-a|2535/2535/3802/3030|1140/1140/0/0|432268180 /154900 /0|old L2 QKVM|1.010|
|VML2TruePile|310326 / us-east5-a|2535/2535/2535/3544|1140/1140/1140/0|432268716 /154380 /+1056|old L2 VM|1.015|
|L14VML17QKVML2TruePile|310327 / us-east5-a|2275/2535/3802/3030|1010/1140/0/0|432343756 /232220 /−1744|old L14 VM/L17 QKVM|1.010|

Each RUN also compares the completed independent2:1 directly. All gain ratios
use `BamMHAMediumPropC256TruePile`; forecast refers to each old tail control.
Two first arms use ordinary3-layer scan×5 + tail×1. Third uses ordinary×4,
L12–14×1 and final×1. This reuses the proven three-layer anchor/stage machinery;
the2:1 parent used two-layer scan. Parameter distributions and initialization
recipes are preserved, while changed scan parameter paths change RNG assignments.
Body Pallas v8t48/fused inputs retained; selective L14/L17 sites use the existing
JAX layout bridge (including the private L14 write), original O reads retained.
L16 attention write always present; final attention/MLP writes absent.

Measured full parameter trees and JAX/Pallas shape agreement pass the formal
preflight (`combined-formal-preflight.log`). Regression gate includes original
BAM47 + family16 + combinations6 + rotating-writer5=74 checks. Combination checks
cover original-tail parameter regression, exact full budgets/write omissions,
scanned private-write/anchor gradients and causality, selective v8t bridge.
Retain normal training health, disable internal/BAM statistics, tuned Splash,
full remat/T4096/global128, checkpoint200/latest2/keep0/TBrotation500.
Run13500, early report1000, review2800/5000. Speed priors: first two within±3%
of2:1, third about−2..−5%; M-cache75x32 unchanged.

Infrastructure audit through main1f3dbc67 finds no missing train/AOT/optimizer/
checkpoint/writer fixes versus family fork781a86a6. New base.yml fields are
experimental tail defaults only. Borrowed retained FLEX `llm-jax-v6e-1-0`,
europe-west4-a/worker0 is ACTIVE/idle, not best-effort. Its worker lock serializes
all three AOTs; each launcher owns only its own formal UE5a training prequeue.

Startup2026-10-23: all three RUNs sealed on`82e5a73a4e03dafabfb55f6d1d130e7dcae4f800`.
CPU74 passed once and was reused under the local gate lock. Retained FLEX AOTs
ran serially under its worker lock; all three training prequeues ran concurrently.
Each worker loaded its exact AOT, produced FIRST_STEP and passed step14; registry
and actual controller environments agree on runtime, both direct controls,
zone-local truepile4096 data and full13500 schedule. Mean step10–14 throughput:
QKVM.6960(−1.6% vs2:1), VM.6872(−2.8%), mixed.6760(−4.4%), within priors.
Their old tail startup controls are.722/.713/.705 (about−3.6%/−3.6%/−4.1%).
Same generic/internal/BAM health and tuned Splash;2:1 parent uses two-layer scan
while these runs use three-layer stages. Proofs`combined-{qkvm,vm,mixed}-startup.json`
and parallel-launch`*-20261023T112902Z/`; compiler lifecycle remains borrowed.


## Completed L1 source on the mixed dense2:1 tail (2026-10-23)

User requested adding the completed L1 residual to the running mixed combination,
hot-replacing independent rank8. New configuration:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17QKVML1L2TruePile`.
Direct sole comparison:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17QKVML2TruePile`;
MHA remains the gain reference. Same implementation family/worktree/branch as above.
All three native sites (L14 C2 VM, L17 C4 QKVM, final C1) use
[embedding, complete L0, complete L1, complete L2, current]. Each retains one
shared D->64 controller, zero dynamic-up and current-identity bias. Expand1/4
both retain hidden64. Total432344211, native routers232675: +455 parameters
(.000316 W_Q), exempt under existing Mudd convention. All ordinary/private widths,
write policies, scan stages, data and13500-step schedule remain the direct parent's.
The extra scan carry captures the completed L1 output exactly once, not its input.
Legacy default source ordering is preserved, including old L5 configurations.
Prior terminal MHA-reference gain1.010 [.995,1.025], throughput cost0..1%; expectation
is that supplying the missing intermediate first-block state can help where adding
another later cumulative anchor (historical L5) did not. This is a hypothesis.

Infrastructure audit through main37d4eea0: main-only changes since audited1f3dbc67
are experiment metadata; no omitted training/AOT/optimizer/checkpoint/writer fixes.
PallasTrue/v8t/48/fused inputs, tuned Splash, normal training health ON,
internal/BAM OFF, full remat, T4096/global128/percore16, checkpoint200/latest2,
TB rotation500 retained. Borrow verified FLEX_START llm-jax-v6e-1-0 worker0 in
EW4a for AOT; keep rank8 training until the new sealed gates/AOT are ready.
Hot-switch only the owned rank8 trainer UE5a xd-v5p-16-310324-maxtext; never
release the borrowed compiler. Initial agent launch ID310328.


### Verified L1 startup and replaced-arm closeout

L1 mixed route runtime dde685157893dc1378af5a833f4e57a3fe653f4c passed CPU77,
sealed configuration, retained FLEX AOT, actual loaded executable/FIRST_STEP/step14.
Mean step10-14 .677step/s vs direct parent.676 (+.15%, effectively unchanged).
Training ID310328 owns inherited xd-v5p-16-310324-maxtext in UE5a;13500-step schedule,
sole direct mixed-parent comparison and MHA-reference ratio registered. Startup proof
`l1-startup.json`. Compiler preserved.

Independent rank8 user-hot-replaced at committed8001 (not a full-schedule verdict).
Final five mature windows7000-7800 vs2:1 +.00398424/R.97748136; early catch-up stalled
rather than producing a sustained late MLP gain. Total parent-456, ordinary MLP+20.8%,
cache unchanged, steady throughput~+2.8% (startup+3.1%). Original terminal1.02 bet
unsupported through replacement. UE5a P1: first READY prequeue10:39:31-12:11:54 UTC
(1h32m23s; registry starts11s later), recovered12:19:26-14:03:54 (1h44m28s, handoff
boundary). CP8001 marker independently checked; old stopped registry verified before
bookkeeping wrapper; already_closed=true, no resource operations, TB SYNC_OK.

Single QKVM combination user-stopped6166 after common5k review. Latest five mature
windows5200-6000 vs original L2 QKVM +.00500920/R.97454565, vs2:1 +.00131196/R.99321580.
Large initial convergence lead reversed by~1.2k vs historical tail; later failed to add
value even to dense2:1. Total+1600 vs old tail, cache unchanged, startup-3.6% vs old
and-1.6% vs2:1. Original terminal1.010 prior unsupported through stop. UE5a P0,
first prequeue READY11:34:38-14:11:08 UTC (2h36m30s); registry starts83s later.
CP6166 committed, physical310325 node/queue verified absent, TB SYNC_OK.

### Two spaced VM sites on the L1 mixed route

User requests an additional independent RUN, not replacement of its L1/QKVM parent.
Configuration `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17VML1L2TruePile`.
Change only final QKVM to VM and restore L16 private memory writer. L16 attention write
stays enabled, L17 writers stay removed. Both native C2 sites and final C1 retain
[embedding,L0,L1,L2,current], shared width64 controllers. Ordinary L14-17 widths
2275/2535/2535/3544; private1010/1140/1140/0. Expected full total432344617 and native
routers232025: direct parent+406 from nonrouter integer rounding+1056 and router-650.
MLP kernels follow local existing budgets; native routers remain exempt. Sole direct
comparison is L1 mixed QKVM parent, MHA-reference gain. Terminal prior1.010 [.995,1.025],
throughput cost0..2%. Implementation has no further architecture change: scoped formal
parameter/write-policy, scanned forward/causal/gradient and v8t bridge checks suffice
on the verified77-check runtime. Main-only audit still37d4eea0 (metadata-only since
previous audit). Initial launch ID310329, UE5a v5p16, same T4096/global128/percore16,
Pallas and tuned Splash/health/CP/TB settings; borrow the retained FLEX compiler only.


Verified two-VM startup: runtime`afb2ee9d1f557529ef07c0ed316432b89db3f098`, physical310329 UE5a v5p16.
Scoped CPU3 groups passed (full five-arm parameter trees/write policies/JAX-Pallas
shape agreement, new two-VM scanned causal/gradient paths including L16 writer,
existing/new selective v8t bridge sites). Initial preparation was blocked by a
legacy test fixture missing L1; fixed fixture only and resealed before launch.
Owned first prequeue was released by the failed gate; retained FLEX compiler
preserved. Successful retry preparation20261023T141904Z, first trainer READY
14:23:55 UTC; actual AOT load/FIRST_STEP/step14 and controller source/data/plan
verified in`two-vm-l1-startup.json`. Throughput0.6678 vs .677 (-1.36%).
Sole direct L1 mixed-QKVM parent and MHA-reference gain registered. All widths,
432344617 total,232025 routers and live L16 write confirmed in formal trees.


## L14 vector capacity refund on spaced VM pair (2026-10-23)

User requests another independent RUN on the two-VM/L1 parent:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MuddL14VML17VML1L2NoL14MemoryMLPTruePile`.
Remove the complete L14 private M branch, not only its hidden dimension: set
`bam_mudd_pre_tail_vm_skip_mlp_write=True` and explicit per-layer private H14=0.
The selective write policy suppresses the ordinary-MLP-to-M fallback and removes
private content, address down/up/prebias and gate/bias parameters. L14 attention
write remains, L15/L16 private writers remain1140, L17 stays write-free. Sources,
native shared controllers, first-block anchors and final readout are unchanged.
L14 content1010*3600=3636000 + controls458000 gives4094000; refund1137 ordinary
widths4093200, L14 ordinary2275->3412. Last four normal widths3412/2535/2535/3544,
private0/1140/1140/0. Expected432343817 total,232025 native routers: parent-800
(-.0005556W_Q), ordinary171057600 (+4093200), cache unchanged.

Sole direct parent is the already running L14VM/L17VM/L1 class, common MHA gain
reference. Terminal prior1.015 [1.000,1.030], throughput0..+2%. This specifically
restores vector-processing capacity at the Mudd VM site while ordinary L15/L16
continue maintaining M; no simultaneous removal at ordinary L15. Early reports
1000 then2000 after5k, keep through5000 (2800 observational review), full13500
schedule. Initial launch ID310330 UE5a v5p16. Same independent family worktree,
branch, data/shapes, tuned Splash/Pallas/health/checkpoint/TB settings as parent.

Infrastructure audit through mainf50f17f6: main-only changes since prior audit are
ledger and closeout notes, no missing operational fixes. The only functional change
is a default-OFF selective pre-tail write-suppression flag and invalid-config guard;
no BAM math, kernel, scan, optimizer or initialization changes. Scoped sealed CPU
checks cover six full formal parameter trees/write policies, new no-fallback
forward/causality/gradients with live L14 attention and L15/L16 memory paths, direct
parent regression, and existing/new JAX/v8t bridge output/gradient equality. Borrow
retained FLEX compiler, preserve its lifecycle; launcher owns only the new UE5a
training prequeue. Main exp.py remains ledger-only for this worktree implementation.


Verified L14-refund startup on`42f04a217636fbd52c3462d624f48320de3a5d95`: sealed CPU4 groups passed,
six formal parameter trees and unchanged parent trees, no ordinary-write fallback,
causal/active gradients for L14 attention and L15/L16 private branches, v8t bridge.
Borrowed retained FLEX AOT and actual executable/FIRST_STEP/step14 verified;
physical310330 UE5a v5p16, initial prequeue READY14:51:06 UTC, preparation retry
20261023T144416Z. Startup0.6744 vs .6678 (+0.99%), within prior0..+2%.
Full432343817 parameters, native routers232025, local3412 width measured. Sole
direct two-VM/L1 parent and MHA-reference ratio registered;13500 schedule intact.
Proof`no-l14-memory-startup.json`; compiler preserved. Implementation remains on
codex/mediumprop-dedicated-m-mlp; main records configuration/launch metadata only.

Single L17 VM/2:1 combination user-stopped7904. Final5(7000-7800) vs historical
L2 VM +.002478494/R.986227696; vs2:1 -.000288720/R1.001674793. Large initial lead
reversed vs historical tail, later approached2:1; slight final recovery does not
establish convergence to historical tail. Prior terminal1.015 unsupported through
stop; full13500 not completed. Total historical-parent+1600, cache unchanged,
startup.6872 vs old.713 (-3.6%); no speed/parameter/cache offset. UE5a P0,
first READY11:34:43-14:54:59 UTC=3h20m16s (registry starts9s later). CP7904 committed,
node/queued resource independently verified absent by closeout; TB SYNC_OK.
Final proofs`closeout-combined-vm.json`,`combined-vm-registry-final.json`,
`combined-vm-final-report.json`. Historical oldVM's18 missing cached step10 points
6850..7020 were restored from its already-synced real TB, preserving prior cache
entries (`old-vm-loss-reexport.provenance.json`); final matched windows are complete.


## Main-branch implementation integration (2026-10-24 JST)

Merged `codex/mediumprop-dedicated-m-mlp` at41d9a626 into refactor-bam,
including original/native shared Mudd tail controllers, QKVM/VM/dual-V variants,
residual anchors, persistent embedding M, staged block scan, omitted writes,
independent/shared/half/compact-rank M-MLPs and all recent combination classes.
Pallas v8t selective-layer matrix-layout bridges and ordinary fused-core writes
are included. Historical experiment runtime hashes and results remain their
original reproduction references; the43 supported classes now run from main.
All43 effective configuration dictionaries match the sealed experiment source.
The seven runtime source files (base defaults and six layer modules) match the
verified experiment implementation byte-for-byte.

Main integration gate:131 CPU checks passed in the pinned environment:
shared BAM47, memory-MLP family16, combinations11, native Mudd23, original tail8,
summary writer5, standard Pythia5, full Mudd4, full-head QKNorm6,
Pallas/QKNorm3 and Pythia2.8B projection/init/scan compatibility3.
Full parameter/write-policy trees, forward/causal/gradient behavior and JAX/Pallas
bridges were validated. Proof:main-merge-gate-20261023T151202Z/passed.json under
/data0/xd/bam_diagnostics/mediumprop-dedicated-m-mlp. No additional training or
compiler resources were created for this integration; existing RUNs retain their
sealed Git/AOT lineage.


## Restart spaced VM routes without L1 (2026-10-24 JST)

User requested removing unproven L1 from both early VM/VM descendants and fresh
retraining, then explicitly requested latest old-run loss/gain plots before handoff.
Reuse the same family worktree, fast-forwarded to latest main6595b1d1 before edits.
No model/operator changes: only configuration overrides disabling L1 globally at
L14 C2/L17 C2/final C1. Sources[embedding,L0,L2,current], all controller widths64;
65*(2+2+1)=325 fewer native router parameters, no MLP compensation under existing
Mudd convention. Ordinary/private widths, attention writes, L16 private writing,
normalization, initialization, data,13500 schedule, kernel/health/CP/TB settings
all retain the respective old E/F recipes.

New configuration prefix BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1Mudd:
- L14VML17VML2TruePile:432344292 total,231700 routers; ordinary2275/2535/2535/3544,
  private1010/1140/1140/0. Primary control is the no-L1 mixed VM/QKVM parent;
  preserve the stopped old L1 VM/VM run for matched early source-removal comparisons.
  Terminal MHA-reference gain prior vs primary1.015 [1.000,1.030].
- L14VML17VML2NoL14MemoryMLPTruePile:432343492 total,231700 routers; ordinary
  3412/2535/2535/3544, private0/1140/1140/0. Primary control is the new no-L1
  VM/VM run, old L1 refund run is the matched source-removal control. Terminal
  primary gain prior1.010 [.995,1.025]. Both throughputs expected within~1% of
  corresponding old runs; cache unchanged.

No parameter-tree continuation: fresh RUN/GCS prefixes and full13500 schedules.
Launch IDs310331/310332 take physical310329/310330 UE5a v5p16 by authorized hot
replacement. Retain old trainers until CPU/config and both AOT artifacts are ready;
collect fresh curves and report them before either stop signal, then hand off.
Borrow retained FLEX worker0 EW4a; centralized worker lock serializes both AOTs,
no new trainer queues/resources. Keep standalone L1 experiment through5k review.
Infrastructure audit main6595b1d1 includes the complete experimental implementation
and operational fixes. CPU configuration-only gate covers eight formal parameter
and write-policy trees, two new causal/active-gradient cases, exact single-source
attribute/initial-function equivalence and legacy/new Pallas bridges. Original
131-check main integration just passed; no further runtime code changed.
