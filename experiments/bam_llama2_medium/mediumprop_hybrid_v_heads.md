# Per-head residual/M values, ordinary-head local O comparison

Implementation worktree `/data0/xd/mediumprop-hybrid-v-heads`, branch
`codex/mediumprop-hybrid-v-heads`, forked from main `68cc1df3`.

RUN/configuration names:
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4KeepLocalOTruePile`
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4DropLocalOTruePile`

Both use the independent dedicated2:1 parent. Every layer uses4 leading ordinary
residual-projected V heads and12 M-read V heads, concatenated before attention;
all16 Q/K retain generalized M reads plus their18-dimensional standard RoPE tail.
No mixture gate, fraction multiplier, Mudd routing or read/write-down sharing.
Keep retains local O for all16 heads. Drop omits it only on the4 ordinary V heads.
All productive layers still write all16 attention heads and run private MLP
writes. L17 has neither attention nor private-MLP writes; L16 retains both.

Actual parameters are pruned, rather than keeping masked unused trainable gates.
W_V is1200x4x75. V keys/dynamic/static gates have12 heads. In Drop, VO-up,
VO pre-bias, O keys/dynamic/static gates also have12 heads. The shared read R256
down stays. Constant zero/closed leading head slots adapt these parameters to
the existing full-head v8t kernel; no new Pallas kernel implementation is needed.
This first implementation does not claim a proportional kernel-contraction FLOP
reduction from dropping O. Forward and all-parameter-gradient equivalence against
the native JAX path is a launch gate.

Each ordinary V head costs90000 and removes2434 V-read parameters. Four heads
add net350264=.243239 W_Q per layer in Keep. Removing their O paths saves42632
more per layer, giving net307632=.213633 W_Q in Drop. Shared Q/K/VO down cannot
be removed. Complete independent private write controls cost458000 per layer.

| Variant | Ordinary/private hidden widths L0-16 | Complete branch parameter ratio | L17 ordinary | Total parameters | Delta vs parent |
|---|---|---:|---:|---:|---:|
| Keep local O |2469/1108|1.998830:1|3848|432113632|+352|
| Drop local O |2477/1112|1.998834:1|3857|432113056|-224|

All first17 layers have exactly the same ordinary/private widths. Ordinary
widths use the uniform `base_mlp_dim` and terminal `bam_last_mlp_dim` override;
`mlp_dim_by_block` and `bam_final_local_mlp_dim` are None. No
`bam_mlp_dim_by_layer` mechanism or ordinary per-layer width list is used.
On user direction, disable `bam_pair_scan`: prefix17 layers scan one layer at a
time; L17 compiles separately. The parent's two-layer packaging is not inherited.
This changes parameter paths/RNG assignment relative to that historical parent;
initialization distributions and layer math remain unchanged. Both new arms use
the same single-layer packaging. No paired speed claim isolates scan granularity.

The preceding whole-layer ordinary-V Every3 arms lost to independent2:1 by
final5 gaps+.01118/+.00887 despite keeping Q/K/O reads. This experiment preserves
M-V transfer in every layer while giving ordinary V its own attention heads.
The ordinary-V fraction also changes from1/3 of layers to1/4 of heads, so any
recovery cannot be attributed exclusively to splitting heads instead of layers.
Historical all-O removal lost+.01795 in an earlier recipe; that is evidence
favoring Keep, but differs from selectively removing O on ordinary heads here.

Before-run terminal multiplier bets versus independent2:1, MHA reference:
Keep1.02 (working range.99-1.05); Drop1.01 (.98-1.04). Approximate terminal gaps
-.0031/-.0016 using the parent's late MHA advantage, not extrapolated measurements.
Startup speed priors .70/.71 step/s versus parent~.707; keep training-health ON,
internal/BAM statistics OFF, tuned Splash1024/fused, Pallas v8t/48MiB/fused inputs,
full remat. T4096/global128/per-device16/BF16/13500. Checkpoint200/latest2,
TB rotation500 with `begin_step` before scalar writes.

Infrastructure audit: this worktree starts at current main68cc1df3, directly
inherits its training/AOT/optimizer/checkpoint/summary-writer fixes, and changes
none of these paths. The decoder's existing explicit-tail path now also accepts
a homogeneous single-layer-scanned prefix, validated against per-layer masks.
Summary-writer regressions are part of the shared CPU gate.

Training ownership: Keep logical310363 (`xd-v5p-16-310363-maxtext`), Drop310364
(`xd-v5p-16-310364-maxtext`), spot UE5a, no passive backup zones.
The recent same-size leases favor UE5a; retain this direction's regional policy.
Borrow only the verified READY/idle FLEX_START compiler `llm-jax-v6e-1-0` in EW4a,
worker0; its queued-resource model is explicitly FLEX_START. AOTs serialize through
the worker lock; the compiler is never enrolled in cleanup. Preparation runs CPU,
AOT and owned training prequeue concurrently, gates all three before formal launch.

Both compare `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile`
and the stronger prior plain-V arm
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1StandardVEvery3DropMemoryTruePile`.
The latter was added on user direction; exclude the weaker KeepMemory arm.
Gain reference `BamMHAMediumPropC256TruePile` for both comparisons. Group plots
start600, with loss gaps above and gain multipliers below; one column per baseline
and both new arms overlaid, retaining each arm's latest complete horizon.
Report both arms together every1000 steps; no interim diary entries in `exp.py`.
User direction: neither arm has a2800/5000 stopping review; continue the13500-step
plan and retain grouped1k reporting.
Artifacts `/data0/xd/bam_diagnostics/mediumprop-hybrid-v-heads`.

Sealed runtime `b5e8b0eb675f251c39e77c1a41c9e6e9ffca3824`:62 CPU checks
passed (47 shared BAM,5 historical plain-V,5 new hybrid-V,5 rotating writer).
Single-layer-scan Pallas/JAX maximum relative gradient errors4.07e-6/3.80e-6.
Both exact AOTs verified, both installers and actual FIRST_STEP gates passed.
Formal assignment2026-10-25 04:44:21/04:44:24 UTC; the same training TPUs
were first observed READY during prequeue at04:32:28/04:32:24 UTC. The initial
two-layer preparation was superseded before any training on user direction;
queues/compiler were retained, all gates repeated on the single-layer runtime.
Main contains ledger-only classes; model implementation remains in the stated
experimental worktree/branch.

Worker source hashes, Loaded-compiled markers, actual GCS configs and both live
registry baseline/gain specifications were verified. Resolved T4096/global128/
per-device16, BF16/full remat, homogeneous17-layer single scan plus explicit L17,
tuned Splash/Pallas, health and TB-rotation settings match the launch plan.
Startup20-99 harmonic throughput Keep.685872/Drop.684498 step/s, about-3.0%/-3.2%
versus the parent's approximate.707 startup record; Drop/Keep-.20%, effectively
equal. This is a package comparison with different scan packaging, not a causal
timing attribution to O removal. Detailed audits and80-step timing evidence are
in `resolved-config-audit.json`, `registry-launch-audit.jsonl`, `startup-summary.json`
under the artifact root. No further interim loss entries belong in this ledger.

User-requested speed diagnostic: one standalone spot v5p-16 in UE5a,
`xd-v5p-16-hybrid-v4-pairscan-prof-1025`. Reuse the verified two-layer Keep AOT
at `a978db2dff5475ad598d31a3a644499ea0203f35`, same class and exact model/batch/
health/kernel settings; effective configuration differs from the live single-layer
runtime only in pair-scan flag, block size2 and explicit-tail count2. Preserve the
13500-step LR schedule; time20-99, collect through100 then stop, no checkpoints or XPlane.
Keep generic health ON, internal/BAM OFF and GCS TensorBoard in its original
central1 region; release the standalone TPU after artifacts are secured.

Completed: two-layer scan .675798 step/s (median.680), versus live single-layer
Keep .685872 (median.690), -1.47% in the same20-99 window. Both have80 samples,
T4096/global128/per-device16/BF16/full remat, tuned Splash/Pallas, training health
ON and internal/BAM OFF. Actual diagnostic configuration and AOT load were checked.
No speed improvement was observed, so retain single-layer scan. This is one
separate-v5p16 diagnostic arm versus the live arm, not a same-VM crossover matrix.
Timing/raw log uploaded from worker to GCS, downloaded locally and verified;
both diagnostic workers stopped. Node and queued resource independently NOT_FOUND
at2026-10-25 05:20:00 UTC. P0, only UE5a; first observed READY05:08:49, observed
READY-to-absence upper bound11m11s. Formal trainers and borrowed FLEX unchanged.
Evidence: `pairscan-timing-summary.json`, `pairscan-config.txt`, `pairscan-train.log`,
`pairscan-profile-release.log`, `pairscan-profile-absence.json` under the artifact root;
GCS `gs://newproject-1-llm_projects_us-east5/log/diagnostics/hybrid_v4_pairscan_20261025`.

## Matched3:1 capacity follow-up

New RUNs in this same worktree/branch:
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1HybridV4KeepLocalOTruePile`
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1HybridV4DropLocalOTruePile`

Each inherits the matching2:1 HybridV4 arm, changing only ordinary/private hidden
widths and its RUN identity/comparison. Both retain4 ordinary-V heads,12 M-V heads,
their original O policy, single-layer scan17+explicit L17, all17 productive
attention/private writers, tuned Splash/Pallas, training health ON/internal+BAM OFF,
and T4096/global128/per-device16/13500. No ordinary per-layer width mechanism.

| Arm | Ordinary/private L0-16 | Complete branch ratio | L17 ordinary | Total parameters | Delta vs matching2:1 |
|---|---|---:|---:|---:|---:|
| Keep O3:1 |2778/799|2.999280:1|3848|432113632|0|
| Drop O3:1 |2787/802|2.999283:1|3857|432113056|0|

Complete private branch cost is3600H+458000, including address/gate projections.
The two hidden widths sum to3577/3589 exactly as their respective2:1 parents,
so every productive layer's total and the global total stay equal; terminal widths
need no further correction. Normal hidden capacity rises~12.5%, private hidden
capacity falls~28%. All four body pairs round to the same padded sum3712 at128
alignment, a cost estimate rather than a measured throughput claim.

Ordinary V makes vector-stream features available to attention and its M writes.
This can change the best vector/private capacity allocation. Historical pure BAM
3:1 lost to2:1 at13500 (final5 gap+.0014874, MHA-reference multiplier.990542),
so the new experiment tests a changed optimum, not a claim that3:1 is universally
better. In Drop, ordinary-head write data is SA(V); in Keep it also contains
localO. I favor a larger3:1 benefit for Drop, where the extra ordinary capacity
feeds that write without a localO term.

Pre-run MHA-reference terminal multiplier bets versus each matching2:1 parent:
Keep1.01 (range.99-1.03), Drop1.02 (.99-1.045). Throughput prior within~2% of
each parent. Compare each new RUN only with its matching2:1 parent; use
`BamMHAMediumPropC256TruePile` as the common reference. Historical pure3:1/2:1
supplies the capacity-allocation prior. Group reports every1000 steps; no2800/5000
stopping reviews, plan13500 for all four.

Infrastructure audit versus latest main8832c269: no main-only changes in training,
AOT, optimizer, checkpoint or summary-writer implementations since model parent
b5e8b0e. Main's missing experimental head fields/script reflect ledger-only status,
not infrastructure fixes to port. Core model code remains identical to the62-test
verified parent. Targeted CPU gate checks both full parameter trees, inherited
protocol and5 rotating-writer regressions; no repeated unrelated full BAM suite.

Keep ID310365 and Drop310366, spot v5p-16 UE5a only. Borrow the reverified
FLEX_START `llm-jax-v6e-1-0` EW4a worker0 for serialized AOT; never release it.
Artifacts `/data0/xd/bam_diagnostics/mediumprop-hybrid-v-heads/ratio3`.


## Independent2:1: separate dynamic V/O addresses

User confirmed the original pure BAM dedicated2:1 parent, not either hybrid-V arm.
RUN `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile`.
Implementation stays in this family worktree/branch. Share the original readR256
down/GELU across Q/K/V/O, but add a separate R256->16x32 V up and512-element
V pre-RMS bias. The original VO up/bias now serves O. Dynamic gates, static read
keys and static gates were already independent and are unchanged. Clone V's up
initialization from O and start its bias at zero; no initial read-amplitude jump.
The two addresses and their gradients then learn independently.

The v8t kernel packs independent V/O keys into its existing key input, retains
the same4N-row read matmul, and handles their RMS reverse independently. Native
JAX uses two reads. No JAX-only fallback; fused inputs, Pallas v8t/48MiB, tuned
Splash1024/fused, health ON/internal+BAM OFF remain enabled.

Extra address parameters131584/layer=.0913778W_Q,2368512 total=1.6448W_Q.
Remove the parent's unused458000-parameter final attention writer; the final
private MLP writer is already absent. Reallocate the productive layers' remaining
MLP budget at complete ordinary/private ratio1.999646:1: widths2510/1128,
uniform forL0-16; terminal ordinary3904. Total432112192, parent-1088
(-.0007556W_Q). No per-layer ordinary-width mechanism. Use the homogeneous
17-layer single scan plus explicit L17, as directed for this family, rather than
the historical independent2:1 parent's unnecessary two-layer packaging.
Initialization distribution/layer math retained; scan packaging/RNG paths differ
from that historical parent, so this is not a bitwise-parent startup claim.

Historical prior: last-block separate-C8 VO at2867d2f ended near parent parity
(last5 through2800 gap-.00000448); the older no-W_O/head-mix family also
separated keys but differs substantially from this generalized full-M recipe.
Before-run bet versus independent2:1: terminal MHA-reference gain1.02
(working range.99-1.05; approximate gap-.0031 using the parent's terminal
advantage), speed within~2% of its~.707 record, scan package differs.

T4096/global128/per-device16/BF16/full remat/13500; checkpoints200/latest2,
TB rotate500. Direct compare only independent2:1; common MHA reference
BamMHAMediumPropC256TruePile. Report1000-step groups from600, loss gap
and gain together. Standard Medium2800-step review;13500 remains the training
endpoint so a review never triggers automatic resource deletion.
Spot UE5a v5p16 ID310367; reverified READY/idle ACTIVE FLEX_START compiler
llm-jax-v6e-1-0 EW4a worker0, borrowed only and never lifecycle-adopted.
Infrastructure audit versus mainfcef789c found no main-only train/AOT/optimizer/
checkpoint/summary-writer changes since2437dfe; no fixes needed.
Full BAM regression plus independent-key native/Pallas forward/all-gradient,
shared-key hybrid regressions, shifted-read reverse and summary-writer gates
precede training. Artifacts: family root/separate-vo.


## Hybrid V heads plus independent V/O addresses

User requested both combinations after reviewing the independent-VO experiment.
RUNs:
- BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4KeepLocalOSeparateVOKeysTruePile
- BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4DropLocalOSeparateVOKeysTruePile

Each inherits only its matching2:1 HybridV4 parent. Four ordinary W_V heads stay
unchanged. Separate dynamic V/O addresses only on the remaining12 M-V heads: one
additional R256->12x32 up and12x32 pre-RMS bias; the shared Q/K/V/O R256 down
remains. Clone each V address from its matching O address at initialization;
ordinary-head O-only addresses, where present, stay unchanged. Dynamic/static
gates and static reads already independent, with no added mixture gates.

Use the existing independent-key v8t kernel unchanged. Adapt compact head keys
by padding the four absent V slots and, for Drop, the four absent O slots before
packing V/O separately. The read matmul still has4N rows. Native and fused
paths must match forward and every parameter gradient. No JAX-only fallback.

| Arm | Ordinary/private L0-16 | Complete branch ratio | L17 | Total | Parent delta |
|---|---|---:|---:|---:|---:|
| Keep O, independent VO |2451/1098|2.000453:1|3831|432115216|+1584|
| Drop O, independent VO |2459/1102|2.000452:1|3840|432114640|+1584|

Extra98688/layer=.0685333W_Q,1776384 total=1.2336W_Q, exactly3/4 of
the pure independent-VO address overhead. Preserve the complete ordinary/
private2:1 allocation after paying this cost, adjust only the terminal ordinary
width for integer balancing. Both parent deltas+.0011W_Q. All first17 widths
uniform, base_mlp_dim plus bam_last_mlp_dim; no ordinary per-layer width list.
Single-layer scan17+explicit L17; retain all17 productive attention/private
writes and neither terminal writer. No Mudd, no new M-shift gates, no cache change.
T4096/global128/per-device16/BF16/full remat/13500, checkpoint200/latest2,
TB rotate500. Pallas v8t/48MiB/fused inputs, tuned Splash1024/fused backward,
generic training health ON/internal+BAM stats OFF.

Before-run MHA-reference terminal gain bets versus matching hybrid parent:
both1.015 (working range.99-1.04); approximate late gap-.0024 using the
parent's late MHA advantage. Speed prior about-1% (range-3%..+1%). I expect
partial overlap: ordinary V changes the information source, while independent
VO changes M addressing, but both relax value/O coupling. Do not assume
ordinary-head O removal necessarily makes the independent-address gain larger.

Only matching hybrid parent in compare_runs; matched common reference
BamMHAMediumPropC256TruePile. Group1k reports, curves from600; default Medium
2800-step review,13500 endpoint. Pure independent VO versus pure2:1 supplies
the other conditional effect for this family's secondary analysis:
Q=R_child/hybrid divided by R_pureVO/pure2:1, with both native R using MHA.
Q measures relative-gain interaction, not exact absolute additivity.
I=(L_hybrid-L_child)-(L_pure2:1-L_pureVO), zero for absolute additive gains,
positive above the additive prediction. Plot Q and I beside direct loss gaps/
gain ratios at complete common windows. Keep both metrics in this family
analysis; do not modify the generic registry report. This tests complementarity
or substitution, not a precise decomposition of the ordinary-V mechanism.

IDs310368/310369, spot v5p16 UE5a, no backup queue. Borrow only the verified
READY/idle FLEX_START llm-jax-v6e-1-0 EW4a worker0; serialized AOT lock and
parallel CPU/training prequeues. Never lifecycle-adopt/delete the compiler.
Worktree/branch unchanged. Audit main55523afe versus5dea945: no main-only
training/AOT/optimizer/checkpoint/summary-writer fixes. Kernel unchanged;
focused CPU gates cover both full parameter trees, masks and cloned initial
keys, native/Pallas logits/all gradients in both O modes, pure-VO regression
and rotating writers. Artifacts: family root/hybrid-separate-vo.
