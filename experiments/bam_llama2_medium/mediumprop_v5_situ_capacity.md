# MediumProp V5 SiTU: matrix capacity and rank scaling

Hold the trained18-layer `BamMediumPropV5SiTU47` backbone and total parameter
budget fixed. Scale M75xV and every address bottleneck together: V16/R128,
V32/R256 reference, V48/R384. These first three points keep R/V at8.
The added V64/R384 point holds rank fixed against V48, isolating further
matrix-width growth at the same total budget. This complements the12/18/27-depth
study; it measures spending parameters on matrix/address capacity versus MLPs.

Runtime2458bb3817e57b6ea0b4b6e929a7c38afb56ba92, branch
`codex/mediumprop-v5-situ47`, existing family worktree
`/home/xd/.codex/worktrees/medium-v5-situ47/maxtext`.
Main retains the ledger; SiTU implementation remains in that runtime family.
Artifacts: `/data0/xd/bam_diagnostics/mediumprop-v5-situ-capacity-20261113/`.

| RUN | TPU ownership | Ordinary/private/tail widths | Parameters | Delta parent | Ordinary/complete memory branch |
|---|---|---|---:|---:|---:|
|`BamMediumPropV5SiTU47M16R128`|`xd-v5p-16-311019-maxtext`|2662/1273/4147|432112388|+160|2.001236:1|
|`BamMediumPropV5SiTU47`|completed reference|2510/1128/3904|432112228|0|approximately2:1|
|`BamMediumPropV5SiTU47M48R384`|`xd-v5p-16-311020-maxtext`|2298/933/3583|432111620|-608|2.000921:1|
|`BamMediumPropV5SiTU47M64R384`|`xd-v5p-16-311021-maxtext`|2203/858/3443|432113860|+1632|2.001183:1|

Adjust the read, attention-write, private-MLP-write and embedding-write ranks.
Complete memory-branch cost includes its address down/up, bias and write gate:
205840/458000/775696 fixed parameters per nonterminal writer for V16/32/48.
Refund/charge each body layer locally and retain approximately2:1 across ordinary
and complete memory branches. Close the remaining embedding/terminal and integer
residue in the terminal ordinary MLP. No per-layer width mechanism is introduced:
single-layer scan over17 identical layers plus the explicit terminal layer,
whose attention and private-MLP writes remain off. Matrix state is1200/2400/3600
elements per token, or .5/1/1.5 times the parent.

Same original Prop recipe: D1200,16x75, RoPE18+NoPE75, noQKNorm; M pre-norm,
unit-initialized learned QK/VO layer scales, read-only M-shift .5, independent
VO addresses, negative attention diagonal. Ordinary MLP SiTU4/7, private MLP
SwiGLU. Keep parent write-content processing and all other initializations.
Train from scratch13500 steps/LR endpoint, T4096 TruePile, per-device16 on
v5p-16 (global128),7077888000 tokens. UE5a primary with UC1a/EW4b backups.
Generic health ON, internal/BAM activation metrics OFF, TBrotate500,
checkpoint200/latest2/no permanent period and forced final checkpoint.

## Runtime and kernel validation

Audited current main e7c1d552 against the reused family before sealing.
Training, AOT, checkpoint, optimizer, max_utils and summary-writer source files
are byte-identical. The MaxText differences are the intentional SiTU support in
base.yml/fusion.py/linears.py, its tests and experiment metadata; no missing
operational fix was found. Model/kernel lineage is preserved.

No new Pallas interface or kernel code: reuse v8t packed reads/fused inputs and
writes, full remat and48MiB VMEM, tuned Splash. CPU tests retain actual V16/V48
in a two-layer feedback model with packed/segmented attention and perturbed signed
QK/VO scales. Native/Pallas per-token loss and every parameter gradient pass;
worst relative gradient errors5.95e-6 and1.38e-5. Full-sized abstract trees verify
all four ranks, complete branch budgets and absence of both terminal writers.
Summary-writer rotation tests pass. Formal AOT still gates TPU layout/VMEM;
verify actual AOT load, both worker commits and FIRST_STEP at launch.

Retained FLEX was freshly checked SUSPENDED. Native AOT preparation allocates
owned temporary v6e spot candidates, cleans those candidates, and leaves FLEX
untouched. Training queue/installation runs concurrently with AOT preparation.

## Comparisons and prior

The initial two arms compare only with direct parent `BamMediumPropV5SiTU47` for routine
loss-gap plots. Register gain(arm versus parent; `BamMHAMediumPropC256TruePile`)
in the native registry. Report all active arms together every2000 steps, with full
cumulative curves from600; continue the depth-study group independently.

Prior: terminal gain multiplier .98 for V16/R128 (range .94--1.01),1.02 for
V48/R384 (.99--1.06). Speed versus parent: +4% (0--8%) and-7% (-3--12%).
Earlier V48 with fixed R256 on the pre-H strongest combination yielded1.01827
at the endpoint; it is historical evidence, not an extra routine baseline here.
Increasing R also costs MLP capacity, so this experiment tests their joint scaling.

## Verified launch

Both RUNs reached actual training steps, loaded their AOTs on both workers and
matched the sealed2458bb3 source. Zone-local TruePile4096 paths were verified.
Recent startup rates .724/.659 step/s versus SiTU18's historical steady .680;
these are unpaired training observations, not a same-machine speed profile.
The capacity group uses2000-step shared reports. Per the user's campaign
clarification, depth-scaling and subsequent studies characterize scaling and
capacity/budget tradeoffs rather than select a new SOTA. There are no
performance-based elimination reviews; train both arms through the full13500
steps unless the user explicitly changes scope. The depth group continues
separately on its existing shared cadence.

M16 uses the standard AOT2458bb3-bdd73e13. The temporary M48 v6e compiler was
preempted before completing an artifact. Its preparer was gracefully cancelled
and all four owned candidates were independently verified absent as both nodes
and queued resources across EW4a/UC1a/UE5a. Compile M48 instead on its already-idle
training worker0 with `prepare_train_aot_on_worker.py --isolated-artifact`, CPU
backend; the worker lifecycle and installed training checkout are retained.
Its verified AOT state is2458bb3-e74a7784-idle, artifact:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/2458bb3/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamMediumPropV5SiTU47M48R384.idle.pickle`.
Both use48MiB VMEM; no Pallas change or larger memory limit was necessary.

## Fixed-rank64-column extension

Runtime 5acf1abc8891dcba5b304a70f164b1ee6443fd13 on the same implementation branch/worktree.
Audit against main8017ffe8 again found training/AOT/checkpoint/optimizer/logger
infrastructure unchanged from the verified runtime. All four address ranks are
384. Body-layer parameters match the parent to+16; embedding, terminal and
integer residuals close through the terminal ordinary MLP. Complete writer
controls cost874256 per body layer. State size4800 elements/token is2x V32
and4/3x V48. Both terminal M writers stay off; single-layer scan17+tail.

Reuse Pallas v8t packed reads/writes and48MiB VMEM. Actual V64 native/Pallas
full-feedback forward and all-parameter gradients pass in the focused CPU gate;
full-size abstract budgets and summary-writer rotation tests pass. Formal AOT
must still verify target layout/allocation, then actual worker AOT/source/FIRST_STEP.

Keep the common V32/R256 reference in the shared figure; also register V48/R384
as a direct fixed-rank comparison for this arm. Native MHA-referenced gain prior
versus V48:1.01 (range .98--1.04); speed approximately4% slower than V48.
The objective remains scaling characterization, so train the full13500 steps.

V64 startup verified: both workers loaded the exact5acf1ab AOT/source on UE5a;
actual FIRST_STEP at12 and subsequent101-step gate pass. Startup .621 step/s is
5.8% slower than V48 .659 (unpaired, identical health settings). CPU all-gradient
maximum relative error2.75e-6. The formal48MiB-VMEM AOT succeeds without a kernel
change. Added to the existing group without resetting its2000-step calendar;
plots retain the common V32/R256 reference, with native V64/V48 fixed-rank
comparisons available separately.

## Completed V16/R128 point

`BamMediumPropV5SiTU47M16R128` completed13500/P0. Against the V32/R256 parent,
the absolute deficit gradually narrowed from about+.034 at600 and+.030 at2000
to final-five+.02649868, while MHA-referenced gain retention declined to .851638.
Thus the .98 central prior substantially underestimated the joint capacity/rank
cost. Refunds to both MLPs did not reverse it by the endpoint. The measured
tradeoff is half the M state, approximately6% faster training, and retention of
about85% of the parent's net loss gain over matched MHA, at the same total budget.
This is a completed scaling data point, not an eliminated contestant.

One uninterrupted UE5a v5p-16 lease: observed READY13:37:09Z through scheduled
completion19:05:06Z on2026-11-13,5h27m57s including preparation. The native
13:43:10Z observer covers a subset of the same lease. CP13500, seed9876/global128
cursor and zone-local TruePile files, TB SYNC_OK, native already-closed status,
and all six regional node/queue identities were independently verified. Evidence
is `m16-final-proof.json`, `m16-local-closeout.txt`, `m16-prequeue.log`, and `k12000/`.
V48 and V64 continue to their full budgets.

## Completed V48/R384 point

`BamMediumPropV5SiTU47M48R384` completed13500/P0. Its large initial lead reduced
during the transient, then held near-.005 through the middle and late stages;
final-five gap-.00473824 and MHA-referenced gain1.026518. The1.02 prior was close.
State size is1.5x the V32 parent, with approximately3.5% slower unpaired training
(.656 versus .680 step/s). Thus the gain survives to the endpoint at matched
parameter/token budget, despite the smaller MLPs required by larger M/R.

One uninterrupted UE5a v5p-16 lease: observed READY13:36:49Z to scheduled
completion19:43:31Z on2026-11-13,6h06m42s including preparation and the recovered
AOT compile on its idle CPU worker. Native13:51:57Z training observation is a
subset of this lease. Final CP13500/cursor, TB SYNC_OK, native already-closed
receipt and all six regional node/queue identities were verified; see
`m48-final-proof.json`, `m48-local-closeout.txt`, `m48-prequeue.log`, and `m48final/`.
Only the late-started V64 arm remains active. Its shared report calendar now
uses its own fixed8k/10k/12k points; previously published per-RUN horizons remain
intact. It still trains to13500 without elimination.

## Final capacity/rank scaling results

All three arms completed the full13500-step budget. Final five native200-step
windows through13400, versus `BamMediumPropV5SiTU47` (V32/R256); gains use the
matched `BamMHAMediumPropC256TruePile` reference:

| M / R | Loss gap | Gain multiplier | M-state size vs parent | Training speed vs parent |
|---|---:|---:|---:|---:|
|75x16 /128|+.02649868|.851638|.5x|about+6%|
|75x48 /384|-.00473824|1.026518|1.5x|about-3.5%|
|75x64 /384|-.00720208|1.040315|2x|about-9.3%|

Speed observations are unpaired, with matched generic health ON/internal metrics
OFF. V64 vs fixed-rank V48 ends at gap-.00246384/gain1.013441, close to the1.01
prior; V48's1.02 prior is also close. V16's .98 prior was far too optimistic.
Both larger points retain their gains through the endpoint. V64 has smaller
ordinary/private MLPs than V48, so its positive fixed-R result directly supports
spending some budget on matrix capacity rather than those MLPs.

The response is asymmetric under this budget: the joint V16/R128 compression
has a substantial cost that MLP refunds do not recover, while enlargement above
V32 yields smaller persistent benefits. The small-arm loss still mixes M and R;
it cannot identify which bottleneck dominates. The most discriminating next
control is V32/R384: compare with V32/R256 to isolate rank, and with V48/V64 at
R384 to obtain a three-point fixed-rank capacity curve. This campaign measures
scaling and tradeoffs; no arm was eliminated for performance.

V64 completed on2026-11-13T23:07:38Z after two preemptions, UE5a->UC1a. Both
recoveries retained the exact5acf1ab executable/parameter tree and schedule.
Committed emergency checkpoints7432 and12568 recovered progress; post-resume
CP7600 and post-migration CP12600/correct UC1 data filenames were verified.
Final CP13500, seed9876/global128/two-host cursor, TB SYNC_OK, native already-
closed receipt and all regional nodes/queues are independently verified.

All V64 v5p-16 READY leases, UTC on2026-11-13:

| # | Zone | Observed READY | End | Duration | Exit |
|---|---|---|---|---|---|
|1|UE5a|16:21:32|19:53:24|3h31m52s|preempted|
|2|UE5a|19:59:10|22:23:13|2h24m03s|preempted|
|3|UC1a|22:36:18|23:07:38|31m20s|scheduled completion|

The first interval includes prequeue/AOT/startup preparation; the native trainer's
16:26:16Z observer is a subset of that lease. Total observed READY time6h27m15s;
the two intervening gaps total18m51s. Active assignment switched to UC1a at
22:36:16Z; UE5a/EW4b pending alternatives were released after verified resume,
not counted as active-zone switches. Evidence: `m64-final-proof.json`,
`m64-local-closeout.txt`, `m64-preemption*-*`, and `final13500/` in the artifact root.
