# XL shared-rank4 joint GELU read projection

RUN: `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayerQKJointGelu256`. Direct baseline: `BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer` (33244e0).
Runtime worktree `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
Main exp.py is ledger only. Owned training TPU `xd-v5p-32-xl-k96-qk-rank4-joint-gelu256-maxtext`.

Replace the 2048->256 dynamic read projection (shared basis4x32, Q mix16x4, K mix16x4)
with 2048->256->GELU->256. Separate Q/K gate logits remain direct linear functions of x,
zero-kernel initialized, with the original .05 gate prior. Existing shared4x32 pre-RMS bias
remains after the basis output; no new projection biases. Static Q/K read keys are unchanged.
Both L and F layers use the new LocalQK projection. Full-M effective-key CFp32 reading,
M96x32/C8, concat96+32, partialRoPE32, LocalVO/FetchedO, and all write paths remain unchanged.

Down uses fan-in unit-variance initialization. Up seeds nonzero basis and mixes using
original segment initializers, rescaled by sqrt(D/hidden/E[GELU(N(0,1))^2]) with E=.425193711
so raw-output second moments match the prior projection. This matches amplitude scale,
not exact activations or the original loss trajectory. Never zero-initialize the up kernel.
Gates remain outside the nonlinear hidden representation.

Added parameters per layer65536=.015625 W_Q (D2048); all24layers1572864=.375 W_Q.
MLP hidden widths6266/6266/6266 ->6255/6255/6256. Across8LLFblocks exact total is preserved:
**1420870528** parameters, verified from actual initialized shapes. Sharding overhead.0731%<2%.
L layers each2048 fewer total parameters, F each4096 more: integer-width rounding cancels
inside each block. No hardware alignment or extra scan partition.

Prediction before training: terminal gap-.001 vsrank4, subjective range[-.004,+.003]; speedflat.
The existing basis-times-head-mix is already nonlinear in x. This tests shared nonlinear feature
formation, not an increase beyond rank4. M-cache unchanged; matrix FLOPs approximately balanced
by MLP deduction, plus a small GELU cost. No claim of guaranteed gain over shared-P DirectC8.

Training: new RUN fromstep0,50000steps,checkpoint250,full24-layerblockscan,AOTv5p-32,
generic health ON and968 BAM read-health metrics. PrimaryUE5a,backupsUC1a/EW4b;
recent owned UE5a leases2h53m/4h36m ended in preemption, both same-zone recoveries succeeded.
Compiler v6e EW4a primary, UC1a/UE5a backups. Review10000; report500-stepwindows in~1000stepbatches.

Validation artifacts `/data0/xd/xl-qk-joint-{tests.log,audit.json,health-trace.log}`.

Validation completed:57 pinned CPU tests PASS; exact total/sharding audit PASS;
full train-step tracing exports968 scalar read-health metrics.

User-directed handoff: separate-C8 RUN paused at committed11193; new RUN starts from0 on retained `xd-v5p-32-xl-qkstatic-vo-c8-ig-k96-directc8-27-maxtext` (UE5a). Unstarted extra queue removed. AOT runtime348fd5a ready, compiler cleanup verified.

Launch verified: runtime348fd5a, AOT loaded, step0 start, FIRST_STEP confirmed. Steps10–14 .5388/s (-1.17% vs rank4 .5452, matched968 health); steps20–24 .5400/s. Evidence `/data0/xd/xl-qk-joint-start-verified.json`.

Monitoring baselines: shared rank4 isolates joint-GELU change; original shared-P DirectC8 measures gain against the strongest current XL variant. Both are direct comparisons. Shared-rank4 briefly paused11777, then user requested resume through at least20000 after noting recent narrowing; original runtime/schedule retained.

User cadence: all owned XL runs report every2000steps, retaining500-step loss windows. Sharedrank4 must be observed through at least20000.

Reviews: JointGELU at10000 last5+.000899 vsrank4,+.004585 vsoriginalC8; continue because catch-up remains.
Sharedrank4 at20000 last5+.001996 vsoriginalC8, lower than+.002798 at15500; continue beyond minimum observation.
Neither has established a net loss gain over originalC8. Keep2000-step reporting.
