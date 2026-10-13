# MediumProp read050 projection sharing

User-authorized three arms, switched before launch from original SOTA to completed
single M-shift read .5. Latest-main fork: 0f7f6d10a2760c205699cf34f9834c892b4926d0.
Worktree /data0/xd/mediumprop-projection-sharing, branch codex/mediumprop-projection-sharing.
Parent model operation restored exactly from 52cf0b3: immutable previous-token M,
columnwise sigmoid .5; mixed M only feeds reads, original M receives writes.
No KV shift, no triple read gates, no address-derived gates.

| arm / RUN suffix after BamMediumPropMShiftReadG050 | bank | MLP sharing | MLP widths | expected params | residue vs432111104 | TPU |
|---|---|---|---|---:|---:|---|
| ReadAttentionSharedR256TruePile | R256 | private R256 | 3887/3762/3887 |432111680|+576|xd-v5p-16-310187-maxtext|
| ReadAttentionSharedR384TruePile | R384 | private R256 |3772/3646/3772|432121472|+10368|xd-v5p-16-310188-maxtext|
| ReadAttentionMLPSharedR384TruePile |R384| shared only6 write layers |3794/3669/3794|432118688|+7584|xd-v5p-16-310189-maxtext|

The third arm's remaining12 layers follow arm2. Full actual parameter-tree counts
matched all three expected totals in the pinned CPU shape probe. Shared reader/attention input yields one D->R GEMM; MLP
recomputes the same weights on its post-attention residual. Distinct up projections
and all existing read/write gates, RMS and bias conventions remain.
Projection-only deltas vsread050: -3.84 W_Q / +1.3568 W_Q / +.3498667 W_Q,
where W_Q=1200². R384 head expansion exceeds down savings in arms2/3, so those
MLPs shrink. Integer-width residuals are recorded above, not claimed exact matching.

Formal training: same13500 schedule, T4096, dataset TruePile4096, scan True,
checkpoint200, max_to_keep2, keep_period0, TB rotation500. Retained FLEX compiler
llm-jax-v6e-1-0 in europe-west4-a is borrowed under its AOT lock; never deleted.
Training-only SPOT v5p16 region us-east5-a, chosen from current owned multi-hour
healthy leases and main's latest regional history. Launcher verifies IDs absent.
Direct comparisons read050 and original SOTA; gain reference MHA,200-step windows,
agent1000-step reports. No new scalar-stat families; omit inherited32-column
M-shift means/stds while retaining14 summary health metrics/layer.

Pre-run forecast after switching parent: terminalloss vs read050 +.003 / -.001 / +.001.
Speed forecast .460 / .460 / .460 step/s vs measuredread050 .458350 (~+.36%).
Arm2 most promising; sharing regularization may help, arm1 capacity compression and
arm3 conflicting read/write tasks are the expected risks. These are bets, not results.

Infrastructure audit: starts latest main0f7f6d10; train.py writer.begin_step precedes
scalars, max_utils.get_summary_writer constructs rotating writer, resolved TB500;
latest checkpoint/JIT-AOT/optimizer paths retained. Runtime writer regressions,
new budget/reuse/direct-deferred/scan/gradient/causality tests and full47 BAM suite
are required CPU gate, cached once per sealed commit across the3 launchers.

Checkpoint diagnosis originalSOTA13500 64e3fc3 on retainedFLEX: joint native-GELU
unit selection and refitted up heads, same6 MLP-write layers,128 unique sequences
96fit/32held,64 sampledpositions/sequence. Heldloss deltas read+attention R384
+.00220165 (SEM.00033778), attention+MLP R384 +.00339578 (.00043283),
all3 R384 +.01056610 (.00109684). This supports train-time validation but does not
prove read050 compatibility. All diagnostic results use capture-free paired forwards;
no checkpoint writes. Artifacts gs://newproject-1-llm_base_models_us-central1/log/bam_diagnostics/mediumprop-r256-sharing/sota13500-20261013/cohort128-r384/.

Status: exact parameter trees and full47 BAM regressions passed; new scan/gradient/causality checks passed. First preparation blocked by a test-only tracer capture bug; no training started. Corrected host callback observes actual post-attention input under remat; resealing and retrying all three gates.
