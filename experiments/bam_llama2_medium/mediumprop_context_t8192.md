# MediumProp T8192 length extension: MHA / Mudd / shared-R256 generalized BAM

Worktree `/data0/xd/mediumprop-context-t2048`, branch `codex/mediumprop-context-t2048`.
Reuse the completed T2048 context-family implementation at e9f77281 rather than latest main model changes. Main exp.py remains ledger only. Three new RUNs inherit their T4096 parents; no parameter-tree or MLP-width change.

| Arm | RUN | Planned TPU | Params |
|---|---|---|---:|
| MHA | BamMHAMediumPropT8192SplashTruePile | xd-v5p-16-310160-maxtext |432121200|
| Mudd | MuddLlama2MediumPropT8192SplashTruePile | xd-v5p-16-310161-maxtext |434076747|
| BAM | BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT8192 | xd-v5p-16-310162-maxtext |432111104|

T4096/global128 -> T8192/global64, per_device_batch_size8 on v5p-16's8 devices. Each step524288 target tokens,13500 steps,7,077,888,000 total targets; retain corresponding parent LR/optimizer/init/model widths. Mudd retains full growing history and historical unrolled dispatch; MHA/BAM scan as parent. All three use Splash. User explicitly accepts Splash/C256 effects as small and directs proceeding without extra backend controls. Historical T4096 MHA/Mudd C256 results remain comparison references; record this one backend difference, do not request additional controls. T2048 aborted Splash prefixes are excluded.

Dataset declaration truepile8192, no hardcoded regional prefix. Selected UE5a replica `gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_8192`. Verified manifest8193-token records,3,200,000 samples,100 materialized shards and coverage50000 at batch64. Source/replica all-shard CRC audits retained in truepile8192-build; this run preserves region-local routing. Record boundaries/order differ across lengths, so primary length conclusions use within-length MHA-relative gaps and their differences/ratios, not raw cross-length loss alone.

Infrastructure audit against main0f7f6d10a2760c205699cf34f9834c892b4926d0: main-only historical commits5ae86564/2a6c2d46 reviewed; effective train/train_compile/optimizer/max_utils/maxtext_utils/base runtime defaults and summary writer already match their operational fixes in this family. No model merge. TB positive period500 -> rotating writer -> begin_step before scalar writes; five writer regression tests in CPU gate. Keep latest2 checkpoints with keep_period0, periodic200 and final save. Formal launch gates exact sealed config, focused full-count/token-budget/history/forward-gradient CPU checks, target-topology AOT, actual FIRST_STEP and14. CPU parameter-shape fixtures disable Splash solely for CPU execution; actual sealed configs and AOT retain Splash. Borrow verified idle ACTIVE FLEX_START llm-jax-v6e-1-0/EW4a worker0, pinned Python; never lifecycle-adopt/delete it. Three AOTs serialize on the worker lock; CPU checks and trainer prequeues overlap. UE5a chosen from latest five same-zone unpreempted formal leases; no backup zones requested.

Define b_T=MHA_T-BAM_T, u_T=MHA_T-Mudd_T. At exact common steps, use five samples step+-20/10 inside200-step windows. Report b_T/u_T, length amplification b8/b4 and u8/u4, second-order (b8/u8)/(b4/u4), added gains Δb=b8-b4 and Δu=u8-u4, Δb/Δu and Δb-Δu. Retain the T2048 family as an extra8/2 view when windows are common. Compute each ratio per window then aggregate; tiny or sign-changing denominators must remain visible and are not interpreted as meaningful performance multiples. Normal1000-step agent reporting,200-step cached windows and silent~10min liveness checks. Planned13500 to completion as a sequence-length comparison, no default2800 cull.

Pre-run bets: at13500 BAM-MHA-.165, Mudd-MHA-.088, BAM-Mudd-.077; b8/u8~1.875, second-order8/4~1.13. Predict Δb~+.0204, Δu~+.0011, Δb-Δu~+.0193 versus observed T4096 endpoint. Kernel throughput MHA.62/Mudd.54/BAM.39step/s; BAM about-18.5% vs T4096.4787. Shape/parameter count preserves each parent's budget; Mudd's historical+.45% versus MHA remains, no newly introduced architecture parameters. Timing and loss bets will be checked with observed runs.

Artifacts `/data0/xd/bam_diagnostics/mediumprop-context-t8192`. Previous conversations consulted: original「查找Medium压缩M列读实验」01a0a354-2e46-7320-86bf-714eb273f0dd and completed handoff01a12fe7-da30-7302-940a-5476f71ebcac; final reportmediumprop_t2048_final_20261012.md supplies authoritative ratios and corrected matched baselines.

## Actual launch proof

All three actually training on sealed runtime dbcaa31f46ce88b9582793b38a57cccb2e9cd1ec. CPU full counts and finite gradients, five writer regressions, exact config and target-topology AOT passed. Actual FIRST_STEP and14 verified for all. Worker logs show Loaded compiled function, SplashTrue/T8192/global64/per-device8/TB500/LR.0003/schedule13500. Mean10–14 MHA.5280/Mudd.4880/BAM.3908;20–99 pending. Registry and controller environments agree on runtime, direct compares, region-local8192 data and AOT URI. Artifact startup14-proof.json and controller-env-proof.jsonl.

Steady20–99: MHA.5260250/Mudd.4859625/BAM.3889500step/s, versus T4096-.26327/-.22863/-.18753. Same-length BAM speed-26.0586% versusMHA/-19.9630% versusMudd. MHA/Mudd speed bets missed(-15.16%/-10.01%), BAM-.27% from bet. Sanity-check affine step-time extrapolation from observed T2048/T4096 predicts approximatelyMHA.508/Mudd.458/BAM.391; actual speeds are compatible with increased attention arithmetic at fixed tokens, so the original MHA/Mudd forecasts were optimistic rather than an identified runtime fault. Baseline MHA/Mudd kernels differ by user authorization; this is a timing sanity estimate, not a measured matched-backend ablation. Artifact startup-speed-proof.json.

## Common1000 review

All running normally/unpreempted. Committed checkpoints MHA1400/Mudd1200/BAM1000, reported cursors1400/1200/1000. Common1000 BAM-MHA-.734407, Mudd-MHA-.498835, BAM-Mudd-.235572. Initial advantages are already shrinking after400/600; no early淘汰 for this context study. TB inventory verifies part00000 and00001 on all three actual GCS event streams after500.

At1000 b8/u8=1.472244 versus b4/u4=1.508583; BAM context8/4=1.257156, Mudd1.288186, second-order.975912. Last600/800/1000 second-order.950605/.981745/.975912, all just below1; first-five mean1.000592. Δb+.150226, Δu+.111597, Δb/Δu1.346149, Δb-Δu+.038629. Absolute extra leading margin favors BAM, while proportional context amplification favors Mudd at this early stage. These are different questions; do not interpret second-order<1 as smaller absolute extra BAM gain.8/2 cumulative second-order1.001671.200/400 Δu negative, raw added-gain ratios retained but not meaningful gain multipliers. No current support for terminal1.13 second-order bet; only1000 observed, continue planned13500.

Cross-length raw losses remain unpaired: MHA8-MHA4+.168656, Mudd8-Mudd4+.057059, BAM8-BAM4+.018430, BAM8-BAM2-.002395 at1000. BAM8-BAM4 crosses behind at1000 after briefly ahead600/800; raw sign flips are not pure context causality. Artifacts report-1000.txt, factors-1000.json, tb-rotation-inventory.txt. Next regular common report2000, shared silent liveness checks in between; do not hide faster RUNs on peer failure.

## Common2000 review

Healthy/unpreempted, committed CP2600/2400/2000, reported cursors2600/2400/2000. Common2000 B8-M8-.396604,U8-M8-.256697,B8-U8-.139906; recent common1200..2000 B8-M8 mean-.495098/B8-U8-.166364. Large startup gains shrink; retain full13500 length plan.

At2000 b8/u8=1.545024, b4/u4=1.585340, b2/u2=1.441008.8/4 context BAM1.268507/Mudd1.301608, second-order.974569; Δb+.083950/Δu+.059482/Δb-Δu+.024468,Δb/Δu1.411353. Recent1200/1400/1600/1800/2000 second-order.952404/.998713/.973351/.960801/.974569: allbelow1, mean.971968 (range.952404.. .998713). Means BAM context1.273209/Mudd1.310488, Δb+.106338/Δu+.077935/extra+.028403.8/2 cumulative second-order rises1.001671@1000->1.072183@2000 (last5mean1.045340). Provisional inference: relative scaling benefit is stronger2k->4k than4k->8k at this stage, despite larger absolute extra BAM leading margin. Terminal bet untested; phase behavior can change.

Raw cross-length losses repeatedly flip: B8-B4 at2000-.014073, B8-B2-.077156; M8-M4+.069877,U8-U4+.010395. These unpaired-record differences do not establish pure context causality. Full cumulative owned direct comparisons inreport-2000.txt; factor seriesfactors-2000.json; recent means/rangessummary-2000.json. Next common3000, shared silent liveness checks in between.

## Common3000 review / revised second-order forecast

Healthy/unpreempted, CP4000/3800/3000 committed, reported cursors4000/3600/3000. Common3000 B8-M8-.306849,U8-M8-.192163,B8-U8-.114687. Common2200..3000 B8-M8 mean-.335801 (previous-.495098), B8-U8-.121030 (previous-.166364): early advantages still shrink but drift has slowed. B8-U8 rebounds slightly at3000 after2800-.109612; broad trend remains shrinking.

At3000 b8/u8=1.596820 versus b4/u4=1.648943, b2/u2=1.426185.8/4 BAM1.269614/Mudd1.311056, second-order.968390; Δb+.065162/Δu+.045592/extra+.019570; Δb/Δu1.429253. Last2200/2400/2600/2800/3000 second-order.965423/.979910/.965223/.941559/.968390, mean.964101 (previous.971968),13 consecutive windows600..3000 below1. Means BAM context1.255955/Mudd1.302784, Δb+.068440/Δu+.049847/extra+.018594.8/2 cumulative second-order1.119644, recent mean1.101355 versus prior1.045340. Cumulative long-context proportional benefit stays positive, while incremental4k->8k proportional amplification slightly favorsMudd so far.

Working terminal second-order forecast revised from pre-run1.13 to near1.00 on this evidence; preserve initial bet for end scoring rather than editing history. Original absolute-loss and speed bets remain archived, not retroactively refitted. All three continue full13500 context plan, no2800 cull.

Raw M8-M4 crosses negative at4000 (-.043272 after3800+.102618); unpaired raw length gaps are still noisy. Artifacts report-3000.txt, factors-3000.json, summary-3000.json. Next common4000; shared silent liveness checks between reports.
