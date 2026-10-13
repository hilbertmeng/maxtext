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
