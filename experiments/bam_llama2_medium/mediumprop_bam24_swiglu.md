# All-in-budget BAM24 ordinary SwiGLU control

The user corrected the27-layer wording: intended arm belongs to the active
all-in-total-budget24-layer study. Do not reuse the mistaken D976/BAM27
configuration, checkpoint or conclusions. Reuse only its retained v5p-16
resource after its old worker is quiescent.

Class `BamMediumPropV5L24TotalBudgetSwiGLUTruePile` inherits directly from
`BamMediumPropV5SiTU47L24TotalBudgetTruePile` (actual parent runtime46e26e7).
Only ordinary SiTU limits change4/7 to0/0; private writers already use
SwiGLU. D1056/16x66, M66x32, all independent R256; ordinary/private/tail
2201/968/3435. Total432120064, unchanged and1136 below the MHA18/24 target.
No parameter reallocation or kernel interface change.

Current main shared layers/training sources remain identical to the actual
parent runtime; keep v8t packed fused read/write,48MiB VMEM, full remat,
tuned Splash and single-layer scan of L0–22 plus explicit L23. Terminal
attention/private M writes stay disabled. Seed9876, global128xT4096,
TruePile4097 records,13500 steps =7077888000 tokens; LR endpoint13500.
Generic health ON/internal OFF, TBrotate500, checkpoint200/latest2/final.

Direct controls: SiTU24 parent, current strongest post-only fixed-router
Mudd24, matched MHA24. Native MHA-referenced gain triples; paired depth
comparison uses the completed pure-SwiGLU18 V5/MHA18, while the earlier
SiTU18 pairing is retained only as a diagnostic of the parent study.
Full budget, no elimination; fixed2k reports. UE5a primary and UC1a backup,
EW4b excluded. Start from scratch under a distinct RUN after handoff.

Prior: final gap-.001 versus SiTU24, gain1.005 (plausible .985–1.025);
speed~.55–.56 steps/s versus parent's~.550 (unpaired). The point is to
remove the activation confound in the depth study, not to declare that
SiTU already explains its results.

Main/refactor-bam. Intended retained TPU xd-v5p-16-311042-maxtext; use new
launcher ID311043 with explicit retained TPU override. Preserve the old
RUN's actual checkpoint/TB/leases and mark it cancelled due to scope
correction after successful new FIRST_STEP. Artifact root:
`/data0/xd/bam_diagnostics/mediumprop-bam24-swiglu-20261115`.

## Accepted startup

Runtime5a5dd84feef9f4a154af2988ccb798e28a40d1d7, main/refactor-bam.
Both full parameter trees have104leaves and432120064 parameters; only the
two ordinary SiTU limits, class name and compare metadata differ. Shape
probes use native read/write paths atT4; the unchanged real Pallas path
passed the actual two-worker exact-AOT/source and FIRST_STEP/step14 gate.
Reuse the unchanged4-test SiTU activation regression; no kernel source
changed. Formal logs include step0, proving fresh initialization rather
than resumption of the mistaken27-layer checkpoint.

The preserved UC1a xd-v5p-16-311042-maxtext now runs the24-layer RUN under
launcher311043. Native hot_switch_run dry-run/manifest and quiescence gates
passed; primary remains UE5a, backupUC1a. Tools83a84d8 fixes handoff to
preserve the registered primary independently of its current allocation;
14handoff regressions passed and deployed/source hashes match.
Startup median over steps55–99 was.546 steps/s, about-.7% versus SiTU24
.550 (unpaired; matching generic/internal health). Artifacts and proofs
are local `startup-proofs.json`, `startup-speed-proofs.json`,
`startup-accepted.json`, and the remote task's `handoff.log`.
The exact CPU AOT/manifest lives under
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/5a5dd84/jax081-i0ae3f58-c5aae279/v5p-16/s13500/`.

Group `mediumprop-bam24-activation`, fixed2000-step cadence, no reviews.
Two Mudd24 norm controls remain in their existing group. The shared
read-only waiter watches both native snapshots and startup receipts;
it does not poll training console logs.

The actual pure-SwiGLU18 reference is the completed 13,500-step RUN
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormQKScaleTruePile`
(runtime f8d07cd). Native paired gains compare SwiGLU24/SwiGLU18 against
MHA24/MHA18; the original SwiGLU18 also has a gain triple against post-only
Mudd18. Thus cross-depth SwiGLU analysis does not silently use SiTU18 as
its denominator. SiTU24 remains the direct activation-ablation parent.
Receipt: `matched-swiglu-depth-gains.json`. Effective attributes of the
SiTU24 parent were verified unchanged from runtime46e26e7. Adding these
gain triples preserves existing loss comparison cursors.
