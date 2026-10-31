# H dedicated memory MLP: Pythia biases

Implementation: `/home/xd/projects/maxtext`, branch `refactor-bam`.
Parent: `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixed`, runtime `9c11469da97a0ade00530448736e55024bfd6e99`.
New EXP/RUN: `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedMemoryMLPBias`.

## Intervention and prior

Add zero-initialized biases to the dedicated memory MLP's hidden and output projections, matching ordinary Pythia GELU MLP. Kernel initialization remains the original BAM recipe. The 31 writing layers add `31*(2944+2560)=170624` parameters: 0.00615% of the parent, 0.02603515625 W_Q. User explicitly exempts this tiny cost from repayment. Ordinary/memory/tail widths remain 7808/2944/9984; total rises from 2775190912 to 2775361536. Both biases inherit zero weight decay, including stacked scan parameters.

Before training: at16k predict gain multiplier1.02 versus H, range0.99–1.05, with MHA the shared reference; speed approximately unchanged. Original H remains the primary direction SOTA baseline. Additional direct references: pure QK80, pre-mask+QK80 combination, Mudd serial ShortTrain and MHA ShortTrain. Do not confuse the combination with pure pre-mask.

Same-seed shared parameters and initial logits are exactly unchanged. Ordinary Flax `param` creation consumes RNG even for zero bias and perturbs scanned kernel draws. The guarded `DenseGeneral.zero_bias_without_rng` path instead uses the public variable API with logical partitioning and no RNG; only the dedicated memory MLP selects it. Both added biases have finite nonzero gradients and changing either affects logits. All historical defaults remain unchanged.

## Launch and infrastructure

Fresh RUN and checkpoint prefix: parameter tree changes. Retain original28000-step cosine schedule, T2048/global1024/per-device16, v5p-128, minimal remat, VMEM60MiB, scan31+tail1, generic healthON/internal+BAM sowOFF, TB rotation1000. No additional health metrics.

Infrastructure reviewed against main `7b89a0fb0918db89732025b8c9b036bf47f1b8d2`: current training/checkpoint coordination and TB rotation retained; model parent is9c11469. New runtime includes `CoordinatedCheckpointSaver` and explicit Orbax1200s timeout. Versioned authoritative external controller release `eeaa0436669621acbf1292f9f86edecbfa23a9f6` supplies1300s trigger and600s rescue budget, rather than changing helpers under the two existing unit-init controllers. Release path on tpu-ag: `/home/lishengping/xd/projects/controller_releases/eeaa043-ckpt600`; mutable registry/log/aot directories link to the shared canonical directories. Runtime, controller header and first checkpoint must be verified at launch.

Borrow retained FLEX_START `llm-jax-v6e-1-0` in europe-west4-a, worker0, through `prepare_train_aot_on_worker.py`; no lifecycle ownership or cleanup. Exact target AOT: v5p-128/s28000. Keep old H on `xd-v5p-128-pythia-h-1030` in us-east5-a until new AOT is verified, then use the journaled hot-switch entrypoint from the versioned release and retain the same training TPU. Preserve both existing unit-init RUNs and their16k guards.

CPU launch evidence and controller hash manifests: `/data0/xd/bam_diagnostics/pythia28-h-memory-mlp-bias-20261031/`. Tests cover formal budget/recipe/no-decay, exact initialization parity/live gradients, native/Pallas outputs and all gradients, summary writer and coordinated checkpoint recovery, plus full BAM regression. Failed preliminary naive-RNG implementation was corrected before sealing; final logs are `cpu-final.log` and `bam-regression.log`.

## Verified launch / parent closeout

Runtime `95a57d35f0beee020812cc595de92c1e4d4db44e` pushed on main; sealed config matches. CPU:23 targeted/host tests and47 BAM regressions pass. Exact AOT SHA256 `ecff4d818e62b632db98ace4c3dd0e6bdaf0b74b0b4ba04540c6842c1e657997`,165063290 bytes; manifest verified. AOT state `/home/lishengping/xd/projects/aot_runs/95a57d3-29352af0.json`; borrowed FLEX remains retained.

Journaled hot handoff saved old H checkpoint18331, no unsaved tail; boundary2026-10-31T08:56:18Z. New RUN logical ID310463 retains `xd-v5p-128-pythia-h-1030`, us-east5-a. All16 workers run the exact source and log `Loaded compiled function!`; actual first update0, then registry FIRST_STEP8; finite loss. CP0 committed with matching cursor/globalbatch1024/16 hosts. Saved runtime config confirms per-device16/no accumulation, bias enabled and Orbax1200; controller header confirms1300/600 and exact authoritative release hashes. Steps20..39 mean.31465, median.316step/s, versus parent.31690 (-.71%), matched health.

Old H stopped18331; final common native window18k. Versus QK80 gap-.0090044/R1.087606 (final5 -.00937676/1.089336); versus pre-mask+QK80 gap-.0071418/R1.068248 (final5 -.00751572/1.070355). Its extra gain remains; versus Mudd multiplier continues declining1.753874(12k)->1.673129(16k)->1.661949(18k), final gap-.0445244. Versus MHA18k gap-.111787. Original16k prior1.02 versus the older gamma comparator measured1.036092;28k terminal untested. Stop is the user's bias replacement, not a losing-result verdict.

Parent zero preemptions, UE5a only: assignment2026-10-30T15:30:10Z to2026-10-31T08:56:18Z; complete READY lease2026-10-30T15:36:02Z to2026-10-31T08:56:18Z,62416s=17h20m16s. New FIRST_STEP preceded old stopped-registry independent check and resource-free `already_closed=true` bookkeeping. Old TB SYNC_OK. Physical TPU/queue retained under the new RUN.

Unified group remains `pythia28-h-unit2x2-1031`,2000-step clock from unit/no-gamma, next4000; remove stopped H active membership and add new bias H without moving calendar. Preserve both unit-init16k stop guards. New bias16k is a review, not an automatic stop. Pair2x2 plots still retain original H and stopped H+gamma references; bias is an additional intervention, not a replacement2x2 cell.
