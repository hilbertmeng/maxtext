# UnitH per-head content adapters

Status: six trainers passed FIRST_STEP and startup20..99 audits on repaired runtime ddcc67a.63 unique CPU checks covered, including8-device sharding. No loss conclusion is claimed here. User authorized these six experiments on the learnable common-read-scale H initialized to one. All Medium experiments prepare transfer to 2.8B.

Worktree `/data0/xd/mediumprop-unit-h-content-adapters`; branch `codex/mediumprop-unit-h-content-adapters`. Main source/fresh infrastructure audit: `fb9e8689`. Runtime manifest: `/data0/xd/bam_diagnostics/mediumprop-unit-h-content-adapters/runtime.json`. Direct parent `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneTruePile` (runtime aa7d8ec), paused checkpoint9341; mature direct comparison ends9200. User previously declined resuming this parent. No extrapolated baseline after9200. MHA reference `BamMHAMediumPropC256TruePile`.

## Design and prior

All adapters are layer/head-specific, data-width75, bias-free. Full version stores only zero Delta[N,75,75], applying x+xDelta. Rank16 stores fan-in normal down[N,75,16] (std1/sqrt75) and zero up[N,16,75], applying x+(x down)up with no additional alpha/r. The fixed identity is not decayed; learned Delta/factors follow the parent's optimizer WD.

Attention write adapter follows the parent's content RMSNorm and precedes gates/address outer products. O adapter transforms complete localO, including dynamic/static gates and learned common read scaling. V adapter transforms the aggregated context: per-head linear adapters commute with token aggregation. Both adapted outputs feed the residual W_O and attention writer. Q/K, M-shift, embedding and private-MLP writers remain unchanged. Layer17 has enabled read adapters, no dead attention write adapter.

Pallas read/write remain enabled, without kernel modifications: read adapters run outside the fused read/Splash; original content RMSNorm and write adapter run outside the write kernel, whose attention factor receives divisor1 to avoid a second norm. Private writer retains divisor1. Native and Pallas must be checked against each other over the whole-model forward and every parameter gradient after random adapter perturbations, including read scale and both LoRA factors.

Parameter repayment keeps the ordinary/full-private-write branch ratio approximately2:1 on the first17 layers; final ordinary MLP handles integer remainder. No per-layer width mechanism, single-layer scan first17 plus explicit tail. Parent432112210 params; dense matches exactly, LoRA +1200 (<0.0003%). No inference M-cache increase. Parameter-neutral dense FLOPs largely repay the extra dense adapter MACs through narrower MLPs; wall time may worsen from smaller grouped GEMMs and moving attention content norm out of the fused writer.

| Arm | Adapter | New params | Ordinary/private/tail | Total params | Prior MHA-ref gain vs UnitH | Speed prior vs UnitH | TPU ID |
|---|---|---:|---|---:|---:|---:|---:|
| 1 | ContentWriteAdapterFull | 1530000 | 2494/1119/3904 | 432112210 | 1.020 | -3% | 310510 |
| 2 | ContentWriteAdapterR16 | 652800 | 2503/1124/3910 | 432113410 | 1.015 | -1% | 310511 |
| 3 | ContentWriteOAdapterFull | 3150000 | 2477/1111/3879 | 432112210 | 1.030 | -5% | 310512 |
| 4 | ContentWriteOAdapterR16 | 1344000 | 2496/1121/3888 | 432113410 | 1.040 | -2% | 310513 |
| 5 | ContentWriteVAdapterR16 | 1344000 | 2496/1121/3888 | 432113410 | 1.020 | -2% | 310514 |
| 6 | ContentWriteVOAdapterR16 | 2035200 | 2489/1117/3883 | 432113410 | 1.035 | -3% | 310515 |

Most favored: write+O rank16; simplest candidate: write rank16. Write+O vs write+V rank16 have identical budgets/MLP widths and directly test the O-before-V judgment. Full vs low-rank also reallocates saved capacity to MLP, so interpret it as the parameter-matched architecture comparison. Ratios are priors, not measured results; endpoint comparisons require matching parent coverage (currently9200).

## Training and validation contract

Six new v5p16 RUNs,13500 steps,T4096,global128/per-device16, checkpoint200/latest2, TBrotation500. UE5a selected from the latest matching5-6h P0 Medium leases recorded in main regional history; backup UC1a/EW4b during recovery. Borrow retained FLEX llm-jax-v6e-1-0/EW4a, never delete it. Single shared CPU proof by exact sealed hash; serialize AOT through its worker lock while training queues provision independently.

Generic training health ON, internal NN/BAM activations OFF. Existing common-read-scale signed values remain visible. Add exactly one effective Delta Frobenius-over-identity metric per layer and active route (17,17,35,35,35,53 per arm); for LoRA measure down@up with rank-sized Gram matrices. Do not substitute the separate factor norms for effective adaptation.

Pinned CPU gates: initialization/learning, V commutation, effective-Delta metrics, formal six parameter trees, six native/Pallas whole-model forward/all-gradient cases, shared47 BAM regressions,5 summary-writer rotation/resume cases. Verify config seal for every class, exact source/AOT/data/parameter count at FIRST_STEP. Preserve all launch proofs in the artifact directory. New worktree inherits current main rotation construction and begin_step; unit tests cover rotation. No historical infrastructure frozen.

Report as one group, fixed2000-step cadence and4000/8000 review milestones; reviews are recommendations pending user confirmation before stop. One shared loss-gap/ratio figure with six curves from600 steps, parent UnitH only (MHA denominator). No routine live-loss diary in exp.py. Retain actual per-run plot horizons and parent9200 limit. Quiet shared waiting between milestones; health at meaningful reports, no repeated TB synchronization for loss-only monitoring.

## Pre-first-step layout repair

Initial sealed commit a59892a passed62 single-device numerical/regression checks and all6 AOTs. Runtime startup rejected the model before FIRST_STEP: new adapter tensors used q_heads (tensor parallelism), while the formal8-device mesh is FSDP. Replicated adapters raised parameter-per-chip overhead to2.4%-5.3%, above the2% launch threshold. This was a validation coverage gap, not a numerical mismatch. No training loss was produced.

Repair: assign the independent adapter head axis to the existing embed/FSDP parameter rule; the75-wide within-head axis remains intact. Keep the safety threshold, all parameters, initialization and math unchanged. Add an explicit8-device virtual-CPU shaped-state test for all6 classes, requiring every adapter leaf to shard by8 and whole-model overhead below tolerance. Retain all6 READY trainers, recompile AOT for the repaired source, then restart on those same resources. The existing62 numerical checks remain applicable to the axis-only change; rerun identity/health and formal-tree checks, plus the new multi-device gate.

## Startup audit

Runtime `ddcc67af56b4dd37b223e8f4454bb8405035bf0b`. All six AOT manifests/config seals/source hashes/data paths/parameter counts verified; first step0, baseline widths and counts listed above. Formal safety check remains2%; measured shaped8-device overhead0.226%. All six original READY training resources retained through the pre-first-step repair; no diagnostic resource was created. Borrowed FLEX retained. Generic health ON/internal+BAM OFF,17/35/53 effective-Delta scalars as appropriate.

| TPU ID | Arm | step/s20..99 | Historical UnitH delta |
|---|---|---:|---:|
| 310510 | ContentWriteAdapterFull | 0.671954 | -0.62% |
| 310511 | ContentWriteAdapterR16 | 0.664169 | -1.77% |
| 310512 | ContentWriteOAdapterFull | 0.652595 | -3.48% |
| 310513 | ContentWriteOAdapterR16 | 0.647808 | -4.19% |
| 310514 | ContentWriteVAdapterR16 | 0.648231 | -4.12% |
| 310515 | ContentWriteVOAdapterR16 | 0.630704 | -6.72% |

These startup timings are descriptive: newer parent read-kernel infrastructure and adapter-specific health differ from historical UnitH. Rank16 is slower than dense at both matched route sets despite fewer adapter parameters; the speed priors were optimistic for the low-rank routes. Loss/capacity judgments remain open. Group clock is write-rank16 (310511), fixed2k cadence/4k+8k reviews, stop only after user confirmation. Main exp.py records only config/provenance/startup/final conclusions, not routine loss reports.
