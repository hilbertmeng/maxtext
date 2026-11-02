# UnitH per-head content adapters

Status: write-full, write+O-r16 and the resumed UnitH comparator completed13500. Six adapter prefixes were stopped earlier; pre-RMS write+O subsequently stopped5498 after user confirmation. Only pre-RMS write-full remains in training. All new2.8B transfers are config-only; no2.8B trainer was launched. All Medium experiments prepare transfer to2.8B.

Worktree `/data0/xd/mediumprop-unit-h-content-adapters`; branch `codex/mediumprop-unit-h-content-adapters`. Main source/fresh infrastructure audit: `fb9e8689`. Runtime manifest: `/data0/xd/bam_diagnostics/mediumprop-unit-h-content-adapters/runtime.json`. Direct parent `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneTruePile` (runtime aa7d8ec), initially paused at9341, then resumed on2026-11-02 at the user's request and completed13500; final mature comparison ends13400. No baseline extrapolation. MHA reference `BamMHAMediumPropC256TruePile`.

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

Six new v5p16 RUNs,13500 steps,T4096,global128/per-device16, checkpoint200/latest2, TBrotation500. UE5a selected from the latest matching5-6h P0 Medium leases recorded in main regional history; backup EW4b during recovery. Borrow retained FLEX llm-jax-v6e-1-0/EW4a, never delete it. Single shared CPU proof by exact sealed hash; serialize AOT through its worker lock while training queues provision independently.

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

## Nonlinear write+O rank16 replacement

User authorized replacing write+V rank16 with write+O rank16 GELU, from scratch on its existing READY trainer. New RUN suffix ContentWriteOAdapterR16GeluTruePile (controllerID310516), actualTPU xd-v5p-16-310514-maxtext/UE5a retained. Direct comparator is the linear write+O rank16 RUN, not an ancestor substitute. Both attention-write and complete-localO adapters become x+GELU(x down)up; each layer/head has its own weights. No V adapter. Up stays zero; down fan-in normal. No added parameters; widths2496/1121/tail3888, total432113410 remain exact matches to the linear comparator. Embedding/private writers, norms, gates, common read scale initialization1, scan, Splash and both Pallas kernels stay unchanged.

GELU cannot commute through context attention; nonlinear V adapters are deliberately rejected. A nonlinear adapter has no global Delta matrix: compact health reports ||.5 down up||_F/||I||_F as the zero-input adapter Jacobian update, tagged jacobian0_update_over_identity. Generic parameter health remains available. This is not the activation-dependent effective output change.

Prior: MHA-reference new/linear-parent gain1.020 (rough range0.99-1.05); speed-.5% versus linear write+O. Unlike the linear r16 cases, its two projections cannot be collapsed to a fixed matrix at each optimizer step.

Infrastructure audit against main9aa2dd5: post-fb9e8689 change16ee2385 adds three default-OFF optional packed-key/fused-QKNorm/Splash-remat flags and corresponding model paths. No later train/checkpoint/summary/optimizer operational repair is missing. Preserve the verified linear-parent runtime behavior; do not introduce those optional optimizations into this GELU comparison. TBrotation500, construction/begin_step and5 summary regressions retained. CPU gate covers identity/learning, independent nonlinear forward+all-grad oracle, zero-input Jacobian health, exact formal-parent parameter trees/budget,8-device FSDP layout, full native/Pallas model logits+all-gradients, unchanged linear adapter cases, and writer rotation. Existing five RUNs retain sealed ddcc67a.

Finish new CPU/config/AOT gates before checkpointing the old write+V and handing over. Collect the old terminal native report once at handoff. Confirm new source/AOT/data/parameter count/FIRST_STEP before transferring ownership; then finalize old RUN with resource deletion disabled and remove it from active group. New RUN cadence2k/reviews4k+8k; review recommendations require user confirmation. Original-group4k calendar remains fixed during preparation.

Sealed GELU runtime b84abc2;12 focused CPU checks and config seal passed; FIRST_STEP/source/data/AOT/parameter audit passed. Startup20..99 0.646145step/s (-0.26% vs linear write+O).

## Closed prefixes and current ownership

User-confirmed4k decisions stopped write-r16 at4426 and write+O-full at4179. Both final checkpoints committed, local TB SYNC_OK, and node/queue independently absent. Write+V-r16 stopped3648 under the user's explicit hot-replacement instruction; its TPU310514 is retained by GELU write+O, so its closeout correctly skipped resource actions after independent stopped-registry verification. Full13.5k endpoints are unmeasured for all three. Final native loss-report JSON and closure audit are retained in the artifact directory.

Four active RUNs: write-full310510, write+O-r16310513, write+V+O-r16310515, GELU write+O310516 on physicalTPU310514. Group clock transferred to linear write+O before removing the former clock; reported4k calendar phase preserved, next regular report6k. Stop requires user confirmation. Relevant3.2k effective-update health is saved as adapter-health-3200.json with full layer-distribution figure; it measures matrix change, not activation-energy change.

The remaining head-sharing proposals await mature GELU/linear-parent trends. The user subsequently authorized only the shared-full-write route1, described below; it is active, while the other sharing configurations remain unimplemented and unlaunched.

## Shared full-write adapter: user-authorized route1

Only the full-write head-sharing route was subsequently authorized. Existing four trainers continue; the other head-sharing proposals remain deferred pending later linear/GELU trends. Direct parent: ContentWriteAdapterFullTruePile on UnitH. New suffix ContentWriteAdapterFullSharedHeadsTruePile, controller/physicalTPU310517, UE5a v5p16, plan13500. Borrow retained FLEX for AOT; recovery backup EW4b per current regional direction.

Each of the first17 layers stores one Delta[75,75], initialized zero, shared by its16 attention-write heads; layers remain independent. Effective map x+xDelta follows each head's original content RMSNorm. Addresses/gates remain per-head, O/V/Q/K unchanged, and L17 still has no writer. No bias or learned identity; ordinary WD acts only on Delta. Projection remains outside Pallas, prepared attention factor still receives divisor1, both kernels enabled. Head-sharing gradients sum contributions from all heads, not an accidental single-head gradient.

Adapter parameters1530000 ->95625, saving1434375 (0.996094 W_Q). Front17 ordinary/private widths2494/1119 ->2510/1128, complete branch parameter ratio1.999646:1. Tail3904 ->3877 handles integer repayment, total432110635 (parent-1575,0.000365%). Repayment means preserving the front-layer2:1 branch allocation, not claiming that tail-adjusted aggregate refund increments are exactly2:1.

The matrix remains small and replicated with logical axes(None,None), avoiding the invalid attempt to shard a singleton head or width75 over8 devices. Verify the whole-model2% sharding threshold on the formal8-device layout without relaxing it. Health keeps update_over_identity: for a shared matrix, its Frobenius/identity statistic equals the RMS over identical per-head matrices. No per-head metric duplication or activation capture.

Prior versus direct full-write parent: MHA-ref gain1.015 (rough range.98-1.04), speed+.5%/approximately flat. Sharing parameters does not reduce the16 per-token head projections by16; replenishing MLPs slightly raises nominal compute. A single ordinary matrix contraction across all heads may improve layout and GEMM efficiency; measure rather than infer from parameter count.

Focused gate: exact head tying and summed-parameter/input gradients, identity initialization, shared/unshared effective-norm equality, formal parameter budget and unchanged controls,8-device replicated-small-matrix tolerance, full native/Pallas logits+all-gradients, unaffected linear/GELU adapter checks,5 writer-rotation regressions. No standalone diagnostic TPU required. Infrastructure audit: branch began fresh-mainfb9e8689; later16ee2385 adds optional default-OFF math-preserving model optimizations, no missing operational checkpoint/log/optimizer fix; preserve parent lineage. Main reviewed at the hash recorded in headshare-runtime.json. New RUN must pass config seal/AOT/actualFIRST_STEP before registration as launched.

Sealed3e3e1e0;13 focused CPU checks passed; FIRST0/source/AOT/data/432110635 params and steps20..99 verified. Startup0.671414step/s (-0.08% vs direct full-write parent), effectively flat. Five active trainers now; new shared-head route has no mature loss conclusion.

## Three-adapter rank16 closeout and native-step reporting groups

User stopped write+V+O rank16 at5857. Final mature5800 gain remained small: gap-.0007858/R1.0036158 vs UnitH; later windows mildly improved, so this is weaker comparative benefit, not proof of zero benefit or a full-schedule endpoint. CP5857 committed, local TB SYNC_OK, TPU310515 node/queue independently absent. One same-zone preemption at05:36:09 was automatically recovered by05:43:50; final P1, and both observed READY leases are recorded in regional history.

Four active trainers: write-full310510, write+O-r16310513, GELU write+O310516 and shared-full-write310517. At the user's explicit request, all four now share the original mediumprop-unit-h-content-adapters report group and UnitH/MHA comparison. Former child groups are archived, members moved with history, and their watchers stopped. Fixed2k public milestones remain on clock310513; each RUN retains native4k/8k review points. Two additional linear pre-RMS write-adapter RUNs are being prepared under the same group, described in mediumprop_unit_h_content_adapters_pre_rms.md. Stop recommendations require confirmation. Architectural parent records remain separate from the common reporting comparator.

## GELU closeout and resumed common comparator

User stopped GELU write+O at committed6406/13500. Mature comparisons through6.2k remain behind the linear write+O counterpart at every common2k+ window (+.00059..+.00368); the4.4k catch-up does not persist. Relative to common UnitH, final5mean gap-.0003465/R1.0016015, final6.2k-.0004324/R1.0020156: effectively little retained gain. The prior1.02 multiplier over the linear parent is unsupported; full endpoint unmeasured. Step20..99 speed.646145 versus linear.647808 (-.26%); parameter/MLP budgets identical. CP6406 committed, localTB SYNC_OK, stopped registry independently checked before metadata-only closeout. No TPU deletion: physical310514 is retained by the resumed UnitH comparator.

P0; one READY lease UE5a/v5p16,2026-11-02T05:07:51Z..07:56:27Z,10116s (2h48m36s), manual stop/handoff. Full lease JSON and native final report are in the artifact directory. Five adapter experiments remain active, one unified group; GELU removed with history, original2k calendar preserved.

At the user's request the common UnitH comparator resumes from9341 to13500 using original runtimeaa7d8ec and existing v5p16/s13500 AOT, on retained physicalTPU310514. Preserve source, initialization, optimizer/LR plan and regional data/output prefix. Both worker hashes/AOT loaded, restored9341/cursor, first logged9342 and committed9400 verified. Launcher starts07:58:23Z. This fills previously missing9.2k+ direct comparisons without extrapolation. Baseline resume is owned and monitored separately from the five plotted experimental lines; its own historical H comparison need not add a plot column.

## Shared-head full-write closeout

User stopped shared-head full write at6611/13500. Versus common UnitH, mature6.4k gap-.0009236/R1.0044663, last5mean gap-.0015143/R1.0070493. Early near-parity with independent-head full write does not persist: sharing retains gain but the gap to the independent route widens. Adapter parameters fall1530000->95625, saving1434375 and refunding to ordinary/private MLPs, while startup speed stays flat(-.08%); no M-cache change. The extra MLP capacity has not compensated the lost head-specific transform freedom over this prefix. Full endpoint unmeasured;1.015 new/full-parent prior unsupported.

P0, single UE5a/v5p16 READY lease2026-11-02T05:59:07Z..08:48:51Z,10184s (2h49m44s), manual stop. CP6611 committed, localTB SYNC_OK, node+queue independently absent. Four adapter RUNs plus the resumed original UnitH comparator remain active. The shared-head RUN is removed from the unified group with history. After the original clock finishes, transfer clock to pre-RMS full-write; its published native2.4k means reported milestone2k/next4k, not the old clock's12k. Keep the single group and native4k/8k recommendations requiring user confirmation.

## Completed matched endpoint

Both original surviving adapters and the resumed common UnitH comparator completed13500. All final checkpoints/TensorBoards are saved and all three TPU nodes and queues independently absent. UnitH keeps its original aa7d8ec source, optimizer/LR plan and data cursor across the9341 pause/resume. Final five common windows are12600..13400; ratios use the original matched MHA.

| Route | Comparator | Final5 gap | Final5 MHA-ref ratio | Outcome |
|---|---|---:|---:|---|
| full | UnitH | -0.0031152 | 1.0175027 | Completed13500;1.020 prior close |
| wo | UnitH | -0.0021474 | 1.0120523 | Completed13500;1.040 prior optimistic |
| unit | Original H | +0.0047828 | 0.9738255 | Completed13500 after9341 resume |

Full-write-only is the strongest completed adapter recipe on this chosen transfer base. Its independent full matrices outperform the more elaborate write+O-r16 option; a capacity-only explanation is not isolated from factorized optimization. This comparison is internal to UnitH: unit initialization itself still trails original H on Medium. The stopped write-r16/GELU/head-shared variants do not establish full-schedule endpoints. Required2.8B configurations are documented separately in pythia2p8_h_unit_content_adapters.md.

Pre-RMS write+O was user-stopped5498 after the4k recommendation. A brief4.8k improvement does not persist against the matched post-RMS route; final5 vs UnitH gap-0.0006086/R1.0026985. CP5498/TB SYNC_OK, node+queue independently absent. Pre-RMS write-full continues in the same group; the next review remains native8k, recommendations pending user confirmation.

All below are v5p16/UE5a, P0. Complete chronological READY lease history for these closeouts (UTC):

| Route/stint | READY start | End | Duration | End reason |
|---|---|---|---|---|
| unit/1 | 2026-10-30T00:15:23Z | 2026-10-30T04:11:56Z | 3h56m33s | Manual pause9341 |
| full/1 | 2026-11-02T03:11:04Z | 2026-11-02T09:17:44Z | 6h06m40s | Completed13500 |
| wo/1 | 2026-11-02T03:09:15Z | 2026-11-02T09:21:48Z | 6h12m33s | Completed13500 |
| unit/2 | 2026-11-02T07:58:26Z | 2026-11-02T09:47:10Z | 1h48m44s | Completed13500 |
| prewo/1 | 2026-11-02T07:33:31Z | 2026-11-02T09:58:30Z | 2h24m59s | User stop5498 |
