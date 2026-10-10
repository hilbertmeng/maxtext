# MediumProp attention parameter allocation: D1152

Worktree `/data0/xd/mediumprop-attention-budget`; branch `codex/mediumprop-attention-budget`; forked latest refactor-bam c7dab980. All four are18-layer AllLocal TruePile4096, independent GELU-LoRA MLP writes at layers1/4/7/10/13/16. Pure-JAX BAM core; Splash attention and SEQ_MINOR enabled. Parent: BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile. Budget432121200 (original MediumProp MHA).

Common loss baselines:18-layer standard parent above, and `BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` (BAM27). The live registries and main ledger retain both, comparing wider attention with greater depth under the same parameter budget. BAM27 historical speed is .385step/s, C256; speed comparisons with these Splash runs are runtime-unmatched. Adding this baseline does not change runtime444cb5f or restart training.

Training: spot v5p-16, primary UE5a based on recent longer leases; UC1a/EW4b passive alternatives if primary remains queued. 13500steps; 200-step loss windows; report about1000steps, review2800/5000. Compiler: idle user-owned FLEX_START llm-jax-v6e-1-0, EW4a worker0; borrowed without lifecycle ownership.

| RUN | TPU | Heads / M / C | Attention + embedding address R / MLP R | MLP widths repeated6 | Parameters | Terminal loss bet vs standard | Historical-speed bet |
|---|---|---|---|---|---:|---:|---:|
| `BamMediumPropD1152H24K96V32C8AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-16-310101-maxtext` | 24x96 / 96x32 / C8 | 384 / 192 | [3445, 3356, 3446] | 432112464 | -0.002 | 0.43step/s vs .520 |
| `BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-16-310102-maxtext` | 24x96 / 96x48 / C12 | 512 / 256 | [3256, 3124, 3257] | 432114000 | -0.010 | 0.40step/s vs .520 |
| `BamMediumPropD1152H32K72V48C12AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-16-310103-maxtext` | 32x72 / 72x48 / C12 | 768 / 384 | [2919, 2700, 2919] | 432125504 | +0.006 | 0.39step/s vs .520 |
| `BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdTruePile` | `xd-v5p-16-310104-maxtext` | 32x72 / 72x64 / C16 | 1024 / 512 | [2484, 2155, 2484] | 432119616 | +0.015 | 0.36step/s vs .520 |

Independent projected QK uses one quarter of head width for RoPE; LocalQK reads full M and truncates to the remaining three quarters. Embedding writes use the full attention head count; MLP output reshapes directly to D/K write heads (12 for K96,16 for K72), with no padding/content projection. MLP write sqrt-N scaling uses its own head count. Paired address-alignment health is omitted when heads differ, because there is no headwise pairing; remaining generic+concat health follows the parent.

AOT correction: the compiler host traces with JAX_PLATFORMS=cpu but targets v5p-16. Splash selection must recognize the explicit TPU compile_topology rather than silently compiling C256. Ordinary CPU fixtures still use C256. Compile target selection, exact four full parameter trees, unequal-head outer products and their gradients, tiny full-model gradients, and full BAM regression gate training. Shared CPU gate cached per sealed runtime commit avoids repeating the suite four times; four training prequeues run concurrently while one borrowed compiler serializes AOTs.

Speed bets were made before selecting the newly optimized runtime; historical .520 was C256. Do not attribute any Splash/runtime speedup to the head/M allocation. Loss bet order: H24K96V48 < H24K96V32 < standard < H32K72V48 < H32K72V64.

Startup verified: all four use runtime444cb5f, loaded the exact AOT and executed real steps on UE5a, reading the UE5a TruePile4096 replica. CPU checks passed46 BAM regressions plus unequal-head/model checks. Early throughput (steps20-99) respectively .4217/.3962/.3589/.3416, versus historical standard .520: -18.9%/-23.8%/-31.0%/-34.3%; generic+concat health matches, attention runtime does not.

Main follow-up3745492e removes backend detection entirely: bam_splash_attention (defaultTrue) controls the core; CPU fixtures explicitly disable it. All47 main regressions passed. The sealed444cb5f runtime retains explicit TPU-target detection and already traced Splash successfully, so these four executables need no replacement.

All8 passive spot candidates (`310101`–`310104`, suffixes `-ew4b`/`-uc1a`) released; both TPU nodes and queued resources verified absent. Cleanup journal: `/data0/xd/bam_diagnostics/mediumprop-attention-budget-cleanup.json`. User-owned FLEX_START compiler retained.

## Forward theoretical FLOPs

Common ideal causal pair count T(T+1)/2, T4096,18layers; multiply and add each count as one FLOP. Fixed reference1W_Q=2BT(1200²), reported per-layer average even when D1152. Includes embedding-write projections/outer and LM head; excludes input embedding lookup, elementwise activation/normalization/gating/softmax, health, optimizer, backward/remat and hardware padding. All18 nominal attention writes counted. Full-K shared-rank4 basis read/expansion follows the source before QK truncation.

| Configuration | Dense W_Q | QK+AV W_Q | M contractions/Gram W_Q | Total W_Q | vs standard | Transformer-only delta |
|---|---:|---:|---:|---:|---:|---:|
| standard16x75 M75x32/C8 | 14.33160 | 3.41417 | 0.16504 | 17.91081 | +0.00% | +0.00% |
| 24x96 M96x32/C8 | 14.42480 | 6.55520 | 0.29400 | 21.27400 | +18.78% | +21.91% |
| 24x96 M96x48/C12 | 14.42320 | 6.55520 | 0.44707 | 21.42547 | +19.62% | +22.77% |
| 32x72 M72x48/C12 | 14.42222 | 6.55520 | 0.43449 | 21.41191 | +19.55% | +22.46% |
| 32x72 M72x64/C16 | 14.41978 | 6.55520 | 0.58756 | 21.56254 | +20.39% | +23.12% |

QK+AV doubles nearly: H×K1200→2304 (+92%); parameter matching keeps Dense arithmetic almost flat (+.62–.65%). Input embedding lookup has no Dense FLOPs, so moving its saved parameters to active projections slightly increases Dense FLOPs; LM head shrinks with D. Learned full-M static reads, compression, shared-rank4 reads/Gram and outer products are counted separately. Applying the same old C256 rounded pair count T(T+256)/2 to every arm gives +19.64/+20.47/+20.40/+21.23%; the main table isolates architecture from that implementation overhead.

24x96 V48 and32x72 V48 have almost identical nominal FLOPs, while32heads measured9.4% lower throughput; total FLOPs cannot explain this difference. Source accounting and category MACs: `/data0/xd/bam_diagnostics/mediumprop-attention-budget-flops.py` and `.json`.

BAM27 depth reference (`BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`):27 layers, unchangedD1200/head16x75/M75x32/C8, nine privateMLP writes, widths[2311,2184,2311]. Same ideal-causal whole-model forward accounting versus18-layer standard gives +9.98366%; Transformer-only+11.53550%. Dense-0.00494%,QK+AV+50%,M contractions/Gram+49.55117%. Matched-C256 pair accounting gives+10.45286%. Historical.385 versus.520step/s (-25.96%) remains larger than this arithmetic increase; no implementation cause established. Category MACs: `/data0/xd/bam_diagnostics/mediumprop-bam27-flops.json`.

## User pause after2000 window

All four paused with committed checkpoints, training TPU nodes/queues released; resumes retain the original13500-step LR plan. Pause checkpoints2054/2043/2052/2038. All use full-M shared-rank4 LocalQK plus static reads; no DirectC8 LocalQK. Loss and lease artifacts: `/data0/xd/bam_diagnostics/mediumprop-attention-budget-report-2000.json`, `mediumprop-attention-budget-pause-state.json`, `mediumprop-attention-budget-leases.json`. Local TB tails synced.

| RUN shorthand | vs standard@2000 | vs BAM27@2000 | Last5 vs standard / BAM27 |
|---|---:|---:|---:|
| H24K96V32C8 | -0.016376 | +0.004201 | -0.019071 / -0.001981 |
| H24K96V48C12 | -0.025139 | -0.004562 | -0.028836 / -0.011746 |
| H32K72V48C12 | -0.018681 | +0.001896 | -0.020900 / -0.003810 |
| H32K72V64C16 | -0.025681 | -0.005104 | -0.026059 / -0.008969 |

Early gains shrink throughout. H24V32 and H32V48 crossed behind BAM27 at2000; H24V48 and H32V64 remain ahead by~.005, but the advantage versus BAM27 is not stable. H32V64 is substantially better than H32V48; H24V48 is better than H32V48. This supports testing address capacity alongside attention allocation, but does not isolate heads, width, D or MLP effects. Results are provisional, not terminal bet judgments.

All four had0preemptions, one UE5a v5p-16 READY lease each (UTC):

| TPU ID | READY start | Pause boundary | READY duration |
|---|---|---|---|
| xd-v5p-16-310101-maxtext | 2026-10-10T04:14:58Z | 2026-10-10T05:40:37Z | 1h25m39s |
| xd-v5p-16-310102-maxtext | 2026-10-10T04:14:52Z | 2026-10-10T05:45:03Z | 1h30m11s |
| xd-v5p-16-310103-maxtext | 2026-10-10T04:14:25Z | 2026-10-10T05:54:21Z | 1h39m56s |
| xd-v5p-16-310104-maxtext | 2026-10-10T04:14:32Z | 2026-10-10T05:58:36Z | 1h44m04s |


## DirectC restarts of the two large-M arms

New independent runs, from step0: H24K96V48/C12 and H32K72V64/C16. Same worktree/branch and attention-budget recipe; only LocalQK dynamic read switches from full-M shared-rank4 to independent per-head keys on the existing shared C projection. Full-M static Q/K reads retained, no key bias; no separate QK compression. Main Splash config-only selection3745492e incorporated. Repaid additional parameters to nearest MHA-budget widths.

| Class | TPU | MLP widths | Expected params | Terminal gap bet vs standard DirectC8 | Speed bet vs original large-M rank4 |
|---|---|---|---:|---:|---:|
| BamMediumPropD1152H24K96V48C12AllLocalMLPWriteIndependentEveryThirdDirectCTruePile | xd-v5p-16-310105-maxtext | [3192,3060,3194] | 432131280 | -.015 | -2% (~.388step/s) |
| BamMediumPropD1152H32K72V64C16AllLocalMLPWriteIndependentEveryThirdDirectCTruePile | xd-v5p-16-310106-maxtext | [2313,1984,2314] | 432115008 | -.010 | -3% (~.331step/s) |

Each compares its paused original rank4 arm, the completed standard18 DirectC8, and BAM27. Original rank4 arms provide paired loss only through2000. Health, TruePile4096, original13500-step schedule retained; report about1000steps, review2800/5000. Formal spotv5p-16 UE5a primary; retain EW4b/UC1a spot alternatives only if primary waits5min. Retained FLEX_START llm-jax-v6e-1-0 EW4a used only as compiler, never auto-cleaned. Targeted CPU gate checks both full parameter trees and actual-K/V/C tiny consumed gradients; existing broad runtime paths were already verified.

Sealed runtime4981467; targeted gates passed both exact full budgets and both actual-K/V/C consumed-gradient checks.

Both DirectC restarts loaded the exact4981467 AOT and reachedFIRST_STEP on UE5a, zone-local TruePile4096 path verified.20-99 mean speeds .3996375/.3276125: +.88%/-4.09% vs their same-health, Splash+SEQ_MINOR rank4 parents .39615/.3415875. H24 small speed gain reverses its -2% speed bet; cause not established. H32 slightly slower than -3% bet. Artifacts: `/data0/xd/bam_diagnostics/mediumprop-attention-budget-direct-registries.json`, `mediumprop-attention-budget-direct-speeds.json`; launch logs in `mediumprop-attention-budget-direct-launch/`. Compiler retained.

Monitoring now also reports advantage ratios versus standard DirectC8 and BAM27: `(MHA18 loss - RUN loss) / (MHA18 loss - BASE loss)`, with the same `BamMHAMediumPropC256TruePile` reference and exact common +/-25-step windows sampled every10 steps. Values greater than1 favor RUN; a nonpositive denominator is not interpreted as an advantage multiple. Keep MHA18 for both comparisons; substituting MHA27 would mix in the deeper MHA control's own degradation. Helper: `/data0/xd/bam_diagnostics/mediumprop-attention-budget-gain-report.py`, deployed read-only cache reader at `tpu-ag:logs/mediumprop-attention-budget-gain-report.py`.

## Standard16-head DirectC address expansion

Fixed D1200,16x75,QK57+RoPE18,18 AllLocal layers, private MLP writes1/4/7/10/13/16. Parent standard DirectC8; existing full-M static Q/K/V/O reads retained. Extend M75x32/C8 to M75x48/C12 and M75x64/C16, proportional attention/embedding/private-MLP address R384/R512, exact MLP repayment. Same worktree and branch as above; pure JAX, Splash SEQ_MINOR, inherited generic+concat/write health. Runtime e332529; paired full budgets and actual K75/C12/C16 forward/consumed gradients passed before launch, sealed runtime configuration verified.

| RUN | TPU ID | M / C | Address R | MLP widths | Parameters / MHA delta | Terminal bet vs DirectC8 | Matched-runtime speed bet |
|---|---|---|---|---|---|---|---|
| `BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | 310107 | 75x48 / C12 | 384 | [3744, 3528, 3744] | 432111616 / -9584 | -0.008 | -3% |
| `BamMediumPropK75V64C16EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile` | 310108 | 75x64 / C16 | 512 | [3567, 3246, 3567] | 432128512 / +7312 | -0.010 | -6% |

C12 also compares completed M75x48/C12 rank4 (~-.007290 vs rank4 V32 parent); bet ~-.002 versus that matched-shape baseline. C16 also compares the new C12 arm; both retain standard DirectC8 and BAM27. Report about1000steps with200-step windows, advantage ratios use common MHA18; review2800/5000. Training spot UE5a primary, add EW4b/UC1a passive candidates if primary waits5min. Borrowed user-owned FLEX_START llm-jax-v6e-1-0 EW4a is compiler-only, never auto-cleaned. Launch artifacts `/data0/xd/bam_diagnostics/mediumprop-direct-address-runs.json`, `mediumprop-direct-address-launch/`.
