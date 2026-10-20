# BAM Llama2Medium experiment memo

This is the authoritative, compact summary of current BAM capability and performance conclusions.
Detailed profiling history is in [`bam_profile_history.md`](bam_profile_history.md); checkpoint and
numerical diagnostics are in [`bam_checkpoint_diagnostics.md`](bam_checkpoint_diagnostics.md).
Configuration-class comments in `MaxText/exp.py` remain the source of truth for individual run
speed, stop step, loss comparison and runtime commit.

Local Q/K/V rank, routing and AlignedRow experiments are consolidated in
[`local_qkv_routing_review.md`](local_qkv_routing_review.md), with the complete
configuration inventory, matched-window evidence, confidence and open questions.

## Current XL comparison reference (2026-10-01)

User-designated current SOTA: `BamXLPropK96EmbedVOnlyQK72LLFTruePile`
(runtime22c2c5c, `/data0/xd/mediumprop-k75-embed`, `codex/mediumprop-k75-embed`).
Ongoing XL RMT combinations and newly launched XL experiments must include this RUN
as a direct loss/speed baseline; retain other decision-relevant controls.
AllLocal remains the LLF ablation control: the existing LLF-AllLocal series already
provides their paired comparison, so do not duplicate its inverse on AllLocal.
LLF itself retains AllLocal/MHA/Mudd. Compare exact common mature windows; when an arm runs ahead
of LLF, report the last common step explicitly rather than extrapolating its loss.
At16000 its last5 versusAllLocal/Mudd are-.027244/-.026602, Mudd gain1.327x.
UE5a v5p-32 .359 versusAllLocal .355 (+1.1%); this reverses the pre-run
-1.4% speed forecast and remains an unresolved timing anomaly (`!?` inexp.py).
Generic+concat health enabled; cross-family health costs are not fully matched.

## Reading conventions

- `dloss = RUN loss - BASE loss`; negative is better.
- `throughput retention = MHA XPlane step / BAM XPlane step`; larger is better.
- `step-time overhead = BAM XPlane step / MHA XPlane step - 1`; smaller is better.
- Compare speed only on the same TPU type, model shape, batch, sequence length and input settings.
- Canonical throughput uses `B=32,T=2048`, bf16 activations, `float32_logits=False`, and step 10–14
  XPlane. v5p-16 values are the 16-device mean.
- Six/eight-layer profiles establish same-shape operator behavior; training-speed decisions require
  full-24 v5p-16 confirmation.

## Capability milestones

### QK concat changes the capacity-allocation prior (2026-09-20)

`BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer` (4a4fdbe)
completed 13,500: vs `BamMediumIndependentLLFMLPPerLayerColOnly`, last five
windows 12,600–13,400 average -0.016407 (range -0.016751 to -0.015690).
Against the still-running `BamMediumIndependentLLFMLPPerLayerColOnlyNoPE32PartialRoPE`,
9,000–9,800 average -0.011695: RoPE alone does not explain the gain.
The initial -0.003 forecast substantially underestimated this package.

Update the prior: full standard Q/K projections are less indispensable than assumed;
standard RoPE32 plus BAM NoPE32 can be a better parameter allocation in Medium.
This is evidence for the combined QK concat + shared rank4 + MLP reinvestment package,
not a causal estimate for concat alone. The unsuccessful V-concat arms caution against
transferring the result to the content path: prioritize Q/K replacement and retain
full standard V while testing additive BAM content capacity. Do not automatically add
this gain to historical K64 gains; fixed QK32 truncation tests a different expansion.
QK concat speed .7208 vs ColOnly .7354 is health-unmatched and cannot establish a
clean throughput penalty. Runtime and complete configuration remain in `exp.py`.

Loss comparisons below use each run's registered parent/reference and are not additive across rows.

| Milestone | Speed | Training result | Main change |
|---|---:|---|---|
| `BamLlama2MediumRmsGateOnly` | ~0.280 step/s | completed 13,500; dloss `-0.0678` vs MHA @13,400 | RMS-normalized read-key direction plus sigmoid gate |
| `...FactorizedLocalQKNoMNorm` | ~0.325 | completed 13,500; mean dloss `-0.0078` vs Factorized @5,600–7,000 | remove whole-M read normalization |
| `BamLlama2MediumV1` | ~0.461 | completed 13,500; mean dloss `-0.0030` vs NoMNorm @12,000–13,400 | all-bf16 and multiply+reduce reads |
| `BamLlama2MediumV1WriteMulDiagonalOne` | ~0.512 | completed 13,500; mean dloss `+0.0016` vs V1 @12,600–13,400 | faster write and one diagonal-one fetched read |
| `BamLlama2MediumV1CompressAbsV8Direct` | ~0.515 | completed 13,500; mean dloss `+0.0096` vs V1 @12,400–13,400 | compress absolute V axis 32→8 |
| `BamLlama2MediumDirectPLocR256Gelu` | ~0.512 | completed 13,500; mean dloss `-0.0044` vs Direct @12,600–13,400 | `P_loc: D→256→nV` with GELU |
| `BamLlama2MediumV2` | **~0.551** | completed 13,500; dloss `+0.00014` vs Direct @13,400 | packed LocalQK, fp32 RMS statistics, diagonal-one and fast writes |

`BamLlama2MediumV2` is the current trained capability milestone. Optimized C256 and LGLL are
throughput/profile candidates layered on V2; they have semantic and short-step validation but not a
new 13,500-step capability result.

## Canonical v5p-16 full-24 throughput

### Matched MHA controls

| MHA-only path | Configuration class | Commit | XPlane | Stable log | Result |
|---|---|---:|---:|---:|---|
| standard dense dot | `Llama2MediumDotProductFullLayerProfile` | `f052fa6` | 1,258.65 ms | ~0.786 | reference |
| `BamAttention` dense control | `BamMHAControlDenseFullLayerProfile` | `f052fa6` | 1,276.37 ms | ~0.775 | 1.41% slower than standard dot |
| `BamAttention` C256 control | `BamMHAControlQChunk256FullLayerProfile` | `a1ad13f` | **1,088.61 ms** | **~0.908** | 17.25% faster than BAM dense control |

The C256 control keeps the packed segment mask and removes redundant chunk-local remat. Its custom
4D contraction is within 0.73% of generic QChunk on v6e; no BAM state or `bam/*` operator exists in
the control.

### Complete G/LGLL × U/U/S/U × BAM-MHA/BAM matrix

`U/U` is explicit layers plus optimized-unrolled query chunks. `S/U` changes only the layer loop to
scan. The v5p full-BAM G/LGLL arms explicitly use `optimized`. At those runtime commits, BAM-MHA
did not execute the full-BAM implementation: every non-`streaming_scan` value dispatched to the
same dedicated `_query_chunk_mha_control`, so its inherited selector value did not change the graph.

| Schedule | Loops | BAM-MHA class | MHA XPlane / step/s | BAM class | BAM XPlane / step/s | Retention | S/U cost MHA/BAM |
|---|---|---|---:|---|---:|---:|---:|
| G | U/U | `BamMHAControlQChunk256FullLayerProfile` @`a1ad13f` | 1,088.61 / ~0.908 | `BamV2QChunk256OptimizedFullLayerProfile` @`165b55b` | 1,455.35 / ~0.675 | 74.8% | — |
| G | S/U | `BamMHAGScanLayerFullLayerProfile` @`1d9e1e1` | 1,094.35 / ~0.904 | `BamV2GScanLayerFullLayerProfile` @`1d9e1e1` | 1,480.44 / ~0.665 | 73.9% | +0.53% / +1.72% |
| LGLL | U/U | `BamMHALGLLQChunk256FullLayerProfile` @`be4174d` | 915.39 / ~1.08 | `BamV2LGLLQChunk256FullLayerProfile` @`be4174d` | 1,245.99 / ~0.788 | 73.5% | — |
| LGLL | S/U | `BamMHALGLLScanLayerFullLayerProfile` @`d5225e2` | 1,119.09 / ~0.884 | `BamV2LGLLScanLayerFullLayerProfile` @`be4174d` | 1,534.31 / ~0.641 | 72.9% | +22.25% / +23.14% |

Optimized G C256 is 17.85% faster than legacy C256 and 22.37% faster than dense BAM. Dynamic LGLL
layer scan reduces compile latency (BAM 686.23→49.49 s) but adds about 23% step time to both models;
use explicit LGLL layers for long training.

### Sequence-length scaling at constant tokens/step

Full-24 v5p-16, C256, all-global attention, bf16 logits. T2048 uses batch 32 and T4096 batch 16.

| Length | Layers | BAM-MHA class | MHA XPlane / step/s | BAM class | BAM XPlane / step/s | Retention |
|---:|---|---|---:|---|---:|---:|
| 2,048 | U/U | `BamMHAControlQChunk256FullLayerProfile` @`a1ad13f` | 1,088.61 / ~0.908 | `BamV2QChunk256OptimizedFullLayerProfile` @`165b55b` | 1,455.35 / ~0.675 | 74.80% |
| 4,096 | U/U | `BamMHAControlQChunk256T4096FullLayerProfile` @`bbfd0ea` | 1,462.25 / ~0.677 | `BamV2QChunk256OptimizedT4096FullLayerProfile` @`bbfd0ea` | 1,910.07 / ~0.517 | 76.55% |
| 2,048 | S/U | `BamMHAGScanLayerFullLayerProfile` @`1d9e1e1` | 1,094.35 / ~0.904 | `BamV2GScanLayerFullLayerProfile` @`1d9e1e1` | 1,480.44 / ~0.665 | 73.92% |
| 4,096 | S/U | `BamMHAGScanLayerT4096FullLayerProfile` @`309448f` | 1,462.14 / ~0.678 | `BamV2GScanLayerT4096FullLayerProfile` @`309448f` | 1,937.20 / ~0.510 | 75.48% |

Doubling sequence length raises BAM/MHA throughput retention by 1.75 pp without layer scan and
1.56 pp with layer scan. BAM's fixed per-token work is amortized by the larger standard-attention
cost, but the C256 mix/fetch path still scales with sequence length, so the gain is modest. At
T4096, layer scan leaves MHA unchanged and makes BAM 1.42% slower than explicit layers, while
reducing BAM first-step compile time from about 18 minutes to 2 minutes 15 seconds.

### XL width and head-shape scaling

Full-24 v5p-32, all-global C256, layer scan, bf16 logits, per-device batch 16. Each retention is a
same-region BAM/MHA XPlane ratio. The repeated 16x128 T4096 MHA control differs by only 0.15%
between UC1a (3,128.69 ms) and EW4b (3,123.97 ms), validating the EW4b pairs. Medium rows are the
v5p-16 references above. Commit `ec0de73` differs from `011d44a` only by the Head32 sharding-audit
tolerance; it does not change the compiled computation.

| Scale / heads | BAM k/v/C | Length | BAM-MHA class | MHA XPlane / step/s | BAM class | BAM XPlane / step/s | Theory FLOPs | Theory params | Retention | vs Medium |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| Medium 16x64 | 32/32/8 | 2,048 | `BamMHAGScanLayerFullLayerProfile` @`1d9e1e1` | 1,094.35 / ~0.904 | `BamV2GScanLayerFullLayerProfile` @`1d9e1e1` | 1,480.44 / ~0.665 | +10.96% | +10.28% | 73.92% | — |
| Medium 16x64 | 32/32/8 | 4,096 | `BamMHAGScanLayerT4096FullLayerProfile` @`309448f` | 1,462.14 / ~0.678 | `BamV2GScanLayerT4096FullLayerProfile` @`309448f` | 1,937.20 / ~0.510 | +11.25% | +10.28% | 75.48% | — |
| XL 16x128 | 64/32/8 | 2,048 | `BamMHALlama2XLHead16x128C256T2048Profile` @`011d44a` | 1,400.17 / ~0.708 | `BamLlama2XLHead16x128V2C256T2048Profile` @`011d44a` | 1,754.93 / ~0.562 | **+7.89%** | **+7.28%** | **79.78%** | +5.86 pp |
| XL 16x128 | 64/32/8 | 4,096 | `BamMHALlama2XLHead16x128C256T4096Profile` @`011d44a` | 3,123.97 / ~0.319 | `BamLlama2XLHead16x128V2C256T4096Profile` @`011d44a` | 3,886.03 / ~0.256 | **+8.24%** | **+7.28%** | **80.39%** | +4.91 pp |
| XL 32x64 | 32/64/16 | 2,048 | `BamMHALlama2XLHead32x64C256T2048Profile` @`011d44a` | 1,554.72 / ~0.640 | `BamLlama2XLHead32x64V2C256T2048Profile` @`ec0de73` | 2,085.69 / ~0.475 | +10.89% | +10.45% | 74.54% | +0.62 pp |
| XL 32x64 | 32/64/16 | 4,096 | `BamMHALlama2XLHead32x64C256T4096Profile` @`ec0de73` | 3,714.83 / ~0.268 | `BamLlama2XLHead32x64V2C256T4096Profile` @`ec0de73` | 4,929.29 / ~0.202 | +11.06% | +10.45% | 75.36% | -0.12 pp |

Theory overhead is relative to the corresponding BAM-MHA Transformer block, excluding embeddings
and the LM head. Parameter counts include the complete configured BAM parameter tree. FLOPs count
matrix projections and contractions using the actual C256 block-pair count `T(T+256)/2`; they omit
lower-order norm, gate, softmax, and elementwise work. Forward/backward scales both sides similarly.

The theoretical ordering matches the width experiment: 16x128 cuts BAM arithmetic/parameter
overhead to about 8%/7%, whereas 32x64 remains near Medium because twice as many heads, wider BAM
V, and two 64->32 LocalQ/K adapters consume the width gain. Measured step-time overhead
(`1/retention-1`) is still much larger: Medium 35.3/32.5%, XL16 25.3/24.4%, and XL32 34.2/32.7%
at T2048/T4096. Moreover, theoretical FLOP overhead rises slightly with length while measured
overhead falls. The context-length gain is therefore a hardware-utilization effect, not arithmetic
amortization; BAM's small contractions remain disproportionately expensive.

### Full-24 layer-scan unroll

Commit `2646f97`, full-24 G C256 BAM on v5p-16. The `u1/u2` pair used one EW4b pod; `u4` used
UC1a after two EW4b preemptions. Stable loss is invariant to unroll within bf16 rounding: at step 0,
`u2/u4 = 10.846203/10.846224`, and their subsequent values differ by at most about `2e-5`.

| Unroll | Configuration class | XPlane | Stable log | `M+dM` add events / ms | Lowered TF / GB |
|---:|---|---:|---:|---:|---:|
| 1 | `BamV2GScanLayerFullLayerProfile` | 1,485.78 ms | ~0.664 | 24 / 3.97 | 100.06 / 1,388.07 |
| 2 | `BamV2GScanLayerUnroll2FullLayerProfile` | 1,488.24 ms | ~0.663 | 12 / 1.96 | 101.45 / 1,404.62 |
| 4 | `BamV2GScanLayerUnroll4FullLayerProfile` | 1,482.00 ms | ~0.663 | 6 / 0.98 | 103.70 / 1,433.46 |

Unroll successfully coalesces the loop-boundary matrix residual add, but increases other lowered
work and traffic; the complete-step spread is only 0.42% and shows no throughput benefit. Keep
`unroll=1`. Scan and non-scan do not start from identical random parameters: at step 10,
`u1/u2 = 10.306737/10.306740`, versus non-scan C256 `10.304788` and dense V2 `10.304796`.
The two non-scan paths agree, while `nn.scan(split_rngs={'params': True})` uses a different parameter
RNG sequence. Thus ordinary from-scratch loss cannot establish scan/non-scan numerical equivalence;
that requires mapping one parameter set into both layouts.

### Dynamic fetched-row rank 8

Commit `db94296`, full-24 G C256 on one EW4b v5p-16. The rank-8 path replaces 16 independent
fetched-row reads with eight dynamic basis reads and a multiply-reduce head expansion.

| Layers | Control class / XPlane / log step/s | Rank-8 class / XPlane / log step/s | Rank-8 throughput |
|---|---|---|---:|
| non-scan | `BamV2GRowRank8ControlFullLayerProfile` / 1,456.213 ms / ~0.673 | `BamV2GRowRank8MulReduceFullLayerProfile` / 1,456.361 ms / ~0.673 | -0.01% |
| layer-scan | `BamV2GScanLayerRowRank8ControlFullLayerProfile` / 1,482.839 ms / ~0.663 | `BamV2GScanLayerRowRank8MulReduceFullLayerProfile` / 1,481.892 ms / ~0.664 | +0.06% |

Rank-8 is speed-neutral under both layer implementations. Non-scan is 1.83% faster for the control
and 1.75% faster for rank-8; layer-scan remains useful only for its much shorter compilation.

### Generalized Factorized LocalQK rank

Rank 1 is the production FactorizedLocalQK path. Rank 2/4 reads multiple dynamic LocalQK bases,
then expands them to heads; the second contraction was paired as dot versus multiply+reduce.

Eight-layer v6e screening:

| Shape | Rank 1 | Rank 2 dot / mul | Rank 2 mul cost | Rank 4 dot / mul | Rank 4 mul cost |
|---|---:|---:|---:|---:|---:|
| Medium 16x64 | 675.37 ms | 799.89 / 707.10 ms | -4.49% throughput | 820.75 / 720.49 ms | -6.26% |
| XL 16x128 | 568.77 | 639.05 / 595.77 | -4.53% | 659.64 / 615.42 | -7.58% |
| XL 32x64 | 777.30 | 876.05 / 818.93 | -5.08% | 898.92 / 837.79 | -7.22% |

Multiply+reduce raises throughput by 13.12/13.92% over dot at Medium rank 2/4, and by
6.98–7.30% on the two XL shapes. It is the only viable rank-expansion implementation.

Commit `8235ccd`, full-24 target profiles, all paired in EW4b:

| Shape / TPU | Rank 1 | Rank 2 mul | Rank 2 throughput | Rank 4 mul | Rank 4 throughput |
|---|---:|---:|---:|---:|---:|
| Medium 16x64 / v5p-16 | 1,480.38 ms | 1,527.18 ms | -3.07% | 1,566.82 ms | -5.52% |
| XL 16x128 / v5p-32 | 1,748.65 | 1,803.44 | -3.04% | 1,842.78 | -5.11% |
| XL 32x64 / v5p-32 | 2,074.42 | 2,140.11 | -3.07% | 2,186.56 | -5.13% |

The XL32 rank-4 value is the mean of two independent step-10 traces (2,188.60/2,184.53 ms,
0.19% apart) captured after repeated spot preemptions. The target-TPU cost is strikingly stable
across model shapes: about 3% at rank 2 and 5–5.5% at rank 4. v6e preserves the ordering but
overstates the penalty. Keep rank 1 for speed; rank 2/4 requires a training-loss gain large enough
to justify this measured target cost.

### Grouped fetched-M rank

Commit `e2c645d`. Fetch rank 2 learns two signed head mixtures, fetches two matrices, and assigns
each contiguous half of the BAM heads to one fetched matrix. `D/M` below means dot for
`bnts,btfn->bfts` and multiply+reduce for grouped Read-M.

Eight-layer v6e implementation screening:

| Shape | Rank 1 | D/D | D/M | M/D | M/M | Winning wall cost |
|---|---:|---:|---:|---:|---:|---:|
| Medium 16x64 | 507.56 ms | 651.13 | 595.12 | 709.66 | 654.44 | +17.25% |
| XL 16x128 | 568.55 | 657.31 | 632.47 | 699.83 | 673.91 | +11.24% |
| XL 32x64 | 769.53 | 951.01 | 886.05 | 1,009.81 | 954.80 | +15.14% |

Dot mix plus multiply+reduce Read-M wins all three shapes. Multiply+reduce is unsuitable for the
large alpha-mix broadcast; dot is unsuitable for the small grouped Read-M contraction.

Full-24 target profiles, paired in UC1a:

| Shape / TPU | Rank 1 | Rank 2 D/M | Wall cost | Throughput retained |
|---|---:|---:|---:|---:|
| Medium 16x64 / v5p-16 | 1,478.47 ms | 1,718.47 ms | +16.23% | 86.03% |
| XL 16x128 / v5p-32 | 1,758.38 | 1,849.96 | +5.21% | 95.05% |
| XL 32x64 / v5p-32 | 2,071.92 | 2,316.08 | +11.78% | 89.46% |

Incremental per-layer theory at C256/T2048 (average causal source length 1,152), in units of one
`W_Q = D*D` projection:

| Shape | Extra parameters | Extra FLOPs | Read-M FLOPs |
|---|---:|---:|---:|
| Medium 16x64 | 0.0156 W_Q | 0.3145 W_Q | unchanged |
| XL 16x128 | 0.0078 | 0.1528 | unchanged |
| XL 32x64 | 0.0156 | 0.1650 | unchanged |

The rank routes partition the existing heads, so Read-M arithmetic and M-cache do not grow; the
extra work is the second alpha mixture/fetch and a doubled temporary fetched-M tensor. Fine-grained
target traces locate the measured wall regression as follows (scope deltas are diagnostic and not
additive because fused scopes overlap):

| Shape | mix-alpha | fetch-M | fetched Read-M | QK logits | softmax | MHA AV |
|---|---:|---:|---:|---:|---:|---:|
| Medium | +49.75 ms | +10.03 | +7.55 | +4.43 | +91.76 | +63.51 |
| XL 16x128 | +22.69 | +5.73 | +4.14 | +0.07 | +37.56 | +0.77 |
| XL 32x64 | +42.58 | +5.95 | +1.72 | +1.37 | +73.50 | +33.53 |

QK-logits FLOPs remain essentially unchanged, whereas the softmax scope is about 2x in every
Rank-2 train trace. Thus the unexpected indirect cost is in softmax or its reverse pass, not QK;
a forward-only trace or HLO inspection is needed to distinguish forward recomputation from a
duplicated softmax VJP. Rank 2 now needs a substantial loss gain to justify its 5–16% target cost.

## Canonical v6e-1 eight-layer matrix

Commit `91cb24a`; every arm has eight layers and explicitly selects optimized, non-streaming C256.
This matrix replaces the invalid older comparison that mixed G/6 optimized with LGLL/8 legacy.

| Schedule | Model | Arm | Configuration class | XPlane | step/s | S/U time cost |
|---|---|---:|---|---:|---:|---:|
| G | BAM | U/U | `BamV2GQChunk256OptimizedEightLayerProfile` | 638.73 ms | 1.566 | — |
| G | BAM | S/U | `BamV2GScanLayerOptimizedEightLayerProfile` | 661.01 ms | 1.513 | +3.49% |
| G | BAM-MHA | U/U | `BamMHAGQChunk256EightLayerProfile` | 467.49 ms | 2.139 | — |
| G | BAM-MHA | S/U | `BamMHAGScanLayerEightLayerProfile` | 482.79 ms | 2.071 | +3.27% |
| LGLL | BAM | U/U | `BamV2LGLLQChunk256OptimizedEightLayerProfile` | 483.28 ms | 2.069 | — |
| LGLL | BAM | S/U | `BamV2LGLLScanLayerOptimizedEightLayerProfile` | 693.20 ms | 1.443 | +43.44% |
| LGLL | BAM-MHA | U/U | `BamMHALGLLQChunk256OptimizedEightLayerProfile` | 357.35 ms | 2.798 | — |
| LGLL | BAM-MHA | S/U | `BamMHALGLLScanLayerOptimizedEightLayerProfile` | 500.49 ms | 1.998 | +40.06% |

| Schedule | U/U retention | S/U retention | S/U − U/U |
|---|---:|---:|---:|
| G | 73.19% | 73.04% | -0.15 pp |
| LGLL | 73.94% | 72.20% | -1.74 pp |

The v5p and v6e results agree: G/LGLL changes retention by roughly one percentage point, while
dynamic LGLL scan penalizes both BAM and MHA. The slight v6e U/U increase (`73.19→73.94%`) means
there is no universal rule that LGLL lowers retention; SWA reduces both standard attention work and
BAM mix/fetch, while fixed BAM projections/read/write remain.

## Current performance picture

### Fused BAM core: per-token MXU (v8t, 2026-10-12)

XLProp full-M generalized read (`...FullMReadGeneralReadTruePile`), v5p-8 paired on one VM, BAM health OFF:
pure JAX 0.453 → v7u fused (VPU dynamic reads) 0.532 → v7 tail 0.547 → **v8t 0.593 step/s = 96.6% of MHA
0.614**; device 1655 vs 1618 ms, of which the parameter-refund MLP widening alone is +114 ms. Kernels 255 → 102 ms.
Branch `claude/bam-pallas-fullm` (`cd06c769`), worktree `.claude/worktrees/bam-pallas-directc10`,
report `experiments/bam_llama2_medium/fullm_general_read_profile.md` on that branch.

- Every token's M_t is a stationary MXU operand: per-token `[R,V]x[V,K]` costs ~13 cycles/token on v5p vs
  ~160 VALU cycles for the VPU broadcast form. All reads of one layer are linear in M_t, so static keys and
  gates fold into one stacked key `X_t[4N,V]`; the write is `M_t += A_t^T C_t`; reverses are matmuls too.
- Keep kernel I/O T-minor ([..,T] slabs; Splash-native [B,N,T,K] outputs); only M is token-major `[B,T,V,K]`.
  Moving key building / output splitting to XLA (v8) made `[B,T,N,40/96]` small-minor tensors cost +330 ms of
  pads/copies/reduces (0.445 step/s). Norms belong in T-minor too (sublane sums are VPU adds; lane sums need XLU).
- Mosaic: stride-T sublane loads/stores cost ~1 op per row; aligned [8,L] blocks + an in-register
  major<->sublane swap are cheap but VALU-heavy, so use them only to replace such gathers. `scr[:, t, :]` on a
  3D ref is not a native strided load.
- Saving kernel outputs under remat costs scan stacking (~43 ms here) comparable to recomputing the read.

### Pure-JAX mix-alpha outcome

Forty JAX-only rewrites of the C256 layer-scan `bncs,bcn->bcs` contraction were screened on
eight-layer v6e, including independent `alpha: BNCS/BCSN` and BAM-owned
`mix_weights: BTN/BNT` layout axes. Multiply+reduce appeared 2.9% faster there and repeated at
full-24 v6e, but reversed on the target v5p-16:

| Full-24 v5p path | XPlane | mix scope | Result |
|---|---:|---:|---|
| BNCS + BTN + einsum | **1483.50 ms** | 84.35 ms | keep |
| BNCS + BNT + einsum | 1483.36 | 84.21 | neutral; XLA folds the logical transpose |
| best BCSN einsum arm | 1488.04 | 83.92 | no gain |
| best multiply+reduce arm | 1505.84 | 108.88 | 1.51% slower |

No pure-JAX rewrite improves the target graph; retain BNCS/BTN/einsum. The exhaustive matrix and
runnable commits are in `bam_profile_history.md` under **Pure-JAX mix-alpha search**. This result
also establishes that same-shape lowering gains from v6e must be confirmed on full-layer v5p.

### Current V2 C256 scan/non-scan paired main profile

Eight-layer UC1a `v6e-1`, `B=32,T=2048`, step 10–14. U/U is
`BamV2GQChunk256OptimizedEightLayerProfile` at `f29e89f`; S/U is
`BamV2GScanLayerOptimizedEightLayerProfile` at `66c8173`. The runtime graph is identical apart
from `scan_layers`: changes after `66c8173` were comments/documentation only. C256 computes 56.25%
of dense causal attention pairs, so MHA QK calibrates one training-state `(W_Q)` as
`5.02476 / 1.125 = 4.46645 TF`. The S/U `while.*` parent contains its body kernels and is excluded
from additive TF/byte totals to avoid double counting.
Named scopes may overlap through fusion; use complete-step time for throughput and scope deltas for
local attribution. XPlane TF/bytes describe the compiled lowering, so their small U/U–S/U changes
do not imply a change in model semantics or theoretical arithmetic.

| Part | Theory `(W_Q)` | U/U ms | S/U ms | S/U Δ | U/U TF / `(W_Q)` | S/U TF / `(W_Q)` | U/U / S/U GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer / optimizer / unscoped | 14.500 | 487.35 | 510.41 | +4.73% | 81.052 / 18.147 | 81.659 / 18.283 | 597.58 / 613.88 |
| └ shared C256 QChunk (overlapping scope) | — | 399.57 | 392.82 | -1.69% | 14.678 / 3.286 | 14.674 / 3.285 | 457.42 / 455.28 |
|   ├ MHA QK logits | 1.125 | 104.92 | 105.52 | +0.57% | 5.025 / 1.125 | 5.025 / 1.125 | 117.36 / 117.36 |
|   ├ MHA softmax | ≈0 | 36.25 | 33.38 | -7.93% | 0.055 / 0.012 | 0.055 / 0.012 | 39.83 / 39.83 |
|   └ MHA AV | 1.125 | 110.20 | 108.61 | -1.45% | 5.101 / 1.142 | 5.101 / 1.142 | 127.47 / 127.47 |
| **write M** | **0.406** | **27.86** | **33.90** | **+21.68%** | **1.577 / 0.353** | **1.804 / 0.404** | **27.11 / 34.19** |
|   ├ `P_loc_down` | 0.250 | 2.40 | 2.61 | +8.76% | 0.725 / 0.162 | 0.829 / 0.186 | 3.54 / 4.05 |
|   ├ `P_loc_up` | 0.125 | 3.20 | 3.68 | +14.76% | 0.732 / 0.164 | 0.837 / 0.187 | 5.31 / 6.06 |
|   ├ write-gate projection | 0.016 | 2.58 | 2.99 | +15.83% | 0.066 / 0.015 | 0.075 / 0.017 | 3.85 / 4.40 |
|   ├ write outer product | 0.016 | 14.95 | 17.09 | +14.28% | 0.044 / 0.010 | 0.050 / 0.011 | 5.64 / 6.44 |
|   └ GELU/RMS/bias/other | ≈0 | 4.72 | 7.53 | +59.67% | 0.010 / 0.002 | 0.012 / 0.003 | 8.76 / 13.24 |
| **mix alpha** | **0.033** | **60.67** | **58.88** | **-2.94%** | **0.215 / 0.048** | **0.209 / 0.047** | **68.07 / 65.35** |
|   ├ head-weight projection | 0.016 | 3.34 | 3.56 | +6.37% | 0.075 / 0.017 | 0.075 / 0.017 | 4.43 / 4.43 |
|   ├ chunked `bnts,btn→bts` | 0.018 | 57.26 | 55.27 | -3.48% | 0.140 / 0.031 | 0.134 / 0.030 | 63.45 / 60.74 |
|   └ diagonal/transform/other | ≈0 | 0.07 | 0.06 | -4.55% | 0.000 / 0.000 | 0.000 / 0.000 | 0.19 / 0.19 |
| AbsV source compression | 0.008 | 4.46 | 4.74 | +6.37% | 0.046 / 0.010 | 0.040 / 0.009 | 6.23 / 6.71 |
| fetch M temporal contraction | 0.281 | 8.00 | 8.20 | +2.55% | 1.232 / 0.276 | 1.240 / 0.278 | 11.56 / 11.75 |
| **read local M for QK** | **0.197** | **22.53** | **22.42** | **-0.49%** | **0.874 / 0.196** | **0.874 / 0.196** | **33.33 / 31.75** |
|   ├ packed key/gate/head-mix projection | 0.191 | 3.39 | 3.63 | +6.95% | 0.848 / 0.190 | 0.848 / 0.190 | 5.15 / 5.14 |
|   ├ key RMS/gate transform | ≈0 | 0.62 | 0.63 | +1.77% | 0.002 / 0.000 | 0.002 / 0.000 | 1.90 / 1.93 |
|   ├ **read-M contraction** | **0.004** | **7.94** | **7.06** | **-11.03%** | **0.013 / 0.003** | **0.014 / 0.003** | **8.62 / 7.25** |
|   ├ head-mix transform/expand | 0.002 | 5.62 | 5.98 | +6.43% | 0.007 / 0.002 | 0.007 / 0.002 | 9.98 / 9.75 |
|   └ other | ≈0 | 4.96 | 5.11 | +3.15% | 0.004 / 0.001 | 0.004 / 0.001 | 7.68 / 7.68 |
| **read fetched M** | **0.664** | **33.09** | **33.63** | **+1.63%** | **2.944 / 0.659** | **2.944 / 0.659** | **31.33 / 31.31** |
|   ├ read-key projection | 0.625 | 4.88 | 5.44 | +11.56% | 2.756 / 0.617 | 2.756 / 0.617 | 7.06 / 7.04 |
|   ├ read-gate projection | 0.031 | 3.10 | 3.33 | +7.42% | 0.144 / 0.032 | 0.144 / 0.032 | 4.47 / 4.47 |
|   ├ key RMS/gate/layout | ≈0 | 4.79 | 4.89 | +2.21% | 0.008 / 0.002 | 0.008 / 0.002 | 7.25 / 7.25 |
|   ├ read-M contraction | 0.008 | 17.63 | 17.28 | -1.99% | 0.033 / 0.007 | 0.033 / 0.007 | 7.52 / 7.52 |
|   └ other | ≈0 | 2.70 | 2.68 | -0.52% | 0.003 / 0.001 | 0.003 / 0.001 | 5.03 / 5.03 |
| **complete step** | **16.089** | **643.94** | **672.17** | **+4.38%** | **87.940 / 19.689** | **88.770 / 19.875** | **775.21 / 794.94** |

Stable log speed is `1.535` U/U versus `1.473` S/U (`-4.02%` throughput). S/U compiles in
`30.71 s` versus `121.44 s` for U/U (`3.95×` faster compile). Its complete-step penalty is
`28.23 ms`: the five BAM scopes plus AbsV compression account for only `5.17 ms`; the residual
accounts for `23.06 ms` (81.7%). Within BAM, write M adds `6.04 ms`, partly offset by mix alpha
(`-1.78 ms`) and LocalQK (`-0.11 ms`). The shared C256 QChunk is itself `6.75 ms` faster under
scan, so query chunking is not the source of the slowdown.

For LocalQK read-M, U/U exposes 20 forward and 32 backward compute events per step; S/U exposes
16 forward and the same 32 backward events. The reduced forward fragmentation saves `0.88 ms`,
but does not alter the backward. Thus scan's Q/K forward coalescing is real and local, while its
whole-step loop/layout cost is larger. Raw artifacts:
`/data0/xd/bam_diagnostics/scan_pair/f29e89f/non_scan/` and
`/data0/xd/bam_diagnostics/qk_batch_read/66c8173/baseline/`.

Manual Q/K batching does not improve this contraction. On the same non-scan v6e-1, commit
`2646f97`, it changes complete step `644.20→650.57 ms` (+0.99%) and LocalQK
`22.61→25.35 ms`. Its intended forward coalescing works (`314→216` events,
`8.03→5.78 ms`), but backward rises `14.58→19.56 ms`; backward read-M traffic rises
`4.97→7.58 GB`. Training is backward-dominated, so keep the separate Q/K implementation.

### XL Rank2 all-F versus shared LLLF paired main profile

Shared LLF [tuple-return refactor timing](xl_shared_tuple_timing.md): same UE5a
v5p-32 full-24 block-scan+AOT, old `1b39c64` .5550 → tuple `9c44858` .5584
steps/s (**+.61%**, steps 10–14). Small measured gain from avoiding intermediate
readout concat/split; not evidence that the larger shared-LocalV overhead is resolved.
The full-fetch XL Rank2 tuple control reaches .5540 vs its reused same-zone/window
old result .5524 (+.29%); LLF's ratio-of-speedups is therefore only about +.32%.

Follow-up [LocalQK/read simplification matrix](xl_read_simplification_timing.md):
UE5a v5p-32 full-24 XL Rank2 scan+AOT, High1/High2/High3/combined each .5540;
shared contraction + unified rank1/rankR (Mid1) .5538 (−.036%). No resolvable
speed change versus .5540. Combined `f549622` matches the recent `1b39c64`
reproduction at every printed step0–100; historical ten-step points differ by
at most 5e-7. Keep the simpler code for maintainability, without claiming speed gain.

Related [LLF JIT/AOT timing check](xl_llf_jit_aot_timing.md): UE5a full-24 scan,
independent LocalV JIT/AOT .5534/.5564 steps/s (−.54%); shared LocalV
.5512/.5532 (−.36%). Switching compilation method does not recover a material
throughput gain; full class names, exact commits and raw logs are in that report.

Full-24, T2048, v5p-32 **EW4b**, runtime `ca17342`, historical all-decay,
health metrics disabled. Implementation and fine-grained report:
[XL fine-grained paired report](xl_lllf_profile.md),
branch `codex/xl-lllf-profile`. Five unique configurations answer the three requested
pairs. Device timings use the complete first step on all eight recorded primary-worker
devices; partial second-step kernel records are excluded.

| Configuration | Scanner | Device step ms |
|---|---|---:|
| BamXLRank2LayerScanProfile | layer scan | 1796.94 |
| BamXLRank2Block4ScanProfile | block4 scan | 1786.64 |
| BamXLLocalSharedLLLFBlock4ScanProfile | block4 scan | 1785.67 |
| BamXLRank2NoScanProfile | none | 1825.86 |
| BamXLLocalSharedLLLFNoScanProfile | none | 1818.73 |

LLLF throughput gains: **+.63%** versus actual layer-scan Rank2, **+.05%** with
matched block4, **+.39%** without scan. Thus scanner grouping is not the principal
explanation for the unexpectedly small architectural gain.

Matched block4 LLLF saves **50.06 ms** of mix+fetch, but adds **38.60 ms** of
LocalV gate projection and shared LocalO/LocalV output gating, **9.78 ms** in write
M and **4.16 ms** in O-read scopes. Other small deltas leave only ~.97 ms net saving.
Gate application is backward/recompute dominated (~27.4 of29.2 ms), especially
coordinate reduction and accumulation from the two consumers. Shared contraction
does not make those two gated consumers free. Non-scan reproduces this cancellation.
Gating already multiplies only compact row/column slices, not the padded tail.
Investigate joint gradient/layout handling for a future paired
optimization; the present measurements do not yet establish a faster implementation.

Historical Medium S/U mix+fetch is **9.98%** of its step; XL is **3.67%**. The
former is 8-layer v6e-1 and the latter 24-layer v5p-32: this is descriptive, not a
hardware-controlled scaling claim. Both temporal fetch/W_Q and MHA AV/W_Q halve
from Medium to XL; fetch/AV remains25%, so reduced fetch leverage cannot be explained
by comparison to AV alone. Detailed scope/theory definitions and raw per-device
JSON are in the linked report. Existing Medium main profiles above/below are retained.

### V2 fine-grained main profile

The most detailed operator breakdown is the six-layer graph of
`BamLlama2MediumDirectPLocR256GeluBf16PackedLocalQK`, commit `9fb6720`, standalone `v6e-1`,
step 10–14. It is the clean precursor from which V2 evolved and remains the primary table for
choosing optimization targets. One training-state `(W_Q)` is calibrated by MHA QK logits as
**3.329 TF/step**.

| Part | Theory `(W_Q)` | XPlane TF / `(W_Q)` | Bytes | Scope ms (6L / layer / 24L linear) |
|---|---:|---:|---:|---:|
| Transformer / optimizer / unscoped | 16.250³ | 72.308 / 21.722 | 564.44 GB | 498.21¹ / — / — |
| └ MHA QK logits² | 2.000 | 6.657 / 2.000 | 115.99 GB | 101.07 / 16.84 / 404.28 |
| **write M** | **0.406** | **1.125 / 0.338** | **23.41 GB** | **35.35 / 5.89 / 141.39** |
| ├ `P_loc_down` | 0.250 | 0.518 / 0.156 | 2.53 GB | 2.33 / 0.39 / 9.32 |
| ├ `P_loc_up` | 0.125 | 0.523 / 0.157 | 3.77 GB | 2.74 / 0.46 / 10.94 |
| ├ write-gate projection | 0.016 | 0.047 / 0.014 | 2.75 GB | 2.63 / 0.44 / 10.52 |
| ├ **write outer product** | 0.016 | 0.032 / 0.010 | 5.91 GB | **18.84 / 3.14 / 75.36** |
| └ GELU/RMS/bias/other | ≈0 | 0.005 / 0.001 | 8.45 GB | 8.81 / 1.47 / 35.25 |
| **mix alpha** | **0.047** | **0.248 / 0.075** | **97.76 GB** | **91.38 / 15.23 / 365.52** |
| ├ head-weight projection | 0.016 | 0.056 / 0.017 | 3.32 GB | 3.60 / 0.60 / 14.39 |
| ├ **`bnts,btn→bts`** | **0.031** | **0.192 / 0.058** | **91.11 GB** | **84.44 / 14.07 / 337.75** |
| └ transform/other | ≈0 | 0.000 / 0.000 | 3.32 GB | 3.35 / 0.56 / 13.38 |
| **fetch M** | **0.508** | **1.678 / 0.504** | **14.19 GB** | **17.16 / 2.86 / 68.64** |
| ├ AbsV source compression | 0.008 | 0.025 / 0.007 | 5.13 GB | **9.99 / 1.67 / 39.96** |
| └ temporal fetch contraction | 0.500 | 1.653 / 0.497 | 9.06 GB | 7.17 / 1.19 / 28.68 |
| **read local M for QK** | **0.197** | **0.655 / 0.197** | **25.54 GB** | **44.01 / 7.33 / 176.03** |
| ├ packed key/gate/head-mix projection | 0.191 | 0.636 / 0.191 | 3.86 GB | 2.93 / 0.49 / 11.70 |
| ├ key RMS/gate transform | ≈0 | 0.001 / 0.000 | 1.52 GB | 0.79 / 0.13 / 3.17 |
| ├ **read-M contraction** | **0.004** | **0.012 / 0.004** | **11.78 GB** | **33.74 / 5.62 / 134.95** |
| ├ head-mix transform/expand | ≈0.002 | 0.005 / 0.002 | 7.45 GB | 5.92 / 0.99 / 23.68 |
| └ other | ≈0 | 0.000 / 0.000 | 0.93 GB | 0.63 / 0.11 / 2.53 |
| **read fetched M** | **0.664** | **2.205 / 0.663** | **21.63 GB** | **22.85 / 3.81 / 91.38** |
| ├ read-key projection | 0.625 | 2.067 / 0.621 | 5.29 GB | 3.33 / 0.55 / 13.30 |
| ├ read-gate projection | 0.031 | 0.108 / 0.032 | 3.35 GB | 2.60 / 0.43 / 10.42 |
| ├ key RMS/gate/layout | ≈0 | 0.004 / 0.001 | 4.23 GB | 2.84 / 0.47 / 11.35 |
| └ **read-M contraction** | **0.008** | **0.026 / 0.008** | **8.76 GB** | **14.08 / 2.35 / 56.31** |
| **complete step** | **18.072** | **78.218 / 23.498** | **746.98 GB** | **708.95 ms** |

1. The 498.21-ms residual is the complete step minus five top-level BAM scopes; it includes
   Transformer, optimizer, communication, unscoped work and idle, not pure Transformer time.
2. MHA QK is already included in the first row and appears only as a common calibration reference.
3. The 24-layer values are linear scope extrapolations, not a replacement for full-24 profiling.

This table exposed the optimization order. Its current status is:

1. `mix alpha`: only `0.031 W_Q`, yet its contraction moved 91.11 GB and occupied 84.44 ms. C256
   template/concat/deferred-read work reduced the mix scope `89.25→44.27 ms`, but it remains the
   largest BAM-specific target.
2. LocalQK read-M: only `0.004 W_Q`, yet the contraction took 33.74 ms. Packed projections did not
   fix this low-utilization reduction; it remains unresolved.
3. AbsV source compression: multiply+reduce was tested and slowed the step 708.95→716.26 ms; keep
   dot. This branch is closed unless a custom kernel/layout changes the lowering.
4. Write outer product: addressed by multiply+reduce, improving the full-24 dynamic-V graph 7.27%.
5. CombinedRead: addressed by diagonal-one, improving the equivalent full-24 graph 3.96%.

Raw artifacts: `/data0/xd/bam_diagnostics/clean_profile_9fb6720_v6e/`.

### C256 improvement on top of the main profile

The optimized six-layer C256 ladder reduced BAM scope from 170.13 to 112.47 ms:

| Component | legacy | optimized |
|---|---:|---:|
| mix alpha | 89.25 ms | **44.27 ms** |
| diagonal update | 33.41 ms | **0.03 ms** |
| fetched read | 37.26 ms | **25.23 ms** |
| copy | 30.93 ms | **4.29 ms** |
| compiler temporary estimate | 14.08 GB | **8.68 GB** |

Remaining structural concerns are:

1. Mix-alpha and small M contractions still have poor realized throughput relative to theoretical
   FLOPs; C256 fixes much of the materialization cost but not small-reduction utilization.
2. Full BAM retains about 72–75% of matched BAM-MHA throughput on the measured C256 schedules.
3. Historical `M_s` remains the principal inference-cache obstacle. AbsV8 shrinks each matrix but
   fixed windows and time-block compression have not provided healthy temporal reduction.
4. Query scan lowers compile memory/latency but is 2.5–3.4× slower and is not a training path.

## Decision ledger

| Change | Capability result | Efficiency result | Decision |
|---|---|---|---|
| RMS-gated read keys | `-0.0678` vs MHA at 13,400 | small cost | keep; only clearly large capability gain |
| soft RMS cap | `+0.0435` vs original BAM at 2,550 | no useful gain | reject |
| three fixed full fetches | `+0.0015` vs RmsGateOnly @2,800 | more compute/cache | reject |
| dynamic one-fetch mix | essentially same loss as multi-fetch | less fetch state/work | keep |
| signed RMS mix vs SoftmaxMix | `-0.0006` @6,200 | similar | no demonstrated capability gain |
| CombinedRead / diagonal-one | same capability within trajectory noise | diagonal-one 3.96% faster full-24 | keep diagonal-one |
| FactorizedLocalQK | `-0.0035` vs Combined; `+0.0057` vs PerHead | far fewer parameters/work | keep |
| remove whole-M normalization | `-0.0078` vs Factorized | slightly faster | keep |
| AbsV8 Direct | `+0.0096` vs V1 final | 11.7% faster; V-axis cache 4× smaller | keep as tradeoff |
| R256-GELU `P_loc` | `-0.0044` vs Direct final | speed neutral | keep |
| packed factorized LocalQK projection | mapped/native-init controls converge to ~loss-neutral vs Direct | isolated fp32 mapped pair: 0.523 vs 0.509 step/s (`+2.8%`); the cross-commit clean endpoint, 0.521 vs historical Direct 0.512, is not a packing-only estimate | keep with sharded `P_loc_up`; inherited by V2 |
| replicate `P_loc_up` | persistent ~`+0.0034` trajectory gap | ~1.5% faster | reject for capability runs |
| CodebookC4 | `+0.0411` vs Factorized @2,800 | faster | reject |
| Window256 | same-batch `+0.1193` | temporal cache 8× smaller | reject |
| best Window256+OldBlock16 | same-batch `+0.0626` | temporal cache 4.27× smaller | reject |
| JAX bilateral block read | full-24 1.04% slower | more copies/traffic | reject |
| three-input mix+fetch einsum | — | 20–29% slower | reject |
| pure-JAX mix-alpha rewrites | semantics healthy | best v6 gain reverses on v5p; no target gain | keep BNCS/BTN/einsum |
| C256 optimized | short-step semantics healthy | 22.37% faster than dense BAM full-24 | keep as speed candidate |
| layer scan, G | semantics healthy | ≤1.72% full-24 cost | optional compile tradeoff |
| layer scan, LGLL | semantics healthy | ~23% full-24 cost | reject for long training |
| streaming query scan | semantics healthy | 2.5–3.4× slower | reject |

## Checkpoint conclusions

Detailed tables and ablations are in
[`bam_checkpoint_diagnostics.md`](bam_checkpoint_diagnostics.md). The durable conclusions are:

- Signed alpha is functionally used: negative mass is 43.2%, and deleting minority-sign content
  raises same-batch loss by 0.0293. This proves checkpoint dependence, not superior training
  parameterization.
- Token 0 has a local sink but explains only 0.33% of Window256's loss damage when the first four
  tokens are restored.
- Adjacent `M_s` matrices are weakly correlated; mean/linear time blocks are not viable read-only
  approximations.
- The old packed LocalQK gap came from replicated `P_loc_up` reduction order, not packing or `btn`
  layout.
- V2's early fast-path divergence decays to `+0.000138` versus Direct at step 13,400.

## Reproduction index

| Topic | Commit/configuration | Artifact or source |
|---|---|---|
| configuration/run conclusions | per-class `code_commit` | `MaxText/exp.py` |
| early and clean fine-grained profiles | `e05d099`, `9fb6720` | `/data0/xd/bam_diagnostics/clean_profile_9fb6720_v6e/` |
| write/read implementation pairs | `fef8e3a`, `2b8e63a` | `/data0/xd/bam_diagnostics/write_read_pair_fef8/` |
| C256 optimization ladder | `821dc8d`, `165b55b` | `/data0/xd/bam_diagnostics/bam_c256_opt/v6e/` |
| fixed BAM-MHA C256 controls | `a1ad13f` | `/data0/xd/bam_diagnostics/c256_control_fix/` |
| v5p full-24 C256/scan | classes in canonical table | `/data0/xd/bam_diagnostics/qchunk_full_v5/` |
| fair v6e eight-layer scan matrix | `91cb24a` | `/data0/xd/bam_diagnostics/bam_scan/v6e8_fair/` |
| pure-JAX mix-alpha matrix | `2cad4cb`…`eed9791` | `/data0/xd/bam_diagnostics/mix_alpha/` |
| M-cache compression | V1 step 13,250 | `/data0/xd/bam_diagnostics/bam_cache_diagnostics_49be222_mb16_final.json` |
| signed alpha | V1 step 13,250 | `/data0/xd/bam_diagnostics/bam_alpha_*_final.json` |
| attention sink / prefix-4 | `a02fc72`, `d3c17a6` | `/data0/xd/bam_diagnostics/bam_*sink*`, `bam_window_prefix4_*` |

Design documents:

- [`bam_scan_design.md`](bam_scan_design.md)
- [`shared_qchunk_swa_design.md`](shared_qchunk_swa_design.md)


### XL all-column K64/K128 main profile

**XL M-size experiment preflight:** inspect the resolved write/read contraction switches
and the historical same-shape operator matrix before launch or speed attribution.
K64 favors mul_reduce while K128 favors dot; neither transfers automatically to a
new K or read architecture. Pair write × read implementations on the full target
shape with matched health settings; include rank-to-head expansion only when present.
Do not compare an unoptimized new size with optimized K128 and call the difference
an architectural cost. K96 concat follow-up: `xl_k96_concat_operator_profile.md`.

`BamXLSharedBasisQKDirectC8MLPPerLayerColOnly` and
`BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE`:
training runtime `4fb2021`; profile classes `BamXLK64OperatorWMRMSMFull` and
`BamXLK128OperatorWMRMSMFull` at `3d59d64`. Same UC1a v5p-32, 24 layers,
B16/device,T2048,C256; generic health ON/BAM sow OFF. Trace-free steps20–24
.6030/.5070 steps/s: K128 -15.92% throughput, consistent with formal UE5a
-15.75%. XPlane device time +19.11%, compiled FLOPs only +.498%.

Forward theory uses W_Q=2D² FLOPs per token averaged over layers; omits
LM-head/optimizer and elementwise arithmetic. XPlane includes backward/remat.
Device event buffers retain one complete step on the primary profiler host's eight TPU cores (JAX process0, gcloud worker1);
partial next-step markers and nested `while` wrappers are excluded. Full step
speed is independently checked using the five trace-free steps above. Fused
source scopes describe compiler attribution, not separable causal components.

| Part | Forward theory W_Q (K64 / K128) | K64 ms | K128 ms | Delta ms | K64 / K128 TF | K64 / K128 GB |
|---|---:|---:|---:|---:|---:|---:|
| Transformer / optimizer / unscoped | 12.7417 / 12.7417 | 1387.70 | 1405.68 | +17.98 | 170.3582 / 170.3579 | 1211.40 / 1214.75 |
| local QKV packed projection | .208333 / .208333 | 21.19 | 21.18 | -0.01 | 2.7533 / 2.7533 | 38.58 / 38.58 |
| write M | .171875 / .179688 | 78.53 | 209.83 | +131.30 | 2.2695 / 2.3527 | 108.70 / 143.27 |
| ↳ outer (subset) | .007813 / .015625 | 24.99 | 133.48 | +108.49 | 0.0759 / 0.1520 | 8.46 / 19.73 |
| C8 compression | .003906 / .007813 | 14.07 | 27.98 | +13.91 | 0.0799 / 0.1597 | 16.78 / 33.55 |
| LocalQK read | .003906 / .007813 | 52.82 | 119.17 | +66.34 | 0.0585 / 0.1104 | 28.64 / 39.51 |
| LocalV read + expansion | .001953 / .003906 | 19.59 | 60.45 | +40.86 | 0.0272 / 0.0517 | 16.95 / 25.55 |
| O key/gate + read | .072266 / .074219 | 42.77 | 82.20 | +39.43 | 0.9580 / 0.9828 | 67.06 / 77.20 |
| route key + head mixing | .004069 / .004069 | 17.35 | 17.44 | +0.09 | 0.0637 / 0.0637 | 25.30 / 25.30 |
| temporal fetch | .046875 / .093750 | 5.06 | 8.43 | +3.36 | 0.6195 / 1.2385 | 4.67 / 8.09 |
| complete step | ≈13.2549 / ≈13.3213 | 1639.64 | 1952.92 | +313.28 | 177.1879 / 178.0707 | 1518.08 / 1605.81 |

Raw data: `/data0/xd/bam_diagnostics/xl-colonly-k-operators/`.
[Operator matrix, provenance and full profile details](xl_colonly_k64_k128_operator_profile.md).
All 16 full-layer combinations were tested on the same VM. K64 retains all
mul_reduce (.6030 steps/s); K128 all-dot reaches .5770 (+13.81% vs its .5070
control). Each model using its best setting leaves K128 4.31% slower than K64,
versus the original 15.92% gap. K128 write/read dot with mul-reduce expansion
already achieves +13.10%; the expansion switch adds .63%.

K128 write-only dot reduces its outer-product kernel 133.48→25.29 ms and raises
throughput 6.31%. K64 saves only 2.36 ms there but adds 34.58 ms of copy kernels;
its throughput falls 1.63%. K64 write/read dot adds 81.38 ms of copies and
83.26 ms whole-step time. Thus the marginal v6e K64 ranking did not transfer
to the full v5p shape. Compiled K128 FLOPs/bytes remain essentially unchanged
under all-dot (178.084 TF / 1609.49 GB); better lowering drives the speedup.
K128 formal training adopted all-dot runtime `9a58407` after checkpoint9024;
UE5a resumed steps9035–39 .5690 steps/s, +13.89% vs pre-switch .4996 and
-4.08% vs K64 .5932 at the same steps, with matched health. Full result matrix,
BF16 value checks, scope limits and retained raw paths are in the linked report.


### Shared-C8 K64: spend QK savings on width before depth (2026-09-20)

The independent-VO-gate K64 family (`8c188b0`, implementation worktree
`/data0/xd/llf-parameter-matched`) completed 13,500 steps. Final five common
windows (12,600–13,400): QK48 with MLP3050/3050/3045 beats QK32 with
MLP2879/2879/2874 by .002674; spending the QK savings on a 25th L instead
(original MLP2879/2879/2874, tail2970) beats QK32 by only .000784 and loses
to the wider 24-layer model by .001890. The 25-layer model is 2.18% slower
than 24-layer QK48, with 44 extra health scalars. Its early deficit shrank but
did not reverse: an added terminal L is not a better use of this budget than
widening existing MLPs. This conclusion is specific to this placement and family.

Untying LocalV/O gates also reversed the early QK48-vs-QK32 ranking relative
to the old shared-gate series. Do not transfer early rankings between gate
schemes or attribute the whole difference to gate freedom: the current
independent implementation also moves gate multiplication from the read key
to the read output, mathematically equivalent but not bitwise equivalent in
BF16. A tied-gate control on that same post-contraction path remains unrun.
Raw cumulative comparisons: `/data0/xd/k64-independent-final-report.txt`;
formatted report: `/data0/xd/k64-independent-final-report.md`.


### XL K96 QK-concat full-shape operator result (2026-09-21)

Same EW4b v5p-32, genericON+968BAM metrics, steps20–24: write/read mul/mul .5150,
mul/dot .5424 (+5.32%), dot/mul .5148 (-.04%), dot/dot .5436 (+5.55%).
K96 needs read-dot; write-dot adds only .22% over read-dot, below a persuasive standalone effect.
This differs from old K64 all-mul and K128 all-dot: rerun the operator comparison for an expanded-M XL shape.
[Full report](xl_k96_concat_operator_profile.md); raw `/data0/xd/bam_diagnostics/xl-k96-concat-operators/`.


### AllLocal local-O writeback geometry (2026-09-22)

AllLocal without fetchedO needs no cross-token M-cache: it retains ordinary attention KV-cache. Expanding M changes forward activations and matrix read/write work; it does not add a persistent source-M cache. Historical fetchedO configurations do retain source matrices for cross-token fetch.

`BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal`,
checkpoint13500 / training runtime `ba29940`: 128 fixed Pile sequences, dual read-O/write
gates>=.1. L2-L22 actual O update / old-content norm median .179; 35.86% oppose
the old direction, only18/19,287,431 reverse its projection. Lifted read/write
address angle median88.63deg; address/content cosine signs disagree30.23%.
Gate association is dominated by component amplitudes, with head-specific opposite
angular/compositional trends; total amplitude alone loses substantial information.
These are observational same-head diagnostics, not full-M or causal ablations.
[Complete distributions, held-out checks and reproduction](alllocal_write_geometry.md).
Diagnostic implementation: `codex/alllocal-write-geometry`, `/data0/xd/alllocal-write-geometry`.
The user requested retaining `xd-v6e-alllocal-geom-3-0922` in EW4a after completion.

### XLProp C256 matched-basic-health main profile

Runtime `1517cce`, same UE5a v5p-32, full28xD1920/head96,T4096,batch8/device.
`BamMHAXLPropC256BasicHealthProfile` vs `BamXLPropK72SharedRank4BasicHealthProfile`:
trace-free20–24 .5610/.3890 steps/s (**BAM -30.66%**), device-step1771.36/2546.17ms.
Both use C256, basic training health ON, BAM sow OFF. The formal Splash MHA is
~.548–.550; formal BAM with concat health~.384. Extra health explains little.

Forward theory uses `1 W_Q = 2BTD²`, averaged over28 layers; F-only work is
weighted9/28. Both arms use the actual C256 pair count `T(T+256)/2`.
Theory excludes LM-head/optimizer and lower-order elementwise work; the first row's
theory therefore covers only Transformer blocks. LM head adds.93810 W_Q to each
per-layer average. Nominal theory counts28 writes; the unused final write can be
removed, subtracting.01147 W_Q from BAM.
XPlane columns include backward/remat, LM head and optimizer. They average the primary
profiler host's eight TPU-core traces over complete kernel-covered device steps;
partial next-step markers and nested while wrappers are excluded. Top-level rows
partition attributed kernel work; `↳` rows are subsets, not extra work. Fused-op source
attribution is not an isolated causal measurement. Complete-step ms includes the
small gaps between kernels, so it differs slightly from their duration sum.

| Part | Forward theory W_Q (MHA / BAM) | MHA ms | BAM ms | Δ ms | MHA / BAM TF | MHA / BAM GB |
|---|---:|---:|---:|---:|---:|---:|
| Transformer / LM head / optimizer / unscoped | 14.26667 / 13.64792 | 1771.12 | 2082.86 | +311.73 | 194.3186 / 184.9380 | 1728.21 / 1860.53 |
| LocalQK basis / gate / head-mix projection | 0.00000 / 0.18750 | 0.00 | 23.14 | +23.14 | 0.0000 / 2.5405 | 0.00 / 43.88 |
| Write M | 0.00000 / 0.32118 | 0.00 | 131.87 | +131.87 | 0.0000 / 4.1859 | 0.00 / 151.86 |
| ↳ P_loc down (subset) | 0.00000 / 0.20833 | 0.00 | 22.78 | +22.78 | 0.0000 / 2.7240 | 0.00 / 45.09 |
| ↳ P_loc up (subset) | 0.00000 / 0.08681 | 0.00 | 26.75 | +26.75 | 0.0000 / 1.1459 | 0.00 / 26.78 |
| ↳ write-gate projection (subset) | 0.00000 / 0.01042 | 0.00 | 12.80 | +12.80 | 0.0000 / 0.1377 | 0.00 / 31.05 |
| ↳ outer product (subset) | 0.00000 / 0.01562 | 0.00 | 35.11 | +35.11 | 0.0000 / 0.1559 | 0.00 / 20.14 |
| C10 compression | 0.00000 / 0.00781 | 0.00 | 18.21 | +18.21 | 0.0000 / 0.1070 | 0.00 / 17.55 |
| LocalQK dynamic + static read / expansion / Gram | 0.00000 / 0.03789 | 0.00 | 141.93 | +141.93 | 0.0000 / 0.5232 | 0.00 / 131.68 |
| ↳ rank-to-head expansion (subset) | 0.00000 / 0.00313 | 0.00 | 31.21 | +31.21 | 0.0000 / 0.0399 | 0.00 / 27.07 |
| Shared LocalVO / FetchedO key + O gate + read | 0.00000 / 0.11849 | 0.00 | 76.36 | +76.36 | 0.0000 / 1.6112 | 0.00 / 82.98 |
| ↳ read-M contraction (subset) | 0.00000 / 0.00391 | 0.00 | 33.91 | +33.91 | 0.0000 / 0.0537 | 0.00 / 14.34 |
| Independent LocalV gate projection | 0.00000 / 0.00707 | 0.00 | 8.28 | +8.28 | 0.0000 / 0.0975 | 0.00 / 21.80 |
| LocalVO output gating | ≈0 / ≈0 | 0.00 | 8.34 | +8.34 | 0.0000 / 0.0041 | 0.00 / 8.24 |
| F route projection + attention-head mixing | 0.00000 / 0.00714 | 0.00 | 38.79 | +38.79 | 0.0000 / 0.1224 | 0.00 / 50.35 |
| F temporal fetch | 0.00000 / 0.13661 | 0.00 | 15.62 | +15.62 | 0.0000 / 1.8511 | 0.00 / 12.39 |
| **Complete step** | **14.26667 / 14.47161** | **1771.36** | **2546.17** | **+774.81** | **194.3186 / 195.9808** | **1728.21 / 2381.27** |
| ↳ Q/K projections (overlapping subset) | 2.00000 / 0.50000 | 181.70 | 75.50 | -106.21 | 27.0911 / 6.7839 | 157.49 / 94.95 |
| ↳ V/O projections (overlapping subset) | 2.00000 / 2.00000 | 176.89 | 184.56 | +7.67 | 27.0753 / 27.0767 | 135.94 / 143.88 |
| ↳ MHA QK logits (overlapping subset) | 1.13333 / 1.13333 | 202.46 | 308.10 | +105.64 | 15.4484 / 15.4129 | 278.24 / 258.64 |
| ↳ MHA AV (overlapping subset) | 1.13333 / 1.13333 | 233.94 | 270.18 | +36.24 | 15.5459 / 15.5587 | 263.05 / 268.39 |
| ↳ MLP (overlapping subset) | 8.00000 / 8.88125 | 637.01 | 744.61 | +107.61 | 99.3523 / 110.2954 | 548.96 / 571.44 |
| ↳ copy kernels (cross-cutting subset) | — | 17.22 | 174.35 | +157.13 | 0.0000 / 0.0001 | 35.60 / 199.26 |

Matched-C256 forward theory14.26667/14.47161 W_Q per-layer average (+1.44%,
excluding LM head/elementwise); XPlane model FLOPs+0.855%, bytes+37.79%, time+43.74%.
Copy kernels17.2->174.4ms. LocalQK141.9ms/write131.9ms outweigh temporal fetch15.6ms,
although temporal fetch dominates the added theoretical arithmetic. Prioritize layout/copies
and small read/write contractions; do not infer the bottleneck from FLOPs alone.
Source scopes and copy classification overlap and must not be summed.
[Reproduction, precise configuration differences and artifacts](xl_prop_matched_health_profile.md).


## RMT static-write removal: arithmetic is not a speed ceiling

Deleting attention/MLP static writes halves their outer-write count (four to
two per layer). The saved forward arithmetic is only0.08 W_Q for the tested
D1200/K48/V75/H16 geometry. That fraction does not bound speed benefit:
full-M materialization/traffic, backward/remat, small contractions and fusion
can dominate. All-RMT-health-OFF measured contraction-category−41.35ms,
offset by formatting+17.14ms, loop fusion+13.56ms and elementwise+10.29ms.
With all41 health statistics retained, row moments, a transposed internal
scan carry and reused input-M RMSs recover pure speed from.401 to.407
step/s, versus original.405. XPlane speed gains+1.52% versus old pure,
+0.40% versus original.
An independent rematerialized full-M add (10.28ms) disappears after changing
carry layout; full-M copy count does not decrease. Compare global kernels
by shape/layout, since compiler source labels can move without new copies.
[Paired repair measurements and reproduction](rmt_dynamic_write_speed_repair.md).


### RMT MHABudget full-layer main profile (2026-09-26)

`RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget`, runtime`d1b1f89`,
same EW4b v5p-16, full18xD1200/head16x75,T4096,batch16/device, M48x75,
C8/tail32 dynamic reads, full48 writes, independent RoPE18 and matrix QK57,
MLP4118. Six LLL blocks, layer remat inside block scan. Static and dynamic
attention/MLP writes remain separate. `rmt_record_dynamic_health=True` (41);
`record_internal_nn_metrics=False`; base training metrics unchanged.

Paired stable20-99 speed: original.379875; carry-only.378050 (-.480%);
headwise W1/Wg/W2.379950 (+.020%, negligible); combined.378025 (-.487%).
Retain the original. Device steps2599.46/2611.94/2598.58/2611.17ms agree.
This is the MHA-budget model, not the older narrower-MLP VectorNorm.

Parameter units: `W_Q=D²=1,440,000`. Forward theory units: `2BTD²`,
per-layer average; C256 pair count is`T(T+256)/2`. Complete-step parameter
and theory columns describe Transformer blocks, excluding embeddings, final
seed/read/norm, LM head and optimizer. Norm/GELU/gating/softmax elementwise
work is omitted; QK Gram contractions are counted. Static/dynamic matrix
QK reads nominally produce75 coordinates before retaining57. LM head adds
2.33481 forward W_Q per-layer average; seed/final matrix contractions add.00444.
Full parameters432112752; Transformer block17,281,712/layer.

XPlane columns include backward/remat/LM head/optimizer and compiler-model
FLOPs/bytes. They average the primary host's eight cores over the first fully
kernel-covered step; subsequent incompletely covered step markers and nested
while containers are excluded. Top rows partition attributed kernels; `↳`
rows overlap and are not extra work. Fused source/scope attribution is not an
isolated causal timing.

| Part | Parameters W_Q/layer | Forward theory W_Q | ms | Step share | XPlane TF | GB |
|---|---:|---:|---:|---:|---:|---:|
| Scan / residual / LM head / optimizer / other | — | — | 415.93 | 16.00% | 12.0881 | 454.77 |
| SwiGLU MLP | 10.295000 | 10.295000 | 440.32 | 16.94% | 70.0393 | 279.91 |
| Independent QK18 projection + RoPE | 0.480000 | 0.480000 | 95.61 | 3.68% | 3.2734 | 74.26 |
| C256 QK logits | 0.000000 | 1.813333 | 206.07 | 7.93% | 12.4033 | 244.74 |
| C256 softmax | 0.000000 | ≈0 | 77.39 | 2.98% | 0.0822 | 82.29 |
| C256 AV | 0.000000 | 1.813333 | 260.82 | 10.03% | 12.5265 | 245.12 |
| Static QKV M read | 0.001600 | 0.120000 | 48.92 | 1.88% | 0.8302 | 59.45 |
| Static MLP M read | 0.000533 | 0.040000 | 40.36 | 1.55% | 0.2810 | 35.41 |
| Static attention + MLP M writes | 0.001067 | 0.080000 | 80.30 | 3.09% | 0.4983 | 85.05 |
| Dynamic QK | 0.240111 | 0.254133 | 262.99 | 10.12% | 1.7692 | 196.33 |
| ↳ QK basis / mix / gate projections (subset) | 0.240000 | 0.240000 | 119.39 | 4.59% | 1.6421 | 108.81 |
| ↳ QK basis M contraction (subset) | 0.000000 | 0.006667 | 28.00 | 1.08% | 0.0453 | 19.31 |
| ↳ QK rank-to-head expansion (subset) | 0.000000 | 0.006667 | 40.06 | 1.54% | 0.0455 | 26.36 |
| ↳ QK Gram / RMS / gating / other (subset) | 0.000111 | 0.000800 | 75.53 | 2.91% | 0.0364 | 41.85 |
| Dynamic VO C8 | 0.133533 | 0.153333 | 151.42 | 5.83% | 1.0694 | 115.59 |
| Dynamic MLP C8 read | 0.120189 | 0.140000 | 127.58 | 4.91% | 0.9642 | 100.44 |
| Dynamic attention + MLP M writes | 0.727489 | 0.806400 | 277.03 | 10.66% | 5.5018 | 242.75 |
| ↳ write address down (subset) | 0.426667 | 0.426667 | 25.52 | 0.98% | 2.9076 | 43.29 |
| ↳ write address up (subset) | 0.273067 | 0.273067 | 28.06 | 1.08% | 1.8699 | 35.24 |
| ↳ write gate projection (subset) | 0.026667 | 0.026667 | 12.72 | 0.49% | 0.1842 | 28.85 |
| ↳ dynamic outer write (subset) | 0.000000 | 0.080000 | 108.80 | 4.19% | 0.4867 | 74.53 |
| ↳ write transforms / other (subset) | 0.001089 | ≈0 | 101.92 | 3.92% | 0.0533 | 60.84 |
| Vector pre-RMSNorm | 0.001667 | ≈0 | 79.24 | 3.05% | 0.0446 | 96.80 |
| RMT health statistics | 0.000000 | ≈0 | 34.19 | 1.32% | 0.0707 | 41.34 |
| Complete step | 12.001189 | 15.995533 | 2599.46 | 100.00% | 121.4421 | 2354.26 |
| ↳ all copy kernels (subset) | — | — | 288.48 | 11.10% | 0.0000 | 225.69 |

Matrix flow (static/dynamic reads and writes, vector norms and RMT health)
occupies1102.03ms (42.39%), versus MLP440.32ms (16.94%). Dynamic QK262.99ms
and dynamic writes277.03ms are major targets. Copies288.48ms (11.10%) cut
across these scopes; do not add them again. Forward theory15.99553 W_Q is
only2.36% above matched MHA15.62667; small theoretical arithmetic does not
imply small layout or read/write cost.

Headwise MLP saves13.6ms formatting but adds20.2ms convolution-fusion work,
including13.2ms in W2 backward scopes; other savings leave no whole-step gain.
Transposed carry makes the scan-buffer update2.37->17.20ms, offsetting its
copy savings. Prior carry gains on the pure-dynamic model do not transfer
to this block-scanned MHA-budget model.
[Configurations, paired results and reproduction](rmt_headwise_carry_profile.md).
Raw artifacts: `/data0/xd/bam_diagnostics/rmt-headwise-carry/`.


### RMT attention-write / MLP-read chunk sweep (2026-09-27)

Full18 combined-boundary model (MLP4078), same EW4b v5p-16; generic health ON,
extra RMT health OFF. Attention C256 unchanged. Original runtime3617fbb and
static-unrolled follow-up4988442; each follow-up has a same-commit control.

| Write/read scheduling | Device step ms | Stable step/s | Throughput vs paired control |
|---|---:|---:|---:|
| Original | 2579.69 | .3848 | — |
| Merge48→16 static read + padded32→8 compression | 2584.47 | .3833 | −.18% |
| Best mapped chunk (merged1024) | 2898.98 | .3420 | −11.01% |
| Mapped64, separate reads | 3544.77 | .2800 | −27.23% |
| Static-unrolled512 | 2688.54 | .3690 | −4.04% |
| Static-unrolled1024 | 2941.76 | .3370 | −12.30% |

Retain original. Mapped1024 adds122 ms formatting,98 ms loop fusion,58 ms
broadcast,28 ms output updates. Unrolling removes most of those costs, but
convolution-fusion time grows78 ms at512 and348 ms at1024. Combining the two
linear reads saves14 ms GEMM time but loses it to layout/elementwise work.
Unlike attention QChunk, these chunks remove no nominal FLOPs; reuse must
actually stay on chip. Simple JAX chunk boundaries do not ensure that.

Unrolled1024 also stalls near initial loss through step30; one-layer CPU
value/gradient agreement does not establish whole-model TPU equivalence.
This candidate is rejected with its numerical discrepancy unresolved.
Original repeat timing differs by .01%, and same-step losses reproduce exactly.
Use raw XPlane: small chunks exceed trace JSON's ~1M-event cap.
[Full16-configuration sweep, kernel table and reproduction](rmt_write_read_chunks.md).
Artifacts: `/data0/xd/bam_diagnostics/rmt-write-read-chunks`.


### Full18 dynamic RMT Pallas optimization (2026-09-27)

Latest matched runtime9ef9053, `codex/rmt-pallas`, `/data0/xd/rmt-pallas`;
implementation stays in the worktree, main exp.py is the ledger. Direct32 combined
boundaries,18 layers/MLP4078,432119360 parameters, unchanged equations/dimensions.
Attention C256 unchanged. All health OFF, exact AOT, same-VM steps20–49.

| Target / global batch | Original RMT step/s | Selected RMT step/s | RoPE MHA step/s | Gain | Selected/MHA |
|---|---:|---:|---:|---:|---:|
| UC1a v5p-16 /128 |.383933|.468967|.724233|+22.15%|64.75%|
| EW4a v6e-1 /4 |1.202465|1.413598|2.475063|+17.56%|57.11%|

Shared recipe: direct layer scan, packed dynamic projections, token-minor128
write forward, fused C8/QK post-read operations, retain attention output and
post-attention matrix through remat. V5p selects
`RMTCombinedLayerScanMajorDirectWriteProfile` with explicit analytic reverse64;
v6e retains`RMTCombinedLayerScanTokenAllSaveStateV6eB4Profile` with fused AD reverse.
V5p previous best.457066→.468967 (+2.60%); v6e new reverse64/128 reaches only
1.385299/1.390132, so neither replaces its paired1.413598 baseline.

Write reverse kernels185.522→76.553ms on v5p (−58.7%) and51.521→33.175/31.336ms
on v6e. However formatting grows60.587/31.686/32.414ms; v5p device time improves
2165.413→2110.980ms while v6e regresses698.403→713.009/710.664ms. Kernel speed
alone is insufficient: producer/consumer layouts and fusion boundaries are
first-order costs. Next candidate is a true fused write→MLP-read path, not the
JAX mapped/unrolled schedules rejected above.

Use actual topology and batch for VMEM screening. The previously observed default
scoped budgets were v5p16MiB/v6e32MiB, NOT physical capacities: physical VMEM is
64MiB/128MiB per TensorCore. Budgets can be raised with
`--xla_tpu_scoped_vmem_limit_kib`; account for full-step XLA headroom and verify
compiler live allocation before reducing fusion scope. Under the old defaults, major128 passes B1 but requires16.38MiB at B16 and fails v5p. Major64
passes actual full-model AOT/training. V6e256 needs31.39MiB with negligible micro
gain. Do not require the two TPU types to share the same winning kernel/config.

This removes38.59%/29.05% of original RMT excess latency over MHA. Absolute
steps/s across different batches/topologies is not a hardware comparison.
FP32 full-layer/mapped-scan gradients checked; BF16 operation order differs,
so no long-run convergence equivalence is claimed. V5p loss49 is7.749873 versus
old7.752004; short profiles are not retraining evidence. Both retained v6es stay
allocated; temporary UC1a target is verified absent after artifact validation.
[Configuration tables, failed paths, tests and reproduction](rmt_pallas.md).
Artifacts: `/data0/xd/bam_diagnostics/rmt-direct-v5p`, `rmt-direct-v6e`.

Separate NoO speed ablation at8ebb3e9 on retained v6e host-1: O-enabled1.424598,
plain NoO1.445265 (+1.45%), gate-on-key V-only kernel1.444064 (+1.37%). The
specialized reverse kernel saves6.254→3.624ms, yet provides no extra whole-step
gain over plain NoO. Loss neutrality on the current RoPE/dynamic-boundary model
is untested; historical ALiBi final-loss evidence does not transfer automatically.
NoO removes345888 parameters (.2402 W_Q), with no MLP reinvestment.
Artifacts: `/data0/xd/bam_diagnostics/rmt-noo-v6e`.


### Complete three-stage NoO Pallas trunk and phase accounting (2026-09-27)

Runtimebe0d8d3, same worktree/branch. Main trunk is attention read, attention write
plus MLP read, MLP write; both directions use complete Pallas kernels with explicit
analytic pullbacks. Attention operator is unchanged. New default removes dynamic O;
therefore retain a matching NoO implementation control as well as original FullO.
All18 layers/MLP4078/T4096, health OFF, exact AOT, same-VM stable log steps20–49.

| Target / global batch | Original FullO | Previous optimized NoO | Selected three-stage NoO | RoPE MHA | vs original / NoO | Selected/MHA |
|---|---:|---:|---:|---:|---:|---:|
| UC1a v5p-16 /128 |.389900|.482733|.534267|.754033|+37.03% / +10.68%|70.85%|
| EW4a v6e-1 /4 |1.206433|1.445767|1.672400|2.521000|+38.62% / +15.68%|66.34%|

Device forward / backward including remat, milliseconds:

| Target | Original | Previous NoO | Selected NoO | MHA |
|---|---:|---:|---:|---:|
| v5p |652.852 /1860.928|605.036 /1438.095|447.905 /1399.914|401.680 /913.100|
| v6e |204.422 /596.668|182.312 /484.257|122.151 /451.255|80.055 /297.910|

Most improvement remains in forward. Backward occupies75.58%/76.50% of the new
v5p/v6e device step; the complete middle reverse alone costs290.043/81.815ms.
Initial reverse32 actually regressed v5p backward1438.095→1457.992ms despite
higher total throughput. Independent block tuning fixes that regression.
Forward tiles attention/middle/write: v5p128/128/128, v6e128/128/256;
reverse tiles128/128/128 and256/128/256. Budgets48/96MiB, not physical capacities.

Important control correction: forcing the enlarged96MiB scoped budget onto the
v6e baseline executable family yielded original/NoO/MHA1.126367/1.399433/2.343467.
Repeating the earlier default-budget executables on the same retained host recovered
the faster baselines in the table. Use these for headline comparisons, and retain
the matched-budget ablation separately. Budget-matched controls alone can inflate
apparent gain if the baseline regresses; compilation budget is itself a tuning choice.

FP32 equations/scan/remat/shared gradients pass; all tuned blocks pass actual TPU
BF16 output/gradient probes. BF16 short training trajectories do diverge, so these
speed profiles do not establish long-run convergence equivalence. Full stage times,
recompute/optimizer attribution limits, exact classes/runtime hashes and all13
traces: [canonical report](rmt_pallas.md). JSON:
`/data0/xd/bam_diagnostics/rmt-three-stage-comparison.json`.
Temporary v5p node/queue released; both user-retained v6e machines remain READY.

### Middle-stage analytic backward follow-up (2026-09-27)

Same-host controls were rerun: UC1a `xd-v5p-16-rmt-midv2-0927-uc1a` globalB128,
EW4a retained `llm-jax-v6e-1-0` globalB4. T4096, full18 layers, all health OFF.
Throughput is steps20–49 divided by their total implied elapsed time (harmonic
mean of logged rates); trace phases10–14 use complete raw XPlanes. Backward
includes remat; phase figures below are primary-core additive attribution.

|Hardware|Configuration|Runtime|step/s|Forward ms|Backward ms|
|---|---|---|---:|---:|---:|
|v5p-16|Original RMT control|ebd5ba0|0.390299|652.263|1857.685|
|v5p-16|Previous three-stage selection|be0d8d3|0.537099|444.172|1391.479|
|v5p-16|`RMTThreeStageMiddleChunk6458Profile`|4dd7038|0.565299|460.818|1281.913|
|v5p-16|RoPE MHA control|ebd5ba0|0.755166|400.657|911.860|
|v6e-1|Original RMT control|9ef9053|1.202299|204.442|599.530|
|v6e-1|Previous three-stage selection|b9194ea|1.661431|122.157|453.533|
|v6e-1|`RMTThreeStageMiddleRecomputeSavedV6eB4Profile`|e30ec00|1.761599|127.144|414.174|
|v6e-1|RoPE MHA control|9ef9053|2.474329|80.346|302.059|

Selected NoO throughput increases5.25%/6.03% over the previous fused selection,
44.84%/46.52% over original RMT, and reaches74.86%/71.20% of MHA throughput
(v5p/v6e respectively). The original-RMT comparison includes the authorized
NoO change; incremental comparisons use NoO on both sides. Parameter count
remains431,773,472. Branch/worktree `codex/rmt-pallas`, `/data0/xd/rmt-pallas`;
implementation is not merged into main. Full BF16 convergence remains untested.

V5p needs a different local choice from v6e: write qchunk64 inside the complete
middle backward plus cached middle outputs reduces body289.912→246.194ms and
external middle remat57.263→0ms. Forward caching costs16.646ms globally. V6e
favors native token-minor reads and write-only recomputation; middle body is
73.028ms with zero external middle remat. Native-minor without recomputation
has the fastest body65.404ms, but total training is slower1.750065step/s.
The30–50% middle-body speed bet was not met: v5p body improves15.08%, v6e's best
body20.00%; these must not be inflated by silently adding eliminated remat.

Rejected probes: scheduling/merged adjoints alone do not help; native checkpoint
placement gains only0.03%; tiled shared-gradient accumulation slows v6e and
only lowers v5p peak54.39→54.18MiB. Single DMA buffers remove7.35MiB of buffer
windows, but actual peak is48.53MiB, not the naive47.04MiB prediction, because
about1.49MiB of previous allocation reuse disappears. Register-spill declarations
remain40.77MiB. The50MiB-budget v5p final confirmation is0.528133step/s (F443.883ms/B1426.873ms),
slower than minor56 and qchunk64; v6e single-buffer reverse is also slower
5.2585ms versus4.6634ms with default buffering. Capacity savings alone do not
imply a speed gain.

Compute/IO estimates must separate MXU and VPU: published v5p ordinary VPU
throughput is about14.34TFLOP/s/chip, about5.2operations/byte against2765GB/s HBM,
versus166MXU FLOP/byte. v6e's918MXU TFLOP/s cannot be used for normalization,
rank-wise VPU products, reductions or rsqrt. Count padded work, actual buffering,
layout conversions and register spills, and verify peak allocation against the
per-TensorCore physical64/128MiB limits. Compiled allocation declarations are
not automatically the live peak. Detailed derivation and links: [report](rmt_pallas.md).

Artifacts: `/data0/xd/bam_diagnostics/rmt-middle-v2/comparison.json`,
`selected-comparison.json`, allocator dumps and all raw profiles in that directory.

Resource closeout: temporary UC1a v5p node/queue verified absent2026-09-27
16:31:40UTC after all7 profiles; no preemption. Both retained EW4a v6e hosts
remain READY and are not enrolled in cleanup.


## 2026-09-28：Rank-H 写更新与 native VPU 后续

完整证据、配置和代码索引见 [rmt_rankh_write.md](rmt_rankh_write.md)。实现位于
`/data0/xd/rmt-pallas-rankh` / `codex/rmt-pallas-rankh`，测量 runtime `dd87162`；
实现已合入主分支 `refactor-bam`，主 `exp.py` 包含可运行配置与历史变体。NoO、18层、T4096、health OFF，
同机复测原始RMT、既有三段融合最优和MHA，分别统计前传及反传（含remat）。

- 最终 v5p-16 UC1a `RMTRankHVPUUnchunkedV5Profile`：0.637335 step/s，
  较既有最优+12.92%，较原始RMT+63.41%，达到MHA速度84.46%。F434.115/B1109.957ms。
- 最终 v6e-1 EW4a B4 `RMTRankHVPUV6Profile`：1.897464 step/s，
  较既有最优+7.91%，较原始RMT+57.79%，达到MHA速度76.68%。F124.835/B375.718ms。
- 静态、动态层写共用D，先合并C=S+g*rD*An，反传只需两组收缩。
  NN+NT两个MXU dot先胜过带零象限的对称矩阵；两头寄存器分块、显式完整展开
  75/48项的VPU再胜出。独立乘/加而非硬件fused FMA，按实际指令计成本。
- 局部kernel胜出不等于融合段胜出：VPU沿用major ABI时明显回退；投影、归一化、
  收缩与epilogue全部保持token-minor之后才兑现整步收益。保留全部三段融合边界。
- 小qchunk不必然省VMEM。旧major qchunk64桥接需求75.84MiB，其中51.82MiB是
  寄存器溢出；直接native-minor K2反而可在原58MiB编译预算通过完整v5p模型。
  必须区分声明scratch、padding、存活期与spill，不能只缩小逻辑tile。
- K1小residual只占约一层13.5MiB，收益小但正向。改逻辑scan carry并未自动消除
  真实HBM复制；MXU/VPU混合没有实现预期收缩重叠，未选用。
- FP32全梯度、三层scan四种remat及两硬件BF16检查通过；未宣称逐位或长期收敛等价。
  原始v6e/MHA曾因统一预算而退速，最终比较使用更快的原已验证AOT控制。


### 2026-09-29：QK 的 M 归一化与投影拆分开销

纯 JAX XLProp NoO 的6层/B1/T4096定位，runtime `0e487d4`，EW4a同一预留
v6e-1；三路均开基础与RMT健康指标。原始 XPlane 的5个训练步均值：

| 配置类 | ms/步 | 相对控制的耗时 |
|---|---:|---:|
| `RMTXLPropNorm6LControlProfile` | 103.560 | — |
| `RMTXLPropNorm6LQKInputProfile` | 106.316 | +2.66% |
| `RMTXLPropNorm6LQKOutputProfile` | 104.263 | +0.68% |

输入归一化版将静态 QKV 投影拆为 QK 和 V，第一个完整profile步的数据布局转换
比控制多2.275ms。将逐token的同一标量归一化移到矩阵QK读出后，保留合并QKV投影，
可回收大部分开销。数学等价不代表BF16轨迹完全一致；尚未验证完整28层v5p-32，
正式训练仍用 `562bfb9`，不能把此表当作完整模型的已兑现加速。
实现worktree `/data0/xd/rmt-xlprop-noo`，分支 `codex/rmt-xlprop-noo`；
原始trace与解析结果 `/data0/xd/bam_diagnostics/rmt-readnorm-speed-0e487d4/`。
详见 [实验记录](rmt_prop_read_norm_training.md)。


## 2026-10-13 JST：MediumProp 静态入口与最终 M 输出（均未胜 SOTA）

直接基线为 `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile`，runtime64e3fc3；TruePile4096、UE5a v5p-16、20–99步吞吐对齐。实现 worktree `/data0/xd/mediumprop-full-m-read-gelu128`，分支 `codex/mediumprop-full-m-read-gelu128`；主exp是ledger only。完整配置、初始化、累计轨迹及资源证据见 [实验记录](mediumprop_embedding_initialization.md)。下表为各停止时最后完整共同窗口，gap=实验−SOTA，不代表未观察的13500步终点。

| 方案 / runtime | 最终checkpoint / 比较窗口 | loss gap | 稳定速度差 | 净参数差（W_Q） |
|---|---:|---:|---:|---:|
| 因子化静态写U/A/S / f6ce0a4 | 4531 / 4400 | +.014657 | +.0444% | -.001033 |
| 独立矩阵embedding / f6ce0a4 | 4427 / 4400 | +.008160 | -.5196% | +83.050278 |
| 自由直接B写 / 0f5f6c4 | 2894 / 2800 | +.014938 | -.0183% | +.006944 |
| 最终M输出 / 0f5f6c4 | 5125 / 5000 | +.004474 | +.0470% | +.005000 |
| 最终M按列门控输出 / 1810a41 | 5121 / 5000 | +.003643 | +.1514% | +.031689 |

- 新增实质参数从MLP抵扣；矩阵lookup和最终门控的MLP宽度按用户明确例外保留。额外lookup表为84.0533W_Q，净增加83.0503W_Q；没有新增前传dense投影，不等于没有lookup/优化器/带宽开销。
- 静态入口去掉原写地址pre-RMS-bias梯度尖峰，仍未改善loss。直接B显著消除因子化静态写的启动劣势，但到2800只比静态写好.002577，仍距SOTA+.014938。梯度范数和参数化启动速度都不能替代后期loss证据。
- 最终门控与父路保持相同MLP、共享参数初始化、初始logits和输出投影梯度幅度。1400..4000持续约-.0016局部优势，到4200..5000缩成均值-.000642，并在4600短暂反转；SOTA差距同期均值+.003894，前五窗+.003929，停止符合用户5000趋势审查。修正中期看法：局部收益未保持同样幅度，没有已兑现的SOTA收益。
- 五路0抢占、均主动停止，checkpoint/资源删除/TB同步已核验；全程UE5a，未切区。吞吐差均不到1%，不能宣称实质加速；M-cache无新增优势。
- 此前逐token门改逐列均值、rank截断的消融建议已撤回优先级：用户目标是突破当前SOTA；当前证据没有显示该失败输出支路值得继续投入。未启动这些后续实验。

历史补充：廉价最终M动态读已跑过：cb65817 Direct32的16个v32条件读头，仅.442511W_Q，动态/静态MLP地址两路到2800相对父版last5+.008508/+.005725，分别停2977/2971。更早V2UnEmbRead亦未建立收益。四读头+投影建议只比旧廉价方案省约19.4%，不能仅凭比本轮dense输出省82%就当作新突破方向；已撤回优先推荐。旧底座不是当前SOTA精确对照，但这些负证据必须纳入新实验选择。详情见上面的实验记录历史审计节。

## 2026-10-13：MediumProp T8192 同步7000步收尾

固定每步524288目标token，保留原参数/LR/优化器；MHA/Mudd使用用户允许的Splash，BAM共享R256+泛化读端保留父实现。相对同长度MHA的BAM/Mudd收益倍数，最近五同步窗口T2048/T4096/T8192为1.4061/1.6840/1.7398；ratio-first二阶4k/2k为1.1977，8k/4k为1.0332（范围1.0234–1.0406）。因此2k→4k收益明显，4k→8k收益较小且初期还曾略偏向Mudd，不能用早期结果固定后期结论。后者绝对额外领先量近期~.018基本稳定；倍数上升部分源自Mudd额外收益消退更快。不同长度记录未配对，不把跨长度raw loss符号当纯上下文因果。

7000步BAM−MHA/−Mudd为−.203961/−.086375，后者近期均值−.087944，收窄速度+.003618/千步，足以回答本轮长度问题。按用户允许的稳定后停止条件收尾，最终CP9639/8908/7140，均0抢占，资源删除/TB同步核验。停止点并非13500终点；初始二阶1.13下注未在观察范围兑现，后来near1预测低估了小幅正收益。BAM速度相对MHA/Mudd为−26.06%/−19.96%，相对T4096−18.75%；参数与推理M-cache没有新增开销。四路M-shift仍保留T4096方向SOTA作直接基线。详见[完整7000步报告](reports/mediumprop_t8192_review_7000_20261013.md)。


## 2026-10-13：MediumProp 动态 M-shift 阶段结论

实现 worktree `/data0/xd/mediumprop-m-shift`，分支 `codex/mediumprop-m-shift`；从最新主分支0f7f6d10派生并审计TB500步轮转。四路runtime52cf0b3，TruePile4096/global128/UE5a v5p-16、参数432111680（较SOTA仅+576偏置）。新门691776参数=.4804W_Q，MLP[3802,3676,3802]抵扣691200核参数。所有比较保留共享R256+泛化读端SOTA，按用户指示移除MHA。

| 方案 | 最终CP / 最新完整窗口 | 实验−SOTA | 速度差vs SOTA | 状态 |
|---|---:|---:|---:|---|
| read .05 | 3699 / 3600 | +.001446 | −4.251% | 判定被read .5支配，停止；P1 |
| persist .05 | 2957 / 2800 | +.007843 | −3.930% | 后期劣势平台，停止；P1 |
| read .5 | 进行中 / 3600 | −.006434 | −4.256% | 最近五窗均值−.007875，继续；P1 |
| persist .5 | 3095 / 3000 | +.007714 | −4.191% | 劣势平台，热替三门；P0 |

持久混合把跨token操作写回跨层状态，此轮两种初始开度均失败；read .5保留约.008 loss收益。read .05自2000至3400落后read .5约.009，最终共同3600仍+.007881，没有速度、预算或cache优势。三路停止的CP提交、资源归属、TB同步已核验；310176/177释放，310179热转移给三门版。原4路均1404自定义健康指标，后续新增已收紧。

用户要求Q/K/VO三个独立v32门，runtime e4e6e17，新RUN `BamMediumPropSharedR256GeneralReadMShiftReadQKVoG050TruePile`。额外两门=.9608W_Q，MLP[3781,3654,3781]补偿，总参数432112832（较父路+1152偏置）。实际324健康指标，无逐列指标；速度.442375step/s，较父路−3.4853%、较SOTA−7.5931%，包含324vs1404日志差异。初始.5零权重，CPU与实际启动检查通过。200/400/600/800步相对父路−.267658/−.082765/−.040955/−.024986；早期优势明显收窄，不能据此宣称终点上限突破。实际配置、目标token数、原始共同loss点及初始函数审计未发现非预期对照差异。400步K门平均前token份额.554，高于Q .277/VO .231，说明模型使用独立时间选择；不把描述性健康当完整因果解释。当前最值得验证的是这一已有训练能否把启动优势转为后期平台收益，无额外消融任务。终点下注仍为新−父−.003、新−SOTA−.009。

两类read均需上一token原始M的每层缓存，三门共用一个缓冲，不是三个；无新增推理M-cache优势，当前实现未验证decode。完整配置、轨迹与资源证据见[实验记录](mediumprop_m_shift.md)。


M-shift后续核验：共享read .5到5000仍−SOTA.008717，最近五窗−.008591；默认MHA参照相对SOTA收益倍数1.045728x，近期均值1.043695较前五1.036278提高。三门到2000−SOTA.015143、−共享父路.008053，MHA参照倍数1.048435x；600–1000约1.0315，1200–2000约1.0427，绝对gap收窄不能等同于比例收益消失。用户指定默认参照为MHA，之前Mudd-first汇报已承认并更正。通用三参数接口与持久监控已集成到权威run_registry.py/9f68649并部署，17报告测试及实际auto-event核验；不再临时写每轮计算脚本。当前两路继续，尚无终点结论。


三门3000：−SOTA.012459、−共享父路.004165，默认MHA参照相对SOTA倍数1.051550x（近期均值1.050195，上组1.042730）。方向收益持续，但共享门本身也在进步，三门额外父路优势近期均值从−.012090缩至−.005692。因此下调三门终点额外收益预期：原−.003下注保留，当前预测接近平手到−.001（中心−.0005）；不能只看SOTA-relative倍数稳定就认为三门提高了共享门最终上限。继续到下一阶段检验是否形成非零收益平台。


4000后预测修正：三门−父路.002785，额外优势收窄速度明显减慢。带非零下限指数拟合不同起始区间得到平台−.0024..−.0036；把收窄直接外推平手过于悲观，撤回3000时近乎平手预测。当前下注终点−.0025，5000−.0028，原始终点−.003仍保留。拟合在4000观测后完成，非事前验证；5000实测检验，不据此宣称已观察到终点平台。共享门7000仍−SOTA.009329，近期均值−.009368；默认MHA参照相对SOTA倍数1.054842，近期均值1.054316，上组1.048949。两路继续。


### MediumProp temporal read shift update (2026-10-13)
Single Read .5 retains loss−SOTA~−.01045 at10k (recent MHA-ref gain/SOTA1.067665); three-gate extra shrinks to recent−.000714 at7k (gain/parent1.003928) for−3.49% throughput. The latter approaches parity, not a validated−.0025 terminal floor. User-authorized two-lag independent sigmoids.5/.3/.2, pureKV.5/.5, and singleM.5+KV.5/.5 all runUE5a with MLP-repaid budgets. Verified speed vs single:−.15%/−9.07%/−11.53%; KV speed optimism was wrong. M75×32 and completeKV2×16×75 both2400 values/token; vectors are not intrinsically smaller here. Inspect optimized HLO before attributing overhead to layout/copies/fusion. Full configs/hashes/counts/health/bets in `mediumprop_m_shift.md` and main exp.py. Temporal-gate reports reuse training skill `report_bam_shift_health.py`, no new column metrics.


2026-10-13 MediumProp M-shift review: singleRead.5 retains SOTA gap−.010521(last5 through12000), MHA-ref gain1.072313, speed−4.26%. Q/K/VO separate gates extra gain fades: parent gap−.001660@5000→last5−.000123@9000, repeatedzero crossings; stopped9198 vs−3.49%parent speed. More conditional gates did not preserve stronger asymptotic gain. Two-lag at2000 remains parentparity(last5+.000156); secondlaggate.0685 concentrates earlylayers(.1090/.0647/.0317). NativeKV2000andhybrid2000pending.
Gate/address feature-sharing proposal(originalSOTA, not M-shift): reuse readGELUR256 for8dynamic/staticQKVO gates, ownaddressGELUR256 forattention/MLP writegates. Ten1200→16kernelsbecome256→16,saving2718720=1.888W_Q; refundMLP3855/3728/3855,+2880rounding. InitialreadzeroW preserved. Fixed006normalwriteW must rescaleagainstlatentRMS; GaussianRMS1 estimate latentRMS.10785,newgateσ.12045 to preserve parentlogitRMS. Actualbatchverifybeforelaunch. Discussion only.
KV optimizedHLO probes reject simplistic transpose cause: directmixlayout leaves12largecopies; sample-firstequivalenthealth merely movescopies to naturalK/Vlayout(total513unchanged). No rolloutwithouttimingproof; borrowedFLEXcompileonly,nolifecyclechange.


### 2026-10-14 单门 read050 的读写投影共享（准备中）

独立最新主分支0f7f6d10工作树 `/data0/xd/mediumprop-projection-sharing`，
分支 `codex/mediumprop-projection-sharing`，runtime `c9a2a360`。三路 RUN：
`BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile` /
`BamMediumPropMShiftReadG050ReadAttentionSharedR384TruePile` /
`BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile`。
最后一路仅6个MLP写回层共享三路R384，其余12层为读+attention写R384。
实际参数432111680/432121472/432118688，MLP3887/3762/3887、
3772/3646/3772、3794/3669/3794。R384扩大up后整体不省参数，MLP据总量抵扣。
TPU310190/191/189（首轮187/188因CPU门禁失败已进入GCP删除，交接用新队列），UE5a SPOTv5p16，预留EW4a FLEX v6e编译；三路FIRST_STEP+14均核验。
canonical20–99速度.464425/.464400/.465800，比read050快1.325/1.320/1.625%；健康记录精简，不是隔离架构的配对测速。
新测试+writer5+full47 BAM封存门禁；TB500轮转，CP200保留最近2。
相对单门read050的跑前下注loss +.003/-.001/+.001，速度.460steps/s。
用户预期更乐观，今后建议开跑不能仅以预期-.001微小改进为充分依据。
原SOTA13500 checkpoint 128序列、96fit/32held同6层诊断：读+attention
压R384 +.002202，attention+MLP +.003396，三路R384 +.010566；
诊断不等同训练上限，且未验证与M-shift的叠加适应。


### 2026-10-14 XLProp 单门 read .5移植（已提交准备）

RUN `BamXLPropSharedR400GeneralReadMShiftReadG050TruePile`，M-shift家族工作树
`/data0/xd/mediumprop-m-shift`，latest-main0f7f6d10基础设施审计无差异；
runtimeeeec67ce，TPU310192 UE5a SPOTv5p32，50000步，CP250/keep2/no永留，TB500。
父路XL sharedR400 generalizedread，D1920/L28/V40，gate1920x40+bias40
×28=2151520=.583637W_Q。MLP[6127,5938,6127]+末层6126，
实际1432383640，父路1432386360余差-2720。只有新增门bias排除WD，旧XL规则保持。
新增健康140个汇总，无列统计。参数树、v40只读carry、末层scan/梯度/因果性和
compacthealth已通过开发检查；封存CPU门禁+AOT+trainer预队列完成，FIRST_STEP+14/AOT加载/实际worker1432383640参数已核验。
跑前下注同XL内部loss-parent-.008、收益倍数1.06–1.08、速度.367steps/s。
跨尺度只比收益倍数，不比lossgap；按step/planned_steps对齐，Medium2700→XL10000。
家族专用report_mshift_cross_scale.py，MHA作各尺度reference，原SOTA作comparator，
2%阶段精确五个10步点、不插值，三项脚本回归通过。旧XL SOTA截止33002，
父路倍数只支持约66%阶段，不外推其后基线。


2026-10-13: KV and M+KV completed13500 (e79a303/77ea4bb), UE5a v5p16 P0/P1. Final12600-13400 means: single gain .011034, KV .006075, joint .013015 vs originalSOTA; extra joint .001981, A~76.19%, C~32.82%. Single did not catch joint; original terminal -.002 bet matched. R256 shared read/attention and tripleR384 still running~9.8k: normalized gains vs single remain~1.02-1.03, early triple advantage largely gone. XL singleMshift at7000: MHA-ref gain1.065168 vs parent; same14% Medium1.016076. Cross-scale only ratios. Artifacts: /data0/xd/bam_diagnostics/mediumprop-m-shift/paired-13400.json and status-latest-reports.json.

2026-10-14 projection sharing final: pairR256 and tripleR384 both completed13500, runtime c9a2a360, direct final5 gap triple-minus-pair -.000141, effectively tied. SingleRead050 gaps -.003499/-.003640; originalSOTA gain ratios1.100491/1.101467; speed+.30% triple vs pair. Initial predictions +.003/+.001 vs single were wrong. Follow-up mixed[256,384,256] and all256 a445bbc8 stopped2859/2861, all CP/resource/TB verified: latest2800 normalized gain vs pair .981786/.965099, no recovery and slower, so combining layerwise did not improve. Six-write-layer R384 capacity improves over all256, but reducing other12 layers to256 also loses vs allR384. Cannot infer per-layer substitutability from whole-model results. XLsingle at10000 crossscale20%1.051370 vsMedium1.032471; known Medium terminal1.076290 is forecast anchor, expect likely crossing unless XL rebounds.

2026-10-14 failure mechanism review (user challenged insufficient analysis): full sharedR256 removes MLP-private256 from six layers; read/attention use pre-attention normalized residual whereas MLP writes use post-attention normalized residual. Hypothesis: excessive tying across purposes/input distributions, recoverable partly by384, not proven gradient conflict. Mixed failure disproves inferring layerwise substitutability from whole-model pairR256-vs-pairR384 comparison. Init M RMS four arms1.25-1.26; @1000 pairR2563.442/tripleR3843.486/mixed3.247/all2562.889, gates live; may reflect write strength/coherence/use, not causal evidence. CPU rank-selector identity audit: original pairR256 and tripleR384 versus identical per-block ranks produce bitwise-equal params/loss/grads (tiny scanned fixture). Artifact /data0/xd/bam_diagnostics/mediumprop-projection-sharing/rank-selector-identity-audit.log. Remaining rank/MLP budget/init confounds explicit. Proposed, NOT LAUNCHED: six write layers use shared128 + read/attention-private128 + MLP-private128 (union384, eachconsumer256), anchored to successful pairR256 and refund saved parameters toMLP. Discriminating diagnostics: consumer-specific Down gradient alignment plus MLP write increment/currentM alignment and functional readout contribution. No new training authorized by this discussion.

2026-10-14 correction incorporating failed pairR384/private-MLP control: prior private-feature hypothesis was overprioritized. At2800, parent gaps: pair256/private256=-0.0077184; pair384/private256=+0.0014794; allshared256=+0.0016432; allshared384=-0.0086464. Triple384 minus pair384 at2000/2200/2400/2600/2800: -0.012053/-0.012498/-0.012327/-0.011196/-0.010126. Sharing MLP hurts at256 but helps at384; neither generic anti-sharing nor necessary-private-feature explanation fits. Pair384->triple384 simultaneously expands MLP address features256->384, couples MLP training into shared Down, and refunds net1,449,984 projection parameters into MLP widths [3772,3646,3772]->[3794,3669,3794] (total parameter difference -2784). Capacity/budget and constructive read-write feature learning remain competing hypotheses, not established causes. Mixed combo preserves six triple384 layers but narrows remaining12; its +0.005814 gap vs triple384 at2800 indicates context-dependent value of those ranks, not incompatible M shapes. Downgrade shared128+private128 proposal from preferred next experiment to speculative. Diagnostic priority: explain historical pair384->triple384 improvement, separating address-feature capacity, functional write/read compatibility and MLP-budget effects; matched-stage evidence required. No new run launched.

2026-10-14 user-authorized resume from committed2859, unchanged a445bbc8/runtime/AOT/s13500, UE5a xd-v5p-16-310193-maxtext. Prior stop was premature: gain vs triple384 .973037->.974139->.978401 at2400/2600/2800. Terminal outcome unresolved. Observe at least5000, continue while credible recovery; report every1000 against pair256 and triple384, MHA-reference ratios. Relaunch submitted02:19:09Z; FIRST_STEP pending.

2026-10-14 cross-layer projection diagnosis (user uniform-rank hypothesis): pair256/triple384 step13500, six consecutive3-layer blocks, reserved Flex EW4a. Dominant directions overlap beyond random, but full shared384 retains90.4–91.2%/84.4–85.8% weight energy. Same32-sequence plain-JIT interventions yield loss costs shared256 +.007053/+.013568; shared384 +.003214/+.006799. Same compression factor3R->R gives similar ~.007 costs. Blockwise384 localization: first block not dominant; excluding first yields+.002867/+.005450. Common-basis A with layer-specific pre-GELU C is supported as a training candidate, not direct feature tying or proof uniform rank causes mixed-run lag. Full report /data0/xd/mediumprop-projection-sharing/experiments/bam_llama2_medium/cross_layer_projection_sharing.md; scripts committed4e248efe; artifacts /data0/xd/bam_diagnostics/cross-layer-sharing-20261014. Mixed run resumed2859, AOT/data/CP3000 verified, nextreview>=5000; do not label terminal failure.


Cross-layer sharing race (2026-10-14): eight from-scratch arms authorized: ReadAttnR384 / TripleR512 × conservative / direct × blocks / thirds; no global/halves/adapter. Runtime f8755258, codex/mediumprop-projection-sharing, `/data0/xd/mediumprop-projection-sharing`. Parameters matched through MLP; audit/bets/arm table in that worktree's `experiments/bam_llama2_medium/cross_layer_race.md`. Owned UE5a v5p16 IDs310200–310207; preparation submitted, FIRST_STEP not yet verified.

2026-10-14 cross-layer race update: authorized original8+square R384 conservative inner384/MLP-shared thirds =9, IDs310200–310207/310216; all nine FIRST_STEP/AOT/full parameter counts verified. Direct runtime f875525; conservative runtime38af4cc fixes bridge sharding without relaxing production tolerance. IDs310208–310215 withdrawn, never trained; four owned AOT nodes/queues verified released, reserved Flex preserved. Nine-row combined report every1000, XL every2000; both historical sharing winners retained, MHA-reference gain sequences registered. Startup evidence /data0/xd/bam_diagnostics/cross-layer-race-20261014/verified-startups.json; speeds in verified-speeds.json.

### 2026-10-14 cross-layer sharing second review

Nine-arm race (runtime38af4cc conservative / f875525 direct, worktree `/data0/xd/mediumprop-projection-sharing`): user explicitly approved stopping conservativeR512 thirds CP5198, directR512 block CP6200, square384 conservative thirds CP4960. All checkpoints committed, corresponding xd-v5p-16-310205/310206/310216 nodes and queues absent, TB SYNC_OK. Four arms remain: conservativeR384 block/thirds, directR384 thirds, directR512 thirds. Main exp.py records final direct-baseline results; full evidence `/data0/xd/bam_diagnostics/cross-layer-race-20261014/second-review-closeout.md`.

At5k conservativeR384 block is near oldTriple384 parity (last5 +.000507) and slightly ahead of oldPair256 (last5 -.000377). R384-family reports require gap/gain sequences and MLP parameter deltas against oldPair256 as well as common oldTriple384; MHA is the gain-ratio reference. Direct512block continued catching up to +.002614 at6k; its user-approved stop was a resource-allocation decision, not evidence of no late upside. Square384 has no demonstrated advantage vs its closest conservative384thirds control, with fewer MLP parameters. Larger MLP late payoff remains a reason to retain conservative384thirds.

XL singleMShift reached20k/CP20k: MHA-reference gain1.069456 vs Medium same40%1.044693; Medium terminal1.076290 remains the relevant late-stage prior. Next XL report22k.

2026-10-14 cross-layer thirds calculated extrapolation: through7200, fit starts2000/3000/4000; rolling heldout4600→6000 and5600→7000. Ratio forecasts vs oldTriple384 center~.988 conservativeR384thirds /~.990 directR512thirds; subjective prior.995/.990 corrected, MLP late payoff remains unobserved hypothesis. User authorized stops: CP7538/7713, TPU+queues absent, TB SYNC_OK. Last complete7400/7600 gaps+.0022584/+.0017974 vsoldTriple384, ratios.987590/.989877. Only conservativeR384block remains in original9; continue newMediumactivationreuse256/384 and twoXL runs. Full reproduction /data0/xd/bam_diagnostics/cross-layer-race-20261014/extrapolation/.


### MLP hidden-channel BAM read, runtime3afcaa9 (2026-10-15)

Three authorized Medium original single.5 arms run from `/data0/xd/mediumprop-mlp-bam-read`, branch `codex/mediumprop-mlp-bam-read`, latest-main parentc5e02ad8. Read actual attention-written M into8x75=600 SwiGLU value channels, no W1 on these channels, existing Wg/W2 retained. Independent R256 generalized column addresses; inherited static/dynamic gates control their mixture, no extra common output gate. Static/dynamic keys combine before ONE contraction. Deferred attention delta is reused, decay once. Added locationswrite6/nonwrite12/all18, locally refunded total MLP widths[3802,3767,3802]/[3893,3676,3893]/[3893,3767,3893]; parameters parent+384/+768/+1152 only. Forward net extra.225/.450/.674MFLOPs per token, dominant full-model+.024%/+.049%/+.073%. CPU59 and actual AOT/FIRST_STEP/tree/config verified. UE5a v5p16 TPU310246/247/248;20–99 .4780875/.468775/.4575625step/s, versus health-OFF same-shape timing control-1.25%/-3.18%/-5.49%. Full18 three-seed initialization finite, read/W1 RMS grows2x→10x; keep native scale, not a matched-amplitude control. Pre-run terminal new/parent MHA-ref R bets1.08/1.12/1.15. [Full design, budgets and evidence](mediumprop_mlp_bam_read.md).

Common1k: write6/nonwrite12/all18 versus original single.5 gap-.003187/-.009198/-.012516, MHA-reference gain1.00545/1.01574/1.02141. All three extra gains contracted from200; nonwrite12 presently offers more favorable effect/overhead than write6. Continue all; original terminal bets not established. Full200–1000 series in mediumprop_mlp_bam_read.md and report-common1000.json.

2026-10-15 attention-only post-residual write review3k: parent gap-.010694 and MHA-reference gain1.04278 (latest five mean1.04472), versus both-change gap-.004530 andgain1.01768. Continue attention-only; both-change/MLP-only user-stopped6167/2231. Full note mediumprop_post_residual_write.md.

2026-10-15 MLP-read follow-up: write6 stoppedCP2008 for hot replacement; gatednonwrite12/all18 authorized2800 review stopsCP2970/2909, resourcesabsent/TBsyncOK. Final2800 originalsingle-reference ratios.997179/1.000524: earlyextraadvantagevanished. New no-read-gates/static-zero all18 runtimec560cd6 active, locally matched432090944params, widths[3898,3772,3898], dynamicaddressesnormal.006/staticSzero; readnotzero. First1k versusgatedall18 gap-.004948/R1.008287, versusoriginalsingle gap-.017464/R1.029876, totalgainstillcontracting. Health-OFF speed.4629875 versusgatedall18.4575625(+1.19%); no historical health-ON timing comparison. Forwardnetextra.631MFLOPs/token(+.0686%). Fullhistory andparameterflow inmediumprop_mlp_bam_read.md.

2026-10-15 attention-only5k parentgap-.007665/R1.038450, latest5R1.040022 stable; versusbothgap-.003497/R1.017185. Sharedattention/MLPwriteR256 actualactivation childnormA/M first1k parentgaps+.003785/+.003404, R.993860/.994477, deficitscontracting; speedflat, extraMLPrefundlocally+85width. Neithernewnormwinnerestablished. Initializationpair6k primaryBAM/Mudd1.720946/1.612302, second-orderunfixed1.366223/1.279973, oldrecipe1.155768/1.082803. Allsixcontinue.

2026-10-15 zero-static/no-read-gates nonwrite12 launchedae19f92 UE5av5p16TPU310251, FIRST_STEP8/CP200committed. Actual432097856params/MLP[3898,3676,3898], localintegerremainder-13824;8CPUchecks/AOT/workerconfigallverified, allhealthOFF. Speed.4709125(+.46% vsoldgated12,+1.71% vsnew18). Reader.2592W_Q/selectedlayer, forwardnet.420864MFLOPs/token(+.04575%). Betnew/oldgated12 MHA-reference1.05[.98,1.12]. New18at2k versusold18gap-.001665/R1.005154(last5R1.009742), versusoriginalsingle.5-.005009/R1.015666 after1.4kpeak1.035364: totalbenefitstillcontracts. Allsevencontinue, nofirst1keffectjudgmentfornew12.

2026-10-15 follow-up: all18zero-static/no-read-gates3k parentgap-.0036138/R1.014456(last5R1.019104), nofullcollapse butgainmodest; new12first1k parent-.0144866/R1.024783, oldgated12-.0052888/R1.008908, new18+.002977/R.995055, faster1.71%. Historicalterminalnew/oldgated1.05betscannotbeverifiedbecauseoldstopped~2800; useoriginalsingle.5 andactive12/18forfutureterminalcomparison. Continueboth. Init8kprimaryBAM/Mudd1.640465/1.557850, second-orderunfixed1.368075/1.299178, oldrecipe1.105249/1.049589; full-originallast5 nowallaboveconditionalrecoverythreshold1.284081.

Newattention-onlyMLPnormreusecontrolruntime6c586f8, UE5aTPU310252, FIRST_STEP6/actual140leaves432111680/MLP[3802,3676,3802]/allhealthOFF verified, 62CPUchecks+AOTpass. GatesandaddressesuseexistingMLPnorm onpostattentionresidual, independentwriteR256preserved. Exactinitialparams/outputmatchparent; learnednormscalestestprovesnewinputs andonecallpernorm. .4855375step/s(+.232% vshealthOFFparent) matches+.3%prior; terminalnew/parentgainbet1.01[.99,1.04]. Shared-writepair3kA/M parentR.996351/.998552, peerM/A1.002209(last5 1.003580), notyetaddedsharinggain. Parentattention-only7kR1.025649(last5 1.030948), shrinking.

CorrelatedUE5a2026-10-15T1232 firstpreemption: initoriginalCP7770->7771, shared-writeA2975->2976, newMLPread12 731->732, allsamezone/exactruntimes/full13500schedule/AOTresumed; firstperiodicCP7800/3000/800committed. Leasesinregionalhistory. All8RUNsactive.

2026-10-15：用户停共享写attention norm于4673，末五对attention单改gap-0.0000327/R1.000132，未建立净收益；MLPnorm继续。attention单改9k对原单门.5末五R1.025854，较8k1.030165收缩。完整证据见mediumprop_post_residual_write.md。

2026-10-15：用户相继停共享写attention norm4673/MLPnorm5354、去读门零静态MLP读18层5117，均CP提交/资源释放/TBSYNC_OK。共享写两路末五对attention单改R约1.0001/1.0010；18层读M末五对原单门R1.00695，较3k1.01910继续缩水。12层读M继续。attention单改/MLPnorm来源对照2k对父路末五R.999459，少一次norm速度+.23%，尚无额外loss收益。初始化组共同10k：BAM/Mudd1.61245/1.54850；对未修复串行二阶1.39025/1.33511，修复效果保持到中后期。详细证据见各组note。

2026-10-16 JST（UTC15日）：MLP读12层也按用户指令停3149，至此MLP读整组均已收尾。attention单改父路11k末五R1.019387/gap-.003120，较10k1.025413/-.004185继续收缩，撤回“已稳定”判断；全程及后段放大图见本地post-residual-write/selective/attention-only-trajectory-20261015T1453.png。改用MLPnorm对照3.2k对父路R.999304/gap+.000174，仍近持平。两者例行报告已合并为一个完整组，不能用“单路数据+一句同组进度”冒充一起报。初始化组共同11k修复/全原BAM-Mudd末五1.622815/1.549095，绝对loss修复路末五领先.005802，最近约1k稳定；与attention单改父路的收缩是不同方向、不同配方，不能混淆。
