# BAM performance history

This document keeps the profiling evidence behind the current conclusions. It is not the
authoritative result summary; use [`bam_exp_memo.md`](bam_exp_memo.md) for current milestones and
canonical throughput. Raw XPlane artifacts are indexed at the end of that document.

## Conventions

- Unless stated otherwise, profiles use `B=32,T=2048`, bf16 activations and step 10–14 XPlane.
- One `(W_Q)` is one `D→D` projection. For `D=1024,n=16,k=v=32`, a forward projection is the
  common unit used for theoretical comparisons.
- Scope wall time is not additive: fused operators can be attributed to several semantic scopes.
- Paired `Δwall` is the whole-step marginal effect of changing a path, not that operator's isolated
  execution time.
- Six-layer profiles screen same-shape kernels; speed involving layout, remat, memory pressure or
  collectives must be confirmed at full 24 layers.

## Read-M lowering

The original PerHead LocalQK graph established that the two LocalQK reads contain exactly twice the
read FLOPs of one fetched read, but cost more than twice the time because their Q/K outputs are
materialized into the attention forward/backward graph.

| Metric | LocalQK / fetched read |
|---|---:|
| theoretical FLOPs | 2.000× |
| XPlane FLOPs | 2.002× |
| XPlane bytes | 2.287× |
| XPlane scope time | 2.353× |
| paired `Δwall` | 3.230× |

For a single bilateral contraction, `btkv,btnv→btnk` and `btkv,btnk→btnv` have equal theoretical
FLOPs. Their measured ratio against MHA QK was only `1.092:1:3.945`, rather than `1:1:128`, because
the small `32×32` reductions achieve about 33–35× lower effective compute throughput than MHA.

### Equivalent implementations

Same v5p-16 graph, original PerHead path:

| Path | XPlane step | LocalQK scope | fetched-read scope | Decision |
|---|---:|---:|---:|---|
| dot to `bntd` + transpose | 3425.7 ms | 565.3 ms | 239.7 ms | reject |
| dot directly to `btnd` | 3412.2 ms | 560.0 ms | 236.2 ms | reject |
| multiply + reduce to `btnd` | **3191.9 ms** | **378.5 ms** | **156.5 ms** | keep |
| packed Q/K dot | 3396.1 ms | 550.4 ms | 236.1 ms | reject |
| squeeze `n_f=1` | 3407.5 ms | 559.3 ms | 269.0 ms | reject |

After FactorizedLocalQK shrank the contraction, multiply+reduce still improved the complete step by
1.59%, but the individual contractions only by 0.5–0.9%; the larger gain came from changed
layout/copy/fusion. This is why the final choice is based on paired whole-step speed.

### Batched Q/K LocalQK read

At `66c8173`, Q/K keys were represented as one `qk=2` axis and passed through the shared
`bam_read(..., return_sides=True)` path. The same-VM v6e-1 eight-layer G C256 S/U pair was negative:

| Path | XPlane | Stable step/s | LocalQK scope | read-M contraction | compile |
|---|---:|---:|---:|---:|---:|
| separate Q/K | 672.17 ms | 1.474 | 22.42 ms | 7.06 ms | 30.71 s |
| batched `qk=2` | 679.40 ms | 1.458 | 26.03 ms | 11.62 ms | 32.80 s |

The separate source graph already lowers to two fused forward contractions, one per matrix side.
Making Q/K an explicit size-two axis therefore does not reduce forward kernel count, while its
gradient over the shared M adds two reductions per layer: forward+backward read-M kernels rise from
6 to 8. Reject the batched path; no full-24 confirmation is warranted.

### Pure-JAX block read

Packing row/column reads into `[[0,M],[M.T,0]]` made the six-layer LocalQK step 0.97% faster, but the
full-24 confirmation was 1.04% slower: read-M traffic rose from 22.25 to 80.49 GB and read time from
80.12 to 129.71 ms. All block-read variants are rejected.

## Pre-RoPE speed anomaly

The apparent Pre-RoPE+QKNorm acceleration was a dtype boundary, not a QKNorm benefit.

| LocalQK injection / QKNorm | v5p-16 XPlane |
|---|---:|
| post / off | 3058.7 ms |
| pre / off | 3072.6 ms |
| post / on | 3094.6 ms |
| pre / on | 2588.6 ms |
| pre / off + cast merged Q/K to bf16 | **2539.0 ms** |

The float32 RMS-gate bias promoted LocalQK output to f32. Without a later dtype boundary, adding it
to bf16 Q/K promoted attention to f32. QKNorm happened to cast the merged values back to bf16;
explicit casting was another 2% faster. Subsequent BAM paths keep activations bf16 deliberately.

## Write and diagonal-one pairs

Equivalent v5p-16 profiles:

| Change | 6-layer result | 24-layer result | Decision |
|---|---:|---:|---|
| dynamic-V write: dot → multiply+reduce | 528.18→498.12 ms | 1914.68→1775.48 ms | keep multiply+reduce |
| static-V write: dot → multiply+reduce | 486.30→491.98 ms | — | keep dot |
| CombinedRead → diagonal-one | 581.70→568.63 ms | 2149.30→2064.12 ms | keep diagonal-one |

Diagonal-one removes the separate `local_o` branch, retains mixed-alpha off-diagonal entries and
sets its diagonal to one. It is algebraically equivalent to CombinedRead but avoids the `Mbar+M`
layout/copy path. The 24-layer gain is 3.96% step time despite nearly unchanged FLOPs and bytes.

AbsV source compression is an exception to the usual read/write result: replacing
`bskv,vc→bskc` dot with broadcast multiply+reduce slowed a v6e step from 708.95 to 716.26 ms and
the source-compression scope from 9.99 to 15.24 ms. Keep dot.

## Factorized and clean profiles

FactorizedLocalQK reduced its theoretical cost from `2.125 W_Q` for PerHeadLocalQK to
`0.197 W_Q`: shared key projection `0.125`, gates `0.0039`, shared contractions `0.0039`, and head
routing about `0.0645`. Its measured XPlane work was `0.199 W_Q`.

The complete Bf16Packed clean table and the live status of every optimization target are retained
in the main memo under **V2 fine-grained main profile**. Relative to the older Factorized graph,
AbsV8 reduced fetch theory `2.000→0.508 W_Q` and fetched read `1.063→0.664 W_Q`; R256-GELU reduced
write `0.531→0.406 W_Q`. Packed projections did not fix the low-utilization LocalQK contraction.

## C256 evolution

### Chunk size and rejected three-input fusion

On the original six-layer v6e graph, BAM C128/C256/C512 measured 608.79/591.78/596.40 ms; C256 won.
Combining mix and fetch into a three-input einsum was 20.4% slower on v5p-16 and 28.8% slower by
v6e log speed, so the two-stage formulation remains.

### Full-BAM implementation ladder

Same UC1a v6e-1, six layers:

| Path | Runtime commit | XPlane | Throughput vs legacy |
|---|---:|---:|---:|
| legacy | `36ebca4` | 592.26 ms | — |
| remove chunk-local remat | `821dc8d` | 536.81 ms | +10.33% |
| concatenate `Mbar`, read once | `821dc8d` | 521.43 ms | +13.58% |
| diagonal template/select | `821dc8d` | 497.61 ms | +19.02% |
| template mask + concat outputs | `821dc8d` | **494.57 ms** | **+19.75%** |

Legacy→optimized reduced BAM total scope `170.13→112.47 ms`, mix alpha `89.25→44.27 ms`,
diagonal update `33.41→0.03 ms`, fetched read `37.26→25.23 ms`, copy `30.93→4.29 ms`, and the
compiler temporary-buffer estimate `14.08→8.68 GB`. Packed-segment forward equivalence was exact;
loss/gradient differences stayed below `9.41e-4` relative and arise from bf16 lowering changes.

### BAM-MHA control and backend fairness

`bam_mha_control=True` keeps BAM's QKV/RoPE/QK-softmax-AV/output implementation but creates no BAM
parameters, M state or `bam/*` XPlane scopes. The original control was 12.1% slower than generic
QChunk. A 2×2 diagnostic localized almost all of the gap to redundant chunk-local remat interacting
with the packed segment mask:

| Segment mask | inner remat | v6e-1 step |
|---|---:|---:|
| packed/batched | on | 413.99 ms |
| packed/batched | off | **371.51 ms** |
| shared causal | on | 371.59 ms |
| shared causal | off | 371.51 ms |

Shared causal masking is not valid for packed Pile data. The production fix keeps the batched
segment mask and removes the nested remat. Switching the remaining 4D contraction to QChunk's
singleton-GQA layout measured 368.69 ms, only 0.04% from generic QChunk, but was not required for
the fair BAM control.

Dense-MHA backend choice strongly affects apparent relative overhead. On six-layer v6e, default
Pallas dense, explicit dense dot and C256 BAM-MHA measured 373.26/481.56/369.15 ms. C256 is 30.5%
faster than explicit dot but only 1.1% faster than Pallas; cross-TPU BAM/MHA ratios must therefore
use the matched control, not a generic MHA result.

## Scan history

Layer scan reduces compile latency but has schedule-dependent runtime cost. The canonical v5p-16
and corrected v6e U/U/S/U matrices are maintained in the main memo.

Streaming query scan is rejected for throughput: source-block remat is required because the
unrematerialized compiler estimate is 40.91 GiB on a 31.25-GiB v6e, and the rematerialized path is
2.7–3.4× slower on v6e and 2.5–2.9× slower on full-24 v5p. Its numerical path was healthy: 16-step
loss differences were below `9e-5` for BAM and `8e-5` for BAM-MHA, parameter counts matched, and a
two-layer scanned checkpoint restored successfully.

The old `cc61013` v6e ratio table mixed G/6 optimized with LGLL/8 legacy inheritance; its
`62.2→75.0%` retention jump is invalid and is intentionally absent from current summaries.
