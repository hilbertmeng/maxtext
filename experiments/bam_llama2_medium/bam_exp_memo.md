# BAM Llama2Medium experiment memo

This is the authoritative, compact summary of current BAM capability and performance conclusions.
Detailed profiling history is in [`bam_profile_history.md`](bam_profile_history.md); checkpoint and
numerical diagnostics are in [`bam_checkpoint_diagnostics.md`](bam_checkpoint_diagnostics.md).
Configuration-class comments in `MaxText/exp.py` remain the source of truth for individual run
speed, stop step, loss comparison and runtime commit.

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
| replicate `P_loc_up` | persistent ~`+0.0034` trajectory gap | ~1.5% faster | reject for capability runs |
| CodebookC4 | `+0.0411` vs Factorized @2,800 | faster | reject |
| Window256 | same-batch `+0.1193` | temporal cache 8× smaller | reject |
| best Window256+OldBlock16 | same-batch `+0.0626` | temporal cache 4.27× smaller | reject |
| JAX bilateral block read | full-24 1.04% slower | more copies/traffic | reject |
| three-input mix+fetch einsum | — | 20–29% slower | reject |
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
| M-cache compression | V1 step 13,250 | `/data0/xd/bam_diagnostics/bam_cache_diagnostics_49be222_mb16_final.json` |
| signed alpha | V1 step 13,250 | `/data0/xd/bam_diagnostics/bam_alpha_*_final.json` |
| attention sink / prefix-4 | `a02fc72`, `d3c17a6` | `/data0/xd/bam_diagnostics/bam_*sink*`, `bam_window_prefix4_*` |

Design documents:

- [`bam_scan_design.md`](bam_scan_design.md)
- [`shared_qchunk_swa_design.md`](shared_qchunk_swa_design.md)
