# V1 codebook-axis diagnostic (2026-08-09)

Checkpoint: `BamLlama2MediumV1/13250`. Four shuffled Pile-eval batches × 32
sequences; the first 64 fit PCA/LS maps and the other 64 report paired loss. All
runs used identical sequence hashes. Held-out baseline loss: `2.362491`.

## Definitions

`K₀,V₀` below denote the two fixed axes of `M`. For either bilateral read, **local
K** means the axis contracted with its read key and **local V** means its output
axis.

- **Local-K / current codebook:** store `ρ_K M:[C,V₀]` and
  `ρ_V Mᵀ:[C,K₀]`; both read keys are restricted to layer-shared C-dimensional
  subspaces.
- **Local-V / proposed:** store `M E_V:[K₀,C]` and
  `Mᵀ E_K:[V₀,C]`; both full read keys are retained, then each latent output is
  decoded by a per-head `D_h:C→local-V` map.
- **Fixed-K₀:** store only `E_KᵀM:[C,V₀]`; it supports both bilateral reads.
- **Fixed-V₀:** store only `M E_V:[K₀,C]`; it supports both bilateral reads.

Local-K and Local-V both cost `C(K₀+V₀)` and must use the same C. Each fixed-axis
form costs half as much at the same C.

## Rank evidence

Median captured energy over layers 1–23:

| compressed quantity | C=4 | C=8 | C=16 |
|---|---:|---:|---:|
| Local-K: runtime row key | .265 | .424 | .649 |
| Local-K: runtime column key | .436 | .648 | .839 |
| Local-V: `y_u` output | .169 | .316 | .578 |
| Local-V: `y_v` output | .350 | .521 | .742 |

The shared-ρ penalty is real: at C=4, per-head optimal key bases increase row-key
energy from `.265→.388` and column-key energy from `.436→.758`. But V1's local-V
outputs are intrinsically less concentrated; per-head output bases add only about
0–3 percentage points. Thus removing head-shared read-key bases does not make the
frozen outputs easier to compress.

## Frozen all-layer substitution

Same C; Local-K and Local-V have equal cache, while fixed-axis forms use half:

| C | Local-K | Local-V | Fixed-K₀ | Fixed-V₀ |
|---:|---:|---:|---:|---:|
| 2 | +4.782 | +4.974 | +4.958 | +3.808 |
| 4 | +3.101 | +3.658 | +3.970 | +2.275 |
| 8 | +0.959 | +1.686 | +1.877 | +0.943 |
| 12 | +0.320 | +0.804 | +0.844 | +0.462 |
| 16 | +0.141 | +0.404 | +0.403 | +0.241 |

Single-layer substitutions remove most cascading distribution shift. Summed local
deltas at C=4 are `.505/.716/.781/.621` in the same column order. Local-V is worse
than Local-K in 19/23 layers; its four wins are layers 3, 5, 6, and 7.

Equal-cache comparisons give each fixed-axis form twice the C:

| cache width | Local-K | Local-V | Fixed-K₀ | Fixed-V₀ |
|---:|---:|---:|---:|---:|
| 256 | C4 +3.101 | C4 +3.658 | C8 +1.877 | C8 +0.943 |
| 512 | C8 +0.959 | C8 +1.686 | C16 +0.403 | C16 +0.241 |

At these two matched budgets, Fixed-V₀ also has the smallest summed single-layer
delta (`.383` and `.143`) and wins 18/23 layers.

## Interpretation

- The proposed Local-V architecture is the exact symmetric test of the shared-ρ
  hypothesis. On frozen V1 it is consistently worse than current Local-K: preserving
  full read keys does not compensate for bottlenecking both read outputs.
- Fixed-V₀ is the strongest zero-shot cache tradeoff. It constrains only one original
  matrix axis, uses half the cache at equal C, and can spend the saved cache on twice C.
- These absolute all-layer penalties are not predictions of retrained loss. A trained
  compressed model can reorganize its keys, outputs, and state; use the probe only to
  rank initialization compatibility and choose ablations.

For a direct retraining test, compare Local-V C4 against current Codebook C4. For the
best diagnostic-supported cache design at the same budget, compare Fixed-V₀ C8.

Artifacts: `/data0/xd/bam_diagnostics/v1_codebook_axis/` contains the global,
layerwise, rank, and head-sharing JSON reports.
