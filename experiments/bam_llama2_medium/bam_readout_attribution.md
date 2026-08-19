# BAM V2 readout attribution

## Setup

- Model: `BamLlama2MediumV2`, checkpoint step 13,250 (trainer commit `1afd942`).
- Cohort: 128 fixed-seed shuffled Pile-eval sequences. P1 uses every token; P2 samples
  16 query positions per sequence.
- Capture/runtime commit: `78a2f9f`; offline analysis/report commit: `5446f3b`, both
  on `codex/readout-attribution`.
- Sign: positive attribution is loss-harmful at the margin; negative is helpful.
- Side names follow the contraction: the **column path** matches the address/V axis and
  returns data/U; the **row path** matches data/U and returns address/V.

## Reproduction

- Capture/analyzer: `experiments/bam_llama2_medium/readout_attribution.py`
- Sharded launcher: `experiments/bam_llama2_medium/run_readout_attribution.sh`
- Cross-shard side analysis: `experiments/bam_llama2_medium/aggregate_readout_attribution.py`
- Model-side guarded capture: `MaxText/layers/attentions.py`

Raw artifacts:

- Local: `/data0/xd/bam_diagnostics/v2_readout_attribution_sides_13250_78a2f9f/`
- GCS: `gs://newproject-1-llm_base_models_us-central1/diagnostics/bam/v2_readout_attribution_sides_13250_78a2f9f/`

## P1: loss-grounded attribution

P1 covers 100,663,296 layer/token/head write records. The same forward pass was
backpropagated through both paths, column-only, and row-only. The residual
`mixed = both - column - row` isolates downstream interactions requiring both paths;
the numerical decomposition closes to `7.25e-10` of total absolute attribution.

| Component | absolute mass / both | harmful mass | net / absolute | gate vs helpfulness (Pearson / Spearman) |
|---|---:|---:|---:|---:|
| both | 100.00% | 49.992% | -0.0160% | 0.00021 / 0.00045 |
| column (address -> data) | 85.04% | 49.981% | -0.0375% | -0.00110 / -0.00092 |
| row (data -> address) | 20.69% | 50.013% | +0.0250% | 0.00147 / 0.00246 |
| mixed interaction | 32.61% | 50.016% | +0.0327% | 0.00207 / -0.00070 |

The column path carries `4.111x` the absolute first-order mass of the row path.
Their record-level patterns are nearly independent: cosine `0.0396`, sampled
Pearson `0.0529`, and Spearman `0.0418`. Of the magnitude that the two paths overlap,
`46.5%` has opposite signs. The sizable mixed term means the two path attributions
cannot be treated as additive standalone modules.

The approximately 50/50 helpful/harmful split and zero gate alignment hold for both
sides separately. Column is more important because it is stronger, not because its
records are cleaner or its write gate is a better value selector.

## P2: structural readout composition

The energy split below uses each side's actual post-gate runtime keys and readout.
Fetch column output has 32 coordinates and row output has 8; Local-Q/K have 32 on
both sides, so the fetch per-coordinate ratio is the fairer scale comparison.

| Read site | column energy fraction | column/row energy | per-coordinate ratio | column/row key RMS | column/row effective strength |
|---|---:|---:|---:|---:|---:|
| fetch | 86.52% | 6.419x | 1.605x | 2.069x | — |
| Local-Q | 92.89% | 13.059x | 13.059x | 3.181x | 3.210x |
| Local-K | 95.02% | 19.083x | 19.083x | 2.955x | 2.777x |

For Local-Q/K, the column/row absolute head-mix ratios are only `0.889x/0.864x`.
Thus their strong column dominance does not come from larger head-mix coefficients;
it is already present in the post-gate key/effective-read strength.

| Read site | side | top-1 abs share | top-8 abs share | coherence | harmful-attribution share |
|---|---|---:|---:|---:|---:|
| fetch | column | 20.79% | 69.26% | 2.354 | 49.90% |
| fetch | row | 21.65% | 76.80% | 2.474 | 50.00% |
| Local-Q | column | 21.53% | 73.09% | 2.858 | 49.70% |
| Local-Q | row | 18.95% | 62.48% | 2.256 | 49.95% |
| Local-K | column | 32.22% | 81.51% | 3.133 | 49.93% |
| Local-K | row | 18.99% | 62.20% | 2.274 | 49.98% |

Column dominance generally grows after the earliest layers:

| Read site | layers 1-8 | layers 9-16 | layers 17-23 |
|---|---:|---:|---:|
| fetch column energy fraction | 60.25% | 83.00% | 87.96% |
| Local-Q column energy fraction | 89.17% | 96.19% | 92.60% |
| Local-K column energy fraction | 93.57% | 97.24% | 92.52% |

Fetch is an instructive exception at its first usable layer: column energy is only
`5.68%`; the column path takes over as M accumulates. Permuting mixed alpha raises
the aggregate column/row energy ratio from `6.419x` to `8.212x`. Learned alpha
therefore routes relatively more energy through the weak row path, even though the
previous analysis found little change in top-k concentration or harmful mass.

Across both sides, the original aggregate conclusions remain: readouts are fairly
top-k concentrated and coherent (`coherence > 1`), while their contributing records
remain almost exactly 50/50 helpful/harmful. Geometry is constructive even when
first-order loss attribution is sign-mixed.

## Cross-check against training ablations

The independent from-scratch fetched-read ablations provide an unusually clean
validation of the side decomposition:

| Run | Removed path | loss penalty vs V2 |
|---|---|---:|
| `BamLlama2MediumV2C256FetchedRowOnly` | column | +0.0443 |
| `BamLlama2MediumV2C256FetchedColOnly` | row | +0.0108 |

The functional penalty ratio is `0.0443 / 0.0108 = 4.102x`, essentially identical
to P1's independently measured column/row absolute-attribution ratio `4.111x`.
P2 agrees directionally but is more skewed in raw energy (`6.419x`): the small row
readout is still useful per unit energy and should not simply be deleted.

## Interpretation

- The column/address-to-data path is the primary BAM read mechanism. It dominates
  loss attribution, output energy, and the cost of causal removal.
- The row/data-to-address path is weaker but nonzero, relatively more important in
  the first fetched layer, and selectively favored by learned alpha versus the null.
- The two sides do different work rather than duplicate one another: their attribution
  correlation is near zero and the mixed downstream interaction is 32.6% of total
  absolute attribution.
- Splitting the sides does not rescue the learned gates as value selectors. Both are
  individually 50/50 sign-mixed with essentially zero gate/value alignment.
- Uniform lifetime decay remains poorly targeted. A useful selector would need to be
  content/query dependent and likely side aware.

The row path's low energy but measurable causal value motivates the concurrent
`BamLlama2MediumV2C256RowBypassWO` training arm: it tests whether giving this answer
a dedicated output projection can preserve its distinct information without sharing
the standard MHA tail columns.

## Caveats and next check

- P1 is exact first-order scaling attribution of the bf16 graph, not finite removal.
- P2 reconstruction error is 0.60% for fetch and 0.41%/0.37% for Local-Q/K; fetch
  retains 99.944% of absolute alpha mass on average.
- P2 samples 16 query positions per sequence; P1 uses all valid positions.
- Because harmful mass is about 50% on both sides, the preregistered next causal check
  remains a small P4 oracle finite ablation against magnitude-matched random/helpful
  controls. It should report column and row paths separately.
