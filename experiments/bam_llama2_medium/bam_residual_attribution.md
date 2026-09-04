# BAM residual attribution

## Reproduction

Both checkpoints were evaluated on the same 128 randomly shuffled Pile-eval
sequences (`T=2048`, seed `9876`, shuffle buffer `32768`). The reusable cohort
contains inputs, targets, positions, segment masks, and per-sequence hashes.

| Model | Config | Checkpoint | Trainer / diagnostic commit |
|---|---|---|---|
| Medium V2 | `BamLlama2MediumV2` | `BamLlama2MediumV2/checkpoints/13250/items` | `1afd942` / `c7bd696` |
| XL16 Rank2 | `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2` | `...LocalQKRank2/checkpoints/49720/items` | `aef0d97` / `2548e55` |

- Cohort SHA-256: `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`
- GCS cohort: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`
- Local cohort: `/data0/xd/bam_diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`
- Medium raw results: `/data0/xd/bam_diagnostics/bam-v2-residual-attr-pile128-final-c7bd696/`
- XL raw results: `/data0/xd/bam_diagnostics/bam-xl16-rank2-residual-attr-pile128-2548e55/`

Each raw NPZ retains every sample's `E` and `V` matrices with shape `[24,6]`.
The component order is MLP, MHA, BAM column-self, column-cross, row-self,
row-cross. Self/cross identifies the same source position versus other source
positions; it does not identify write-record provenance. Column read uses the
V/address-side key and emits a K/data-side vector; row read does the converse.

## Method

For sample `i`, layer `l`, and component `c`, the normalized energy is

`E[i,l,c] = mean_valid_tokens ||z[i,l,c]||_2 / ||h_L[i]||_2`.

Integrated gradients use ten-point Gauss-Legendre quadrature on
`h(alpha)=alpha*h_L`, with the final-RMS denominator frozen at its value for
the original `h_L`. This makes the path scale-sensitive while retaining the
learned RMS scale and logits head. Positive `V` means that the component
reduces loss. `V` is normalized per sample by that sample's total path
contribution. The dimensionless efficiency below is `mean(V) / mean(E)`, not
the mean of per-sample ratios.

MHA includes LocalQK's nonlinear influence through Q/K and attention weights.
BAM row/column terms include only fetched-M readout and are split exactly into
self and cross-position reads before projection through the layer's actual
output projection.

## Aggregate result

Values are cohort means; `±` is the 95% confidence interval across the 128
paired sequences. `E` and `V` are summed over all 24 layers before averaging.

| Component | Medium E | Medium V | V/E | XL E | XL V | V/E |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 2.0387±.0200 | .3962±.0136 | .1943 | 1.9444±.0232 | .4923±.0091 | .2532 |
| MHA | 1.0037±.0108 | .2174±.0068 | .2166 | .8041±.0172 | .1815±.0068 | .2257 |
| BAM col self | .9151±.0141 | .2587±.0125 | .2826 | .6503±.0102 | .2098±.0072 | .3225 |
| BAM col cross | .3963±.0077 | .0720±.0041 | .1816 | .3190±.0058 | .0759±.0047 | .2379 |
| BAM row self | .4753±.0125 | .0391±.0029 | .0823 | .3162±.0083 | .0355±.0018 | .1124 |
| BAM row cross | .1975±.0072 | .0165±.0005 | .0837 | .1325±.0039 | .0050±.0004 | .0380 |
| **BAM total** | **1.9842±.0342** | **.3863±.0149** | **.1947** | **1.4180±.0208** | **.3262±.0096** | **.2301** |

Same-cohort paired XL-minus-Medium changes in normalized contribution are:
MLP `+.09609±.00922`, MHA `-.03595±.00367`, and total BAM
`-.06007±.00765`. Thus XL reallocates about 9.6 percentage points of total
loss-reducing contribution toward MLP, supplied by about 3.6 points less MHA
and 6.0 points less BAM.

## Depth

| Layers | Medium BAM E | Medium BAM V | V/E | XL BAM E | XL BAM V | V/E |
|---|---:|---:|---:|---:|---:|---:|
| 0–5 | .1085 | .0025 | .0232 | .0481 | .0012 | .0257 |
| 6–11 | .3591 | .0114 | .0318 | .2259 | .0068 | .0303 |
| 12–17 | .5703 | .0868 | .1521 | .3440 | .0515 | .1497 |
| 18–23 | .9463 | .2856 | .3018 | .8001 | .2667 | .3333 |

BAM energy is present much earlier than BAM utility. The last six layers
produce 74% of Medium BAM contribution and 82% of XL BAM contribution, with
roughly ten times the contribution-per-energy efficiency of layers 6–11.

## Main findings

- Same-position BAM read dominates normalized BAM contribution: 77.1% in
  Medium and 75.2% in XL. Cross-token read is nevertheless material: .0885
  and .0809 of total normalized contribution, respectively.
- Column/data-side output dominates row/address-side output: the contribution
  ratios are 5.94× in Medium and 7.04× in XL. XL row-cross contribution nearly
  collapses (`.0050` versus Medium `.0165`), while column-cross remains stable
  (`.0759` versus `.0720`).
- XL uses less BAM energy but uses it more efficiently overall (`V/E=.2301`
  versus `.1947`). Its BAM utility is also more top-heavy. This argues for
  separating capacity from placement: simply enlarging a read path need not
  help if it adds energy in early or row-side components with low attribution.
- The strongest actionable targets are late-layer column reads and preserving
  cross-token column read. Early-layer BAM and row-cross read are candidates
  for compression or reduced allocation, but this report measures attribution,
  not a causal ablation; retraining is still required.

## Closure

Endpoint loss is reproduced exactly for both models. Medium residual closure
mean is `4.09e-9`. XL's direct parallel cross-layer sum has mean error `.00334`,
but the captured initial link, every inter-layer link, and final link each close
exactly; normalized Gram totals are `.999994` (Medium) and `.999992` (XL).
The XL residual discrepancy is therefore cross-layer floating-point summation,
not a missing component. Mean IG quadrature closure error is `.01507` on path
contribution `8.4117` for Medium and `.01573` on `8.7202` for XL (both about
0.18%).
