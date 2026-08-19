# BAM V2 readout attribution

## Setup

- Model: `BamLlama2MediumV2`, checkpoint step 13,250 (trainer commit `1afd942`).
- Cohort: the same 128 shuffled Pile-eval sequences as the delta-rule diagnostic;
  P1 uses every token and P2 samples 16 query positions per sequence.
- Capture/analyzer commit: `65fc768` on `codex/readout-attribution`.
- Scope: P1 and P2 only; P3 was not run, and the triggered P4 remains a proposed
  follow-up rather than part of this result.
- Sign: positive attribution is loss-harmful at the margin; negative is helpful.
  P1 uses an exact per-record scale gradient, algebraically equivalent to the
  Frobenius product between the write and its full downstream cotangent.

## Reproduction code

The executed diagnostic code is preserved on
`codex/readout-attribution@65fc768`:

- Driver, capture harness, analyzer, and artifact writer:
  `experiments/bam_llama2_medium/readout_attribution.py`
- Guarded model-side capture hooks: `MaxText/layers/attentions.py`
- Shared diagnostic extraction helpers: `MaxText/bam_diagnostics.py`

For an exact copy of the main script, run
`git show 65fc768:experiments/bam_llama2_medium/readout_attribution.py`.

## P1: loss-grounded write attribution

P1 covers 100,663,296 layer/token/head records (including the structurally dead
last-layer writes).

| Metric | Result |
|---|---:|
| harmful attribution mass | 49.992% |
| net / absolute attribution mass | -0.0159% |
| sampled attr mean / std | 5.90e-5 / 0.06624 |
| sampled attr p10 / p50 / p90 / p99 | -0.01659 / 0 / 0.01683 / 0.11022 |
| attr / gate, per-layer mean range | -2.35e-4 to 4.55e-4 |
| attr / gate std, layer 0 / layer 22 | 0.328 / 0.0198 |
| gate-weighted mean attr | -1.48e-5 |
| gate vs helpfulness, Pearson / Spearman | 0.00023 / 0.00046 |
| per-head harmful-mass range | 49.903–50.098% |
| per-layer harmful-mass range, layers 0–22 | 49.80–50.13% |

The positive and negative first-order masses cancel almost exactly. This is not
localized to a bad head or a depth band, and the write gate has effectively zero
alignment with marginal record value. Layer net effects change sign without a
monotone depth pattern; layer 23 is exactly zero because its write is never read.

## P2: structural composition at read sites

All values below are means over nonempty read layers; top shares are projections
onto each arm's own reconstructed readout.

| Metric | Fetch | Permuted-alpha null | Local-Q | Local-K |
|---|---:|---:|---:|---:|
| top-1 absolute share | 18.77% | 18.31% | 19.44% | 26.82% |
| top-8 absolute share | 64.34% | 63.63% | 66.20% | 72.28% |
| coherence | 2.463 | 2.406 | 2.693 | 2.871 |
| harmful-attribution absolute share | 50.05% | 50.04% | 49.94% | 50.23% |
| reconstruction relative norm | 0.60% | — | 0.41% | 0.37% |

Permuting alpha barely changes concentration, coherence, or harmful mass. The
heavy top-k concentration is therefore a structural property of these rank-1
contributions, not evidence that learned alpha uniquely selects a few records.

The fetch decomposition retains 99.944% of `|alpha|` mass on average (median
100%; p90 100%) with top-1,536 sources. Its reconstruction error is 0.60% mean
(0.55% median, 0.87% p90); local-Q/K reconstruction errors are 0.41%/0.37% mean.
These residuals are the combined support-truncation and bf16 accumulation floor.

Fetch source depth is strongly recency-weighted: gaps <=1/2/4/8/16 account for
16.8/31.4/51.1/76.5/95.9% of absolute contribution mass. Concentration falls as
the matrix accumulates more records:

| Use layers | Fetch top-1 / top-8 | Local-Q top-1 / top-8 | Local-K top-1 / top-8 |
|---|---:|---:|---:|
| 1–8 | 27.3% / 84.3% | 29.1% / 82.5% | 37.4% / 86.1% |
| 9–16 | 15.5% / 54.2% | 17.0% / 64.4% | 28.0% / 72.8% |
| 17–23 | 12.7% / 53.1% | 11.3% / 49.6% | 13.4% / 55.9% |

For fetched reads, the diagonal/self records supply 76.4% of signed output
alignment and cross-token records supply 23.6%. Thus the high top-k shares are
not evidence that cross-token fetch alone is a pure pointer.

## Interpretation

- H1's proposed signature is rejected: harmful mass is not low and the learned
  gates do not identify helpful records.
- The result supports H2's sign-mixed premise but not a simple targeted-decay
  mechanism: every head and depth contains an almost exact 50/50 mixture of
  helpful and harmful marginal mass.
- H3 holds only in the geometric sense that a few records dominate each readout,
  especially local-K and early layers. The same concentration under permuted alpha
  means this does not establish learned retrieval semantics; concentration also
  becomes more distributed with depth.
- Coherence is greater than one, not much less than one, so the read vectors add
  constructively in representation space. Geometric superposition can therefore
  be coherent while its loss attribution remains sign-mixed.
- A uniform or depth/coordinate-only forget rule has no measured selector for the
  harmful records. This is consistent with the negative Fixed/Learned LambdaBands
  training results; any useful suppression mechanism needs content/query-dependent
  selectivity rather than a lifetime band alone.

Per the preregistered decision rule, the approximately 50% harmful mass triggers a
small P4 finite ablation before treating the first-order signs as causal removal
effects. The most direct check is to suppress the oracle top-harmful record set on
the same batches and compare it with magnitude-matched random/helpful controls.

## Caveats

- Attribution is first-order scaling, not finite removal; record interactions are
  not assigned separately.
- The bf16 production path makes tiny numerical finite differences quantized. A
  high-signal top-256 direction has the correct sign but 17.7% relative slope error;
  the reported P1 values themselves are exact autodiff gradients of that bf16 graph.
- P2 uses 16 query positions per sequence and the stated top-1,536 fetch support.

Artifacts:

- Local: `/data0/xd/bam_diagnostics/v2_readout_attribution_13250_65fc768/`
- GCS: `gs://newproject-1-llm_base_models_us-central1/diagnostics/bam/v2_readout_attribution_13250_65fc768/`
