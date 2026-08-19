# BAM fetched-read head rank

## Setup

- BAM: `BamLlama2MediumV2`, checkpoint step 13,250 (trainer commit `1afd942`).
- MHA: `Llama2Medium`, checkpoint step 13,500.
- Cohort: the same 128 fixed-seed shuffled Pile-eval sequences for both models;
  32 evenly spaced query positions per sequence, or 4,096 sites per layer.
- Fixed-basis validation: fit on the first 64 sequences and evaluate on the held-out 64.
- Capture commit: `c464f92`; analyzer/loss-ablation commits: `b586381`/`d0863e6`,
  branch `codex/bam-head-rank`.

Reproduction:

- Capture: `experiments/bam_llama2_medium/head_rank_diagnostics.py`
- Capture launcher: `experiments/bam_llama2_medium/run_head_rank_diagnostics.sh`
- Analyzer: `experiments/bam_llama2_medium/analyze_head_rank_diagnostics.py`
- Held-out loss test: `experiments/bam_llama2_medium/head_rank_ablation.py`
- Loss-test launcher: `experiments/bam_llama2_medium/run_head_rank_ablation.sh`
- Temporary guarded model capture: `MaxText/layers/attentions.py` at the capture commit.

Artifacts:

- BAM local/GCS: `/data0/xd/bam_diagnostics/v2_head_rank_bam_13250_c464f92/` and
  `gs://newproject-1-llm_base_models_us-central1/diagnostics/bam/v2_head_rank_bam_13250_c464f92/`
- MHA local/GCS: `/data0/xd/bam_diagnostics/v2_head_rank_mha_13500_c464f92/` and
  `gs://newproject-1-llm_base_models_us-central1/diagnostics/bam/v2_head_rank_mha_13500_c464f92/`
- Analysis: `/data0/xd/bam_diagnostics/v2_head_rank_analysis_c464f92/head_rank_analysis.json`
- Loss ablation: `/data0/xd/bam_diagnostics/v2_head_rank_ablation_d0863e6/head_rank_ablation.json`

## Spaces and metrics

The space choice is part of the experiment, not a presentation detail:

- BAM row path: shared `r_row:[n,32]` and native `y_V:[n,8]` coordinates.
- BAM column path: shared `r_col:[n,8]` and native `y_U:[n,32]` coordinates.
- BAM/MHA comparison: each head is mapped through its own output block,
  `c_h = y_h W_O[h] in R^1024`. No MHA native-head cosine is reported because its
  head coordinates are private.

At every layer, `E8` is the top-eight singular-value energy fraction and `r95` is
the number of head modes needed for 95% energy. Two ranks answer different questions:

- local rank: SVD of one token's `16 x d` head matrix;
- global rank: SVD after concatenating all sampled sequence/token/feature axes. This
  tests whether one fixed head basis generalizes across inputs.

Head-mean-centered spectra give the same conclusions; the observed concentration is
not an artifact of one shared mean component.

## Main result

The conjectured native-space similarity is real **per token**, but it does not imply
that a fixed subset/basis of fetched-read heads is redundant. Row heads are less, not
more, globally compressible than column heads; both become functionally diverse after
their private `W_O` blocks.

### Depth trend

Values are averages over the indicated layers; layer 0 is omitted because the incoming
matrix stream is zero.

| Space | Path | Layers 1-8 E8 / r95 | Layers 9-16 E8 / r95 | Layers 17-23 E8 / r95 |
|---|---|---:|---:|---:|
| native key, post-gate | row | 0.658 / 14.88 | 0.626 / 14.88 | 0.574 / 15.00 |
| native key, post-gate | column | 0.859 / 11.62 | 0.851 / 11.88 | 0.835 / 13.00 |
| native readout | row | 0.733 / 14.62 | 0.684 / 14.75 | 0.644 / 15.00 |
| native readout | column | 0.874 / 11.25 | 0.829 / 12.00 | 0.787 / 13.57 |
| residual contribution | BAM row | 0.676 / 14.75 | 0.626 / 15.00 | 0.613 / 15.14 |
| residual contribution | BAM column | 0.819 / 12.38 | 0.718 / 13.75 | 0.685 / 14.57 |
| residual contribution | BAM fetch total | 0.713 / 14.50 | 0.667 / 14.50 | 0.657 / 14.86 |
| residual contribution | pure MHA | 0.697 / 14.25 | 0.780 / 13.38 | 0.806 / 13.14 |

The depth directions are opposite: BAM row redundancy decreases steadily, while pure
MHA becomes more concentrated. Beyond the first third, the complete BAM fetched read is
less redundant than MHA in the fair residual space.

The post-RMS to post-gate change is modest. Across layers 1-23, row-key E8 rises
`0.584 -> 0.621`; column-key E8 rises `0.817 -> 0.849`. Gates add some concentration,
especially on the column side, but do not create a low-rank row path.

### Local versus fixed rank

| Quantity | Layers 1-8 local median r95 | Layers 9-16 | Layers 17-23 |
|---|---:|---:|---:|
| row key | 11.12 | 11.62 | 12.00 |
| column key | 5.88 | 6.38 | 6.14 |
| row native readout | 3.62 | 4.12 | 4.14 |
| column native readout | 3.88 | 4.88 | 4.14 |
| row residual contribution | 11.25 | 11.38 | 12.00 |
| column residual contribution | 10.50 | 11.62 | 11.57 |
| BAM fetch total contribution | 12.50 | 12.75 | 12.57 |
| pure-MHA contribution | 9.75 | 10.00 | 9.71 |

The row native readout is locally low-rank partly because it has only eight coordinates.
Its basis changes across tokens: global row-read `r95` remains 14-15. This structural
ceiling therefore does not justify a fixed four/eight-head implementation.

Private `W_O` blocks also matter. Native BAM readouts have mean absolute inter-head
cosines around `0.4-0.6`; after per-head `W_O`, the corresponding residual contributions
are almost orthogonal (`~0.03-0.08`). Apparent native similarity is not functional
duplication unless the implementation reconstructs every original head before `W_O`.

## Held-out fixed-basis test

The basis is fitted from post-gate keys on 64 sequences and evaluated on the other 64.
Entries are retained residual-contribution energy.

| Basis -> target | Depth | rank 4 | rank 8 | rank 12 |
|---|---|---:|---:|---:|
| row key -> row residual | L1-8 | 0.374 | 0.620 | 0.820 |
| row key -> row residual | L9-16 | 0.324 | 0.575 | 0.795 |
| row key -> row residual | L17-23 | 0.318 | 0.570 | 0.793 |
| column key -> column residual | L1-8 | 0.463 | 0.749 | 0.923 |
| column key -> column residual | L9-16 | 0.331 | 0.621 | 0.864 |
| column key -> column residual | L17-23 | 0.333 | 0.595 | 0.819 |

Even an oracle basis fitted directly to residual contributions does not make BAM more
compressible than MHA:

| Self-fitted residual basis | Depth | rank 4 | rank 8 | rank 12 |
|---|---|---:|---:|---:|
| BAM fetch | L1-8 | 0.450 | 0.709 | 0.880 |
| BAM fetch | L9-16 | 0.388 | 0.664 | 0.874 |
| BAM fetch | L17-23 | 0.392 | 0.645 | 0.846 |
| pure MHA | L1-8 | 0.455 | 0.694 | 0.875 |
| pure MHA | L9-16 | 0.537 | 0.778 | 0.923 |
| pure MHA | L17-23 | 0.544 | 0.806 | 0.936 |

The held-out retained energies nearly equal the all-cohort spectra, so the failure is not
PCA overfitting to the first 64 sequences.

## Same-cohort loss test

The fitted key bases were frozen, applied to every token of the held-out 64 sequences,
and inserted after the runtime RMS/gate. The baseline loss is `2.313345`.

| Compressed side | rank 2 dloss | rank 4 | rank 8 | rank 12 |
|---|---:|---:|---:|---:|
| row | +0.13781 | +0.08388 | +0.03561 | +0.01289 |
| column | +1.27292 | +0.52592 | +0.05619 | +0.01522 |
| both | +2.05984 | +0.80290 | +0.11364 | +0.03034 |

Column keys are geometrically more compressible, but column compression hurts loss more
because this path is functionally stronger. This agrees with the independent attribution
and training ablations that found about `4.1x` more absolute importance on the column path.

## Interpretation

- A static reduction from 16 to 8 fetched-read heads is rejected for both sides. Its
  same-cohort loss penalties are much larger than normal diagnostic noise.
- The original expectation that row heads are especially redundant is reversed. Row-key
  global rank is almost full and becomes fuller with depth.
- A dynamic, token-dependent factorization remains geometrically possible on the column
  side: `r_col` is a `16 x 8` matrix and thus has exact local rank at most eight; observed
  local `r95` is about six. The corresponding subspace rotates across tokens, which is why
  one fixed basis fails.
- Row keys are `16 x 32` and need about 11-12 local modes for 95% energy, leaving little
  head-count reduction. The low local rank of the eight-dimensional row *answer* cannot be
  selected before reading M and is therefore not by itself an efficient routing rule.
- Any follow-up should test a dynamic column-key factorization that reconstructs all 16
  per-head answers before their private `W_O` blocks. Its factorization/mixing cost must be
  counted; low rank alone does not guarantee a speed win.

## Layerwise appendix

Each entry is `E8 (r95)`.

| L | row key | col key | row read | col read |
|---:|---:|---:|---:|---:|
| 1 | 0.713 (14) | 0.882 (11) | 0.798 (14) | 0.912 (10) |
| 2 | 0.623 (15) | 0.758 (14) | 0.670 (15) | 0.777 (14) |
| 3 | 0.630 (15) | 0.831 (13) | 0.669 (15) | 0.859 (12) |
| 4 | 0.616 (15) | 0.841 (12) | 0.670 (15) | 0.877 (11) |
| 5 | 0.666 (15) | 0.904 (11) | 0.750 (15) | 0.933 (10) |
| 6 | 0.696 (15) | 0.935 (9) | 0.821 (14) | 0.938 (9) |
| 7 | 0.607 (15) | 0.825 (12) | 0.668 (15) | 0.831 (12) |
| 8 | 0.710 (15) | 0.893 (11) | 0.819 (14) | 0.867 (12) |
| 9 | 0.752 (14) | 0.845 (12) | 0.815 (14) | 0.849 (12) |
| 10 | 0.583 (15) | 0.839 (12) | 0.649 (15) | 0.835 (12) |
| 11 | 0.569 (15) | 0.783 (14) | 0.623 (15) | 0.783 (13) |
| 12 | 0.659 (15) | 0.869 (12) | 0.765 (14) | 0.828 (12) |
| 13 | 0.594 (15) | 0.826 (12) | 0.651 (15) | 0.796 (12) |
| 14 | 0.630 (15) | 0.851 (12) | 0.718 (15) | 0.829 (12) |
| 15 | 0.649 (15) | 0.947 (9) | 0.660 (15) | 0.929 (10) |
| 16 | 0.572 (15) | 0.851 (12) | 0.588 (15) | 0.780 (13) |
| 17 | 0.596 (15) | 0.833 (13) | 0.653 (15) | 0.762 (14) |
| 18 | 0.551 (15) | 0.779 (14) | 0.599 (15) | 0.729 (14) |
| 19 | 0.580 (15) | 0.850 (13) | 0.599 (15) | 0.769 (14) |
| 20 | 0.566 (15) | 0.840 (13) | 0.694 (15) | 0.778 (14) |
| 21 | 0.577 (15) | 0.783 (14) | 0.672 (15) | 0.734 (14) |
| 22 | 0.564 (15) | 0.884 (11) | 0.630 (15) | 0.887 (12) |
| 23 | 0.584 (15) | 0.876 (13) | 0.657 (15) | 0.847 (13) |

| L | BAM row residual | BAM col residual | BAM fetch total | BAM internal MHA | pure MHA |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.742 (14) | 0.869 (11) | 0.732 (14) | 0.801 (14) | 0.746 (14) |
| 2 | 0.605 (15) | 0.756 (14) | 0.629 (15) | 0.739 (14) | 0.801 (13) |
| 3 | 0.584 (15) | 0.819 (13) | 0.732 (15) | 0.787 (12) | 0.724 (14) |
| 4 | 0.600 (15) | 0.848 (12) | 0.728 (15) | 0.683 (15) | 0.635 (15) |
| 5 | 0.691 (15) | 0.878 (11) | 0.710 (14) | 0.656 (15) | 0.705 (14) |
| 6 | 0.722 (15) | 0.856 (11) | 0.719 (15) | 0.673 (15) | 0.598 (15) |
| 7 | 0.641 (15) | 0.724 (14) | 0.670 (14) | 0.725 (14) | 0.615 (15) |
| 8 | 0.822 (14) | 0.803 (13) | 0.787 (14) | 0.819 (13) | 0.755 (14) |
| 9 | 0.748 (14) | 0.769 (13) | 0.720 (14) | 0.696 (14) | 0.736 (14) |
| 10 | 0.601 (15) | 0.742 (14) | 0.674 (15) | 0.759 (14) | 0.824 (14) |
| 11 | 0.581 (15) | 0.669 (14) | 0.626 (15) | 0.700 (14) | 0.794 (14) |
| 12 | 0.702 (15) | 0.662 (14) | 0.631 (15) | 0.746 (14) | 0.761 (14) |
| 13 | 0.598 (15) | 0.716 (13) | 0.653 (14) | 0.743 (14) | 0.737 (13) |
| 14 | 0.657 (15) | 0.739 (14) | 0.650 (15) | 0.769 (13) | 0.892 (11) |
| 15 | 0.579 (15) | 0.735 (14) | 0.710 (14) | 0.671 (14) | 0.759 (13) |
| 16 | 0.546 (16) | 0.712 (14) | 0.670 (14) | 0.834 (12) | 0.736 (14) |
| 17 | 0.631 (15) | 0.635 (15) | 0.608 (15) | 0.825 (13) | 0.846 (13) |
| 18 | 0.571 (15) | 0.614 (15) | 0.591 (15) | 0.708 (15) | 0.777 (13) |
| 19 | 0.567 (16) | 0.662 (15) | 0.644 (15) | 0.823 (13) | 0.824 (12) |
| 20 | 0.666 (15) | 0.671 (15) | 0.659 (15) | 0.758 (14) | 0.784 (14) |
| 21 | 0.631 (15) | 0.672 (15) | 0.610 (15) | 0.805 (13) | 0.776 (14) |
| 22 | 0.599 (15) | 0.797 (13) | 0.783 (14) | 0.836 (13) | 0.806 (13) |
| 23 | 0.623 (15) | 0.745 (14) | 0.708 (15) | 0.811 (13) | 0.827 (13) |
