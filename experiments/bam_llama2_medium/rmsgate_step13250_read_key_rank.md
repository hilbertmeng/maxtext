# RmsGateOnly step 13250 read-key rank by transform stage

Checkpoint: `gs://newproject-1-llm_base_models_us-central1/log/BamLlama2MediumRmsGateOnly/checkpoints/13250/items`

## Method

- Randomized Pile eval cohort: 32 sequences, seed 9876, shuffle buffer 32768; 65,492 valid tokens.
- Sampled every 32nd position. For every layer, token and head, stacked `full[fetch=0:2]`
  and `local_o` keys into separate `3x32` row-key and column-key matrices at three stages:
  raw projection, `_rms(raw, 1e-4)` before gating, and `2*sigmoid(gate)*rms(raw)`.
  Row and column inhabit different spaces and are independently normalized/gated.
- Metric: `sigma_1^2 / sum_i sigma_i^2`. Layer 0 keys are exactly zero because its incoming
  matrix stream is zero; it is excluded below.
- Cohort fingerprint: `eb7ba595e3e30340e4b1686a276bb3356e1810060bf89a5812850dfe20ad4edc`.
  Individual sequence hashes are in the JSON artifact.

## Results

Values are medians over all sampled sequence/token/head matrices.

| Layer | Row raw | Row RMS | Row gated | Col raw | Col RMS | Col gated |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | .587 | .582 | .690 | .644 | .483 | .655 |
| 2 | .531 | .482 | .658 | .557 | .468 | .775 |
| 3 | .544 | .495 | .662 | .550 | .508 | .702 |
| 4 | .513 | .466 | .705 | .634 | .529 | .749 |
| 5 | .521 | .494 | .701 | .554 | .506 | .678 |
| 6 | .523 | .499 | .680 | .553 | .523 | .654 |
| 7 | .527 | .499 | .689 | .543 | .516 | .675 |
| 8 | .674 | .588 | .815 | .565 | .485 | .641 |
| 9 | .511 | .467 | .663 | .513 | .467 | .661 |
| 10 | .538 | .495 | .750 | .533 | .485 | .686 |
| 11 | .500 | .464 | .686 | .502 | .462 | .626 |
| 12 | .731 | .567 | .897 | .571 | .483 | .720 |
| 13 | .571 | .510 | .719 | .487 | .446 | .631 |
| 14 | .531 | .488 | .680 | .490 | .447 | .608 |
| 15 | .698 | .605 | .832 | .517 | .485 | .659 |
| 16 | .543 | .502 | .694 | .502 | .464 | .650 |
| 17 | .859 | .636 | .954 | .525 | .460 | .690 |
| 18 | .502 | .462 | .647 | .492 | .454 | .626 |
| 19 | .493 | .452 | .641 | .489 | .444 | .631 |
| 20 | .503 | .467 | .638 | .504 | .461 | .655 |
| 21 | .519 | .484 | .635 | .506 | .467 | .632 |
| 22 | .498 | .461 | .657 | .527 | .473 | .688 |
| 23 | .513 | .463 | .677 | .540 | .488 | .675 |

Across layers 1--23, median-layer top-1 energy evolves as follows:

| Side | Raw | RMS, pre-gate | Post-gate |
|---|---:|---:|---:|
| Row | .527 | .494 | .686 |
| Column | .527 | .473 | .659 |

For reference, three independent RMS-normalized Gaussian vectors in 32 dimensions give about
`.423` median top-1 energy. Thus the normalized directions are correlated above chance, but only
moderately. Raw magnitude differences add a little concentration; learned gate imbalance adds
the largest increase. Row exceeds `.8` post-gate in layers 8/12/15/17, while no column layer does.

## Conclusion

The content directions themselves are not close to rank 1: the typical post-RMS/pre-gate
rank-1 relative errors are `.711` row and `.726` column. Most apparent post-gate low-rankness
comes from routing-amplitude specialization, not near-collinearity of the three content keys.
This supports explicitly separating content and route, but not forcing one content direction:
a side-independent, rank-configurable content factorization is safer than global rank 1.

## Head-axis comparison

For each fixed source (`fetch_0`, `fetch_1`, or `local_o`), the 16 heads form a `16x32`
matrix. `joint` concatenates the three 32-dimensional source keys within each head, forming
`16x96`; row and column remain separate. Values below are median-layer top-1 energy over
layers 1--23 on the identical cohort.

| Source | Row raw | Row RMS | Row gated | Col raw | Col RMS | Col gated |
|---|---:|---:|---:|---:|---:|---:|
| `fetch_0` | .192 | .175 | .265 | .211 | .180 | .277 |
| `fetch_1` | .187 | .172 | .262 | .218 | .187 | .276 |
| `local_o` | .206 | .189 | .255 | .217 | .187 | .251 |
| joint | .138 | .125 | .197 | .178 | .140 | .218 |

Independent row-normalized Gaussian baselines are `.152` for `16x32` and `.111` for
block-normalized `16x96`. Thus head content directions are only mildly more correlated than
chance: before gates, the first component explains `.172--.189` for individual sources and
`.125/.140` jointly. Gates again create most of the concentration (`.251--.277` per source,
`.197/.218` jointly), indicating head-amplitude specialization rather than a shared rank-1
content direction.

The head axis therefore should not be collapsed to one factor. Its normalized rank-1 error is
about `.90`, even worse than the three-source axis's `.71--.73`; any head factorization should
retain configurable rank and be justified by an ablation. The source axis has higher absolute
top-1 energy partly because it has only three rows (`3x32`, random baseline `.423`). After
dimension-specific chance correction, `(observed-baseline)/(1-baseline)`, RMS content excess is
`.123/.087` on the source row/column axes, versus only `.024--.044` for individual head axes and
`.016/.033` jointly. The head-direction rank-1 signal is therefore genuinely weaker.

## Artifacts

- Three-stage report:
  `/home/xd/bam_diagnostics/rmsgate_step13250_key_rank_stages_random32/bam_diagnostics.json`
- Head-axis report (also contains the source-axis statistics):
  `/home/xd/bam_diagnostics/rmsgate_step13250_head_axis_rank_random32/bam_diagnostics.json`
- TPU: spot `xd-v6e-1-bamdiag`; diagnostic wall time 103.5s (setup 39.9s, data 14.7s,
  compile+forward 41.8s, host statistics 6.6s).
- Head-axis rerun wall time 63.5s (setup 14.9s, data 14.5s, compile+forward 6.4s,
  host statistics 26.7s); sequence hashes and eval loss exactly match the first run.
- Code base: `c21de96`; overlay SHA256
  `30f83606b1a72ba9b87a05552b0472f6b11e12abde9eb977051e429e55e6a9df`.
