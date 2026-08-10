# V1 step 13250 gate diagnostics

Read-only run on `BamLlama2MediumV1/checkpoints/13250/items` at code
`845650dc4e1c25bce0f2e10bcce2a1afd4c206ac`.  The cohort is 128/128 unique
shuffled Pile-eval sequences (`seed=9876`, cohort `0b414befd8c15850`), split into
4 batches of 32.  Every eighth token was sampled: 32,749 valid positions per
layer.  Eval loss was 2.37945.  Raw report:
`/data0/xd/bam_diagnostics/bam_gate_diag_v1_13250_845650d.json`.

`fetch` below is V1's `W_R_gate`: it gates the single CombinedRead over fetched
M plus local M, not a pure fetched-M-only read.  Factorized LocalQ/K has one
row/column sigmoid gate shared by all heads; its per-head effective magnitude is
`gate * abs(RMS-normalized signed head_mix)`.

## Gate opening

| gate | init | mean | p10 | p50 | p90 | `<.01` | `>.5` |
|---|---:|---:|---:|---:|---:|---:|---:|
| write | .100 | .21794 | .05664 | .16504 | .46289 | .01% | 8.11% |
| fetch row | .005 | .00589 | .00250 | .00507 | .01013 | 89.45% | 0 |
| fetch col | .005 | .00996 | .00377 | .00848 | .01770 | 61.26% | 0 |
| local Q row | .005 | .00407 | .00068 | .00221 | .01160 | 87.97% | 0 |
| local Q col | .005 | .00948 | .00273 | .00806 | .01782 | 59.38% | 0 |
| local K row | .005 | .00368 | .00079 | .00200 | .01038 | 89.25% | 0 |
| local K col | .005 | .00784 | .00247 | .00671 | .01440 | 65.67% | 0 |

No read gate is saturated open.  Write is a broad continuous gate rather than
a binary switch: only 0.36% exceed .9 and 0.09% exceed .99.  Small read-gate
values do not by themselves prove an unimportant read: RMS-normalized key
direction, M magnitude, and the downstream readout also determine contribution.

## Depth structure

Means over layers 0-7 / 8-15 / 16-23:

| gate | early | middle | late |
|---|---:|---:|---:|
| write | .11888 | .20428 | .33066 |
| fetch row | .00611 | .00474 | .00683 |
| fetch col | .00638 | .00941 | .01409 |
| local Q row | .00814 | .00261 | .00146 |
| local Q col | .01233 | .01135 | .00475 |
| local K row | .00743 | .00223 | .00139 |
| local K col | .00995 | .00968 | .00391 |

The branches specialize by depth.  LocalQ/K is strongest early/middle and
falls sharply late, whereas column-side CombinedRead and writing strengthen
toward the top.  Fetch-column and write layer means correlate at .79; fetch-col
and local-Q/K-col correlate negatively (-.40/-.33).

All four read gates in layer 0 remain exactly at their .005 initialization.
This is expected because the incoming M is zero before the first write, so the
current layer-0 BAM reads are no-ops.  Removing them should first be verified by
same-batch equality, then is an exact speed simplification for this flow.

Largest layer means:

- write: L16 .565, L17 .423, L22 .380;
- fetch col: L16 .0221, L17 .0207;
- local-Q row/col: L1 .0169 / L4 .0186;
- local-K row/col: L1 .0142 / L8 .0153.

## Head and input selectivity

Across all layers, mean write/fetch openings are nearly uniform by global head
(CV 6.1% / 2.8-3.7%), but this average hides rotating per-layer roles.  Median
within-layer head CV is 27% for write and 20-22% for fetch; write L5 spans
.0257-.5502 across heads (21x).

Factorized LocalQ/K is much more head-selective after its signed head mix:

| effective magnitude | median within-layer max/min | median head CV |
|---|---:|---:|
| local Q row | 9.5x | .735 |
| local Q col | 33.1x | 1.147 |
| local K row | 10.4x | .724 |
| local K col | 37.6x | 1.127 |

Q and K select almost the same layer/head locations: effective Q/K correlation
is .980 on row reads and .955 on column reads.  Their keys and signed mixes can
still differ, so this supports testing shared routing, not blindly sharing the
whole Q/K read.

The gates are genuinely input-dependent.  After subtracting between-head
variance, median within-head token CV is .47 for write, .52/.39 for fetch
row/col, and .48-1.24 for effective LocalQ/K.  A static-gate replacement is
therefore not supported by these measurements.

## Implications

1. Test the exact layer-0 read removal.
2. Use same-batch checkpoint ablations to rank layer schedules: late LocalQ/K
   removal and early fetch removal are the data-supported candidates.  A fixed
   alternate-layer policy ignores the learned depth split.
3. Sharing only Q/K routing is plausible from the .95-.98 correlation, but its
   parameter/FLOP saving is small; the valuable target is eliminating selected
   M-read contractions by layer.
4. Gate magnitude is a screening signal, not causal evidence.  Before retraining,
   measure loss plus actual read-vector norm changes for each proposed layer
   ablation.
