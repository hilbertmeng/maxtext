# V1 layerwise AbsV allocation diagnostic

Checkpoint: `BamLlama2MediumV1/13250`. Runtime code: `b4deeb3` on branch
`codex/absv-layerwise-diagnostic-run-b4deeb3`; the maintained runner is `64d35f5`.
The fixed shuffled Pile-eval cohort has 128 sequences: 64 fit each layer's Fixed-V
projection/decoder, 32 select widths, and the final 32 only validate the selected schedule.
The checkpoint is read-only.

This is a frozen-checkpoint oracle: `Mbar` is projected to C dimensions and decoded back to
the original shape before the normal read. It ranks cache allocations without introducing
heterogeneous production shapes, but its absolute loss penalties do not predict a retrained
native-compression model.

## Allocation rule

For every layer and `C in {0,4,8,12,16,24,32}`, measure the paired selection-batch loss when
only that layer is compressed. A discrete rate-distortion DP then minimizes the sum of these
single-layer deltas under `sum(C_l) <= budget`. This is the cache-allocation analogue of
water-filling: capacity goes where its next increment removes the most loss. It uses no fixed
layer numbers or depth fractions and therefore applies unchanged to a different layer count.
Layer 0 is forced to C=0 because its incoming M is exactly zero.

For a genuinely new model, first train an uncompressed/wide-bottleneck checkpoint, calibrate
the schedule with this rule, verify it at another checkpoint if the model is still early, then
freeze the physical widths for the formal retrain. With no trained activations, no data-driven
method can know the layer importance in advance.

The layer-count-independent procedure is therefore:

1. Measure each layer's paired causal loss curve `D_l(C)` on a calibration checkpoint.
2. Use the budget DP above to initialize `C_l` by equalizing marginal loss reduction per cache
   dimension, not by imposing a depth trend.
3. Because layer effects interact, refine that seed at fixed total cache with whole-model
   pairwise exchanges (`C_i -= delta`, `C_j += delta`) on the selection split; stop when no
   exchange improves loss, then evaluate once on the untouched validation split.

This works for any number of layers. A non-monotonic result is expected whenever measured
marginal utilities are non-monotonic; monotonic widths are an optional constraint, not an
optimality principle. If no pilot checkpoint can be trained, uniform width is the only
assumption-free starting point and cannot be called model-specific optimum.

## Held-out result

Uniform C8 costs 192 dimensions across 24 layers. `monotonic` is the simple depth rule
`C4/C8/C12` over successive thirds. `auto` is selected only on the preceding 32 sequences.

| schedule | cache width | selection dloss | validation dloss | validation vs uniform | penalty change |
|---|---:|---:|---:|---:|---:|
| uniform C8 | 192 | +0.95611 | +0.92972 | -- | -- |
| monotonic C4/C8/C12 | 192 | +1.40171 | +1.38612 | +0.45640 | +49.1% |
| auto DP | 184 | +0.68628 | +0.68283 | -0.24689 | -26.6% |
| auto DP | 192 | +0.63652 | +0.62738 | -0.30233 | -32.5% |

At equal cache, auto C192 beats uniform C8 on all 32 validation sequences. Auto C184 uses
4.2% less cache and wins on 31/32. Selection and validation effects agree closely, so the
gain is not a selector-batch fluctuation. The monotonic schedule is substantially worse than
uniform: importance is not a smooth function of normalized depth.

The DP's additive prediction for C192 is +0.207, versus the exact combined +0.627. Cascading
cross-layer distribution shift is therefore large; single-layer curves reliably rank these
schedules but cannot be summed as an absolute loss estimate.

## Layer structure

`Mbar V` is the shared V-axis spectrum of the actually mixed historical matrix; `read-yv` is
the corresponding row-read output spectrum. Both are fit-batch measurements.

| L | Mbar V energy@8 | read-yv energy@8 | read-yv r95 | C184 | C192 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0 | 0 | 0 |
| 1 | 0.422 | 0.378 | 29 | 32 | 32 |
| 2 | 0.570 | 0.473 | 28 | 8 | 8 |
| 3 | 0.580 | 0.612 | 26 | 8 | 8 |
| 4 | 0.580 | 0.606 | 26 | 4 | 4 |
| 5 | 0.541 | 0.576 | 26 | 4 | 4 |
| 6 | 0.556 | 0.633 | 26 | 4 | 4 |
| 7 | 0.524 | 0.587 | 27 | 4 | 4 |
| 8 | 0.517 | 0.581 | 28 | 4 | 4 |
| 9 | 0.518 | 0.539 | 28 | 12 | 12 |
| 10 | 0.561 | 0.609 | 27 | 4 | 4 |
| 11 | 0.548 | 0.544 | 28 | 8 | 8 |
| 12 | 0.549 | 0.550 | 28 | 8 | 8 |
| 13 | 0.510 | 0.521 | 28 | 4 | 4 |
| 14 | 0.498 | 0.489 | 29 | 8 | 8 |
| 15 | 0.458 | 0.475 | 29 | 12 | 12 |
| 16 | 0.552 | 0.530 | 28 | 24 | 24 |
| 17 | 0.459 | 0.426 | 29 | 4 | 12 |
| 18 | 0.503 | 0.447 | 29 | 4 | 4 |
| 19 | 0.389 | 0.403 | 30 | 4 | 4 |
| 20 | 0.431 | 0.402 | 30 | 8 | 8 |
| 21 | 0.417 | 0.381 | 30 | 8 | 8 |
| 22 | 0.391 | 0.441 | 30 | 0 | 0 |
| 23 | 0.392 | 0.447 | 30 | 8 | 8 |

The spectra become somewhat harder late (`read-yv` energy@8 averages .552/.539/.435 over
L1-7/L8-15/L16-23), but spectral energy alone does not determine importance: its correlation
with the C8 single-layer loss penalty is only -0.25. L1 and L16 dominate causal sensitivity,
whereas many neighboring layers tolerate C4 and L22 tolerates C0. Use spectra to describe the
compression geometry and paired loss to allocate capacity.

Artifact: `/data0/xd/bam_diagnostics/v1_abs_v_layerwise_b4deeb3/v1_abs_v_layerwise_b4deeb3.json`.
