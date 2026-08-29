# XL fetched-M AbsV width diagnosis

## Finding

The C32 failure starts at the fetched-read `W_R` Jacobian, before any forward
output differs.  The implementation keeps per-coordinate M/read-key scale
fixed as AbsV width `C` grows, but has no `1/sqrt(C)` width calibration.  Thus
fetched-read energy and the `W_R` Jacobian norm grow as `sqrt(C)`.  C32 starts
with a 2x branch/Jacobian scale relative to C8; Adam then gives the extra
coordinates essentially the same per-coordinate update, so the mismatch is
not self-correcting.

This is the root scaling defect.  Larger readout energy, clipping, and later
P_loc/LocalQK gradient changes are downstream symptoms.

## Full-24 training evidence at step 0

All non-`W_R` gradients are identical across C8/C16/C32.  The global gradient
square increase is numerically equal to the `W_R` gradient square increase.

| config | `||g(W_R)||` | global raw grad | `global^2-W_R^2` | `W_R^2/global^2` |
|---|---:|---:|---:|---:|
| C8 | 3.53174 | 6.09930 | 24.72831 | 33.5% |
| C16 | 5.05033 | 7.08760 | 24.72830 | 50.8% |
| C32 projected | 7.16557 | 8.72202 | 24.72828 | 67.5% |
| C32 native | 7.18660 | 8.73931 | 24.72828 | 67.6% |

Across layers, C16/C8 `W_R` gradient ratios average 1.435 and C32/C8
ratios average 2.036, close to `sqrt(16/8)` and `sqrt(32/8)`.

## Which `W_R` coordinates grow

A fixed-data two-layer XL trace splits the zero-initialized projection into
the K-wide row key and C-wide column key:

| C | row-key L2 | row-key RMS | col-key L2 | col-key RMS | total L2 |
|---:|---:|---:|---:|---:|---:|
| 8 | 3.5906 | .001753 | 3.6459 | .005035 | 5.1171 |
| 16 | 5.4452 | .002659 | 5.2230 | .005101 | 7.5452 |
| 32 projected | 7.3167 | .003573 | 7.3429 | .005071 | 10.366 |

- Row-key parameter count is fixed; its per-coordinate gradient grows as
  `sqrt(C)` because every row key affects C output coordinates.
- Column-key per-coordinate gradient stays fixed, but its parameter count grows
  with C, so its total gradient also grows as `sqrt(C)`.

The initialized fetched M has the matching scaling:

| C | mean covariance eigenvalue | covariance trace |
|---:|---:|---:|
| 8 | 3.603 | 28.827 |
| 16 | 3.626 | 58.016 |
| 32 projected | 3.635 | 116.331 |
| 32 native | 3.636 | 116.336 |

Per-coordinate state energy is invariant, while total state energy grows
linearly with C.

The first downstream divergence is also visible in the full-24 TB history:

| group raw-grad L2 | C8 step 0 | C32P step 0 | C8 step 20 | C32P step 20 |
|---|---:|---:|---:|---:|
| `W_R` | 3.5317 | 7.1656 | .7700 | 1.6514 |
| `P_loc_up` | 0 | 0 | .0869 | .1732 |
| `W_local_qk_packed` | .0303 | .0303 | .0040 | .0073 |
| `fetch_head_mix` | 0 | 0 | .0792 | .3562 |
| standard V | 3.0241 | 3.0241 | .2379 | .2973 |
| standard O | 3.0196 | 3.0196 | .2865 | .3603 |

Thus P_loc/LocalQK/mix are not independent initial causes: they are identical
or zero at step 0 and diverge only after `W_R` has opened the fetched-read loop.

## Why it happens

`_matrix_for_read` gives M unit RMS per matrix element, not unit Frobenius norm.
For the zero-initialized `W_R`, the read transform's zero-point slope is

`read_key_scale * gate_init / sqrt(read_epsilon)`

which is `2 * .005 / sqrt(1e-4) = 1`.  This is a coordinate-wise identity
Jacobian, but it is not width invariant.

For `M:[K,C]`:

- column read `M @ r_C` has fixed K outputs whose variance grows with C;
- row read `M.T @ r_K` has per-coordinate variance fixed but C outputs;
- either side therefore has total squared read energy proportional to C.

The implementation then adds this read to `y_std` and immediately uses the
sum both for the residual output and as the next BAM write's data factor.
Consequently the initial mismatch propagates into P_loc, LocalQK, and future M
states.

## Causal intervention

For C32, changing only the fetched-read gate initialization from `.005` to
`.0025 = .005*sqrt(8/32)` leaves the step-0 forward exactly unchanged but
restores the C8 Jacobian scale:

| metric | C8 | C32 default | C32 gate-calibrated |
|---|---:|---:|---:|
| step-0 raw grad | 16.2567 | 18.5889 | 16.2373 |
| step-0 row-key grad L2 | 3.5906 | 7.3167 | 3.5681 |
| step-0 col-key grad L2 | 3.6459 | 7.3429 | 3.5811 |
| step-10 fetched/std norm | .01246 | .02832 | .01373 |
| step-9 `P_loc_up` grad L2 | .03353 | .07348 | .03569 |

The per-coordinate `W_R` Adam update is about `2.0e-7` for every width, which
explains why the default C32 path opens about four times as many coordinates at
the same individual amplitude.  In the full runs, `W_R` parameter RMS remains
nearly identical through 4k steps, and the learned scalar gate differs by much
less than the 2x compensation required.

| parameter statistic @4k | C8 | C16 | C32P |
|---|---:|---:|---:|
| `W_R` parameter RMS | .010658 | .010634 | .010590 |
| `W_R_gate` kernel RMS | .007190 | .006644 | .006200 |
| `abs(W_R_gate_b0)` RMS | 4.90496 | 4.90722 | 4.90991 |

The gate kernel adapts downward by only about 14% from C8 to C32 and the bias
hardly changes; neither approaches the 50% branch-scale correction required.

## Medium/XL C8 cross-scale check

The completed Medium V2 step-13,250 and XL16 Rank2 step-49,720 checkpoints were
run on the same 32 fixed Pile-eval sequences.  Here `y_bam` is fetched-M readout
at the `o_head = y_std + y_bam` addition.  The two models have `(K,C,H)` of
`(32,8,64)` and `(64,8,128)`.

| fetched-read scale | Medium | XL16 Rank2 |
|---|---:|---:|
| mean `||y_bam||/||y_std||`, layers 1-8 | .926 | .941 |
| layers 9-16 | 1.349 | 1.253 |
| layers 17-23 | 2.875 | 2.455 |
| energy-weighted across layers | 2.187 | 2.284 |
| mean `cos(y_bam,y_std)`, layers 1-23 | -.0467 | -.0550 |

The mean of the 23 matched-layer XL/Medium ratios is `.995`; XL is slightly
stronger early and weaker late.  The energy-weighted aggregate is only 4.5%
higher in XL because it weights layers by absolute `y_std` energy.  Thus the
healthy XL C8 checkpoint does not have a forward fetched-read scale explosion
relative to Medium.

TensorBoard per-parameter raw-gradient histories, aligned by relative training
progress, give a complementary result:

| relative progress | 20% | 40% | 60% | 80% | 98% |
|---|---:|---:|---:|---:|---:|
| Medium all-BAM / standard-attention | .687 | .617 | .546 | .518 | .490 |
| XL all-BAM / standard-attention | .758 | .686 | .599 | .535 | .490 |
| Medium fetched / standard-attention | .316 | .299 | .282 | .279 | .272 |
| XL fetched / standard-attention | .448 | .416 | .378 | .339 | .318 |

At 98%, LocalQK/standard is `.376/.354` and write/standard is `.158/.117`
for Medium/XL.  XL therefore has a persistently stronger fetched branch but
weaker LocalQK/write branches; their total BAM gradient budget converges to the
same `.490` ratio.  This is a branch-composition difference, not global BAM
instability.

The equal initial gate prior across these two C8 models is dimensionally
reasonable: fetched-read energy relative to one standard head scales as
`2*K*C/H * gate^2`, and `2*K*C/H = 8` for both.  This does not make one fixed
gate prior universally optimal.  Widening XL from C8 to C32 makes that factor
four times larger, so the branch norm doubles unless calibrated.

The apparent Medium/XL C32 contradiction is also not a matched ablation.
Medium's `BamLlama2MediumV2C256CompressedVLocalQK` comparison changes the
LocalQK source from the native full `32x32` M to the compressed `32x8` fetched
view; compression costs about `.008` loss, while fetched read remains C8 in
both runs.  XL's C32 runs leave LocalQK on the same full M and widen only the
fetched cache/read from C8 to C32, exposing the width-scaling defect above.

The next clean training control is therefore C32 with an explicit
`sqrt(C_ref/C)` fetched-read calibration (`C_ref=8`).  It can be implemented in
the gate prior, compressed-state view, or final fetched-read output; those are
equivalent at initialization, while the gate-prior form still permits learned
amplitude recovery.  Scaling the raw `W_R(x)` before RMS normalization would be
cancelled by RMS; a fixed calibration must act on the post-RMS runtime key or
readout.

## Reproduction

- Script: `xl_abs_v_gradient_diagnostics.py`
- Cross-scale readout script: `xl_abs_v_width_diagnostics.py`
- Raw reports: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/c32_abs_v/`
- Cross-scale readouts: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/cross_scale_readout/`
- Diagnostic commits: `34bc7a9`, `1b9c0fc`, `ac7bca3`
