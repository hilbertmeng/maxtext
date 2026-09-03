# XL fetched-M AbsV width diagnosis

## Finding

Increasing fetched-M AbsV width from C8 to C32 introduces the same initial
width-scaling pressure in Medium and XL: fetched-read energy and the `W_R`
Jacobian grow as `sqrt(C)`, so the step-0 fetched gradient is about 2x larger.
This is the root cause of XL C32's optimization failure, but it is not a
universal claim that C32 must hurt loss.  Medium partially compensates through
smaller runtime gates, exits the clipping regime much sooner, and gains enough
representation capacity for native C32 to beat C8.

Adam's coordinate-wise normalization does not itself provide the missing
`sqrt(C_ref/C)` branch calibration.  The model may nevertheless learn a partial
correction through its gates and surrounding projections.  XL does not learn
enough correction soon enough; Medium does, and its residual 15% stronger
readout is useful rather than fatal.

## Is XL Rank2 C8 still redundant?

No evidence supports reducing the fetched-M cache from C8 to C4. On the
completed XL Rank2 checkpoint at step 49,720, 128 fixed Pile-eval sequences give:

| C8 object, layers 1--23 | top-4 energy | top-6 energy |
|---|---:|---:|
| learned `32->8` cache projection | 62.5% mean (50.0--70.0%) | -- |
| actually mixed fetched `Mbar` | 71.2% mean (62.4--85.9%) | 88.1% |
| fetched row-read output | 74.3% mean (62.5--80.0%) | 89.3% |

A shape-preserving SVD intervention that reduces only one layer's learned
`32->8` cache projection to rank 4 raises same-batch loss in every layer. The
median increase is `+.01097`; the most sensitive layers are L2 `+.45612`, L4
`+.10609`, L1 `+.09842`, and L3 `+.05566`. Applying rank 4 to all layers raises
loss by `+4.30384`; even all-layer rank 6 raises it by `+2.66793`. This frozen
intervention cannot rule out recovery through from-scratch retraining, but it
shows that C8's trailing directions are used rather than obviously redundant.
The concurrent native C4 run is the decisive trainable-capacity check; its first
500-step gap is already `+.03740` versus C8, consistent with the diagnosis.

Reproduction:

- runner: `run_xl_rank2_c8_redundancy.sh`
- analysis: `xl_abs_v_width_diagnostics.py`
- code commit: `6a60760`
- report: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl_rank2_c8_redundancy/6a60760/report.json`

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

## Matched Medium native-C32/C8 check

This is the clean Medium comparison:

- native C32: `BamLlama2MediumV1`, checkpoint 13,250;
- C8: `BamLlama2MediumV1CompressAbsV8Direct`, checkpoint 13,250.

Only the fetched cache/read view changes width.  M writes and LocalQK both keep
the full native `32x32` M; this is **not** the later experiment that makes
LocalQK read a compressed `32x8` M.  Both checkpoints were evaluated on the
same 32 fixed, unique Pile sequences.

| fetched-read metric, layers 1-23 | C8 | native C32 | C32 / C8 |
|---|---:|---:|---:|
| mean `||y_bam||/||y_std||` | 1.946 | 2.268 | 1.166 |
| pooled-energy `||y_bam||/||y_std||` | 2.168 | 2.499 | 1.153 |
| inverse `||y_std||/||y_bam||` | .461 | .400 | .868 |
| column/U contribution over standard | 1.967 | 2.177 | 1.107 |
| row/V contribution over standard | .911 | 1.227 | 1.346 |
| post-RMS/post-gate read-key entry RMS | .02414 | .01892 | .784 |
| same-batch eval loss | 2.39451 | 2.38917 | -0.00534 absolute |

The C32 model therefore does have a stronger fetched branch, especially on the
new V coordinates, but nowhere near the naive 2x norm increase.  Its learned
runtime gate is 21.6% smaller than C8's; equivalently C8 opens its narrower
read key 27.6% more.  The late-layer pooled ratios are 2.854/3.491 for C8/C32,
so the residual scale difference is concentrated where BAM readout is already
strongest.  The training-curve result agrees with the same-batch probe: C8 is
about `+.0096` loss worse over steps 12,400-13,400.

Medium also shows the predicted early gradient problem; it is just transient:

| Medium training interval | C8 raw-grad mean / clipped fraction | C32 raw-grad mean / clipped fraction |
|---|---:|---:|
| 0-200 | 1.458 / 50% | 1.602 / 65% |
| 200-400 | 1.100 / 65% | .904 / 25% |
| 400-600 | .720 / 5% | .634 / 0% |

At step 0, the fetched parameter-group gradient is `2.526 -> 5.132` (2.03x)
and global raw grad is `4.236 -> 6.156` (1.45x), with clipping threshold 1.
Thus Medium does not escape the width law.  It exits the excess-clipping phase
by about step 400, and sampled fetched/standard gradient ratios over the full
run average almost identically (`.252` C8, `.251` C32).

XL differs mainly in persistence:

| XL training interval | C8 raw-grad mean / clipped fraction | native C32 raw-grad mean / clipped fraction |
|---|---:|---:|
| 0-250 | 2.576 / 100% | 3.083 / 100% |
| 250-500 | 1.555 / 96% | 2.074 / 100% |
| 500-1,000 | .588 / 2% | .828 / 30% |

XL C32 remains in the clipping regime substantially longer, then settles at
about `+.0045` loss versus C8 from steps 3,500-8,500.  Its trained pooled
fetched/standard ratio is 2.764 versus 2.177 for C8 (+27%); these checkpoints
are not progress-matched (about 7.8k versus 49.7k), so this number is supporting
evidence rather than the primary causal comparison.  The same-initialization
step-0 and step-10 interventions above establish the causal width effect.

Why the loss sign differs is therefore a benefit/cost tradeoff, not a different
mechanism:

1. The extra 24 fetched coordinates occupy 37.5% of a Medium H64 head but only
   18.75% of an XL H128 head, so their marginal representational value is much
   larger in Medium.
2. Healthy XL C8 already assigns more gradient budget to fetched read than
   Medium C8, while XL's Partial-RoPE + rank-2 LocalQK path supplies additional
   memory-read capacity.  Extra fetched width is consequently more redundant.
3. Medium learns a stronger gate compensation and rapidly leaves clipping;
   XL C32 retains a larger forward/Jacobian mismatch long enough to alter its
   optimization path.

The decisive follow-up is not another C8/C32 comparison, but native C32 default
versus native C32 with `sqrt(8/C)` calibration in both Medium and XL.  This
separates C32's capacity gain from its scale cost.  Scaling raw `W_R(x)` before
RMS would be cancelled; calibration must act on the post-RMS key, gate prior,
or final fetched readout.

## Absolute startup clipping versus MHA

`raw_grad_norm > 1` is not by itself a BAM failure: the matched MHA controls
also spend their early steps in the clipping regime.

| model / interval | MHA clipped | BAM C8 clipped | BAM C32 clipped |
|---|---:|---:|---:|
| Medium 0-200 | 77.5% | 50% | 65% |
| Medium 200-400 | 10% | 65% | 25% |
| XL 0-250 | 100% | 100% | 100% |
| XL 250-500 | 76% | 96% | 100% |
| XL 500-1,000 | 2% | 2% | 30% |

At step 0, removing the fetched `W_R` group's gradient in quadrature recovers
the matched MHA raw gradient almost exactly:

| scale | MHA raw grad | BAM C8 raw grad | C8 fetched grad | BAM without fetched term |
|---|---:|---:|---:|---:|
| Medium | 3.4007 | 4.2361 | 2.5258 | 3.4007 |
| XL | 4.9706 | 6.0993 | 3.5317 | 4.9706 |

Thus C8 adds a real BAM-specific startup gradient, but its duration is healthy;
the pathological signal is the width-dependent excess and its persistence.
Using `sqrt(C_ref/C)` with `C_ref=8` fixes only that width dependence: C8 stays
unchanged and C32 gets a factor `.5`.  An absolute `1/sqrt(C)` would also shrink
the already-working C8 path by `.354` and conflates two independent choices.

If C8 itself is to be weakened, use a separate reference gain
`a_ref * sqrt(8/C)` or reduce the fetched gate prior.  This controls BAM's
overall startup learning strength, while the square-root term controls width
invariance.  Raising the global clipping threshold does not solve the relative
problem: it preserves the oversized fetched direction and merely increases the
total update.

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

## Reproduction

- Script: `xl_abs_v_gradient_diagnostics.py`
- Cross-scale readout script: `xl_abs_v_width_diagnostics.py`
- Raw reports: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/c32_abs_v/`
- Cross-scale readouts: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/cross_scale_readout/`
- Matched Medium C32/C8 readouts: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/medium_native_c32_vs_c8/`
- Diagnostic commits: `34bc7a9`, `1b9c0fc`, `ac7bca3`
