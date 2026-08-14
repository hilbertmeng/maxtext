# BAM checkpoint and numerical diagnostics

This document contains checkpoint-dependent findings and deterministic training-trajectory
diagnostics. Current milestones and throughput live in [`bam_exp_memo.md`](bam_exp_memo.md);
low-level performance history lives in [`bam_profile_history.md`](bam_profile_history.md).

## Cohort

The V1 checkpoint diagnostics use step 13,250, commit `49be222`, and 128 randomly selected Pile eval
sequences. The original `4×32` cohort was executed as `8×16` microbatches with every sequence hash
preserved. Baseline loss was `2.379473`, within `5e-6` of the original batch-32 result. All reported
ablations below are same-checkpoint, same-batch, read-only substitutions; they do not include model
adaptation through retraining.

## M-cache temporal compression

Dynamic RMS mix is neither one-hot nor purely local: its unit-L2 head coefficients have maximum
absolute value 0.549 on average, effective head count `6.03/16`, and adjacent-token cosine 0.831.
Mixed signed alpha has effective support mean/median `43.2/16.8`; top-16/64/128 positions contain
`59.0/75.5/83.3%` of absolute mass, while the latest 128/256/512 contain `68.6/78.7/88.5%`.

The smooth route coefficients do not imply smooth cached matrices. Aggregate `M_s` cosine at lags
1/2/4/8/16/32 is only `0.249/0.179/0.140/0.113/0.096/0.085`; relative delta RMS is
`1.225/1.282/1.312/1.332/1.344/1.352`. Layer 1 lag-1 cosine is 0.014.

| Compression | Nominal cache reduction | dloss | fetch-M rel-RMS | BAM output rel-RMS |
|---|---:|---:|---:|---:|
| B8 mean | 8× | +0.4104 | 0.7525 | 0.7072 |
| B8 linear | 4× | **+0.2236** | **0.6358** | **0.6107** |
| B16 mean | 16× | +0.4859 | 0.8080 | 0.7468 |
| B16 linear | 8× | +0.3617 | 0.7282 | 0.6916 |
| B32 mean | 32× | +0.4844 | 0.8432 | 0.7688 |
| B32 linear | 16× | +0.4196 | 0.7857 | 0.7348 |

Linear beats mean, but every block-only scheme harms all 128 sequences. Temporal proximity is not a
healthy compression basis for this checkpoint.

| Read approximation (`T=2048`) | Cache reduction | dloss | fetch-M rel-RMS | BAM output rel-RMS |
|---|---:|---:|---:|---:|
| Window256 | 8.00× | +0.1193 | 0.3725 | 0.3988 |
| Window256 + OldBlock16 mean | 5.57× | +0.0811 | 0.3050 | 0.3496 |
| Window256 + OldBlock16 linear | 4.27× | **+0.0626** | **0.2818** | **0.3261** |

OldBlock16 recovers useful history but remains far above the usual `<0.01` same-batch tolerance.
Further cache work should use alpha-conditioned sparse or hierarchical memory rather than fixed
time averaging.

Artifact:
`/data0/xd/bam_diagnostics/bam_cache_diagnostics_49be222_mb16_final.json`.

## Signed dynamic mix

The observed 57.9% negative-alpha figure is an element count, not contribution mass. Absolute-mass
weighting gives positive/negative `56.8/43.2%`. Per-query negative-mass mean/p50/p90 is
`46.1/42.9/90.6%`; cancellation `1-|sum(alpha)|/L1(alpha)` is `46.9/48.4/86.4%`.

Layer behavior is heterogeneous: negative mass is 0.3% in layer 1 but 93.5/91.8% in layers 3/7.
Negative element count can disagree with mass, so mass is the useful statistic. Minority-sign mass
correlates with fetch/output error at `0.87/0.74` when removed.

RMS-mix coefficients have L1 mean/p50 `3.126/3.180` out of a maximum 4 and 70.6% coefficient-level
cancellation. The zero-sum contrast subspace carries 91.2% energy, close to the 93.75% expectation
for a random 16-D unit vector; this fact alone does not show learned preference for contrast.

| Same-batch ablation | dloss |
|---|---:|
| make mix coefficients absolute, preserve L2 | +2.8718 |
| remove negative mix coefficients, preserve L2 | +1.7539 |
| make alpha absolute | +1.5727 |
| keep raw positive alpha only | +0.3865 |
| keep dominant-sign mix, restore L2 | +0.1279 |
| keep dominant-sign alpha without rescaling | **+0.0293** |
| keep dominant-sign alpha, restore L2 | +0.0627 |
| common mean component only | +0.3867 |
| zero-sum contrast component only | +0.1855 |

All variants worsened all 128 sequences. The checkpoint uses signed routing and both common and
contrast components, but this proves co-adaptation rather than globally superior parameterization:
independent RmsMix and SoftmaxMix training differed by only `-0.0006` near step 6,200.

A principled future comparison is positive base routing plus a separately gated zero-sum contrast:

```text
w = g_base * softmax(z_base) + g_ctr * normalize_L1(z_ctr - mean(z_ctr))
```

Initialize the contrast gate near zero and the base gate open. This retains subtraction while
separating selection, contrast and gain.

Artifacts: `/data0/xd/bam_diagnostics/bam_alpha_*_final.json`.

## Attention sink and fixed-window failure

There is a local token-0 sink, not a broad prefix sink. For queries at position ≥1024, token 0 has
1.73× per-token enrichment; layers 8/11/13 reach `6.49/5.67/3.95×`. Yet token 0 carries only
0.114% of total absolute mass, and first-2/4/16 aggregate enrichment is only `1.04/0.67/0.39×`.

For all queries affected by Window256, token 0/first16/first64 explain only
`0.93/3.11/9.40%` of removed mass. Keeping the first four tokens changes Window256 loss from
`2.498772` to `2.498375`: only `0.000397`, or 0.33% of the window damage. Thus the sink is real but
does not explain fixed-window failure; useful old-history mass is dispersed.

Artifacts:

- `/data0/xd/bam_diagnostics/bam_alpha_sink_diagnostics_a02fc72_final.json`
- `/data0/xd/bam_diagnostics/bam_window_prefix4_diagnostics_d3c17a6_final.json`

## PackedLocalQK stable-gap root cause

The approximately `+0.0034` gap of the old packed eps1e-4 run versus Direct came from
`bam_replicate_ploc_up=True`, not packing, `btn` layout or native packed initialization:

- `btn`-only matched native PackedOnly exactly through every common step 0–41.
- replicated `P_loc_up` alone exactly matched the old run through step 56.
- a mapped-step-0 control still diverged at step 2 when replication was enabled, excluding initial
  parameter-value differences.
- Native PackedOnly's transient vanished by 1,800–2,600 (`-0.0010` mean gap versus Direct).

Replication changes `P_loc_up/kernel` axes from `('embed','q_heads','v_factor')` to
`(None,'q_heads','v_factor')`. On v5p-16 this replaces an 8-way FSDP-sharded reduction with
replicated computation and gradient all-reduce. The projection is mathematically unchanged, but
collective and reduction order changes perturb bf16 training. Replication gained about 1.5% speed
and produced a persistent loss penalty, so capability runs keep `bam_replicate_ploc_up=False`.

## V2 fast-path trajectory

V2 combines diagonal-one read with multiply+reduce dynamic-V writes. Against the same fp32 Packed
Native parent, each change alone showed no sustained early harm, while the combination initially
diverged:

| `RUN - Native` | 200 | 400 | 600 |
|---|---:|---:|---:|
| diagonal-one only | -0.01877 | -0.00639 | -0.00367 |
| write multiply+reduce only | +0.02758 | +0.00234 | -0.00142 |
| both (V2) | +0.07236 | +0.02037 | +0.01544 |

V2−Native then narrowed to `+0.00374` at 2,600, averaging `+0.00436` over 1,800–2,600. Primitive
checks showed unchanged mathematics but non-bitwise bf16 reductions: diagonal-one forward rel-RMS
`1.48e-5`, M-gradient `8.02e-4`; write forward rel-RMS `1.49e-8`, u1/u2 gradients about
`2.57e-3`; combined alpha-gradient about `3.38e-3`. Loss was identical at step 1, differed at
`1e-5` scale at step 2, and optimizer dynamics amplified the trajectory split.

After full 13,500-step training, V2 was only `+0.000138` versus long-running Direct at step 13,400.
The early divergence is therefore an initialization/reduction-order trajectory, not a persistent
capability loss. Native ended at 2,800, so this late result cannot separately identify the two fast
changes' asymptotic effects.

Reproduction script: `diagnose_v2_fast_path_numerics.py` at commit `90806a4`.
