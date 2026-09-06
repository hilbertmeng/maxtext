# Nonnegative fetch-alpha controls

Branch `codex/bam-nonnegative-mix`, worktree `/data0/xd/bam-nonnegative-mix`,
branched from `e610c3634a14414a84d2e13290eda16637f9516b`. Diagnostic hooks remain
in the separate `codex/bam-row-mediation` branch.

All runs inherit `BamLlama2MediumV2C256ScanAotControl`: full-24, C256,
layer-scan + v6e-precompiled AOT, C8, 13,500 planned steps, checkpoint every 200.
Keep fixed-one fetch diagonal, LocalQK, read keys/gates, and write path unchanged.
Clip **after mixing attention heads**, not the head-mixture coefficients; set the
diagonal to one afterward. MHA attention itself is unchanged.

| Class suffix after `BamLlama2MediumV2C256` | Mixture | Direct comparisons | Predicted final gap vs control |
|---|---|---|---:|
| `SoftmaxMix` | `w=softmax(Wx+b); alpha=sum(w*A)` | ScanAotControl | +.003 |
| `ClippedAlphaMix` | `w=Wx+b; alpha=max(sum(w*A),0)` | ScanAotControl, SoftmaxMix | +.001 |
| `StaticClippedAlphaMix` | per-layer `w[n]`; `alpha=max(sum(w*A),0)` | ScanAotControl, ClippedAlphaMix | +.007 |
| `RmsGeluAlphaMix` | `w=RMSNorm(Wx+b)/sqrt(n); alpha=GELU(sum(w*A))` | ScanAotControl, ClippedAlphaMix | +.010 |

The GELU follow-up replaces SoftmaxMix only after its v6e AOT artifact is ready.
Unlike the original three arms it permits negative alpha. GELU is applied before
the fixed-one diagonal and only to BAM's mixed route. Near zero it halves the
cross coefficient; neither nonnegativity nor matched cross/self scale is claimed.
Versus ClippedAlphaMix it changes both coefficient normalization and activation;
versus ScanAotControl only the post-mix GELU is added. Parameter count is unchanged.

These numbers are pre-run bets, not measurements. The first two preserve the
current dynamic projection's regular initialization and zero bias. The static
coefficients start at `1/n`, can become signed, and are shared across tokens but
not layers. They are named `fetch_head_mix_bias` and excluded by the existing
`.*bias$` weight-decay rule. Attention heads remain input-dependent even in the
static-mixture arm. Different initial mixtures confound early optimization-speed
comparisons; do not infer final capacity from a step-200 gap.

Primary TPU region: UE5a, based on recent formal v5p lease history. Request formal
v5p-16 only after the corresponding `prepare_train_aot.py` artifact verifies.
Check FIRST_STEP and steps 10–14 speed; report unexpected speed changes.

Alongside cumulative 200-step loss gaps and checkpoints, monitor existing
fetched gate/read-health and gradient/clipping metrics. New per-layer TB tags
`bam/fetch_route/layer_NNN/` retain preclip-negative fraction, final-zero fraction,
cross mass/query, cross L2 RMS/query, coefficient mean/RMS/negative fraction.
Masked/padding and diagonal edges are excluded from cross-edge denominators;
sum chunk sufficient statistics before taking ratios. This distinguishes a
healthy self path from a cross path made inactive by clipping.

After the standard incremental TB sync/read-health report, run
`experiments/bam_llama2_medium/report_fetch_route_health.py RUN... --steps 0,200,...`.
It reuses the same local scalar cache, prints complete horizontal route series
for each layer band and RUN, and represents unavailable historical tags as `--`.
Pass all three RUNs together for direct comparisons rather than subtracting
health metrics. These reporting-only changes do not alter the AOT runtime hash.

Implementation: `MaxText/layers/attentions.py`; TB export: `MaxText/train.py`;
configuration classes: `MaxText/exp.py`; value/gradient/masking tests:
`MaxText/tests/bam_attention_test.py`. Runtime hashes and measured speed/results
will be recorded in the configuration classes after launch.

## Launch and early health

All three loaded the v6e-precompiled function and reached FIRST_STEP on UE5a
v5p-16. Runtime `feef2596e78b916ba27fa2f9fb5ec9233e18da8a`; source and AOT
manifests are recorded in their RUN registries. Steps 10–14 average throughput:
Softmax .650, dynamic clip .6456, static clip .6576 steps/s, versus control .660.
The new route-metric reductions are included in those numbers.

At step 50, dynamic clip's L8–15 cross-edge zero fraction reached .99975 and
cross mass/query .000620, compared with about .98 mass and zero clipped edges
for Softmax/static. This early cross-route collapse was stronger than predicted;
watch recovery versus persistent hard-clip inactivity, without changing the arm.
These are training-batch TB statistics, not the fixed-cohort checkpoint probe.
TB remains at the inherited UC1 summary prefix; dataset and checkpoint/output
prefixes are zone-local UE5a. Use the registry for checkpoint checks and the
class's actual TensorBoard prefix for incremental health sync.
