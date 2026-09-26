# Dynamic embedding and unembedding on RMT MHABudget

Implementation: `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`.
Parent: `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget` (140dd4b).
New RUNs use fresh initialization, 13500 steps, UE5a v5p-16, checkpoint/loss
interval200. Existing three MHABudget/LLF RUNs are not changed.

Both arms retain the parent's static seed, static final read, final full-M RMSNorm,
RoPE18, full48 dynamic intermediate writes, tail32 dynamic intermediate reads,
LLL block scan, optimizer, data, and per-layer dynamic health.
Only one boundary is made dynamic in each arm; neither adds fetched O.

## Boundary definitions

Embedding: add an independent full48 write from raw token embedding e to the
native static seed. Content is D->16x75 without bias, per-head RMS-normalized.
Address is D->R256->16x48, GELU between projections, with zero pre-RMS bias,
per-head RMSNorm. Each head's sigmoid gate has learned D->16 kernel and0.1
bias opening. No sqrt(heads) scaling, matching the historical no-MHA-V BAM
embedding write. Sum gated address/content outer products in native48x75
orientation. Both matrix write contractions use the inherited setting.

Unembedding: keep final full-M RMSNorm, read the first16 rows as a D1200 proxy,
then learned vector RMSNorm. Independently compress only the remaining32 rows
to C8, yielding75x8. Generate zero-initialized D->16x8 keys without bias,
RMS-normalize, read16x75, gate with independent D->16 gates initialized0.05
and multiply the inherited0.2 read scale. Add to the native static48->16 read;
flatten and use the same LM head. Static read is not gated. New compression
is independent of the middle layers.

## Exact parameter repayment

`W_Q=D^2=1440000`; SwiGLU width1 costs3600 parameters/layer. A repeated
three-layer block permits total-width repayment in increments21600. Select
nearest total budgets while spreading widths by at most1; no hardware rounding.

| RUN suffix after `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget` | Extra boundary parameters | W_Q | MLP widths/block | Full parameters | vs MHA |
|---|---:|---:|---|---:|---:|
| `DynamicEmbedding` | 1963792 | 1.363744 | 4088/4088/4087 | 432110944 | -10256 |
| `DynamicUnembedding` | 174272 | .121022 | 4115/4115/4116 | 432114224 | -6976 |

MHA control:432121200. Parent:432112752, widths4118/4118/4118.
Embedding extra:1440000 content +307200 address-down +196608 address-up
+768 pre-RMS bias +19200 gate kernel +16 gate bias.
Unembedding extra:1200 proxy norm +256 compression +153600 key
+19200 gate kernel +16 gate bias.

All inherited health settings stay on. Each new boundary adds compact static/
dynamic RMS, ratio, cosine, gate mean/std and fractions below0.05, above0.5,
above0.95. No extra Gram diagnostics.

## Validation and bets

`run_rmt_boundary_cpu_tests.sh` includes full-size abstract parameter audits,
embedding equivalence and gradients against the original `EmbeddingBamWrite`
after swapping matrix axes, finite full-model gradients, exact zero-initialized
unembedding equivalence to the parent at equal test MLP widths, and existing
block-scan health checks. Launcher also runs the pinned BAM suite locally.

CPU tests, AOT on the two retained EW4a compilers, and UE5a trainer prequeues
run concurrently. Retained compilers are borrowed only and never cleaned up.

Pre-run13500-step bets vs MHABudget: DynamicEmbedding -.004;
DynamicUnembedding -.008. Loss ordering: DynamicUnembedding <
DynamicEmbedding < MHABudget. Steady-speed predictions: -.01 and0.00,
respectively. Dynamic output addresses may release a fixed-read bottleneck;
embedding dynamism costs a larger amount of MLP capacity and is less certain.
Direct comparisons: each arm vs MHABudget only. No peer or MHA loss reports.
First800 steps: report every200 steps; thereafter about1000 steps per batch.

## Startup

Runtime `d7cb6c12b91337f10fc5fcc3721fe6dd5f672f7c`, pushed. Both isolated
launches passed3 RMT checks and47 pinned BAM checks, verified AOT loaded and
FIRST_STEP. CPU tests ran on the local workstation; AOT and trainer prequeues
ran concurrently.

| RUN suffix | UE5a training TPU | EW4a borrowed compiler | Launcher UTC |
|---|---|---|---|
| DynamicEmbedding | xd-v5p-16-2609264-maxtext | llm-jax-v6e-1-0 (STANDARD guaranteed) | 2026-09-26T13:35:17Z |
| DynamicUnembedding | xd-v5p-16-2609265-maxtext | llm-jax-v6e-1-1 (FLEX_START) | 2026-09-26T13:35:05Z |

Neither retained compiler is owned by training cleanup. Launcher evidence:
`/data0/xd/rmt-dynamic-embedding-launch.log` and
`/data0/xd/rmt-dynamic-unembedding-launch.log`; startup logs and health cache
in `/data0/xd/rmt-vectornorm-mha-budget-startup/`.

Boundary health atsteps0/10/20/40: embedding dynamic/static ratio
3.087/3.556/5.920/12.962, gate mean~.0995 throughout. The native seed RMS is
~.00594, while normalized dynamic write RMS grows. This boundary is quickly
dominated by the dynamic route, rather than remaining a small perturbation.
Unembedding dynamic/static ratio0/.00369/.01241/.07805; gate mean
.05029/.05029/.05055/.10296. No gates exceed.5 at these early points.

Steady20-99 throughput (inverse mean duration from rounded step/s logs):
DynamicEmbedding .3792954 (+.4978% vs parent .3774168);
DynamicUnembedding .3765610 (-.2268%). Same UE5a v5p-16, same inherited
layer and basic health, plus9 boundary metrics in each new arm. No separate
health-disabled timing control. The embedding speed bet (-1%) had the wrong
sign; both measured costs are small. Startup speed evidence:
`/data0/xd/rmt-vectornorm-mha-budget-startup/boundary-speeds.json`.


## Uncompressed dynamic unembedding arm

`RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembeddingDirect32`
keeps the same static full48 final read and dynamic proxy/gates as the C8 arm.
It removes only the boundary32x8 compression and replaces the zero-initialized
D->16x8 key with D->16x32; RMSNorm now spans32 key coordinates. Direct dynamic
read consumes the entire tail32x75 state. Middle-layer C8 reads stay unchanged.

Boundary parameters634816 =1200 proxy RMSNorm +614400 key +19200 gate +16 bias,
.440844 W_Q; vs C8 +460544 (.319822 W_Q). MLP widths4108/4108/4109 give
432121168 total parameters,32 below MHA budget. Only compare to MHABudget.
Pre-run13500-step bet: gap-.006; steady speed-1% vs MHABudget. A larger
output-address space can remove the C8 read bottleneck, at the cost of MLP width.

CPU checks extend full-tree parameter audit and full-model zero-read/finite-gradient
checks to this arm, plus nonzero-key direct32 read/value/gradient equivalence.
Launch uses the verified idle FLEX_START llm-jax-v6e-1-1 in EW4a only for AOT;
new UE5a training TPU xd-v5p-16-2609266-maxtext is separately owned.

Runtime `a5364424af6e2e03f16a83754d5263016c681551`, pushed. Launched
2026-09-26T15:01:08Z on UE5a xd-v5p-16-2609266-maxtext after4 RMT and47 BAM
checks passed. Verified compiled artifact loaded, actual step20 and onward.
Steady20-99 .3757134 step/s (-.4513% vs MHABudget .3774168), same inherited
health plus9 boundary fields. Prequeue first observed READY14:55:59Z; registry
controller observation15:01:12Z. Evidence: /data0/xd/rmt-dynamic-unembedding-direct32-launch.log,
/data0/xd/rmt-unembedding-direct32-step14.log, /data0/xd/rmt-unembedding-direct32-speed.json.
Only MHABudget is registered as baseline. First800 report every200, then~1000-step
batches; review2800.

Future local boundary changes use `run_rmt_unembedding_direct32_cpu_tests.sh`
with `--cpu-test-scope targeted`, retaining full-size parameter audit, target
full-model initialization/gradients/health, nonzero direct-read equivalence,
and existing C8 V/O/fetch key regression. These checks passed in~1 minute
with3 bounded CPU groups. Full BAM regression also passed in141s with4 groups
(vs359s serial). Training runtime remains a536442; these workflow/tests-only
updates do not hot-switch running models.
