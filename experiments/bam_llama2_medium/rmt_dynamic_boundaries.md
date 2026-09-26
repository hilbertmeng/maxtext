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
Direct comparisons: both vs MHABudget and MHA; Unembedding also vs Embedding.
