# H write recipe and common V/O scaling on Pythia 2.8B

Configuration-only preparation, requested 2026-10-30. No AOT, training RUN or
TPU allocation was submitted. Implementation worktree:
`/data0/xd/mediumprop-common-read-raw-write`, branch
`codex/mediumprop-common-read-raw-write`. Main `MaxText/exp.py` records the
classes as ledger entries; main's common-read/raw-write guard intentionally
rejects execution until that implementation is merged.

Common comparator:
`BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80LocalODynamicNormScale020`.

Let the class prefix be
`BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixed`.

| Priority | Full class name | Learned V/localO scalar initialization | Dynamic-O gamma |
|---|---|---|---|
| 1 | `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneLocalODynamicNormScale020` | all32 start at1 | .2 |
| 2 | `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedLocalODynamicNormScale020` | `1/sqrt(1+2*l)`, l=0..31 | .2 |
| 3 | `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixed` | `1/sqrt(1+2*l)`, l=0..31 | absent |

All three retain attention-write content RMSNorm, use raw private-MLP content
with divisor1, and use raw embedding-write content divided by .0020825863.
Address RMSNorms and write gates remain. The directly learned signed scalar
multiplies complete dynamic+static V and localO, all heads, before context
attention and write-content construction. Q/K have no added scale; full100-D
QKNorm remains before RoPE. The v8t read folds the scalar into its static V/O keys and
post-RMS dynamic V/O scales, so no [B,T,N,V] epilogue follows the kernel.
Generic health retains all32 signed values; internal/BAM activation metrics
remain disabled. Existing scale weight-decay exemption applies.

Inherit target dimensions D2560/L32/N32, M80x64, all R640, GELU7808/2944 and
tail9984, LayerNorm, original BAM/embedding .006 initialization and Pythia
Transformer initialization. Scan31 single layers plus explicit tail1; the tail
has its read scalar but no attention or private-MLP write. Preserve original28k
training schedule/batch/data, tuned Splash and minimal remat.

With gamma, total parameters are2,775,256,448, current gamma comparator+32
(0.000004883 W_Q). Without gamma, total2,775,190,912, comparator-65,504
(-0.009995117 W_Q). These negligible differences do not alter MLP dimensions.

## Embedding divisor

Only embedding requires a transferred fixed divisor. It has no feedback from
later M updates, so a full-network iterative calibration is unnecessary.
For independent .006 Gaussian embedding/projection entries, per-coordinate
content variance is `v=D*.006**4=3.31776e-6`. With K80 and epsilon1e-6,
`q/v ~ chi2(80)/80`; match mean raw and normalized write energy using

`tau = sqrt(v / E[q/(q+epsilon)]) = .0020825863071568967`.

This includes the finite-head correction, beyond `sqrt(v+epsilon)`. Head cross
terms vanish in the ensemble approximation. Actual fixed initialized weights,
gates, addresses and BF16 rounding are independently checked by
`calibrate_pythia_embedding_write.py`: 512 Gaussian embedding samples for
calibration and512 held out, target D2560/N32/K80/V64, initialization seed0.
Energy ratios fixed/normalized are1.0017066 and .9978411; empirical divisors
.0020847183/.0020806862. These are an embedding-only distributional check,
not an actual-token Pile loss evaluation or a per-layer amplitude constraint.
Small MLP stays raw without any amplitude compensation.

Results: `/data0/xd/bam_diagnostics/pythia2p8-common-read-raw-write/embedding-calibration.json`.

## Pallas and validation

Reuse the family branch's existing v8t support: common read scaling folded into the read,
`content_divisors=(None,1.0)` for attention/private-MLP write groups, embedding
write in XLA, and existing dynamic-O gamma read support. No kernel changes or
new VMEM buffers. Keep the target's60MiB scoped budget.

The first runtime multiplied complete V and localO in XLA after the read; Splash
could not fuse it, costing24.3ms/step at32 layers. Paired v5p-8 UE5a timing,
32 layers, 8/device, minimal remat, health ON (`claude/pythia-common-read-fold-b8`):

| Arm | XLA epilogue b59a903 | Folded 1f9ec4f |
|---|---|---|
| `BamPythia2p8BCommonReadParentProfile` | .6380 step/s, 1516.2ms | .6390, 1516.2ms |
| `BamPythia2p8BCommonReadHProfile` | .6280, 1538.1ms | .6350, 1520.8ms (+1.11%) |
| `BamPythia2p8BCommonReadUnitGammaProfile` | .6280, 1542.0ms | .6360, 1521.6ms (+1.27%) |

The folded read returns attention glue to the parent; +2.5ms remains in the read
reverse's scale reduction. 16/device with minimal remat exceeds v5p-8 HBM by
6.26GiB, so v5p-8 timing uses the V64-profile 8/device recipe. Versus F32 truth,
folded BF16 full-model gradients have lower global relative L2 (.0098 vs .0105);
the BOTH parity check uses the global bound with per-leaf relative tolerance .07.

Focused checks are in `MaxText/tests/bam_pythia_common_read_raw_write_test.py`:
actual2.8B abstract parameter trees/budgets and isolated config deltas; scanned
initial values and32 signed health values/WD; three scanned GELU/LayerNorm,
full-head QKNorm models comparing native vs Pallas logits and all gradients.
The latter uses reduced model widths/depth with target K80/V64/QK100; it is not
a formal topology speed or memory test. Artifacts are in the diagnostic folder
above. Before formal training, audit current main operational fixes and prepare
the exact chosen training-topology AOT. No training-result prediction is sealed
by this configuration-only task.
