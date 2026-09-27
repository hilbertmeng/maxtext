# Dynamic RMT boundary combination in refactor-bam

Merged from `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`,
reference source commit `d49f5eec`. Historical training hashes remain in exp.py:
18-layer `be5491f`, 22-layer `a8d5bcd`.

Supported combined configurations:

- `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32`:
  18 layers, MLP4078, three-layer block scan, 432119360 parameters.
- `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32L22`:
  22 layers, MLP3212, direct layer scan, 432083008 parameters.

Both retain matrix48x75, first16-row vector pre-norm, tail32 dynamic middle
reads (C8 for VO/MLP), shared rank4 QK57 plus independent RoPE18, full48
static and dynamic writes, dynamic embedding write, and Direct32 final read.
Final full-matrix normalization and all layer/boundary health exports remain.
Parameter names and scan axes match the original training checkpoints.
This runtime supports training; autoregressive decode is not implemented.

## NoO option

Set `rmt_dynamic_o_enabled=False` for either combined configuration.
Only the extra dynamic O read/gate disappears. Static V read, dynamic V
read, shared VO key/compression, attention result and both attention writes
remain. Embedding and unembedding also remain. O health amplitudes/gates
report zero; the health schema stays unchanged.

No automatic MLP repayment accompanies this switch. At unchanged widths:

| configuration | O enabled | NoO | reduction |
|---|---:|---:|---:|
| 18 layers | 432119360 | 431773472 | 345888 = .2402 W_Q |
| 22 layers | 432083008 | 431660256 | 422752 = .2936 W_Q |

Each removed gate is `(D+1)*heads=19216` parameters per layer.
This is the same option as the historical ALiBi Full48NoO experiment;
no claim is made that the later combined configurations have undergone
this ablation. No new training run was launched by this merge.

## Scope and validation

Only RMT decoder dispatch, its LM-head normalization treatment, health
export, and optional additive attention bias touch shared code. Other
BAM/embedding mechanisms from the experimental worktree were not copied.
Unmerged LLF/fetch, headwise/transposed layouts, static-MLP, single-outer,
pure-dynamic-write and full-matrix-read variants fail explicitly rather than
silently executing another model under an old ledger class name.

Validation uses the pinned local CPU environment:

- Full BAM regression: all46 tests passed, four bounded CPU groups.
- Combined boundary initialization, finite gradients and Direct32 numerical
  read/gradient checks passed.
- Full-size `jax.eval_shape` parameter counts passed for18/22 layers, with
  O enabled and disabled.
- NoO regression uses nonzero dynamic keys; verifies live V, zero O health,
  finite gradients and retained attention writes.
- Layer-scan regression verifies independently parameterized final layer,
  final-layer gradient, all41 health fields per layer and both boundaries.
- Direct comparison against the untrimmed reference RMT source: identical
  seeded parameter trees, forward outputs, health and parameter gradients
  for block scan and layer scan, with O enabled and disabled, using nonzero
  dynamic keys (FP32 CPU). O-enabled block/layer scan also matched exactly
  in BF16, including gradients.

Evidence logs: `/tmp/rmt-merge-bam-suite.log`, `/tmp/rmt-merge-tests.log`,
`/tmp/rmt-merge-new-tests.log` (budget test passed; initial standalone NoO
health helper used the block-scan setting), `/tmp/rmt-merge-noo-retest.log`,
`/tmp/rmt-merge-depth.log`, `/tmp/rmt-source-equivalence.log`,
`/tmp/rmt-source-equivalence-bf16.log`, `/tmp/rmt-merge-bias-test.log`.
The additive-bias check verifies both FP32/BF16 logits and exact causal masks.
