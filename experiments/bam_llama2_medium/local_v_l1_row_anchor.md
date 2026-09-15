# LocalV L1 row anchor

## Experiment and reproduction

- RUN: `BamMediumIndependentLLFLocalVRank4BLocalORowDecodeL1Anchor`.
- Worktree: `/data0/xd/llf-l1-row-anchor`; branch: `codex/llf-l1-row-anchor`.
- Implementation starts from `286da7a55e00f89c38945a0e575109aec6d12d41`, the actual LocalORowDecode implementation, not its ledger-only main class.
- Direct comparisons: `BamMediumIndependentLLFLocalVRank4RoutingBLocalORowDecode` and `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow`.
- 24 layers, LLF block scan (8 iterations), C256, AOT, 13,500-step schedule, checkpoint every 200 steps and final checkpoint.
- Generic training health ON; only the new row-anchor BAM statistics ON. Historical all-health-OFF throughput is not a matched speed control.

## Definition

Layer indices start at zero. LocalV uses full M32×32, rank4 routing B and key scale 1; LocalQK stays rank1 legacy.
LocalO row8 is decoded into the full address space by its own compression transpose, as in LocalORowDecode.
Both inject their 32D row result into head coordinates 32:64. F layers are unchanged.

Let `u_l` be native LocalV row output after its existing gates and head mixing, before MHA V addition.
Save `anchor = u_1`, shape `[batch, token, 16, 32]`, without stopping its gradient.
For later Local layers use `u'_l = (1-g_l) u_l + g_l anchor`;
`g_l = sigmoid(W_l x_l + b_l)` has shape `[batch, token, 16, 1]`.
The gate projection is zero-initialized and the bias gives initial opening .1.
L0/L1 remain native. Columns and all existing read gates are unchanged.

Scan carry is `(hidden, M, anchor)`; only global L1 updates anchor.
Later blocks and F layers forward that same array. Layer index is `3 * block_index + offset`.
There is no conditional branch (`lax.cond`) and no first-block parameter special case.
All 16 Local layers own gate parameters; the first two are inactive, preserving uniform scanned parameter shapes.
Extra parameters: `16 × (1024×16+16) = 262400` (0.25024 W_Q in the whole model).
The carry adds 512 bf16 elements/token (1 KiB); it adds no persistent decoding M/KV cache.

## Verification

Pinned CPU commands (from the worktree):

```bash
bash /home/xd/projects/maxtext/.claude/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh /data0/xd/llf-l1-row-anchor
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=MaxText:MaxText/tests /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_row_anchor_test.py
```

- Original 47-test suite passed.
- Analytic L1-only capture and downstream gradient test passed.
- Two complete LLF blocks: scan versus identical mapped parameters unrolled, forward and all parameter gradients passed tolerance checks. Nonzero read biases exercise nonzero anchors.
- The unrolled anchor is bitwise unchanged after block1; scan per-layer anchor RMS is bitwise unchanged after L1. Scan/unroll cross-graph differences are floating-point roundoff, not a bitwise-equivalence claim.
- Jaxpr retains scan and contains no `cond`.
- Complete scan and non-scan training signatures, checkpoint/schedule, raw_grad_norm and scoped health exports passed.

## Health and prediction

`bam/row_anchor/layer_XXX/`: active flag, gate mean and five bins over [0,1], native/anchor RMS,
weighted-native/weighted-anchor RMS, and their per-vector cosine. Inactive first-block gates are explicitly flagged.
Compare layer1 anchor RMS against every consuming layer at a common step; they must agree.
Together with loss and raw gradients, these distinguish useful reuse, gate shutdown, and excessive L1 domination.

Pre-run bet: final gap -.002 versus LocalORowDecode, -.002 versus BAlignedRow;
architectural throughput change about -1.5%/-2.4%, respectively, before unmatched health overhead.
Full coordinates solve the private compression-coordinate mismatch, not the possible semantic/head specialization mismatch across layers.
No claim of benefit is made before training.
