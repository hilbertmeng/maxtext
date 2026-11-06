# Inference fast path: design, cost analysis and negative results

Design of `fast_inference.py` and its kernels (`bam_kernels.py`, `prefill_attention.py`,
`decode_attention.py`) for Pythia unit-H BAM, and of the Pythia-2.8B baseline it is timed against.
Usage, environment and the speed table are in [README.md](README.md). Numbers are A800 80GB PCIe, BF16,
torch 2.14 / Triton 3.8, 32 layers, 32 heads, decode at a 1984-token context.

## Goal and fairness policy

The target is the BAM / Pythia throughput ratio, not BAM's absolute speed: an optimization that speeds
BAM but helps Pythia more is rejected.

- Components both models have use the same implementation in both: inductor (`max-autotune`) GEMMs for
  the x / QKV, out / dense, fc_in / fc_out and h projections, inductor LayerNorms, the split-KV Triton
  decode attention kernel, RoPE tables indexed by position, CUDA graphs over one decode step.
- Only BAM-specific parts get BAM-specific kernels: the M read and write, the strictly causal Q/K 100 /
  V 80 prefill attention, BAM's K cache layout, and the BAM-only small linears.
- Rejected for this reason: a generic one-token Triton GEMV for the shared linears (`SHARED_GEMV=1`)
  speeds up both models but Pythia more (bs1 ratio 0.891 -> 0.843).

## Structure of one BAM layer

Per layer the fast path runs: LN1 + one `x_w` GEMM (M shift, query, key, read / write bottleneck downs,
the 4 + 4 read gates and the write gate concatenated) -> GELU + key-up GEMMs -> the **read kernel** ->
attention -> out projection -> LN2 -> one `h_w` GEMM -> memory-MLP output and MLP write-address GEMMs.
The write of layer i is deferred and fused into the read of layer i + 1 (decode) or run as its own
kernel (prefill), so M is touched once per layer.

- `h_w` = [main MLP fc_in | memory MLP fc_in | MLP write gate | MLP address down]: the main MLP, the
  memory MLP and the first projection of the address LoRA share one GEMM. This matters at decode bs1
  (one weight pass instead of four: ~13% of the step), little in prefill (compute bound). Their GELUs
  (exact for the MLPs, tanh for the bottleneck) differ per column range; decode applies them in the GEMV
  prologue, prefill leaves them to inductor (a hand-written epilogue would save ~1%).

## M read and write

- Algebra. The read view is the previous-token M-shift blend followed by whole-M RMS over 80 x 64. The
  RMS is a scalar per token, so it is applied to the read outputs, not to M. The four arms' dynamic and
  static reads fold into one key per row,
  `R = 0.1 sigmoid(g) rms(up) + (1.1 / 0.99) sigmoid(h) S`, so all reads are one matmul `R @ M~^T`. The
  attention write and the MLP write are one matmul `[K, 2N] @ [2N, V]`.
- Read kernel: per token, one program per arm group ((q, k) / (v, local o)). Keys are computed before
  the M work (prefill -8%). Q/K get QKNorm + RoPE on the projected tail + 1/sqrt(100); decode also writes
  K into the tail buffer and V into the cache. Prefill: ~0.75 ms per layer at 16k tokens (bandwidth
  floor ~0.52).
- Write: separate kernel in prefill (fused into the read it needs 255 registers and spills: 3.2 vs
  1.4 ms); fused into the read in decode, where the previous token's M comes from the cache.
- Decode config per batch: rows <= 16 use one arm group per program and 4 (rows <= 4) or 2 programs per
  token; bs64 puts both groups in one program (M written once). Read kernel per layer
  12.1 / 14.4 / 38 us -> 9.2 / 10.1 / 13.6 us at bs1 / 16 / 64.

## Attention and caches

- Strict causality: no token attends to itself (MaxText negative-diagonal attention). Prefill: query i
  attends to keys j < i, the first query returns 0. Decode: the new query attends to cache positions
  < pos; its own K/V is appended for later tokens.
- Prefill attention: Triton flash attention with Q/K loaded as 64 + 32 + 16 column blocks and V as
  64 + 16, local O added in the epilogue. Q/K are stored 112 wide (200-byte rows do not vectorize).
  Faster than SDPA flash padded to 128.
- Decode K cache ('kt'): transposed 64-position blocks `[B, N, S/64, 100, 64]` (coalesced reads), plus a
  double-buffered rows-layout tail `[B, N, 2, 64, 100]` holding the last two blocks. The read kernel
  writes the new key contiguously into slot `(pos // 64) % 2`; at step r of block j it moves head r of
  block j - 1 into kt (amortized over the block). Scattered single-position writes into kt cost
  7-15 us per layer; moving a whole block in one step is a latency spike.
- Decode attention runs a constant-trip-count main loop masked to the full kt blocks
  (`lim = max(cb - 1, 0) * 64`), then the two tail tiles. A `tl.where` pointer select, a runtime branch
  or a runtime loop bound inside the loop kills the software pipelining (bs16 +30%).
- Pythia keeps the rows layout (fastest for it in the full step; kt for Dq 80 is no faster).
- Static per-layer buffers; `attend_and_append` / `fused_read` do not declare their cache mutation,
  otherwise inductor clones and copies back whole caches. Sound because only that op touches a layer's
  cache within a step and reads are < pos.

## Where the remaining gap is

Decode, per step (attention microbenchmark with CUDA graphs, 32 layers, context 1985):

| | Pythia rows Dq 80 | BAM kt + tail Dq 100 | kt + tail Dq 80 | rows Dq 100 | BAM - Pythia, step |
|---|---:|---:|---:|---:|---:|
| bs1 | 0.45 ms | 0.61 ms | 0.52 ms | 0.60 ms | 0.71 ms |
| bs16 | 7.14 ms | 7.41 ms | 7.08 ms | 9.49 ms | 1.55 ms |
| bs64 | 25.2 ms | 27.8 ms | 26.4 ms | 31.6 ms | 4.04 ms |

- The extra attention cost is bytes, not FLOPs: K width 100 vs 80 with V 80 in both is +12.5% K/V per
  position. One query per head gives ~1 flop per byte of K, far below A800's ~150 flop/byte ridge, and
  BAM's attention runs at the same or higher bandwidth than Pythia's (1581 vs 1458 GB/s at bs16,
  1684 vs 1652 at bs64).
- Attention is ~23% / 17% / 65% of the step gap at bs1 / 16 / 64. The kt layout absorbs most of the
  width cost at bs16 (rows Dq 100 would add 2.3 ms). At large batch and long context the ratio tends to
  ~1 / 1.125 unless K is narrowed or the K cache quantized.
- The rest of the bs1 / bs16 gap is BAM-only work: small GEMMs / GEMVs (~0.85-0.9 ms), the read kernel
  (0.36-0.42 ms), extra GELU and LayerNorm passes (~0.2-0.3 ms), minus BAM's narrower shared MLP
  (parameter matching).
- Prefill: read + write kernels (~38 ms at 8 x 2048, floor ~28 ms) and the lower efficiency of BAM's
  small-K GEMMs (~9 ms) at equal FLOPs. BAM's two LayerNorms per layer run near bandwidth and are inherent
  to its serial residual.
- Other GPUs (prediction, before measurement): A100 about the same ratio; H100 prefill lower (more FLOPs
  per byte makes BAM's bandwidth-bound kernels a larger share). RTX PRO 6000 Blackwell measured
  0.888 / 0.876 prefill, 0.899 / 0.963 / 0.880 decode (kernels tuned on A800).

## Negative results

- Fusing the write into the prefill read, also with recomputing the previous token's M (register
  spills: 3.2 / 1.50 vs 1.17-1.4 ms).
- M write as extra programs inside a GEMV launch (raises the GEMV's registers); k-split write programs.
- Splitting the prefill read over more programs (SUB); TOK / warps sweeps beyond the defaults.
- An activation inside a Triton GEMM prologue (breaks the operand pipeline: 5x slower); hence one GELU
  pass + a grouped small-M GEMM for multi-token decode.
- Block-diagonal memory-MLP / address GEMM; a separate GEMM for the exact-GELU columns of `h_w`
  (inductor still runs the GELU as its own kernel); splitting `h_w` in prefill.
- 128-wide masked Q/K tiles in the prefill attention; chunked (64 + 32 + 4) decode K loads; KT blocks
  smaller than 64; padding Pythia's heads to 96 for flash (slower in the graph).
- Shared-linear GEMV (`SHARED_GEMV=1`): faster for both, worse ratio (see policy).

## Measurement notes

- Run one benchmark process per machine at a time: two concurrent compiles on different GPUs crashed a
  CUDA graph capture (`cudaErrorStreamCaptureInvalidated`); the same run alone passes.
- `max-autotune` sometimes picks slower GEMMs for Pythia in one process (bs16 12.55 vs 11.1-11.3 ms,
  bs64 35.1 vs 30.9 ms); BAM's timings were stable to <1%. Report decode as the median of several runs
  alternating BAM and Pythia processes.
- Check `nvidia-smi --query-compute-apps` right before timing; another user's training on the GPU makes
  timings vary by 30-50%. torch.profiler kernel totals are inflated relative to the CUDA-graph step time;
  use them for breakdowns only.
