# Pythia 2.8B unit H: faster read/attention implementation (same math)

Production class `BamPythia2p8BHUnit300BFastKernel` = `BamPythia2p8BHUnit300B` with three implementation
changes. The math, initialization and parameter/optimizer trees are unchanged, so its checkpoints are
interchangeable with `BamPythia2p8BHUnit300B`.

| Change | Config | What it does |
|---|---|---|
| Packed read keys | `bam_pallas_packed_read_keys` | One einsum `btd,drc->bcrt` writes the full-M read keys straight into v8t's `[B,C,S*N,T]` layout (S=4: q,k,v,o; S=3: q,k,shared vo). Pre-biases are a constant-one contraction row. The kernel reads Q/K/VO through block views; the reverse writes one key cotangent block. This replaces the split/reshape/concat/transpose copies around `_fused_input_projections`. |
| Fused full-head QKNorm | `bam_pallas_fused_qk_norm` (+ `bam_pallas_vmem_mib=63`) | The read epilogue normalizes complete Q/K rows and applies the norm scales plus the Q logit scale. Row statistics and per-head selections are MXU dots against 0/1 matrices (bf16 hi/lo split), so they arrive lane-broadcast. The reverse uses saved xhat and 1/r; xhat is single-buffered for VMEM. |
| Splash residuals | `bam_splash_residual_name`, `remat_policy=minimal_save_splash` | Splash's own `residual_checkpoint_name` tags its output and logsumexp. The policy keeps them instead of re-running the Splash forward inside the backward. |

## Speed (v5p-8, 8/device, 32 layers, minimal remat unless noted, generic health ON / internal OFF)

All entries in one row come from the same commit and the same TPU. Step rate is the reciprocal of the mean
step time over the profiled run.

| Comparison | step/s |
|---|---|
| Unit H implementation → packed + fused (minimal) | .6356 → .6780 (+6.7%) |
| + Splash residuals | .6958 (+9.5% vs. unit H) |
| Old long-training implementation (`R768` read, standard-arm QKNorm) | .6729 |
| Same-remat H vs. old implementation | +0.3% to +0.8% |
| Fair: old architecture also packed (`[B,C,3N,T]`) | H .6780 vs. .6901 (−1.75%); with Splash residuals .6958 vs. .7097 (−1.96%) |

The remaining fair gap is about 24 ms/step:
- The complete-head QKNorm epilogue is VPU-throughput bound, about 55 cycles per 128-token head slab.
- H writes the memory MLP every layer, which costs about 40 ms in the write kernel and the MLP-write factors. The smaller MLP repays it.

Training predictions scale from v5p-8 and are not yet measured on the formal v5p-256 run.

## Tried and rejected

- Splash block sizes (49 configurations), read chunk and tiles, write chunk and tiles: the current settings
  are best or tied. A 256-token read tile exceeds VMEM.
- Single-layer vs. 3-layer-block scan: tied.
- Fused QKNorm with lane reductions: first version −2.0%, later tied. Per-head XLU transposes, T-minor statistics, and HIGHEST-precision or 8-head-batched MXU statistics were all slower.
- Saving the read outputs or the packed keys in addition to Splash: slower (scan residual stacking copies).
- `everything` remat: 164 GB of HBM on v5p-8; not feasible on v5p-256 either.
- Key pre-bias inside the read kernel: exceeded VMEM by 5 MB.
- Gradient reduce-scatter constraint and the older async-collective flags: no HLO change. The TPU backend
  already lowers these collectives this way, and the libtpu default already enables async collective fusion.

## Reproduction

- Exploration branch and worktree: `claude/pythia-h-speed`, `/data0/xd/pythia-h-speed`. It contains the profile classes, read/Splash microbenchmarks and ablation switches.
- Profiles: `/data0/xd/bam_diagnostics/pythia2p8-h-speed-*-20261102/`.
- Category breakdowns: `catsum.py`; per-op diffs: the scratch `topops.py`.
- Correctness: `MaxText/tests/bam_fast_kernel_test.py` checks:
  - native vs. Pallas logits and every F32 gradient;
  - BF16 vs. F32 truth (fused 0.0194 vs. native 0.0358 global L2);
  - the shared-VO packed path;
  - parameter-tree identity with `BamPythia2p8BHUnit300B`.
