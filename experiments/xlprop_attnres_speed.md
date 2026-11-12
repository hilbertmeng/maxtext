# XLProp TruePile Full AttnRes training-speed optimization

Date 2026-11-12. Config `Llama2XLPropTruePileMHAAttnRes` (28 layers, D1920, 20x96 heads, T4096,
8/device, Splash1024, full growing depth history). Baseline `Llama2XLPropTruePileMHA`.
Worktree `/home/xd/projects/maxtext_mqy/xlprop-hc-speed`, branch `xd/xlprop-hc-speed`, merged
into `mqy/refactor-bam-hc`. Diagnostic TPUs `xd-v5p-8-ar*`, `xd-v5p-32-ar-*`, all released.
Companion report for mHC/DHC: [xlprop_hc_speed.md](xlprop_hc_speed.md).

## Conclusion

Math is unchanged; only the TPU implementation of the depth reads changed.

| v5p-32 UC1a, steps/s (median) | AttnRes | vs MHA formal 0.550 | vs MHA Splash1024+minimal 0.633 |
|---|---:|---:|---:|
| XLA reads (as configured before) | 0.453 | -17.6% | -28.4% |
| **Pallas reads + saved read outputs (final)** | **0.520** | **-5.5%** | **-17.9%** |

v5p-8: 0.510 -> 0.636 (+24.7%): -19.4% vs MHA Splash1024+minimal 0.789, **+3.1% vs MHA formal 0.617**.

## Theoretical overhead

57 reads (attention and FFN of every layer plus the output) cover 1653 (read, source) pairs per
token; read l touches every earlier source, so traffic is quadratic in depth. FLOPs are ~5D per
pair, ~0.2% of the model. Each pair moves at least one full source slab (126 MB/device on v5p-8):

| Implementation floor | Slab passes / pair | ~GB/step | ~ms @ 2.2 TB/s | vs 1260 ms MHA step |
|---|---:|---:|---:|---:|
| per-read kernels, remat re-runs reads | 1 fwd + 1 remat + 3 bwd | 1040 | 470 | ~37% |
| + saved read outputs (final design) | 1 fwd + 3 bwd | 830 | 380 | ~30% |
| lazy reverse (not implemented) | ~1 + 2 | ~620 | ~280 | ~22% |

Measured final overhead vs equally configured MHA is 19% (v5p-8) / 18% (v5p-32), below the
per-read estimate because the read kernels overlap with other work and run at ~2.2-2.3 TB/s.
Reaching the lazy-reverse floor would need the reverse of all later reads of a source to be
evaluated at that source (one dout read per pair), which autodiff's per-read cotangents do not allow.

## What cost time and what fixed it

| Step | v5p-8 | v5p-32 | Cause / fix |
|---|---:|---:|---|
| XLA reads | 0.510 | 0.453 | per pair: separate RMS/score/sum passes forward; backward adds one full cotangent into every earlier source (read-modify-write per pair) at ~1.1 TB/s |
| Pallas reads, stacked [K,B,T,D] accumulator | 0.594 | 0.433 | one-pass read and reverse kernels; on v5p-32 XLA copied the whole 7 GB accumulator before ~42 in-place updates/step (255 ms) |
| per-source accumulator tuple | 0.590 | 0.466 | no stacked copies; ~840 per-slab transposes remain on v5p-32 (XLA laid sources out T-minor `{1,2,0}`, Pallas needs row-major) |
| row-major pin per shard | 0.592 | 0.486 | copies 175 -> 89 ms/step; a global `with_layout_constraint` dropped the sharding (0.20 step/s) and was replaced by a per-shard pin inside `shard_map` |
| + `minimal_save_attnres` (final) | **0.636** | **0.520** | read outputs saved instead of re-running each read under remat (+7.4% / +7.2%) |

Read kernels run at the HBM roofline (v5p-8 XPlane: forward ~99 GB in 46 ms, reverse ~296 GB
in ~128 ms). Accumulator aliasing was dropped: the kernel reads and writes each accumulator
either way, and inside remat XLA copied non-donatable aliased inputs first.

## Implementation (`MaxText/layers/attnres_pallas.py`, flag `attnres_pallas`)

- `read(G, sources, u)`: per token tile, flat-RMS keys `rsig_k * (x_k . u)`, softmax and weighted
  sum in one pass over the K sources (VMEM-resident tiles, 16 tokens for K > 24).
- Source gradients live in per-source accumulators `G = (G_0..G_56)` threaded through the stack;
  reads see sources through `stop_gradient`, their reverse adds into `G[k]` in one pass (read
  source + read/write accumulator) and accumulates the query/key-scale gradient per core;
  `register(G, x_k, k)` hands `G[k]` back as `x_k`'s cotangent.
- `remat_policy='minimal_save_attnres'` = minimal plus the named read outputs (`attnres_read`).
- `attnres_pallas_interpret` for CPU tests only (AOT compiles under `JAX_PLATFORMS=cpu`).

## Correctness

- `residual_connections_test.test_attnres_pallas_matches_default`: logits and every parameter
  gradient match the XLA path (no remat, minimal, minimal_save_attnres; kernels interpreted).
- `hc_pallas_shard_test`: four host devices with the real `shard_map` wrappers.
- Real training from the same initialization (losses at steps 1/10/30), v5p-8: XLA
  10.857944 / 10.401771 / 9.753161, Pallas 10.857934 / 10.401763 / 9.753195, + saved reads
  10.857936 / 10.401773 / 9.753184; v5p-32 Pallas vs + saved reads agree to 4e-6.

## Measurements

Profile classes `Llama2XLPropTruePileMHAAttnRes{,Pallas,PallasSaveRead}SpeedProfile` (exp.py).
Commits c6b1fd8 (baseline arms), 44a63b2 / 14fcf4f / c7615ad / d7956ec / 6018d2f (kernel
iterations), 381a532 (final). Paired matrices per VM, steps 16-99 median; v5p-32 means are
lower than medians because of a few slow steps per run (all arms alike).
Artifacts: `/data0/xd/bam_diagnostics/xlprop-hc-speed-20261111/` (orch logs `orch-ar*.log`,
XPlanes `xp/ar*`), GCS `.../diagnostics/profile_matrix/{c6b1fd8,44a63b2,14fcf4f,c7615ad,d7956ec,6018d2f,381a532}/`.
