# XLProp TruePile mHC / DHC4Tanh training-speed optimization

Date 2026-11-11. Configs `Llama2XLPropTruePileMHAmHC` and `Llama2XLPropTruePileMHADHC4Tanh`
(28 layers, D1920, 20x96 heads, T4096, 8/device, four residual streams). Baseline
`Llama2XLPropTruePileMHA`. Worktree `/home/xd/projects/maxtext_mqy/xlprop-hc-speed`, branch
`xd/xlprop-hc-speed`, merged into `mqy/refactor-bam-hc`. Diagnostic TPUs: spot `xd-v5p-8-hc*`
and `xd-v5p-32-hcfin-*`, all released.

## Conclusion

The routing math is unchanged; only its TPU implementation changed. The final mHC/DHC
runtimes match or beat the formal MHA run, and are 11.6-12.8% slower than MHA with the same
Splash blocks and remat, which is the theoretical memory-traffic floor.

| v5p-32 UE5a, steps/s | Original | Final | Speedup | vs MHA formal 0.550 | vs MHA Splash1024+minimal 0.632 |
|---|---:|---:|---:|---:|---:|
| `Llama2XLPropTruePileMHAmHC` | 0.398 | **0.559** | +40.5% | **+1.6%** | -11.6% |
| `Llama2XLPropTruePileMHADHC4Tanh` | 0.386 | **0.551** | +42.7% | **+0.2%** | -12.8% |

Original overhead vs the formal MHA was -27.6% (mHC) and -29.8% (DHC). The remaining
-11.6% / -12.8% vs equally configured MHA matches the ~12% memory-traffic estimate below.

## Theoretical overhead (reference)

- FLOPs: the routers add < 1% per layer (mHC 0.86M, DHC 0.34M FLOPs/token/layer vs 104M), and
  the parameter-matched FFN shrink (5120 -> 5053 / 5115) pays them back. Compute is not the cost.
- Memory traffic: the residual state is 4xD. With each connector reading the state once and
  each update reading/writing it once, forward + remat + backward move ~100 full-state slabs
  per layer, ~350 GB/step per device (v5p-8, 8x4096 tokens). At a practical ~2.2 TB/s this is
  ~160 ms, i.e. ~12% over the 1260 ms MHA (Splash1024 + minimal) step; ~9% without remat of
  the state.

## Bottleneck (XPlane, v5p-8)

| Implementation | Extra vs MHA (same Splash/remat) | Where |
|---|---:|---|
| Original `[B,T,n,D]` XLA | ~600 ms/step | 1336 full-state ops/step, 694 GB at 1.1 TB/s; stream axis tiled as T(4,128); FP32 normalized-state copies |
| Stream-major XLA | 482 ms | update backward split into dot/reduce fusions (~2.4x bytes, ~250 ms); connectors ~250 ms incl. a 127 GB/s outer-axis reduce |
| + Pallas update + connector | 210 ms | Sinkhorn: ~10k tiny XLA ops/step (~74 ms) |
| + Pallas Sinkhorn | ~110 ms | at the byte floor |

XLA's own elementwise fusions (MHA included) top out at ~1.1-1.25 TB/s here, so the gain had to
come from fewer passes and Pallas kernels, not from fusion tweaks. Per-stream router matmuls and
a two-stage RMS reduction measured slower (mHC 0.512 vs 0.568) and were reverted.

## Changes

All behind config flags (default off; the formal classes enable them):

- `SplashTunedBlocks1024` and `remat_policy='minimal'` (memory fits at 8/device on v5p).
- `mhc_stream_major_layout`: state `[n,B,T,D]`, coefficients `[k,(n,)B,T]`; router norms folded
  into the router dot (`norm(x)@W = rsig*(x@(g*W) - mu*sum(g*W)) + b@W`). Same parameter tree
  and initialization.
- `mhc_pallas_update` (`layers/hc_pallas.py`): fused residual update; the custom VJP returns
  dstate, dbranch and the mixing/post gradients in one pass.
- `mhc_pallas_connector`: one-pass read (mHC: router dot, flat RMS, sigmoid pre, read; DHC:
  per-stream LayerNorm, tanh, read weight, read). The state is passed through to the update so
  its gradient is summed inside the reverse kernel; dW accumulates per TensorCore.
- `mhc_pallas_sinkhorn` (mHC): all 20 iterations in one kernel; the reverse recomputes the
  iterates in VMEM.
- `mhc_pallas_interpret`: CPU tests only. AOT compiles TPU executables under
  `JAX_PLATFORMS=cpu`, so interpret mode must never be inferred from the backend (doing so ran
  the interpreter inside real training at 0.13 step/s).

## v5p-8 ablation (steps/s, steps 16-99 median, one VM per matrix, MHA control in each)

| Arm | mHC | DHC |
|---|---:|---:|
| MHA formal (Splash 512, full remat) | 0.617 | 0.617 |
| MHA + Splash1024 + minimal | 0.789-0.794 | 0.789-0.794 |
| Original | 0.437 | 0.431 |
| + Splash1024 | 0.478 | 0.472 |
| + Splash1024 + minimal | 0.535 | 0.532 |
| + stream-major | 0.568-0.573 | 0.574-0.575 |
| + Pallas update | 0.643 | 0.700 |
| + Pallas connector | 0.671 | 0.704 |
| + Pallas Sinkhorn (final mHC) | 0.704 | - |
| Final vs MHA same settings | -11.3% | -11.3% |
| Final vs MHA formal | +14.1% | +14.1% |

Profile classes `Llama2XLPropTruePileMHA*SpeedProfile` pin each arm (exp.py). Commits:
bd1f705 (original arms), ead6a1e (stream-major), deb2e6f (update), bd166c9 (connectors),
861c2da (Sinkhorn, final kernels).

## Correctness

- CPU full-model test (`residual_connections_test.test_stream_major_layout_matches_default`):
  logits and every parameter gradient of stream-major, + Pallas update, + connector and
  + Sinkhorn match the original layout (no remat and minimal remat; kernels interpreted).
- `hc_pallas_shard_test`: four host devices with the real `shard_map` wrappers, including the
  cross-device reduction of replicated router gradients (relative gradient error <= 1.5e-6).
- Real 4-chip training from the same initialization: losses of all implementations agree to
  ~1e-4 (steps 1/10/30), e.g. mHC 10.858607 / 10.400096 / 9.751766 vs stream-major XLA
  10.858603 / 10.400547 / 9.752048.

## v5p-32 confirmation (formal topology)

Spot `xd-v5p-32-hcfin-ue5a` (us-east5-a), one VM for both matrices, AOT for v5p-32,
16 chips, global batch 128, steps 16-99 median.

| Arm | Profile class | Commit | step/s |
|---|---|---|---:|
| MHA formal (Splash 512, full) | `Llama2XLPropTruePileMHASpeedProfile` | 8e1a712 | 0.550 |
| MHA + Splash1024 + full | `Llama2XLPropTruePileMHASplashSpeedProfile` | 6752818 | 0.623 |
| MHA + Splash1024 + minimal | `Llama2XLPropTruePileMHASplashMinimalSpeedProfile` | 8e1a712 / 6752818 | 0.632 / 0.632 |
| mHC original | `Llama2XLPropTruePileMHAmHCSpeedProfile` | 8e1a712 | 0.398 |
| mHC + update + connector | `...mHCSplashMinimalPallasConnectorSpeedProfile` | 8e1a712 | 0.537 |
| mHC final, minimal | `...mHCSplashMinimalPallasSinkhornSpeedProfile` | 6752818 | **0.559** |
| mHC final, full remat | `...mHCSplashFullPallasSinkhornSpeedProfile` | 6752818 | 0.548 |
| DHC original | `Llama2XLPropTruePileMHADHC4TanhSpeedProfile` | 8e1a712 | 0.386 |
| DHC final, minimal | `...DHC4TanhSplashMinimalPallasConnectorSpeedProfile` | 8e1a712 | **0.551** |
| DHC final, full remat | `...DHC4TanhSplashFullPallasConnectorSpeedProfile` | 6752818 | 0.545 |

Minimal remat stays faster than full on v5p-32 for all three models, so the formal classes
use it. The MHA formal speed matches its historical UE5a run (~0.543).

Artifacts: `/data0/xd/bam_diagnostics/xlprop-hc-speed-20261111/` (orchestration logs, XPlane,
analysis tools `tools/xpscope.py`, `tools/xpdetail.py`); GCS
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/{bd1f705,ead6a1e,221cdbc,deb2e6f,bd166c9,8e1a712,861c2da}/`.
