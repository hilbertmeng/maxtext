# Two-Pass Global Flash-DCMHA (Pallas) vs QChunk & MHA: Training Throughput

Single source of truth for all-global $T = 4096$ query-wise Dynamic Composed Multi-Head Attention (`DC2`, `attention='dc_flash'`) on TPU v5p in MaxText (`MaxText/layers/dc_pallas.py`).

## 1. Kernel Design Summary

For query-wise `DC2` (`pre_compose=True, post_compose=True, query_wise=True, key_wise=False, static_proj=False`) with $G$ head groups (`dc_num_groups`), $M = H / G$ heads per group, and dynamic rank $R$ (`dc_dynamic_rank`):

$$\tilde{S}_{g,h,k,q} = (1 + d^{\text{pre}}_{g,h,q}) S_{g,h,k,q} + \sum_{r=1}^{R} w^{\text{pre}}_{2,g,r,h,q} \sum_{m=1}^{M} w^{\text{pre}}_{1,g,r,m,q} S_{g,m,k,q}, \qquad P = \operatorname{softmax}_{k \le q}(\tilde{S})$$

$$\tilde{P}_{g,h,k,q} = (1 + d^{\text{post}}_{g,h,q}) P_{g,h,k,q} + \sum_{r=1}^{R} w^{\text{post}}_{2,g,r,h,q} \sum_{m=1}^{M} w^{\text{post}}_{1,g,r,m,q} P_{g,m,k,q}, \qquad O_{g,h,:,q} = \sum_{k \le q} V_{g,h,:,k} \tilde{P}_{g,h,k,q}$$

Key implementation properties (`MaxText/layers/dc_pallas.py`):
1. **Unified 4D group-parallel grid `(B, G, N_q, ...)`**: handles arbitrary $(G, M, R)$ in one parameterized Pallas kernel. Setting $G = 4, M = 4, R = 1$ shrinks per-instance VMEM score tiles by $4\times$ and unlocks `1024x1024` (`compute=512`) blocks matching `SplashTunedBlocks1024` (vs `512x512/256` at $G = 1, M = 16, R = 2$).
2. **Fused single-kernel two-pass forward (`dc2_flash_fwd_fused`)**: Pass 1 streams $K$ only ($V$ DMA pinned to block 0) for exact online-softmax `LSE`; Pass 2 streams $(K, V)$ in the same kernel with stationary $Q$ preparation in VMEM, emitting $\bar{O} = V P$ and rank-$R$ summaries $G_r = V_{\text{flat}} g_r$ with zero $O(T^2)$ HBM traffic.
3. **Within-group $(M \cdot D)$-wide MXU channel factorization**: cross-head rank-$R$ contractions execute on the BF16 MXU via $K_{\text{flat}}^\top (w^{\text{pre}}_1 \odot Q)_{\text{flat}}$ and $V_{\text{flat}} g_r$ rather than VPU head loops.
4. **Exact $O(T)$ backward $\Delta$ decomposition (`dc2_flash_bwd_dw1` + `dc2_flash_bwd_fused_kv`)**: using saved $\bar{O}$,
   $$\Delta_{g,m,q} = \langle \bar{O}_{g,m,:,q}, d\bar{O}_{g,m,:,q} \rangle + \sum_{r=1}^{R} w^{\text{post}}_{1,g,r,m,q} \, dw^{\text{post}}_{1,g,r,m,q},$$
   eliminating the $M$-head $V^\top dO$ matmul and $80\%$ of VPU reductions from Backward Pass 1.
5. **Optional residual checkpointing (`full_save_splash`)**: setting `bam_splash_residual_name='bam_splash_residual'` saves $(\bar{O}, G_r, \text{LSE})$ across layer rematerialization, matching SplashAttention's `residual_checkpoint_name` contract.

Numerical parity (`MaxText/tests/dc_pallas_test.py`): IEEE FP32 (`Precision.HIGHEST`) max absolute error $\le 1.5 \times 10^{-6}$ (forward) and $\le 2.5 \times 10^{-5}$ (all gradients `dq, dk, dv, dpre, dpost`, with and without `SegmentIds`) for both $(G=1, R=2)$ and $(G=4, R=1)$.

---

## 2. Paired `v5p-8` Benchmark ($T = 4096$, All-24-Layer Global)

Backbone: `Llama2Medium` ($L = 24, D = 1024, H = 16, d_h = 64, d_{\text{ff}} = 2816$, `sliding_window_size=None`, `scan_layers=True`), `v5p-8` (4 chips FSDP, `per_device_batch_size=16`, `65,536` tokens/chip/step, global `262,144` tokens/step), Pile dataset (`truepile4096`), `record_training_health_metrics=True`.

### 2.1 Primary Paired Comparison (`qk_norm=True` Aligned Across All Arms, Commit `d08a683f`)

| Arm (`MaxText/exp.py`) | Groups / Rank | Kernel & Blocks | Remat Policy | Step/s | Device Step (ms) | Ratio vs Same-Policy MHA+QKNorm | Full Attention Sublayer (ms) | Pure Attention Kernel (ms) |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `OverheadMediumLlama2MHAQKNormSplashSpeedProfile` | — | `Splash` (`1024/1024/512`) | `full` | **.774** | 1284.0 | 100.0% | 911.5 | 443.7 |
| `OverheadMediumLlama2MHAQKNormSplashSaveResSpeedProfile` | — | `Splash` (`1024/1024/512`) | `full_save_splash` | **.845** | 1173.4 | 100.0% | 795.3 | 330.5 |
| `OverheadMediumLlama2DC2G4R1QChunkSpeedProfile` | $G=4, R=1$ | `QChunk=512` (`remat`) | `full` | **.206** | 4833.1 | 26.6% | 4441.2 | 4219.3 |
| `OverheadMediumLlama2DC2PallasSaveResSpeedProfile` | $G=1, R=2$ | `dc_flash` (`512/512/256`) | `full_save_splash` | **.626** | 1584.4 | 74.1% | 1197.5 | 889.9 |
| `OverheadMediumLlama2DC2G4R1PallasSpeedProfile` | $G=4, R=1$ | `dc_flash` (`1024/1024/512`) | `full` | **.635** | 1561.0 | **82.0%** | 1189.9 | 919.0 |
| `OverheadMediumLlama2DC2G4R1PallasSaveResSpeedProfile` | $G=4, R=1$ | `dc_flash` (`1024/1024/512`) | `full_save_splash` | **.753** | **1314.8** | **89.1%** | **931.5** | **663.0** |

### 2.2 Supplementary Controls (Ungrouped `QChunk` & `qk_norm=False` MHA, Commit `80465891`)

| Arm (`MaxText/exp.py`) | Groups / Rank | `qk_norm` | Kernel & Blocks | Remat Policy | Step/s | Device Step (ms) | Full Attention Sublayer (ms) | Pure Attention Kernel (ms) |
|---|---:|:---:|---|---|---:|---:|---:|---:|
| `OverheadMediumLlama2MHASplashSpeedProfile` | — | Off | `Splash` (`1024/1024/512`) | `full` | **.839** | 1182.2 | 819.3 | 443.7 |
| `OverheadMediumLlama2MHASplashSaveResSpeedProfile` | — | Off | `Splash` (`1024/1024/512`) | `full_save_splash` | **.919** | 1079.9 | 706.4 | 330.5 |
| `OverheadMediumLlama2DC2GlobalSpeedProfile` | $G=1, R=2$ | On | `QChunk=512` (`remat`) | `full` | **.302** | 3295.0 | 2915.5 | 2691.2 |
| `OverheadMediumLlama2DC2PallasSpeedProfile` | $G=1, R=2$ | On | `dc_flash` (`512/512/256`) | `full` | **.515** | 1926.4 | 1555.7 | 1239.4 |

### 2.3 Per-Call Pallas Sub-Kernel Breakdown (24 Layers Total, XPlane TPU:0)

| Sub-kernel (24 layers summed, ms/step) | Ungrouped $G=1, R=2$ (`full`) | Ungrouped $G=1, R=2$ (`SaveRes`) | Grouped $G=4, R=1$ (`full`) | Grouped $G=4, R=1$ (`SaveRes`) | Reference `Splash` (`full` / `SaveRes`) |
|---|---:|---:|---:|---:|---:|
| Forward (`dc2_flash_fwd_fused` / `splash_mha_fwd`) | $2 \times 330.2 = 660.5$ | 297.5 | $2 \times 256.0 = 512.0$ | **256.0** | $2 \times 113.8 = 227.6$ / 113.8 |
| Backward Pass 1 (`dc2_flash_bwd_dw1`) | 137.4 | 148.3 | 107.0 | **107.1** | — |
| Backward Pass 2 (`dc2_flash_bwd_fused_kv` / `splash_mha_dkv`) | 441.6 | 444.1 | 300.0 | **300.0** | 216.1 / 216.7 |
| **Total Attention Kernel (ms/step)** | **1239.4** | **889.9** | **919.0** | **663.0** | **443.7 / 330.5** |

Key takeaways:
- **Speedup over `QChunk`**:
  - Ungrouped $(G=1, R=2)$: end-to-end **$1.71\times$** (`full`) / **$2.07\times$** (`SaveRes`); pure attention kernel **$2.17\times$** (`full`) / **$3.02\times$** (`SaveRes`) over `OverheadMediumLlama2DC2GlobalSpeedProfile` (`.302 step/s`).
  - Grouped $(G=4, R=1)$: pure JAX `QChunk` regresses to `.206 step/s` (`4833.1 ms`) due to $M=4$ short-axis XLA padding, whereas the 4D Pallas kernel reaches **`.753 step/s` (`1314.8 ms`)** — **$3.66\times$** end-to-end speedup and **$6.36\times$** pure-kernel speedup over grouped `QChunk` (**$2.49\times$** end-to-end over ungrouped `QChunk`).
- **Overhead vs strictly matched `MHA + QKNorm`**:
  - Even with all 24 layers running global $T = 4096$ `Pre + Post` dynamic composition, grouped `DC2 (G=4, R=1)` achieves **$82.0\%$** (`full`) and **$89.1\%$** (`SaveRes`, `+141.4 ms` / `+12.0%` step time) of `MHA + QKNorm` training throughput.

---

## 3. Reproduction & Trace Archives

Unit test (CPU or TPU):

```bash
PYTHONPATH=MaxText python MaxText/tests/dc_pallas_test.py
```

Paired `v5p-8` AOT + XPlane benchmark (via `tpu-ag`):

```bash
# Request v5p-8 with start_standalone_tpu.sh ... install_xd_maxtext_jax081.sh d08a683f4fa28ff87d387c3b5978eb694a9fd505
DONE_STEP=30 TOPO=v5p-8 bash /data0/xd/bam_diagnostics/tools/profile_orch.sh \
  d08a683f4fa28ff87d387c3b5978eb694a9fd505 dc2g4r1 /data0/xd/bam_diagnostics/dc2-g4r1-20261128/repro "TPU_NAME=ZONE" \
  mha_qkn=OverheadMediumLlama2MHAQKNormSplashSpeedProfile \
  mha_qkn_saveres=OverheadMediumLlama2MHAQKNormSplashSaveResSpeedProfile \
  dc2_g4r1_pallas=OverheadMediumLlama2DC2G4R1PallasSpeedProfile \
  dc2_g4r1_pallas_saveres=OverheadMediumLlama2DC2G4R1PallasSaveResSpeedProfile \
  dc2_g1r2_pallas_saveres=OverheadMediumLlama2DC2PallasSaveResSpeedProfile \
  dc2_g4r1_qchunk=OverheadMediumLlama2DC2G4R1QChunkSpeedProfile
```

Archived XPlane traces and logs:
- Commit `d08a683f` (primary 6-arm `G=4, R=1` & `qk_norm=True` round): `/data0/xd/bam_diagnostics/dc2-g4r1-20261128/run-a1/`
- Commit `80465891` (ungrouped `G=1, R=2` & `qk_norm=False` controls): `/data0/xd/bam_diagnostics/dc2-pallas-20261127/fast-a1/`
