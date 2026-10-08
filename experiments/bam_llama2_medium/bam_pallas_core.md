# Fused Pallas core for AllLocal DirectC10 BAM

Worktree/branch: `/home/xd/projects/maxtext/.claude/worktrees/bam-pallas-directc10`,
`claude/bam-pallas-directc10` (base: DirectC10 runtime `ca4491a`). Main `exp.py` remains the ledger.
Target: `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile`
(28xD1920, 20x96, M96x40/C10, QK72+RoPE24 concat, static+dynamic LocalVO, R400 writes,
R384 independent MLP writes at layers1/4/.../25). Same equations and parameters; no architecture change.

## Design

Only two kernels touch M; everything else (projections, attention, MLP) stays in XLA.

- M carried token-minor `[B,V=40,K=96,T]` between layers (embedding write emits it directly).
- **Read** `M -> (Q,K,V,LocalO)`: one MXU dot `[S_q;S_k;S_v;S_o;P]ᵀ(90x40) @ M(40 x 96T)` gives the four
  static reads and the C10 compression; direct C10 Q/K and shared LocalVO reads are VPU contractions
  over C=10 with register-blocked head accumulators; RoPE'd standard QK24 is written into the same
  output heads (no XLA concatenate). Analytic reverse recomputes the forward state, accumulates
  `dS/dP` on chip per batch element, and builds `dM` with one MXU dot.
- **Write** after the MLP: `M + Σ_n A_n⊗C_n` over the attention group and, on MLP-write layers, the
  independent MLP group, in one pass. Content/address RMS norms and sigmoid gates are inside the kernel.
  Reverse: `dM_in = G` (identity); `dC`, `dA` register-blocked VPU contractions, then norm/gate reverses.
- Kernel bodies operate on VMEM refs with register-blocked accumulators; pure-jnp per-tile functions
  are kept as the reference.

Code: `MaxText/layers/bam_pallas.py`; wiring `attentions.py::_pallas_core_call`, `fusion.py`
(`bam_pallas_core`), `models.py` embedding write. Tests: `MaxText/tests/bam_pallas_test.py`
(kernel vs original token-major math: FP32 values + all gradients <1e-4 rel, bf16 <2e-2),
`MaxText/tests/bam_pallas_model_test.py` (tiny full model, scan+remat+pair scan+final layer+MLP/embedding
writes: identical FP32 loss, max grad rel err 8.6e-6).

## v6e-1 full-model paired profile (B2/device, T4096, synthetic, health: generic ON / BAM OFF)

Retained FLEX_START `llm-jax-v6e-1-0` (lock held), runtime `335cebf`, MHA at the same VM.
Trace-free steps20–39; XPlane steps10–14 (exclusive time, coverage ≥98.9%).

| Arm | step/s | vs MHA | device step ms |
|---|---:|---:|---:|
| `Llama2XLPropTruePileMHA` (e30c1b8) | 2.256 | — | 439.4 |
| `...DirectC10NoHealthTruePile` (pure JAX) | 1.205 | 53.4% | 819.8 |
| `...DirectC10PallasTruePile` | **1.793** | **79.5%** | **547.8** |

Pallas vs pure JAX +48.8% throughput. Copies 156.9→7.8 ms; C256 attention core 291.6→222.3 ms
(its excess was Q/K concat layout); kernels total 82.3 ms (read F 9.5, read B+remat 37.3, write F 11.4,
write B 24.2). Remaining gap to MHA 108 ms: kernels 82, MLP +32 (wider after parameter refund),
scan/other +14; BAM projections together are 10 ms cheaper than MHA's QKVO.

Microbench (one layer, B2): read F/F+B 0.29/0.68 ms vs XLA 1.66/2.14; write(attn) 0.40/0.68 vs
0.48/1.04; write(attn+MLP) 0.70/1.29 vs 0.90/2.69. Read reverse ≈41k vector ops per 128-token
tile ≈4.5 ops/cycle: near VALU-bound; register-accumulated dMc (`blocked2`) only −3%.

VMEM: v5p physical 64 MiB/TensorCore, v6e 128 MiB (jax `tpu_info.py`); 16/32 MiB are default scoped
compiler budgets. The v5p arm sets per-kernel `vmem_limit_bytes` to 48 MiB (`bam_pallas_vmem_mib`).

## v5p-16 paired profile

PENDING.
