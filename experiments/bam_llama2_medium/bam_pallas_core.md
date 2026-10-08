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

## v5p-16 paired profile (runtime `49d3784`, AOT, UE5a spot, B8/device, health generic ON / BAM OFF)

Classes `Llama2XLPropMHAPallasCoreProfile`, `BamDirectC10NoHealthPallasCoreProfile`,
`BamDirectC10PallasCoreProfile` (`bam_pallas_vmem_mib=48`). Same VM `xd-v5p-16-pallascore-1008-ue5a`,
`run_profile_matrix.sh`, steps to 45 (trace 10–14). Node/queue deleted after artifact pull.

| Arm | step/s | vs MHA | device step ms |
|---|---:|---:|---:|
| MHA (Splash) | 0.593 | — | 1676.1 |
| pure-JAX DirectC10 | 0.371 | 62.6% | 2669.8 |
| **Pallas core** | **0.509** | **85.8%** | **1940.6** |

Losses at steps 46–49 agree with pure JAX to ~4e-4. Remaining +264 ms vs MHA: kernels +223
(read F 32.8 / read B incl. remat 117.5 / write F 21.3 / write B 51.8), MLP +141 (wider after the
parameter refund; FLOPs predict ~+114), attention C256 vs Splash +31, scan/other +37; projections
−113 and QKNorm/RoPE −55 versus MHA. Artifacts `/data0/xd/bam_diagnostics/bam-pallas-core/v5p-pallascore/`.

## v5p kernel analysis without a v5p: target compiles and bundle counts

Kernels are cross-compiled for `v5p-16` with the local libtpu (`MaxText/tests/bam_pallas_compile.py`,
`LIBTPU_INIT_ARGS=--xla_jf_dump_to`); `bundle_stats.py` reads per-bundle slot utilization (v5p
capacities MXU4 XLU3 VALU4 VLOAD3 VSTORE1), `loop_bundles.py` loop body lengths (dynamic cost =
static + (trips−1)·body). Local libtpu 0.0.23 reproduces worker counts within ~5–9%.

Original (`blocked2`) bundles per 128-token tile: read F 9.3k, read B 20.6k, write F 7.4k,
write B 19.5k (attn) / 37.0k (attn+MLP). Arithmetic floor (no FMA on v5p, 2 VALU ops/MAC) ≈ 1/2–1/3
of that. Findings:

- Read F: ~60% of VALU ops are relayout (`[V,K,T]→[V,K·T]` reshape for the static MXU dot):
  7.8k selects, 4.5k unpacks, 3.9k packs per tile. Read B and write B are bound by spill stores
  (single store slot): 9.6k / 8.3k spill stores per tile.
- Mosaic constraints found: sublane-strided loads/stores need 32-bit data; strided stores cost one
  store op per row (one store slot), strided loads one load op per row (three slots); `fori_loop`
  unroll must be 1 or full; loop carries with non-multiple-of-8 sublanes (`[20,T]`) crashed libtpu
  (pad to 24). Long Python-unrolled bodies let the scheduler hoist loads and spill.
- Loop-structured, VALU-dense bodies reach the VALU floor (write F inner loop 49 bundles/k vs
  floor 50; write B dC/dA loops at floor). Short loop bodies with load→use chains are latency-bound
  (~2× floor) unless manually unrolled.

| Variant (bundles/tile, v5p) | read F | read B | write F | write B (attn / attn+MLP) |
|---|---:|---:|---:|---:|
| `blocked2` (measured 85.8%) | 9.3k | 20.6k | 7.4k | 19.5k / 37.0k |
| v3 (FP32 staging, strided stores, emulated BF16 rounding) | 12.6k | 28.6k | 15.4k | 18.0k / 34.4k |
| v4 (k-major M, unrolled) | 7.7k | 20.8k | 13.7k | 18.3k / 35.9k |
| v5 (k-major, loops) | ~11.4k dyn | ~20.1k dyn | ~8.0k dyn | — |
| **v6 write B (loops, v-major M)** | — | — | — | **~13.2k / ~26k dyn** |
| loop-structured read B (blocked2 math) | — | ~24.7k dyn | — | — |

Combination `bam_pallas_body='v6'` (blocked read, blocked2 read reverse, blocked2 write forward,
v6 write reverse) was predicted ≈ −11% kernel time. **v5p paired measurement refuted it** (runtime
`e2e82a8`, `xd-v5p-16-pallasv6-1008-ue5a`, target JIT, same VM):

| Arm | step/s | device ms | read F | read B | write F | write B |
|---|---:|---:|---:|---:|---:|---:|
| MHA | 0.592 | 1677.9 | — | — | — | — |
| Pallas blocked2 (+BF16 read-reverse dots) | 0.508 | 1943.3 | 33.3 | 114.8 | 21.3 | 51.8 |
| Pallas v6 | 0.505 | 1954.4 | 33.6 | 113.2 | 20.5 | **65.4** |

The loop-structured write reverse is 26% *slower*, although static bundles × trip counts predicted
−32%. Static schedules omit memory stalls and loop-boundary bubbles; for loop kernels they are not a
valid speed proxy (straight-line kernels matched better). Decisions now require measured kernel
time (v6e microbench, then v5p). Activation-dtype static dots in the read reverse: −1.6 ms.
Default remains blocked2. Artifacts `/data0/xd/bam_diagnostics/bam-pallas-core/v5p-pallasv6/`.

MXU block-diagonal prototype (`MaxText/tests/bam_mxu_write_proto.py`, token-major M,
`blockdiag(A_tᵀ)@stack(C_t)`): g=4 → 42 bundles/token vs ~57 for the VPU write forward; g=6 worse.
Construction (mask/select, conversions) and serial group dependence dominate; not adopted.
Contractions over K (dkey, dA) cannot be token-batched this way.

FLEX_START retained hosts `llm-jax-v6e-1-0/1-1` were suspended by the service on 2026-10-08
(created 2026-10-01; FLEX_START duration). Diagnostics now use spot `xd-v6e-1-bamdiag-*`.
