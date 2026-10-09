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

## v5p measured tuning (single-layer sweep + paired full model, `xd-v5p-8-pallastune-1009-ue5a`)

Single layer, v5p-8 one chip, B8 T4096, ms (`MaxText/tests/bam_pallas_benchmark.py`, read/write tile 128):

| Body, block | read F | read F+B | write attn F+B | write attn+MLP F+B |
|---|---:|---:|---:|---:|
| `blocked`, 4 (production) | 0.858 | 2.036 | 1.685 | 3.073 |
| `blocked`, 2 | 0.855 | 2.034 | 1.558 | 2.881 |
| `blocked2`, 1 | 1.393 | 1.974 | 1.571 | 2.910 |
| `blocked2`, 2 | 0.850 | 1.998 | **1.488** | **2.737** |
| `blocked2`, 3 | 0.853 | 1.974 | 1.649 | 2.782 |
| `blocked2`, 4 | 0.866 | 1.973 | 1.637 | 3.125 |
| v6 write reverse, 4 | 0.856 | 1.981 | 2.104 | 3.743 |
| `blocked2`, 4, parallel-tile reverse (`1fecf6a`) | 0.864 | **1.829** | — | — |

Tile 256 (read) does not fit 60 MiB scoped VMEM; write tile 256 is slower (2.009 / 4.207).
Parallel-tile reverse: per-tile `dsw` partials instead of an on-chip accumulator, grid
(`parallel`,`parallel`).

Paired full model, v5p-8 target JIT, same VM, step/s:

| Runtime | MHA | Pallas `blocked`/4 | Tuned (`blocked2` read 4, write 2) |
|---|---:|---:|---:|
| `7cc41b1` | 0.615 | 0.538 (87.5%) | 0.541 (88.0%) |
| `1fecf6a` (+ parallel-tile reverse) | 0.614 | — | 0.542 (88.2%) |

The single-layer gain of the parallel-tile reverse (−7% read F+B) is ≈+0.2% in the full model.
Block/body tuning is exhausted at ≈+0.8% total. Further kernel gains need removing the
`[V,K,T]→[V,K·T]` relayout and spills (bundle analysis above), not schedule knobs.

MXU block-diagonal prototype (`MaxText/tests/bam_mxu_write_proto.py`, token-major M,
`blockdiag(A_tᵀ)@stack(C_t)`): g=4 → 42 bundles/token vs ~57 for the VPU write forward; g=6 worse.
Construction (mask/select, conversions) and serial group dependence dominate; not adopted.
Contractions over K (dkey, dA) cannot be token-batched this way.

FLEX_START retained hosts `llm-jax-v6e-1-0/1-1` were suspended by the service on 2026-10-08
(created 2026-10-01; FLEX_START duration). Diagnostics now use spot `xd-v6e-1-bamdiag-*`.

## v7: k-major end-to-end kernels + glue fusions (v5p-8, `xd-v5p-8-pallasv7b-1009-ew4b`, target JIT)

Read-reverse ablation on the v-major `blocked2` body (local v5p cross-compile, bundles/tile): the
two reverse static dots cost 10.7k of 22.4k; Mosaic relayouts `d [90,96,128]→[90,12288]`
((16,128)→(1,256)→(16,128)) and reads M by 96 strided row loads. Any per-head [K,T] layout needs a
(j,k) sublane transpose for a j-contraction; dtype tricks (BF16 staging, FP32 dots) were worse.
`layers/bam_pallas_v7.py` (body `v7` rolled / `v7u` unrolled) therefore keeps M k-major
[B,K,V,T] and does all per-token math on k-major [N,T] head slabs, so `st = S·M_k`, `dM = Sᵀ·D`,
`dS = D·Mᵀ` are relayout-free MXU dots on lane-concatenated k blocks.

| Runtime | Arm | step/s | % MHA | Note |
|---|---|---:|---:|---|
| `3faeae9` | v7u, k-major q/k/v outputs | 0.446 | 72.6 | XLA attention on k-major operands +384 ms |
| `896f2ad` | tuned `blocked2` / v7u head-major I/O | 0.541 / 0.549 | 88.1 / 89.4 | pure kernels 150.4 → 139.0 ms |
| `8731faa` | v7u / + fused input projections | 0.548 / 0.555 | 89.3 / 90.4 | one dot for Q/K RoPE, C10 keys, W_R, gates, P_loc_down, W_gw |
| `7149ff2` | + 1/√d query scale folded into the read kernel | 0.556 | 90.6 | |
| `93aa1e6` | + read-reverse cotangents transposed k-major by XLA | 0.533 | 86.8 | layout propagates into attention backward |

MHA 0.614 in every matrix. Best: `BamDirectC10PallasV7UFusedCoreProfile` (`7149ff2`), step 1777 ms vs
MHA 1619 (`8731faa` trace, before the scale fold). Pure kernels (ms/step, tuned → v7u): read F 21.1 →
17.8, remat 20.7 → 17.8, read B 45.7 → 50.0, write F 17.8 → 15.4, write B 45.2 → 38.1; XLA dM
`add_any` 15.5 → 7.1 (M passthrough output of the read; the reverse adds the write-path dM).

Lessons:
- The consumer fixes the output layout. Kernel outputs feed XLA's attention; anything but the
  head-major [B,N,K,T] it consumes copy-free gets propagated into the attention dots (forward
  outputs: +380 ms; backward cotangents via an XLA transpose: −4%). Convert inside the kernel
  (strided FP32 row gathers through VMEM), even at +11 ms read-reverse cost.
- Rolled `fori_loop` passes again lost to unrolled code on hardware (read F+B 2.61 vs 1.67 ms per
  layer) despite fewer static bundles. Packed BF16 pair gathers via `ref.bitcast(uint32)` need an
  unsqueezed batch dim (`memref_bitcast` rank check) and spilled badly (32.9k bundles); dropped.
- BAM's dozen small input projections cost more as separate XLA dots (each reverse writes a full
  [B,T,D] dx) than as one concatenated dot (−19 ms/step at the same ~150 TFLOP/s as MHA's QKV).
- Write kernels are VALU-bound at ~1.3× the VPU floor (dA 6.2k, dC 7.5k of 15.2k bundles/tile);
  the per-token contractions have no shared MXU operand.

Remaining gap to MHA (`8731faa` trace, ms/step): SwiGLU MLP +142 (wider after the parameter refund),
attention core +35 (XLA C256 QChunk vs Splash, with fewer BAM attention FLOPs: 20×96 vs 16×128),
kernels ~139 + glue, scan carry of M +17. Kernels are near their practical floor; 92–93% needs the
attention core (Splash-class kernel for d=96, ideal ≈ −71 ms) or the MLP width, not more core work.

## Formal XL run (v5p-32)

RUN `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasV7TruePile`, runtime
`efc1977` (this branch/worktree), TPU `xd-v5p-32-2910131-maxtext` UE5a, 50k plan, loss windows 500,
compare_runs DirectC10TruePile (pure JAX, `ca4491a`, stopped 20,185) and Llama2XLPropTruePileMHA. Launched
2026-10-09 via launch_train_parallel (targeted CPU checks, AOT on llm-jax-v6e-1-0). Step 101: 0.494 step/s
(+45.3% vs DirectC10 0.340 with BAM health ON; 91.0% of MHA 0.543). Bet: 0.47–0.50 step/s; loss gap to
DirectC10 within ±.005 early, |mean| < .002 after 2k, no trend (>.01 drift would indicate a numerical bug).

## Attention core: layout pin vs Splash (v5p-8 `xd-v5p-8-layout-1009-ew4b`, target JIT, MHA = Splash, 0.614)

| Runtime | Arm | step/s | % MHA | Note |
|---|---|---:|---:|---|
| `b0d540b` | pure JAX (C256) | 0.368 | 59.9 | attention 986 ms vs 614 in Pallas arms: XLA layout propagation |
| `b0d540b` | + `with_layout_constraint` q/k/v B,N,K,T | 0.385 | 62.7 | attention → 641, but the constraint lowers to async copies (+398 ms, 48+95 GB/step) |
| `b0d540b` | + constraint B,T,N,K | 0.319 | 52.0 | |
| `a1f64f2` | Pallas v7u fused (C256) / + Splash | 0.554 / **0.572** | 90.2 / **93.2** | BAM Splash kernels 522 ms (MHA 546); +18 ms transposes around them |
| `a1f64f2` | pure JAX + Splash | 0.449 | 73.1 | |
| `748bc53` | Pallas + Splash, SEQ_MINOR q/k/v | 0.568 | 92.5 | copies 70→48 ms, Splash kernels 522→556 ms |
| `748bc53` | pure JAX + Splash, SEQ_MINOR | **0.480** | **78.2** | best pure-JAX path (+30% vs C256) |

Both MHA and BAM have q/k/v head_dim 96; MHA resolves `attention='autoselected'` → Splash. Splash accumulates
logits/softmax in FP32 (C256 keeps BF16 logits with float32_logits=False): same math, slightly different rounding.
Best overall: Pallas v7u + fused inputs + Splash (HEAD_DIM_MINOR), `BamDirectC10PallasV7USplashCoreProfile`.
For pure-JAX research variants: Splash with `bam_splash_seq_minor=True` (needs full-causal or LocalMask windows and
no fetch). Lesson: an XLA layout constraint is not a cheap relayout under SPMD; a fixed-layout custom call is.
