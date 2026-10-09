# Fused Pallas core for AllLocal DirectC10 BAM（v5p）及 BAM Splash attention

2026-10-10 合入主分支 `refactor-bam`。本文前半部分是最终设计、用法、测速汇总和核心 kernel 的前传／反传伪代码；
后半部分「实验记录」按时间保留全部 profile、bundle 分析和被否决的路线（英文原始记录）。

- 实现分支：`claude/bam-pallas-directc10`（worktree `.claude/worktrees/bam-pallas-directc10`，基于 DirectC10
  runtime `ca4491a`）。各 runtime hash 见下文和 `MaxText/exp.py` 台账。合入主干的是最终版：v-major
  `blocked`/`blocked2` 与 k-major `v7`/`v7u` kernel、融合输入投影、M passthrough、query scale 折叠、
  BAM Splash attention。v3–v6、`loop`、`kmajor` body、XLA layout constraint 和 k-major cotangent 选项只保留在实验分支。
- 目标模型：`BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile`（28×D1920，
  20×96 heads，M 96×40/C10，QK72 + RoPE24 拼接，static + dynamic LocalVO，R400 写，第 1/4/…/25 层 R384
  独立 MLP 写）。方程与参数完全不变。

## 用法

| 配置项 | 默认 | 含义 |
|---|---|---|
| `bam_pallas_core` | off | 用融合 kernel 替代 XLA 的 M 读写（要求 AllLocal DirectC10 配方，`_check_pallas_core` 列出全部前提） |
| `bam_pallas_body` | `'blocked'` | `'v7u'`：k-major 最终版（推荐）；`'v7'`：同上但反传 pass 用 `fori_loop`（实测更慢）；`'blocked'`/`'blocked2'`：v-major 旧版 |
| `bam_pallas_fused_inputs` | off | BAM 全部输入投影（RoPE Q/K、C10 key、W_R、4 个读门、P_loc_down、W_gw）拼成一次 dot |
| `bam_pallas_vmem_mib` | 编译器默认 | v5p 用 48（物理 64 MiB／TensorCore；16 MiB 只是默认 scoped 预算） |
| `bam_pallas_write_block` / `bam_pallas_read_block` | 4 | v-major body 的寄存器分块（v5p 最优：写 2、读 4） |
| `bam_splash_attention` | **on** | BAM attention core 用 Splash kernel 代替 C256 query-chunk einsum |
| `bam_splash_seq_minor` | `null` | Splash q/k/v 用 `SEQ_MINOR`；`null` 时纯 JAX 路径为 True、融合 kernel 路径为 False |

最优配置类：`...DirectC10PallasV7SplashTruePile`（v7u + fused inputs + Splash）。
v7 body 下 M 以 k-major `[B,K,V,T]` 跨层携带（embedding write 直接产出），与 v-major `[B,V,K,T]`
及 XLA 路径的 token-major `[B,T,K,V]` 不能互换 checkpoint 之外的中间状态；参数树完全相同，checkpoint 可互换。

**Splash 的启用条件**（`attentions.py::_bam_splash_enabled`，两处调用：纯 JAX `__call__` 和融合路径
`_pallas_core_call`）：`bam_splash_attention=True`、TPU 后端、序列长度为 128 的倍数、该层没有 BAM fetch
（`'full'` mode 需要 attention 概率 α，不能用 Splash）。窗口层用 `CausalMask & LocalMask((w-1, 0))`，与 C256 的
`source > target - w` 等价。不满足条件时自动回到 C256/dense einsum（CPU 单测、interpret 模式、decode）。
Splash 在 kernel 内以 FP32 累加 logits/softmax，而 C256 在 `float32_logits=False` 时 logits 为 BF16：
数学等价，舍入不同，打开后的新 RUN 与历史 runtime 不逐位可比（以 runtime hash 区分谱系）。
沿用 MHA 的 `sa_*` block 尺寸；q 已在进入前缩放（融合路径在 read kernel 内折叠 1/√d）。

## 测速汇总（同机配对，target JIT，MHA = Splash 20×96）

| 平台 | 配置 | step/s | MHA 的 % |
|---|---|---:|---:|
| v6e-1 B2 | 纯 JAX (C256) / 融合 `blocked` | 1.205 / 1.793 | 53.4 / 79.5 |
| v5p-16 B8 AOT `49d3784` | 纯 JAX / 融合 `blocked` | 0.371 / 0.509 | 62.6 / 85.8 |
| v5p-8 `7cc41b1`–`1fecf6a` | 融合 `blocked` / tuned `blocked2` | 0.538 / 0.542 | 87.5 / 88.2 |
| v5p-8 `896f2ad`–`7149ff2` | v7u / + fused inputs / + q-scale 折叠 | 0.549 / 0.555 / 0.556 | 89.4 / 90.4 / 90.6 |
| v5p-8 `a1f64f2`/`748bc53` | **v7u + fused inputs + Splash** | **0.572** | **93.2** |
| v5p-8 `748bc53` | 纯 JAX + Splash SEQ_MINOR（纯 JAX 默认） | 0.480 | 78.2（C256 为 59.9） |
| v5p-32 正式 RUN `efc1977` | v7u + fused inputs (C256) | 0.491 | 90.4（MHA 0.543） |

数值：kernel 对原 token-major 数学 FP32 值和全部梯度 < 1e-4、BF16 < 2e-2；tiny 全模型 FP32 loss 相同、梯度
相对误差 ≤ 8.6e-6。v5p-32 正式 RUN 与纯 JAX DirectC10 的 loss：step 0–50 差 ±2e-4，早期轨迹分叉 +0.0095@500，
2500–17000 共 30 个窗口均值 −0.00017（范围 [−0.0009, +0.0010]），即舍入噪声。

剩余差距（v5p-8，融合 + Splash 1719 ms vs MHA 1618 ms）：SwiGLU MLP +142 ms（参数退还后更宽，架构所致）、
融合 kernel 约 139 ms + XLA glue、M 的 scan carry 约 +17 ms；BAM 的 Splash kernel 522 ms 略低于 MHA 的 546 ms
（同形状；差异来自 scan 中 3 个位置里 2 个位置的前传快 14%，推测为并发 DMA/collective 争用较少，未验证）。

## 伪代码

以下省略 BF16 的逐条 cast、物理 padding 和分布式分片，只保留数学运算、融合边界、分块和保存／重算策略。
所有反传均为显式解析式；热路径没有 `jax.vjp`。共用子程序均在所在 Pallas kernel 内展开。

### 整层连接关系

```text
Forward (每层; 写层另有 MLP 组):
  raw                  = x @ [W_qrope | W_krope | W_lq_c8 | W_lk_c8 | W_R | W_lq_g | W_lk_g | W_R_g | W_lv_g
                              | P_loc_down | W_gw]            # 一次 dot (bam_pallas_fused_inputs)
  (qs, ks)             = RoPE(QKNorm(raw.q, raw.k))         # 标准 QK 24 维
  (Q, K, V, Oloc, Mp)  = ReadKernel(M_in, S', keys, gates, qs, ks)   # Mp = M_in（passthrough 输出）
  Y                    = Splash(Q, K, V) + Oloc             # Q 已含 1/√d
  out                  = W_O · Y ;  x' = x + out ;  m = MLP(x')
  M_out                = WriteKernel(Mp, groups = [(Y, raw.W_gw + b, P_loc_up(gelu(raw.P_loc_down))),
                                                   (m, mlp_gate, mlp_address)  # 仅 MLP 写层])

Backward:
  (δMp, δfactors)      = WriteKernel_Backward(δM_out)       # δMp = δM_out（恒等）
  δY ... δ(Q,K,V)      = Splash_Backward, W_O/MLP backward (XLA)
  δM_in                = ReadKernel_Backward(δQ, δK, δV, δOloc, δMp)   # 在 kernel 内加上 δMp，不再有 XLA add_any
```

M 有两个消费者（读、写），passthrough 让写路径的 δM 直接进入读反传 kernel 累加。
全模型 `remat_policy='full'`：反传前重算一次 ReadKernel 前传（写前传的输出在反传中不需要，被 XLA 消除）。

### 记号与执行约定

下面公式写单个 token；kernel 对一个 token tile（T=128，占满 lane）向量化执行，grid 为
`(batch, token_tile)`，两维都是 `parallel`（v5p megacore 可切分 tile）。

|记号|含义／形状|
|---|---|
|`M`|每 token 状态 `[K, V] = [96, 40]`；k 为 data（列读输出维），v 为 address|
|`N, C, Kq, R`|`20, 10, 72, 24`：heads、压缩秩、直接读的 QK 列数、RoPE 列数（`Kq + R = K`）|
|`NP`|`24`：每个 head 组 pad 到 8 的倍数，使各组起始于 sublane 边界|
|`S'`|静态读矩阵 `[4·NP + C, V] = [106, 40]` = `[S_q; S_k; S_v; S_o; P]`（P 为 C10 压缩）|
|`rq, rk, rr`|直接 Q/K key 与共享 LocalVO key，`[C, N]`；`lq, lk, lo, lv` 读门 logits `[N]`|
|`qs, ks`|RoPE 后的标准 QK 坐标 `[R, N]`|
|`σs(l)`|`s · sigmoid(l)`，`s = key_scale = 0.2`；`σs'(l) = s · σ(l)(1-σ(l))`|
|`D`|静态读的 cotangent，`[106]` 行／每 k|
|`ACC(θ, x)`|沿 tile 的 token 求和后写出 per-tile partial，kernel 外沿 tile 与 batch 求和|

v7 布局：M 在 HBM／VMEM 为 k-major `[K, V, T]`，`M_k = M[k]` 是一个 `[V, T]` slab；沿 lane 拼接若干 k 的
slab 是免费的，所以静态读、压缩、`δM = S'ᵀ D`、`δS' = D Mᵀ` 都是无 relayout 的 MXU dot。逐 token 的动态
运算在 k-major 的 `[N, T]` head slab 上进行（N 在 sublane）。q/k/v/o 及其 cotangent 在 HBM 中为 head-major
`[N, K, T]`（XLA attention 免拷贝消费的布局），kernel 内用 sublane-strided FP32 行读做 k-major↔head-major 转换。

### 片上共用子程序

```text
Nε(z):        a = rsqrt(mean(z²) + ε);  return a·z
NBε(z, u):    a = rsqrt(mean(z²) + ε);  return a·(u - z·mean(u·z)·a²)      # Nε 的反传

Keys(rq, lq, rk, lk, rr):               # ε = 1e-4，对 c 归一化
  nq, nk, nr = Nε(rq), Nε(rk), Nε(rr)    # 每个 head 沿 C=10
  kq = σs(lq)·nq ;  kk = σs(lk)·nk       # 广播到 c
  return kq, kk, nr, nq, nk
```

### Kernel 1：Read 前传

```text
ReadForward(M[K,V], S'[106,V], rq, lq, rk, lk, rr, lo, lv, qs, ks):     # 每 token tile
  kq, kk, nr = Keys(...)
  for k-block of 16:                                    # MXU，lane 拼接 16 个 M_k
    St[:, k] = S' · M_k                                 # [106] = 静态 q/k/v/o (4×NP) + 压缩 mc (C)
  mc[c, k] = St[4·NP + c, k]
  for k-block of kb=4:                                  # VPU，寄存器累加器
    for c in 1..C:
      for k in block:
        if k < Kq:  aq[k] += kq[c]·mc[c,k] ;  ak[k] += kk[c]·mc[c,k]     # [N] slab, mc 行广播
        y[k]  += nr[c]·mc[c,k]
    for k in block:
      Q[:,k] = (k < Kq) ? aq[k] + St[S_q rows, k] : qs[k-Kq]
      K[:,k] = (k < Kq) ? ak[k] + St[S_k rows, k] : ks[k-Kq]
      V[:,k] = σs(lv)·y[k] + St[S_v rows, k]
      O[:,k] = σs(lo)·y[k] + St[S_o rows, k]            # LocalO
  Q *= 1/√d                                              # attention scale 折叠进 kernel
  out[n] = gather_k(scratch[k·NP + n])                   # FP32 VMEM → head-major [N,K,T]
  return Q, K, V, O ;  Mp = M（passthrough，不读写 HBM）
```

### Kernel 2：Read 解析反传

```text
ReadBackward(M, S', keys..., δQ, δK, δV, δO, δMp):
  kq, kk, nr, nq, nk = Keys(...)
  mc[:, k] = P · M_k                                     # 只重算压缩行（MXU）
  ct[i][n,k] = δ{Q,K,V,O}[n,k] (FP32)；δQ *= 1/√d       # head-major 暂存，per-k 用 strided 行读取 [N] slab
  # Pass A（k 外层；Python 展开）: VO 前传状态、门梯度、D 的静态行
  for k:
    y = Σ_c nr[c]·mc[c,k]
    gV += δV_k ⊙ y ;  gO += δO_k ⊙ y
    δy_vo[k] = σs(lv)·δV_k + σs(lo)·δO_k
    D[S_q rows, k] = (k < Kq) ? δQ_k : 0 ;  D[S_k rows, k] = (k < Kq) ? δK_k : 0
    D[S_v rows, k] = δV_k ;  D[S_o rows, k] = δO_k
  δlv = σs'(lv)·gV ;  δlo = σs'(lo)·gO
  δqs[r] = δQ_{Kq+r} ;  δks[r] = δK_{Kq+r}
  # Pass B（每个 c；k 内层）: key 梯度在寄存器累加；压缩行的 cotangent 用 sublane 归约
  for c in 1..C:
    for k:
      δkq[c] += δQ_k·mc[c,k] (k<Kq) ;  δkk[c] += δK_k·mc[c,k] (k<Kq) ;  δnr[c] += δy_vo[k]·mc[c,k]
      D[4·NP + c, k] = Σ_n ( kq[c,n]·δQ_k[n] + kk[c,n]·δK_k[n] + nr[c,n]·δy_vo[k][n] )
  δlq = σs'(lq)·Σ_c δkq[c]·nq[c] ;  δrq = NBε(rq, σs(lq)·δkq)      # k 路同理
  δrr = NBε(rr, δnr)
  # 静态读反传：每 16 个 k 一个 MXU block
  for k-block of 16:
    δM_k = S'ᵀ · D_k + δMp_k                             # δMp 与 δM 输出 aliased
    ACC(S', D_k · M_kᵀ)                                  # per-tile partial
  return δM, δS', δrq, δlq, δrk, δlk, δrr, δlo, δlv, δqs, δks
```

### Kernel 3：Write 前传（attention 组与 MLP 组合并为一次）

```text
WriteForward(Mp[K,V], groups g = (X_g[N,K], ℓ_g[N], A_g[N,V])):      # 写层 40 个 head，其余 20 个
  for each head i of all groups:                          # ε = 1e-6
    C[i,k] = sigmoid(ℓ[i]) · Nε(X[i,:])[k]               # 沿 K 归一化（content／data）
    Â[i,v] = Nε(A[i,:])[v]                               # 沿 V 归一化（address）
  for k-block of kb=4:
    acc[k] = Σ_i C[i,k] · Â[i]                            # [V] slab，C 行广播；寄存器累加
    M_out[k] = Mp[k] + acc[k]                             # input_output_aliases，原地写
```

### Kernel 4：Write 解析反传

```text
WriteBackward(G = δM_out[K,V], groups):
  重算 C, Â（同前传）
  Gf[k] = G[k] (FP32) ;  G2[v] = gather_v(Gf)            # v-major [K] slab，strided 行读
  for head block of 8:  δÂ[i] = Σ_k C[i,k] · Gf[k]        # [V] 累加器，C 行广播
  for head block of 3:  δC[i] = Σ_v Â[i,v] · G2[v]        # [K] 累加器，无逐 (i,k) sublane 归约
  for each head i:
    δℓ[i] = σ(ℓ)(1-σ(ℓ)) · Σ_k δC[i,k]·Nε(X[i])[k]
    δX[i] = NBε(X[i], sigmoid(ℓ[i])·δC[i])
    δA[i] = NBε(A[i], δÂ[i])
  return δMp = G（恒等）, (δX, δℓ, δA) per group
```

### XLA glue

```text
FusedInputProjections(x, names):        # bam_pallas_fused_inputs
  W = concat_out([cast(kernel_n)·gradscale_n for n in names])   # 参数不变；W_R 的 kernel_gradient_scale 保留
  y = x · W ;  return split(y) + biases                          # 反传只写一次 δx

BamSplash(Q, K, V, segments, window):   # bam_splash_attention
  mask = Causal (& Local(window-1, 0))
  return shard_map_batch(vmap(make_splash_mha(mask, sa_* blocks)))(Q, K, V, segments)
```

## 实验记录（按时间，英文原始记录）

### Initial design (v-major `blocked`, superseded by v7)

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


### v6e-1 full-model paired profile (B2/device, T4096, synthetic, health: generic ON / BAM OFF)

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

### v5p-16 paired profile (runtime `49d3784`, AOT, UE5a spot, B8/device, health generic ON / BAM OFF)

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

### v5p kernel analysis without a v5p: target compiles and bundle counts

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

### v5p measured tuning (single-layer sweep + paired full model, `xd-v5p-8-pallastune-1009-ue5a`)

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

### v7: k-major end-to-end kernels + glue fusions (v5p-8, `xd-v5p-8-pallasv7b-1009-ew4b`, target JIT)

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
attention core +35 (XLA C256 QChunk vs Splash; both models use 20 heads × 96),
kernels ~139 + glue, scan carry of M +17. Kernels are near their practical floor; 92–93% needs the
attention core (Splash for d=96: measured −80 ms, see below) or the MLP width, not more core work.

### Formal XL run (v5p-32)

RUN `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10PallasV7TruePile`, runtime
`efc1977` (this branch/worktree), TPU `xd-v5p-32-2910131-maxtext` UE5a, 50k plan, loss windows 500,
compare_runs DirectC10TruePile (pure JAX, `ca4491a`, stopped 20,185) and Llama2XLPropTruePileMHA. Launched
2026-10-09 via launch_train_parallel (targeted CPU checks, AOT on llm-jax-v6e-1-0). Step 101: 0.494 step/s
(+45.3% vs DirectC10 0.340 with BAM health ON; 91.0% of MHA 0.543). Bet: 0.47–0.50 step/s; loss gap to
DirectC10 within ±.005 early, |mean| < .002 after 2k, no trend (>.01 drift would indicate a numerical bug).

### Attention core: layout pin vs Splash (v5p-8 `xd-v5p-8-layout-1009-ew4b`, target JIT, MHA = Splash, 0.614)

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
Splash per call (ms, same shapes 20×96, T4096): forward-in-backward MHA 22.37 vs BAM 22.39 (identical); first forward
MHA 22.39, BAM 22.39 at one scan position and 19.27 at the other two; dq 22.32/21.88; dkv 30.45/28.6. The BAM total
(522 vs 546 ms) is context (concurrent traffic hypothesis), not less work; untested.
