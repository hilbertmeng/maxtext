# XLProp full-M generalized column read：理论 FLOPs、实测速度与 profile（为 Pallas fused kernel 做准备）

2026-10-11。目标：`BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile`
（runtime `b83223f`，`codex/mediumprop-full-m-read-gelu128`；纯 JAX + Splash，SEQ_MINOR 自动开启），
基线 `Llama2XLPropTruePileMHA`（Splash，20×96 heads）。profile 分支 `claude/fullm-general-profile`（`a9a7a15` =
`b83223f` + 两个 profile 类），TPU `xd-v5p-8-fullm-1011-ew4b`（已释放）。MHA 的 v5p-8 trace 复用
`bam-pallas-core/v5p8-splash2/mha-ops.json`（同型号机器，6 个矩阵均为 0.614 step/s）。
数据：`/data0/xd/bam_diagnostics/fullm-general-read/`；分析脚本 `bam_pallas_core/table_fullm.py`。

## 结构（每层，每 token）

- 读 key：共享 GELU 瓶颈 `x[1920] → 400 → gelu`，各臂独立 up：Q、K、VO 各 `400 → N×V = 20×40`，加
  general pre-bias。另有 4 个读门（lq, lk, R, lv）与 4 个 general static 门（q,k,v,o），均为 `D → N`。
- 读：无 C10 压缩，Q/K/VO 动态读直接收缩整张 M：`col[n,k] = Σ_v key[n,v] M[k,v]`（Q/K 计算 96 列后截 72）；
  VO 为一次未门控读，再分别乘 v、o 门。静态读 `M · S_{q,k,v,o}`（q/k 72 列），乘 general static 门。
- 写：attention 写 `M += Σ_n C[n,k] A[n,v]`（R400 地址瓶颈），第 1/4/…/25 层另有 R384 独立 MLP 写；embedding 写。

## 理论 FLOPs（参数对齐：BAM 1,432,386,360 vs MHA 1,432,398,720）

每 token、28 层前传（MAC×2）：

| 项 | BAM | MHA |
|---|---:|---:|
| token 侧 matmul（2×参数）：MLP、投影、门、LM head | 2,670.5 M（参数 1,335.3 M） | 2,670.9 M（1,335.5 M） |
| 其中 attention 投影 / MLP | 投影 ~252 M 参数，MLP 980.5 M | QKVO 412.9 M，MLP 825.8 M |
| 因果 attention（N·(d_qk+d_v)·T/2，T=4096） | 440.4 M | 440.4 M |
| BAM M 运算（静态读 268.8k + 动态 Q/K 153.6k + VO 76.8k + 写 76.8k + MLP 写均摊 24.7k MAC/层） | 33.6 M | — |
| **前传合计** | **3,144.5 M** | **3,111.3 M** |
| 训练（×3）/ 全重算执行（×4） | 9.43 G / 12.58 G | 9.33 G / 12.45 G |

理论上 BAM 只多 **+1.1%**。XPlane 实际执行（不含 Splash 自定义调用的 FLOPs）：BAM 166.6 TF/step/device，
MHA 163.1（+2.2%，含逐元素运算）。M 运算只占 BAM FLOPs 的约 1%。

## 实测速度

| 平台 | 配置 | step/s | MHA 的 % |
|---|---|---:|---:|
| v5p-32 正式 RUN（另一 session，`xd-v5p-32-310118-maxtext`） | health ON | 0.3826 (20–99) / 0.379 @10.7k | 70.5 |
| v5p-8 配对 `a9a7a15` | health ON（与正式训练一致） | 0.439 | 71.5 |
| v5p-8 配对 `a9a7a15` | BAM health OFF | 0.455 | 74.1 |

BAM health（concat + general 统计）成本：日志 step/s −3.6%，device 时间 +41 ms（2204.5 vs 2163.4 ms，+1.9%；
其余为 host 侧统计传输）。

## Profile（v5p-8，device ms/step；BAM health OFF 列附实测吞吐）

| 部分 | MHA | BAM health ON | BAM health OFF | TF/s | GB/s |
|---|---:|---:|---:|---:|---:|
| BAM health statistics (sow) | 0.0 | 30.8 | 0.0 | | |
| Attention core (Splash) | 579.1 | 553.1 | 551.6 | | |
| SwiGLU MLP | 504.6 | 615.1 | 616.5 | 197 | 198 |
| Standard Q/K(/V) projections | 280.0 | 40.3 | 39.6 | 171 | 455 |
| QKNorm + RoPE | 50.1 | 4.9 | 4.9 | | |
| O projection | 88.7 | 97.6 | 96.8 | 140 | 179 |
| Full-M key bottleneck + up projections | 0.0 | 51.8 | 51.5 | 124 | 687 |
| Read/static gate projections（9 个 D→N） | 0.0 | 66.6 | 66.5 | 17 | 867 |
| Static Q/K/V/O reads | 0.0 | 126.8 | 122.2 | 8 | 1087 |
| LocalQK full-M dynamic read | 0.0 | 149.1 | 148.4 | 3 | 757 |
| LocalVO full-M dynamic read + gating | 0.0 | 80.4 | 78.2 | 4 | 674 |
| Attention write (factors + outer product) | 0.0 | 98.0 | 107.9 | 39 | 717 |
| MLP write (projections + merge) | 0.0 | 55.5 | 50.6 | 27 | 729 |
| Embedding write | 0.0 | 7.6 | 7.6 | 63 | 337 |
| Layer norms | 24.3 | 25.9 | 25.9 | | |
| LM head / loss | 56.1 | 55.8 | 55.8 | 171 | 256 |
| Scan carry / optimizer / other | 34.9 | 141.0 | 138.0 | | |
| **Device step** | **1617.9** | **2204.5** | **2163.4** | | |
| ↳ copy kernels | 12.7 | 198.2 | 184.4 | | |
| HBM traffic / step | 293 GB | 860 GB | 853 GB | | |

## 差距归因（BAM health OFF − MHA = +545 ms）

| 来源 | Δ ms | 性质 |
|---|---:|---|
| BAM M 运算（静态读 122 + 动态 Q/K 148 + VO 78 + attention 写 108 + MLP 写 51 + embedding 写 8） | +515 | 全部 memory-bound：3–40 TF/s、0.7–1.1 TB/s；每个 op 单独读写 M 及 [N,K] 中间量 |
| scan carry / copies / 其他 glue | +103 | 拷贝 kernel 184 ms（MHA 13） |
| SwiGLU MLP 更宽（参数退还） | +112 | 架构 |
| 投影：BAM 标准 Q/K + QKNorm + O + full-M key + 门 259 ms vs MHA QKVO + QKNorm 419 ms | −160 | 其中 9 个门投影 66.5 ms 只有 17 TF/s |
| Splash attention | −28 | BAM 略快 |

HBM 流量是根本：BAM 每步 853 GB，MHA 293 GB（2.9×）；多出的 ~560 GB 以约 1 TB/s 的实际带宽流过，
正好对应 ~550 ms 的差距。

## 对 Pallas fused kernel 的含义与下注

1. **可融合目标约 515 ms + 部分 glue（copy/scan 约 100 ms）**。一个 read kernel（静态读 + 三个 full-M 动态读 +
   门/偏置）与一个 write kernel（attention + MLP 组合并）让 M 每层只读写一次，中间量留在 VMEM。
2. **与 DirectC10 的差别：动态读的 VPU 工作量约 4×。** DirectC10 每臂只收缩 C=10（19.2k MAC），这里收缩
   V=40（76.8k MAC/臂，Q/K/VO 三臂）。v7 k-major 形式下即每 k 对 40 个 v 广播 `M_k[v]` 行乘 `[N,T]` key slab，
   前传 ~69k VALU op/tile，反传约 2×（δkey 与 δM）。静态读仍是无 relayout 的 MXU dot（v7 方案可直接复用）。
   per-token 的 `[N,V]×[V,K]` 没有跨 token 共享的 MXU 操作数，仍是 VPU 运算。
3. **不依赖 kernel 的速赢**：
   - 9 个门投影 + full-M down/up + 标准 Q/K 拼成一次 dot（DirectC10 上 −19 ms；这里小投影更多，预计 −40~−60 ms）；
   - health 统计改为低频或在 kernel 外按需计算（−41 ms device，日志 −3.6%）。
4. **下注**：v7 式 fused kernel 总耗时约 250–300 ms（DirectC10 为 139 ms，主要因动态读 VPU 量 ~4×），
   替代 ~515 ms XLA M 运算和约 80 ms glue → BAM health OFF 从 2163 ms 降到约 1850 ms，
   约 **85–88% of MHA**（当前 74.1%）；叠加输入投影融合可再 +2~3 点。若 read 反传 VALU 实测超过前传 3×，
   应优先考虑把三臂动态读共享 M_k 行广播（一次加载 M_k[v]，同时累加 Q/K/VO 三个 key）。

## Pallas fused kernel 实现与实测（分支 `claude/bam-pallas-fullm`，`9fa69e2` / `1ea3cbb`）

实现：复用 v7 k-major 核心（`layers/bam_pallas_v7.py`），full-M 读即「压缩矩阵取单位阵」（C=V=40，压缩行就是
M 行）；新增 per-token static-read gate（前传乘在静态读上；反传 D 的静态行改为 gate⊙cotangent，gate 梯度
`Σ_k δ⊙st` 在 pass A 累加，需要在反传中用一次 MXU dot 重算全部静态读）；key 走共享 GELU 瓶颈 + 各臂 up +
general pre-bias；融合输入投影分两段（x 侧一次 dot 含 8 个门、full-M down、标准 Q/K、P_loc_down、W_gw；
hidden 侧一次 dot 含三个 up）。写 kernel 与 DirectC10 相同，直接复用。新增可选 `bam_pallas_hm_reverse`：动态读的
反传改为 head-major 行广播累加（dMc_k = Σ_{arm,n} dy_k[n]·key_n，dkey_n = Σ_k dy_k[n]·M_k），去掉逐 (v,k) 的
sublane 归约。CPU 等价（`bam_pallas_model_test.py::test_full_m_general_read`）：loss 一致到 1e-7，梯度相对误差
≤1.5e-5；DirectC10 用例同样通过（含 hm_reverse）。

v5p bundle（每 128-token tile）：read F 20.5k（VALU 下限 16.1k），read B 63.8k → hm 54.0k（VALU 189k → 158k，
spill store 32k → 24k）。k_block=2 只把 read F 降 4%。

| v5p-8 配对（同一台 ew4b） | step/s | MHA 的 % | device ms |
|---|---:|---:|---:|
| 纯 JAX，health OFF | 0.453 | 73.8 | 2165.9 |
| 融合 v7u | 0.519 | 84.5 | 1895.0 |
| + 融合输入投影 | 0.528 / 0.525 | 86.0 / 85.5 | 1866.7 |
| **+ head-major read 反传** | **0.532** | **86.6** | **1846.5** |

kernel 耗时（ms/step，最优版）：read F 43.6、remat 43.6、read B 114.4（hm 前 134.4）、write F 15.3、write B 38.1，
合计 255。HBM 流量 853 → 386 GB/step；copy kernel 184 → 67 ms。下注（kernel 250–300 ms、85–88% of MHA）命中。

剩余差距（1846.5 vs MHA 1617.9 = +229 ms）：kernel +255、MLP 更宽 +113、XLA glue（fused scope 48 + other 87−35）
约 +100；BAM 投影 −201、Splash −57。下一步按收益排序：read B（114 ms，VALU 下限约 40k bundle 对 54k 实际，
spill 仍 24k）> read F（43.6×2，spill 8k）> fused 输入投影（136 ms、126 TF/s，可查形状/拼接方式）> scope 内 glue。
正式训练用 `...FullMReadGeneralReadPallasTruePile`（health OFF，未启动；当前正式 RUN 为 health ON 纯 JAX）。

## 继续优化：v7 收尾 → v8 / v8t（per-token MXU）

v5p-8 同一台 ew4b，全部配对（同 commit 内对照臂复测 0.544–0.547，漂移 <0.5%）。

| 版本（commit） | 改动 | step/s | MHA 的 % | device ms |
|---|---|---:|---:|---:|
| v7 HM（1ea3cbb） | 上节最优 | 0.532 | 86.6 | 1846 |
| rolled B1/B2 / rolled A（625c0f4） | 反传 fori_loop | 0.532 / 0.527 | — | 静态 bundle 估计 −20% 未兑现 |
| + split 代替 slice、wrapper 外转置、token-major q/k/v/o（515f5cb） | glue 47→28 ms，融合投影 137→115 ms | 0.539–0.541 | 88.1 | 1812 |
| + forward 存 VO 读 y（7707cee） | 反传不重算 y | 0.543–0.544 | 88.6 | 1806 |
| + remat 保存读输出（a05a18e） | 去掉 remat 前传 45 ms，但 scan 堆叠 +43 ms | 0.545–0.547 | 89.1 | 1798 |
| v8（d0f411e） | token-major M，XLA 构 key/拆 Y | **0.445** | 72.5 | 2212 |
| **v8t**（5286504） | v7 的 T-minor I/O + 核内 per-token MXU | 0.582 | 94.8 | 1685 |
| **v8t 调优**（1d53d70） | 4N 行不 pad、dY 对齐块+寄存器 swap | **0.593** | **96.6** | **1655** |

**v8 的关键发现**：每个 token 的 M_t 是 MXU 的 stationary 操作数，per-token `[R,V]×[V,K]` 在 v5p 上约
13 cycles/token（v7 VPU 约 160）；一层所有读都是 M_t 的线性函数，合成一次 `X_t[4N,V] @ M_t`（静态 key、
门控都折进 X）；写为 `M_t += A_tᵀ C_t`；反传也都是 matmul。核时间 202→100 ms。但 v8 把 key 构造/Y 拆分交给
XLA，`[B,T,N,40/96]` 小尾维张量在 XLA 中 pad/copy/reduce 极差（glue +330 ms），整体 0.445。
**v8t**：保留 v7 的 T-minor 核 I/O（key/门/因子 [..,T] slab，输出 Splash 原生 [B,N,T,K]），核内用
XLU 转置把 key slab 变成 per-token `X_tᵀ[V,4N]`，per-token MXU 后用 VMEM 重排回 head-major token 行；
M 只在核内/carry 中 token-major `[B,T,V,K]`。

v8t 每 128-token tile v5p bundle（v7u → v8t 调优）：read F 20.6k → 8.1k，read B 49.8k → 12.4k，
write F 6.2k/12.1k（1/2 组）→ 3.7k/5.2k，write B 15.2k/28.2k → 6.0k/10.2k。实测核时间：read F 21.3 ×2（含
remat）、read B 32.4、write F 10.4、write B 16.4，合计 102 ms（v7u 255）。

| v8t 调优（v5p-8 profile，device ms） | MHA | v8t 调优 | 差 |
|---|---:|---:|---:|
| BAM 核 | 0 | 101.8 | +102 |
| 融合输入投影 + O 投影 + QKNorm/RoPE + glue | 418.7（QKV+O+QKNorm） | 226.5 | −192 |
| Splash | 579.1 | 530.1 | −49 |
| SwiGLU MLP（参数退还变宽） | 504.6 | 618.3 | +114 |
| attention/MLP/embedding 写的 XLA 投影 | 0 | 24.8 | +25 |
| scan carry / 其他（M 堆叠 21、Splash 残差 copy 等） | 34.9 | 82.5 | +48 |
| layer norm / LM head | 80.4 | 70.1 | −10 |
| **device step** | **1617.9** | **1655.2** | **+37** |

除去参数退还导致的 MLP 变宽（+114 ms，架构本身），BAM 其余部分已比 MHA 净快 77 ms。

### 经验（可迁移）

1. **per-token 矩阵运算优先上 MXU**：每 token 一个小矩阵乘，只要能把每 token 的状态作为 stationary 权重，
   MXU 比 VPU 广播累加快一个数量级；所有对同一 M_t 的线性读应先在 key 侧合并成一次 matmul。
2. **小尾维张量（40/96 宽）不要跨 XLA/核边界做 token-major**：XLA 对其 pad/copy/reduce 很差；边界保持 T-minor，
   在核内用 XLU 转置换布局；归一化等逐元素运算也应在 T-minor（sublane 归约是 VPU 加法，lane 归约要 XLU）。
3. **Mosaic 访存**：sublane-strided 访问（stride=T 行）约每行一条指令；对齐 [8,L] 块加载 + 寄存器内
   major↔sublane swap 便宜得多；但 swap 本身是 VALU，只用于替换昂贵的 strided gather。3D ref 的 `scr[:, t, :]`
   不是原生 strided load。
4. **remat 保存核输出**只在重算核比 scan 堆叠（约 1 TB/s 写+读）贵时才划算；v7 下打平，v8t 下更不划算。
5. **静态 bundle 估计 rolled loop 会误导**（B1/B2 rolled 估 −20%，实测 0）；以 unrolled 静态 bundle 为主、硬件配对为准。

### 剩余与下一步

- 核内：read F 8.1k bundle（HBM 下限约 12 ms vs 实测 21 ms）与 write B 仍有余量；其余（read B、write F）已接近 HBM 下限。
- XLA：key cmajor 转置约 13 ms、写 content/读输出的 relayout copy 约 10 ms、`y_std + local_o` 8 ms、融合投影 split/concat 12 ms。
- loss 对拍（cd06c76，v5p-8 同机 600 步，v8t vs 纯 JAX health OFF，同数据同 seed）：v8t − 纯 JAX 的平均差
  0–100 步 +0.0002、100–200 +0.002、200–300 −0.045、300–400 −0.029、400–500 −0.009、500–606 −0.0025
  （每步噪声 |d| 约 0.008）；最后 40 步窗口 4.2065 vs 4.2054、4.1365 vs 4.1433。快速下降段的暂时领先已收敛，
  无系统偏差。v8t 0.593 vs 纯 JAX 0.453 step/s（+31%）。
