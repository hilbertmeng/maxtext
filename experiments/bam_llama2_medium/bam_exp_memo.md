把配对消融的 `Δwall` 直接叫作“各部分耗时”不够严谨。`Δwall` 是删掉某部分后整个编译图的边际变化，不是该算子自身时间。

这里按 \(D=1024,n=16,k=v=32,T=S=2048,n_f=1\) 计算。定义一次完整 \(D\!\to D\) 的 \(W_Q\) 投影为 1 单位；在当前 24 层、batch 32、full-remat 训练配置下，约对应 **6.597 XPlane-TF/step**。

| 部分 | 理论 \(W_Q\) | XPlane compute：TF / \(W_Q\) | XPlane scope | 配对 `Δwall` |
|---|---:|---:|---:|---:|
| 标准 Transformer | 16.250 | 未单独归因 | 1824.9 ms¹ | 1208.1 ms² |
| normalize M | 0.003 | 0.006 / 0.001 | 86.5 ms | 与 write 合计357.6 ms |
| write M | 0.531 | 3.354 / 0.508 | 133.5 ms | 同上 |
| read local M for QK | 2.125 | 14.086 / 2.135 | 564.5 ms | 1047.6 ms |
| mix alpha | 0.047 | 0.414 / 0.063 | 320.5 ms | ≈0 ms |
| fetch M | 2.000 | 13.058 / 1.979 | 122.9 ms | 509.8 ms |
| read fetched M | 1.063 | 7.037 / 1.067 | 239.9 ms | 324.3 ms |
| 未覆盖/idle | — | — | 133.1 ms | — |
| **完整 step** | **22.019** | BAM scopes合计 **5.754 \(W_Q\)** | **3425.8 ms** | **3421.5 ms** |

1. 1824.9 ms 是完整 BAM 图中未落入 BAM named scopes 的 standard/optimizer/unscoped work，不是纯 Transformer 算子时间。  
2. 1208.1 ms 是独立 MHA profile 的完整 module wall-time。

理论 FLOPs 来自：

- Transformer：QKV/O投影 \(4\)，两次attention contraction \(2S/D=4\)，SwiGLU \(3d_{\rm ff}/D=8.25\)，合计 **16.25**。
- LocalQK：两个key投影 \(2\) + 两组门投影 \(0.0625\) + 两次双侧M读取 \(0.0625\) = **2.125**。
- Fetched read：一个key投影 \(1\) + 门投影 \(0.03125\) + 双侧M读取 \(0.03125\) = **1.0625**。
- Fetch：\(S k v / D^2=2\)。
- Mix：head权重投影 \(0.015625\) + alpha跨头混合 \(Sn/D^2=0.03125\)，合计 **0.046875**。
- Write：\(P_{\rm loc}=0.5\) + write gate \(0.015625\) + outer-product \(0.015625\)，合计约 **0.53125**。

你问的 LocalQK 与 fetched read，理论上确实正好是：

\[
2.125/1.0625=2
\]

XPlane 也证实实际计算量完全符合：

\[
14.086/7.037=2.002
\]

所以没有隐藏的第三次读取。超出2倍的是时间：

| 指标 | LocalQK / fetched read |
|---|---:|
| 理论 FLOPs | 2.000× |
| XPlane FLOPs | 2.002× |
| XPlane bytes | 2.287× |
| XPlane scope time | 2.353× |
| 配对 `Δwall` | 3.230× |

原因有两层：

- LocalQK 的两个结果必须分别物化并写回 Q、K，随后进入 \(T\times T\) attention及其反向图；XPlane显示它搬运328.0 GB，而fetched read只有143.5 GB。因此虽然FLOPs正好2倍，内存流量已经是2.29倍，实际计算吞吐也从29.34降到24.95 TF/s，最终scope时间成为2.35倍。
- `Δwall` 还会放大到3.23倍，因为移除LocalQK会改变attention、dynamic alpha及其反向传播的整个融合和临界路径；移除fetched read只剪掉attention之后的一条内容读取分支。因此配对消融适合回答“删掉它整步能快多少”，不能当作算子自身执行时间。

## M-read lowering paired profile

同一 `v5p-16`、同一 commit、step 40–44 XPlane；A–E 数学语义相同，理论 FLOPs 不变。

| 路径 | 稳态 step/s | XPlane step | local-QK scope | fetched-M scope |
|---|---:|---:|---:|---:|
| A: dot → `bntd` + transpose | 0.289 | 3425.7 ms | 565.3 ms | 239.7 ms |
| B: dot → `btnd` | 0.290 | 3412.2 ms | 560.0 ms | 236.2 ms |
| C: multiply + reduce → `btnd` | **0.311** | **3191.9 ms** | **378.5 ms** | **156.5 ms** |
| D: B + packed Q/K | 0.292 | 3396.1 ms | 550.4 ms | 236.1 ms |
| E: B + squeeze `n_f=1` | 0.291 | 3407.5 ms | 559.3 ms | 269.0 ms |

- C vs A: step `-6.8%`; vs B: local-QK/fetched scopes `-32.4%/-33.7%`。XLA未物化 `btnkv`；显式写法降为 loop fusion，而 dot 路径是 convolution fusion + output copy。FLOPs、scope bytes基本未降，收益来自更合适的 lowering/layout/backward fusion。
- B、D 仅有 `0.4–0.9%` 整步收益；D 将 local contraction 的 data-formatting 数减半，但被完整 scope 稀释。E 无收益且 fetched scope 变慢。

## Pre-RoPE QKNorm speed anomaly

同一 `v5p-16`、同一模型/批量、step 40–44 XPlane：

| LocalQK 注入 / QKNorm | 稳态 step/s | XPlane step |
|---|---:|---:|
| post / off | 0.326 | 3058.7 ms |
| pre / off | 0.325 | 3072.6 ms |
| post / on（只规范标准 Q/K） | 0.322 | 3094.6 ms |
| pre / on（规范标准+LocalQK） | 0.385 | 2588.6 ms |
| pre / off + 只 cast 合并 Q/K 到 `bf16` | **0.393** | **2539.0 ms** |

根因是隐藏的 dtype 提升，而不是 Pre-RoPE 或 RMSNorm 本身：RmsGate 的 `float32` 门偏置令 LocalQK 读出成为 `f32`；不经后续 dtype 边界时，加回标准 `bf16` Q/K 会把 attention 提升为 `f32`。只有 pre+QKNorm 会在合并后由 RMSNorm 输出端降回 `bf16`。只-cast 对照相对 pre/off 加速 **21.0%**，且比 QKNorm 再快 **2.0%**，完成因果验证。

配对优化 HLO 与 XPlane 一致：

- pre/off 的每层 Q/K 反向各生成一份独立 `bf16[32,16,2048,2048]` masked-softmax 梯度 fusion，RoPE/attention 的 Q/K 操作数仍为 `f32`。
- pre/on 的 Q/K 操作数为 `bf16`，一份 masked-softmax 梯度由两个 Q/K 反向 consumer 共享；总 XPlane bytes 从 6483.8 降至 5168.2 GB（`-20.3%`），FLOPs 基本不变。
- HLO 峰值仅从 56.38 降至 56.02 GiB；收益来自每步动态搬运和更好的 fusion/lowering，不是峰值显存或算术量下降。

## `dot_btn` bilateral read vs MHA QK logits

同一 `v5p-16`、bf16、`B=32,T=S=2048,n=16,d=64,k=v=32`，将层数缩为6但保持单层形状、sharding和full remat不变；step 10–14 XPlane，两个host trace、8个设备平均。时间是完整训练step中归属于该scope的forward、remat、backward和data-formatting总和，不是孤立microbenchmark。

| scope | 理论相对FLOPs | XPlane TF/step | bytes/step | scope time | 有效TF/s | 有效GB/s |
|---|---:|---:|---:|---:|---:|---:|
| 1a `btkv,btnv->btnk` | 1 | 0.02597 | 4.832 GB | 14.47 ms | 1.80 | 334 |
| 1b `btkv,btnk->btnv` | 1 | 0.02577 | 4.429 GB | 13.25 ms | 1.95 | 334 |
| MHA `btnd,bsnd->bnts` | 128 | 3.32631 | 55.579 GB | 52.25 ms | 63.64 | 1064 |

实测耗时比为 **1.092 : 1 : 3.945**，不是纯FLOPs的 `1:1:128`，也低于事前估计的 `1:1:8–20`。小read contraction的矩阵维只有 `16×32×32`，其有效算力比MHA低约33–35倍；实际scope流量比也只有11.5–12.5倍，而MHA fusion的有效带宽约为小contraction的3.2倍，因此MHA只慢3.6–3.9倍。每侧read另有约2.8 ms data formatting，占scope的19–21%。

1a与1b的普通forward、remat和output copy几乎相同；1a慢9.2%来自backward fusion：6层合计6.49 vs 5.26 ms，且搬运2.013 vs 1.611 GB。即1a的反向layout比1b更差，不是前向dot本身更慢。

这种逐层算子profile可优先用6层：本次launch到first step约137秒，且单层算子shape完全不变。比较scope比值或除以层数的时间即可；整步速度、跨层/optimizer临界路径、通信或层数相关fusion仍须用full层数复核。

## Optimized FactorizedLocalQK main profile

最新 `NoMNorm + all-bf16 + CombinedRead + FactorizedLocalQK`，6层、step 10–14、两个host trace/16个设备平均；BAM两类读取均使用 `multiply+reduce`。6层的一个训练态 (W_Q) 单位约为 **1.649 TF/step**。

| 部分 | 理论 (W_Q) | XPlane TF / (W_Q) | bytes | scope ms（6L / 每层 / 24L线性外推） |
|---|---:|---:|---:|---:|
| 标准Transformer / optimizer / unscoped | 16.250³ | 42.397 / 25.707 | 426.38 GB | 394.3¹ / — / — |
| └ MHA QK logits² | 2.000 | 3.329 / 2.018 | 58.00 GB | 55.16 / 9.19 / 220.65 |
| write M | 0.531 | 0.729 / 0.442 | 17.21 GB | 22.70 / 3.78 / 90.79 |
| mix alpha | 0.047 | 0.103 / 0.062 | 49.75 GB | 38.47 / 6.41 / 153.89 |
| fetch M | 2.000 | 3.163 / 1.918 | 11.30 GB | 35.98 / 6.00 / 143.93 |
| **read local M for QK** | **0.197** | **0.328 / 0.199** | **36.73 GB** | **39.83 / 6.64 / 159.32** |
| ├ read-key projection | 0.125 | 0.207 / 0.126 | 8.97 GB | 3.98 / 0.66 / 15.94 |
| ├ read-gate projection | 0.004 | 0.007 / 0.004 | 8.08 GB | 3.86 / 0.64 / 15.44 |
| ├ key RMS/gate transform | ≈0 | 0.001 / 0.000 | 0.76 GB | 0.79 / 0.13 / 3.17 |
| ├ **read M contraction** | 0.004 | 0.006 / 0.004 | 6.49 GB | **22.12 / 3.69 / 88.48** |
| └ head-mix projection/transform/expand | 0.064 | 0.107 / 0.065 | 12.42 GB | 9.07 / 1.51 / 36.30 |
| **read fetched M** | **1.063** | **1.758 / 1.066** | **29.13 GB** | **46.99 / 7.83 / 187.95** |
| ├ read-key projection | 1.000 | 1.651 / 1.001 | 9.77 GB | 10.74 / 1.79 / 42.96 |
| ├ read-gate projection | 0.031 | 0.052 / 0.032 | 4.26 GB | 2.64 / 0.44 / 10.56 |
| ├ key RMS/gate/layout transform | ≈0 | 0.003 / 0.002 | 3.17 GB | 2.76 / 0.46 / 11.02 |
| └ **read M contraction** | 0.031 | 0.052 / 0.031 | 11.93 GB | **30.85 / 5.14 / 123.41** |
| **完整step** | **20.088** | **48.477 / 29.394** | **570.50 GB** | **578.23 ms** |

1. 394.3 ms只是整步减去五个BAM顶层scope后的wall residual；其中混有标准Transformer、optimizer、通信、未命名操作和idle，不能当作纯Transformer算子时间。XPlane TF/bytes列则是所有未落入这些BAM scope的XLA操作总和。
2. MHA QK logits已经包含在“标准Transformer”内，仅作读M的共同参照，不重复相加。
3. 16.250仅是标准Transformer前向理论量；该行实测还含optimizer和其他unscoped工作，所以不可直接比较理论/实测两列。

理论量从旧版 PerHeadLocalQK 的 **2.125 (W_Q)** 降到 FactorizedLocalQK 的 **0.197 (W_Q)**：两侧共享key投影0.125、门0.0039、共享M contraction 0.0039、head routing约0.0645。XPlane测得0.199 (W_Q)，吻合。

### `dot_btn` vs `multiply+reduce`（同一6层图）

| 指标 | dot | multiply+reduce | 变化 |
|---|---:|---:|---:|
| 完整step | 587.41 ms | **578.23 ms** | **-1.56%**（step/s +1.59%） |
| local-QK总scope | 40.05 ms | 39.83 ms | -0.54% |
| └ local read-M contraction | 22.23 ms | 22.12 ms | -0.52% |
| fetched-read总scope | 48.35 ms | **46.99 ms** | **-2.82%** |
| └ fetched read-M contraction | 31.12 ms | 30.85 ms | -0.86% |
| fetch-M scope | 32.72 ms | 35.98 ms | +9.98% |

结论：在已经因FactorizedLocalQK大幅缩小读形状的最新路径上，`multiply+reduce`仍有真实但较小的整步收益（+1.59%），远小于旧PerHead路径的+6.8%。直接归到M contraction的时间只降0.5–0.9%；较大的fetched-read总scope收益来自layout/copy/fusion重排。不同lowering还把部分融合算子重新归到key-transform/fetch scope，因此细分scope适合定位，最终优劣以配对完整step为准。

## Write outer 与 diagonal-one 配对 profile

同一 `v5p-16`、相同 batch/data、step 10–14 XPlane；六层使用 commit `fef8e3a`，
24 层复核使用仅增配置类的 `2b8e63a`。配对同 step loss 最大差约 `2.4e-4`，语义一致。
原始 trace 位于 `/data0/xd/bam_diagnostics/write_read_pair_fef8/`。

| M-write 路径 | 层数 | dot / multiply+reduce step/s | XPlane step | `write_outer` | `write_m` | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Direct dynamic-V | 6 | 1.861 / **1.975** | 528.182 / **498.124 ms** (-5.69%) | 8.913 / 5.297 ms | 22.431 / 15.068 ms | multiply+reduce |
| Direct dynamic-V | 24 | 0.515 / **0.554** | 1914.679 / **1775.476 ms** (-7.27%) | 39.684 / 19.328 ms | 99.885 / 55.601 ms | multiply+reduce |
| Static-V | 6 | **2.022** / 2.009 | **486.295** / 491.984 ms (+1.17%) | **1.539** / 5.168 ms | **4.747** / 8.336 ms | 保留 dot |

两种写法的前向理论 outer FLOPs 相同，均为每层 `0.015625 W_Q`。Direct 的 dot 被降为
`convolution_add_fusion`，显式 multiply+reduce 变成更适合该图的 `multiply_reduce_fusion`；
24 层下全图 XPlane TF/bytes 也从 `143.926/2050.1 GB` 降到 `120.405/1701.1 GB`，
说明收益扩散到 remat/backward fusion，而不只是前向 outer。Static 的 dot 本已被 XLA 降为
高效 `multiply_reduce_fusion`；手写 multiply+reduce 反而生成更差的反向/layout 图。

| 等价 read 路径 | 层数 | Combined / diagonal-one step/s | XPlane step | `fetch_m` | `read_fetched_m` | `mix_alpha` |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 6 | 1.690 / **1.742** | 581.695 / **568.628 ms** (-2.25%) | 35.980 / 20.855 ms | 47.622 / 36.337 ms | 38.480 / 42.167 ms |
| V1 | 24 | 0.459 / **0.478** | 2149.300 / **2064.121 ms** (-3.96%) | 131.310 / 69.930 ms | 157.383 / 115.503 ms | 123.021 / 134.063 ms |

Diagonal-one 直接去掉 `local_o`：保留 mixed-alpha 的非对角元、把对角置 1 后只 fetch/read
一次；它与 CombinedRead 的“fetch 对角置 0、再加本地 M”代数等价。24 层全图 XPlane
TF 仅 `-0.19%`、bytes 反而 `+0.35%`，所以约4%的收益主要来自避免独立 `Mbar + M`
所带来的 layout/copy 和更好的临界路径，而非理论计算量下降。后续 V1 变种应优先使用
diagonal-one 实现。

## Pure-JAX bilateral block read

同一 `v5p-16`、commit `ece8eb2`、bf16、batch/data与step 10–14 XPlane。把
行/列读写成块矩阵 `[[0,M],[M.T,0]]`；LocalQK把Q/K作为两个RHS列，full fetch把16个头
作为RHS列。块矩阵含一半零元素，但普通dense lowering仍计算和搬运这些零块。

| 6层路径 | 稳态 step/s | XPlane step | Δwall | 对应 read-M：TF / GB / ms |
|---|---:|---:|---:|---:|
| 原路径 | 1.690 | 576.960 ms | — | Local 0.00628 / 6.49 / 22.12；fetch 0.05151 / 11.93 / 30.86 |
| Local block dot | **1.709** | **571.364 ms** | **-0.97%** | 0.02409 / 9.49 / 20.61 |
| Local block multiply+reduce | 1.685 | 580.252 ms | +0.57% | 0.01937 / 20.39 / 34.64 |
| Fetch block dot | 1.640 | 596.173 ms | +3.33% | 0.11489 / 14.83 / 29.83 |
| Fetch block multiply+reduce | 1.533 | 637.482 ms | +10.49% | 0.11013 / 30.07 / 74.61 |

Local block-dot在6层用更大dot提高了利用率，虽read-M FLOPs为原路径3.84倍，scope仍快
6.8%；其余三路均无效。完整24层复核使用commit `26f43ec`（仅新增配对配置类，算子代码
仍为`ece8eb2`）验证唯一的6层胜者：

| 24层路径 | 稳态 step/s（step 20–40） | XPlane step | Local scope / read-M | read-M bytes |
|---|---:|---:|---:|---:|
| 原路径 | **0.46110** | **2,137.186 ms** | 137.52 / 80.12 ms | 22.25 GB |
| Local block dot | 0.45610 | 2,159.358 ms (**+1.04%**) | 193.61 / 129.71 ms | 80.49 GB |

24层图中block-dot lowering新增大量归属于`reshape`的output copy；read-M流量增加2.62倍、
时间增加61.9%，使六层的局部收益反转。结论：不采用任何block-read路径，V1继续使用分别
执行的row/col `multiply+reduce`。六层适合筛选算子，但涉及layout、remat和内存压力的结果
必须用完整层数复核。

## V1 M-cache temporal compression diagnostics

V1 step 13250、commit `49be222`、spot `v6e-1`；随机 Pile eval 128 条（4×32，执行时切成
8×16 microbatch），与原始 batch32 cohort 的 128 个序列哈希逐条一致。microbatch baseline
loss `2.379473`，与原结果 `2.379468` 相差约 `5e-6`。以下均为同 checkpoint、同 batch 的
只读替换，不含重训适应；完整结果：
`/data0/xd/bam_diagnostics/bam_cache_diagnostics_49be222_mb16_final.json`。

Dynamic RMS mix 并非单头选择：系数 L2 norm `1.000`、最大绝对系数均值 `0.549`、有效头数
均值 `6.03/16`；系数相邻 token cosine `0.831`。混合后的 signed alpha 有 `57.9%` 负值，
有效支持均值/中位数 `43.2/16.8`；top-16/64/128 的绝对质量为 `59.0/75.5/83.3%`，最近
128/256/512 为 `68.6/78.7/88.5%`。每序列的 window256 质量均值 p10/50/90 为
`63.9/79.8/91.8%`：alpha 有一定稀疏性和局部性，但固定 256 窗口并不可靠。

系数随 token 平滑不代表 `M_s` 平滑。跨层合计的 `M_s` cosine（lag 1/2/4/8/16/32）为
`0.249/0.179/0.140/0.113/0.096/0.085`，相对 delta RMS 为
`1.225/1.282/1.312/1.332/1.344/1.352`；layer 1 的 lag-1 cosine 仅 `0.014`。因此直接对
相邻 `M_s` 求均值或线性拟合不是健康的压缩基底。

| 压缩 | nominal cache reduction | dloss | fetch-M rel-RMS | combined-M rel-RMS | BAM output rel-RMS |
|---|---:|---:|---:|---:|---:|
| B8 mean | 8× | +0.4104 | 0.7525 | 0.5824 | 0.7072 |
| B8 linear | 4× | +0.2236 | 0.6358 | 0.4846 | 0.6107 |
| B16 mean | 16× | +0.4859 | 0.8080 | 0.6242 | 0.7468 |
| B16 linear | 8× | +0.3617 | 0.7282 | 0.5617 | 0.6916 |
| B32 mean | 32× | +0.4844 | 0.8432 | 0.6488 | 0.7688 |
| B32 linear | 16× | +0.4196 | 0.7857 | 0.6070 | 0.7348 |

Linear 始终优于 mean，但所有纯 block 方案都严重破坏模型；最佳 B8 linear 仍使 128/128
序列 loss 变差，平均 `+0.2236`。

| 读取方案（T=2048） | effective cache reduction | dloss | fetch-M rel-RMS | BAM output rel-RMS | 改善序列 |
|---|---:|---:|---:|---:|---:|
| Window256 only | 8.00× | +0.1193 | 0.3725 | 0.3988 | 0/128 |
| Window256 + OldBlock16 mean | 5.57× | +0.0811 | 0.3050 | 0.3496 | 0/128 |
| Window256 + OldBlock16 linear | 4.27× | **+0.0626** | **0.2818** | **0.3261** | 1/128 |

OldBlock16 能恢复被窗口丢失的信息，linear 最好，但 `+0.0626` 仍远高于只读近似通常可接受
的 `<0.01`。当前 checkpoint 不支持把 Window256+OldBlock16 视为健康替换；若继续压缩
M-cache，应优先尝试由 alpha 动态选择的稀疏/分层记忆，而不是按时间邻近直接压缩 `M_s`。

## V1 signed dynamic-mix diagnostics

V1 step 13250，同一 Pile eval 128 条 cohort；sign/mass 与只读消融分别由 commits
`1bbf708`、`94961c7`、`c7ba633` 采集。完整结果位于
`/data0/xd/bam_diagnostics/bam_alpha_*_final.json`。

`57.9%` 是负 alpha 的**元素个数**，不能直接解释为负贡献强度。全体 query 按绝对质量
加权后，正/负质量为 `56.8/43.2%`；逐 query 的负质量占比 mean/p50/p90 为
`46.1/42.9/90.6%`。逐 query cancellation `1-|sum(alpha)|/L1(alpha)` 的
mean/p50/p90 为 `46.9/48.4/86.4%`；选择每条 route 的占优符号后，表示局部双极对比的
少数符号质量平均约为其一半，即 `23.4%`。

层间差异极大：layer 1 的负质量仅 `0.3%`，layer 3/7 分别为 `93.5/91.8%`。计数与质量
也可相反：layer 4 有 `83.4%` 负元素但负质量仅 `39.0%`；layer 8 只有 `41.9%` 负元素，
负质量却为 `63.5%`。删除少数符号后的逐层 fetch/output error 与少数符号质量相关系数为
`0.87/0.74`，说明质量比计数更有诊断意义。

RMS mix 的系数 L2 固定为1，但 L1 mean/p50 为 `3.126/3.180`（16维理论上限4）；负质量
占比 `46.6%`，正负 cancellation 均值 `70.6%`。把系数分解为 common mean direction 与
zero-sum contrast direction 后，后者占 `91.2%` 能量；但16维随机单位向量的期望本来就是
`93.75%`，所以高 contrast 能量大部分可能是参数化几何，而非训练主动选择。

| 同 batch 只读消融 | 控制变量 | dloss |
|---|---|---:|
| `mix_abs_l2` | 系数 L2 不变，全部改正 | +2.8718 |
| `mix_positive_l2` | 删除负系数，系数 L2 不变 | +1.7539 |
| `alpha_abs` | alpha 每点绝对值及 L1/L2 不变，只翻负号 | +1.5727 |
| `alpha_positive_raw` | 仅保留正 alpha，不补幅度 | +0.3865 |
| `alpha_positive_l2` | 仅保留正 alpha，恢复原 L2 | +1.0934 |
| `alpha_negative_l2` | 仅保留负 alpha，恢复原 L2 | +2.2908 |
| `mix_dominant_sign_l2` | 每 query 保留占优符号系数，恢复 L2 | +0.1279 |
| `alpha_dominant_sign_raw` | 每 query 只删少数符号 alpha | **+0.0293** |
| `alpha_dominant_sign_l2` | 同上并恢复 alpha L2 | +0.0627 |
| `mix_mean_mode_raw` | 只保留 common mean component | +0.3867 |
| `mix_contrast_raw` | 只保留 zero-sum contrast component | +0.1855 |

所有消融均为 `0/128` 序列改善。`alpha_abs` 证明当前 checkpoint 确实依赖符号，而非只依赖
绝对幅度；但 naïve positive-only 把整体负 fetch（在 CombinedRead 中相对固定 `+1` local
项也有意义）与局部双极对比混在一起，严重夸大了“负值本身”的作用。保留占优符号的消融
表明局部双极对比有稳定但中等的贡献：删除它使
loss `+0.0293`，恢复 L2 反而更差，说明正负内容与总增益已共同校准。mean-only 与
contrast-only 都明显变差，二者均有功能；contrast 的作用不只制造负 alpha，也会重排同号
源位置的相对强度。

这只能证明 checkpoint 已与 signed route 共适应，不能证明 signed 参数化的训练最优点更好；
此前独立训练的 RmsMix 与 SoftmaxMix 在约6200步几乎打平（`-0.0006`）也支持这一保留意见。
后续若改动态混合，优先在最终 V1 上配对重训：positive SoftmaxMix，以及“正向基路由 +
独立零和 contrast + 显式有界门”的方案。后一方案先混成一个 alpha 再 fetch，不增加 M 读取：

`w = g_base * softmax(z_base) + g_ctr * normalize_L1(z_ctr - mean(z_ctr))`

其中 contrast gate 小值初始化、base gate 开启；这样保留源级减法表达力，同时把正向选择、
对比方向和幅度解耦，避免当前 L2-only 归一化造成接近最大 L1、强 cancellation 与尺度漂移。

### Mixed-alpha attention sink

V1 step 13250、同一 128 条 cohort，commit `a02fc72`；统计实际参与 fetch 的
post-mix/post-diagonal alpha。完整结果：
`/data0/xd/bam_diagnostics/bam_alpha_sink_diagnostics_a02fc72_final.json`。

存在**局部 token-0 sink**，但不是宽前缀 sink。对 query position >=1024，按 alpha 绝对质量
汇总，position 0 的逐 token 富集为 `1.73x`，layer 8/11/13 分别为
`6.49x/5.67x/3.95x`；但 first 2/4/16 的整体富集仅
`1.04x/0.67x/0.39x`。token 0 自身只占总绝对质量 `0.114%`，正负质量约
`49.3/50.7%`，不是单一符号的 sink。

它只能小幅解释 Window256 损伤。对所有受窗口影响的 query (position >=256)，token 0、
first 16、first 64 分别只解释被 Window256 删除绝对质量的 `0.93%/3.11%/9.40%`；对
position >=1024 则为 `0.39%/1.43%/4.50%`。因此绝大多数被删质量来自分散的旧历史，
不是序列开头。alpha 质量不能排除早期 `M_s` 被下游放大的可能；若需严格因果归因，做同
batch 的 `Window256 + keep first K` 只读消融。

K=4 的同-cohort 因果消融（commit `d3c17a6`）确认该解释很弱：Window256 only 与
Window256+keep-first-4 的 loss 分别为 `2.498772/2.498375`，相对未压缩 V1 的 dloss 为
`+0.119299/+0.118902`。保留前4个 token 仅改善 `0.000397`，追回窗口损伤的 `0.33%`；
8个 microbatch 中7个改善，但幅度很小。fetch-M/combined-M/BAM-output rel-RMS 分别从
`0.372531/0.298635/0.398787` 降至 `0.371736/0.298052/0.398147`，也只改善
`0.21%/0.20%/0.16%`。因此 token-0 sink 真实存在，但不是 Window256 失效的主要原因。
完整结果：`/data0/xd/bam_diagnostics/bam_window_prefix4_diagnostics_d3c17a6_final.json`。

## PackedLocalQK stable-gap root cause

The `~+0.0034` loss gap of
`BamLlama2MediumDirectPLocR256GeluPackedLocalQKReadGateInit005Eps1e4` versus Direct is caused by
`bam_replicate_ploc_up=True`, not the packed projection, `btn` output layout, or native packed
initializer:

- `btn`-only exactly matched native PackedOnly at every common step 0–41 (max loss error 0).
- replicated-`P_loc_up`-only exactly matched the old eps1e4 run at every common step 0–56
  (max loss error 0); adding `btn` also matched it exactly through step 48.
- Native PackedOnly's initialization transient vanished: mean dloss `-0.0010` versus Direct over
  steps 1,800–2,600, while it remained about `-0.0051` better than the replicated old run.
- A mapped-step-0 control fixes parameter initialization yet diverges immediately when replication
  is enabled (loss differences appear by step 2), excluding different initial parameter values.

The flag changes `P_loc_up/kernel` axes from `('embed','q_heads','v_factor')` to
`(None,'q_heads','v_factor')`. On v5p-16 the `embed` logical axis maps to the 8-way FSDP mesh, so
this changes the 256-dimensional input axis from 8-way sharded to replicated. The mathematical
projection is unchanged, but its FSDP path changes from parameter all-gather/gradient reduce-scatter
to replicated computation/gradient all-reduce. Different collective and dot reduction order creates
small floating-point differences, which training amplifies into a different trajectory. Replication
gives about 1.5% speed, but this run's trajectory is persistently worse, so capability comparisons
should keep `bam_replicate_ploc_up=False`.

## Bf16Packed clean main profile

`BamLlama2MediumDirectPLocR256GeluBf16PackedLocalQK` 的六层同构图，commit
`9fb6720`，standalone `v6e-1`，step 10–14 XPlane。一个训练态 `(W_Q)` 单位由
MHA QK logits 校准为 **3.329 TF/step**。XPlane 只取决于训练图，不依赖参数值，
因此无需把24层 checkpoint 裁成不兼容的六层参数树。

| 部分 | 理论 (W_Q) | XPlane TF / (W_Q) | bytes | scope ms（6L / 每层 / 24L线性外推） |
|---|---:|---:|---:|---:|
| 标准Transformer / optimizer / unscoped | 16.250³ | 72.308 / 21.722 | 564.44 GB | 498.21¹ / — / — |
| └ MHA QK logits² | 2.000 | 6.657 / 2.000 | 115.99 GB | 101.07 / 16.84 / 404.28 |
| **write M** | **0.406** | **1.125 / 0.338** | **23.41 GB** | **35.35 / 5.89 / 141.39** |
| ├ `P_loc_down` | 0.250 | 0.518 / 0.156 | 2.53 GB | 2.33 / 0.39 / 9.32 |
| ├ `P_loc_up` | 0.125 | 0.523 / 0.157 | 3.77 GB | 2.74 / 0.46 / 10.94 |
| ├ write-gate projection | 0.016 | 0.047 / 0.014 | 2.75 GB | 2.63 / 0.44 / 10.52 |
| ├ **write outer product** | 0.016 | 0.032 / 0.010 | 5.91 GB | **18.84 / 3.14 / 75.36** |
| └ GELU/RMS/bias/other | ≈0 | 0.005 / 0.001 | 8.45 GB | 8.81 / 1.47 / 35.25 |
| **mix alpha** | **0.047** | **0.248 / 0.075** | **97.76 GB** | **91.38 / 15.23 / 365.52** |
| ├ head-weight projection | 0.016 | 0.056 / 0.017 | 3.32 GB | 3.60 / 0.60 / 14.39 |
| ├ **`bnts,btn->bts`** | **0.031** | **0.192 / 0.058** | **91.11 GB** | **84.44 / 14.07 / 337.75** |
| └ transform/other | ≈0 | 0.000 / 0.000 | 3.32 GB | 3.35 / 0.56 / 13.38 |
| **fetch M** | **0.508** | **1.678 / 0.504** | **14.19 GB** | **17.16 / 2.86 / 68.64** |
| ├ absolute-V source compression | 0.008 | 0.025 / 0.007 | 5.13 GB | **9.99 / 1.67 / 39.96** |
| └ temporal fetch contraction | 0.500 | 1.653 / 0.497 | 9.06 GB | 7.17 / 1.19 / 28.68 |
| **read local M for QK** | **0.197** | **0.655 / 0.197** | **25.54 GB** | **44.01 / 7.33 / 176.03** |
| ├ packed key/gate/head-mix projection | 0.191 | 0.636 / 0.191 | 3.86 GB | 2.93 / 0.49 / 11.70 |
| ├ key RMS/gate transform | ≈0 | 0.001 / 0.000 | 1.52 GB | 0.79 / 0.13 / 3.17 |
| ├ **read M contraction** | **0.004** | **0.012 / 0.004** | **11.78 GB** | **33.74 / 5.62 / 134.95** |
| ├ head-mix transform/expand | ≈0.002 | 0.005 / 0.002 | 7.45 GB | 5.92 / 0.99 / 23.68 |
| └ other | ≈0 | 0.000 / 0.000 | 0.93 GB | 0.63 / 0.11 / 2.53 |
| **read fetched M** | **0.664** | **2.205 / 0.663** | **21.63 GB** | **22.85 / 3.81 / 91.38** |
| ├ read-key projection | 0.625 | 2.067 / 0.621 | 5.29 GB | 3.33 / 0.55 / 13.30 |
| ├ read-gate projection | 0.031 | 0.108 / 0.032 | 3.35 GB | 2.60 / 0.43 / 10.42 |
| ├ key RMS/gate/layout transform | ≈0 | 0.004 / 0.001 | 4.23 GB | 2.84 / 0.47 / 11.35 |
| └ **read M contraction** | **0.008** | **0.026 / 0.008** | **8.76 GB** | **14.08 / 2.35 / 56.31** |
| **完整step** | **18.072** | **78.218 / 23.498** | **746.98 GB** | **708.95 ms** |

1. 498.21 ms 与旧表相同，只是整步减去五个BAM顶层scope后的wall residual，不是纯
   Transformer算子时间。
2. MHA QK logits 已含在标准部分，只作共同参照。
3. 新旧绝对 wall time 不可直接比较：旧表是 `v5p-16` 的16设备平均，本表是单设备
   `v6e-1`。代码差异和理论量用于因果判断；scope占比及相对 MHA QK 的耗时仅作跨硬件佐证。

### 相对旧 FactorizedLocalQK main profile

| 部分 | 旧→clean 理论 (W_Q) | 整步占比旧→clean | scope / MHA QK旧→clean | 代码原因 |
|---|---:|---:|---:|---|
| write M | 0.531→0.406 | 3.93%→4.99% | 0.411→0.350 | `P_loc: D→256→nV`，理论降23.5% |
| mix alpha | 0.047→0.047 | 6.65%→12.89% | 0.697→0.904 | 图未简化；已成为最大BAM wall瓶颈 |
| fetch M | 2.000→0.508 | 6.22%→2.42% | 0.652→0.170 | AbsV8，把fetch的V轴32压至8，理论降74.6% |
| local QK read | 0.197→0.197 | 6.89%→6.21% | 0.722→0.435 | packed投影只合并launch/layout，理论量不变 |
| fetched read | 1.063→0.664 | 8.13%→3.22% | 0.852→0.226 | AbsV8缩小row key/output，理论降37.5% |
| **BAM合计** | **3.838→1.822** | **31.81%→29.73%** | **3.335→2.085** | **BAM理论量降52.5%，完整图理论量降10.0%** |

代码对比确认，两版共同使用 NoMNorm、all-bf16、CombinedRead、FactorizedLocalQK 和两类
`multiply+reduce` read。clean 新增的结构差异只有：AbsV8、R256-GELU `P_loc`、packed
LocalQK投影，以及BAM RMS统计使用activation dtype；最后一项不改变理论量，且当前
bf16/fp32 packed实测速度近乎相同。

改善最明确的是 AbsV8：fetch与fetched-read的理论量和归一化wall都大降。packed LocalQK
把投影本身压到2.93 ms，但真正的local-M contraction仍用33.74 ms完成仅0.004 `(W_Q)`，
所以LocalQK总体只小幅降占比。当前未解决项按优先级为：

1. `mix alpha` 的 `bnts,btn->bts`：仅0.031 `(W_Q)`，却占整步11.91%、搬运91.11 GB；
   应优先配对验证流式融合mix+fetch，或Pallas kernel，避免物化完整 `bts`。
2. local-M contraction：仅0.004 `(W_Q)` 却占4.76%；packed投影没有触及这个小矩阵归约
   的低利用率。
3. AbsV8 source compression：理论仅0.008 `(W_Q)`，却比真正的0.500 `(W_Q)` temporal
   fetch更慢（9.99 vs 7.17 ms）；应配对测 `dot` 与 broadcast multiply+reduce。
4. write outer product占write scope的53%；clean尚未启用已验证的write multiply+reduce。
5. clean仍是CombinedRead，未启用已验证等价且更快的diagonal-one路径；这两项属于已知可
   回收速度，不是新瓶颈。

原始结果：`/data0/xd/bam_diagnostics/clean_profile_9fb6720_v6e/`。

### AbsV source compression配对

同一 `v6e-1`、六层图、step 10–14，只把 `bskv,vc->bskc` 从dot改为broadcast
multiply+reduce：

| 路径 | 稳态 step/s | XPlane step | source compression | fetch M总scope | BAM总scope |
|---|---:|---:|---:|---:|---:|
| dot | **1.399** | **708.95 ms** | **9.99 ms** | **17.16 ms** | **210.74 ms** |
| multiply+reduce | 1.383 | 716.26 ms | 15.24 ms | 24.50 ms | 218.94 ms |
| 变化 | -1.14% | +1.03% | +52.5% | +42.8% | +3.89% |

multiply+reduce无效且明确更慢。XPlane中它生成独立的
`multiply_reduce_fusion`，反向仍伴随大块output copy；没有像较早的read-M contraction那样
借由融合消除不利dot lowering。scope重归属也使后续temporal contraction从7.17升到8.71 ms，
所以最终以完整step判负，生产默认保留dot。配对结果：
`/data0/xd/bam_diagnostics/clean_source_mul_2386d1d_v6e/`。

## V2 early-loss divergence

V2同时把CombinedRead改为diagonal-one，并把write outer的dot改为multiply+reduce。以fp32
Packed Native为共同基准的2×2配对表明，两项单独都没有持续loss负作用：

| gap = RUN - Native | 200 | 400 | 600 |
|---|---:|---:|---:|
| diagonal-one only | -0.01877 | -0.00639 | -0.00367 |
| write multiply+reduce only | +0.02758 | +0.00234 | -0.00142 |
| V2（两项同时） | +0.07236 | +0.02037 | +0.01544 |

组合相对Native的gap继续收窄，但到Native可比区间末端仍未归零：

| step | 800 | 1,000 | 1,200 | 1,400 | 1,600 | 1,800 | 2,000 | 2,200 | 2,400 | 2,600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 - Native | +.01065 | +.00874 | +.00811 | +.00575 | +.00547 | +.00451 | +.00516 | +.00445 | +.00392 | +.00374 |

最近五点（1,800–2,600）均值`+0.00436`，是两项fast改动组合相对直接父配置的因果
loss代价。V2相对长期Direct基线在2,400–4,000均值约`+0.00253`，同时快约5.8%；这只
说明总体速度/loss折中较小，不能用来否定fast组合的负作用。Native止于2,800，故无法据此
证明该代价最终会消失。
同一v5p-16上的bf16原语诊断确认数学语义未变，但归约/反向路径不逐位相同：diagonal-one
前向rel-RMS误差`1.48e-5`、M梯度`8.02e-4`；write前向rel-RMS误差仅`1.49e-8`，但
u1/u2梯度约`2.57e-3`；组合后alpha梯度约`3.38e-3`。训练loss在step 1完全相同，step 2
才出现`1e-5`级差异，之后差值多次换符号并被优化动态放大。因此早期大gap是两个代数等价
实现改变浮点归约与反向顺序后造成的确定性轨迹分叉，不是read/write语义错误；但观测区间
内仍有约`+0.0044`的持续loss代价。复现脚本：`diagnose_v2_fast_path_numerics.py`
（commit `90806a4`）。

V2 后续训满13,500；相对长期Direct在13,400步的gap降至`+0.000138`，说明整套V2相对
Direct的早期轨迹差最终基本消失。Native基准止于2,800，因此不能把这一晚期结论严格拆成
diagonal-one/write-mul本身相对Native的独立因果效果。

## Shared QChunk + SWA profile

实现见 `shared_qchunk_swa_design.md`。语义验证覆盖 dense/C256 的前向、loss 和参数梯度；
profile 使用 bf16、`B=32,T=2048`、step 10–14。v6e-1 主矩阵为 commit `0caa467`；v5p-16
早期交叉验证为 commit `1f40820`。跨配置结论只使用同一 TPU 型号内配对。

### All-global chunk size

| 配置 | v6e XPlane step | 相对自身 dense 吞吐 | 相对匹配 MHA 的 BAM overhead |
|---|---:|---:|---:|
| MHA dense | 373.31 ms | — | — |
| MHA C256 | 376.58 ms | -0.9% | — |
| BAM dense | 683.15 ms | — | 83.0% |
| BAM C128 | 608.79 ms | +12.2% | — |
| **BAM C256** | **591.78 ms** | **+15.4%** | **57.1%** |
| BAM C512 | 596.40 ms | +14.5% | — |

C256 最快，但与 C512 只差0.78% step time；选择C256还因其alpha/M中间规模更小。MHA
C256 本身略慢，故 BAM C256 的 +15.4% 不是普通MHA chunk收益。dense→C256 时显式
`bam_total`只从173.96降至169.64 ms，而整步从683.15降至591.78 ms；同时XPlane累计的
non-BAM算子时间从1197.44降至1018.50 ms。主要收益来自整个联合图的lowering、remat和
中间值生命周期改善，不能只归因于标注scope内的mix/fetch FLOPs。

### Local/global schedules

| L:G | 层数 | MHA XPlane | BAM XPlane | BAM overhead |
|---:|---:|---:|---:|---:|
| 0:1（global C256） | 6 | 376.58 ms | 591.78 ms | 57.1% |
| 1:1 | 6 | 323.39 ms | 499.06 ms | 54.3% |
| 3:1（LGLL） | 8 | 376.77 ms | 579.91 ms | 53.9% |

每行只在相同层数/调度的MHA与BAM间计算overhead。1:1相对全global时，MHA/BAM分别
提速16.4%/18.6%；增加local比例后BAM overhead仅小幅下降至约54%，说明SWA同时缩短
MHA QK/AV与BAM mix/fetch，而每层write、LocalQK和运行时读键投影仍保留。

### Three-input and TPU-type check

将 `bncs,bcn->bcs` 和 `bcs,bskv->bckv` 合成三输入einsum没有得到融合收益：v5p-16
两阶段C256为479.01 ms，三输入为601.89 ms，吞吐下降20.4%；v6e日志也从约1.680降至
1.196 steps/s（-28.8%）。保留两阶段实现。

同一dense/C256 BAM配对在v5p-16只提升4.20%（499.11→479.01 ms），而v6e提升15.44%
（683.15→591.78 ms）。TPU间绝对速度不可直接比较；目标full-24结果统一见下表。

### Full-24 MHA/BAM throughput

Canonical results use `v5p-16`, `B=32,T=2048`, `float32_logits=False`, and the 16-device mean of
the step 10–14 XPlane. The timing source is UC1a unless marked otherwise; class plus runtime commit
defines the reproducible configuration.

| MHA-only path | Configuration class | Runtime commit | XPlane step | Stable log | Relative result |
|---|---|---:|---:|---:|---:|
| standard dense dot | `Llama2MediumDotProductFullLayerProfile` | `f052fa6` | 1,258.65 ms | ~0.786 | reference |
| `BamAttention` dense | `BamMHAControlDenseFullLayerProfile` | `f052fa6` | 1,276.37 ms | ~0.775 | 1.41% slower than standard dot |
| `BamAttention` C256, fixed 4D | `BamMHAControlQChunk256FullLayerProfile` | `a1ad13f` | **1,088.61 ms** | **~0.908** | +17.25% throughput vs BAM dense |

The current C256 control removes redundant chunk-local remat but retains the 4D contraction. It is
6.80% faster than its pre-fix `f052fa6` result (1,162.67 ms, ~0.854 steps/s). An identical EW4b
run gave 1,083.54 ms/~0.911, only 0.47% faster and with matching device/log direction, so no Pile
input slowdown was observed; UC1a remains canonical for historical comparability.

Full BAM overhead is only reported for semantically matched control/full paths:

| Attention | MHA-only control class | MHA step | Full BAM class | BAM step | BAM/MHA throughput |
|---|---|---:|---|---:|---:|
| dense | `BamMHAControlDenseFullLayerProfile` | 1,276.37 ms | `BamV2DenseFullLayerProfile` | 1,780.90 ms | 71.7% |
| C256, legacy inner-remat | `BamMHAControlQChunk256FullLayerProfile` @`f052fa6` | 1,162.67 ms | `BamV2QChunk256FullLayerProfile` | 1,715.14 ms | 67.8% |
| **C256, optimized** | `BamMHAControlQChunk256FullLayerProfile` @`a1ad13f` | **1,088.61 ms** | `BamV2QChunk256OptimizedFullLayerProfile` @`165b55b` | **1,455.35 ms** | **74.8%** |

Optimized C256 is +17.85% throughput versus legacy C256 and +22.37% versus dense BAM; stable logs
are ~0.675 steps/s. Its 16-device range is 1,453.44--1,457.40 ms. Relative to the matched C256 MHA
control its time overhead is 33.7%, down from legacy C256's 47.5%; the canonical matched
throughput-retention figure is 74.8%.

### v6e controlled overhead recheck

Commit `da35a43`, one standalone `v6e-1`, six layers, `B=32,T=2048`, all
`float32_logits=False`; every arm uses the step 10–14 primary XPlane and also retained a step 2–6
insurance trace. `BAM/MHA` is throughput retained; `time overhead = BAM/MHA step time - 1`.
The earlier v6e MHA arms inherited `float32_logits=True` while BAM used `False`, so that table was
not a fair overhead pair. The recheck reproduces dense MHA within 0.01%, BAM dense within 0.14%,
and BAM C256 within 0.08%; only MHA C256 moves 376.58→369.15 ms after the dtype correction.

| Comparison | MHA step | BAM step | BAM/MHA | time overhead | C256 throughput gain |
|---|---:|---:|---:|---:|---:|
| default dense (`autoselected` Pallas/flash) | 373.26 ms | 684.12 ms | 54.6% | +83.3% | — |
| C256 | 369.15 ms | 592.28 ms | 62.3% | +60.4% | MHA +1.1%; BAM +15.5% |
| matched explicit dense dot | 481.56 ms | 684.12 ms | 70.4% | +42.1% | MHA dot→C256 +30.5%; BAM +15.5% |

因此v5p-16默认配置中“C256使相对BAM overhead变大”**没有在v6e默认后端复现**；v6e
反而从+83.3%降至+60.4%。但matched-dot控制确实从+42.1%升至+60.4%，说明现象不是
简单测量错误，而是强烈依赖dense MHA分母的后端：默认dense MHA走融合Pallas/flash，
C256走显式`accelerator.QChunk`。v6e上两者几乎打平；v5p-16 full-24中C256 MHA比
dense Pallas快13.8%，所以v5p的分母加速远大于BAM，ratio才恶化。跨TPU不可迁移该ratio。

MHA细分进一步验证这个解释：

| MHA路径 | 完整step | attention core | QK | softmax | AV | Q/K/V/O投影 |
|---|---:|---:|---:|---:|---:|---:|
| default dense Pallas/flash | 373.26 | 182.85¹ | fused | fused | fused | 39.57 |
| explicit dense dot | 481.56 | 317.76 | 123.14 | 60.59 | 134.03 | 26.97 |
| C256 | 369.15 | 195.67² | 87.02 | 21.98 | 83.84 | 28.90 |

1. Pallas call 168.43 ms + layout/transpose 14.39 ms；custom call没有可靠的XPlane FLOP
   metadata，不能把其0 TF读成零计算。
2. QK/softmax/AV合计192.84 ms，chunk slice/update等其余2.83 ms。C256比显式dense dot
   快30.5%，主要来自不计算因果上三角；但其core比Pallas dense慢约7%，完整图只打平。

BAM自身dense→C256的scope变化如下。scope wall是XPlane算子时长之和，不是互斥的
critical-path分解，不能机械相加为完整step；但同scope配对可定位抵消项。

| BAM/共享部分 | dense | C256 | C256 - dense |
|---|---:|---:|---:|
| **完整step** | **684.12** | **592.28** | **-91.84 ms (-13.4%)** |
| MHA QK+softmax+AV | 264.15 | 186.54 | -77.61 |
| mix alpha总计 | 103.41 | 89.66 | -13.74 |
| ├ alpha contraction | 88.67 | 53.48 | -35.19 |
| └ diagonal update | 7.38 | 33.43 | +26.06 |
| fetch M | 9.47 | 4.81 | -4.66 |
| read fetched M | 23.20 | 37.01 | +13.82 |
| read local M for QK | 15.56 | 17.71 | +2.15 |
| write M | 22.05 | 20.99 | -1.06 |
| **五个BAM顶层scope合计** | **173.69** | **170.19** | **-3.50 (-2.0%)** |

C256另有50.18 ms未落入上述语义scope的chunk scaffolding：mask/select 27.59 ms、slice
13.52 ms、dynamic update 5.41 ms、AbsV源压缩3.15 ms、其余0.51 ms。即目标
`mix_alpha` contraction确实省35.19 ms，但重复的diagonal更新、fetched read及chunk
scaffolding抵消了大部分BAM专属收益；整步91.84 ms收益主要来自共享MHA核心少77.61 ms，
而不是BAM顶层scope只少3.50 ms。这直接促成了下面的C256实现优化。

### C256 BAM implementation optimization

同一UC1a `v6e-1`、六层、`B=32,T=2048`、step 10--14 XPlane逐项累积；每一行只增加
表中所述改动。

| 路径 | Configuration class | Runtime commit | XPlane step | vs 前一项吞吐 | vs legacy吞吐 |
|---|---|---:|---:|---:|---:|
| legacy | `BamV2QChunk256SixLayerProfile` | `36ebca4` | 592.26 ms | — | — |
| 去掉chunk-local remat | `BamV2QChunk256NoRematSixLayerProfile` | `821dc8d` | 536.81 ms | +10.33% | +10.33% |
| 拼接全部 `Mbar` 后只read一次 | `BamV2QChunk256DeferredReadSixLayerProfile` | `821dc8d` | 521.43 ms | +2.95% | +13.58% |
| 预计算diagonal mask/select，消除scatter | `BamV2QChunk256DiagSelectSixLayerProfile` | `821dc8d` | 497.61 ms | +4.79% | +19.02% |
| template mask + concat输出 | `BamV2QChunk256OptimizedSixLayerProfile` | `821dc8d` | **494.57 ms** | **+0.62%** | **+19.75%** |

关键scope由legacy→optimized变化：`bam_total` 170.13→112.47 ms，`mix_alpha`
89.25→44.27 ms，diagonal update 33.41→0.03 ms，fetched read 37.26→25.23 ms，
copy 30.93→4.29 ms；编译日志的临时buffer估计由14.08降至8.68 GB。逐chunk加法mask
把显式select挪进softmax fusion，但整步497.99 ms，反比optimized慢0.69%，故删除。
代数式diagonal correction会因bf16重结合产生1.39e-3前向相对误差及最高4.15e-3梯度
相对误差，也未采用。

commit `165b55b`的packed-segment同参验证中，optimized与legacy前向逐元素一致；loss及参数
梯度差异不超过9.41e-4相对量级，来自去掉冗余remat后的bf16 reduction lowering。最终
full-24目标TPU结果统一更新到上面的权威表。

Recheck artifacts: `/data0/xd/bam_diagnostics/qchunk_v6e_recheck_final/`；matched-dot和
中间分析文件位于`/data0/xd/bam_diagnostics/qchunk_v6e_recheck_complete/`。
本轮实现优化位于`/data0/xd/bam_diagnostics/bam_c256_opt/v6e/`。

### Strict `BamAttention` MHA-only control

Commit `775a938`增加`bam_mha_control=True`：保留`BamAttention`同款QKV、RoPE、
QK/softmax/AV和输出投影，但不创建BAM参数，不分配或传递M，也不执行任何BAM读写。
三路六层`v6e-1`配对具有完全相同的57-leaf参数树与逐项初始化；训练loss仅有bf16数值级
差异，XPlane中两个control的`bam/*`算子数均为零。

| MHA路径 | Configuration class | XPlane step | 稳态日志 | 相对变化 |
|---|---|---:|---:|---:|
| `Attention(dot_product)` | `Llama2MediumDotProductSixLayerProfile` | 481.50 ms | 2.055 | 基准 |
| `BamAttention` dense | `BamMHAControlDenseSixLayerProfile` | 503.17 ms | 1.968 | step time +4.50% |
| C256, legacy inner-remat | `BamMHAControlQChunk256SixLayerProfile` @`775a938` | 413.99 ms | 2.390 | +21.54% throughput vs自身dense |

普通`Attention(dot_product)`复现旧值481.56 ms。它比BAM dense control快4.5%，来自
不同的等价MHA lowering：前者使用五维GQA=1及未归一化`exp→AV→除sum`，后者使用
显式`softmax→AV`；不是残余BAM开销。C256 control也比通用`accelerator.QChunk`旧值
369.15 ms慢12.1%。commit `7d673c0`的同机2×2消融将其定位为batched segment mask与
重复inner-remat的编译交互：

| Configuration class | segment mask | inner remat | v6e-1 step |
|---|---|---:|---:|
| `BamMHAControlQChunk256SixLayerProfile` @`775a938` | batched | on | 413.99 ms |
| `BamMHAControlQChunk256SixLayerProfile` @`a1ad13f` | batched | off | 371.51 ms |
| `BamMHAControlQChunk256SharedMaskSixLayerProfile` | shared causal | on | 371.59 ms |
| `BamMHAControlQChunk256SharedMaskNoInnerRematSixLayerProfile` | shared causal | off | 371.51 ms |

修复提交`a1ad13f`的同机通用`Llama2MediumGQChunk256SixLayerProfile`复测为368.84 ms，
固定4D control为371.51 ms（+0.73%）。关闭inner-remat消除了原差距的94%以上；再把
control换成通用QChunk同款五维singleton-GQA core后为368.69 ms（已打平），验证余差来自四维自定义contraction与
五维GQA lowering/layout细节。shared mask会丢失packed Pile的segment隔离语义，不能
作为训练修复；正确选择是保留batched segment mask并去掉嵌套在整层remat内的
chunk-local remat。

与同型TPU、同六层图的完整BAM结果配对：

| 路径 | MHA-only configuration | MHA-only | Full BAM configuration | 完整BAM | BAM增量 |
|---|---|---:|---|---:|---:|
| dense | `BamMHAControlDenseSixLayerProfile` | 503.17 ms | `BamV2DenseSixLayerProfile` | 684.12 ms | +180.95 ms / +35.96% |
| C256, legacy | `BamMHAControlQChunk256SixLayerProfile` @`775a938` | 413.99 ms | `BamV2QChunk256SixLayerProfile` | 592.28 ms | +178.29 ms / +43.07% |

dense→C256时BAM绝对增量仅少2.66 ms，与旧scope分析的五个BAM顶层scope只少3.50 ms
一致。因此C256的主要收益来自共享MHA核心，几乎未降低BAM专属读写成本；相对BAM
overhead反而上升。Artifacts：
`tpu-ag:/home/lishengping/xd/projects/diagnostics/bam_mha_control/`。

Full-24结果只保留在上面的`Full-24 MHA/BAM throughput`权威表中，不在此重复快照。

Artifacts:

- fixed C256 controls: `/data0/xd/bam_diagnostics/c256_control_fix/`
- v6e: `/data0/xd/bam_diagnostics/qchunk_profiles/` and
  `/data0/xd/bam_diagnostics/qchunk_profile_v6e_local/`
- v5p-16: `/data0/xd/bam_diagnostics/qchunk_profile_1f40820/`
- full-24 v5p-16 traces: `/data0/xd/bam_diagnostics/qchunk_full_v5/`; complete XPlane files
  are retained on `tpu-ag:/home/lishengping/xd/projects/qchunk_full_v5/`.
