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
