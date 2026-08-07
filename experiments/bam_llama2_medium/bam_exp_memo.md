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
