# BAM 与 Transformer / MHA 架构图

目标配置：`BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan`。

图基于当前工作树的配置继承链与实际实现，不把台账中的历史 runtime `f6af33c` 当作当前源码快照。绘图时的 Git HEAD 与各源文件 SHA256 见 `source_manifest.json`；源文件可能有其他任务的未提交修改。此次只新增此目录，没有修改模型或实验配置，也没有启动训练或 TPU。

## 文件

- `bam_transformer_interaction.svg` / `.pdf` / `.png`：主图，Transformer 主干、Local/Fetch 两种 MHA 接入方式、公共写回。
- `bam_read_write_details.svg` / `.pdf` / `.png`：矩阵双向读取、完整/压缩视图、head routing、Fetch 与写入公式。
- `bam_architecture.pdf`：两张图合并为两页矢量 PDF。
- `render_architecture.py`：可修改的生成源码，仅依赖 Matplotlib；SVG 保留文字，PDF 嵌入字体。

重新生成：

```bash
python experiments/bam_llama2_medium/architecture_shared_read_llf/render_architecture.py
```

图内使用英文标签和数学符号，便于论文、汇报及后续编辑。下面用中文说明关键交互。

## 读图

蓝色为 Transformer，青色为 BAM 读取与路由，橙色为 BAM 写回。

模型包含 24 层，按 `[L,L,F] × 8` 执行。每层都有完整的因果 MHA 和 SwiGLU；L 是 BAM 的当前 token 读取方式，不是滑窗注意力。`scan` 是执行机制，各层参数独立。

每层有两条深度方向的信息流：残差向量 `h` 与每个 token 自己的矩阵 `M`。进入 attention 的 `x = RMSNorm(h)` 同时用于普通 Q/K/V 投影、BAM 读键/门/路由投影、写入地址和写门。FFN 更新残差流，不直接读取或更新 M。

1. 所有层：在完整 `M[t,32,32]` 上做 rank-1 factorized LocalQK 双向读取，形成 16 个 head 的 ΔQ、ΔK，加在普通 Q/K 的 RoPE **之后**。本配置关闭 QKNorm，也没有 Partial RoPE。
2. L 层：将 address 轴从 32 压缩到 8，得到 `S=M P_c`。每个 head 的同一个 pre-gate 双向读结果 R，分别经过独立的 V gate 和 O gate，注入标准 V 与 attention head 输出。注入 V 的内容会随普通 MHA 权重传播到后续 token。共享的是该层内 O/V 的读结果，不是它们的 gate，也不是跨层参数。
3. F 层：普通 V 投影参与 MHA；BAM 用由 x 生成的有符号 head mixture 混合已有 MHA attention weights，形成一条矩阵传输路由。self diagonal 设为 1，跨 token 的 route 不再做 softmax。路由对各源 token 的压缩矩阵 S 加权求和，再做 query-conditioned 双向读取和门控，将结果加入 head 输出。
4. 所有层：融合后的 O（在 W_O 之前）的前 32 个坐标，经 RMS 归一化后成为写入 data；由本地 x 经 `1024 → 256 → GELU → 16×32` 投影得到 address（准确算子顺序是 down → GELU → up + bias），每头 address 也做 RMS 归一化。按 head 写门加权 outer product、对 16 个 head 求和，累加到完整 M，传给下一层。

## 维度与符号

| 项 | 本配置 |
|---|---|
| Residual width | 1024 |
| MHA | 16 个 Q heads，16 个 KV heads，每头 64 维 |
| FFN | SwiGLU，intermediate width 2816 |
| 初始状态 | `M_0=0`，每个 token 一个 `32×32` 矩阵 |
| LocalQK | 完整 M，独立 Q/K 读键，rank 1，32 data + 32 address |
| Local O/V 与 Fetch | 压缩 M 视图 `32×8`；16 个读取头 |
| 压缩读结果 | 每头 `[32 col/data, 8 row/address, 24 zeros]`，直接放入 64 维 head |
| 跨层持久状态 | 始终是完整 `32×32`，无 whole-matrix read RMSNorm |
| 读取 gate | side-specific `2×sigmoid(logit)`；Local O/V 各自有 row/col gate |
| 写回 | full M 累加，retention λ=1，无动态遗忘 |
| C256 | attention query chunk size=256，不改变全局因果语义 |
| C8 | BAM read/cache view 的 address 压缩宽度=8 |

`M[k,v]` 中 k/U 是 data，v/P_loc 是 address，不应与标准 attention 的 K/V 语义混淆。列读 `M r_v → k`；行读 `Mᵀ r_k → v`。图中的 `RMS` 表示按相关向量坐标归一化；batch 维、epsilon 和 dtype cast 为可读性省略。LocalQK 公式中的 z 包含 side gate，而通用双向读取示意的 z 展示归一化读键收缩本身。

图按有效序列 token 绘制。实际 MHA/Fetch 同时应用因果和 segment mask；Fetch diagonal-one 是实现中覆盖路由对角线的操作。压缩视图体现该结构的 M-cache 设计，但当前 `BamAttention.__call__` 限定 train mode；本图不声称已有可运行的 prefill/decode cache 实现。

## 源码依据

| 内容 | 入口 |
|---|---|
| 配置与继承 | `MaxText/exp.py`：目标类、`BamLocalFetchBase`、`BamLlama2MediumV2`、`BamLlama2MediumV1`、`Llama2Medium` 及其继承链 |
| Q/K、V、O 接入时序 | `MaxText/layers/attentions.py`：`BamAttention.__call__` |
| MHA 权重与 Fetch 复用 | 同文件：`_attention_op`、`_dynamic_bam_fetch_mix_weights`、`_bam_fetch_op` |
| LocalQK | 同文件：`_read_local_qk`、`factorized_head_bam_read`、`_add_local_qk` |
| 压缩、共享 read 与 gate | 同文件：`_compress_full_fetch_state`、`_read_fetched_m`、`_gate_local_output`、`_pack_fetched_bam_heads` |
| 矩阵写入 | 同文件：`_write`、`_update_bam_matrix` |
| 主干、两条 carry 与 LLF | `MaxText/layers/fusion.py`：`SubDecoderLayer`、`FusionDecoderLayer`、`BamLayerPair` |
| M 初始值 | `MaxText/layers/models.py`：`initial_bam_matrix` |
| QKNorm、SwiGLU 默认值 | `MaxText/configs/base.yml` |

验证范围：按上述实际前传核对图中的路径、读取顺序、维度和共享关系；生成 SVG/PDF/PNG，检查文字画布边界并目视检查两图。没有运行训练或模型数值测试。
