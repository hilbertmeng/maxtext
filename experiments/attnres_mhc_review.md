# Full AttnRes / mHC：实现与论文对照

工作树 `/home/mengqy/Projects/Bam/maxtext`，分支 `mqy/refactor-bam-hc`，
当前基于 `origin/refactor-bam` 的 `0c7e097`（2026-11-07 rebase；初始实现基线为 `17a78b3`）。
初版仅做 CPU 验证；2026-11-07 按用户要求撤销两个 Llama 正式实验，改为串行 Pythia mHC，训练状态见下文。

## 入口与范围

- `MaxText/exp.py`：可复用的 `AttnResConfig`、`MHCConfig`；
  正式实验 `PythiaMediumPropT2048SerialMLPmHC(MHCConfig, PythiaMediumPropT2048SerialMLP)`；
  两个原 Llama 正式实验已删除，测试仅用 mixin 临时组合 Llama 基线。
- `MaxText/layers/residual_connections.py`：深度聚合、mHC 路由/更新、跨层状态。
- `MaxText/layers/fusion.py::SubDecoderLayer`：在已有 ATTN、MLP 前后接入新连接。
  复用原来的 RoPE、因果/packing mask、KV cache、Attention、SwiGLU、分支 RMSNorm 和权重初始化。
- `MaxText/layers/models.py::Decoder`：仅在 `residual_connection != 'none'` 时走新 stack；
  使用原来的 embedding 和最终 RMSNorm / LM head / loss。
- `MaxText/residual_config.py`：兼容性检查和 mHC 参数预算；`base.yml` 提供解析器字段。

只实现 **Full AttnRes**，没有 Block 模式。连接使用 unscanned 层循环，
支持 remat、训练、prefill 和逐 token decode。显式拒绝 scan、pipeline、BAM/Mudd、MoE、
MTP、deep embedding、变宽 MLP 等未经组合验证的配置。配置 mixin 可被其他兼容的 dense fusion 或串行 Pythia 类继承；Pythia 必须使用 fusion 路径。

## 官方来源

- [AttnRes 论文](https://arxiv.org/abs/2603.15031)：Full 公式、embedding source、最终聚合、
  §5 中 pseudo-query 必须零初始化的说明。
- [MoonshotAI 官方仓库](https://github.com/MoonshotAI/Attention-Residuals)：
  检查时公开论文、图和 README 中的 Block 伪代码，没有完整训练源码。
  本实现按论文的 Full 定义编写，没有把 Block 的组内累加当作 Full。
- [DeepSeek mHC 原论文](https://arxiv.org/html/2512.24880v1)：§4.2 的路由约束和 Appendix A.1 的
  n=4、alpha 初值 0.01、Sinkhorn 20 次。
- [mHC ICML 正式版](https://raw.githubusercontent.com/mlresearch/v306/main/assets/xie26aa/xie26aa.pdf)：
  第 17 页 Appendix A.6 / Algorithm 1 明确给出零投影、循环选路的 pre bias、
  residual 对角/非对角 bias、post bias 和 pre epsilon。controller 初始化现按此版本实现。
- [DeepSeek 官方 TileKernels 参考](https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/torch/mhc.py)：
  `mhc_pre_norm_fn_ref`、`mhc_pre_split_mixes_ref`、`sinkhorn_normalize_ref`、`mhc_post_ref`。
  [DeepInfer 接入](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/v4_modules/mhc_ops.py)
  可作为官方推理调用路径参考。
- [Depth Memory 固定版本 `9a6865f`](https://github.com/davidgonmar/depth-memory/blob/9a6865feeb43a153a4501cc9a6457e047ffef3c3/pretrain/OLMo-core/src/olmo_core/nn/hyper_connections.py)：
  `MHCController.reset_parameters` 对照 A.6；`MHCLearnedStreamReduce` 提供 V4 风格的
  最终读出及其初始化。后者是独立适配，不是原始 mHC Algorithm 1 的必需部分。

## Full AttnRes 前传

每个 token 独立在深度轴上聚合。记 source 列表 `S=[e, a0, f0, a1, f1, ...]`：

```text
k_i = RMSNorm_key(v_i)                 # 带可学习 gain，只归一化 key
p_i = softmax_over_sources(q_l · k_i)   # 不除 sqrt(D)，不跨 token 做 softmax
h   = sum_i p_i * v_i                  # value 是原始 source
a_l = Attention(RMSNorm_attn(h))
S.append(a_l)
h   = AttnRes_mlp_l(S)                 # 与 ATTN 的 query/norm 独立
f_l = MLP(RMSNorm_mlp(h))
S.append(f_l)
```

每层 ATTN 前、MLP 前各聚合一次；最后再用独立 `output_connection` 聚合所有 source，
随后进入原来的最终 RMSNorm 和 LM head。保存的是各分支输出，**不是累计 residual state**。
第一个 ATTN 只有 embedding 一个 source，直接返回 embedding，不创建数学上无效的 query/norm 参数。

初始化：所有 pseudo-query **严格为 0**，key gain 为 1，因此初始权重是 source 的均匀平均。
query=0 时 key gain 的首步梯度为 0 是公式自然结果；query 本身仍可学习。
RMS epsilon 使用 `attnres_norm_epsilon=1e-6`；分支 norm 仍用 Llama2Medium 的 epsilon。
路由的 RMS 统计、score、softmax 和加权求和用 FP32，再转回模型 dtype；source 持久保存为模型 dtype。

## mHC 前传

状态 `X` 为 `[B,T,n,D]`，embedding 复制成 n 路。ATTN、MLP **分别有独立 router**。
每个 router 归一化整个 `n*D`，不逐 stream 分别归一化：

```text
z = RMSNorm_affine(flatten(X))
u = z @ Phi                           # Phi: [nD, n²+2n]
pre_logits, post_logits, res_logits = split(alpha * u + bias)
pre  = sigmoid(pre_logits) + mhc_pre_epsilon
post = 2 * sigmoid(post_logits)
R    = Sinkhorn(res_logits.reshape(n,n))
h    = sum_i pre_i * X_i
y    = F(RMSNorm_branch(h))            # F 为已有 ATTN 或 MLP
X'_j = sum_i R_ij * X_i + post_j * y
```

`R[source,destination]` 的存储方向与官方 `mhc_post_ref` 一致。
如果按论文列向量写法表示，所乘矩阵是这里存储的 R 的转置；测试使用非对称循环矩阵，
避免双随机矩阵行列和都为 1 掩盖转置错误。
ATTN 后的完整 n 路状态直接供同层 MLP 路由；MLP 后完整传到下一层，过程中不压成单路。

Sinkhorn 与官方稳定化参考顺序对应：

```text
A = softmax_rows(logits) + eps
A /= column_sum(A) + eps
重复 19 次：A /= row_sum(A)+eps；A /= column_sum(A)+eps
```

有限迭代与 epsilon 下是近似双随机，而非数学上的精确投影。
router、Sinkhorn、pre 加权读和 residual/post 更新用 FP32；读出 h 和更新后的 X 转回模型 dtype。
这对应官方 BF16 参考在 pre-read / post-update 的转换边界；实现也允许 FP32 用于核验。

最后读出默认使用 `MHCLearnedStreamReduce`，不是四路平均：

```text
z = RMSNorm_affine(flatten(X))           # head 独立的 norm gain，长度 nD
g = sigmoid(alpha_h * (z @ Phi_h) + b_h) + mhc_pre_epsilon
h = sum_i g_i * X_i                     # 独立 sigmoid gate，不做 softmax
logits = original_lm_head(original_final_RMSNorm(h))
```

`Phi_h` 为 `[nD,n]`，`alpha_h` 为 `[1]`，`b_h` 为 `[n]`。head 使用 FP32 归一化、
投影和加权求和，输出转回 residual dtype；默认参数 dtype 仍是 FP32。
`mhc_learned_head=False` 可切回无参数 mean adapter，便于独立比较 head 的作用。
数学和初始化对照 Depth Memory 的 V4 风格 head，没有复制其 CUDA kernel 的 TF32 舍入。

## mHC 初始化：A.6 controller 与 V4 风格 head

| 项目 | 默认值 | 来源 / 含义 |
|---|---|---|
| stream 数 n | 4 | 原论文 Appendix A.1 |
| 三组 alpha | 各 0.01，可学习 | 原论文 Appendix A.1 |
| Sinkhorn 次数 | 20 | 原论文 Appendix A.1 |
| pre / post 非线性 | sigmoid / 2×sigmoid | 原论文 §4.2，官方参考一致 |
| controller Phi | 严格为 0 | ICML Appendix A.6；`mhc_router_init_std=0.0` |
| affine norm gain | 全 1，长度 nD | 本次启用；官方 forward 支持可选 norm weight |
| pre 静态 bias | 选中一路 +3，其余 −3 | A.6；ATTN 为 `(2*l)%n`，MLP 为 `(2*l+1)%n` |
| post 静态 bias | 全 0 | A.6；零投影下 post 严格为 1 |
| residual 静态 bias | 对角 +3，非对角 −3 | A.6；经 Sinkhorn 后接近单位阵，非严格 identity |
| router RMS epsilon | 1e-6 | Llama 适配选择；原论文训练表的 norm epsilon 为 1e-20 |
| Sinkhorn epsilon | 1e-6 | 官方 TileKernels 稳定化参考；原论文理想公式无此数值项 |
| pre 额外 epsilon | 1e-6 | ICML A.6 / Algorithm 1 明确包含；head sigmoid 也使用此 epsilon |
| head Phi | Normal(0, 0.02²)，`[nD,n]` | Depth Memory 的 V4 风格适配 |
| head alpha / bias | 0.01 / 全 0 | 同上，与 controller 的三组 alpha 独立 |
| head RMS | 独立 affine gain 全 1，epsilon=1e-6 | 同上；head 始终带 affine，`mhc_norm_affine` 仅控制 controller |
| 最终 n 路到单路 | 默认 learned sigmoid read | `mhc_learned_head=False` 则使用 mean adapter |

因此本实验复现的是 **mHC 的连接方程和约束**，不是复刻 DeepSeek 的全部训练配置。
controller 的初始动态项严格为零，因此 pre 选中一路约 0.952575，其余约 0.047427，post 为 1。
四路 embedding 相同时，首个 pre 读出约为 `1.094856 * e`，随后分支 RMSNorm 处理其尺度。
这不表示有限 epsilon、BF16 舍入下与普通残差逐元素相同。
零 Phi 下 controller alpha 和 norm gain 的首步梯度为零是公式的结果；Phi 仍可获得梯度。
head 的 Phi 随机初始化，因此 head 的初始 gate 通常只是接近 0.5。
Llama 组合测试使用 `init_method='006normal'`；正式 Pythia 实验使用继承的 Pythia 初始化，
包括小初始化 embedding、按深度缩放的 Attention/MLP 输出投影和零偏置。没有套用论文 MoE/MLA 参数。

新增参数继续遵循基线的 weight-decay 规则：`*scale`、`*bias` 不 decay；
controller/head 的 Phi 和 AttnRes query 会 decay。
两个正式配置 dropout=0；若显式开启，新连接对 ATTN / MLP 的写入分支分别 dropout，
MLP 内部的 dropout 仍由现有 MlpBlock 控制。
分支参数路径增加了 `residual_stack`，Flax 按路径折叠随机种子；因此保留的是初始化分布和顶层 seed，
不保证 ATTN/MLP 的每个初始权重与基线逐元素相同。新结构也不能直接按旧模型的参数树恢复 checkpoint。

## 参数预算与训练配置

仓库要求非微小额外参数从 MLP 抵扣，因此 mHC 按实际 D、n、L、激活分支数计算抵扣，
不是在实验中硬编码一个只适合 Medium 的宽度。可用 `mhc_match_parameter_budget=False` 取消抵扣。

| 实验 | 有效 MLP 宽度 | 完整参数量 | 对 Llama2Medium 的差值 |
|---|---:|---:|---:|
| Llama2Medium | 2816 | 411,616,256 | 0 |
| Llama2MediumAttnRes | 2816 | 411,714,560 | +98,304（约 0.024%） |
| Llama2MediummHC | 2749 | 411,613,461 | −2,795（约 −0.0007%） |
| Llama2MediummHC，`mhc_learned_head=False` | 2749 | 411,592,976 | −23,280（约 −0.006%） |

mHC 每 router 参数量为 `nD(n²+2n) + nD + (n²+2n) + 3`，共 48 个 router，
新增 4,916,496；learned head 的 `nD*n + nD + n + 1` 另加 20,485，合计 4,936,981。
MLP 减少 67 个 hidden unit，抵扣 4,939,776，仍是最接近预算的整数宽度。
关闭 learned head 时也会从预算中移除其开销；不同尺寸下的 MLP 抵扣重新计算，不硬编码 67。
AttnRes 有 48 个非平凡读出，每个 query+gain 共 2D 参数。首个 ATTN 读出不创建参数，末端独立读出计入。
表中参数由完整 D=1024、L=24、vocab=50432 的 `jax.eval_shape(model.init)` 核验，没有实际分配大权重。

除残差架构和上述预算调整外，两实验保留基线数据、RoPE、学习率、WD、初始化 seed、步数、checkpoint 等配置。
2749 未按 TPU 矩阵 tile 对齐；这里优先严格匹配参数，实际吞吐需要 TPU 测量。
目前是可微的 JAX 参考实现，没有移植官方 GPU 融合 kernel；Full AttnRes 保留全历史，深度聚合计算随层数平方增长。

## 验证与复现

固定 CPU Python：`/home/mengqy/.cache/bam-diag-cpu-py312/bin/python`（JAX 0.8.1 / Flax 0.12.1）。
2026-11-07 对齐 A.6 初始化并加入 learned head 后，相关测试 **15/15** 通过（85.46 秒），
包含新增 head 的公式/梯度、默认零投影的循环选路、mean 开关及更新后的完整参数计数；
日志：`~/.local/state/maxtext-cpu-tests/20261107T033758Z-1442517`。
2026-11-07 rebase 至 `0c7e097` 后，新增测试 **12/12** 再次通过（90.26 秒）；
日志：`~/.local/state/maxtext-cpu-tests/20261107T031415Z-1404934`。
2026-11-05 验证结果：新增 **12/12** 通过，已有 BAM **47/47** 通过；
完整尺寸、默认 full-remat 初始化的抽象参数计数另行复核通过。
日志目录分别为 `~/.local/state/maxtext-cpu-tests/20261105T101729Z-2473167`、
`20261105T101150Z-2461955` 和 `20261105T101807Z-2473961`（后两者在同一父目录）。
新增 `MaxText/tests/residual_connections_test.py` 覆盖：

- 零 query 均匀平均、仅 key 归一化、深度 softmax、value/各 source 的梯度。
- mHC 与官方前传公式的独立 NumPy 对照、参数有限差分梯度、非对称矩阵方向、极值有限性和行列和。
- A.6 零投影/循环 bias、首步梯度、learned head 的独立 NumPy 对照/有限差分和初始化。
- 默认参数初始化、MLP 预算、非法组合报错、完整参数量。
- 两模型真实前向/反向、因果性、跨层 source / stream 持续性、末层 ATTN/MLP 梯度。
- remat 开关的 loss/gradient 一致性、BF16 前向有限、prefill + 两个 decode token 与整段计算一致。
- learned/mean head 开关、浅层尺寸下 head 预算对 MLP 抵扣的影响、完整模型参数计数。

运行新增测试：

```bash
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$PWD" \
  --test-file MaxText/tests/residual_connections_test.py \
  --python /home/mengqy/.cache/bam-diag-cpu-py312/bin/python --jobs 2 --cores-per-job 4
```

现有 BAM 回归：

```bash
MAXTEXT_CPU_PYTHON=/home/mengqy/.cache/bam-diag-cpu-py312/bin/python \
MAXTEXT_CPU_TEST_JOBS=4 MAXTEXT_CPU_TEST_CORES=4 \
  .agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh "$PWD"
```

未测 TPU 编译、速度、峰值内存、训练收敛；CPU 检查不能替代这些验证。


## 串行 Pythia mHC 实验（2026-11-07）

- Worktree `/home/mengqy/Projects/Bam/maxtext`，分支 `mqy/refactor-bam-hc`。
- RUN / 配置 `PythiaMediumPropT2048SerialMLPmHC`，计划 13,500 step。
- 直接基线：已完成的 `PythiaMediumPropT2048SerialMLP`（d434e69）和
  `MuddPythiaMediumPropT2048SerialMLP`（3538f6c），不重复训练。
- D1200 / L18 / 16x75 heads / RoPE18 / GELU；T2048、global batch256、
  legacy2048 数据、优化器/LR/seed/schedule/checkpoint 策略继承原串行 Pythia。
- 4 streams、36 个 router、20 次可微 Sinkhorn、零 Phi / 循环 pre-bias、learned head。
  不加入 Mudd 或 BAM。原 Pythia LayerNorm 和 attention/MLP 偏置全部保留。
- `pythia_use_fusion=True`：复用已验证的串行 Pythia fusion；`decoder_block='pythia'`
  保证归一化/初始化语义不变。显式拒绝并行 Pythia 组合。
- `scan_layers=False` 是当前新连接 stack 的执行要求；generic training health 开，
  internal NN/BAM health 关，TB GCS 每500步轮换。
- 4,344,977 新参数 = 3.017 W_Q；按包含 hidden bias 的每单位 FFN 成本43,218抵扣101。
  有效 FFN4800→4699。完整 abstract model.init 核验：基线432,360,000，
  mHC432,339,959，余差−20,041（−0.00464%）。
- 旧 Mudd 的 router 为额外参数且采用逐层变宽 FFN，因此它是历史效果参考，
  不是严格匹配总参数的连接单因素对照。
- CPU：连接公式/初始化/Sinkhorn正常反传/整模型remat/bf16/因果性/缓存/Pythia预算
  17/17 通过；Pythia fusion 前向和梯度对齐回归4/4；summary writer5/5。
  日志目录：`~/.local/state/maxtext-cpu-tests/20261107T040349Z-1473147`、
  `20261107T040519Z-1474942`、`20261107T040511Z-1474493`。
- 跑前判断：预期优于串行 Pythia，对 Mudd 胜负无强先验；未经 TPU 优化的四流路由
  预期慢于原 Pythia，先验证实际吞吐/内存再解释曲线。
  初始误以为历史health=False代表健康指标未计算；实际旧runtime无条件计算，见下文核验。
- 基线 loss cache：`training_reports/cross_scale_20261025/loss_cache/`（Bam 项目根目录）。
- ag2 编排使用用户目录 `/home/lishengping/mengqy/tpu_mhc_20261107` 的脚本副本；
  不修改 ag2 `/home/lishengping/xd/`。
- runtime `695c28d` 已提交并推送；sealed-config gate通过；v5p-16 AOT已完成并验证。
- 2026-11-07 04:14 UTC 开始 Spot v5p-16 排队，主区UE5a、候补EW4b；
  TPU `xd-v5p-16-6110701-maxtext`。尚未验证 FIRST_STEP，不宣称训练已成功启动。
- 2026-11-07 05:23 UTC 按用户要求扩展候补至`us-central1-a`；通过MODE=attach替换
  外层控制器，保留UE5a 04:14:40、EW4b 04:20:06原队列身份和时间，新增UC1a
  05:23:34 Spot队列。三地仍为同一v5p-16 RUN/runtime/AOT，不并行训练多个副本。
  UC1 legacy2048的286个TFRecord与UE5副本按相对名称、size、CRC32C全部一致；
  该旧数据集不使用TruePile的manifest.json检查。启动脚本副本及本地launch.sh均已更新候补策略。
  v5p公开区域参考：https://docs.cloud.google.com/tpu/docs/regions-zones 。
- 权威 RUN registry 在 ag2 用户目录 `tpu_mhc_20261107/run_registry/`；checkpoint区域跟随该记录，
  初始计划 `gs://newproject-1-llm_projects_us-east5/log/PythiaMediumPropT2048SerialMLPmHC/checkpoints/`。
- `mqy-pythia-mhc` report group每2k更新；本地只读观察服务
  `mqy-pythia-mhc-monitor.service` 每10分钟检查，报告loss/gap和速度，4k/8k提醒复核，
  不自动做loss驱动的停训决策。服务代码及状态：
  `/home/mengqy/Projects/Bam/training_reports/mhc_pythia_medium_20261107/`。
- TensorBoard复用6008端口，本地镜像`/data0/mengqy/tensorboard_logs/`，
  两个基线已通过HTTP核验存在`learning/loss`；新mHC按2k报告同步，结束再同步。
  `./tensorboards/`有对应入口。当前用户对`/data0/xd/tensorboard_logs`无写权限，
  未改其权限或共享服务。新训练TB仍写中央区GCS的原生轮换目录。
- 速度先报告稳定步的step/s、token/s及训练时间；历史值为Pythia0.72949、Mudd0.57875 step/s。
  当前Splash1024/fused配置不同于历史控制；两条旧runtime也实际计算了健康指标。
  仓库`perf/step_time_seconds`实际记录step/s，不能将其当秒/step取反解释。


## v6e-16 备选（2026-11-07）

- 用户授权尝试v6e-16；新配置/RUN `PythiaMediumPropT2048SerialMLPmHCV6e16`。
  继承原mHC配置，仅改model_name及per_device_batch_size=16；16个v6e chip对应global batch256，
  与8个v5p chip×32一致。模型/优化器/13500步schedule/数据不变，参数量仍为432,339,959。
- runtime `717ce4a` 已推送并通过sealed-config检查；v6e-16 AOT在保留FLEX_START编译机完成、
  上传并验证manifest。原v5p图不能用于v6e，二者独立封存。
- 原v5p两次分配均在首步验证前被SERVICE回收，第二次发生在安装阶段；未采集到loss，
  且独立确认无committed checkpoint。暂停旧控制器并验证无旧writer，才启动v6e控制器。
  原监控服务已停；旧候补暂留，v6e AOT首步通过后释放。期间若旧候补READY闲置超过10分钟，
  只释放该任务的闲置实例，避免无限待机；清理脚本有registry身份、paused状态及queue identity检查。
- `xd-v6e-16-6110702-maxtext`：UE5a主区，EW4a/UC1b候补，Spot，1秒后扩区。
  2026-11-07 06:03 UTC三地队列均已接受，尚未验证FIRST_STEP。
- 新report group `mqy-pythia-mhc-v6e`；本地服务`mqy-pythia-mhc-v6e-monitor.service`，
  每10分钟检查、每2k同步同step的Pythia/Mudd loss gap与速度。日志仍在TensorBoard6008，
  本地`/data0/mengqy/tensorboard_logs/PythiaMediumPropT2048SerialMLPmHCV6e16/`。
  本地启动/监控/状态：`/home/mengqy/Projects/Bam/training_reports/mhc_pythia_medium_v6e_20261107/`。
- 预期：同global batch下v6e速度应不低于v5p，但四流路由的带宽操作可能限制收益；
  以去掉启动阶段后的实测为准。历史Pythia/Mudd是v5p-16且旧Splash，实际也计算健康指标，
  速度差不能归因于mHC一项。


## Return to v5p-16 (2026-11-07)

The user requested a return to v5p-16. The v6e controller was stopped and all three
candidate zones were verified quiescent. No completed training step, loss history,
or committed checkpoint was found for the v6e RUN. All three v6e candidate queues and nodes have been independently verified absent;
compiled artifacts and storage prefixes are retained.
The v5p-candidate release watcher was terminated before resuming the original RUN,
so it cannot delete resumed v5p resources. The original `PythiaMediumPropT2048SerialMLPmHC`
RUN resumes with runtime `695c28d`, its existing v5p-16 AOT, batch32 per chip/global256,
UE5a primary, and EW4b/UC1a backups. No model or optimizer change is introduced.
Monitoring returns to `mqy-pythia-mhc-monitor.service`; the v6e monitoring service is stopped.


### Unreachable READY allocation before FIRST_STEP

The UC1a node created at 2026-11-07 06:33:42 UTC reported READY, but both workers'
TCP22 connections timed out over private and public IPs from tpu-ag, and public
connections also failed from the local host. The retained compiler accepted SSH
from tpu-ag. The inspected project VPC rules allow SSH; no firewall rules or keys
were changed. This is an unusable allocation, not a measured model/compile slowdown.
Before recycling, the controller was fenced and the launcher log proved no train
process had been launched; loss history and committed checkpoints were absent.
That UC1a node/queue generation was verified released, and the v5p controller was
restarted. UE5a/EW4b retain their accepted queues; UC1a is requested again. The
replacement still uses runtime `695c28d` and its original AOT.
The local monitor now alerts once per lease when a READY host has no first step
and SSH errors after ten minutes. That alert never deletes resources automatically.


### FSDP controller placement fix

A usable UC1a allocation launched the train process at 07:03:46 UTC. Startup then
failed before FIRST_STEP: the model had 432,339,959 global parameters but 58,052,759
parameters per chip, 7.4206% above ideal eight-way sharding, exceeding the unchanged
2% startup guard. The mHC controller and learned-head parameters had no logical
partition annotations, so every FSDP rank replicated them. AOT compilation and the
single-device functional tests did not exercise this runtime guard.

Router/head kernels now use logical axes `('embed', None)` and their affine RMS
scales use `('embed',)`. Routing formulas, initializers, parameter names/shapes,
optimizer settings and global batch are unchanged. The 17 formula/model tests pass.
A new isolated virtual-device test checks the formal full-width model on FSDP8/16:
54,251,759 parameters per chip / 0.3872% overhead on eight devices, and 27,245,459 /
0.8298% on sixteen. Its counterfactual replicated-controller placement reproduces
the actual 58,052,759 / 7.4206% failure exactly. The sharding tolerance remains 2%.
CPU logs: `~/.local/state/maxtext-cpu-tests/20261107T070805Z-1662419` and
`20261107T071046Z-1665147`. Runtime `c1c46834d54f2f0d4f921495b01bf59e18910806`
passed the sealed-config gate and its v5p-16 AOT was published at 07:18:33 UTC.
The old 695c28d/717ce4a artifacts predate this fix and must not be used with the
patched parameter placement.

The retained training allocation was reclaimed during compilation: its queue
reported `SUSPENDED` with `stateInitiator=SERVICE`, and its node was verified absent.
On restart, regional recovery refused the unjournaled UC1a queue left by the earlier
manual recycle. With the controller stopped, the native ownership check and
`Recovery.refresh` rebound the exact 06:53:03.719643199Z queue generation under the
recovery lock. The controller then restarted with the new runtime/AOT; UE5a/EW4b
accepted queues retain their original creation times. Local ten-minute monitoring
is active again.

At the 08:34 UTC status check, the patched RUN had completed cached step709 and
had a verified committed checkpoint at648. The recovery journal records a resumed
first step661 after checkpoint648. It was reclaimed again at08:26:35 UTC and is
requesting Spot capacity in UE5a/EW4b/UC1a; no worker is currently training.
The first steady 40-step sample (steps10–49) has harmonic throughput0.535489 step/s,
median0.536, or280.75k tokens/s. At that rate13,500 steps take7.00 active hours;
remaining work from checkpoint648 takes6.67 hours, excluding queue/preemption
overhead. Historical Pythia/Mudd speeds are0.729488/0.578750 step/s, with older
Splash settings; their old train.py computed health metrics despite health=False.
These are practical runtime comparisons rather
than isolated mHC overhead. TensorBoard files were synchronized locally. To fill
preemption gaps,23 missing period10 points through the native-log horizon709 were
imported from TensorBoard without replacing any native-log values. The48 overlapping
points agreed within4.9e-7. Complete native comparison windows at200/400/600 now show
mHC minus Pythia gaps−0.02335/−0.10271/−0.10370 and mHC minus Mudd gaps
+0.17945/+0.20779/+0.02520. At600 the respective losses are3.641464,3.745159 and
3.616268 (±25, period10, five paired samples). mHC improves on Pythia and nearly
matches Mudd in this early window; longer-term ordering remains unmeasured.

### Same-step training-health check (2026-11-07)

Local artifacts: `training_reports/mhc_pythia_medium_20261107/health/` in the Bam
project root. Native TensorBoard has mHC scalar samples0–740 at period10. Neither
mHC nor the controls has a nonfinite recorded scalar in this inspected early span.
All75 common samples have exactly identical learning rates and valid-token counts.
Both historical controls actually recorded global/per-leaf gradient and parameter
norms. Their train.py at d434e69/3538f6c computes these unconditionally, despite
record_training_health_metrics=False in the saved configurations. Earlier claims
of health-OFF runtime comparisons were incorrect; the correction does not change
the measured speed values or remove differences in Splash/execution paths.

At the native600 window (580/590/600/610/620), mean raw gradient norms are
mHC0.479192, Pythia0.475473, Mudd3.744961. From140 through740, none of mHC's61
recorded raw norms exceeds clipping threshold1; the maximum is0.935374. Initial
raw norms22.493/21.897/23.331 all undergo clipping. Recent650–740 mean mHC raw norm
is0.446398, versus Pythia0.464919 and Mudd2.041242. Thus mHC does not show a sustained
early gradient increase or clipping saturation. Sampled gradients cannot rule out
unrecorded single-step spikes or later-training changes.

The large mHC param_norm is an initialization/parameterization effect: initial
global norms are574.364 versus Pythia382.841. New mHC controller/head leaves have
norm429.054, dominated by one-initialized affine normalization scales and +/-3
routing biases. Removing those added leaves gives pre-update backbone norms
381.846 at0 and409.263 at600, versus Pythia382.841 and410.262. The remaining small
difference includes the deliberately narrower FFN; there is no evidence here of
abnormal backbone norm inflation. Global param_norm is post-update, whereas the
per-leaf total_params values used for this subtraction are pre-update.

At the600 window, squared gradient norms summed over disjoint leaf groups give
mHC18.44% in added controllers/head,44.89% in attention,23.76% in FFN, and8.88% in
the vocabulary projection. Pythia's vocabulary projection accounts for8.63%.
The leaf sums reproduce global gradient norms within7.5e-7 relative error.
Mudd's larger norm is primarily its dynamic router leaves (97.71% of that window's
gradient energy), not evidence of an mHC failure. Token accuracy at600 is
39.126%/38.025%/39.511% for mHC/Pythia/Mudd, consistent with their loss ordering.
Activation RMS, route-weight distributions, optimizer-state/update norms and
Sinkhorn constraint errors are not recorded, so those cannot be assessed from
these TensorBoard records. Continue watching gradient groups and parameter growth
as training resumes; no training change is justified by this early check.

## TruePile4096 MHA mHC experiment (2026-11-07)

User request: add mHC to `BamMHAMediumPropC256TruePile` and train a separate RUN.
Class/RUN: `BamMHAMediumPropC256TruePilemHC`, same feature worktree/branch
`/home/mengqy/Projects/Bam/maxtext` / `mqy/refactor-bam-hc`.
Training TPU ID6110703: `xd-v5p-16-6110703-maxtext`, Spot, UE5a primary and EW4b/UC1a
backups. Preserve the independent Pythia mHC RUN and borrowed compiler.
Use tpu-ag's copied scripts under `/home/lishengping/mengqy/tpu_mhc_20261107`;
do not modify its `/home/lishengping/xd/` tree. Local run artifacts are under
`training_reports/mhc_truepile_medium_20261107` in the Bam project root.

Parent recipe: D1200/L18/H16x75, RoPE74, RMSNorm, SwiGLU3200, normal0.006
initialization, bf16 logits, AdamPax3e-4/WD0.1, seeds9876, full remat, global
batch128 x4096, TruePile4097-token records, original13,500-step schedule.
Keep `bam_enabled=True` and `bam_mha_control=True`: this selects the parent's exact
BamAttention MHA-control implementation, with every layer mode `none` and no M
read/write parameters. Do not substitute generic Attention. Validation now permits
only this pure mHC MHA-control combination; real BAM streams, matrix writes, shifts,
Pallas, tail routes and other incompatible connection modes remain rejected.
The existing SubDecoderLayer already implements that branch; no attention forward
formula changes are required.

mHC inherits the same four-stream,36-controller,20-Sinkhorn, learned-final-head
configuration as the existing Pythia experiment. Unrolled execution is required
by the connection stack (`scan_layers=False`); full remat and health capture remain
enabled, with internal NN metrics disabled and TensorBoard rotation500.
New controller/head parameters4,344,977 =3.017345 W_Q are repaid by67 SwiGLU units
at64,800 parameters/unit. Effective FFN3200→3133. Actual full abstract init counts:
parent432,121,200, mHC432,124,577, difference+3,377 (+0.0007815%).
The FSDP virtual-mesh startup check gives54,055,277 parameters/chip on8 devices
(+0.073506% over ideal) and27,050,327 on16 (+0.157514%), within unchanged2% guard.

The main worktree is still0c7e097, already an ancestor of this feature branch;
no additional main-only train/AOT/optimizer/checkpoint/summary-writer fixes need
porting. The active Pythia mHC runtime has already demonstrated AOT restore and
committed checkpoints on this operational lineage. Relevant launch gates include
the expanded19 formula/model/config regressions, both formal FSDP configurations,
and5 summary-writer tests. New end-to-end coverage includes the real BAM-MHA-control
branch, finite forward/backward, causal mask, full remat gradient equivalence and
bf16, plus full-width parameter counting and rejection of incompatible matrix modes.

Direct comparison runs: `BamMHAMediumPropC256TruePile` (historical0d31a3d,
UE5a~0.714 step/s) and `MuddLlama2MediumPropTruePile` (historical4fee490,
~0.630 step/s, extra router budget). Both native histories are retained as baselines.
Pre-run expectation: mHC should improve on pure MHA; uncertain relative to Mudd.
Expect routing overhead to reduce speed versus pure MHA, subject to the separate
scan/kernel/runtime differences; use measured post-warmup timing, not a claimed
isolated architecture penalty. Dataset manifests agree across all three regions:
TruePile4096/global128/seed9876, extended50,000 steps, original13,500 prefix retained.
Target training stays13,500 steps. Formal training starts only after CPU gates,
sealed-config verification and the exact v5p-16 AOT; actual FIRST_STEP is required.

Preparation passed at runtime `de3d1b1f04229e075af2c4fd409f44f234dcdf28`:
19 model/formula checks (`~/.local/state/maxtext-cpu-tests/20261107T091016Z-1797055`),
both configurations' FSDP placement (`20261107T091017Z-1797072`), and5 writer checks
(`20261107T091102Z-1797947`). All100 regional training-shard size/CRC records agree;
the audit is `tpu-ag:ROOT/truepile_mhc_data_audit.json` with ROOT as specified above.
Original TensorBoard configs also confirm the inherited data/optimizer/init/attention
settings against both historical baselines; their full native loss histories are
imported into ROOT's isolated registry cache.

The borrowed-compiler AOT became ready at09:19:07 UTC, with artifact
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/de3d1b1/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamMHAMediumPropC256TruePilemHC.pickle`.
The AOT-ready guard started the separate native controller, whose registration
matches the full runtime hash and UE5a TruePile4096 path. Group `mqy-truepile-mhc`
was created at09:19:36 UTC with2k public cadence and4k/8k reviews. The shared ChatServer
wake now checks both independent mHC RUNs every10 minutes; the old standalone bot
monitor remains stopped to avoid duplicate public delivery. FIRST_STEP and speed
remain pending allocation; declared_status=running alone is not training evidence.
