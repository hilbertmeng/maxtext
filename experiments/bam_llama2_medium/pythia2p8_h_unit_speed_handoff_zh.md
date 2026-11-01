# 调查交接：Pythia 2.8B H 全1路相对旧 BAM 长训慢约 5–6%

## 目标与当前判断

定位并优化 H 全1路的实际执行成本，优先寻找保持模型数学、初始化和参数预算不变的实现优化。

已确认速度差真实存在，而且 v5p-128 上已有相近差距；目前没有证据证明“这次转成 300B 长训”额外造成了退化。但“历史上就慢”和“落在宽泛的跑前下注范围内”都不能作为停止调查的理由。需要用历史逐项实验和 profile 定位成本，不要仅凭算子数量猜测。

**此前提出的两个主要嫌疑——MLP 拆分／更频繁写 M，以及 NoPE QK60→80——缺少历史实验支持，已撤回优先归因。**详见下文反证。

## 1. 速度现象

相同 global batch 1024、T2048、minimal remat；通用训练健康统计 ON，内部 NN/BAM/shift 激活健康统计 OFF。以下是不同历史租期的记录，不能冒充同机配对实验。

| 拓扑／窗口 | 旧 BAM | H 全1 | H 相对旧 BAM |
|---|---:|---:|---:|
| v5p-256，同为 log step 20–49 | .59384653 step/s | .56232261 step/s | **−5.31%** |
| v5p-128，历史稳态测速记录 | .337699 step/s | .316382 step/s | **−6.31%** |

计时用每步耗时均值的倒数，即 steps/s 的调和平均。新路最初 20–39 步为 .561686、median .565；后续 100–199 步为 .558219、median .562，不能用一次启动停顿解释。新旧早期窗口均包含短暂慢步，不能只剔除新路慢步。

最初汇报用旧路后期约 .585 对比新路 .5617，得到约 −4%；更合适的同窗口比较是上表的 **−5.31%**。跑前中心下注 .60 没有用好已有 v5p-128 证据：按其速度比推算，v5p-256 应预估约 .55636，实际 .56232 比这个估计高约 1.07%。

## 2. 必须优先考虑的历史反证

这两项实验都是之前累计汇总图中的历史选手，各自有直接父路测速：

| 改动 | 直接父路 | 实验路 | 结论 |
|---|---:|---:|---|
| 每第三层 MLP 写 M → 每层约 2:1 普通／小 MLP | 启动约 .326 | .3294–.3295 | 启动约 **+1%** |
| 同上，后期记录 | 约 .329 | 约 .325 | 约 **−1.2%**，没有建立大幅减速 |
| 截断 NoPE QK60 → 完整 QK80，总 Q/K80→100 | SeparateVO 约 .317 | .316706，median .318 | **基本不变** |

均为 v5p-128、global batch 1024、B16/device、minimal remat、相同健康设置；不应把约 1% 的跨租期差异说得过死。

对应配置：

- 2:1：`BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1`，runtime `6c60bfd`；直接父路 `BamPythia2p8BSotaIndependentR768PallasShortTrain`，当时已使用优化内核 `0579479`。
- QK80：`BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysQK80`，runtime `59ef09e`；直接父路是去掉末尾 `QK80` 的 SeparateVO 类，runtime `b79aae7`。

2:1 把逻辑上的前向 MLP 矩阵乘从 64 次增到 126 次，MLP 写 M 从 10 次增到 31 次，但实测吞吐没有相应大幅下降。因此不能把这些计数当成当前 5–6% 减速的因果解释，也不能把逻辑矩阵乘次数当作最终执行 kernel 数。

## 3. 更值得定位的阶段

### A. SeparateVO 这一步已有明确减速

`...MemoryMLPRatio2to1AllR640` → `...AllR640SeparateVOKeys`：约 **.328 → .317 step/s，慢约 3.4%**。

这一步不止一个变化：

- V/O 使用独立动态读地址，但仍共享 R640 read down。
- 三层 block scan 改成单层 scan（31 层 scan + 1 层显式尾层）。
- MLP 宽度重新找平为普通 7808、小 MLP 2944、末层 9984。
- scoped VMEM 预算从 56 MiB 提高到 60 MiB；当时独立 VO read backward 需要约 58.13 MiB，v5p 物理 VMEM 为 64 MiB。

应拆开判断成本主要来自独立 V/O 的 read/reverse，scan 组织／collective 调度，还是宽度与融合变化。预算上调本身不是已经证明的耗时原因。不要简单把整段 −3.4% 全记到 V/O 独立地址上。

### B. 旧长训与完整 QKNorm ShortTrain／内核谱系

旧长训实际 runtime 是 `1ac1183`，QKNorm 尚未覆盖拼接后的完整 Q/K；后续 ShortTrain 修复了 `bam_qk_norm_after_concat=True`，并经历 `2a108de → b902c69 → 0579479` 内核切换。

历史 v5p-128 记录：旧长训配方 .337699；修复后 ShortTrain 初期 minimal 约 .3092；两次内核优化后约 .327–.329。完整 QKNorm 与配套实现可能解释另一部分差距，但这些数字不是一次只改 QKNorm 的严格配对结果，不能据此直接归因。

**当前主干同名旧配置已经带上后来的修复。复现旧长训必须用实际 runtime `1ac1183`，不能只看／运行今天的同名配置类。**

## 4. 已检查／可以排除的简单解释

- **MLP 隐宽未按 128 对齐：已排除。**运行时配置与完整参数形状均确认：普通 7808=61×128、小 MLP 2944=23×128、末层 9984=78×128。
- **global batch 或累积不同：已排除。**v5p-256 为 128 个 JAX device、32 个 worker、B8/device，无梯度累积；每步 2,097,152 个 token。
- **忘记加载 AOT／误用源码：已排除。**32 个 worker 均是指定 runtime，均有 `Loaded compiled function!`，实际首步与 checkpoint 0 已验证。
- **多开了激活健康统计：已排除。**运行配置确认通用健康 ON，内部 NN/BAM/shift 健康 OFF。
- **新 logit scale 多加了一个计算阶段：源码不支持此解释。**此次把已有 query 缩放运算的编译期常数改为 1/sqrt(100)，没有新参数或额外 tensor pass。策略为 `bam_qk_logit_scale_dim='qk_dim'`，自动取实际 Q/K 宽度；旧配置默认 None 保留旧分母。原生/Pallas 实际 attention 输入及所有参数梯度一致性检查已通过。
- **FSDP 大矩阵没有分片：已排除。**完整分片审计通过；每 device 22,968,832 参数，相对理想分片开销 5.939%，是小参数复制造成的，所有大矩阵均正常分片。
- 日志里的 Transformer TFLOP/s 估算没有完整计入 BAM／小 MLP，不能拿新旧打印的 TFLOP/s 直接判断真实 FLOPs 或硬件利用率。

## 5. 建议调查／调优顺序

1. 优先复查 SeparateVO 前后的实际 runtime 和参数形状，构造保持数学不变的 scan 布局对照；同时检查 read/reverse 的 HLO、layout、copy、collective、HBM/VMEM 成本。
2. 拿到同机、同 batch／健康／remat 的 profile，以实际耗时决定先调哪里。不要为证明猜测而重新跑已被历史结果否定的大矩阵。
3. 若热点确在双 MLP，可检验共用输入的第一投影能否合并，或减少 layout／通信碎片；这是待 profile 支持的优化方向，不是已经确认的根因。
4. 性能实现改动应保持 H 全1数学、初始化、参数量、精度与完整 QKNorm 修复。先验证 forward／梯度，再报告配对速度及适用范围；改变 scan 参数树时应明确映射／兼容性，不能直接拿不兼容 AOT 续跑。
5. 小拓扑可用于算子归因，最终需在 B8/device 的 v5p-256 路径确认。小拓扑不应为保持 global batch 而擅自改变每设备 batch。

希望交付：能解释多少耗时的 profile 证据、具体热点、保持数学不变的优化 patch、配对测速和必要的数值验证，而不是只给笼统的架构成本解释。

## 6. 当前训练与复现入口

当前正式训练继续运行，由本会话负责监控；诊断请使用独立资源，不要直接停训、热切换或改其 runtime。

- RUN／EXP：`BamPythia2p8BHUnit300B`。
- 实际 runtime：`fb915b61c7326cd5c680e740c2ff9e4409abd142`，位于 `refactor-bam` 历史；后续主干提交有调查文档和启动元数据。
- TPU：`xd-v5p-256-311101-maxtext`，`us-east5-a`。
- 计划：143000 步，LR 1.6e-4，warmup 1430；报告 3k/8k/13k/...。
- checkpoint：每 1000 步，最新两份，永久保留 3000/13000/.../143000。救援已配套：Orbax 1200s、外层 1300s 触发、600s 救援预算。
- 精确 AOT：`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/fb915b6/jax081-i0ae3f58-c17f538a/v5p-256/s143000/BamPythia2p8BHUnit300B.pickle`；manifest 同路径加 `.manifest.json`。
- 预留 FLEX 编译机：`llm-jax-v6e-1-0` / `europe-west4-a` / worker 0。用共享编译锁和 `prepare_train_aot_on_worker.py`；借用、不拥有生命周期，勿删除。
- 训练侧编排：tpu-ag `/home/lishengping/xd/projects/`；权威脚本 `/home/xd/projects/xd_tpu_scripts/`。按项目 AGENTS 与最新 training/diagnostics skill 操作。

本地直接可读证据：

- 新路：`/data0/xd/bam_diagnostics/h-unit-300b-launch-20261101/`，包括 `runtime.config.txt`、`startup-worker.log`、`startup-speed.json`、`timing-followup.log`、`speed-review.json`、`all-worker-runtime.log`、AOT state/manifest、checkpoint 0 游标与提交标记。
- 旧路相同 20–49 步原日志：`/data0/xd/bam_diagnostics/pythia2p8b-r768-v5p128-speed-20261021/v5p256-matched20-49.log`。
- 旧路速度说明：`experiments/bam_llama2_medium/pythia2p8b_v5p128_speed.md`。
- 2:1、SeparateVO、QK80 完整类与结论：主干 `MaxText/exp.py`。
- SeparateVO/QK80 细节：`experiments/bam_llama2_medium/pythia2p8b_residual_mlp_budget.md`；QK80 原始测速：`/data0/xd/bam_diagnostics/pythia2p8b-residual-mlp-budget-20261025/qk80-launch/speed.json`。
- 完整 QKNorm 和两次内核切换：`experiments/bam_llama2_medium/pythia2p8b_shorttrain.md`。
- 本次长训说明：`experiments/bam_llama2_medium/pythia2p8_h_unit_300b.md`；启动／汇报计划：同目录 `pythia2p8_h_unit_300b_plan.json`。
