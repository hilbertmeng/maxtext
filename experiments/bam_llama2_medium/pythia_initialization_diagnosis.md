# Pythia 初始化与 BAM 有效尺度诊断

当前源码 `6e4002bc`（`/data0/xd/pythia-mediumprop-t2048`）。本诊断只读取实际模块，不改变在训代码、checkpoint 或资源。并行和串行 GELU 参数初始化保持一致，已有逐参数相等回归检查。

## 实际规则

Medium D=1200、L=18。原配方普通 kernel std=.006；Pythia small_init=sqrt(2/(5D))=.0182574186，残差输出 wang_init=2/(L*sqrt(D))=.0032075015。token embedding、BAM 普通非零地址投影、写门和 embedding 写入均使用 small_init；attention/MLP 向量输出使用 wang_init。零初始化的读门、M-shift 门等保持其原有规则。

Pythia LayerNorm epsilon=1e-5 通过全局配置同时影响 BAM 写入 RMS，原值1e-6；读 key epsilon 已显式设1e-4，两边未变。该耦合需要在解释迁移结果时明确，不能把“RMS 会消除尺度”作为假定。

## CPU 初始化测量

实际 `EmbeddingBamWrite`，D1200/H16/K75/V32/rank256，bf16 激活、float32 参数。三个随机种子，每个256个按对应初始化分布生成的 token embedding 行；配对使用相同高斯方向和权重随机种子。只测初始化，不是训练 checkpoint 或真实 Pile loss 的因果测评。

| 设置 | seed U 归一化后 RMS | seed 地址归一化后 RMS | 初始 M RMS |
|---|---:|---:|---:|
| 原 embedding + 原 BAM投影 + epsilon1e-6 | .778264 | .059740 | .018519 |
| Pythia embedding + Pythia BAM投影 + epsilon1e-5 | .963751 | .467838 | .179397 |
| Pythia embedding + 原 BAM投影 + epsilon1e-5 | .766358 | .057513 | .017555 |

完整 Pythia 设置初始 M RMS 为原来的9.6873倍，三个种子倍率9.6935/9.6932/9.6753。embedding 写门均值均约 .1000；该差异来自投影与 epsilon 所处尺度区间，而非 gate 均值改变。原地址归一化前 RMS 约5.986e-5，远小于sqrt(1e-6)，因此 epsilon 不可忽略。Pythia 地址归一化前约.0016875。

完整8组因子组合及原始分种子数据：`/data0/xd/bam_diagnostics/pythia-init/embedding-seed-init.json`；可复用脚本 `probe_embedding_seed_init.py` 与运行日志 `probe.log` 同目录。使用固定CPU环境 `/data0/xd/conda/envs/maxtext-cpu/bin/python`、JAX_PLATFORMS=cpu、PYTHONPATH 指向上述源码，8核运行。

## 判断与待验证方案

初始化成为有定量证据的优先嫌疑，但该探测不证明它已经解释终局收益缩水。原迁移验证覆盖了Pythia投影初始化规则、前向与梯度数值对齐，遗漏了BAM初始矩阵和读写有效尺度的匹配检查。

用户已授权在串行 GELU BAM 上启动两路（2026-10-15），并要求暂停并行 SwiGLU 三路以供热替换。只恢复 BAM 专用投影不足以复现原始初始化链路；最终设计如下。

| 项目 | 串行 GELU 父路 | BAM 初始化修复 | 全量原初始化对照 |
|---|---|---|---|
| token embedding std | .0182574 | .006 | .006 |
| BAM 专用非零 kernel std（含 seed、读、attention/MLP 写门和地址） | .0182574 | .006 | .006 |
| 标准 QKV / 主 MLP 输入 / unembedding | Pythia small | Pythia small | .006 |
| attention/主 MLP 输出 | Pythia wang .00320750 | Pythia wang .00320750 | .006 |
| BAM 写入 RMS epsilon（seed、attention、MLP） | 1e-5 | 1e-6 | 1e-6 |
| 向量 LayerNorm epsilon | 1e-5 | 1e-5 | 1e-5 |
| 读 key epsilon | 1e-4 | 1e-4 | 1e-4 |

两路均保留串行残差、GELU、LayerNorm、偏置结构、RoPE18、单门 M-shift .5 和读/attention 写共享 R256；MLP 写 down 独立。原本为零的门 kernel、门偏置开度、归一化有效 scale=1 已核对与原配方一致，不改变。这里 epsilon 是前向计算规则，明确作为 BAM 适配修复登记，不冒称纯初始化变化。全量原初始化对照不是恢复整个 Llama 架构，也不是把所有参数都随机化：语义常量、零门与偏置保持各自原规则。

配置类：`BamPythiaMediumPropT2048SerialMLPBamInitFix`、`BamPythiaMediumPropT2048SerialMLPOriginalInit`，同一实现工作树 `/data0/xd/pythia-mediumprop-t2048`，分支 `codex/pythia-mediumprop-t2048`。两路参数均432356690，MLP[5832,5643,5832]，新增0 W_Q，不涉及参数偿还。T2048 × global256，13500步原schedule，从头训练，checkpoint200/latest2、TB500、关闭健康指标。

原串行 BAM 为直接方向基线；串行 MHA、串行 Mudd 为共同参考，报告相对父 BAM 的收益倍数以及 BAM/Mudd 倍数完整序列。两路之间保留直接比较。收益倍数默认参考 MHA，不使用 Mudd 代替 MHA。

开跑前下注：以串行 MHA 为共同参考，相对父 BAM 终局收益倍数，BAM修复1.12（1.02–1.25）、全量原初始化1.20（1.05–1.40）；若父族终局BAM/Mudd约1.14，对应约1.28、1.37。初始M尺度的机制证据支持认真押注，但尚未证明终局loss因果。初始化本身不增加训练FLOPs，预期没有速度损失；新runtime还修复健康开关，实际速度改善应独立注明，不能算初始化收益。

基础设施审计以主分支 `c5e02ad8` 为准，比较 train/train_compile/max_utils/maxtext_utils/optimizers/checkpointing；发现家族工作树缺少 `record_training_health_metrics` 的计算门控（配置虽false，旧train仍计算norm），本次移植该主分支修复。其余相关运维路径一致，已有TB500旋转与 begin_step、最终checkpoint/保留策略。主分支新增M-shift统计导出不影响本实验（关闭sow）。保留模型谱系，不整体合并主干。

验证包括完整参数树/形状与父路相同、逐参数初始化角色及语义常量、实际全宽 seed M 与原尺度一致、两路有限前向和非零梯度/因果性、原Pythia/串行回归、完整BAM回归及summary writer。CPU/AOT通过前不启动训练。

暂停记录：SwiGLU MHA 4527、Mudd3603、BAM3174，均有提交checkpoint；MHA TPU310241已释放，Mudd310242和BAM310243保留热替换。暂停为资源转向，结论仍属阶段性，不能记为失败。TPU选UE5a，沿用已READY的v5p-16，无新区域申请。预留FLEX `llm-jax-v6e-1-0` / EW4a，实际queued resource为FLEX_START，两路编译通过worker锁串行执行。


## 启动证据（2026-10-15）

Runtime `eaa95f811e4ca774a680e4f3323fc29efe812343` 已推送并通过SEALED_CONFIG。新增初始化5项、Pythia/串行回归4项、完整BAM47项、summary writer5项均通过；日志在 `/data0/xd/bam_diagnostics/pythia-init/`。实际全宽bf16 seed M RMS=.018492952，和用原epsilon的同参数模块逐元素完全相等。

两路都在原UE5a READY v5p-16上热替换，从头step0训练，不加载SwiGLU旧checkpoint：
- `BamPythiaMediumPropT2048SerialMLPBamInitFix`：物理TPU `xd-v5p-16-310243-maxtext`，旧RUN BAM SwiGLU保留CP3174；FIRST_STEP=5，.518step/s。
- `BamPythiaMediumPropT2048SerialMLPOriginalInit`：物理TPU `xd-v5p-16-310242-maxtext`，旧RUN Mudd SwiGLU保留CP3603；FIRST_STEP=9，.518step/s。

FLEX上的两个AOT均完成，状态 `/home/lishengping/xd/projects/aot_runs/eaa95f8-36b3968a.json`（修复版）和 `eaa95f8-78a1a3a6.json`（全量原初始化）。两份manifest的commit/topology/13500schedule已在hot-switch dry-run核对。预留编译机保持不变。暂停三路TB均SYNC_OK。

全量原初始化worker实际参数树156叶、432356690参数；实际配置已核对T2048/batch32 per-device、串行、MLP宽度、两类epsilon、checkpoint200/latest2/TB500、health OFF，AOT Loaded。修复版和稳定速度继续核对。特别注意.006是原配方的普通随机投影规则；如static_v_key本来使用专用初始化，仍保留其原有规则，语义零门及bias等不改成随机值。


两路worker配置和实际参数树已全部核对，均156叶、432356690参数、AOT Loaded、health OFF。固定20–99步各80个速度样本：BAM修复.5175875step/s，全量原初始化.5166625step/s；相对串行父路.5103625为+1.42%/+1.23%。两路差异约.18%，未显示初始化本身的速度影响；共同小幅提速与健康计算门控修复混杂。运行登记中的基线和两组收益倍数（各自相对父BAM、相对串行Mudd，均用串行MHA参考）已添加。


### Core gain report2000

Report BAM/Mudd ratios FIRST, each with own-family MHA. Steps1000/1200/1400/1600/1800/2000: BAM fix2.505087/2.516143/2.302020/2.246572/1.996239/2.099398; all-original1.921299/1.965807/1.813596/1.789764/1.646009/1.739619. Unfixed serial ends this sequence1.624957; exact architecture-matched old recipe1.485257. At2000 second-order fix versus unfixed1.291971, versus old recipe1.413492; all-original1.070563/1.171258. Same-family new/parent ratio equals the ratio of BAM/Mudd ratios algebraically because MHA and Mudd cancel, but reporting only that number hides the cross-recipe question. User explicitly corrected the omitted primary metric; put full BAM/Mudd and second-order sequences before parent gap in future reports.

Initialization fix absolute parent gain shrinks-.074227@1000→-.053175@2000 while second-order stays~1.29. All-original crosses parent at1200 and reaches-.012851@2000, still.040324 behind fix but catching up. Native window data in /data0/xd/bam_diagnostics/pythia-init/core-gain-common2000.json. Both checkpoints2000 committed. Early multipliers exceed old recipe even WITHOUT repair, so whether initialization fixes late relative degradation remains unproven; focus matched4k–6k crossover region and terminal. Speeds.5175875/.5166625 vs parent.5103625 include actual health-off compute fix.


### Core gain report3000

Both run3000 checkpoints committed. BAM/Mudd ratios at2000/2200/2400/2600/2800/3000: repair2.099398/2.023374/2.014201/1.986014/1.915420/1.892021; all-original1.739619/1.683254/1.700656/1.705939/1.655198/1.664113. Unfixed serial1.624957/1.554058/1.532138/1.515202/1.449346/1.432027. Repair/unfixed1.291971/1.301994/1.314635/1.310725/1.321576/1.321219; all-original/unfixed1.070563/1.083135/1.109989/1.125882/1.142031/1.162068. Cross-recipe ratios continue dropping (3000repair1.268986,all-original1.116127), while same-family initialization effect grows. Parent gap repair-.046445/all-original-.023433 at3000. Original's gap to repair shrinks.084702@1000→.040324@2000→.023012@3000.

Use known serial terminal1.147469 and matched-old-recipe latest5 through10000=1.473443 to define recovery threshold new/parent1.284081. If current second-order persists, conditional terminalrepair1.516058/all-original1.333437; this is not fitted extrapolation nor observed terminal. Matched old run stopped10160, so1.473443 is a late reference, not a full13500 endpoint. Repair's terminal new/parent forecast raised1.12->1.30[1.20,1.40]; preserve original1.12[1.02,1.25] pre-run bet. Original pre-run1.20[1.05,1.40] retained. Continue both;4k–6k is decisive for maintaining the recovery threshold.

Reusable experiment-specific reporter `report_pythia_init_gains.py` joins registry gain ratios, does not recompute raw loss windows. Invocation takes report-all JSON, completed serial BAM loss-report, matched old-recipe loss-report, through-step and output. Artifact /data0/xd/bam_diagnostics/pythia-init/core-gain-common3000.json; actual registry triple ratios remain the shared authority.

## Common4k update

Actual steps4042/4102, CP4000 both. BAM-specific repair/all-original BAM-over-Mudd ratios1.801077/1.630746, each using own-familyMHA. Second-order versus unfixed serial1.329051/1.203361; versus matched old recipe at4k1.215081/1.100169. Parent-BAM gaps-.0427632/-.0264286, mutual gap.0163346 down from.0230114 at3k. Initializer repair contribution remains strong; cross-recipe relative edge continues shrinking. Conditional terminal if current second-order holds1.525045/1.380820; not fitted/observed terminal. Recovery threshold1.284081 versus matched old late10k gain1.473443; repair arm above, full-original below. Continueboth to5k, retain forecasts1.30[1.20,1.40] and1.20[1.05,1.40]. Full200-step primary series in core-gain-common4000.json; original report report-common4000.json. Report helper now selects exact init RUN names from shared multi-group report-all instead of misclassifying unrelated arms.
