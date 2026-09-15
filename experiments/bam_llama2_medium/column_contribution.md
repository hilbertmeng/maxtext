# Medium / XL 列读贡献及行列联合干预

2026-09-15 UTC 完成。与此前行读分析使用同两个checkpoint、同64条Pile，分析到路径、层、LLF单元、深度及L/F组，不做头粒度。新增每模型376个命名场景，共48128个序列loss；去重后43776个干预forward，另有一致性检查。

## 目的与结论

目的仍是减少BAM读参数和计算：先确认低损伤行读删除是否依赖保留列读，再辨别两方向哪些功能重叠、哪些组合不能同时删，并以投影成本衡量取舍。这比单独再画列读层排名多回答了“另一方向是否能兜底”和“此前行读稀疏模式是否可迁移”两个问题。

- **Q/K和V的列读在两个模型中都比行读更必要。** Q/K列全关的损伤远大于行全关，V也一致；行读低损伤不是整个BAM读取都不重要。
- **O是行、列都不能轻视的路径，联合损伤有强正交互。** 一方向的原生删除损伤会低估两方向共同删除的损伤；冻结模型中存在功能补偿的证据，但不能仅凭交互确定内部算法。
- **列读的层分布不能沿用行读结论。** 行V只留L1时损伤约0.01；列V只留L1时仍接近列V全删的损伤。行O的重要层组合也保不住列O的大部分功能。
- **优先保留列读、针对行读压缩的方向得到支持。** 列key投影总预算Medium/XL仅6.5/4 W_Q，行key为15.5/17 W_Q；O行key均12 W_Q，仍是主要参数靶点。该判断是候选排序，尚无结构裁剪、重训恢复或实际速度收益证明。

## 整路行、列、联合关闭

单位nats/token，mean ±1.96 paired SE；其余路径保持原生。

| Model | Path | Row | Column | Both | Interaction: both−row−column |
|---|---|---:|---:|---:|---:|
| Medium | Q | +0.001763 ± 0.000544 | +0.566636 ± 0.097229 | +0.572173 ± 0.093861 | +0.003774 ± 0.008077 |
| Medium | K | +0.001745 ± 0.000475 | +0.579851 ± 0.101778 | +0.580010 ± 0.095469 | -0.001585 ± 0.012111 |
| Medium | V | +0.032059 ± 0.003465 | +0.718291 ± 0.075069 | +0.841235 ± 0.079182 | +0.090885 ± 0.013021 |
| Medium | O | +1.834453 ± 0.124822 | +1.633999 ± 0.169881 | +5.474973 ± 0.160312 | +2.006521 ± 0.143483 |
| XL | Q | +0.009587 ± 0.001915 | +0.638879 ± 0.077843 | +0.682377 ± 0.080700 | +0.033911 ± 0.008240 |
| XL | K | +0.008583 ± 0.001203 | +0.596312 ± 0.077368 | +0.640286 ± 0.080412 | +0.035391 ± 0.011390 |
| XL | V | +0.075087 ± 0.006636 | +1.123724 ± 0.161487 | +1.583307 ± 0.176777 | +0.384495 ± 0.031124 |
| XL | O | +1.824252 ± 0.115970 | +3.005766 ± 0.109965 | +6.216031 ± 0.166770 | +1.386012 ± 0.214555 |

| 模型 | QK：行 | QK：列 | QK：联合 | 全四路：行 | 全四路：列 | 全四路：联合 |
|---|---:|---:|---:|---:|---:|---:|
| Medium | +0.002594 | +0.600387 | +0.604006 | +2.240091 | +2.697250 | +5.892181 |
| XL | +0.017588 | +0.781163 | +0.928745 | +2.935038 | +4.100448 | +6.712411 |

![整路行列联合删除](/data0/xd/bam_diagnostics/column-row-contribution-0914/row_column_paths.png)

Medium O列−行的配对差误差跨零，因此不能断言Medium的O行比列更重要。XL QK行读在列读原生时删除仅损失+0.017588，但列读已关闭后再删行读增量为+0.147582；Medium对应+0.002594与+0.003619。XL的行读低损伤更依赖列读保持完整。全四路两方向交互Medium为+0.954841，XL却为−0.323075 ±0.262073；不把O的正交互推广到任意大组合。

## L/F、深度及行稀疏模式迁移

L/F均指0-based层号，F=2,5,8,…,23；F没有Local V。表中均为列读干预，行读保持原生。

| 列读干预 | Medium | XL |
|---|---:|---:|
| QK：L全关 | +0.183987 ± 0.023643 | +0.507560 ± 0.080260 |
| QK：F全关 | +0.225297 ± 0.037795 | +0.175790 ± 0.021721 |
| O：L全关 | +0.724446 ± 0.083852 | +0.603210 ± 0.041276 |
| O：F全关 | +0.428046 ± 0.052627 | +0.867023 ± 0.046857 |
| V：只留L1 | +0.713003 ± 0.073333 | +1.124986 ± 0.161315 |
| V：全关 | +0.718291 ± 0.075069 | +1.123724 ± 0.161487 |
| O：只留L1/F2/F5/F11 | +1.471648 ± 0.174138 | +1.517045 ± 0.132060 |
| O：只关F5/F11 | +0.018044 ± 0.002056 | +0.073514 ± 0.005039 |
| QK全关+V只留L1+O只留上述4层 | +2.421131 ± 0.178724 | +2.793037 ± 0.145607 |

| 列读深度组关闭 | Medium | XL |
|---|---:|---:|
| QK 前1/3 | +0.030311 ± 0.002505 | +0.299551 ± 0.038987 |
| QK 中1/3 | +0.389305 ± 0.070764 | +0.128544 ± 0.009041 |
| QK 后1/3 | +0.044489 ± 0.004650 | +0.068056 ± 0.007460 |
| V 前1/3 | +0.038507 ± 0.003740 | +0.206960 ± 0.030128 |
| V 中1/3 | +0.212204 ± 0.019773 | +0.172924 ± 0.022201 |
| V 后1/3 | +0.112922 ± 0.008443 | +0.084033 ± 0.006025 |
| O 前1/3 | +0.142509 ± 0.010742 | +0.974335 ± 0.088626 |
| O 中1/3 | +0.410205 ± 0.046021 | +0.413908 ± 0.026638 |
| O 后1/3 | +0.671120 ± 0.122415 | +0.533903 ± 0.096026 |

![逐层行列联合删除](/data0/xd/bam_diagnostics/column-row-contribution-0914/row_column_layers.png)

同路径的Medium/XL图共享纵轴；误差条未校正多重比较。完整24层、8个LLF单元、半幅及组合表见[完整列读表](/data0/xd/bam_diagnostics/column-row-contribution-0914/column_tables.md)，两方向Shapley及交互见[交互表](/data0/xd/bam_diagnostics/column-row-contribution-0914/row_column_interactions.md)。

## 跑前下注检验

跑前预测QK列全关Medium +0.01–0.10、XL +0.03–0.20；实际为+0.600387/+0.781163，两个区间都明显低估。‘V列比行重要’和‘O行列有明显非加性交互’得到支持。没有实测裁剪后的速度，本轮不作速度下注兑现声明。原预测保存在比较目录的`plan.json`。

## 方法、预算与复现

两个实际训练checkpoint分别为`BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow` step13500、`BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis` step30091。训练源码分别`77401da6f83a5aa6ddd61994e028c3c694221518`、`c664e8211c299c9c780ce1667553cf995130c453`，诊断worktree为`/data0/xd/llf-row-contribution`、`/data0/xd/xl-row-contribution`，分支同名加`codex/`前缀。生产MaxText目录相对指定训练源码无修改。模型规模、rank、路由、共享和训练时长均不同，跨模型差异不能仅解释为scaling。

数据沿用行读实验相同64条Pile，取seed9876、T2048、128条cohort文件的前64条。共同文件SHA256 `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`；五个输入/target字段逐样本hash一致。每条序列先按有效target归一化，然后序列等权平均，单位nats/token，正Δloss表示删除损伤。误差为±1.96配对SE，未做多重比较校正。

在BAM读取最终输出上干预，packed prefix `[0:bam_k]`为列读，tail为行读。列读对应`M @ r_v → k/data`，行读对应`Mᵀ @ r_k → v/address`。在Local Q/K/V读取及Local/Fetch O读取出口按层、路径、方向乘0或0.5，其余输出与gate保留，完整重算下游attention和M写入；不是固定下游激活的直接归因。第0层M为零、Fetch没有Local V，这些无操作有独立普通forward对照检查。

共376个命名场景，合并为342种唯一mask（每模型21888个干预forward，另有热身和重复一致性检查）：16个历史行读整路组合对照；179个列读场景（16路径组合、5半幅、88单层、32 LLF单元、9重点联合、7 QK深度、7 V深度、8 Q/K-L/F、7 O深度）；174个对应行列联合场景（不含5半幅）；另7个行读O深度以补齐此前未测的对照。先规范化已验证无操作，再按完全相同的mask去重，复用同一次forward；输出仍保留每个命名场景的对应loss。

每条序列均重跑并逐值核对原始16种行读路径组合及native。采用batch1固定编译形状，测量并验证scalar/async结果一致后选择较快者，不尝试曾导致bf16 lowering漂移的vmap或设备端loop。Tiny CPU测试覆盖两种head宽度、动态层号、行列切片、路径隔离和gate不变。checkpoint只读、`only_eval=True`，关闭健康指标/激活采集。

对于同一范围，`ΔR`、`ΔC`、`ΔRC`分别是只关闭行、只关闭列、行列都关闭的loss变化。交互为`I=ΔRC−ΔR−ΔC`：正值表示共同删除损伤超过两次单删之和，不能仅凭它命名具体机制。两玩家Shapley为`φR=(ΔR+ΔRC−ΔC)/2`、`φC=(ΔC+ΔRC−ΔR)/2`，逐序列和为`ΔRC`。Q/K/V/O四路径Shapley另在列读16组合上计算，行读保持原生，不能和这两方向分配混用。

### 参数口径

`W_Q=D²`，只计算动态read-key投影kernel，不包含gate、mixing、bias、norm、M写入/缓存投影等全部BAM参数。Local列key输出宽度`rank*bam_v`，行key为`rank*bam_k`；XL共享Q/Kbasis只计一次。O列key每层为`D*16*C`，行key为`D*16*K`，两个模型均C=8。

| 模型/方向 | Q/K | V | O | 总量 |
|---|---:|---:|---:|---:|
| Medium列 | 1.5 | 2 | 3 | 6.5 W_Q |
| Medium行 | 1.5 | 2 | 12 | 15.5 W_Q |
| XL列 | 1.5 | 1 | 1.5 | 4 W_Q |
| XL行 | 3 | 2 | 12 | 17 W_Q |

对应L/F组O列key成本Medium为2/1 W_Q，XL为1/0.5 W_Q；O行key均8/4 W_Q。Q/K列L/F两模型均1/0.5 W_Q，行Medium1/0.5、XL2/1 W_Q。第0层按实际存储kernel计预算，尽管其读结果为零。共享消费者尚在时不能把投影预算重复算为可删收益。

相同M尺寸和同样读key数量R时，行列两个方向的矩阵收缩各需与`K*V*R`成比例的乘加量；fetched压缩M对应`K*C*H`。投影预算不对称不代表收缩成本同比例不对称，实际混合、归一化、padding和算子融合也需单独衡量。

本实验是输出干预，未实现结构裁剪，未测训练/推理速度收益或M-cache减小。投影参数少不必然同比例降低端到端时间，后续真实裁剪需让投影和收缩尺寸实际缩小。低秩、跨层共享、静态key及重训仍未在本轮执行。

### 执行入口

runner为两worktree的`experiments/bam_llama2_medium/column_contribution.py`，launcher为`run_column_contribution.sh`。Medium runtime `c174418d4926d4382859f7c41d1f2fc6127078b9`，XL `51a55643e50ad4412a7236cebf3c85741d4169fa`。设置`COLUMN_OUTPUT`、`COLUMN_GCS`、`COLUMN_REFERENCE_GCS`、`COLUMN_START=0 COLUMN_STOP=64`。`reference.npz`为原行读16种组合的64×16 loss和输入hash，运行时每条都检查。

分析：`summarize_column_contribution.py NEW_ROOT LEGACY_ROW_ROOT`；作图：`plot_column_contribution.py MEDIUM_ROOT XL_ROOT OUTPUT_ROOT`；完整表格：`column_result_tables.py MEDIUM_ROOT XL_ROOT OUTPUT_ROOT`。原始输出`worker0/column_NNN.npz`，每模型64文件；summary中保留376新场景及历史行读补充，配对数组在`paired_results.npz`，校验记录在`verification.json`。数据只经worker→GCS→本机，tpu-ag仅编排。

本地新数据根：`/data0/xd/bam_diagnostics/medium-column-contribution-13500-0914`、`/data0/xd/bam_diagnostics/xl-column-contribution-30091-0914`；GCS在`gs://newproject-1-llm_projects_europe-west4/log/diagnostics/`下同名目录。比较产物、跑前下注、资源清单在`/data0/xd/bam_diagnostics/column-row-contribution-0914/`。

### 校验与资源收尾

两模型各64份原始NPZ，五字段输入hash与历史行读完全一致；逐样本16组历史行读对照最大误差均为0，普通native与无操作对照通过，scalar/async loss完全一致。各模型GCS文件数量与本地大小相符，`verification.json`保留逐文件SHA256。

实际执行为2台europe-west4-a v6e-1（每模型1台）；另2个us-central1-a候选未执行诊断，在主机首步验证后撤销。4个候选均已通过权威删除脚本停止创建器并验证node及queued-resource不存在，无保留机器。准确资源名、worktree、runtime及释放凭据见比较目录`resources.json`、`released_*.json`和各模型`release.log`。这4个候选与上一轮已释放的10台资源分开记账。

相关：[行读粗粒度比较](/home/xd/projects/maxtext/experiments/bam_llama2_medium/row_contribution_scale_comparison.md)、[行读头粒度](/home/xd/projects/maxtext/experiments/bam_llama2_medium/row_head_contribution.md)。低秩、共享、静态key、重训仍暂停，本轮没有启动这些实验。
