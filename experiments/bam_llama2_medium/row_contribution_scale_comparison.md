# Medium / XL 行读贡献：相同64条Pile的配对比较

2026-09-14，全部完成。每个模型172场景×64序列=11008个loss（含重复对照），共22016个loss；最近追加V分段7场景及Q/K层类型8场景。低秩压缩、跨层共享、静态key和幅度校准按用户要求暂停，未运行。

## 核心结论

- **Medium的Q/K行读是低损伤精简候选；这个结论不能直接搬到XL。** Q/K同时全删：Medium **+0.002594 ± 0.000681**，XL **+0.017588 ± 0.002767**，XL约6.8倍；两者配对差 **+0.014994 ± 0.002475**。XL的64条序列全部变差。
- **O在两者中都是主要行读功能来源，但关键层发生迁移。** O整体删除均损失约1.83。Medium单层最强为 **F11、F5**；XL为 **F5、L1**，XL的F11效应很小。不能用同一个单层排名指导两种模型。
- **V在XL整体更重要，增强主要集中于L1。** V整体删除损失从0.032059增至0.075087；L1单独删除从0.019173增至0.053799。删除其余V、仅留L1，在两者中都损失约0.010–0.011。
- **O的弱层有累积作用。** 两个模型都只保留L1/F2/F5/F11时，仍损失约0.11。Q/K全删、V只留L1、O只留这四层的激进组合，在Medium/XL分别损失0.136437/0.160447，不能称为近乎无损。
- **XL的跨路径交互更强。** 全行读关闭损失2.935038，高于Medium的2.240091；全关闭减去四路独立关闭之和，XL为1.017529，Medium为0.370072。单层/单路knockout都不可简单求和当作总贡献。

这些是两个实际训练完成模型的冻结checkpoint因果必要性比较。XL同时改变了rank、共享basis、路由、宽度、训练时长和历史数值/WD谱系，**不是只改变规模的受控实验**；不能把差异全部解释为scaling，也不能推断重训能补回多少。

## 模型与共同口径

| 项目 | Medium | XL |
|---|---|---|
| 完整配置 | `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow` | `BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis` |
| Checkpoint | 13500 | 30091 |
| 训练源码 | `77401da6f83a5aa6ddd61994e028c3c694221518` | `c664e8211c299c9c780ce1667553cf995130c453` |
| 层数/结构 | 24层，8×LLF | 24层，8×LLF |
| D / heads × head_dim | 1024 / 16×64 | 2048 / 16×128 |
| M的K/V/C | 32/32/8 | 64/32/8 |
| Q/K | 独立rank1 | 共享rank4 basis、独立gate/mixing |
| Local V | rank4 | rank4 |
| 原生序列平均loss | 2.472505 | 2.225765 |

两者均取既有seed9876、T2048的128条Pile cohort的前64条，实际有效target总量130974。五个batch字段逐序列hash全部一致。所有表格使用64条序列的**配对Δloss均值 ± 1.96标准误**，单位nats/token，正值为干预后变差。每条序列先按有效target归一化，再在序列间等权平均；不把token视为独立样本。误差未做多重比较校正，细小层排名只作探索性观察。

## 四路总贡献

| 路径 | Medium：独立关闭 | XL：独立关闭 | XL−Medium：配对差 | Medium：Shapley | XL：Shapley |
|---|---:|---:|---:|---:|---:|
| Q | +0.001763 ± 0.000544 | +0.009587 ± 0.001915 | +0.007824 ± 0.001734 | -0.000055 ± 0.000805 | +0.024076 ± 0.002647 |
| K | +0.001745 ± 0.000475 | +0.008583 ± 0.001203 | +0.006838 ± 0.001185 | -0.000541 ± 0.000681 | +0.050970 ± 0.006300 |
| V | +0.032059 ± 0.003465 | +0.075087 ± 0.006636 | +0.043029 ± 0.004611 | +0.222376 ± 0.020642 | +0.534092 ± 0.031779 |
| O | +1.834453 ± 0.124822 | +1.824252 ± 0.115970 | -0.010201 ± 0.061629 | +2.018311 ± 0.126560 | +2.325900 ± 0.135173 |
| 所有行读关闭 | +2.240091 ± 0.132233 | +2.935038 ± 0.162944 | +0.694947 ± 0.081063 | — | — |

Shapley精确遍历16种路开关组合，对四路的边际关闭损失分配，总和严格等于全行读关闭损失。O/V/Q/K占总量：Medium约 **90.10% / 9.93% / −0.002% / −0.024%**；XL约 **79.25% / 18.20% / 0.82% / 1.74%**。Medium Q/K微负且误差跨零，是受损网络多上下文的平均分配，不能据此称原生Q/K有害。

O单路损失跨模型差为−0.010201±0.061629，未显示明确差异；XL的V/Q/K依赖更大。所有路径一起被削弱时出现更大损伤，与XL更大的全行读损失相符。

## Q/K同时删除：深度分组

层号从0起算，区间含端点；每组只关该层段的Q和K行读，V/O及列读照常。

| 关闭层段 | Medium | XL | XL−Medium：配对差 |
|---|---:|---:|---:|
| 全部0–23 | +0.002594 ± 0.000681 | +0.017588 ± 0.002767 | +0.014994 ± 0.002475 |
| 前1/2：0–11 | +0.001510 ± 0.000530 | +0.010459 ± 0.002567 | +0.008949 ± 0.002350 |
| 后1/2：12–23 | +0.000927 ± 0.000379 | +0.006529 ± 0.000774 | +0.005602 ± 0.000831 |
| 前1/3：0–7 | +0.000622 ± 0.000356 | +0.008277 ± 0.001676 | +0.007655 ± 0.001676 |
| 中1/3：8–15 | +0.001392 ± 0.000445 | +0.003249 ± 0.000940 | +0.001858 ± 0.000736 |
| 后1/3：16–23 | +0.000491 ± 0.000316 | +0.004576 ± 0.000631 | +0.004086 ± 0.000692 |

Medium后三分之一的删除损失均值最小，中三分之一最大；XL前三分之一最大，中三分之一最小。XL后半仍有0.006529损失，并非整体无用。这个分组把共享Q/K一并关闭，适合判断XL共同row basis的删减价值；不能靠“某路单层很弱”直接删除共用投影。

![Q/K深度分组比较](/data0/xd/bam_diagnostics/row-contribution-scale-comparison-0914/qk_depth_comparison.png)

## V行读分段删除（追加验证）

与Q/K使用相同完整层段、同64条Pile；仅删除区间内Local层的V行读，Q/K/O和全部列读保留。native以及全层V关闭与原始实验逐样本完全一致。

| 关闭层段（0起算） | Medium Δloss | XL Δloss | XL−Medium配对差 |
|---|---:|---:|---:|
| 全部0–23 | +0.032059 ± 0.003465 | +0.075087 ± 0.006636 | +0.043029 ± 0.004611 |
| 前1/2：0–11 | +0.026665 ± 0.003228 | +0.071104 ± 0.006335 | +0.044439 ± 0.004713 |
| 后1/2：12–23 | +0.004491 ± 0.000665 | +0.002943 ± 0.000666 | -0.001548 ± 0.000884 |
| 前1/3：0–7 | +0.024818 ± 0.003067 | +0.068168 ± 0.005956 | +0.043349 ± 0.004477 |
| 中1/3：8–15 | +0.004405 ± 0.000893 | +0.003196 ± 0.001223 | -0.001209 ± 0.001079 |
| 后1/3：16–23 | +0.001605 ± 0.000356 | +0.001606 ± 0.000429 | +0.000001 ± 0.000518 |

**两尺度的V作用都偏前，XL相对Medium增加的V依赖主要在前段。** XL前1/3删除损失0.068168，Medium为0.024818；后1/3两者均约0.001605。XL后半删除损失0.002943，比Medium的0.004491更小，不能从XL整路V更重要推断其每个深度段都更重要。

后1/3在两个模型上都是本次分组中损伤最小的V候选，约+0.0016而非完全无损；两模型都能理论删掉5个Local层的行key kernel，即0.625 W_Q（各自V行key kernel的31.25%）。后半对应8个Local层、1 W_Q。未做实际结构裁剪或重训，分段效应不可相加当作唯一贡献份额。

三个三分段实际包含6/5/5个Local层；第0层M为零，是无操作，故每段均含5个非第0层Local读。半段各8个Local层。Fetch层无Local V。

复现：Medium runtime `b5f27a78`、XL runtime `901bb070`；各自worktree的`row_contribution.py`，`ROW_STAGE=vdepth ROW_START=0 ROW_STOP=64 ROW_VARIANT_BATCH=1`。分别使用保留的`xd-v6e-rowko-0-ewa4a-0914`、`xd-v6e-rowko-2-ewa4a-0914`，输出`/tmp/medium-row-vdepth`、`/tmp/xl-row-vdepth`；没有新增或删除TPU。每套新增7×64=448个loss，原始checkpoint/生产源码不变。产物在各模型artifact根目录的`vdepth/`，包括metadata、逐序列loss、summary、verification及resource_manifest。GCS对应同模型诊断前缀下`vdepth/`，64个文件数量/大小已核对。聚合命令`summarize_row_targeted.py ROOT --stage vdepth`，比较脚本同时输出vdepth逐序列配对差。两台运行进程已退出，继续保留机器。

## Q/K在Local与Fetch层的作用（追加验证）

同64条，Q、K各自和同时分别关闭全部Local或全部Fetch层；V/O和全部列读保留。两模型native与全层Q/K同删对照均与原始结果逐样本完全一致。

| 关闭行读 | Medium Δloss | XL Δloss | XL−Medium配对差 |
|---|---:|---:|---:|
| 仅Q / 全部L层 | +0.000624 ± 0.000284 | +0.005066 ± 0.001109 | +0.004442 ± 0.001115 |
| 仅Q / 全部F层 | +0.001419 ± 0.000442 | +0.002836 ± 0.000618 | +0.001417 ± 0.000643 |
| 仅K / 全部L层 | +0.000443 ± 0.000279 | +0.005817 ± 0.000941 | +0.005373 ± 0.000947 |
| 仅K / 全部F层 | +0.001227 ± 0.000377 | +0.001878 ± 0.000592 | +0.000652 ± 0.000640 |
| Q/K同时 / 全部L层 | +0.000992 ± 0.000360 | +0.010379 ± 0.001684 | +0.009387 ± 0.001669 |
| Q/K同时 / 全部F层 | +0.001805 ± 0.000508 | +0.004577 ± 0.001174 | +0.002772 ± 0.001076 |
| Q/K同时 / 全部层 | +0.002594 ± 0.000681 | +0.017588 ± 0.002767 | +0.014994 ± 0.002475 |

**看整组删除损伤，Medium是F更重要，XL是L更重要。** Q/K同删的L−F配对差：Medium **-0.000813 ± 0.000570**；XL **+0.005802 ± 0.001284**。Q和K分别删除也支持相同方向，合并Q/K没有掩盖相反的单路趋势。

但L有16层，F只有8层。下面对两个组做精确两玩家Shapley分配：`φL=(ΔL+Δ全部−ΔF)/2`，`φF=(ΔF+Δ全部−ΔL)/2`，每个样本的两项严格加和等于全部Q/K关闭损失。这里V/O固定为原生，玩家为“L层Q/K”和“F层Q/K”，与前面的四路径Shapley上下文不同，不混用。

| 量 | Medium | XL |
|---|---:|---:|
| 全部−L单关−F单关 | -0.000202 ± 0.000314 | +0.002633 ± 0.000526 |
| L组Shapley | +0.000890 ± 0.000346 | +0.011695 ± 0.001742 |
| F组Shapley | +0.001704 ± 0.000523 | +0.005893 ± 0.001271 |
| L组Shapley / 行key kernel W_Q | +0.000890 ± 0.000346 | +0.005847 ± 0.000871 |
| F组Shapley / 行key kernel W_Q | +0.003408 ± 0.001047 | +0.005893 ± 0.001271 |
| 每W_Q归一化后的L−F配对差 | -0.002517 ± 0.001038 | -0.000046 ± 0.000935 |

Medium L/F的QK行key kernel成本为1/0.5 W_Q；XL共享QK对应2/1 W_Q。**按这一成本归一化，Medium F组的Shapley均值约为L组3.8倍，XL两类几乎持平，差值误差范围跨零。** 因此Medium若精简Q/K，L组更值得优先考虑；XL不能仅凭F组总损伤更小，就断言F组单位参数更不重要。这里是冻结网络功能分配除以理论kernel预算，不是测得的速度收益，也不证明逐层/逐参数贡献均匀。第0层虽然无操作，仍按实际存储kernel计入成本。

复现：Medium runtime `6be80e7f`、XL runtime `b0905334`；各自worktree的`row_contribution.py`，`ROW_STAGE=qktype ROW_START=0 ROW_STOP=64 ROW_VARIANT_BATCH=1`。继续使用保留的node0/2（完整名`xd-v6e-rowko-0-ewa4a-0914`、`xd-v6e-rowko-2-ewa4a-0914`，EW4a），输出`/tmp/medium-row-qktype`、`/tmp/xl-row-qktype`，每模型新增8×64=512个loss。产物在各模型本地/GCS诊断根目录的`qktype/`；原始checkpoint及模型源码不变。64个文件GCS数量/大小、完整cohort字段hash、scalar模式及配对对照检查通过。分组mask检查确认Q/K各自范围、L/F互补与V/O未修改。两台进程已退出，机器继续保留。

聚合：`summarize_row_targeted.py ROOT --stage qktype`；两组交互/分配：XL诊断worktree的`summarize_qk_layer_types.py ROOT`，保存`layer_type_allocation.json/npz`；比较脚本增加qktype表及逐序列配对差。

## 按Local/Fetch及关键层联合删除

以下组合在Medium初次筛查后确定，XL使用**完全相同的层集合**；它们不是在XL重新挑选的最优稀疏方案。全部64条用于探索性比较，没有额外留出确认。

| 干预（仅行读） | Medium | XL | XL−Medium：配对差 |
|---|---:|---:|---:|
| 关闭全部Local O | +0.118504 ± 0.011030 | +0.266719 ± 0.023290 | +0.148215 ± 0.016982 |
| 关闭全部Fetch O | +0.798373 ± 0.074481 | +0.295879 ± 0.038782 | -0.502494 ± 0.045108 |
| O只留F5/F11 | +0.209151 ± 0.018238 | +0.741197 ± 0.053994 | +0.532046 ± 0.042741 |
| O只留L1/F2/F5/F11 | +0.113688 ± 0.009591 | +0.105512 ± 0.008403 | -0.008176 ± 0.005608 |
| 只关O的F5/F11 | +0.633634 ± 0.070279 | +0.189102 ± 0.030889 | -0.444532 ± 0.044451 |
| V只留L1 | +0.010307 ± 0.001334 | +0.010857 ± 0.002178 | +0.000550 ± 0.001760 |
| Q/K全关；V留L1；O留L1/F2/F5/F11 | +0.136437 ± 0.010856 | +0.160447 ± 0.011825 | +0.024010 ± 0.006260 |

Medium关闭Fetch O比关闭Local O损伤大得多（0.798 vs 0.119）；XL两组损失接近（0.296 vs 0.267）。这与XL早期Local层L1作用增强一致。只留F5/F11在XL损失0.741，加回L1/F2后降至0.106；这项联合对比没有进一步分离L1和F2各自的恢复份额，但单层结果显示L1效应远强于F2。

## 层分布和非加性

![四路逐层比较](/data0/xd/bam_diagnostics/row-contribution-scale-comparison-0914/layer_comparison.png)

| 单层行读关闭 | Medium | XL |
|---|---:|---:|
| O / L1 | +0.020602 ± 0.002343 | +0.120853 ± 0.011078 |
| O / F2 | +0.023229 ± 0.002587 | +0.014230 ± 0.002534 |
| O / F5 | +0.114982 ± 0.012251 | +0.186768 ± 0.031426 |
| O / F8 | +0.007336 ± 0.000970 | +0.020243 ± 0.001914 |
| O / F11 | +0.204305 ± 0.028066 | +0.004585 ± 0.000774 |
| V / L1 | +0.019173 ± 0.002465 | +0.053799 ± 0.005086 |

O逐层独立关闭的总和仅Medium 0.427738、XL 0.385240，均远小于整路关闭约1.83。LLF单元联合删除也有明显交互：Medium的O第1/3单元（层3–5/9–11）分别0.207863/0.256132；XL第0/1单元（层0–2/3–5）分别0.229669/0.252568，而第3单元仅0.008461。单元分布同样显示O核心更偏前层，但不能把这些数相加当作分配后的贡献。

完整24层×4路与8个LLF单元表：
- [Medium完整报告](/home/xd/projects/maxtext/experiments/bam_llama2_medium/row_contribution.md)。
- [XL完整结果与复现](/home/xd/projects/maxtext/experiments/bam_llama2_medium/xl_row_contribution.md)。

## 参数/计算上的含义

`W_Q=D²`；仅计行key投影kernel，不含gate、mixing、norm、矩阵收缩及其他BAM参数。

| 行key kernel总量 | Medium（D=1024） | XL（D=2048） |
|---|---:|---:|
| Q/K | Q 0.75 + K 0.75 = **1.5 W_Q** | 共享Q/K **3 W_Q** |
| V | 2 W_Q | 2 W_Q |
| O | 12 W_Q | 12 W_Q |
| 合计 | **15.5 W_Q = 16,252,928权重** | **17 W_Q = 71,303,168权重** |

Medium可独立去掉Q或K的row key kernel；XL只去掉一路输出时，共享basis仍被另一路使用，不能把3 W_Q简单平分为两个可独立删除的预算。XL关前/后半QK对应理论1.5 W_Q，每个三分之一对应1 W_Q；Medium对应0.75及0.5 W_Q。

结合损失：Medium优先Q/K；XL若接受小幅冻结模型损失，中三分之一Q/K是本次分组里损伤最小的候选。O虽然占投影大头，直接按少数尖峰保留仍损失明显；本次暂停的低秩/共享方法没有任何实验结论。V只留L1能删掉15/16层V行投影，但两尺度仍约+0.01，需按可接受损伤衡量。

本次干预在已算出的行输出上乘0或0.5，**没有实现裁剪kernel、测得训练/推理加速，或改变M-cache**。理论kernel乘加减少只在以后真实结构改造时成立，不能用当前诊断耗时估算训练加速。上述候选均未重训。

## 方法与复现

- 使用最新主工作树流程skill，分别从两模型实际训练commit建立诊断worktree；生产模型源码和checkpoint参数不变。Medium分支/worktree：`codex/llf-row-contribution` / `/data0/xd/llf-row-contribution`；XL：`codex/xl-row-contribution` / `/data0/xd/xl-row-contribution`。
- Medium原始141场景runtime `f619f39135e5eba3a7852d0d1ea6f0d338f2e625`，联合9场景 `4c587ab087ddf931fc970ff2e5b5447a346a58ba`，深度7场景 `715854d5`。XL全部157场景runtime **`46eb17081c57fd188d07f4e16be47ded2a6c7364`**。
- 在Linen `_read_local`返回值上分别缩放Q/K/V行读，在`_read_fetched_m`返回值上缩放Local/Fetch O行读。Medium的行输出区间从32开始，XL从64开始；前面的列读、gate、mixing均保留。XL共享basis内部计算照常，独立Q/K输出可分别开关。所有位置、下游attention和M写入照常重算。这是全网络因果必要性，不是固定下游激活的直接归因。
- 每模型16全路组合+5半幅控制+88单层关闭+32单元关闭+9联合删除+7深度组。V只存在16个Local层；第0层M为零，四路关闭为无操作。
- 只读restore、`only_eval=True`，所有通用/BAM健康采集关闭；不保存原始激活。模型形状、完整resolved overrides、五字段序列hash和checkpoint URI保存于worker metadata。
- 共同cohort文件SHA256：`68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`。source是seed9876的128条文件，只执行索引0–63，未扩大样本数。
- 全开干预与普通forward、层0/F-V无操作检查通过；每套模型重复native与全层QK关闭均逐值完全一致。两模型全部五字段hash一致，raw loss全有限，64条无重复。原始worker文件已核对GCS数量/大小，本地保留SHA256清单。
- Medium47项、XL53项历史BAM单测通过；Tiny Linen拦截、列读保留、gate/层/路径选择测试通过；QK分组覆盖和互补性检查通过；Shapley精确求和及配对验证通过。XL模型源码相对`c664e82`未改。
- Medium起初尝试variant batch8，因native数值不等价触发回退，全部最终结果为scalar；默认已修为1。XL从头使用scalar。XL在每节点一次恢复后每条157场景约4.22秒，host RSS约12GB；三个节点并行，非批处理加速。首次同步快照有1个对象未完成，最终完整同步成功且64文件验证通过，诊断结果无缺片。

### 路径与命令

XL checkpoint：`gs://newproject-1-llm_projects_europe-west4/log/BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis/checkpoints/30091/items`，存在`commit_success.txt`；Medium checkpoint在`gs://newproject-1-llm_projects_us-east5/log/BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow/checkpoints/13500/items`。

XL运行`experiments/bam_llama2_medium/run_row_contribution.sh`，设置`ROW_STAGE=suite ROW_VARIANT_BATCH=1`，三分片`ROW_START/STOP=0/22,22/43,43/64`，独立`ROW_OUTPUT`及`ROW_GCS`。该launcher调用`row_contribution.py`、`RowContributionProbe`。XL原始`worker*/suite_NNN.npz`用`split_row_suite.py ROOT`拆成all/targeted/depth，属于同一forward套件的分组，不是额外实验。随后执行`summarize_row_contribution.py ROOT`、`summarize_row_targeted.py ROOT [--stage depth]`。

比较命令：`compare_row_contribution.py MEDIUM_ROOT XL_ROOT OUTPUT_ROOT`。脚本在XL worktree的`experiments/bam_llama2_medium/`。导出逐序列配对差、`comparison.json`、`comparison_tables.md`、PNG/PDF。图使用两模型同一子图纵轴，四路子图之间尺度不同。

- Medium产物：`/data0/xd/bam_diagnostics/llf-row-contribution-13500-0914`；GCS同名目录位于`gs://newproject-1-llm_projects_europe-west4/log/diagnostics/`。
- XL产物：`/data0/xd/bam_diagnostics/xl-row-contribution-30091-0914`；GCS同名目录位于同一诊断根目录。
- 比较产物：`/data0/xd/bam_diagnostics/row-contribution-scale-comparison-0914`。
- 大产物均worker→GCS→本机；tpu-ag只编排，无模型产物中转。

### 保留资源

按用户要求保留三台`v6e-1`，均在`europe-west4-a`：`xd-v6e-rowko-0-ewa4a-0914`、`xd-v6e-rowko-2-ewa4a-0914`、`xd-v6e-rowko-3-ewa4a-0914`。Medium完成后转用于XL，现全部诊断Python进程已退出；未新开训练、未删除这三台、未借用其他任务资源。资源及分片归属在两个产物根目录的`resource_manifest.json`。
