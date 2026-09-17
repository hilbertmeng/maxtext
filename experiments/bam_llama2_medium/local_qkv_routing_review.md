# Local Q/K/V routing、rank 与行读对齐：实验总账

更新：2026-09-12。范围为 Medium/XL 独立 LocalV LLF 主线，及用于判断可迁移性的全-F、Paired40 对照。本文不把相关性诊断当作训练因果证据。

## 先读结论

1. **C有明确但有适用边界的正面证据：Medium LocalV rank4上C≈B、优于legacy；XL LLF LocalQK rank2上C优于legacy。** 后者是只改Q/K routing的直接配对，20.5k–29.5k约−.0013，置信度较高。相反，Medium Paired40全F的LocalQK rank2改C在观察期较差；首批Medium Q/K rank1、V rank2同时改A/B/C也未胜过legacy。关键未解项是这些具体方向差异的原因，而不是用“上下文依赖”代替解释。
2. **Medium 最强完整训练证据是 V rank4+B+AlignedRow，Q/K 仍为 rank1 legacy。** 末期较历史独立LLF约 −.00479、较 Clean约 −.01280；不是“给原本有益的 rank4 再加两个改进”：rank4 legacy 本身先造成了损害。
3. **B 与 C-fp32 在 V rank4 上效果接近，C 有微小晚期均值优势及约 +.47% 吞吐。** 不足以声称 C 在所有通路、rank 或规模上更好。Medium 没有完整训练的“V rank4 C+AlignedRow”直接对 B+AlignedRow 配对；从 C 与 B 接近推断能组合，仍是推断。
4. **XL 的组合迁移有小正收益，且 Q/K 也统一为 C 后进一步改善。** 但 XL 没拆开 V rank4、routing C、AlignedRow 三个因素；不能据此确认每项单独可扩展。Q/K C 下 rank2→4 正在训练，22k前已出现持续小收益。
5. **pre-RMS bias 并非 Gram 算法成立的障碍。** C 的 Gram 和读取都使用同一个 `A=Wx+b` 即可。NoBias 到4k大体打平，尚无最终结论；不能从 bias 几何能量小推断其训练作用为零。

## 数据入口与可信度

- [完整配置类清单、继承与台账结果](local_qkv_experiment_inventory.md)：包含训练、复现、速度控制；明确区分历史注释和实时状态。
- [同窗口配对原始 gap 数据](local_qkv_pairwise_evidence.json)：每个点记录 step/gap/共同样本数，不拼接不同基线。
- 权威配置及 runtime hash：`MaxText/exp.py`。历史实现主线为 `codex/local-read-gram`、`/data0/xd/local-read-gram`；早期 LF/LLF 在 `codex/bam-alternating-local-fetch`，XL 独立LLF在 `codex/xl-lllf-profile`。
- 本次将 local-read-gram 的 attention 实现及配套测试、配置默认值集成到主工作树 `refactor-bam`。历史 RUN 仍使用其已登记 hash；**合入不意味着当前代码必然精确重现历史轨迹**，也没有切换在训 RUN。
- gap=实验−直接基线，负值有利。每窗口取共同 `step%10==0`、中心±25步数据；Medium 点距200，XL点距500。表中的区间均值与单点分开标。
- **高**：同谱系直接配对、长窗口/接近训满支持（只对该上下文）；**中**：短训、组合因素或时段有限；**低/未定**：外推、未隔离或证据缺项。均非多seed统计显著性结论；多数只有单seed。
- Medium LLF 为修复 AOT WD 的 Clean 谱系（bias/scale/read-gate-b0/gw_b0 skip WD）；XL 为匹配历史 Rank2 显式 all-decay。两者还不同于历史 Medium Paired40 的 AOT all-decay。跨尺度不能把这些差别消去。
- 历史吞吐涉及不同 runtime、健康统计和部分区域；只有条件匹配时才作严格比较。损失报告中的控制不是自动等于测速控制。

## 运算语义：先避免把不同实验混为一谈

LLF 中 L=LocalO+LocalV，F=fetched read；各层仍有 LocalQK 和写 M。独立 LocalV 默认读**完整 M**，LocalO 读 V压缩后的 M。Medium M=32×32/C8，XL M=64×32/C8；不是 XL 32×32。row-key 在 k/data轴，读出 v/address；col-key 在 v/address轴，读出 k/data。

对单侧，`A:[b,t,R,d]` 为动态基底，`H:[b,t,N,R]` 为动态 head mix；先用 R 个键读 M，再展开到 N 头。Q/K/V 各自有投影、偏置与门，不意味着互相共享键。packed 只是合并投影计算。

| routing | 基底变换 | H归一化 | 门的粒度 | 关键区别 |
|---|---|---|---|---|
| legacy | 每个基底 RMS | (N,R)联合，另除√R | 每条Q/K/V每侧一个共享门 | 有符号 mix，门不区分basis或head |
| shared_rank_gate | 每个基底 RMS | 仅N，无√R除数 | 每侧每basis一个门 | 不是 B，也不是 head_rank_gate |
| A：head_gate_n | 每个基底 RMS | 仅N | 每侧每head一个门 | 不同head在同一basis上参与RMS归一化 |
| B：head_gate_r | 每个基底 RMS | 仅R | 每侧每head一个门 | R=1时 H近似退化为符号（epsilon附近除外） |
| C：effective_key | 保留原始A | 不单独归一化H；归一化组合键HA | 每侧每head一个门 | `G=AAᵀ; norm²=(HG*H).sum(-1)`，避免显式N×d键 |

C 的 fp32/activation 选项只管 Gram/norm²统计；缩放、rsqrt与门沿用激活 dtype。A/B的 mix RMS使用通用 RMS helper 的统计路径，**不是 C 的 gram dtype 开关**。`scale_placement=mix/output` 是在rank→head展开前后施加相同缩放；有限精度可能不完全一致。

A/B 的实训同时采用 `key_scale=2/√R` 校准；C 通常用2。因此 A/B对legacy、或 R改变，并非只改一个归一化轴而完全保持数值尺度。AlignedRow 是把 LocalV row 的完整V输出乘 LocalO同一 `abs_v_cache_projection[V,C]` 后再注入；列读仍读完整M，不是把两侧都改读压缩M。

## Rank × 通路 × 规模：容量增加到底在哪里有效

以下统一把gap写为**高rank−低rank**。Q/K实验均同时改变Q与K，而且通常同时改变行列两侧；没有Q-only或K-only的因果对照，不能分别断言Q或K需要更高rank。LocalV只在L层生效，LocalQK在所有层生效，二者的容量与计算增量也不同。

| 规模、架构 | 改变通路/其余条件 | rank变化 | 高rank−低rank及趋势 | 能成立的结论/置信度 |
|---|---|---|---|---|
| Medium 全F普通V2 | Q+K，legacy | 1→2 | +.10967@200→+.00481@2000→+.00287@3400，停3521 | 观察期未获益，但在收窄；最终负作用未证实，中 |
| Medium 全F Paired40 | Q+K，legacy | 1→2 | 训满，−.00368@13400；吞吐−2.7% | 该上下文rank2有用，高；末期数字为单点 |
| Medium 全F Paired40 | Q+K，legacy | 2→4 | +.00050@7400、慢3.1%，停7542；但rank4对rank1仍−.00300 | rank4没有胜过rank2，不等于rank4对rank1无益；中，缺终局 |
| Medium 独立LLF | Q+K，legacy；V rank2 legacy | 1→2 | 早优在1400穿正，2000–3400均值+.00303；慢3.32% | 观察期增加QK rank不划算；中 |
| Medium 独立LLF | V双侧，legacy；QK rank1不变 | 2→4 | 400穿正，1600–2800均值+.00637；慢.97% | rank4 legacy有明确观察损害；中高，低rank底座更好 |
| Medium 独立LLF | V双侧，B+AlignedRow；QK rank1不变 | 2→4 | 2400–3400均值约−.00982（由降rank实验反向记号），rank2版停3580 | 与legacy方向相反；rank4在这个组合中更好，中；rank与scale规则一起变化 |
| Medium 独立LLF | 仅V行侧，B+AlignedRow；列rank4 | 2→4 | 低rank早优消失，高rank末期12200–13400均值−.001055 | 行侧rank4的晚期价值获训满支持，高；不能据此定量推出列rank收益 |
| XL 全F Partial | Q+K，legacy | 1→2 | −.02351@500→约−.010@4k–10k→−.00713@21500 | 收益衰减但长期保留，高（共同观察范围）；不依赖Medium式Paired40 |
| XL 独立LLF | Q+K，C-fp32；V rank4 C+AlignedRow | 2→4 | 早差、后转优，11.5k–22k连续负；末8点−.001017；公平health-OFF吞吐−2.62% | 当前支持rank4小收益，中高，仍在训 |
| XL 独立LLF | V双侧，rank与C/AlignedRow同时改变 | 2→4 | 18k–21k组合gap约−.001295 | 只能确认组合有益，不能确认单独V rank4有益；rank效应未定 |

这里有三组应主动关注的交互，而不是单一routing排名：

1. **QK rank1→2**：Medium普通V2/LLF观察期无益，Medium Paired40及XL Partial有益。Paired40不是必要条件（XL反例）；partial RoPE是否关键仍未隔离，不能把规模当作唯一解释。
2. **QK rank2→4**：Medium Paired40 legacy没有优于rank2，XL LLF C却有晚期收益。这同时改变了规模、架构和routing上下文，尚不能判断是XL需要更多基底，还是C使更高rank更可用。
3. **V rank2→4**：Medium legacy是负、B+AlignedRow是正。容量收益明显受参数化/对齐条件影响。但这是两个不同对齐条件下的rank对比，不能严谨称为已隔离的rank×routing交互，更不能只归功于B。

**当前缺项**：Q与K各自升rank；LocalV仅列rank变化；XL固定routing/AlignedRow只改V rank；Medium固定AlignedRow的rank2/4×legacy/B/C完整配对。没有这些，不能给出各通路的通用最优rank或贡献排序。

## 其余维度与组合交互：哪些已隔离，哪些仍混杂

| 维度 | 已有证据 | 仍不能推出的结论 |
|---|---|---|
| routing作用对象 | Medium QKV全改不利、V-only rank4 B/C有利；XL QK-only C有利 | 不能将首批损害直接归到Q或K，缺单路对照 |
| 行列两侧 | AlignedRow行侧rank4晚期胜rank2；DirectCol替代动态列读较差 | 列侧rank4对rank2的纯收益未隔离，不能用不同时段差值减出来 |
| 对齐方向 | LocalV V→C对齐与LocalO C→V解码均曾获益；后者未胜前者且较慢 | 支持兼容性假说，但压缩/解码也改幅度和梯度，不是纯坐标消融 |
| M访问空间/读键机制 | 完整M动态列读优于压缩M DirectCol；静态列读及静态+动态未获持续收益 | 完整/压缩M和动态低秩/直接逐头同时改变，不能单独归因某一项 |
| 迁移目标 | LocalV上C有效，照搬给LocalO列读无效 | V注入与O注入的作用不等价，不能靠相同contraction公式保证迁移 |
| bias与数值 | MixBias观察期有害；C activation弱于fp32；V C去bias在训近零 | mix bias、basis bias、Gram统计dtype是不同轴；不能合并成“bias有害” |
| 成本/质量 | AlignedRow同时改善末期loss与速度；行rank降至2却未变快；XL QK4有损失改善也有约2.6%速度成本 | 理论rank/FLOPs下降不保证实测增速；健康统计不同的旧速度不可直接相除 |

## 主线矩阵及结果

以下简称只用于本文；完整类名和 runtime 在后面的映射表与清单中。所有“早停”结果只覆盖观察区间，不能宣称13500/50000终局已知。

| 简称 | Q/K：rank,routing | LocalV：row/col rank,routing | 对齐 | 结果（gap与基线、窗口） | 状态/置信度 |
|---|---|---|---|---|---|
| M-Legacy | 1,legacy | 2/2,legacy | 无 | vs历史LLF：早期−.002～−.003收窄至约0；末六点到10k均值−.00038 | 暂停10607；中高 |
| M-AllA | 1,A | 2/2,A | 无 | vs M-Legacy：1000穿正，1800–2400约+.003～+.005 | 停2741；观察区间高、终局未定 |
| M-AllB | 1,B | 2/2,B | 无 | vs M-Legacy：早优消退，1600–2400约+.0015～+.0034 | 停2720；中 |
| M-AllC | 1,C-fp32 | 2/2,C-fp32 | 无 | vs M-Legacy：600穿正，800–2400约+.004～+.006 | 停2727；中 |
| M-AllC-bf16 | 1,C-act | 2/2,C-act | 无 | vs Legacy约+.007～+.009；vs fp32到2400约+.0038，速度仅+.12% | 停2735；中 |
| M-QK2 | 2,legacy | 2/2,legacy | 无 | vs M-Legacy：1400穿正，2000–3400均值+.00303 | 停3497；中 |
| M-V4 | 1,legacy | 4/4,legacy | 无 | vs M-Legacy：400穿正，1600–2800均值+.00637；速度−.97% | 停2894；中高 |
| M-V4A | 1,legacy | 4/4,A | 无 | vs M-V4B：4000–5000约+.001～+.0017平台；速度+.61% | 停5717；中高 |
| M-V4B | 1,legacy | 4/4,B | 无 | vs M-V4：1600–2800均值−.00779；vs M-Legacy：9400–10400约−.00146 | 完成13500；高（baseline较短） |
| M-V4C | 1,legacy | 4/4,C-fp32 | 无 | vs B：早期+.0548快速收窄，12200–13400均值−.000209，仍穿零；速度+.47% | 完成13500；B≈C高，C严格更优低 |
| M-BAlign | 1,legacy | 4/4,B | V→C | vs B：600–2600较差，2800穿负后扩大，12200–13400均值−.002836；速度+1.20% | 完成13500；高 |
| M-Row2 | 1,legacy | 2/4,B | V→C | vs BAlign：800–3400早优消退，6000后多为正；12200–13400均值+.001055；速度−.29%（反预期） | 完成13500；高 |
| M-V2BAlign | 1,legacy | 2/2,B | V→C | vs BAlign：2400–3400均值约+.00982；vs M-Legacy到3400仍+.00701且收窄 | 停3580；中，终局未定 |
| X-V4CAlign | 2,legacy | 4/4,C-fp32 | V→C | vs历史Rank2：25k–30k均值−.00294；vs独立LLF：18k–21k均值−.001295（重算） | 停30054；组合收益中高 |
| X-QKVC | 2,C-fp32 | 4/4,C-fp32 | V→C | vs X-V4CAlign：早优收窄，20.5k–29.5k保持约−.0013 | 停30013；高（上下文内） |
| X-QK4C | 4,C-fp32 | 4/4,C-fp32 | V→C | vs X-QKVC：早差后转负，11.5k–22k连续负；末8点均值−.001017；同health-OFF速度−2.62% | 在训，22k快照；中高、终局未定 |

### 其它相关消融，避免只留下成功案例

| 简称 | 改动及基线 | 观察结果 | 解读/置信度 |
|---|---|---|---|
| M-MixBias | M-Legacy全部QKV加mix bias | 2600–4000均值+.00520，停4037 | 无观察收益；中高 |
| M-Softplus | M-Legacy全部读门sigmoid→softplus，初开度匹配 | 800–2400仍+.0023～+.0044，2600降至+.00154；暂停2739 | 未胜但不能排除后续趋零；中 |
| M-DirectCol | BAlign的LocalV列键直接读C8，取消rank混合；行读在压缩M | vs BAlign 2200–4000约+.004～+.005；速度+.46%；停4197 | 完整M动态4basis胜过此压缩直接读方案；不能仅归因动态vs静态；中高 |
| M-ODecode | B上改LocalO row用压缩矩阵转置C→V | vs B 2800后约−.002～−.003；vs BAlign反复穿零且慢.87%；停9199 | 两种对齐均有支持，但对齐也改变表示/幅度；中高 |
| M-OColC | BAlign上LocalO列改完整M、rank4 C | vs BAlign早优在1400反转，3000–4400均值+.00430；慢2.05%；停4481 | LocalV上的成功不能直接迁移到LocalO；中高 |
| M-StaticCol | BAlign的LocalV列改静态可学习[V,N]键+动态head门 | 2000–3200均值+.00789；快.12%；停3251 | 失去动态键带来明显观察损害；晚期未定 |
| M-StaticDynamic | BAlign加静态列分支 | 800–2600微负波动，2800–3400转正；慢1.44%；停3487 | 没有持续收益；中 |
| M-CNoBias | M-V4C仅去LocalV读键bias | 200大正gap迅速衰减，1800起反复穿零，至4000近零 | 在训；未定 |

## 为什么看起来矛盾：能解释什么、不能解释什么

### 1. Rank与routing存在条件效应，而不是一个通用“rank收益”

同一1600–2800窗口，`M-V4−M-Legacy=+.006367`，`M-V4B−M-V4=−.007790`，相加为`−.001423`。这是同窗口可闭合的实证账，**不能**拿不同末期窗口机械相加。

M-V2BAlign没有把rank4的成功复制到rank2；只降row到2在训满时也损失+.001055。可确认rank4 BAligned上下文中row rank4有用；不能用早期双侧降rank约+.01减去末期单侧降rank约+.001，宣称col贡献+.009。缺少同窗口、同scale的col-only降rank及完整2×2。

### 2. “都改QKV”与“只改V”不是同一试验

首批AllA/B/C同时改了Q、K（rank1）和V（rank2）；后批只改V（rank4）。尤其B的R=1近似符号mix，是可分析的结构退化候选；但尚没有Q-only/K-only配对证明它就是损害根因。所有现有QK routing试验基本是Q、K一起改变，**没有足够证据给Q与K分别排贡献名次**。

### 3. 跨Medium/XL不能用共同改动本身解释方向相反

Medium全F Paired40 Rank2上C比legacy观察期较差（5600–6600均值+.00215，尚缓慢收窄）；XL LLF CAligned上下文中QK改C较好（约−.0013）。两边不仅尺寸不同，还同时不同：LLF/全F、Paired40读后V压缩与partial RoPE、QKV其它routing/坐标、WD谱系。结论是**存在上下文依赖**，不是“C在大模型必好小模型必坏”。

历史全F证据也要保留：Medium Paired40 QK1→2训满改善−.00368@13400；XL Partial QK1→2在21.5k仍−.00713；Medium普通V2 QK1→2却短训较差。这否定“Rank2只能依赖Paired40获益”的强结论，但不足以锁定partial RoPE为唯一原因。

SharedRankGate是另一条线：Medium Paired40末期−.00156；Medium LF LocalV末期仅−.00029；XL全F Rank2到6851约+.006～+.007。与A/B/C不同，不能把它们合成同一门控实验。PairedOrth补救XL SharedRankGate仅部分损害，未证明恢复到无adapter基线。

PairedIdentity也未补救：相对PairedOrth的gap从+.02370@500缩至2k–4.5k的+.00227～+.00328；相对历史Rank2同期+.00754～+.00844，暂停4799。它没有支持“只需把正交初始化换成identity就能消除XL负作用”的猜想，终局尚未验证。

### 4. AlignedRow的晚期收益真实，但“纯坐标原因”仍待证

BAlign在600–2600较差、2800后转优，最终约−.00284，说明短训可能错杀。反向让LocalO解码回完整V也有收益，支持坐标一致性假说。但压缩/解码同时改变有效线性算子、维数和梯度；不是只换无害的坐标标签。CAlign在Medium尚无直接长期配对，XL又同时改rank与routing，不能宣布因素已经完全解耦。

### 5. bias与Gram的数学合理性不是消融结论

`A=Wx+b`时，`||HA||²=H(AAᵀ)Hᵀ`严格成立；必须用包含bias的同一个A计算Gram。当前实现如此。bias诊断在XL QK4 checkpoint16000显示：bias范数占比小（row约.57–1.82%，col约1.36–2.81%），对joint rank4能量比例只改变不足.3个百分点；这说明所见冗余主要来自Wx，不说明去bias不会改变训练。NoBias为Medium V而不是XL Q/K，作用对象也不同。

## 剩余未定项与建议顺序（不是已授权的新训练）

复现控制也提供了确定的实现证据：`BamMediumPaired40Rank2CurrentControlRepro` 的原始0–100步在20步后分叉；`BamMediumPaired40Rank2HistoricalInitRepro` 仅恢复packed模块名对应的RNG路径后，0–100每10步重新匹配到最大约1e-6，RMS顺序未回退。因此本次C对照的前提不是“当前代码天然等价”，而是先排除了该初始化差异。该短训只验证早期轨迹，不保证任意后续重构长期同轨。

1. 先观察在训 M-CNoBias 与 X-QK4C；前者已接近零，后者有小收益，均不应仅凭局部两三点判终局。
2. 若目标是Medium最优配置统一：补**M-BAlign仅V B→C**最直接；现有M-V4C缺AlignedRow，不能当作已跑该配置。
3. 若目标是机制：固定QK与对齐，完成V rank2/4×legacy/B或C的因子配对，统一并记录scale规则；再决定是否值得分row/col。现有数据不是完整2×2。
4. 若目标是scaling：在XL同一LLF底座分别拆V rank、routing、AlignedRow，或在Medium同LLF的QK rank2上单改C。避免跨全F/LLF解释为纯尺寸差异。
5. Q/K分别routing、去bias与mix bias的交互、C-fp32/activation在最优V配置的对比都未被隔离；不要提前下“无bias才正确”“fp32永远更好”的结论。
6. 所有~1e-4级优势没有多seed支持；优先考虑结构简洁性和速度，但明确这是工程取舍而非严格质量排名。

## 简称与配置全名

主矩阵简称按清单中全名后缀对应：

| 简称 | 配置类全名 |
|---|---|
| M-Legacy | BamMediumIndependentLLFRoutingLegacy |
| M-AllA / M-AllB | BamMediumIndependentLLFRoutingA / BamMediumIndependentLLFRoutingB |
| M-AllC / M-AllC-bf16 | BamMediumIndependentLLFRoutingCFp32 / BamMediumIndependentLLFRoutingCActivation |
| M-QK2 | BamMediumIndependentLLFRoutingLegacyQKRank2 |
| M-V4 | BamMediumIndependentLLFRoutingLegacyLocalVRank4 |
| M-V4A / M-V4B / M-V4C | BamMediumIndependentLLFLocalVRank4RoutingA / BamMediumIndependentLLFLocalVRank4RoutingB / BamMediumIndependentLLFLocalVRank4RoutingCFp32 |
| M-BAlign | BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow |
| M-Row2 | BamMediumIndependentLLFAlignedRowLocalVRowRank2 |
| M-V2BAlign | BamMediumIndependentLLFLocalVRank2RoutingBAlignedRow |
| M-MixBias / M-Softplus | BamMediumIndependentLLFRoutingLegacyMixBias / BamMediumIndependentLLFRoutingLegacySoftplusReadGate |
| M-DirectCol | BamMediumIndependentLLFLocalVRank4RoutingBAlignedDirectCol |
| M-ODecode | BamMediumIndependentLLFLocalVRank4RoutingBLocalORowDecode |
| M-OColC | BamMediumIndependentLLFAlignedRowLocalOColRank4CFp32 |
| M-StaticCol / M-StaticDynamic | BamMediumIndependentLLFAlignedRowLocalVStaticCol / BamMediumIndependentLLFAlignedRowLocalVStaticPlusDynamicCol |
| M-CNoBias | BamMediumIndependentLLFLocalVRank4RoutingCFp32NoBias |
| X-V4CAlign | BamXLIndependentLLFLocalVRank4CFp32AlignedRow |
| X-QKVC | BamXLIndependentLLFLocalQKVCFp32AlignedRow |
| X-QK4C | BamXLIndependentLLFLocalQKRank4CFp32AlignedRow |

全F及复现/测速控制完整名称、runtime、原始结果备注见[清单](local_qkv_experiment_inventory.md)。未完成或台账缺失的观察不会当作阴性证据。

## 重现与实现边界

代码入口：`MaxText/layers/attentions.py` 的 `_LocalReadArm`、`_packed_local_arms_init`、`_read_local`、`factorized_head_bam_read`、`_gram_read_norm2`。Q/K/V fallback沿用配置约定；row/col rank可分别覆盖；AlignedRow由`bam_local_v_share_output_coordinates`控制。均为可选路径，未将新routing设为普通BAM默认值。

本次不把实验worktree的 `train.py` health-OFF兼容开关带入主干；主干通用健康统计继续开启。NoBias历史health-OFF运行仍在其已封存commit。历史head_rank_gate保持台账，已被实验分支移出的实现不恢复。其余本轮可选消融路径随该实验族集成，默认关闭，不作为推荐生产配置。

集成来源：`codex/local-read-gram` 的 `0dcd3e10fcf981330f1c3fc42884ba4e855d4d2b` 工作树实现；主干原有未提交的LocalQKV统一重构保留。合入后 `attentions.py` 与该来源逐字一致，SHA256为 `75891da720d67e37c89b62a072a39ea8844fec1b179350ca6706d026e452d3a7`。这不是声称主干完整训练runtime等于该历史commit。

生成清单：`python3 experiments/bam_llama2_medium/export_local_qkv_inventory.py`。重算配对：在tpu-ag执行 `collect_local_qkv_evidence.py`（只读loss缓存，输出JSON；不改compare_runs、不启训练）。JSON中的截止步为请求上限，实际可用末步必须看series；例如NoBias导出时只到4000。

检查：固定CPU入口 `bash .claude/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh /home/xd/projects/maxtext`，另跑 `bam_gram_read_test`、`bam_local_fetch_test`、`bam_config_test`。这些验证前向/梯度/参数形状与配置路径，**不等价于历史13500/50000步训练复现或重新测速**。

本次验证：attention 53项通过；补充17项中15项直接通过，2项测试适配后复测通过（通用健康统计应存在；模拟receiver补齐门激活与可选decoder属性）。共70项测试完成，`git diff --check`通过。没有通过修改生产计算规避失败；主干通用健康统计保留的行为已由scan/non-scan两路测试确认。

诊断旁证：[QKV键相似性](local_qkv_key_diagnostics.md)、[XL QK4基底](xl_qkr4_basis_diagnostics.md)、[逐层分侧bias](xl_qkr4_bias_layer_summary.md)。诊断机按用户要求保留。
