# Q/K/V/O行读：逐层×逐头因果必要性与弱头组合验证

2026-09-14，Medium/XL全部完成。相同64条Pile，各测1344个非零、有效层/路径/head组合，并额外完成前32条选头、后32条验证的联合删除。此前的整路、逐层、深度、Local/Fetch比较见[统一报告](/home/xd/projects/maxtext/experiments/bam_llama2_medium/row_contribution_scale_comparison.md)。

## 主要发现

1. **确有特别重要的head，最清晰的是V和O。** Medium O/F5/h15单独删除 **+0.024758 ± 0.003052**，XL O/L1/h10 **+0.009005 ± 0.001065**；V最强均在L1，Medium h3 **+0.002880 ± 0.000777**，XL h5 **+0.009625 ± 0.001671**。这些head在54–64/64条序列上删除后变差。
2. **“重要层”未必只有一个不可替代的head。** Medium O/F11整层关闭损失0.204305，但最强单头仅0.003184；16个singleton效应相加也只有0.011518。XL O/F5同样：整层0.186768、最强head0.005771、singleton和0.010449。多head联合删除存在很强非加性，不能用单头分数求和代替整层功能。
3. **V的弱头是较清晰的低损伤候选，XL尤其明显。** 每层删最弱8/16个V行头，留出集Medium/XL损失 **0.003552/0.002587**；删最强8个则 **0.025852/0.066595**。同一留出集V整路关闭为0.031304/0.079104。选择弱头明显好于选择强头，但没有做真实结构裁剪或重训。
4. **O能区分强弱，弱头整体也并非零作用。** 每层删弱4/16个O行头，Medium/XL **+0.015134/+0.013193**；弱8/16为 **+0.044804/+0.035535**，远小于强8/16的 **+0.936117/+0.937051**。对应结构裁剪有较大的O行key预算，但“热图中很白”不能解释为可任意累计删除。
5. **Q/K细头排名的可靠性有限，尤其Medium。** Medium前后两半Top4平均重合约1个，与从16头随机挑4头的期望重合数1相当；留出集top−bottom组差也都跨零。XL Q/K组级强弱差较清晰，但单层top4排名仍不稳定。XL Q/L16/h4是比较突出的单头候选（+0.000601），不据大量探索性比较宣称其它头严格无用或有害。
6. **四路都按弱头一起裁的损伤接近两个模型。** 每层每路删弱4/16：Medium/XL **+0.016434/+0.017090**；弱8/16：**+0.049899/+0.047400**。这是实际联合干预，不能从四路单独实验简单相加推得。

这里比较的是两个不同训练谱系的冻结checkpoint：XL还改变宽度、rank、basis共享、路由、训练时长与历史数值/WD设置；不把差异全部归因于规模。

## 哪些头特别重要

下表“单头最大”按全64条均值选择，误差未校正多重比较。集中度与重合数在每条路径的有效非零层之间汇总，**不是最大head所在层的集中度**。

| 模型 | 路径 | 单头最大：层/head | Δloss ±1.96 SE | 删除后变差序列 | Top4集中度中位数 | 两半Top4平均重合数 |
|---|---|---|---:|---:|---:|---:|
| Medium | Q | F2 / h1 | +0.000252 ± 0.000233 | 36/64 | 52.3% | 1.00/4 |
| Medium | K | F8 / h8 | +0.000203 ± 0.000159 | 39/64 | 55.4% | 0.91/4 |
| Medium | V | L1 / h3 | +0.002880 ± 0.000777 | 54/64 | 62.2% | 1.47/4 |
| Medium | O | F5 / h15 | +0.024758 ± 0.003052 | 63/64 | 41.7% | 1.83/4 |
| XL | Q | L16 / h4 | +0.000601 ± 0.000268 | 44/64 | 83.4% | 1.09/4 |
| XL | K | L1 / h13 | +0.000408 ± 0.000272 | 41/64 | 80.5% | 1.43/4 |
| XL | V | L1 / h5 | +0.009625 ± 0.001671 | 63/64 | 88.9% | 1.40/4 |
| XL | O | L1 / h10 | +0.009005 ± 0.001065 | 64/64 | 52.1% | 1.35/4 |

Medium O/F5的h15占该层正singleton分数和74.7%，Top4合计89.2%，是最清楚的单头尖峰之一。XL O/L1的h10占48.6%，Top4占82.5%。这些百分比只描述单头删除分数分布，不是对整层删除损失的Shapley分配。

![24层×16头的四路分布](/data0/xd/bam_diagnostics/row-contribution-heads-0914/head_heatmaps.png)

每条路径在Medium/XL之间使用同一色标，四条路径之间色标不同；白色接近0，红色删除后变差，蓝色删除后改善，灰色表示Fetch层无Local V。接近零且不稳定的头可作为候选，不把蓝色探索性均值当作“有害头”的证明。

完整每路Top4列表见[重点head表](/data0/xd/bam_diagnostics/row-contribution-heads-0914/head_top_tables.md)，每路Top8/Bottom8含均值/SE/正号序列数见[JSON](/data0/xd/bam_diagnostics/row-contribution-heads-0914/head_top_tables.json)。全部1344项及前后半排名保存在各模型的`heads/head_summary.json`。

## 弱/强head联合删除：后32条验证

`bottom4/8`：每个有效层中按前32条singleton均值选最弱4/8头并一起删除；`top4/8`选最强。`all_Q/K/V/O`是同一后32条的整路关闭对照，**不是之前报告的全64条均值**。`all_bottom4/8`同时删除四路各自选出的弱头。以下均为后32条Δloss ±1.96配对SE。

| 逐层删除的行head | Medium：后32条 Δloss | XL：后32条 Δloss |
|---|---:|---:|
| all_Q | +0.001492 ± 0.000655 | +0.008201 ± 0.001366 |
| all_K | +0.001529 ± 0.000488 | +0.008693 ± 0.001394 |
| all_V | +0.031304 ± 0.004750 | +0.079104 ± 0.010514 |
| all_O | +1.864532 ± 0.176531 | +1.871118 ± 0.162219 |
| Q_bottom4 | +0.000219 ± 0.000272 | +0.001223 ± 0.000592 |
| Q_top4 | +0.000657 ± 0.000494 | +0.002678 ± 0.000826 |
| Q_bottom8 | +0.000726 ± 0.000357 | +0.002337 ± 0.000690 |
| Q_top8 | +0.000989 ± 0.000481 | +0.003989 ± 0.000963 |
| K_bottom4 | +0.000324 ± 0.000295 | +0.001322 ± 0.000519 |
| K_top4 | +0.000468 ± 0.000325 | +0.003547 ± 0.000836 |
| K_bottom8 | +0.000461 ± 0.000391 | +0.002506 ± 0.000751 |
| K_top8 | +0.000710 ± 0.000425 | +0.005213 ± 0.001123 |
| V_bottom4 | +0.000956 ± 0.000581 | +0.001252 ± 0.000548 |
| V_top4 | +0.016744 ± 0.002865 | +0.045486 ± 0.006547 |
| V_bottom8 | +0.003552 ± 0.001063 | +0.002587 ± 0.000927 |
| V_top8 | +0.025852 ± 0.004238 | +0.066595 ± 0.008729 |
| O_bottom4 | +0.015134 ± 0.003177 | +0.013193 ± 0.002494 |
| O_top4 | +0.349076 ± 0.047816 | +0.218510 ± 0.032453 |
| O_bottom8 | +0.044804 ± 0.007451 | +0.035535 ± 0.004914 |
| O_top8 | +0.936117 ± 0.116150 | +0.937051 ± 0.112857 |
| all_bottom4 | +0.016434 ± 0.003148 | +0.017090 ± 0.003002 |
| all_bottom8 | +0.049899 ± 0.007980 | +0.047400 ± 0.006020 |

![强弱组在留出集的效果](/data0/xd/bam_diagnostics/row-contribution-heads-0914/head_group_validation.png)

V/O的强弱组区分很明显。Medium Q/K则不同：top4−bottom4配对差分别为0.000438±0.000525、0.000144±0.000359，均跨零，当前数据不足以稳定选出最该留的细头。XL对应为0.001455±0.000818、0.002225±0.000921，有组级选择信号，但这不消除单层排名的不确定性。

低损伤候选应从这些**联合实测**出发：V弱半头的loss损伤小；O弱半头比强半头损伤小得多，却仍有0.036–0.045左右的损伤。不存在已验证的“任意删弱头都无损”。四路弱头组损伤的主要量级接近O组，但没有进一步对联合组内部做Shapley分配。

## 口径与计算

对每个非零层、每条Q/K/V/O路径、每个head，单独关闭其BAM行读输出，完整重算下游loss。两个模型均16 heads；Q/K/V是Local read的输出head，O是fetched read的输出head（这两个checkpoint的fetched heads恰好也是16）。不是删除完整MHA head：列读、普通attention及其余行读保持原生。

每模型24层×4路×16头；第0层M为零，64个已知零项省算；8个Fetch层没有Local V，128项不适用。实际1344个head干预×相同64条Pile序列＝86016个配对测量/模型。每个分片、每条序列另跑native和四个整路关闭对照。图中的层号/head号均从0起，L/F分别表示Local/Fetch。不同层或不同模型的同号head没有预设的语义对应。

正值表示删除后loss升高，单位nats/token；每条序列按有效target归一化，然后等权平均。表中误差为±1.96配对标准误，未做多重比较校正。全部细粒度排名是探索性结果，不能将每个head的Δloss直接相加当作唯一因果份额。Top4集中度专指该层最大4个正singleton分数之和占全部正singleton分数之和。

组合验证用前32条的**有符号singleton均值**在每层/每路径独立排序，选择bottom/top 4或8个head；之后在后32条评估。这32条仅从本轮head选择中留出，之前的粗粒度研究用过同一64条，并非全新数据集。所有层一起执行相同稀疏率，共16种单路径组＋四路径bottom4/bottom8两种联合组，另5对照＝23×64 loss/模型。

### 实现、核验与耗时

- 冻结checkpoint、`only_eval=True`，关闭所有激活/健康统计采集；一次restore、复用同形状编译。Linen拦截`BamAttention.__call__`取得实际动态层号，在`_read_local`/`_read_fetched_m`最终输出只对行切片乘bool mask，gate本身保持原值。
- Tiny CPU验证覆盖两种head宽度、动态层索引、单head选择、其他路径/列读保留、gate不变以及完整1344场景覆盖。生产`MaxText`模型源码相对训练commit未修改。
- 每分片每样本重复native/Q/K/V/O五个对照，与原始整路诊断逐值比较，容差1e-6；首样本验证普通forward、第0层/F-V无操作，每条样本再验证一个head/group与串行执行一致。原始loss均有限，cohort五字段hash核对一致。
- 对比scalar、同一jit kernel异步提交、设备端`lax.map`。异步与scalar完全一致，约快3–5%；设备端loop有约7×10⁻⁴偏差，被拒绝，没有用于统计。后续runner默认关闭loop候选，显式`HEAD_TRY_LOOP=1`才重新验证。单纯改用`where`也曾破坏native一致性，最终采用与原整路干预一致的乘法mask。
- 实际每序列：Medium每分片453场景约5.36秒，XL每分片273/274场景约7.58秒。Medium按3个head分片、XL按5个head分片，每片均覆盖64条；共8个运行/排队资源上限。一个Medium资源分配迟迟未完成，改由已完成分片的机器串接最后一片。两台早期资源被云端抢占后，仅在确认终态后补位。
- 原始文件由worker直接上传GCS，本机从GCS拉取。活跃阶段同步排除不断覆盖的`.log`文件，避免对象版本变化引起404；最终静止后同步完整日志。逐文件大小/SHA256、数值和覆盖验证保存在`verification.json`。

## 参数与结构裁剪含义

`W_Q=D²`。O行key每层为0.5 W_Q，16个输出head可对应独立投影输出块；若以后真实裁剪每层4/8个O行head，包括第0层的无效行读块，理论可删全模型**3/6 W_Q**行key kernel（原12 W_Q的25%/50%）。本次非零层组合只测层1–23，第0层另有严格无操作检查；不裁第0层时对应2.875/5.75 W_Q。

Q/K/V的行key basis在head之间共享，XL Q/K还在路径之间共享。少掉4/8个输出head不能按25%/50%线性扣掉这些basis kernel，只可能减少相应mixing、gate和收缩工作；去掉整套basis需要所有消费者不再使用。上述成本不包括全部BAM参数。

诊断mask未实施真实结构压缩，没有测到训练/推理加速，也不减少M-cache。冻结模型可删性不等于重训后最优结构；低秩、跨层共享、静态key和幅度校准实验仍按用户要求暂停。

## 复现与资源

| 项目 | Medium | XL |
|---|---|---|
| 完整配置 | `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow` | `BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis` |
| checkpoint step | 13500 | 30091 |
| 训练commit | `77401da6f83a5aa6ddd61994e028c3c694221518` | `c664e8211c299c9c780ce1667553cf995130c453` |
| head矩阵runtime | `1cc43c80531ec733b03181389676b37246ca8b29` | `8dd1d35424b707480e19f4dc1352084f05d6170d` |
| group验证runtime | `34470eeca6845e7b201a6cbe5f234460437346c4` | `6e90ba25e576009d1e562ceb230ee3a8fae16af3` |
| worktree | `/data0/xd/llf-row-contribution` | `/data0/xd/xl-row-contribution` |
| 分支 | `codex/llf-row-contribution` | `codex/xl-row-contribution` |
| artifact目录名 | `llf-row-contribution-13500-0914` | `xl-row-contribution-30091-0914` |

Medium checkpoint：`gs://newproject-1-llm_projects_us-east5/log/BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow/checkpoints/13500/items`。
XL checkpoint：`gs://newproject-1-llm_projects_europe-west4/log/BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis/checkpoints/30091/items`。

共同cohort：`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`，文件SHA256 `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`。使用索引0–63，每条2048 token，总有效target 130974，不增加样本。

两worktree均含`experiments/bam_llama2_medium/row_head_contribution.py`和`run_row_head_contribution.sh`。按上表runtime checkout后，设置：

```bash
HEAD_OUTPUT=/tmp/MODEL-row-heads-shardN \
HEAD_GCS=gs://newproject-1-llm_projects_europe-west4/log/diagnostics/ARTIFACT/heads/workerN \
HEAD_REFERENCE_GCS=gs://newproject-1-llm_projects_europe-west4/log/diagnostics/ARTIFACT/heads/reference.npz \
HEAD_SHARD=N HEAD_SHARDS=3_or_5 HEAD_START=0 HEAD_STOP=64 HEAD_CHUNK=16 \
bash experiments/bam_llama2_medium/run_row_head_contribution.sh
```

分片按场景索引取模，5个对照每片都执行。`reference.npz`取原始整路实验native/Q/K/V/O的64×5 loss及输入hash。全开与各单路关闭必须和该参考一致。

用`summarize_row_heads.py MODEL_ROOT`生成完整head均值/SE/排序、`paired_head_gaps.npz`和按前32条选出的`group_scenarios.json`。group阶段设置`HEAD_STAGE=groups HEAD_SHARD=0 HEAD_SHARDS=1`、独立输出/GCS目录，并将`HEAD_SCENARIOS_GCS`指向该模型的`heads/group_scenarios.json`；同一launcher会下载不可变配方，metadata记录配方SHA256。group结束运行`summarize_row_head_groups.py MODEL_ROOT`，保存all64/first32/heldout32及top−bottom配对差。

图脚本`plot_row_heads.py MEDIUM_ROOT XL_ROOT OUTPUT_ROOT`和`plot_row_head_groups.py MEDIUM_ROOT XL_ROOT OUTPUT_ROOT`。下载脚本`pull_row_head_artifacts.sh GCS_HEADS_PREFIX LOCAL_HEADS_DIR [--complete]`，只有全部uploader结束后才用`--complete`纳入日志。运行脚本及聚合/绘图脚本已提交两个诊断分支，图和原始NPZ不写进Git。

本地模型artifact根目录为`/data0/xd/bam_diagnostics/`＋上表目录名；本次数据均在其`heads/`子目录。GCS根为`gs://newproject-1-llm_projects_europe-west4/log/diagnostics/`＋同目录名＋`/heads/`。比较图、重点表和资源清单在`/data0/xd/bam_diagnostics/row-contribution-heads-0914/`，同名GCS诊断目录有归档。

完整逐层/逐头结果：各模型`heads/head_summary.json`，原始64序列数组`heads/paired_head_gaps.npz`，排序配方`heads/group_scenarios.json`，联合组结果`heads/group_summary.json`和`heads/paired_group_gaps.npz`。原始head文件Medium 192个、XL 320个；原始group文件各64个。两模型共172032个有效单头干预测量，含分片重复对照和group组共保留177536个loss。

### 保留机器

按用户“跑完别删”的要求未删除任何本任务机器。当前7台READY的v6e-1，均`europe-west4-a`，诊断Python进程已退出：

| 机器 | 本轮工作 |
|---|---|
| `xd-v6e-headko-m0r-ewa4a-0914` | Medium head shard0 + groups |
| `xd-v6e-headko-m2-ewa4a-0914` | Medium shard2后串接shard1 |
| `xd-v6e-rowko-2-ewa4a-0914` | XL shard0 + groups |
| `xd-v6e-headko-x1r-ewa4a-0914` | XL shard1 |
| `xd-v6e-headko-x2-ewa4a-0914` | XL shard2 |
| `xd-v6e-headko-x3-ewa4a-0914` | XL shard3 |
| `xd-v6e-headko-x4-ewa4a-0914` | XL shard4 |

第8个资源`xd-v6e-headko-m1-ewa4a-0914`仍在PROVISIONING，未重复分配任务，待其就绪也保留。早期`xd-v6e-rowko-0-ewa4a-0914`/`xd-v6e-rowko-3-ewa4a-0914`被云服务抢占，确认PREEMPTED、队列SUSPENDED后才请求替代；没有因补位超过8台运行/排队上限。历史终态资源未删除。当前权威归属为比较产物的`resources.json`、`retained_nodes.json`、`retained_queues.json`以及两个模型的`heads/group_resource.json`；此前粗粒度报告的三台保留状态是旧阶段快照。
