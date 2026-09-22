# AllLocal 四分量、写门与 local O 写回几何

状态：2026-09-22 已封存诊断并提交资源，尚无结果。

模型 `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal`，最终 checkpoint13500，训练 runtime `ba299404f17b366a4e41bce5fef9306f5dfe8a17`。主台账是 ledger only；使用历史实现，不能用当前主干同名配置替代。

## 事前预测（数据产生前固定）

1. 四部分不会普遍正交；跨 token BAM 与本地 O 同向、冲突两类都会出现。更具体下注：除最早层外，local O 往往是四部分中模长最大的，本地 attention BAM 项通常最小。
2. 写门的几何关联主要存在于层/头内部，混合全局会掩盖它；组合特征会比任意单个夹角更有解释力。猜至少半数普通层，几何量在控制头身份/位置之后，能解释留出数据至少15%的剩余写门方差。探索不优先用户举例的任何方向。
3. 实际 O 写回主要加强旧内容或轻度抵消；反向超过旧内容的情况少于1%，集中在少数层/头/位置。不同意用未经门控和归一化的读出模长代替实际写入强弱。

下注的目的：约束事后解释，显式记录哪些先验被推翻，而不是据此筛选分析结果。

## 范围和口径

固定 Pile cohort，T2048，先64条；前32用于探索，后32用于留出复核。只有会改变结论的不确定性才增至128条。有效输入token保留 layer×token×head 记录，L0零记忆/未定义夹角单列；不把L1尖峰作为重点。

四分量为原始V的attention加权和a、本地BAM V的attention项b、其他token的BAM V attention和c、本地O注入d，均取前48维。记录10个上三角Gram元素，可重建全部模长和两两夹角；另存实际写门、self alpha和读门。b/d共线是结构校验，不作为发现。使用真实attention权重，以FP32计算诊断分解，并记录与实际低精度o_head的重建误差；原始前向不改变。

读键从C8转换为r=Pq（q为实际RMS处理后的读键）；写地址p为实际写地址归一化输出。记录同头cos(r,p)。O分量使用实际已门控d和总写入data RMS分母：wO=gw*d/sqrt(mean(o_head_front²)+eps)。在单位写地址p_hat处，旧内容=M p_hat，O更新内容=wO*||p||。因此实际更新/旧内容模长比rho=||wO||*||p||/||M p_hat||，包含读门、写门、总data归一化及外积地址幅度。记录两者夹角、模长及rho；相对平行分量rho*cos<-1代表超过旧内容在原方向上的幅度，不把“负cos且rho>1”误当作充分条件。此为实际前向O分量分解，不是删除O后重算归一化的因果消融。只做同头，不做跨头比较、全M能量归因或新的架构干预。

## 双门筛选（运行前补充）

主要统计以local O读门sigmoid与写门sigmoid均>=0.1的位置为准；0.05和0.2两档作敏感性检查，未筛选作附录对照。原始记录全部保留。逐层逐头记录符合条件的token数、独立序列数、探索/留出各自覆盖度，样本不足时扩至128，不通过重复抽样伪造覆盖。层0零读出不进入双门有效样本。写门是筛选变量，因此筛选后规律描述“开门后的开度”，不能外推成开门原因。

## 探索与展示

完整经验分布来自全部有效token；不以均值替代。保存逐头分位数、二维经验直方图、条件写门分布、逐序列分箱计数和和，支持按序列重抽样。幅度用绝对norm及四模长之和占比，夹角六对全部覆盖；加入总和相干程度以识别多分量组合。近零/未定义角度保持NaN，不当作正交。

从每序列每头随机取128个token作可控开销的探索建模，分布本身不抽样。比较以head和position预测写门的基线与加入全部几何特征的模型，严格按序列32/32分离；报告验证误差改善、分组打乱的重要性和逐头秩相关。用于解释写门的特征均不乘写门；actual write ratio依赖写门，禁止用其预测写门造成泄漏。相关几何特征间可互相替代，打乱重要性不是因果归因。后续根据候选规律增加分组和可读图，而不是止于模型分数。

## 复现与资源

worktree `/data0/xd/alllocal-write-geometry`，分支 `codex/alllocal-write-geometry`；初始诊断runtime `97749c0f7e30af8ab4160c5a57176251c61f245f`，生产MaxText源码相对训练runtime无改动。使用Flax方法拦截采集，原始forward原样返回。

入口 `experiments/bam_llama2_medium/run_write_geometry.sh`，脚本 `write_geometry.py`、`analyze_write_geometry.py`、`benchmark_write_geometry.py`；GEOMETRY_N=64，only_eval=True，checkpoint只读。TPU采集时4个CPU线程并行落盘/分位数，随后最多24个CPU进程按层分析，单进程数值库线程限制1。先比较相同两层串行与双进程结果及速度，CPU资源/内存与实际并行时间保留。

本地 `/data0/xd/bam_diagnostics/alllocal-write-geometry-13500-0922`；GCS `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/alllocal-write-geometry-13500-0922`。候选 `xd-v6e-alllocal-geom-*-0922`，三个诊断区竞速，资源清单由resources.json记录。原始sample_NNN.npy约5.4GB/64序列；保存token/position/valid、输入hash、前向数值一致性记录和完整分布数据。产物验证后释放全部资源。
