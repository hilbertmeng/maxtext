# MediumProp V48：MLP写入内容的每头坐标变换

父版 BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile（c1a68fe，13500步完成）。新RUN BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdContentTransformTruePile。
实现worktree /data0/xd/mediumprop-k75-embed，分支codex/mediumprop-k75-embed；主exp.py仅台账，模型改动未合入主分支。

零起始1/4/7/10/13/16的MLP写入层各有独立T[16,75,75]，单位阵初始化；y_head @ T 后沿用每头内容RMSNorm、原写门、独立动态地址与原M合并调度。向量残差仍加入未变换的原MLP输出；attention与embedding写入不变。不增加bias/激活，不混合头。

每层+90000=.0625W_Q，六层+.375W_Q。每个MLP宽度单位3600参数；写入层精确减25，宽度[3765,3525,3765]，总432115072，与父版一致（MHA−6128）。单写入层变换理论计算为一个W_Q的1/16；十八层平均1/48。

动机：独立地址解开写入位置，本实验解开向量残差输出和矩阵内容坐标。单位阵保证新增映射本身不制造初始化差异，但抵扣MLP宽度后的整模型并不与父版逐值等价。
新增健康：变换前后RMS比与每头内容余弦（及绝对/平方余弦），只记录六个写入层；保留父版全部健康统计。
13500步、200步loss窗口、1000步批量报告，2800/5000判定。TPU xd-v5p-16-2910077-maxtext，主区UE5a；编译借用EW4a FLEX_START llm-jax-v6e-1-0，禁止自动回收。产物/data0/xd/bam_diagnostics/rmt-readnorm-launch/content-transform-*。

运行源码524c9ad931d7aba359b484286ef01da34ea98afb。两项针对性检查39.0s通过：单位阵同宽父版输出等价、原参数初始化逐值一致、有效变换梯度、六层范围与参数/健康指标；47项回归四组并行137.3s通过。受保护FLEX_START编译机完成目标v5p16/s13500 AOT；18:03 UTC正式UE5a训练通过FIRST_STEP，18:04继续到21步；两worker源码524c9ad、目标AOT和UE5a TruePile路径已核对。稳态速度待测，不将首步.444当作稳态。

抵扣后主要线性MAC保持一致：新增16×75²=90000，MLP每写层减少3×1200×25=90000；只忽略少量激活/归一化点运算。不能把未抵扣时的1/16 W_Q计算开销当作新run的净计算增加。UE5a原prequeue保留，16:21 UTC加排EW4b被动候选，先到READY的正式启动后需等待FIRST_STEP再释放未选候选。

UE5a主训练启动后释放未选EW4b候选；记录/data0/xd/bam_diagnostics/rmt-readnorm-launch/content-transform-ew4b-release.log。

20:03 UTC迁入已通过实际SSH检查的UC1a候选；源码/AOT/13500步计划不变，恢复checkpoint98。已达226步并提交200 checkpoint；200步六层变换RMS比1.006–1.027、内容余弦.99899–.99969，写门均有限，raw梯度2.547。UC1a约.502step/s，对父版.503约−.2%；原UE5a资源已核验释放。

2026-10-06用户停止：最终checkpoint8674提交；先前已发生第6次抢占，12:03:12 UTC终止恢复；资源删除重试后于12:14:22完成，TPU/queued-resource均不存在，TB同步完成。相对V48父版，初始瞬态后一直在零附近波动；最后完整8600窗口的五点均值-.000263（-.000779～+.000248），没有稳定收益。匹配健康速度.502/.503，约-.2%；本实验在等参数、主要线性MAC相同的条件下未显示优势。主exp.py已记结论并删除下注。

区域路径UE5a→UC1a（切换2026-10-05 20:03:00 UTC），6次抢占、6个READY租期均记录在主`experiments/tpu_region_preemption_history.md`。EW4b只作被动候选，未作为正式训练区。官方收尾摘要`tpu-ag:/home/lishengping/xd/projects/logs/closeout-20261006T121422Z.json`；完整loss表`/data0/xd/bam_diagnostics/rmt-readnorm-launch/content-transform8600-report.log`。
