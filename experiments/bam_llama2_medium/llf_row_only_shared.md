# LocalV 只共享 LocalO 行读

实验位置：`/data0/xd/llf-row-only-shared`，分支 `codex/llf-row-only-shared`。从主工作树 `refactor-bam` 的 `ac7fccc6` 建立，并复制其未提交的 LocalV 模式重构、配置、模型传递、测试及相关文档。主工作树仍保留原有未提交改动；本分支把复制的依赖连同实验实现一起固定到 runtime commit。

实验类与 RUN：`BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B`。正式训练 TPU `xd-v5p-16-llf-row-shared-maxtext`，主区 `europe-west4-b`，v5p-16，13500 步，checkpoint 每 200 步，Generic health ON、BAM sow OFF。runtime commit `502b479d69088845e93be07c37b67e1d180dbf38`；后续台账元数据 commit 不改变 RUN 的 runtime hash。registry 的比较对象是 `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow`（直接基线）、`BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan`（LocalV 行列都从 LocalO 共享读出，旧 runtime 谱系）、以及 `BamMediumIndependentLLFBAlignedRowSharedRowRank4CFp32`（共享 row rank4 基底的近邻架构）。`BamMediumIndependentLLFLocalVRank4RoutingBAlignedRowSharedRead` 保留作历史参考：它在独立 rank4-B LocalV 行列读之外，额外叠加一条共享 LocalO 行列读，并未替换独立读出。

L 层的 LocalO 仍从压缩的 32×8 M 用每头 40 维 `W_R` 读出。LocalV 不再投影或收缩独立的行键，而是共享 LocalO 尚未门控的 16 头行读结果，另外用自己的门控。LocalV 的列读继续以独立的 rank4 B 键从完整 32×32 M 读出，然后以每头混合系数和独立列门控扩展。LocalO 自身保持原有门控；Q/K、F 层及写入未改。历史 M-cache 大小不变。与 BAlignedRow 相比，L 层投影输出宽度减少 192，即每层节省 192×1024 = 196608 个投影权重，约 0.1875 `W_Q`；16 个 L 层共节省 3145728 个权重。

跑前下注：相对 BAlignedRow，3000–4000 步 dloss 预计 +0.002～+0.007，稳定速度预计 +0～+1%。主要依据是 LocalO 的每头行键及完整 LocalV rank4 列键都保留，而两侧都 shared 的旧实验还移除了独立列读；LocalO/LocalV 行列都各自共享基底的实验同时改变了 LocalO 的行读，这次也没有改变。预测有不确定性，跑后按相同步数和相同健康设置核对。

验证：新增 LocalV 模式测试 13 项、固定 CPU BAM 测试 43 项、配置测试 8 项全部通过。保存的重构前源码与本分支对 `BAlignedRow` 的 rank4 LocalV 参数、前传、M 输出、参数/输入/M 梯度逐元素完全对齐。目标 v5p-16 AOT 编译、上传及 manifest 验证通过，worker 确认 `Loaded compiled function!`；`FIRST_STEP` 在 step13，step46 约 0.686 steps/s。历史 generic-health-ON BAlignedRow UE5a 为 0.6836 steps/s，但与本次 EW4b 跨区，暂不能作为匹配测速结论。RUN 正在继续训练。
