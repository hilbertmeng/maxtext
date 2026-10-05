# XLProp BAM V60/C15

RUN：BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile。
父版：BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile。
工作树/分支：/data0/xd/mediumprop-k75-embed / codex/mediumprop-k75-embed。
新 TPU：xd-v5p-32-2910074-maxtext，优先 UC1a，排不到可加排 UE5a/EW4b；依据近期同拓扑租期，UC1a 集中 maintenance 后已恢复。借用 EW4a FLEX_START llm-jax-v6e-1-0 编译，编译机不归本任务回收。

迁移 Medium M75x48/C12 的地址扩容方案：M96x40/C10 -> M96x60/C15，压缩率维持4:1。Attention/embedding 写地址 R400->600，独立 MLP 地址 R384->576。QK72+RoPE24、D1920、28层、20头、每第三层的1/4/.../25写入位置、dot写和dot_btn读、健康开关、初始化、WD和TruePile T4096数据保持父版。

新增 BAM 参数 39,262,800 = 10.65072 W_Q，普通层1,132,180，MLP写入层1,885,220，共享embedding784,400。精确抵扣后MLP三宽[6092,5773,6093]，末层6094；总1,432,396,680，MHA1,432,398,720，差-2,040（最近可实现整数总参数）。各层近似等总参数，未硬件取整。独立向量残差保留，无fetchedO、无跨token M-cache，前传matrix激活增长50%。

下注：终局相对父版-.010，速度.330step/s vs父版.347（-4.9%）；Medium目前后期收益约-.007，为正向跨尺度迁移提供依据，不能证明XL收益大小。
50000步，500步loss窗口，2000步成批报告；评估点6000/10000，相对父版和LLF判断收益。checkpoint250、每4000永久保留、最近2份，终局checkpoint。
基线：直接父版、XLProp LLF、Mudd、同数据MHA；同时报/Mudd和/LLF收益倍数。

验证：定向CPU全参数树/形状/MLP写入health，含末尾L的7层有限梯度；sealed-runtime，v5p-32 AOT及实际FIRST_STEP。CPU/AOT/训练排队由并行launcher编排。
产物：/data0/xd/bam_diagnostics/rmt-readnorm-launch/xl-v60-*。

R600不能整分16路FSDP；打开bam_replicate_ploc_up，同时让embedding地址up沿用该选项。仅这些up权重不分片其R轴，保留精确R600，参数与数值公式不变。此共享分片选项改动额外运行完整BAM回归。

首步参数分片保护记录：模型1,432,396,680，每芯片109,403,880，理想89,524,792.5，额外22.2051%。28层加embedding共20.88M地址up复制，解释其中19.575M/chip的额外量；其余约.304M/chip为原有小参数复制。新类sharding_tolerance设.23，不改模型或数值。初次失败预队2910073已删除，正式2910074保留重启。

实际runtime e0125ba，FIRST_STEP1及九个MLP写入层health已确认，20–99步TB稳态中位数.336step/s（父版.347，−3.2%，同健康设置）。首次训练180步左右遇maintenance、未达首个250步checkpoint，auto-train同区重建；重启后仍需核对AOT loaded和首步。
