# MediumProp BAM V48/C12

RUN：BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile。
父版：BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile。
工作树/分支：/data0/xd/mediumprop-k75-embed / codex/mediumprop-k75-embed；runtime c1a68fe；TPU xd-v5p-16-2910070-maxtext，UC1a（可加排UE5a/EW4b）。编译借用EW4a FLEX_START llm-jax-v6e-1-0，不回收。

保留独立D1200向量残差，不从M取控制向量。完整M75×32→75×48，C8→C12，压缩率维持4:1。Attention P_loc、embedding写地址、独立MLP写地址R256→R384。QK57＋RoPE18、16头、18层、TruePile T4096、写入层1/4/7/10/13/16不变。

新增参数10732544＝7.45316 W_Q，其中各普通层472704、MLP写入层790400、共享embedding317696。按每层总参数近似均衡并分摊共享增量，精确MLP宽度[3765,3550,3765]，总432115072，MHA432121200，差−6128。无硬件取整。

下注：终局五点均值相对父版−.010；稳态.490step/s，相对父版.520约−5.8%。AllLocal无fetchedO，不需要跨token M-cache；前传M大小增50%，读写开销随之增加，普通attention KV-cache不变。

局部CPU完整预算/读写形状/健康、两block有限梯度；sealed-runtime检查和v5p-16 AOT；CPU/AOT/排队并行。13500步计划，200步loss窗口，1000步成批报告，2800/5000判定。
产物：/data0/xd/bam_diagnostics/rmt-readnorm-launch/medium-v48-*。

已确认AOT loaded、FIRST_STEP7、区内TruePile路径、六个写入层health。20–99 TB稳态中位数.503 step/s（父版.520，−3.3%，健康开关相同）；小于速度下注−5.8%。

首1000步：相对父版200/400/600/800/1000 gap −.165954/−.045164/−.031297/−.024075/−.016921。早期领先迅速收窄，未证明终局容量收益；维持终局−.010下注。1060步raw grad .527，writer health有限。

完成13500步，终局checkpoint13500；官方closeout已释放TPU及queue，TB增量同步成功。末五点12600/12800/13000/13200/13400，相对父版gap −.006433/−.007480/−.007259/−.008262/−.007015，均值−.007290，范围[−.008262,−.006433]。8k后优势维持约−.007，未出现明确晚期坍缩。

下注复盘：终局−.010高估了收益，实际−.007290；速度押.490（−5.8%），实测.503（−3.3%）。扩地址空间的净收益成立，但新增容量不能等比例转为loss收益；XL迁移仍需单独验证。结论台账不记下注与抢占过程。
