# XLProp HD64 TruePile 对照

任务：保持28层和20头，将 XLProp 的 head_dim/model_dim/MLP 等比缩小到64/1280/3413；TruePile T4096，24000步，LR2.5e-4，warmup240步，v5p-32，global batch128。

| RUN | 工作树 / 分支 | TPU ID | 参数量 | runtime |
|---|---|---|---:|---|
| Llama2XLPropHD64 | /data0/xd/mudd-xlprop-truepile / codex/mudd-xlprop-truepile | 2910067 | 679645440 | 5a330c5 |
| MuddLlama2XLPropHD64 | 同上 | 2910068 | 683751601 | 1bdd511 |
| BamXLPropHD64K64EmbedVOnlyQK48AllLocalMLPWriteIndependentEveryThirdTruePile | /data0/xd/mediumprop-k75-embed / codex/mediumprop-k75-embed | 2910069 | 679644600 | 21e5ef7 |

主 exp.py 只登记实验；Mudd 完整历史和精确逐层宽度、BAM 稀疏独立 MLP 写入的实现分别留在上述工作树。全部先排UC1a，资源紧张时可同时排UE5a/EW4b，确认 FIRST_STEP 后释放其他候选。借用EW4a用户 FLEX_START llm-jax-v6e-1-0编译；不得回收。

Mudd：原28个MLP宽度各乘2/3、取最近整数，不重新按128取整。总参数比MHA多4106161（2.506 W_Q，0.604%），保持原Mudd的参数口径。

BAM：M64×40/C10；LocalQK48与标准RoPE16拼接；地址输出维不变，因此P_loc/embedding地址R400、MLP独立地址R384不变。全L，无W_V；MLP写入零起始1/4/…/25层。MLP宽度[4030,3818,4033]×9＋末层4030，总参数比MHA少840。

三路均完成24000步。终局五点均值：Mudd−MHA −.080172；BAM−MHA −.111092，BAM−Mudd −.030920，收益倍数1.386×。BAM优势早期收窄，18k–23.5k相对Mudd保持约−.031；收益倍数由中期约1.43×缓慢降至约1.38×，后期仍保留明确优势。

验证与监控：局部CPU参数/前传梯度和受影响路径回归、sealed-runtime检查、正式拓扑AOT、FIRST_STEP；三项准备并行，CPU失败阻止启动并释放其训练候选。XL loss每500步窗口，约2000步成批报告；内部约10分钟检查资源与健康。产物：/data0/xd/bam_diagnostics/rmt-readnorm-launch/hd64-*。

上线：三路UC1a均已确认AOT loaded和FIRST_STEP。稳态MHA/Mudd/BAM .711/.620/.446 step/s，Mudd−MHA−12.8%，BAM−MHA−37.3%、−Mudd−28.1%。MHA/Mudd原生attention，BAM C256且额外concat/write health开启，速度口径需保留。相对原XLProp BAM/MHA约−36.1%，HD64相对损失接近；绝对速度下注均偏高。

BAM收尾：最终checkpoint24000提交，UC1a/UE5a/EW4b的同名TPU与队列均确认不存在，`closeout_runs_local.py`返回`already_closed=true`，本地TB `SYNC_OK`。14次抢占、15段READY租期和UC1a→EW4b→UC1a的正式区域切换已写入区域历史；[候选队列记录](/data0/xd/bam_diagnostics/rmt-readnorm-launch/hd64-bam-passive-candidates-final.json)单独保存，不作为区域切换。
