# MediumProp BAM：MLP 双残差分配

父版：`BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`（V32/C8，runtime23c692b）。不沿用 V48/C12；保留 D1200、18层、16×75头、QK57+RoPE18、TruePile T4096。六个 MLP→M 写入层为零起始1/4/7/10/13/16；写地址 R256，内容每头归一化。

工作树 `/data0/xd/mediumprop-k75-embed`，分支 `codex/mediumprop-k75-embed`，runtime616c7389cc69e6b3cbe82435bcc9857a4ad0965e。主 exp.py 仅台账。两个新类的完整前缀同父版，分别在 TruePile 前增加 ResidualComplement、ResidualOff。

令原始 MLP 输出按头拆为 y，现有写门为 g=sigmoid(logits+bias)。矩阵仍按原公式写 g·RMSNorm(y)⊗address；反向门控版仅将向量更新改为 (1−g)y；切断版仅在这六层去掉 y 的向量更新。其余十二层不变。先算原始 y 的矩阵写入，再处理向量分支；不改参数树或 MLP 宽度 [3901,3774,3901]，总432096128，与父版相等。新增 vector residual amplitude 健康统计，写门继续保留。

两路主要检验向量/矩阵流的角色，不指望删向量相加带来明显提速。

借用 EW4a FLEX_START llm-jax-v6e-1-0 编译，不拥有其回收。拟用 xd-v5p-16-2910076-maxtext（NoVO）交接 Complement；新抢占式 xd-v5p-16-2910079-maxtext 启动 Off（ContentTransform按用户指示继续观察），均 UC1a。13500总步数，200 loss窗口，1000步报告；2800/5000判定，只有出现稳定正收益或其他价值才继续。

CPU 检查：完整47项回归通过；局部完整参数/健康、逐头路由、相同输入下矩阵更新不变、非写入层不变、切断版经 M 的有限非零梯度。AOT与CPU并行；所有门槛通过后方可热交接。新 RUN 从零训练，不恢复旧 run 参数。

Complement 已在 UC1a 从 NoVO 的3303步提交checkpoint处完成训练机交接；新run从零启动，AOT loaded / FIRST_STEP8，初始 .518step/s（父版.520，−.4%）。NoVO按已完成热替换收尾，官方closeout已确认already_closed并跳过资源操作；TB SYNC_OK。Off另申请抢占式2910079，AOT与已通过CPU检查复用，UC1a队列03:38进入PROVISIONING。

启动编排修复：launch_train_parallel.py原只识别新编译AOT_READY，无法解析复用时的reused=true，导致最初2910079训练预队列被清理。脚本修复ca8e995，12项测试通过，已部署相同SHA2568b68b059。Off重排使用独立资源xd-v5p-16-2910079-retry-maxtext；原资源清理仍持续核对，避免复用旧名字时被旧删除任务误伤。

Complement20–99步TB八点中位速度.517，vs父版.520为−.6%。50/100步写门走势与父版近似；梯度量级亦相近（100步7.239 vs父版6.596），未发现新增异常。矩阵写入与向量保留比例的对应健康指标已验证。

03:43前后UC1a发生集中抢占，Complement首个租期结束；已回到UC恢复队列，UE5a/EW4b加排候选。ContentTransform按用户指示继续、未停。Off原预队列2910079已确认node与queue删除；retry资源在三区原队列保留，候选只能在QR ACTIVE、worker可SSH之后接管，首步通过才清理其他候选。两路代码及CPU/AOT门槛均已完成；Off尚未拿到FIRST_STEP。

## Complement 结论

停止于6452步。相对V32父版，早期落后收窄，4400–6400步约 +.0043，后期没有明确趋零；最后五个200步窗口均值 +.004237，范围 +.003691至+.004590。UC1a约.517 step/s，父版.520，−.6%，健康开关相同。反向门控向量更新未带来净收益。最终checkpoint已提交、TPU与队列已删除，TB同步完成。

## Off 结论

停止于7045步。相对V32父版，早期落后快速收窄，3200–7000步约+.0035，后期没有明确趋零；最后五个200步窗口均值+.003588，范围+.002772至+.003966。相对Complement最后五点（共同到6400）−.000717。UC1a约.516 step/s，父版.520，−.8%，健康开关相同。切断这六层的向量更新损失较小，但没有净收益。最终checkpoint已提交、TPU与队列已删除，TB同步完成。
