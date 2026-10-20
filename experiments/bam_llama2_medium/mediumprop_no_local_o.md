# MediumProp LocalO消融

RUN：`BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdNoLocalOTruePile`。父基线：`BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`（23c692b，.520step/s）。
实现worktree `/data0/xd/mediumprop-qk75-sparse`，分支 `codex/mediumprop-qk75-sparse`，runtime `e0198538524592af50b9b943908560ec59b2881e`。artifact root `/data0/xd/bam_diagnostics/mediumprop-no-local-o`。

仅删除LocalO静态32×16投影、独立动态门控与注入；LocalV保留原非零静态投影、共享C8动态读法、.05门和.2scale。attention输出仍过W_O并写M，但两处内容均不再包含LocalO。保留每第三层独立动态MLP→M。由于原代码的local_o标签同时选择V读法，独立开关不更改layer mode，避免误改LocalV。

每层删除1200×16+16+32×16=19728参数；18层355104=.2466W_Q。MLP每层补6维，宽度[3907,3780,3907]，总参数432129824（vs MHA +8624，vs父版+33696）；这是统一三种层型均匀回补下最接近MHA的整数宽度。

测试：完整参数树核算，只有O参数被删除，健康指标保留V/MLP写入；未删参数初始值逐叶与父版完全相等（删除参数的位置保留初始化RNG消耗）；三层block-scan前传和全梯度与父版将O门/静态O置零的参考实现相符。两组focused CPU并行42.83s通过，sealed config通过。未重复无关通用测试。

目的：量化LocalO的价值，不要求获胜。13500总步数，至少5000才进行效果判定；2800不会因正gap判停。之后根据差距稳定性决定是否续至终局。约1000步报告，200步loss窗口；唯一直接基线为原标准版。下注终局+.010loss、稳态速度+.5%。

两个预留FLEX编译机均SUSPENDED，使用本任务管理的spot v6e AOT队列，EW4a优先、UC1a/UE5a备用。正式训练优先UE5a，资源久等时保留原队列并加EW4b/UC1a。新实验从step0开始，TruePile4096数据按启动区域自动匹配。

正式训练 `xd-v5p-16-2910127-maxtext` / UE5a，FIRST_STEP0，worker源码和AOT加载均核验e019853。注册数据路径为UE5本区 `gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096`。AOT所有自有编译候选已释放（cleanup_failures=[]）。

启动已到114步，初始V门.050331、末层静态V RMS1.952、动态V0、raw grad norm117.919均符合保留初始化。20–99日志步速均值.524725（vs标准.520 +.91%），中位数.533（+2.5%）。25步首次TB flush附近出现一次约140秒停顿（worker0/1分别计入28/27步，.007step/s），随后恢复，未重启或回滚；checkpoint线程未报错。I/O归因尚未做独立测量，不作为模型计算变慢。原始证据`startup-worker.json`，健康`startup-health.json`。比+.5%速度下注更快，但正常步提速包含删去O分支及其健康统计的代价；不把它全归给GEMM FLOPs减少。

日志修复：原运行到909步，TensorBoardX GCS503使event线程退出、主线程阻塞日志队列；800checkpoint已提交。缓存原0–909loss后停止controller，TERM无效后仅KILL本RUN两个挂起进程，保留UE5a训练机。host-only runtime `0e9ec8b9ce6b62d719fd2b09a98f7cec6e7a68c6` 给GCS事件写入增加有界重试/本地spool，避免后台线程异常退出；两个503故障注入测试通过（含队列持续排空及恢复后完整前缀补传）。模型、优化器、数据配置不变；新hash AOT准备中，待从800恢复并验证重放loss连续性。

恢复完成：spot编译三区等不到资源，改用本任务保留的v5p-16 worker0 host做离线CPU AOT（isolated artifact），三个编译候选清理成功，cleanup_failures=[]。从800恢复；新runtime0e9ec8b，AOT加载成功。801–909共109个重放loss与旧日志完全一致到六位小数，917 FIRST_STEP gate通过，约.532step/s。证据replay-check.json、resume-worker.log、aot-repair-idle.log。正式继续至少5000步。

5000判定：最近五点相对标准版+.024962（+.024229..+.026192），2000/3000/4000/5000单点+.043370/+.031331/+.025654/+.024229；仍缓慢收窄，继续观察。常数+1/t及常数+1/t²，拟合起点2000/2600/3000、终点5000，13500外推+.01591..+.02165；模型敏感性范围非置信区间，原+.010下注偏低。仅能说明删LocalO净代价，不能据此排除共享读键存在冲突。梯度近期正常，无新抢占。

终局：完成13500，末五窗口12600–13400相对标准版+.017949（+.017484..+.018449）。早期差距+.594608@200迅速缩小，5k后继续缓慢收窄，约10k后处于+.018左右。正常步速约.532（vs.520 +2.3%）；初始化20–99均值.524725含一次I/O停顿，+0.91%。O被删的健康统计自然也不再计算，速度不是纯理论FLOPs对比。

认识：LocalO并非只有早期优化作用，等参数补MLP后仍保留约.018终局代价。删除同时移除了M→向量残差和LocalO参与attention写回M两条路径，不能单凭本消融区分其贡献，也不能排除共享读键冲突。若继续做因果拆解，优先保留LocalO进入W_O、仅取消其参与M写回；不将本次结果解释成所有O主导都好或所有共享冲突都不存在。原+.010下注低估约80%，5000时外推+.016..+.022覆盖终局。

收尾：本地closeout_runs_local.py确认already_closed=true、不重复操作资源；TB SYNC_OK。UE5a v5p-16，无抢占/跨区，READY 2026-10-08 15:57:22→2026-10-09 00:02:05 UTC，共8h04m43s（包含同机日志修复停顿）。13500checkpoint已提交。终局report文件`/data0/xd/bam_diagnostics/prop-full-qk-monitor/reports-20261009T000126.json`。
