# BAM Pythia 2.8B：v5p-128 保持总 batch 的实测

模型／EXP：`BamPythia2p8BSotaIndependentR768Pallas`。
实现工作树 `/home/xd/projects/maxtext`，主分支 `refactor-bam`；worker 固定检出
`ce946eda1feefdba1b659f04b5a13d2ecb1c52b4`。它与旧 v5p-256 runtime `1ac1183`
之间只有台账文档差异，模型、优化器和训练计算没有改动。

## 结果

保持 global batch1024、T2048、143000步学习率计划，梯度累积1；训练健康统计ON，
BAM内部健康OFF，minimal remat。独立短测从头初始化，无checkpoint，完成log step0–49后停止。
下表统一取20–49共30步，以每步秒数平均后求倒数；日志单步速率保留三位小数。

| 拓扑 | 芯片／worker | 每设备batch | 秒／步 | step/s |
|---|---:|---:|---:|---:|
| v5p-256，原正式训练同窗口 | 128／32 | 8 | 1.68394 | .5938465 |
| v5p-128，本次短测 | 64／16 | 16 | 2.96122 | .3376990 |

v5p-128中位数.338，范围.337–.339，单步速率总体标准差.000586，约708206 tokens/s。
芯片减半后保留56.8664%吞吐，每步芯片时为原来的87.9254%，即省12.0746%；
同芯片单价下对应相同的计算费用比例。对照旧正式长训约.585step/s，则保留57.7263%
吞吐、芯片时省13.3844%。短测没有覆盖周期checkpoint，不能将它当作全天端到端吞吐。

跑前下注.32step/s，范围.29–.35；实测落在范围内，比中心预测快5.53%。
worker0的原始XPlane包含8个TPU core plane、各5个完整train-step span，平均2.90870秒，
最小2.90667、最大2.91276秒；与包含主机侧工作计时的2.96122秒一致。
这里的XPlane范围是worker0的全部core，未声称捕获全部16个host。

## 启动和停止验证

- 日志确认 `Loaded compiled function!`，且实际跑出step0。
- worker HEAD为上面的完整commit；解析后global batch1024、per-device16、累积1、steps和LR schedule均143000。
- 精确50步停止控制在打印step49后终止此任务trainer，完整日志只有0–49；16个worker均确认TASK_TRAINER_GONE。
- worker直接上传profile到GCS，再拉回本机，核验XPlane166513824字节、trace JSON28491126字节。
- TPU与同名queued resource已核验不存在；原正式RUN仍暂停在CP37042，本次没有读写其checkpoint或续训。

任务TPU：`xd-v5p-128-r768speed-20261021-ue5a`，`us-east5-a`，零抢占。
READY首次观测UTC05:18:40；FIRST_STEP05:22:54；矩阵完成05:25:52；节点和队列删除核验05:29:16
（均2026-10-21；最后时间为核验时间）。无备用区或其他TPU参与。

## 复现

用户提供且已验证的AOT：

`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/ce946ed/jax081-i0ae3f58-c17f538a/v5p-128/s143000/BamPythia2p8BSotaIndependentR768Pallas.pickle`

manifest同路径追加`.manifest.json`；size407702216、CRC32C `ApSj7g==`。
安装器hash `0ae3f58971e6b06d9eac2aa752115ea43f66b9869553698dea20ac7d501d1186`。
本次使用的矩阵runner hash `a27edeb0ae50b4fbb920713743374d32a81a860614ad0f7fcd767423f5452716`，
compiled smoke helper hash `cf6b2e9fda587a9a310f4693d229c68720fb7ec66e0fce807cda20a7c8de8912`。

runner为tpu-ag `/home/lishengping/xd/projects/run_profile_matrix.sh`，参数为TPU、UE5a、上述完整commit、
label `r768-v5p128` 和完整EXP。设置 `PROFILE_STEPS=143000 PROFILE_DONE_STEP=49`
`PROFILE_SKIP=10 PROFILE_DURATION=5 PROFILE_PERIOD=-1 PROFILE_TRACE_COUNT=1`，
`PROFILE_MATRIX_ID=r768-v5p128-speed-20261021`，AOT_ROOT为上面pickle所在目录。
实际RUN：`Profilece946ed_r768-v5p128_r768-v5p128-speed-20261021_0_BamPythia2p8BSotaIndependentR768Pallas`。

GCS证据根：
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/ce946ed/r768-v5p128/`

本机证据根 `/data0/xd/bam_diagnostics/pythia2p8b-r768-v5p128-speed-20261021/`：
`train.log`、`speed-summary.json`、`v5p256-matched20-49.log`、`profile/`、
`workers-stopped.txt`、`create.log`、`release.log` 和任务控制脚本。

本次发现的旁路问题是BucketLogger将profile本地输出路径当成GCS，备份线程退出；trainer正常运行。
测速结束后已在compiled-smoke helper修复：通过矩阵已有的PROFILE_BASE_OUTPUT将日志备份指向GCS，
profiler TensorBoard文件继续留在worker本地。主机脚本修复commit `4fb0e24`，已用mock argv验证
有无GCS override两种路由并部署；此修复未用于上述测量，不改变ce946eda模型或AOT。
