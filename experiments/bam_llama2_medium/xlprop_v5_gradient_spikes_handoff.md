# XLProp V5 极端梯度尖峰：诊断交接

日期：2026-11-08。目标是定位**反向梯度第一次异常放大的层与算子**，并区分真实的网络 Jacobian 放大与核的数值／实现问题。本文件供并行排查使用；以下将实测事实与推断分开。

## 1. 现象与正式训练

- RUN：`BamXLPropV5`，当前 EXP：`BamXLPropV5FastKernel`。
- 正式训练于2026-11-08按用户要求停止止损；TPU `xd-v5p-32-311001-maxtext`（`us-central1-a`）已释放，node和queued resource均确认不存在。最终checkpoint **13007**；TB已同步。不得自行恢复训练。checkpoint及独立诊断副本保留。
- 当前 runtime：`7c90924b87ec446ab81ac57eadcda42219230e2a`。
- 初始 runtime：`862caddd9be22c2d4d87fadc5c0dabecb270546f`。
- T4096、global batch128、8/device；计划50000步；BF16运算、FP32参数；原 Prop 配方。
- D1920、28层、20头×96维；M96×40；地址低秩R400；独立VO；完整NoPE96＋RoPE24，QK实际120维，打分缩放仍为`1/sqrt(96)`。
- 概念M尺寸是96×40；正式carry的实际布局是`[B,T,V,K]=[B,T,40,96]`，加性probe应匹配这个布局。
- M pre-norm；V/O共用逐层scale、Q/K共用逐层scale，都从1初始化；**没有QKNorm**。
- 串行SwiGLU、RMSNorm epsilon1e-6、原`.006`初始化。普通/小MLP4048/1827、末层6277；末层两项M写关闭。attention写内容保留RMSNorm，小MLP原幅度写，embedding固定除以`.00203054060713255`。
- AdamW：LR2e-4，beta=.9/.95，epsilon1e-8，WD=.1，warmup1%，cosine下限.1；global clipping阈值1。

实测TensorBoard样本（每10步记录，因此不是逐步峰值上界）：

| step | raw grad norm | clipped grad norm | 说明 |
|---:|---:|---:|---|
| 4100 | .332 | .332 | 旧核正常段 |
| 4140 | 353.786 | 约1 | 热切前已有尖峰 |
| 7880 | .237 | .237 | 新核正常段 |
| 7910 | 3379.258 | 约1 | 7.9k尖峰 |
| 8970 | 6.1438e15 | 约1 | 后续严重不稳定 |
| 9340 | 2.5326e18 | 约1 | 已记录最大raw norm |
| 10000 | .1871 | .1871 | 已恢复低梯度，但loss仍有损失 |

裁剪值是实际记录的`learning/grad_norm`，不是人为`min(raw,1)`。约9k窗口对旧XL单门.5的loss gap达`+4.00643`；10k恢复至`+.0043232`。10k以MHA为共同收益参照，对旧XL单门.5的收益倍数`.97144`，对Mudd为`1.60709`。

用户反馈2.8B的对应架构未出现同类尖峰。**不能仅据此归因QKNorm**：2.8B还有Pythia初始化/LayerNorm/GELU等差异，且32头自然对齐；XL的20头会进入新增的whole-packed-keys路径。

## 2. 热切边界与核差异

从原RUN的checkpoint4750及其原配套数据游标恢复，保留了旧核在checkpoint之后的训练尾部。同批4751–4770的loss平均绝对差`3.24e-5`、最大`1.05e-4`；四个worker的source/AOT/restore一致。

FastKernel相对旧运行仅改变：

```python
bam_pallas_packed_read_keys = True
bam_splash_residual_name = 'bam_splash_residual'
remat_policy = 'full_save_splash'  # 旧：full
```

另外`bam_pallas_v8.py`新增`_WholeKeys`，20头的Q/K/V/O键按完整`[C,80,T]`读取，避免不合法的20行per-arm BlockSpec；32头不走这个特殊路径。

正式AOT：
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/7c90924/jax081-i0ae3f58-c17f538a/v5p-32/s50000/BamXLPropV5FastKernel.pickle`

环境：JAX0.8.1；installer `0ae3f58971e6b06d9eac2aa752115ea43f66b9869553698dea20ac7d501d1186`，compiler `17f538a5cd26b264603e6ff3c3683ae9243e234eeaf4571aaf4bf0819180e442`。如使用manifest，请以AOT旁`.manifest.json`中的完整hash为准。

**旧核有尖峰，不足以排除新路径还有另一类问题；20步对齐也不是全训练数值正确性的证明。** 4.14k/7.91k与9.34k的层间梯度形状明显不同，应分开看。

## 3. 当前层间定位：实测的是参数梯度

9340步的参数raw gradient L2 norm：

| 参数路径后缀 | L0 | L1 | L2 |
|---|---:|---:|---:|
| `self_attention/query/kernel` | 8.891e17 | 5.839e16 | .1308 |
| `self_attention/key/kernel` | 1.769e17 | 8.859e16 | .1799 |
| `self_attention/W_lv_c8_up/kernel` | 2.855e17 | 6.159e16 | .3264 |
| `self_attention/W_R_up/kernel`（localO） | 5.441e16 | 1.383 | .5293 |
| `self_attention/out/kernel`（W_O） | 4.044e17 | .9773 | .2940 |

L1的attention写门／地址也远小于其Q/K/V相关参数；L2以后没有同量级爆炸。8970、9100也呈现类似的L1分叉。相对健康8800，9340的L1 Q/K/V相关参数放大约18–20个数量级，而同层localO/W_O仅约1–2个数量级。

**这将可疑区间收窄到L1的context-attention及其上游读端反传，但尚未直接测得Q/K/V张量cotangent。** 不应把`W_lv_c8_up`参数梯度称为`dV`，也不能因embedding地址bias最大就把它认定为源头。若真实attention张量cotangent正常，异常还可能来自Pallas读核reverse，而不是Splash。

真实softmax反向应满足，令`O=P@V`、`G=dL/dO`：

```text
dV = P.T @ G
dP = G @ V.T
dZ = P * (dP - rowsum(P * dP))
dQ = dZ @ K / sqrt(d)
dK = dZ.T @ Q / sqrt(d)
```

对正确概率矩阵，`||dV||F <= sqrt(T) * ||G||F`；T4096时为64倍。该界作用于**张量梯度**，不直接作用于地址投影参数梯度。大logits本身不能解释1e18；需要测上游G、Q/K/V幅度及算子VJP。

## 4. 可用checkpoint与数据

尖峰期间checkpoint9250已从GCS软删除保留区取回，并复制到独立诊断前缀：

```text
gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-gradient-localization-20261108/checkpoints/9250/
```

- `items/`可正常部分restore参数，参数数目`1,432,396,996`。
- 30个文件、`15,989,895,218`字节；诊断副本与原软删除版本的逐对象size/CRC32C已核对一致。
- 含原TrainState、commit marker和`skip_file_and_step.json`。
- 正式RUN的最新续训指针没有回退。原9250前缀也被恢复为live，属于本次恢复产生的额外副本；诊断完成后应处理这份重复数据。

9250游标：seed9876、global128、4个原infeed host；cur_files为UC1数据副本的`pile.tfrecord.b9000`与`pile.tfrecord.b9500`，`file_in_data=9`、`step_in_file=251`。

按原process-independent global stream和skip count导出的**下一global batch（9251）**：

```text
gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-gradient-localization-20261108/results/cohort.npz
```

大小1,571,357字节，包含128条T4096的inputs/targets/positions/segmentation。探针以seed0打乱序列索引，先取32条逐条运行，记录原索引及SHA256。

**当前小batch探针不等于精确重放9340。** 如需精确重放，应从完整9250 TrainState＋游标，按正式global128与原拓扑执行9251–9340；只读单序列探针用于定位算子及数值机制。

## 5. 已做测试与当前探针状态

已验证：

1. CPU小尺寸FP32 strict-causal＋sink参考，与独立dense softmax的前向及全部Q/K/V梯度一致；首行输出为零。
2. 预留FLEX上，实际TPU、T4096、N20、QK120/V96：仅首token空历史输出有上游梯度时，下面四路的输出及dQ/dK/dV均严格为0：
   - fused backward＋full_save_splash；
   - fused backward＋full；
   - 非fused backward＋full_save_splash；
   - fused backward＋full_save_splash＋单位scale完整头QKNorm。

因此，**该合成测试没有复现empty-row backward异常**。不能外推成真实checkpoint所有张量均正确。

完整模型首版探针：参数restore和cohort导出成功；host callback在TPU编译/执行时报`Unknown send channel 28 for host callbacks`，未产生可信的完整模型cotangent。失败worker进程217717已清理；这不是模型训练失败。

当前runner采用**零值加性probe求导＋Flax intermediates**，保留原scan和参数，不使用host callback。9250 checkpoint的32条单序列探针已完成：

- 平均loss `10.5562334`；embedding写地址bias梯度L2最大 `17.2684`，未复现灾难性放大。
- 捕获L1实际Q/K/V及context cotangent，同张量Splash／FP32参考反向对拍未出现数量级偏差；各case及Q/K/V中最大相对RMS误差 `.09403`。模型内梯度相对独立Splash最大相对RMS误差 `.19119`；部分参考梯度很小，完整逐case绝对误差见结果文件。
- 最高embedding梯度的sequence66中，M输入cotangent RMS为L0 `8.30e-6`、L1 `1.17e-6`、L2 `1.03e-6`；没有9340所见的L1灾难性分叉。

**此结果不能排除核或网络问题**：探针使用B1，正式每核B8；权重冻结于9250，尚未重放9251–9340的优化器更新。后续如继续排查，应优先匹配B8／正式拓扑或重放原TrainState，不能将本次未复现写成根因已排除。

Runner：[`xlprop_gradient_localization.py`](xlprop_gradient_localization.py)。`--empty-row-check-only`是已通过的合成不变量入口。

## 6. 运行方法与资源协调

模型源码必须使用immutable runtime `7c90924b87ec446ab81ac57eadcda42219230e2a`。探针脚本可用本仓库最新版本；它会校验runtime并记录runner SHA256。没有模型参数更新或checkpoint保存。

预留机：`llm-jax-v6e-1-0`、`europe-west4-a`、worker0；用户允许使用的FLEX，**不拥有其生命周期，勿删除**。本机tpu-ag共用编译互斥锁：

```text
/home/lishengping/xd/projects/aot_runs/worker-europe-west4-a-llm-jax-v6e-1-0-0.lock
```

并行排查前先协调该锁及worker进程，避免与本探针/AOT竞争。正式训练TPU保持独立。tpu-ag只用于编排；大产物由worker直接上传GCS。

worker已有runtime快照`/tmp/maxtext-offline-aot-7c90924b87ec`。将runner复制到`/tmp/xlprop_gradient_localization.py`后：

```bash
env JAX_PLATFORMS=tpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 \
  /home/lishengping/miniconda3/bin/python /tmp/xlprop_gradient_localization.py \
  --runtime-dir /tmp/maxtext-offline-aot-7c90924b87ec \
  --runtime-commit 7c90924b87ec446ab81ac57eadcda42219230e2a \
  --checkpoint-root gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-gradient-localization-20261108/checkpoints \
  --step 9250 --max-samples 32 \
  --output /tmp/xlprop-v5-gradient-localization
```

追加`--empty-row-check-only`并使用另一个output目录，可复跑已通过的不变量测试。单v6e诊断使用JIT；不能加载v5p-32正式AOT。

## 7. Target-topology replay: FSS reverse numerical failure confirmed

Claude replayed the full9250 TrainState and exact data cursor onv5p-32 with the
formal AOTs. This task independently downloaded both TB artifacts and checked
9251--9350 FSS loss against the original console cache: all100 points match
within4.94e-7, the console's six-decimal rounding. Full currently has75 recorded
points through9325 in the downloaded artifact.

At9251 the two arms have the same pre-update state/batch and exactly equal
recorded loss10.654891014099121. Parameter-gradient L2 norms:

| Metric | FSS7c90924 | Fulld54c4c5 |
|---|---:|---:|
| Global raw |10903.9238|19.06523|
| L1 query/kernel |14.43092|.176303|
| L1 key/kernel |21.29503|.065320|
| L1 W_lv_c8_up/kernel |201.97667|.220419|
| L1 W_R_up/kernel |.369236|.369555|
| L1 out/kernel |.319405|.319146|

Changing only remat/residual retention removes the isolated Q/K/V-parameter
amplification, while localO/W_O remain within.1%. At9275 full has loss2.361958
and raw.389161; FSS has loss3.844815 and raw32.5158. Full's75-step maximum raw
is32.7276; FSS includes norms above1e18 and2 nonfinite norm samples (the latter
may be FP32 norm-square overflow, not necessarily nonfinite gradient leaves).

**Conclusion:** B-type instability is a reverse numerical problem introduced by
`full_save_splash`, not evidence of the V5 architecture's training ceiling.
Both arms retain packed keys. The proposed mechanism is saved Splash output/lse
combined with differently rematerialized Q/K. Code inspection confirms backward
uses `p=exp(qk-logsumexp)` directly. The downloaded TB contains parameter norms,
not actual Q/K/V/G cotangents: it does not itself measure the activation-norm
inequality or directly capture an lse mismatch. Keep that narrower distinction
when explaining the exact mechanism. A-type spikes predating FSS remain outside
this fix's demonstrated scope.

Evidence:

- GCS:`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-replay-20261108/{tb-fss,tb-full}/`.
- Local:`/data0/xd/bam_diagnostics/xlprop-v5-config-20261108/claude-replay/`;
  `selected-metrics.json`, `summary.json`, `analyze.py`, original console loss
  subset, TB files and plot provenance retained.

The duplicate standalone request`xd-v5p-32-xlprop-spike-replay-1108` inUC1a was
cancelled; its node and queued resource are verified absent. Claude owns the
paired replay resource and its cleanup.

The user authorized restarting from8750 with
`BamXLPropV5FastKernelFullRemat`, runtime
`d54c4c5f1339b01e406932ec28be0e398722b918`. Original RUN remains stopped at13007.
Separate RUN named for the new EXP is submitted for50000 steps, TPU
`xd-v5p-32-311002-maxtext`, UC1a primary and UE5a/EW4b backups. Recovered8750 was
copied into the new RUN and verified by size/CRC across30 files
(15,676,906,451bytes); original global128/4-host cursor restored. Verification:
tpu-ag`logs/xlprop-v5-restore-8750/copy-verification.json`.
Expected throughput .52step/s (about-3.7% vs priorFastKernel .54).
Startup verified: all4 worker source hashes matchd54c4c5, all loaded the AOT
and restored8750 with global128/per-device8. FIRST_STEP8754 at.523step/s;
subsequent8782 at.521. Same-batch8751--8770 mean absolute loss difference
2.92e-5, maximum7.4e-5; first8751 loss exactly2.343012 in both console histories.
The formal RUN is now genuinely training. Group`xlprop-v5-full-remat`, user-updated
5000-step cadence phased at10000 (offset0):10000 delivered, next15000. Scoped
watcherPID137311 on tpu-ag; mechanical health polling remains unchanged.
Startup acceptance also crosses the original B onset: at9079 throughput.520;
25 recorded raw norms over8751--9000 have maximum.291019, no clipping. This
is a recorded-sample statement, not an upper bound on unlogged steps. Evidence:
local`claude-replay/full-resume-through9000-health.json`.
The follow-up through9350 has60 samples, maxraw.381264; at9340 raw.197039
and loss2.206448, with normal L1 projection norms. New9250 checkpoint/cursor
committed. Source:`claude-replay/full-resume-through9350-health.json`.
For cumulative reports, the original actual loss history through8750 is imported
into the child cache as its common prefix; full remat applies only from8751.
Provenance:tpu-ag`logs/xlprop-v5-restore-8750/shared-prefix-provenance.json`.
The corrupted original post8750 branch is not imported.

## 8. 优先判别实验

1. 直接测L1的context输出cotangent G及dQ/dK/dV，检查dV的64倍界，并记录前向Q/K/V范围、logits范围；不要从参数梯度替代推断。
2. 同一捕获Q/K/V/G，对比FP32参考、fused/non-fused Splash。再对比模型中观察到的梯度与独立算子的梯度，以区分局部算法问题与整图remat/残差保存/布局问题。
3. 若attention VJP正常，继续比较**20头whole-packed read reverse**与原unpacked/native读端。特别检查标准Q/K passthrough、Q/K/V/O各支cotangent及M passthrough；32头的正常结果不能覆盖20头特殊分支。
4. 单位scale完整头QKNorm干预可检验幅度/条件数嫌疑，但会改变前向与Jacobian；它能消除尖峰，也不自动证明原核正确或保证重训收益。
5. 同时看每层dL/dM与向量残差cotangent，定位第一处跃迁发生在read、context attention还是写回，区分跨层反馈累计与单算子异常。

希望反馈：**第几层、哪个算子的输入/输出cotangent发生多少倍跃迁；同张量参考误差；最小修复或最有判别力的下一步**。无需先写大规模健康指标或启动新训练。

## 9. 已有数据和日志

本机：

```text
/data0/xd/bam_diagnostics/xlprop-v5-config-20261108/gradient-localization/
  tb-selected-steps.json             # 多个健康/尖峰step的逐叶参数梯度、参数范数及scale
  tags.txt
  ckpt-9250-soft-deleted.json
  checkpoint-copy-verification.json

/data0/xd/bam_diagnostics/xlprop-v5-config-20261108/fast-kernel-switch/
  old-final-steps.json
  new-workers-proof.json
  overlap-comparison.json

/data0/xd/tensorboard_logs/BamXLPropV5/
```

GCS诊断根：

```text
gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-gradient-localization-20261108/
  checkpoints/9250/
  results/cohort.npz
  results/probe.log                 # 最近一次探针日志；请检查是否真正完成对拍
  empty-row/empty-row-check.json
  empty-row/probe.log
```

分析极端梯度时用FP64或缩放后的范数计算，避免将FP32平方溢出误认为原始张量非有限。
