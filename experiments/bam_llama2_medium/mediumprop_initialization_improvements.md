# MediumProp 原标准版初始化改进

实现：`/data0/xd/mediumprop-qk75-sparse`，分支 `codex/mediumprop-qk75-sparse`，runtime `95443aff59c2d3ee21166eb11f0098043ca72023`。
直接基线：`BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`（历史 runtime `23c692b`）。

两路仅改初始化；完整 M75×32/C8、18层、MLP [3901,3774,3901]、432096128 参数、原 VO .05 门控和 .2 scale、concat/write 健康指标均保留。

| RUN | 单一改动 | 终局下注：RUN−原标准 | 速度下注 |
|---|---|---:|---:|
| BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadKeySmallInitTruePile | W_R 高斯 std=.001/√1200；pre-RMS 读键 RMS 约 .001 | −.006 | 原 .520 step/s ±1% |
| BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdStaticVOrthogonalInitTruePile | 静态 V 32×16 投影列正交、单位范数；其余保持原版 | −.003 | 原 .520 step/s ±1% |

前者保留原非零静态 V 启动，检验小幅非零动态读初始化；后者保持单头初始方差，去除静态读头间初始相关。不存在“共同能量占比/末层保留率越高越好”的目标，最终以 loss 和训练健康判断。

Focused CPU：原版/两路完整参数量核对；同 seed 参数逐叶对照，只有指定 W_R/static_v_key 变化；实际原始读键 RMS、静态 V Gram 检查；两路 scanned forward/backward finite。sealed config 通过。CPU 日志：`/data0/xd/bam_diagnostics/mediumprop-init-improvements/cpu`。

AOT 借用用户预留 FLEX_START `llm-jax-v6e-1-0`（EW4a），不持有其生命周期。训练热替换 UE5a `xd-v5p-16-2910121-maxtext` / `xd-v5p-16-2910122-maxtext`，13500步；新 RUN 从0步启动，不继承旧实验参数。

收尾及启动证据追加在本文件。artifact root：`/data0/xd/bam_diagnostics/mediumprop-init-improvements`。

交接脚本修复：`hot_switch_run.py --allow-uncheckpointed-tail` 仅用于新实验替换；确认所有旧 worker 已退出后，保留最新已提交 checkpoint，另记未保存的已训练尾段。默认续训安全要求不变。SIGTERM 后可能留有少量未保存尾段；明确新实验替换场景允许保留最近已提交状态，而不虚称尾段已保存。B 旧 worker 此次因 GCS 503 上传异常未退出，核验2000步 committed后清理精确匹配的旧进程。13项脚本单测通过；编排源码和部署 SHA256 核对一致。

## 启动与旧实验收尾

两份 AOT 均在预留编译机完成，未触碰编译机生命周期。新两路 FIRST_STEP 分别4 / 1，worker源码95443af、AOT加载和UE5a TruePile路径核对通过，fresh step0，健康统计开启。

旧B：checkpoint2000，loss到2025；1200–2000五点相对原静态V零版 +.001850、相对标准 +.096176。旧C：checkpoint2012，loss到2010；1000–1800五点相对B −.034966、标准 +.065719。C收益在收窄、对标准落后平台明显；两路均不支持替代标准。完成收尾后删除台账下注，保留诊断文档里的预期及复盘。

下注修正：原B终局vs静态V零 −.080 的预测显著过度乐观，恢复动态幅度没有修复整体架构启动；少量静态V对B的早期显著改善也不能推出它优于原标准。此处是终止前结果，并非完整终局。两路均0抢占、同区热替换；完整lease见区域历史。

稳态20–99（80点）小读键 .519788step/s，vs原标准.520 −.04%；静态正交 .517738，vs标准 −.44%；median .521 / .520。均为相同concat/write health，未增加训练算子。初始V/O门均约.050331（bf16）；raw grad norm分别118.08/130.84，健康统计finite。两路旧TB均由`closeout_runs_local.py`同步，确认SYNC_OK且already_closed=true，跳过新run拥有的TPU。

旧实验后续健康（末层，1000→1800/2000）：原标准V门 .099→.129、O门 .373→.323；B .087→.086 / .0445→.0448；C .050→.0466 / .040→.0406。B/C gate 并未恢复原版学习路径，且原版 attention/V RMS 保留率维持约.25，低于B约.47/C约.53但loss更好。后续改进不能把恢复高保留率或压低LocalO占比当目标，应同时核对门控学习、loss及原始读幅。完整健康原始序列：`/data0/xd/bam_diagnostics/mediumprop-init-improvements/old-terminal-health.json`。

静态正交的更精确解释：同一个float32 RNG key、32×16形状下，标准高斯矩阵A与正交Q共享列空间（A=QR）；改变的是head坐标的条件数，而不是换一个读取子空间。seed3样例列空间相对残差2.38e-7，AᵀA特征值.0933–2.5796，QᵀQ约为I。因此此路检验的是初始head混合/尺度条件改善能否帮助学习。量化后的微小偏差不改变这个设计意图。

1000步：小读键相对标准gap200/400/600/800/1000 = +.054106,+.018277,+.004496,+.003417,+.003981；正交 = −.238000,−.040515,−.014032,−.004579,−.000993。初期优势快速消退，两路均继续2800判定。当前数据削弱终局−.006/−.003下注。正交末层retention约.60、原版约.25，但loss趋于一致；不以retention大小选优。

2000步：小读键1200/1400/1600/1800/2000相对标准 +.004546,+.004485,+.000746,+.002716,−.000579（last5+.002383）；正交 +.001573,+.003410,+.001385,+.003848,+.002797（last5+.002603），1200起落后，均继续2800。原始loss/健康分别在report packet133522、小读键loss-2000.json、health-2000.json/health-small-2000.json。小读键健康完整窗口到1800。

理论修正：rms(r)=r/√(mean(r²)+ε)，Jacobian横向1/√(mean(r²)+ε)，沿r方向ε/(mean(r²)+ε)^(3/2)；r=0时为I/√ε，键梯度非零且放大系数最大。门控第一步梯度为零只是短暂现象，不能作为终局收益依据。正交改善初始head坐标条件可加快启动，但没有改变读取子空间/模型表达力，当前没有持久收益。

2800步：小读键last5+.001304/latest+.000867；正交last5+.002042/latest+.002145。仍未盈利，落后有收窄，继续5000。静态V参数只恢复三份小叶：1000/2000/13500平均列范数.934/.889/.701，WD-only因子.973/.944/.800（理论初始期望范数约.992）。不能将终局范数移植为最佳初始化，但为适度减弱静态V提供下一步线索。可复现只读CPU脚本`/data0/xd/bam_diagnostics/mediumprop-init-improvements/analyze_standard_static_v.py`，缓存NPZ和geometry.json避免重复读GCS；未申请TPU诊断机。

## 静态正交版收尾（2026-10-08）

用户要求停止，已用 `scripts/closeout_runs_local.py` 收尾：stopped3771，checkpoint3771 committed，0抢占，训练TPU及queue确认不存在。完整终止报告：`/data0/xd/bam_diagnostics/mediumprop-init-improvements/static-orth/closeout-summary.json`；loss序列：同目录 `loss-final.json`。

相对原标准版，3000/3200/3400/3600四点为 +.001832/+.001228/+.002196/+.004926；2800–3600最后五点均值 +.002465，范围 +.001228～+.004926。早期−.238@200快速消失，1200起一直落后；未形成loss或速度收益（matched-health −.44%）。终局下注−.003未获支持，台账已删除下注。

认识更新：改善初始读头条件数可以改善启动，但本次收益没有持续；较高attention/V保留率也没有带来更低loss。因此不能继续把这些代理指标当成初始化优化目标；也不能据此声称原初始化全局最优。小读键版继续观察。

UE5a v5p-16 唯一READY租期：2026-10-08 12:25:20 → 14:29:40 UTC，2h04m20s；正常手动停止，非抢占。TB后台同步已核验SYNC_OK，本地目录位于 `/data0/xd/tensorboard_logs/BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdStaticVOrthogonalInitTruePile`。

4000步小读键：3200/3400/3600/3800/4000相对标准 +.000567/+.000180/+.001919/+.001968/+.001540，last5+.001235，范围+.000180～+.001968；接近持平但未形成持续改善，继续5000判定。完整序列 `small-key/loss-4000.json`；report-all的checkpoint4000已核验，pending事件只到3800，因此用loss-report刷新到成熟4000窗口。

小读键4000健康：末层V门.1297（原版.1434）、O门.3566（.2920），attention/V RMS比.3584（.2741）、O/attention1.7597（1.9480）；raw grad norm .278（.293），全程finite。它改变了两路权重分工，没有转化为loss收益；零初始化的键并未出现无法学习的迹象。完整健康 `small-key/health-4000.json`。

## 小读键版收尾（2026-10-08）

5000判定：相对原标准版4200/4400/4600/4800/5000五点 +.002330/+.000539/+.001322/+.001909/+.001353；均值+.001490，范围+.000539～+.002330。早期+.054@200快速缩小，后期落在+.001～+.002平台，没有持续反超趋势，也无速度、参数或cache收益，判停。最终stopped5073，checkpoint5073 committed；`closeout_runs_local.py`确认TPU/queue释放及TB `SYNC_OK`。主台账已删除−.006下注。完整loss/收尾/健康在artifact root的`small-key/loss-5000.json`、`closeout-summary.json`、`health-5000.json`。

5000末层健康：V/O门.1342/.3609（原标准.1479/.2869），attention/V比.3645（.2800），O/attention1.806（2.018）；raw grad norm .266（.262），无失稳。门控和写入分工不同，但loss没有受益；“zero read key造成长久学习障碍”被本次训练否定，不能用该理由继续押非零键初始化的收益。

UE5a v5p-16 唯一READY租期：2026-10-08 12:27:53 → 15:15:00 UTC，2h47m07s；0抢占，正常判停。资源assignment起点12:27:51。

两项自主初始化尝试均未带来持续收益：正交化提高初期启动速度但收益消失，小非零动态键改变门控轨迹但最终落后千分量级。原标准初始化目前仍是本方向最优已验证方案；这些结果不是全局最优的证明。后续若继续，较有判别力的是单独、适度降低静态V的初始幅度（保留原高斯方向与零动态键），把幅度效应与此前同时改VO读幅、门开度的改法分开。训练后列范数下降只是线索，不能作为最佳初始幅度的直接证据；该后续尚未启动。
