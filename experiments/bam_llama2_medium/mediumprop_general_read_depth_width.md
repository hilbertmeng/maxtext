# 泛化读端的深度与矩阵宽度扩展

工作树 `/data0/xd/mediumprop-full-m-read-gelu128`；分支 `codex/mediumprop-full-m-read-gelu128`；runtime `7d9d9ed7f476f14bc0a8980aaf2ce94affc9feaf`。主 exp.py 仅登记配置；完整 M GELU 读实现尚在实验分支。

父版 G18：18层 M75×32、共享读键 R256＋泛化读端，历史静态门 `1.1·sigmoid(logit)/.99`（原门初始 .99、实际乘数初始1）。新增三路全部沿用；不继承纯sigmoid重训路。

| 标签 | 层数 | M | 共享读键R | attention / embedding / 独立MLP写地址R | MLP三相宽度 | 实际参数 | 相对同层数MHA |
|---|---:|---|---:|---|---|---:|---:|
| A | 18 | 75×48 | 384 | 384/384/384 | 3540/3325/3540 | 432106720 | −14480 |
| B | 27 | 75×32 | 256 | 256/256/256 | 2223/2096/2223 | 432150656 | +7856 |
| C | 27 | 75×48 | 384 | 384/384/384 | 1952/1736/1952 | 432147232 | +4432 |

D1200、16头、QK57＋RoPE18、无W_V、保留W_O；完整M列读，无C压缩。所有层为L；独立MLP→M写入在零起始1/4/7/…，18层6次、27层9次。沿用父版3层block scan和健康指标，不同时变更调度实现。W_Q=1440000；各层按同层数MHA剩余预算就近整数配平，无硬件取整。

## 跑前下注

13500步相对G18：A −.003，B −.014，C −.015；速度分别−10%/−20%/−30%。C相对B押−.001，相对A押−.012。深度×扩M二阶值押+.002，即略低于简单叠加。
依据：原rank4加深收益约−.0206；读出改进与深度可能部分重叠。之前固定读R256的V48扩展收益仅约−.0025，R384在V32不胜R256；不能把扩R当成必然增益。

## 二阶指标与边界

所有差值在相同TruePile同步数±25窗口计算，负值表示增益叠加更好。

1. 深度×泛化读架构：`(B−G18)−(BAM27−BAM18)`。BAM18/27均为历史共享rank4标准版；这是整套读架构与深度的交互，不是泛化门单独消融。BAM27仅到10600完整窗口。
2. 扩V＋比例R×读架构：`(A−G18)−(DirectC12_V48−DirectC8_V32)`；同步附历史rank4 V48−V32作为背景。此指标包含从压缩直读改到完整M非线性读、泛化门及扩R，不能单独归因于某一个。
3. 旧固定读R的参照：`(A−G18)−(SharedR256_V48−SharedR256_V32)`；衡量当前泛化读＋比例R方案相对旧固定R扩V的差别，仍不能分离R384本身。
4. 最干净的当前四格：`(C−B)−(A−G18)`，等价`(C−A)−(B−G18)`；直接回答加深是否改变比例扩M的收益。
5. 历史DirectC深度×扩V参照：`[(C−B)−(A−G18)]−[(DirectC12_27−DirectC8_27)−(DirectC12_18−DirectC8_18)]` **不可计算**：历史没有DirectC8_27。不用rank4 BAM27冒充该缺失格。
   可计算较弱参照为 `(C−A)−(DirectC12_27−DirectC12_18)`；历史DirectC12_27完整窗口仅至7800。

收益倍数用同一MHA分母各自的同步数loss优势比；跨18/27层对比时另报MHA27−MHA18，不能把加深MHA基准变化隐掉。历史速度仅作未匹配runtime参照，新三路同runtime/health互比为主。

## 启动

UE5a SPOT v5p-16，ID310121/310122/310123。借用EW4a用户非抢占式 `llm-jax-v6e-1-0` 编译；绝不纳入回收。CPU集中验证三棵完整参数树、实际18/27层scan的tiny前传梯度和健康；同runtime结果复用。CPU/AOT/训练排队并行，全部通过后才启动。

- A：`BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile`
- B：`BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile`
- C：`BamMediumPropL27K75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile`

三路FIRST_STEP已确认（A5/B3/C5），训练端加载compiled function；有效配置、各自写地址R、历史门公式、MLP宽度和同区TruePile路径符合sealed runtime。初期50步全层健康指标有限。启动安装阶段三台初始机器均被抢占，脚本同区重排后成功；没有已训练步数丢失。

20–99步速度：A .448175（−6.38%），B .352950（−26.27%），C .321813（−32.77%），均对18层V32泛化父版 .4787。A低于预期代价；B超过−20%下注6.27个百分点，原因未做matched profile，不归因于扫描或FLOPs；C略超−30%预期。证据：`/data0/xd/bam_diagnostics/mediumprop-general-read-expansion/startup-verification.json`、`health-startup.json`。

共同每1000步报告；`report_factors.py`在tpu-ag根据缓存精确计算二阶差值，`r200`仍为gap绝对值变化百分比。三路各自loss登记直接父对照，四格/历史交互由该辅助报告呈现；缺失或停止后的窗口不填补。

## A/C收尾与认识更新（2026-10-12）

A停9701，C停6826；`scripts/closeout_runs_local.py`已验证最终checkpoint提交、两台TPU/queue消失，独立本地同步均SYNC_OK。B及XL两路继续训练。产物 `/data0/xd/bam_diagnostics/mediumprop-general-read-expansion/`，含closeout-AC.json、report-AC-final.md、summary-AC-final.json；远端收尾summary `logs/closeout-20261012T002614Z.json`。

- A−G18：早期大幅落后逐渐缩至零附近，8800–9600最后五点均值+.000635，范围[−.000109,+.001009]；9400短暂穿零后回正，未形成持续优势。相对历史V48/R256非泛化版均值−.001076；该差包含读R扩大和泛化读端，不能单独归因。
- C−B：初期+.089缩至约+.002–.003，6000–6800最后五点均值+.002817；没有继续逼近零。C−A均值−.019259，说明它赢A靠加深，扩V没赚。
- 扩V下注A−.003、C−B−.001方向没有兑现；C对G18的总优势不能掩盖扩V的失败。A/C分别比对应V32父版慢6.38%/8.82%，停止。
- B−G18在6200–7000均值−.020819；深度×读架构二阶均值+.000068，接近零：目前读法收益与深度收益近似叠加，我对深度收益折扣的下注缺乏实证支持。B未完成，保留原下注到终局。
- 深度×扩V在6000–6800均值+.001601；扩V在27层更不划算，接近跑前+.002的弱负交互预期。
- 不能推出大M本身无效：同时增加读写地址R、削减MLP。若继续定位，最清楚的控制是V48保持读R256、增加泛化读端，保持其他V48设定，分离读R384的分配代价；尚未启动。

READY租期全部UE5a SPOT v5p-16：A两期17:55:20→17:58:09（2m49，首步前抢占；prequeue补齐registry缺失起点）、18:04:43→次日00:21:39（6h16m56，最终停止）；C三期17:55:32→17:58:55（3m23，首步前抢占）、18:05:03→18:45:01（39m58，抢占）、18:52:15→次日00:21:42（5h29m27，最终停止）。没有切区或被动备选申请。

### 固定R历史对照对上述判断的修正

历史固定R256、未泛化读端时，扩V收益仍为+.002485（DirectC为+.007560，保留33%）；共享R256读法在V48上的收益+.006441（V32为+.011516，保留56%），均为12600–13400最后五点。准确表述是收益打折而非收益变负；gap差分+.005075不表示任一改动本身有害。证据 `historical-fixedR-width-interaction.json`。因此R384不能解释原有折扣；未启动的V48/R256＋泛化读端只能进一步区分新增R的代价与泛化读端的作用，不能期待自动恢复DirectC的扩V收益。
