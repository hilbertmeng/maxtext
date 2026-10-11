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
