# BAM 研究工作入口

本文件记录项目特有的入口与约定；具体操作在对应 skill 中维护。

## 沟通与独立判断

- 简单干脆、开门见山：先给明确结论，再给最关键的依据。抓住问题实质，少铺垫、少重复，不用面面俱到的长篇解释掩盖没有判断。
- 有依据就明确判断、预测和下注；只强调会改变结论的不确定性，不拿“不能完全排除”稀释清楚的趋势。证据不足就直说缺什么，不编造确定性。
- 保持独立思考。用户的前提、推理或结论有错，直接指出错误及依据；不要一味附和、阿谀奉承，也不要为了显得独立而刻意反对。
- 遇到反直觉结果，主动核对实现、对照条件和历史证据，提出最有判别力的排查；不要用“存在交互”等泛泛解释草率收场，等用户一步步推动。
- 自己判断错了就直接承认并修正，不找理由维护原结论。简洁不等于省略用户要求的数据、报告格式或关键证据。

## 代码与实验台账

主工作树：`/home/xd/projects/maxtext`，主分支：`refactor-bam`。
从历史 commit 创建工作树时，显式读取主工作树的本文件和最新流程 skill；历史模型源码仍保持其指定 commit。

主 `MaxText/exp.py` 汇总**所有**实验，包括独立worktree中的实验和已回退的实现。
因此**配置类存在不等于实现仍在主干**：`ledger only` 记录对应的实现分支/worktree/runtime hash。
同一继承实验族的实现放在同一代码位置；较大且不确定的架构实验先在独立worktree验证，主干保留台账。

| 问题 | 入口 |
|---|---|
| 为什么设计 BAM、原始电路动机 | [/home/xd/projects/bam_attention/DESIGN.md](/home/xd/projects/bam_attention/DESIGN.md)；概念设计不代表当前实现全部具备 |
| 某实验到底改了什么、结果怎样、如何复现 | [MaxText/exp.py](MaxText/exp.py)：完整配置类名、继承链、7位 runtime hash、速度/区域、停止或暂停步数、相对结论 |
| 已有速度瓶颈、FLOPs、scaling与优化经验 | [bam_exp_memo.md](experiments/bam_llama2_medium/bam_exp_memo.md) |
| BAM前传与投影/归一化/门控 | [attentions.py](MaxText/layers/attentions.py)：`BamAttention`、`bam_read`、`factorized_head_bam_read`、`_attention_op`、`_bam_fetch_op` |
| M跨层携带、layer scan、标准Transformer部分 | [fusion.py](MaxText/layers/fusion.py)、[models.py](MaxText/layers/models.py)、[accelerator.py](MaxText/layers/accelerator.py)；配置解析见 `pyconfig.py` |
| 训练/AOT一致性、优化器、TB指标 | `MaxText/train.py`、`train_compile.py`、`optimizers.py`、`maxtext_utils.py` |
| 归因/因果诊断复用 | [readout attribution](experiments/bam_llama2_medium/bam_readout_attribution.md)；[残差IG](/data0/xd/bam-row-mediation/experiments/bam_llama2_medium/bam_residual_attribution.md)、[row消费者定位](/data0/xd/bam-row-mediation/experiments/bam_llama2_medium/bam_row_consumer_positions.md) 的 Reproduction 节给出脚本、分支、checkpoint及数据 |

常用锚点而非可互换基线：`BamLlama2MediumV2`（历史milestone）、
`BamLlama2MediumV2C256ScanAotCleanControl`（修正WD的Medium控制）、
`BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2`（历史XL16 Rank2）。
`V2`、旧 `ScanAotControl` 和 `CleanControl` 的数值/WD谱系不同。

术语按运算理解：`M[k,v]` 的 k/U 是 data，v/P_loc 是 address；列读 `M @ r_v → k`，
行读 `Mᵀ @ r_k → v`。LocalQK可读完整M而fetch读压缩M；BAM fetched读头数也可独立于MHA头数。

## 工作流入口

三个 skill：[$tpu-training](.agents/skills/tpu-training/SKILL.md)、
[$tpu-diagnostics](.agents/skills/tpu-diagnostics/SKILL.md)、[$tpu-ag](.agents/skills/tpu-ag/SKILL.md)。
`.agents/skills/` 链接到 `.claude/skills/`，是同一份内容。
编排脚本权威源码在 `/home/xd/projects/xd_tpu_scripts`，部署在 tpu-ag 的 `/home/lishengping/xd/projects`；
tpu-ag 主仓库不是运行时源码分发源：worker从Git取RUN指定commit，环境包和AOT从GCS取。

| 阶段 | 脚本入口（用法见相应skill） |
|---|---|
| 本地验证 | diagnostics skill 的 `scripts/run_bam_unit_tests.sh WORKTREE`，使用已固定的CPU环境 |
| 准备AOT | tpu-ag `prepare_train_aot.py` |
| 正式启动/续训 | tpu-ag `run_exp_xd.sh` → `auto_train_xd_maxtext.sh` |
| 监控 | tpu-ag `run_registry.py status` / `report-all`；当前RUN及 `compare_runs` 登记在 `run_registry/<RUN>.json` |
| TB健康指标 | 本机 training skill 的 `scripts/sync_tensorboard_incremental.py` + `report_bam_read_health.py` |
| 停止/收尾 | tpu-ag `closeout_runs.py`；暂停、热切换另见 training skill |
| checkpoint诊断/测速 | diagnostics skill 的runner；配对测速用 `scripts/run_profile_matrix.sh` |

[区域/抢占经验](experiments/tpu_region_preemption_history.md) 是选区依据，不把历史默认区当作固定最优区。
大产物走 worker → GCS → 本机；tpu-ag只编排，不中转/解析XPlane。
本地诊断数据在 `/data0/xd/bam_diagnostics`，TB在 `/data0/xd/tensorboard_logs`。

## 项目约定与历史陷阱

- 实验文档登记该任务的worktree、分支、RUN和 `xd-` TPU归属；主 `exp.py`、memo及tpu-ag编排脚本由多个session共享。
- 已启用仓库级、跨worktree的同名分支 [防误推hook](.githooks/README.md)；实验分支不能直接推到 `refactor-bam`。
- auto-train负责机械健康检查、抢占恢复、生成报告和TB收尾；agent负责解释、决策及交付。
  auto-train报告不会自动唤醒已退出的agent；持续监控用 training skill 的主动拉取循环。
- 用户要求新实验跑前“下注”：给出相对基线的loss/速度预期，事后检验；参数开销用 `W_Q=D²` 单位。
  推理M-cache也是BAM实验的重要目标，不只看训练loss/速度。
- 历史AOT入口曾漏传优化器WD规则；JIT/AOT、scan、RMS dtype/epsilon及初始化也曾造成可观轨迹差异。
  解释旧实验前区分实际runtime谱系，不能仅凭配置类名称认定同一基线。
