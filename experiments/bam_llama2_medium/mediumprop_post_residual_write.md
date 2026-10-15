# MediumProp post-residual write controls on single M-shift read .5

User authorized run on 2026-10-15. RUN `BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile`.
Worktree `/data0/xd/mediumprop-m-shift`, branch `codex/mediumprop-m-shift`.
Owned training TPU `xd-v5p-16-310240-maxtext`, intended zone us-east5-a; retained FLEX compiler `llm-jax-v6e-1-0` in europe-west4-a is borrowed only.

Parent `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile`, original runtime52cf0b3: independent attention-write R256 and MLP-write R256, read-internal shared R256 and generalized read, single read-only M-shift initialized .5. No read/attention-write projection sharing. xA=x+Attention(N_A(x),M); xF=xA+MLP(N_F(xA)). Attention write gate/address now consume N_A(xA); MLP write gate/address consume N_F(xF). Reuse the same named RMSNorm modules/parameters at the updated inputs, no stop-gradient, original projections/biases/initialization. Attention content remains original pre-W_O head output; MLP content remains original MLP output. Defer attention write control generation until after residual addition; retain one matrix decay per layer and the immutable read-only M-shift carry semantics. Sparse MLP-write layers unchanged.

No added parameters (0 W_Q); MLP widths [3802,3676,3802], total expected432111680 including the inherited576 gate-bias excess. No budget repayment needed. T4096 / globalbatch128 /13500 steps /checkpoint200 /latest2 /keep_period0 /finalcheckpoint /TBrotation500 /no QKNorm. All bam_record_* flags, record_internal_nn_metrics and generic training health disabled. Direct comparisons: independent parent, proven Medium shared-read/attention R256 direction SOTA, MHA. Both gain ratios use MHA as reference.

Pre-run bet: MHA-referenced gain versus independent parent terminal1.02, range .99--1.05. Architectural steady throughput expected1--3% lower; inherited historical parent timing includes health capture, so direct observed speed delta is confounded until matched evidence. New work retains all existing five runs; shared wake/normal1k Medium and2k XL reports.

Infrastructure audit: current main d4dc7617, read current AGENTS and training/tpu-ag skills. Compared train.py, train_compile.py, max_utils/maxtext_utils, optimizers, checkpointing, summary writer, base.yml, pyconfig. Operational files identical; train.py differences versus main are retained M-shift metric emitters only, current main base/pyconfig additions are QKNorm options unrelated to this runtime. TB positive500, rotating writer and begin_step inherited. Ported the verified generic health-off guard from `/data0/xd/mediumprop-qknorm/MaxText/train.py` (sole difference vs this family's train.py), default True preserves prior runs. No model/numerical main merge. UE5a chosen from recent comparable history: strong historical multi-hour completed leases, current first preemptions ~2--3h followed by prompt same-zone restoration; no user backup zones requested. No FLEX training capacity requested.

Validation: explicit tests for full-size exact parent budget/tree and resolved health-off, identical tiny parameter initialization, observed norm/projection inputs against actual residual additions on writing/non-MLP-writing layers, invariant content/output in an isolated block, causal scanned forward and finite/nonzero backward gradients. Include summary-writer regression and full pinned BAM suite because fusion/attention shared paths changed. Seal configuration and exact AOT before launch, require actual FIRST_STEP and speed gate.

## Verified launch

Runtime b49a3923d4154004e097b864bbad771976933a14; exact CPU3+writers5+fullBAM47, sealed config and v5p16/s13500AOT passed. UE5a TPU310240 FIRST_STEP6, actual432111680 parsed from140 worker parameter leaves, health flagsOFF, T4096/global128, correct4096 data, AOTloaded. Mean20--99 .4841625step/s (80points); historical parent .45835 has healthON, so raw+5.63% includes logging difference and cannot test architecture-only speed prior. All five prior runs remain active; report new run1k native200, parent and MediumsharedR256 comparisons retained, both gain ratios MHA-referenced. Artifacts /data0/xd/bam_diagnostics/post-residual-write/{state.json,startup-worker.log,launch.log}.

### Monitoring override, 2026-10-15

User removed `BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile` from this RUN baseline list and gain-ratio monitoring. Keep only the original single read .5 parent and MHA; one gain-ratio triple uses MHA reference and that parent comparator. Registry updated live, class compare_runs updated; runtime remains b49a392 with unchanged training/parameters. Historical comparison figures above are retained solely as history and will not recur in reports.

### Common2k, original parent only

Progress2180/CP2000 committed. Own-parent gaps1/1.2/1.4/1.6/1.8/2k -.024810/-.021286/-.022045/-.015110/-.011158/-.014164. MHA-reference gain ratios1.042444/1.044202/1.050423/1.039949/1.032034/1.044299. Last5 mean gap-.0167526, ratio1.0421815. Absolute benefit contracts but normalized benefit holds~1.04 over1--2k. Retain terminal1.02 [.99,1.05]. Original parent and MHA remain the only registered baselines. Next3k.


## 2026-10-15 补充两路：只改attention / 只改MLP

用户授权与现有双改路同组比较，所有路保持原单门read.5作为直接父路、MHA作收益倍数参考，不重新引入读/attention写共享R256对照。

新配置 `BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile` 与 `BamMediumPropMShiftReadG050PostMLPWriteControlsTruePile`。把既有双改开关拆为两个独立opt-in开关；旧总开关仍保留原语义。仅改attention时，MLP门/地址仍取加完attention后的prenorm残差、未看到MLP本次输出；仅改MLP时，attention门/地址仍取attention输入，MLP控制取加完本次MLP输出的残差。复用各自原norm参数；写内容不变。

两路都432111680参数、MLP[3802,3676,3802]，新增0 W_Q无需偿还。原配方T4096/global128、13500schedule、checkpoint200/latest2/TB500、所有健康统计关闭。运行源码仍在 `/data0/xd/mediumprop-m-shift` / `codex/mediumprop-m-shift`。

下注：相对原单门.5、共同MHA参考，attention单改终局收益倍数1.025（.995–1.06），MLP单改1.005（.98–1.035）。主要先验是attention新聚合上下文更值得用来控制写入，MLP点式变换提供额外收益有限；如MLP单改显著更好则推翻。现有双改3k倍数1.0247，近期五均1.0317且收窄；不把它当终局证据。两新路预期速度与双改相当或略快，MLP单改少18次attention控制norm，attention单改少6次MLP控制norm，具体以相同20–99窗口实测。

基础设施重新对照主分支c5e02ad8：train/train_compile/max_utils/maxtext_utils/optimizers/checkpointing的差别只剩本工作树已有更多shift健康导出，sow已关；record_training_health_metrics计算门控、TB500/rotating writer/begin_step和checkpoint修复已带上。保留历史模型谱系，无需整体合并主干。

CPU gate扩展为旧双改+两新路：全规模参数树与父路一致、实际门/地址输入来源拦截验证、单层向量输出不变/无MLP写层的MLP单改M完全不变、scan有限前向和各写投影非零梯度、因果性、实际健康输出为空、summary writer。同fingerprint文件锁避免两并发启动脚本重复跑CPU检查。两路AOT共享预留FLEX的worker锁串行编译；UE5a新v5p-16预排队与CPU/AOT并行。
