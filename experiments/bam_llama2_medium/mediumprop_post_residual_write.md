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


### User stops both6167 and MLP-only2231, 2026-10-15

Explicit user stop, neither full terminal run. Both-change CP6167 and MLP-only CP2231 committed; nodes/queues verified absent, TB SYNC_OK. Attention-only continues, retaining both-change as direct stopped baseline plus original single.5/MHA. Both-change final mature window6000: parent gap-.002929, parent gain1.015793; last5gap-.003379/gain1.017569. Early2k gain1.044299 continued contracting, so 1.02 terminal central forecast was already undercut before stop, not a terminal observation. MLP-only final2200: parent gap+.000657/gain.997849, last5gap-.001082/gain1.002096; transient1400 benefit disappeared and crossed back behind1800. Versus both-change2200 gap+.011103, last5+.013503. Attention-only same2k gap-.018476/gain1.057784 vs original parent, and-.004312/gain1.012913 vs both-change. Evidence favors updating attention write controls, not mechanically updating every write control.

Resource leases UTC: both TPU310240 UE5a v5p16 P1,06:03:45–06:36:28 (32m43s,preempted),06:44:38–09:52:23 (3h07m45s,user stop). MLP-only TPU310245 UE5a v5p16 P0,08:32:30–09:52:26 (1h19m56s,user stop). No region switches. Closeout summary remote logs/closeout-20261015T095524Z.json; local post-residual-write/selective closeout-user-stop.json, both-final6167.json, mlp-final2231.json. Actual canonical speeds.4841625/.4846375 remain similar to attention-only.4844125.

### Attention-only common3k (2026-10-15)

Current3039/committedCP3000. Both-change stopped6167 and MLP-only stopped2231 at user request. Attention-only continues. Parent gaps at1k/2k/3k -.031872/-.018476/-.010694, MHA-reference gain1.054524/1.057784/1.042778. Latest five2200–3000 parent mean gap-.01232764 (range-.0141766..-.0106938), mean gain1.0447173. Versus both-change at3k gap-.0045298, gain1.0176845; latest five mean gap-.00354576, gain1.0125884. Versus MHA3k gap-.2606746. Attention-only is better than both at every available common window; evidence places the benefit primarily in attention write controls, while additionally changing MLP controls offsets part of it. Benefit versus parent still contracts; no terminal claim. Speed.4844125 essentially same as health-OFF both-change timing control(+.05%). Full cumulative native gaps/r200/gains in local report-common3000.json; missingboth800 retained, not interpolated. Next normal report4000.

### Proposed: share post-attention write R256 activation

User request2026-10-15: derive from attention-only post-residual write arm, keep read down independent, share attention/MLP write address R256 down and actual GELU activation in the six MLP-write layers. Implemented as the two norm-source RUNs below. Both query the same post-attention residual, but parent uses two independently learned prenorm scales. Preserve attention write path exactly: h=GELU(attention_norm(x+attention_output) @ P_loc_down), independently map h through attention P_loc_up and MLP mlp_address_up. MLP gate and MLP itself retain their existing post_attention_layer_norm input; attention gate remains its original full-D gate, not moved to R256. Native write content normalization and independent address normalization remain unchanged. No new gate/health metrics. Sharing must preserve gradient contributions from both write address consumers, and execute down/GELU once, not twice with equal parameters. Earlier pre-attention read/attention activation reuse is a different information time.

Remove six D1200xR256 kernels:1843200params=1.28 W_Q total. Refund locally: MLP-write width3676->3761 (+85), other slots3802; final widths[3802,3761,3802]. MLP adds1836000params, integer remainder7200below parent; expected total432104480, to verify via actual full parameter tree before any launch. Dominant dense MACs nearly cancel after refund, so expect approximately flat throughput, not the speed-up of removing an uncompensated projection. Pre-run forecast relative attention-only parent: gain multiplier1.03 [.99,1.08], MHA reference. Primary risk shared R256 capacity; potential gain coordinated write features plus larger MLP. Retain direction parent attention-only as direct baseline, also original single.5 and MHA for continuity.

### Launch preparation: two norm sources for shared write R256

User explicitly requested both norm sources. Classes `BamMediumPropPostAttentionWriteSharedR256AttentionNormTruePile` and `BamMediumPropPostAttentionWriteSharedR256MLPNormTruePile`, runtime branch/worktree unchanged (`codex/mediumprop-m-shift`, `/data0/xd/mediumprop-m-shift`). Direct parent attention-only `BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile`; preserve original single.5 plus MHA continuity baselines. Share only the write address down/GELU, actual activation and two gradient consumers. Norm source changes only in six MLP-write layers; other12 unchanged. Attention and MLP full-D write gates retain their own parent norm inputs in both arms. No common address up; no change to read down or native write factor RMS.

Actual full18 shaped parameter trees verified432104480 each, same shapes; no mlp_address_down leaves, P_loc_down retained. Widths[3802,3761,3802], parent-7200params integer remainder(-.005 W_Q total). Four focused checks pass: full budgets/healthOFF, actual unequal learned norm scales/single projection/shared GELU inputs/nonwrite-layer unchanged, joint two-consumer gradient additivity, scanned finite gradients and causality. Shared write feature computed before deferred attention factor creation; MLP-norm arm reuses the existing MLP norm result without recomputation. Inference does not add a cross-token/cross-layer state. Full BAM/oldpost-residual/summary-writer regression gate follows before FIRST_STEP.

Pre-run bets new/attention-only MHA-reference terminal gain: attentionnorm1.03 [.99,1.08]; MLPnorm1.04 [.99,1.10], slight preferenceMLPnorm. Both measured throughput expected approximately flat(+-1%) versus health-OFF parent. Dense kernels saved1843200MAC/token across6, refund1836000, net-7200MAC/token; conventional dominant-FLOP change negligible, not a forecast large speedup.

Operational audit against current mainc5e02ad8: no missing subsequent training/AOT/optimizer/summary-writer/checkpoint fixes. Branch retains full health-OFF guard, TBrotation500 begin_step, CP200/latest2/keep0, full13500 schedule, TruePile4096/T4096/global128/per-device16. Differences in main base.yml/pyconfig.py are Pythia-only rotary/validation settings, outside original recipe; preserve model lineage. Extra branch train.py changes concern M-shift health summaries, disabled for these runs. Spot UE5a v5p16 chosen from current task same-zone successful leases; new TPU IDs310249/310250. Verified retained llm-jax-v6e-1-0/EW4a FLEX_START ACTIVE, expiry2026-10-16T01:56:30Z; borrow AOT only with per-worker serialization, never lifecycle cleanup.

### Shared-write norm arms: actual startup verified

Both running frozen99485ac2f7c9414f641eaf79ed80c5c34b1fff35, UE5a v5p16 TPU310249(attention norm)/310250(MLP norm), FIRST_STEP and AOTloaded verified. Worker params432104480 each, exactwidths[3802,3761,3802], rank256, T4096/global128, zone-localTruePile4096, CP200/latest2/keep0/TBrotate500, generic/internal/allbam_record_* OFF verified. CPU59 checks pass(4new+3oldpost-residual+47BAM+5writer). Steady20–99(80/80samples) .4845000/.4842625step/s, versushealthOFF attention-onlyparent.4844125 +.018%/-.031%, flat as predicted. Two registered parent/originalsingle gain ratios use MHA; MLP-norm additionally compares its attention-norm sibling, same reference. No initial-loss verdict; first normal pairedreport1k. Allartifacts under local post-residual-write/shared-write.

### Attention-only common4k

Actual4088, committedCP4000; sharedwrite-normchildren286each, reporttheirfirstpairedlossat1k. Parentgap1k/2k/3k/4k -.031872/-.018476/-.010694/-.008295, MHA-reference gain1.05452/1.05778/1.04278/1.03816. Latestfive3200–4000meanparentgap-.009666, gain1.042190 versus3kmean1.044717, slowcontraction. Versusboth-change4k gap-.003161, gain1.014209; lastfive-.00325324/1.0138446. VersusMHA4k-.225652. Continue5k, retainoriginalterminalbet1.025[.995,1.06]. Fullcumulative200–4000 report-common4000.json in post-residual-write/selective.

### Shared-write norm common1k

Actual1036/1063, CP1000both. Relativeattention-onlyparent, gaps200/400/600/800/1000: attentionnorm+.0258912/+.01526/+.0142692/+.0088062/+.0037848, MLPnorm+.0131158/+.0143044/+.0115386/+.0096808/+.0034042. MHA-reference new/parent ratiosattention .984884/.990460/.986162/.988576/.993860, MLP .992342/.991058/.988810/.987441/.994477. Initialdeficitcontracting; sharedadditionalgainnotestablished, continue2k. MLP-minus-attentiongaps-.0127754/-.0009556/-.0027306/+.0008746/-.0003806, peerMHA-reference ratios1.007573/1.000603/1.002685/.998852/1.000621, noclearnormwinner. Relativeoriginalsingle.5 at1k gaps-.028087/-.0284676, ratios1.04805/1.04870 vsparent1.05452. MHA1kgaps-.6126294/-.6130100. Canonicalspeedchanges+.018%/-.031% flat. Retainterminalbets1.03/1.04; largerordinaryMLPrefundmaymatterlater, notestablishedhere. Fullcumulativeparent/peer/original/MHAgapsandratios inshared-write/report-common1000.json.

### Attention-only common5k

Original-single.5 gaps4000/4200/4400/4600/4800/5000: -.008295/-.008245/-.008279/-.008825/-.008071/-.007665; MHA-reference new/parent1.038162/1.038800/1.039675/1.042654/1.040529/1.038450. Latest5mean gap-.00821704, ratio1.04002154. Versus both-change5k gap-.003497, gain1.017185, latest5-.00349312/1.01665838. MHA gap-.207017. Standardized benefit remains~1.04 while rawgap contracts; continue6k, no terminal claim or forecast revision. CP5000committed, full report selective/report5000.json.

### Shared write norm common2k

CP2000both. Parentgaps at1k/1.2k/1.4k/1.6k/1.8k/2k: attention+.0037848/+.0031866/-.00017/+.002656/+.002342/+.0023576, MLP+.0034042/+.0034414/+.000435/+.0001374/+.002166/+.0011046. MHA-reference new/parentA .993860/.993750/1.000367/.993338/.993591/.993029; M .994477/.993250/.999060/.999655/.994073/.996734. Latest5meanparentgaps+.00207444/+.00145688, ratios.994815/.996554. Latest5MLP-minus-A gap-.00061756, peerR1.001753. No additionalsharedbenefit yet, slightMLPnormedge small. Originalsingle2k ratios1.050410/1.054329, inheritedattention-onlygainmainlyresponsible. Speedflat; retainbets/continue3k. Fullcumulativeshared-write/report-common2000.json.

### Attention-only common6k

CP6000committed. Parentgap5k/5.2k/5.4k/5.6k/5.8k/6k -.007665/-.006443/-.0068612/-.0074576/-.0060712/-.0064722; MHA-reference ratios1.038450/1.032563/1.035443/1.038908/1.031608/1.034900. Latest5gap-.00666104/R1.034684 versus5kR1.040022, slowcontraction. Both-change6k gap-.0035434/R1.018810, last5-.00328196/1.016821 stable. MHA6kgap-.191923. Continue7k, retainterminal1.025forecast. Sharedreportpendingeventstill5800 althoughlive>6025, manualregistryloss-report refreshedcomplete6000, no interpolation. Source selective/report6000.json. Stoppedbothbaselineends6000mature, do not repeatunchangedhorizon afterthisreport.
