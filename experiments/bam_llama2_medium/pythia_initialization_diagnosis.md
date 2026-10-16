# Pythia 初始化与 BAM 有效尺度诊断

当前源码 `6e4002bc`（`/data0/xd/pythia-mediumprop-t2048`）。本诊断只读取实际模块，不改变在训代码、checkpoint 或资源。并行和串行 GELU 参数初始化保持一致，已有逐参数相等回归检查。

## 实际规则

Medium D=1200、L=18。原配方普通 kernel std=.006；Pythia small_init=sqrt(2/(5D))=.0182574186，残差输出 wang_init=2/(L*sqrt(D))=.0032075015。token embedding、BAM 普通非零地址投影、写门和 embedding 写入均使用 small_init；attention/MLP 向量输出使用 wang_init。零初始化的读门、M-shift 门等保持其原有规则。

Pythia LayerNorm epsilon=1e-5 通过全局配置同时影响 BAM 写入 RMS，原值1e-6；读 key epsilon 已显式设1e-4，两边未变。该耦合需要在解释迁移结果时明确，不能把“RMS 会消除尺度”作为假定。

## CPU 初始化测量

实际 `EmbeddingBamWrite`，D1200/H16/K75/V32/rank256，bf16 激活、float32 参数。三个随机种子，每个256个按对应初始化分布生成的 token embedding 行；配对使用相同高斯方向和权重随机种子。只测初始化，不是训练 checkpoint 或真实 Pile loss 的因果测评。

| 设置 | seed U 归一化后 RMS | seed 地址归一化后 RMS | 初始 M RMS |
|---|---:|---:|---:|
| 原 embedding + 原 BAM投影 + epsilon1e-6 | .778264 | .059740 | .018519 |
| Pythia embedding + Pythia BAM投影 + epsilon1e-5 | .963751 | .467838 | .179397 |
| Pythia embedding + 原 BAM投影 + epsilon1e-5 | .766358 | .057513 | .017555 |

完整 Pythia 设置初始 M RMS 为原来的9.6873倍，三个种子倍率9.6935/9.6932/9.6753。embedding 写门均值均约 .1000；该差异来自投影与 epsilon 所处尺度区间，而非 gate 均值改变。原地址归一化前 RMS 约5.986e-5，远小于sqrt(1e-6)，因此 epsilon 不可忽略。Pythia 地址归一化前约.0016875。

完整8组因子组合及原始分种子数据：`/data0/xd/bam_diagnostics/pythia-init/embedding-seed-init.json`；可复用脚本 `probe_embedding_seed_init.py` 与运行日志 `probe.log` 同目录。使用固定CPU环境 `/data0/xd/conda/envs/maxtext-cpu/bin/python`、JAX_PLATFORMS=cpu、PYTHONPATH 指向上述源码，8核运行。

## 判断与待验证方案

初始化成为有定量证据的优先嫌疑，但该探测不证明它已经解释终局收益缩水。原迁移验证覆盖了Pythia投影初始化规则、前向与梯度数值对齐，遗漏了BAM初始矩阵和读写有效尺度的匹配检查。

用户已授权在串行 GELU BAM 上启动两路（2026-10-15），并要求暂停并行 SwiGLU 三路以供热替换。只恢复 BAM 专用投影不足以复现原始初始化链路；最终设计如下。

| 项目 | 串行 GELU 父路 | BAM 初始化修复 | 全量原初始化对照 |
|---|---|---|---|
| token embedding std | .0182574 | .006 | .006 |
| BAM 专用非零 kernel std（含 seed、读、attention/MLP 写门和地址） | .0182574 | .006 | .006 |
| 标准 QKV / 主 MLP 输入 / unembedding | Pythia small | Pythia small | .006 |
| attention/主 MLP 输出 | Pythia wang .00320750 | Pythia wang .00320750 | .006 |
| BAM 写入 RMS epsilon（seed、attention、MLP） | 1e-5 | 1e-6 | 1e-6 |
| 向量 LayerNorm epsilon | 1e-5 | 1e-5 | 1e-5 |
| 读 key epsilon | 1e-4 | 1e-4 | 1e-4 |

两路均保留串行残差、GELU、LayerNorm、偏置结构、RoPE18、单门 M-shift .5 和读/attention 写共享 R256；MLP 写 down 独立。原本为零的门 kernel、门偏置开度、归一化有效 scale=1 已核对与原配方一致，不改变。这里 epsilon 是前向计算规则，明确作为 BAM 适配修复登记，不冒称纯初始化变化。全量原初始化对照不是恢复整个 Llama 架构，也不是把所有参数都随机化：语义常量、零门与偏置保持各自原规则。

配置类：`BamPythiaMediumPropT2048SerialMLPBamInitFix`、`BamPythiaMediumPropT2048SerialMLPOriginalInit`，同一实现工作树 `/data0/xd/pythia-mediumprop-t2048`，分支 `codex/pythia-mediumprop-t2048`。两路参数均432356690，MLP[5832,5643,5832]，新增0 W_Q，不涉及参数偿还。T2048 × global256，13500步原schedule，从头训练，checkpoint200/latest2、TB500、关闭健康指标。

原串行 BAM 为直接方向基线；串行 MHA、串行 Mudd 为共同参考，报告相对父 BAM 的收益倍数以及 BAM/Mudd 倍数完整序列。两路之间保留直接比较。收益倍数默认参考 MHA，不使用 Mudd 代替 MHA。

开跑前下注：以串行 MHA 为共同参考，相对父 BAM 终局收益倍数，BAM修复1.12（1.02–1.25）、全量原初始化1.20（1.05–1.40）；若父族终局BAM/Mudd约1.14，对应约1.28、1.37。初始M尺度的机制证据支持认真押注，但尚未证明终局loss因果。初始化本身不增加训练FLOPs，预期没有速度损失；新runtime还修复健康开关，实际速度改善应独立注明，不能算初始化收益。

基础设施审计以主分支 `c5e02ad8` 为准，比较 train/train_compile/max_utils/maxtext_utils/optimizers/checkpointing；发现家族工作树缺少 `record_training_health_metrics` 的计算门控（配置虽false，旧train仍计算norm），本次移植该主分支修复。其余相关运维路径一致，已有TB500旋转与 begin_step、最终checkpoint/保留策略。主分支新增M-shift统计导出不影响本实验（关闭sow）。保留模型谱系，不整体合并主干。

验证包括完整参数树/形状与父路相同、逐参数初始化角色及语义常量、实际全宽 seed M 与原尺度一致、两路有限前向和非零梯度/因果性、原Pythia/串行回归、完整BAM回归及summary writer。CPU/AOT通过前不启动训练。

暂停记录：SwiGLU MHA 4527、Mudd3603、BAM3174，均有提交checkpoint；MHA TPU310241已释放，Mudd310242和BAM310243保留热替换。暂停为资源转向，结论仍属阶段性，不能记为失败。TPU选UE5a，沿用已READY的v5p-16，无新区域申请。预留FLEX `llm-jax-v6e-1-0` / EW4a，实际queued resource为FLEX_START，两路编译通过worker锁串行执行。


## 启动证据（2026-10-15）

Runtime `eaa95f811e4ca774a680e4f3323fc29efe812343` 已推送并通过SEALED_CONFIG。新增初始化5项、Pythia/串行回归4项、完整BAM47项、summary writer5项均通过；日志在 `/data0/xd/bam_diagnostics/pythia-init/`。实际全宽bf16 seed M RMS=.018492952，和用原epsilon的同参数模块逐元素完全相等。

两路都在原UE5a READY v5p-16上热替换，从头step0训练，不加载SwiGLU旧checkpoint：
- `BamPythiaMediumPropT2048SerialMLPBamInitFix`：物理TPU `xd-v5p-16-310243-maxtext`，旧RUN BAM SwiGLU保留CP3174；FIRST_STEP=5，.518step/s。
- `BamPythiaMediumPropT2048SerialMLPOriginalInit`：物理TPU `xd-v5p-16-310242-maxtext`，旧RUN Mudd SwiGLU保留CP3603；FIRST_STEP=9，.518step/s。

FLEX上的两个AOT均完成，状态 `/home/lishengping/xd/projects/aot_runs/eaa95f8-36b3968a.json`（修复版）和 `eaa95f8-78a1a3a6.json`（全量原初始化）。两份manifest的commit/topology/13500schedule已在hot-switch dry-run核对。预留编译机保持不变。暂停三路TB均SYNC_OK。

全量原初始化worker实际参数树156叶、432356690参数；实际配置已核对T2048/batch32 per-device、串行、MLP宽度、两类epsilon、checkpoint200/latest2/TB500、health OFF，AOT Loaded。修复版和稳定速度继续核对。特别注意.006是原配方的普通随机投影规则；如static_v_key本来使用专用初始化，仍保留其原有规则，语义零门及bias等不改成随机值。


两路worker配置和实际参数树已全部核对，均156叶、432356690参数、AOT Loaded、health OFF。固定20–99步各80个速度样本：BAM修复.5175875step/s，全量原初始化.5166625step/s；相对串行父路.5103625为+1.42%/+1.23%。两路差异约.18%，未显示初始化本身的速度影响；共同小幅提速与健康计算门控修复混杂。运行登记中的基线和两组收益倍数（各自相对父BAM、相对串行Mudd，均用串行MHA参考）已添加。


### Core gain report2000

Report BAM/Mudd ratios FIRST, each with own-family MHA. Steps1000/1200/1400/1600/1800/2000: BAM fix2.505087/2.516143/2.302020/2.246572/1.996239/2.099398; all-original1.921299/1.965807/1.813596/1.789764/1.646009/1.739619. Unfixed serial ends this sequence1.624957; exact architecture-matched old recipe1.485257. At2000 second-order fix versus unfixed1.291971, versus old recipe1.413492; all-original1.070563/1.171258. Same-family new/parent ratio equals the ratio of BAM/Mudd ratios algebraically because MHA and Mudd cancel, but reporting only that number hides the cross-recipe question. User explicitly corrected the omitted primary metric; put full BAM/Mudd and second-order sequences before parent gap in future reports.

Initialization fix absolute parent gain shrinks-.074227@1000→-.053175@2000 while second-order stays~1.29. All-original crosses parent at1200 and reaches-.012851@2000, still.040324 behind fix but catching up. Native window data in /data0/xd/bam_diagnostics/pythia-init/core-gain-common2000.json. Both checkpoints2000 committed. Early multipliers exceed old recipe even WITHOUT repair, so whether initialization fixes late relative degradation remains unproven; focus matched4k–6k crossover region and terminal. Speeds.5175875/.5166625 vs parent.5103625 include actual health-off compute fix.


### Core gain report3000

Both run3000 checkpoints committed. BAM/Mudd ratios at2000/2200/2400/2600/2800/3000: repair2.099398/2.023374/2.014201/1.986014/1.915420/1.892021; all-original1.739619/1.683254/1.700656/1.705939/1.655198/1.664113. Unfixed serial1.624957/1.554058/1.532138/1.515202/1.449346/1.432027. Repair/unfixed1.291971/1.301994/1.314635/1.310725/1.321576/1.321219; all-original/unfixed1.070563/1.083135/1.109989/1.125882/1.142031/1.162068. Cross-recipe ratios continue dropping (3000repair1.268986,all-original1.116127), while same-family initialization effect grows. Parent gap repair-.046445/all-original-.023433 at3000. Original's gap to repair shrinks.084702@1000→.040324@2000→.023012@3000.

Use known serial terminal1.147469 and matched-old-recipe latest5 through10000=1.473443 to define recovery threshold new/parent1.284081. If current second-order persists, conditional terminalrepair1.516058/all-original1.333437; this is not fitted extrapolation nor observed terminal. Matched old run stopped10160, so1.473443 is a late reference, not a full13500 endpoint. Repair's terminal new/parent forecast raised1.12->1.30[1.20,1.40]; preserve original1.12[1.02,1.25] pre-run bet. Original pre-run1.20[1.05,1.40] retained. Continue both;4k–6k is decisive for maintaining the recovery threshold.

Reusable experiment-specific reporter `report_pythia_init_gains.py` joins registry gain ratios, does not recompute raw loss windows. Invocation takes report-all JSON, completed serial BAM loss-report, matched old-recipe loss-report, through-step and output. Artifact /data0/xd/bam_diagnostics/pythia-init/core-gain-common3000.json; actual registry triple ratios remain the shared authority.

## Common4k update

Actual steps4042/4102, CP4000 both. BAM-specific repair/all-original BAM-over-Mudd ratios1.801077/1.630746, each using own-familyMHA. Second-order versus unfixed serial1.329051/1.203361; versus matched old recipe at4k1.215081/1.100169. Parent-BAM gaps-.0427632/-.0264286, mutual gap.0163346 down from.0230114 at3k. Initializer repair contribution remains strong; cross-recipe relative edge continues shrinking. Conditional terminal if current second-order holds1.525045/1.380820; not fitted/observed terminal. Recovery threshold1.284081 versus matched old late10k gain1.473443; repair arm above, full-original below. Continueboth to5k, retain forecasts1.30[1.20,1.40] and1.20[1.05,1.40]. Full200-step primary series in core-gain-common4000.json; original report report-common4000.json. Report helper now selects exact init RUN names from shared multi-group report-all instead of misclassifying unrelated arms.

## Common5k: initialization contribution persists

BAMrepair/all-original primary BAM-over-Mudd1.712527/1.598142 using ownfamilyMHA. Second-order versusunfixedserial1.341942/1.252310, versusmatchedoldrecipe1.152453/1.075477. Parent gaps-.0403476/-.0297714, mutualgap.0105762 versus3k.0230114 and4k.0163346. Repair contribution3k/4k/5k1.321219/1.329051/1.341942; full-original1.162068/1.203361/1.252310. Strong evidence for BAMinitialization-chain mismatch while retainingserialGELU/LN; terminalrecovery stillunobserved. Full-original1.20centralforecastnowconservative; originalbets retained, no new numerical forecast issuedat5k. Conditionalterminalifcurrentcontributionshold1.539837/1.436987 versusknownoldlate10k1.473443 (notoldfullterminal); recoverythreshold1.284081. Continueboth6k.

CompletedserialMudd4800-windowcachewasmissingraw4780/4790/4800 afterpreemption. Recoveredexactpoints fromalready-syncedTensorBoard, importedonlythose3, preservedexistinglogvalues; overlap4810/4820difference<=1.9e-7 matchesrounding. Sharedregistryrecomputedallratios, nointerpolation/privatewindowaveraging. Repairprovenance /data0/xd/bam_diagnostics/pythia-serial/mudd-4800-cache-repair.json. Full200-stepsequence core-gain-common5000.json andreport-common5000-repaired.json.

## Common6k: initialization benefit strengthens

At6000 primary BAM/Mudd ratios repair1.720946/all-original1.612302, each own-family MHA. Repair/unfixed1.366223 versus5k1.341942; all-original/unfixed1.279973 versus5k1.252310. Versus matched-old-recipe same-stage1.155768/1.082803. Recovery threshold1.284081 is exceeded by repair, nearly reached by all-original; it compares a conditional new terminal with the old late10k reference, not an observed new or full-old terminal. If current contributions hold conditional terminal1.567699/1.468729. Still no evidence that restoring standard trunk initialization improves on BAM-only adaptation; no claim about curing MHA absolute loss deficit. Original forecasts retained. Complete native series core-gain-common6000.json and report-common6000-mature.json. BothCP6000 committed; next7k.

## Common7k

CP7000both. PrimaryBAM/Mudd repair1.675229/all-original1.578812; second-orderunfixed1.364435/1.285905; second-orderoldrecipe1.131197/1.066091. Repaircontribution6k1.366223->7k1.364435 stable~1.36, whileprimaryandcross-recipeedgestillcontract. Parentgaps-.0385608/-.0302516. Full-originalsinglepoint1.285905justaboverecoverythreshold1.284081, butlatest5meanbelowthreshold; conditionalestimate notobservedterminal. Fullnative6k/6.2k/6.4k/6.6k/6.8k/7kprimary repair1.720946/1.696401/1.681294/1.686960/1.657650/1.675229, fulloriginal1.612302/1.591275/1.582752/1.592641/1.569566/1.578812. Completeallprimary/second-orderin core-gain-common7000.json, registryreport report-common7000.json. Bothcontinue8k; preservebets.

## Common8k after first preemption

CP8000both, actual8355/8037. Repair/all-original primary BAM/Mudd1.640465/1.557850, new/unfixed1.368075/1.299178, new/matchedold1.105249/1.049589. Wholelastfiveall-original ratios1.299344/1.296346/1.306686/1.303847/1.299178 exceed recoverythreshold1.284081; bothwouldrecoveroldlateadvantageIFcontributionpersists, notobservedterminal. Repaircontributionstable~1.37, full-original~1.30. Repair8kparentgap-.0368916, original-.0299862, mutual.0069054. Primaryandcross-recipeedgesstillcontract; currentbestinitadaptationretainsPythiastandardtrunk. Core8kquestion evidence favorsBAMinitialization-chain/writeepsilonadaptation asmajorfactor, notsoleisolatedinitcause. Source core-gain-common8000.json andreport-common8000.json.

OriginalarmfirstUE5apreemption2026-10-15T12:32:51Z, READY08:16:33–12:32:51(4h16m18s), CP7770committed. Recovered7771 exacteaa95f8/full13500schedule/AOT, resumedLRconsistentwithstep7771, CP7800then8000committed. No from0restart. Native7800windowentirelypostrestore, previous7600cache retained. Continue9k, preserveoriginalbets.

## Common9k

BothCP9000committed actual9356/9047. Primaryrepair/all-original1.632546/1.558084, new/unfixed1.386955/1.323694, new/matchedold1.095827/1.045845. Newprimaryoverrecent~1kstablewhileunfixeddeclines, soinitialization-chain/writeepsilonadaptationbenefitpersistsmidlate. Parentgaps-.0372112/-.0311278, mutual.0060834. Full200-stepseries core-gain-common9000.json, report-common9000.json. Continue10k; no terminalclaim. OriginalarmsecondcorrelatedUE5apreemptionCP9104, priorfull13500schedule/runtimeto beresumed; keep separatefromperformancejudgment.

## Common10k

BothCP10000committed; actualfix10790/original10078. BAM/Mudd9k/9.2k/9.4k/9.6k/9.8k/10k fix1.632546/1.623608/1.629457/1.630342/1.617951/1.612451, original1.558084/1.540857/1.550398/1.551406/1.547384/1.548499. Versusunfixedsecondorderfix1.386955/1.386921/1.388831/1.406790/1.389173/1.390247, original1.323694/1.316233/1.321447/1.338678/1.328585/1.335108. Last5matchedoldsecondorder1.101396/1.050448. Matchingoldstopped10160, lastusablewindow10000; futureprimaryreportscontinuewithoutinventingoldwindows. Parent10kgaps-.0359184/-.0308434; mutual.005075. Primaryrecentdeclineslightlywhileunfixeddeclinesfaster,sorepaircontributionpersistsmidlate. PreferBAM-specificchainfixoverfulltrunkoldinit. Speeds.5175875/.5166625, no newtimingclaim. Continue11k. Fullnativeprimaryseriescore-gain-common10000.json; allrawregisteredgaps/r200report-common10000.json. No terminalclaim.

### User query: absolute loss of the two initialization arms

At10400samecomplete±25window, BAM-specificfixmeanloss2.4164636 versusall-original2.4220994, gap original-minus-fix+.0056358. Full1k..10kgaps+.0847022/+.0403236/+.0230114/+.0163346/+.0105762/+.0093202/+.0083092/+.0069054/+.0060834/+.005075; latest10200/10400+.0062438/+.0056358. Fixalwaysahead, longhorizonleadcontracts; lasttwowindowsreboundaround.005–.006, no observedcrossoverorjustifiedimminentcrossingclaim. StillpreferBAM-onlyfix, terminalgapunresolved. Thisisad-hocuserquery, normalreportcursorremains10000,next11000. Sourceoriginal-vs-fix-current.json.

## Common11k

BothCP11000committed, actualfix11757/original11056. BAM/Mudd10k/10.2k/10.4k/10.6k/10.8k/11k fix1.612451/1.631395/1.621777/1.625284/1.621031/1.614589, original1.548499/1.552103/1.550717/1.559603/1.541114/1.541937. Last5means1.622815/1.549095 compared10k1.622762/1.547709, essentiallyheld. New/unfixedsecondorder11k1.394575/1.331823,last5means1.396143/1.332728. Mutualabsoluteoriginal-minus-fix+.005075/+.0062438/+.0056358/+.005201/+.0062496/+.0056794, last5+.00580192: longhorizoncontractionhaspausedlatest~1k, currentevidencemorefavorsretainedlateadvantagethanimminentcrossing. Parent11kgaps-.0357116/-.0300322; Muddgaps-.0480446/-.0423652,MHAgaps-.1262182/-.1205388. Oldmatchedreferenceends10000, helpercorrectlyretains10200..11000witholdnull; no oldwindowextrapolationorprimaryseriestruncation. Continue12k. Fullnativecore-gain-common11000.json/report-common11000.json.

### Hypothetical old-architecture / Pythia-standard-init trio — forecast only, NOT launched

Userasksforecast: originalpre-migrationSwiGLU/RMSNorm/serialcircuitandtrainingrecipe, onlystandardTransformerparameterinitializersswitchfrom.006toPythia; BAM-specifickernels/gates/norms/writeepsilonandMuddroutingunchanged. Literalinterpretationincludesstandardtokenembedding. NoRUN/class/resources/AOTsubmitted. CounterfactualforecastA=L_MHA-L_BAM same-stagechange+.005(range-.010..+.020), terminalR=(L_MHA-L_BAM)/(L_MHA-L_Mudd) central1.50(range1.35..1.70), judgmentrangesnotconfidenceintervals. MatchedoldT2048late10kreferenceAmean.13806184/Rmean1.473443, notoldterminal.

EvidencecurrentGELU/LNpairfix-versus-originalabsoluteleadlast5~.005802 supportsPythiastandardtrunkforBAM; butMHA/Muddold-initcontrolsabsent,andSwiGLUtwoinputprojectionsscaleresponsediffersfromGELU,soimprovementcannotbecopiedtorelativegains. Crucialboundary: BOTHcurrentinitarmsrestoretokenembedding.006, whileliteralhypotheticalchangesitto.018257. ExistingactualEmbeddingBamWrite3seedprobe, oldprojections/oldepsilon1e-6: oldembeddingM_RMS.0185188161, Pythiaembedding.0688865806(~3.72x). Do notconfusenewepsilon1e-5case.01755485withliteraloriginal-epsiloncounterfactual. BAM-specialparametersunchangeddoesnotmeaninitialMunchanged. Thiswasexplainedtouser; forecastexplicitlyincludesembeddingchange. Ifuserlaterexcludesembedding, updateforecastandimplementationbeforelaunch. Initializerrolesneeddecouplingfromdecoder_blockwhenimplementing: currentfusionattn/MLPresidual-outputoverridegatedonpythiablock; changingonlyinit_methodwouldnotimplementfullWangoutputinitializationonfusion. Noimplementationdoneinthisquery.

## Common12k

Actualfix12744/committedCP12600, original12033/CP12000, bothcoverreporthorizon12000. Primary11k/11.2k/11.4k/11.6k/11.8k/12k fix1.614589/1.620287/1.611376/1.612199/1.617422/1.620809, original1.541937/1.543963/1.541972/1.543381/1.550075/1.549371; last5means1.616418/1.545752. New/unfixedsecondorder12k1.412167/1.349925,last5means1.398433/1.337303. Absoluteoriginal-minus-fix+.0056794/+.0059302/+.0055308/+.0054298/+.0053428/+.0055218, last5+.00555108vs11k+.00580192: modestnarrowingbutadvantageremains. Parent12kgaps-.0365654/-.0310436(last5-.03618552/-.03063444), Mudd-.0479854/-.0424636(last5-.04843716/-.04288608), MHA-.1252804/-.1197586(last5-.12702048/-.1214694). CurrentGELU/LNcircuitstillfavorsPythiastandardtrunkwithBAMchainrestored; thisdoesnotmeasureoldSwiGLUcounterfactual. Continue13k/possibleearlierfixcompletionasynceventwithwholegroupcontext. Fullnativecore-gain-common12000.json/report-common12000.json.

### User clarifies counterfactual motivation: BAM-init-fix embedding is original .006

UserfocusesontheBAM-specificfixarmasabasis. Confirmedactualtoken_embedding_init_method='006normal'; bothcurrentinitarmsuse.006tokenembedding, BAM-specialprojections.006/writeepsilon1e-6; standardtrunk/unembeddingretainPythiaonlyinfixarm. Correctedpriorinterpretation: reproducingthispartitiononoldarchitecturekeepsembedding.006. PriorforecasttableexplicitlyassumedembeddingPythiaandisNOTapre-runbetforthisclarifiedpartition. 3.72xinitialMwarningdoesnotapplywhenembeddingkept.006. NoRUNsubmissionorcodechangeauthorizedbythisfactualquestion. Beforeanyfuturelaunch, registerforecastfortheactualagreedpartitionratherthansilentlyreusingtheoldcounterfactualbet.

### Revised forecast after explicit embedding=.006 clarification — NOT launched

Userexplicitlyrequestsreevaluationwithtokenembeddingoriginal.006, ordinaryTransformerprojectionrolesPythia, originalpre-migrationarchitecture/recipe,BAMinitialization/writeepsilonallunchanged. Revisedforecast: same-stageBAM-vs-MHAadvantageincrementcentral+.005(range-.005..+.015), newtrioterminalBAM/MuddMHA-refRcentral1.55(range1.40..1.70). Subjectivejudgmentranges,notstatisticalCI. PredictBAMownabsolutelossmodestlylower. This supersedes literalembedding-changedprior1.50forecast; noRUNsubmissionyet.

Illustrativeanchorcomputedfromexistingregistrywindows,notfreshrawaveraging: oldmatchedT2048late10klast5A=.13806184, U=.09370232; currentGELUpairordinary-initpositiveeffectlast5=.00555108. HoldingMHA/Muddhypotheticallyfixedyields(A+.00555108)/U=1.53265063. ThisisNOTa predictionderivationbecausetheproposedtriochangesallthreeordinaryinitializations. Maintransferableevidenceis~.00555standard-trunkbenefit, notthelargeBAM-chainrepairbenefitbecauseoldarchitecturealreadyhastunedBAMinitialization. SwiGLUtwoinputprojectionsandunmeasuredMHA/Muddresponsesremainmaterialuncertainties. Testfalsifier: BAMownlossimprovesbutAandRdonot ->genericinitializerimprovement, notBAMrelativegain. NoBAMinitialMscalechangefromembeddingunderclarifiedpartition.

### Baseline correction: original MediumProp T2048 trio

UseraskswhichcurrentRthe1.55betcompareswith. OriginalcontextlengthtrioBAMclass BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048 (no later single.5/read-attention-down-sharing additions), MHA BamMHAMediumPropC256T2048Matched, Mudd MuddLlama2MediumPropC256T2048. Allcompleted13500. OriginalBAMterminal12600–13400last5A=.121129,Muddadvantage=.088513,Rmean1.368510. Previous1.55forecastincorrectlyanchoredlatermatched-circuitBAMclass BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048, stopped10160/lastfull10000,Rmean1.473443. Explicitlyexplainedbaselinemixandbothvalues.

ForSTRICToriginaltrioonlyinitializerchange(embedding.006, standardTransformerprojectionrolesPythia, BAM/otherarchitecture/recipeunchanged), revisedcenterR1.43(range1.30..1.60), Aincrementstill+.005(range-.005..+.015). Illustrativeanchor(.121129+.005)/.088513=1.425, prior1.55wouldbe13.3%relativeincreasefromoriginalR1.36851andisnotsupportedascentralforecast. Thisreplaces1.55ONLYforthisoriginaltrioarchitecture; keepoldmatchedcircuitnumbersclearlylabeled. NoRUNauthorized/launched. OriginaltrioBAMvs laterSOTABAMmustbeexplicitinanyfuturelaunchclass, neverinferfromT2048labelalone. Source mainexp.py originalclass14260+ andmediumprop_context_t2048.md; latermatchedclass15600+.

Reporthelpernowaccepts --arm-report RUN JSON asfallbackforacompleted/acknowledgedarmwhileitspeerruns. Livependingreporttakesprecedence; verifiesratioowner; keepsoriginaloutputsidenticalwithoutfallback. Fiveactual12kdataregressionspassed:unchangedlive,nullpendingfallback,omittedcompletedentryfallback,missingcacherejected,wrongarmrejected. Provenance directory pythia-init/completed-arm-report-check (actualpathpythia-init). Preparedforasynchronouscompletion, notnewlosswindowimplementation.

## 初始化组：共同13k与修复路完成（2026-10-15 UTC）

共同13k的BAM/Mudd收益倍数：修复路1.617556，全原初始化1.545702；末五1.618638/1.551212。共同串行MHA参考。相对未修复串行BAM的二阶倍数为1.414845/1.351996。完整序列见 `/data0/xd/bam_diagnostics/pythia-init/core-common13000.json`，全原路仍训练至13500。

修复路已完成13500，最终CP13500提交，节点/队列独立查询均NOT_FOUND，最终TB同步SYNC_OK。终局末五12600–13400：BAM/Mudd1.616629、相对未修复串行版二阶1.408882；对未修复BAM/Mudd/MHA的loss gap分别−.03585036/−.04711444/−.12353160。跑前1.12[1.02,1.25]明显低估；3k修订1.30[1.20,1.40]仍偏低。最初看好全原初始化超过局部修复，实际已训共同阶段持续相反；不能把embedding+BAM投影+写epsilon组合修复归因为单个std。

资源：UE5a v5p16 `xd-v5p-16-310243-maxtext`，P0，无换区；唯一READY租期08:18:55–15:40:23 UTC，共7h21m28s。终局报告 `fix-final13500.json`、资源验证 `fix-final-resources.json`、租期 `fix-final-leases.txt`、收尾 `fix-completed-closeout.log` 均位于上述目录。权威registry状态complete；本地幂等closeout包装器already_closed时的stopped标签不改变实际完成状态。

## 未启动假设：仅修复路embedding恢复Pythia（2026-10-15 UTC）

用户询问预期，未授权开跑。父路为已完成的串行Pythia BAM初始化修复，MHA/Mudd对照固定。只将token_embedding_init_method从006normal清空以继承init_method=pythia；token embedding独立正态std=sqrt(2/(5D))=.0182574186（D1200），原std=.006，标准差3.042903倍、方差9.259259倍；unembedding独立且不变。BAM投影.006和写epsilon1e-6均保留。

既有3种子实际EmbeddingBamWrite初始化探针给出：原embedding初始M RMS均值.0185188161，Pythia embedding .0688865806，约3.72倍；并非真实Pile loss或训练因果测评，不能把尺度变大直接等同于loss变差。现有修复/全原两路embedding均.006，没有这项单独对照。

主观下注：终局BAM loss较修复路增加+.009，范围−.003～+.020；以修复路实测2.36197836为锚约2.371。终局BAM/Mudd收益倍数（串行MHA参考）中心1.50，范围1.35～1.65，当前实测1.616629。两项须一致：固定对照MHA−Mudd优势约.07641716，loss每增加.001，倍数约下降.0131。判断为损失部分修复收益，仍强于未修复路；范围为判断区间，非统计置信区间。禁止把该预测当新实验结果或写进exp.py。

## 未启动假设：Pythia embedding的BAM侧尺度补偿（2026-10-15 UTC）

推荐仅在EmbeddingBamWrite输入使用固定alpha=.006/sqrt(2/(5D))=.3286335345，不改变向量残差的Pythia embedding。保留BAM全部.006投影、写epsilon1e-6与门偏置；无新增参数/GEMM，逐元素乘法开销可忽略。备选只将直接接收embedding的W_emb_u/W_emb_v_down/W_emb_g初始化std降至.0019718012，W_emb_v_up仍.006；该方案虽初始前向等价，但Adam相对初值的更新比例改变，优先固定输入scale。不能将全部BAM投影一起缩小。

已有实际模块的配对CPU验证（Gaussian token rows，不是真实Pile loss）：三种子bf16初始M RMS，baseline.0185188161、无补偿.0688865806、固定输入scale.0184793665、仅三首投影缩初值.0185186025；bf16逐元素M相对L2误差.72786%/.71580%，cosine约.999946。FP32固定输入scale相对误差3.4375e-7；其余BAM配置/权重完全一致。可复现脚本 `/data0/xd/bam_diagnostics/pythia-init/probe_embedding_compensation.py`，结果 `embedding-compensation-init.json`，日志 `embedding-compensation-probe.log`；固定CPU环境、JAX_PLATFORMS=cpu、8核。未改模型源码、未启动训练。

补偿后下注：当前修复路为父，固定MHA/Mudd对照，终局BAM/Mudd倍数中心1.61，范围1.49–1.68；绝对loss相对父路中心+.0005，范围−.005～+.010。范围为判断区间，非统计CI。预期追回不补偿时所押的大部分损失，但向量残差初始尺度、首层LayerNorm及残差更新比例仍改变，不能保证整网轨迹等价。

## 初始化组完成：最终配对结论（2026-10-15 UTC）

两路均完成13500，末五12600–13400：未修复串行BAM loss2.39782872、BAM/Mudd1.147469；BAM专用修复loss2.36197836、倍数1.616629、相对未修复二阶1.408882；全原初始化loss2.36709296、倍数1.549689、二阶1.350542。共同串行MHA loss2.48550996、Mudd2.40909280。全原比修复loss高.00511460，修复领先贯穿后期。修复预下注二阶1.12[1.02,1.25]显著低估；全原1.20[1.05,1.40]中心低估、仍在区间内。最初认为全原更好，实际相反；支持在该串行GELU/LN配方保留Pythia标准干路，但不等于对其他骨架可直接迁移或隔离embedding效应。

全原CP13500提交、节点/队列独立查询NOT_FOUND、TB SYNC_OK；自动registry complete/clean-exit于16:02:18UTC，未被幂等收尾包装器改变。UE5a v5p16 TPU310242，P2，无换区，三段完整READY租期：08:16:33–12:32:51（4h16m18s，抢占），12:38:35–13:28:09（49m34s，抢占），13:35:04–16:02:18（2h27m14s，正常完成）。最终registry/loss/资源/租期/收尾文件位于 `/data0/xd/bam_diagnostics/pythia-init/original-final*` 和 `original-completed-closeout.log`。两次抢占均恢复原eaa95f8/full13500计划，包含优化器和数据位置，未重置训练。

## Pythia embedding初始化来源核验（2026-10-15 UTC）

直接依据是官方Pythia配置采用init_method=small_init，以及GPT-NeoX将该initializer传给词embedding的完整调用链：
- 配置：https://github.com/EleutherAI/pythia/blob/main/models/410M/pythia-410m.yml
- 公式及非截断normal实现：https://github.com/EleutherAI/gpt-neox/blob/v1.0/megatron/model/init_functions.py#L92-L100
- 模型向embedding传入init_method：https://github.com/EleutherAI/gpt-neox/blob/v1.0/megatron/model/gpt2_model.py#L174-L184
- 词表权重使用同一initializer：https://github.com/EleutherAI/gpt-neox/blob/v1.0/megatron/model/word_embeddings.py#L49-L55

公式small_init的研究来源为Nguyen & Salazar2019《Transformers without Tears》§2.2，https://aclanthology.org/2019.iwslt-1.17.pdf：将attention初始化从方差1/D缩到FFN D→4D对应的2/(5D)。该论文§3.2的非FixNorm词embedding使用std=1/sqrt(D)，所以它不直接证明embedding应取2/(5D)；embedding的该选择来自Pythia实际代码。不得将官方配方迁移等同于已证明对BAM最优。GPT-NeoX函数注释中的论文年份2010是上游笔误，实际2019。

## 用户提供的PaxML/Praxis复现代码核验（2026-10-15 UTC）

固定源码paxml9f12f128808b2496b1b55f40e3477753db3e6896、praxis3d80c945f5d469f836bda6b7d9e61e17280219f7，xd/dev。用户锚点为PilePythia3B128x1FixRot，显式混入PythiaInit。configure_gpt3_task计算small_init std=sqrt(2/(5D))，赋给model.params_init与unembedding；独立embedding未覆盖initializer，经BaseLayer.copy_base_hparams/create_child/create_variable继承同一Gaussian。WeightInit.Gaussian的scale即std，实际init_var使用scale*jax.random.normal；无额外fan缩放，查表后scale_sqrt_depth=False。attention输出与FFN第二投影由wang_init=2/(L*sqrt(D))覆盖，两个相关gshard内部初始化覆盖默认False且该配置未开启。当前Medium D1200/L18得到.0182574186/.0032075015，与当前完整Pythia迁移普通角色初始化一致；修复路embedding.006为有意实验偏离，不能称完整Pythia复现。

部分7B AlignInit配置混入另一PythiaInit7B，显式std .02/output .0025，不属于上述small/wang数值；不能仅凭Pythia类名归为同一初始化。这里只确认代码，不据此推断未提供的历史运行hash/checkpoint实测。完整固定源码与audit-summary.json在 `/data0/xd/bam_diagnostics/pythia-paxml-init-audit-20261016/`。未改模型、未启动训练。

## Pythia embedding with compensated seed write — launch definition

User resolved the architecture to the initialization group's serial Pythia GELU/LayerNorm and then removed unchanged controls. Launch only `BamPythiaMediumPropT2048SerialMLPPythiaEmbeddingScaledWrite`; reuse completed `PythiaMediumPropT2048SerialMLP` (runtime d434e69) and `MuddPythiaMediumPropT2048SerialMLP` (3538f6c). Both controls already initialized embedding with Pythia small_init, std sqrt(2/(5D))=.0182574185835 at D1200, so neither needs rerunning.

Direct model parent is `BamPythiaMediumPropT2048SerialMLPBamInitFix` (eaa95f8). The only model changes are `token_embedding_init_method=''` and fixed `bam_embedding_write_input_scale=.3286335345030997`, applied before all three seed input projections. The vector residual sees the full Pythia embedding. BAM-specific .006 kernels, write epsilon1e-6, semantic zeros/biases/norm scales, standard Pythia initialization, serial GELU/LN and MLP widths remain identical. 432356690 parameters, delta0 W_Q. No router-budget changes.

Forecast: terminal BAM/Mudd R1.61 [1.49,1.68], matched serial MHA reference; loss versus repaired parent +.0005 [-.005,.010]. No material speed change expected; historical speed comparisons need matched health-compute settings. T2048/global256, legacy2048,13500 steps/native200, CP200/latest2/no permanent retention, TB rotation500, health all off. TPU xd-v5p-16-310253-maxtext, UE5a first. Borrow llm-jax-v6e-1-0 EW4a worker0 without lifecycle ownership; worker lock serializes the two AOTs.

Infrastructure audit against current main334f26a9: train/train_compile/optimizers/max_utils/maxtext_utils/checkpointing/summary_writer inspected. Main-only train difference is unused M-shift health emission; no compatible checkpoint/logging/AOT/WD fix missing. Writer construction uses configured positive rotation500 and calls begin_step before scalars; summary-writer tests in targeted launch gate. Preserve verified model lineage instead of merging main. Audit patch: /data0/xd/bam_diagnostics/pythia-init/new-family-infra-reviewed.patch.

Before either FIRST_STEP, user requested fresh bets: supersede preliminary1.61 with terminal BAM/Mudd1.60 [1.45,1.72], serial MHA reference. Repaired parent1.616629; expected loss delta about+.0013. Compensation preserves initial M, not vector residual scale or relative embedding updates. Speed expected unchanged. This updates a launch forecast, not an interim training conclusion.

Startup verified: runtime f98ce98a0c964d561cf8c9bee33069ed4cdb782b, FIRST_STEP and Loaded compiled function confirmed; actual worker parameter shapes sum to 432356690 (156 leaves), all health flags OFF. Canonical steps20–99 mean 0.5151875 step/s over80 samples. Matched health-off init-fix parent .5175875: -0.464% throughput.

## Embedding sensitivity mechanism analysis and MHA/Mudd forecasts

Question: under fixed Pythia standard Transformer initialization and original BAM-specific initialization, why does small token embedding help BAM, and would it transfer to MHA/Mudd? Existing comparison is small embedding versus large embedding **plus fixed seed-input compensation**; it does not isolate embedding std alone. Initial M is paired almost exactly, yet compensation also changes learned-embedding updates entering the seed branch. Keep this distinction in causal claims.

Official Pythia410M config states layer_norm_eps1e-5 (https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json); NeoXv1.0 default agrees (https://github.com/EleutherAI/gpt-neox/blob/v1.0/megatron/neox_arguments/neox_args.py#L119). At D1200, std .0182574186 versus .006 changes expected first LN RMS .985329→.884652. This is not an effectively scale-invariant normalization at the small std.

Actual-width first-layer CPU probe: experiments/bam_llama2_medium/analyze_pythia_embedding_sensitivity.py; PYTHONPATH=MaxText:MaxText/tests, JAX_PLATFORMS=cpu, pinned /data0/xd/conda/envs/maxtext-cpu/bin/python, taskset0-7. Full D1200/L18 initialization formulas, actual first-layer widths4800/2432/5832 for MHA/Mudd/BAM, Gaussian64 unique token vectors, synthetic256-way fixed head. One parameter seed19, embedding-input seed9876, head seed600, fp32. This is not a Pile eval, trained checkpoint diagnosis, or evidence of final-loss causality. Artifact /data0/xd/bam_diagnostics/pythia-init/embedding-sensitivity-first-layer.json.

Small/large first attention RMS: MHA and Mudd .017776/.020186; BAM .0005246/.0005258. Post-attention MLP-input LN RMS: MHA/Mudd .974099/.991391, BAM .885716/.985391. MLP-output RMS: MHA .082773/.083670, Mudd .059074/.059821, BAM .080801/.091648. The same embedding change thus changes BAM's first MLP output by13.43%, versus1.08/1.26% for MHA/Mudd. BAM seed M .018594099/.018594101 is matched. Weak initial attention residual lets epsilon-dependent embedding scale reach the first MLP almost undiminished in BAM. After the MLP, its .08-level residual already exceeds embedding scale, so do not claim the embedding dominates every deep layer.

All three models have about3x larger input gradients with smaller embeddings in this probe. The actual Adam-PAX count0 unit-LR normalized-update proxy is ~1 for both scales, so relative update amplitude is3.043x larger for small embeddings. This is a common optimizer-conditioning mechanism, not evidence that smaller embeddings reduce training gradient spikes. In compensated BAM, fixedalpha=.3286335 also scales updates of the learned seed-input component; matching M at initialization does not preserve its subsequent effective learning rate.

Mudd stores raw embedding/history (mudd_prenorm=False), combines them before Q/K/V/MLP norms, and initially selects only the newest history via zero second router projection and last-history bias1. Small embedding affects later learned history mixtures; it can reduce random shortcut magnitude but also weaken direct embedding access. Therefore no robust ordering of Mudd versus MHA benefit is established. LM head is untied and unchanged, so an embedding-only change does not directly rescale its weights.

Completed TB inventory: repaired and full-original-init arms each have545 total tags, zero gradient/BAM health tags; unfixed serial has889 health tags, but differs in BAM kernels/write epsilon as well. No paired health trend can establish the mechanism. Catalog artifact initialization-health-catalog.json.

Forecast, no new training launched: only token embedding to.006 in the existing serial Pythia controls, all other initialization, widths and Mudd router budget unchanged. MHA terminal loss delta center−.006, subjective range[−.015,+.004]; Mudd center−.006, range[−.020,+.008]. Expect some common benefit, smaller than compensated BAM's observed ~.011 loss difference; Mudd has wider uncertainty, not an asserted larger gain. For unchanged repaired BAM, use terminal H=2.48550996,U=2.40909280,B=2.36197836. If both baselines improve .006, R=(H−.006−B)/(H−U)≈1.538. In general R_new=(H−h−B)/(H−U−h+u). Both absolute losses improving does not force R down: if h>R/(R−1)*u≈2.62u, R rises.
