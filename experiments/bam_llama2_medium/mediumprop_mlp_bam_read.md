# MediumProp BAM reads into SwiGLU hidden channels

User-authorized three training runs, 2026-10-15. Worktree `/data0/xd/mediumprop-mlp-bam-read`, branch `codex/mediumprop-mlp-bam-read`, derived from current main `c5e02ad86121a7c57c3d859168f2d9c68bd2c68a`. Main exp.py is ledger-only for these arms.

Parent: `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile`, not the newer read/attention-write shared-down variant. D1200, L18, n16, M75x32, original SwiGLU/RMSNorm recipe, single read-only M-shift .5. Attention read Q/K/VO share R256; attention and MLP write down projections remain independent. Direct comparisons retain this parent and `BamMHAMediumPropC256TruePile`; gain ratio is new/parent with MHA reference.

## Circuit and ordering

Each selected MLP obtains eight full-K75 column reads, flattened to 600 hidden channels. Query is the existing post-attention normalized vector residual. Read the actual carried M after this layer's attention write and before this layer's MLP write, per explicit user clarification. Existing attention M-shift remains read-only; no additional temporal mixing is applied to the MLP's post-write matrix.

`hidden = concat(x @ W1, Read(M_after_attention, x)) * SiLU(x @ Wg); y = hidden @ W2`.

The new memory channels have Wg/W2 but no W1. An independent D1200->R256 GELU->8x32 reader retains the parent's dynamic key RMS epsilon1e-4, key scale .1, pre-RMS bias, static key initialization, dynamic gate opening .05 and compensated static gate opening .99. Separate dynamic/static gates control their relative content, while Wg gates the combined channels; no additional common output gate. A fine-grained Wg does not algebraically absorb two independent gates preceding summation. Both address terms are combined before ONE contraction with M; CPU compares against the two-read formula in f32/bf16.

In the six deferred-write layers, materialize `lambda*M + delta_attn` for this new read and retain the already-computed delta_attn. Final carry remains `lambda*M + (delta_attn+delta_mlp)` with the parent's operation order; do not repeat the attention outer product or decay twice. In other layers the existing post-attention M is already available.

## Exact per-layer budget and dominant computation

Parent widths [3802,3676,3802], parameters432111680. Zero-based MLP write layers1/4/7/10/13/16.

| location | RUN | total hidden widths | ordinary W1 widths in selected slots | params vs parent |
|---|---|---|---|---|
| write6 | BamMediumPropMShiftReadG050MLPReadWriteLayersTruePile | [3802,3767,3802] |3167|+384|
| nonwrite12 | BamMediumPropMShiftReadG050MLPReadNonWriteLayersTruePile | [3893,3676,3893] |3293|+768|
| all18 | BamMediumPropMShiftReadG050MLPReadAllLayersTruePile | [3893,3767,3893] |3293/3167|+1152|

Reader392464 parameters=.27254444 W_Q per selected layer: down307200, up65536, gate kernels19200, pre-bias256, static key256, gate biases16. Replacing600 W1 channels saves720000=.5 W_Q. Refund remaining savings into91 TOTAL hidden channels, costing327600=.2275 W_Q, leaving+64 parameters per selected layer. Budget is refunded in the same layer, not distributed elsewhere. No hardware width rounding.

Per selected layer/token: reader dense391936 MAC + single matrix read19200 MAC + MLP change(-392400 MAC) =18736 MAC=37472 forward FLOPs. Tiny elementwise additions, norms, activations and gates are omitted in conventional dominant-matmul accounting. The attention write is reused, not charged twice.

| selected layers | forward extra FLOPs/token | approximate whole-model increase | forward+backward extra FLOPs/token, 3x approximation |
|---:|---:|---:|---:|
|6|224832|.02444%|674496|
|12|449664|.04888%|1348992|
|18|674496|.07332%|2023488|

Denominator: actual parent dense-kernel parameter tree371468288 MAC/token (embedding lookup excluded, unembedding included), plus causal QK/PV88495200 MAC/token atT4096, total919926976 forward FLOPs/token. Existing parent BAM contractions omitted from denominator, making percentages slightly conservative. Rematerialization and kernel scheduling are not captured; report actual speed independently. Full parameter tree audit artifact `/data0/xd/bam_diagnostics/mediumprop-mlp-bam-read/parameter-audit.json`.

## Predictions, checks, execution

Pre-run terminal new/parent gain-ratio bets (MHA reference): write6 **1.08 [.98,1.18]**, nonwrite12 **1.12 [.98,1.25]**, all18 **1.15 [.97,1.30]**. Primary hypothesis: more consumers and nonlinear M processing help the upper bound; extra channels on nonwrite layers distribute the new path rather than concentrating M->MLP->M feedback. All18 has the most potential and greatest interference/initialization risk; ranking is a prediction, not evidence. Expected throughput changes approximately-2%/-4%/-6% versus health-off timing control `BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile` (.4841625 step/s). This same-shape control is used only for timing; retain the original single.5 parent as the direct loss baseline.

Focused checks: full parameter shapes and selected-layer budgets; interception verifies actual post-attention query/M and single decay including lambda=.73; finite scanned gradients and causality in all three arms; one-contraction equivalence; full47 BAM regressions; affected Pythia MLP initialization/bf16-remat/NeoX parity; five summary-writer checks. Initialization-only full-width actual-layer amplitude probe uses three seeds and synthetic Gaussian embedding rows; it does not establish trained loss or a best initial scale. No new training health metrics.

Infrastructure audit: derived from latest mainc5e02ad8, training/AOT optimizer-WD/checkpoint/logging paths retained. Verify TB rotation500 with writer begin_step; CP200/latest2/keep_period0, plan13500, T4096 globalbatch128, matched Pile data. Spot UE5a v5p16 selected from current successful same-zone leases. Borrow retained FLEX_START compiler llm-jax-v6e-1-0/EW4a; AOT lock serializes compilation, CPU and three training prequeues run concurrently. No compiler lifecycle ownership. Report all three as a group every1000 steps, parent/MHA loss and gain-ratio sequences, verified speed; no routine M-shift health synchronization.

Historical audit: August23 `BamLlama2MediumV2C256MlpWriteR128` moved MLP hidden INTO M, the opposite direction. RMT read into MLP input and LocalO residual/input proposals are different circuits. User confirmed the hidden-read proposal has not previously been tested.

Full18 initialization probe (3 seeds, 32 synthetic token rows/seed): mean read/W1 RMS ratios L0/L1/L2/L8/L17 =2.005/2.837/4.126/6.971/10.128; last-layer vector RMS1.009. All finite. These memory channels are not amplitude-matched to ordinary W1 channels; preserve native read scale in this first location comparison rather than silently introducing a new normalization rule. Artifact init-amplitudes.json. The temporal/data dependence and depth accumulation are architectural differences from ordinary W1.


### FIRST_STEP and actual-worker audit

All three running from sealed runtime3afcaa9e8d0efaf15b66dc4355b145cf49ed1ae5 on UE5a v5p16: write6 TPU310246, nonwrite12 TPU310247, all18 TPU310248. CPU59 checks passed across new-path4, affected-Pythia3, summary-writer5 and BAM47; final focused launch gate reuses one fingerprint across three scripts. AOT loaded and FIRST_STEP verified for each. Actual full worker parameter trees148/156/164 leaves sum432112064/432112448/432112832, exactly matching local audit. Actual TruePile4096 UE5a dataset, T4096/per-device16(global128), CP200/latest2/keep0/TBrotate500, single read.5, all bam_record_* plus generic health OFF verified. No extra metrics.

Canonical20–99 speeds, all80 steps present: write6 .4780875, nonwrite12 .4687750, all18 .4575625 step/s. Versus health-OFF timing control `BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile` (.4841625 step/s), which has the same matrix multiply shapes and costs: -1.25%/-3.18%/-5.49%. It is not a replacement loss baseline. These actual overheads are much larger than .024%/.049%/.073% dominant-FLOP increments, consistent with added small ops/state dependencies; no kernel-level causal attribution established. Bets -2%/-4%/-6% roughly matched. Current loss is finite and falling; first formal group comparison at common1000, no verdict from startup loss. Artifacts worker-config-verification.json, speed-verification.json under local mediumprop-mlp-bam-read diagnostics.

## Common 1,000-step report (2026-10-15)

Actual progress write6/nonwrite12/all18:1076/1116/1083, committed CP1000 each. Gain multipliers use original single.5 as comparator and MHA as reference; values taken directly from registered report, not recomputed from raw loss.

|step|write6 gap|write6 R|nonwrite12 gap|nonwrite12 R|all18 gap|all18 R|
|---:|---:|---:|---:|---:|---:|---:|
|200|-0.139044|1.09874|-0.073807|1.05241|-0.156299|1.11099|
|400|-0.044189|1.02975|-0.044727|1.03011|-0.067108|1.04518|
|600|-0.014623|1.01511|-0.027268|1.02817|-0.034160|1.03529|
|800|-0.006174|1.00850|-0.014963|1.02060|-0.018081|1.02490|
|1000|-0.003187|1.00545|-0.009198|1.01574|-0.012516|1.02141|

All18 > nonwrite12 > write6, but each extra gain is contracting. Nonwrite12 delivers roughly three quarters of all18 extra loss improvement at1k with lower measured overhead (-3.18% versus-5.49% health-OFF timing control). Continue all three; next group report2000. Original terminal bets remain historical predictions, not supported by the early declining ratios. Full parent/MHA cumulative gaps and r200 in report-common1000.json.

## Ungated all18 replacement, user authorized2026-10-15

Class `BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesTruePile`, same family worktree/branch. Derive gated all18, remove BOTH dynamic and static read gate projections/biases entirely, retain dynamic address RMS/pre-bias/fixedkey_scale.1 and raw statickeys. No fixed.05 multiplier replacement. Static parent gate was compensated to unit initial gain; remove its whole compensated factor, so static initialization amplitude stays unchanged. Dynamic component grows20x relativeparent opening.05, not the whole readout. Existing MLP SiLU Wg and W2 retained for all600 memory channels. Query remains MLP norm after attention, M from actual attention write, singleMcontraction and decay/write ordering retained; no newhealth metrics.

Readerparams373248/layer versus392464parent; saved19216/layer(345888total=.2402W_Q). Refund+5 TOTAL MLP channels per layer over gatedall18: widths[3898,3772,3898], expected432090944parameters, -21888versusgatedall18 and -20736versusoriginalsingle.5 (integer-rounding remainders). Ordinary W1 widths3298/3172/3298. Dense/Mread net17536MAC perselectedlayer/token=35072forwardFLOPs, all18 net631296FLOPs/token (+.0686246% versusoriginalsingle.5 dominant computation), slightly less than gatedall18extra674496. Parameter refund consumes most saved dense MACs; removal of small gate operations may improve actual throughput.

Pre-run new/all18 terminal gain(MHA reference)1.03 [.97,1.10], speed+1% [0,3%] versushealthOFF gatedall18. Direct comparisons all18directionparent, originalsingle.5, MHA; report ratio sequences versusboth BAMcomparators. AllhealthOFF, T4096/global128, plan13500/CP200/latest2/TBrotate500. Auditlatestmainc5e02ad8 unchanged operational paths since verified3afcaa9, no missing runtimefixes. CPU exact tree, ungatedformula f32/bf16 plus sameweights gatedreference, scannedfinitegradient/causality and postattention/singledecay checks; parentfour regressions and summarywriterlaunchgate.

Hot-replace only-write6 `BamMediumPropMShiftReadG050MLPReadWriteLayersTruePile` on retainedTPU310246/UE5a after CPUgateandnewAOT are ready; do not stop source during compilation. NewRUN starts fresh parameters/data/schedule. Usehot_switch_run.py ownership/checkpoint/worker-exit gate; preserveoldCP/TB, neverdelete this physicalTPU througholdregistry afterhandoff. BorrowexistingFLEXcompiler only, serialized behindthetwo shared-write normAOTs. Other two original read arms continue.

## User correction: zero static addresses, fresh RUN

`BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesStaticZeroTruePile` replaces the briefly started normal-static ungated RUN; no loss conclusion from that temporary runtime. Static[32,8] keys initialize exactlyzero in every MLP read. Dynamic down[1200,256]/up[256,8,32] remain iidnormal(.006), GELUhidden, pre-RMSbias0, addressRMS epsilon1e-4/key_scale.1. Thus only static read startszero; entire reader remainsnonzero. Wg/W2 retained, all18 selected, post-attention M/read ordering same. Statickeys remaintrainable; verify nonzero gradient from step0 and exactly unchanged other initialization leaves for equal seeds. Counts432090944 and widths[3898,3772,3898] unchanged. Old gatedall18/nonwrite12 controls retain their original nonzero static initialization.

User correction arrived after temporary f6137cc RUN reachedFIRST_STEP; initialization changes require a new freshRUN/data cursor, not resuming its checkpoint. Borrow FLEX to compile exact new runtime, then hot_switch from temporarynormal-static to staticzero on sameTPU310246. Preserve both old checkpoints/TB without deleting reusedphysicalresource. Direct effect control remains gatedall18 (+ originalsingle.5/MHA); temporary shortRUN is not an effectiveness baseline. Before-start forecast new/gatedall18 MHA-reference terminalgain1.05 [.98,1.12], speed+1% [0,3%]; improvedinitial amplitude is a hypothesis, not observed benefit. Currentlatestmainc5e02ad8 infrastructure audit unchanged since prior gate; three initializer-specific tests plus five summary-writer checks, earlier nongated13-path checks preserved.

## Common2k and write6 replacement closeout

Nonwrite12/all18 actual2071/2029, committedCP2000. Againstoriginalsingle.5 at1k/2k: nonwrite12gap-.009198/-.003750, MHA-reference gain1.01574/1.01173; all18gap-.012516/-.003344, gain1.02141/1.01046. All18firstslightlybehind12at2k(.000406). Both earlieradvantagescontracted to~1%; terminalbets1.12/1.15 currentlyunsupported, continue3k. Canonicalsamehealthtimingcontrolchanges-3.18%/-5.49% retained. Full200–2000report-common2000.json.

Write6userhot-replacedatCP2008, no full2kwindow(samplesafter2020unavailable). Lastnative1800parentgap-.0001938/gain1.000556; early200gap-.139044/gain1.09874 fadedtoessentiallytie. Fullcloseoutwrite6-hot-final.json. UE5a v5p16 TPU310246, P0, assignment09:35:05–10:48:22UTC; singleREADYlease09:35:09–10:48:22UTC(1h13m13s), hotreplacement, no regionchange. CP2008committed and TBSYNC_OK; physicalTPU retained/reassigned, neverdelete via oldregistry. OldRUNdeclaredstopped after successorFIRST_STEP verified. This is a user replacement, not a terminal failure verdict.

## Current status: static-zero successor running

Frozenruntimec560cd62ce2d8563b56891655bdc3c7d47659528, `BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesStaticZeroTruePile`, FIRST_STEP/AOTloaded/actual432090944params verified. Worker flagsstaticinitzero, readgatesFalse, fullhealthOFF, widths[3898,3772,3898], T4096/global128/TruePile4096, CP200/latest2/keep0/TBrotate500, keyepsilon1e-4/scale.1. Threezero-init-specificchecksplusfivewriterpass; paired3seedreaderinitmatchesallotherparameters exactly, staticgradnonzero, dynamicreadnonzero; scannedfinitegrad/causalitypass. Canonical20–99(80/80points).4629875step/s, healthOFF gatedall18.4575625reference +1.18563%, matches+1%prediction. New/all18terminalbet1.05[.98,1.12] preregisteredinsealedruntimebeforeAOT; noformaltrainedlossverdictuntil1k. Originalgated12/all18review2800 peruserconditionalstopauthorization.

Temporarynormal-staticungatedf6137cc successorstopped228/CP228, TBSYNC_OK, solelyuserinitializationcorrection, notarchitecturalfailure. Itscanonical.463075step/s +1.20475%versusgatedall18 recordedonlyasstartupreproduction. P0/UE5av5p16 retainedTPU310246: READY10:49:26–10:59:16UTC (9m50s), hotreplacement. Zero-staticnewRUNstartsfrom0, notfromtemporarycheckpoint. Oldwrite6hasCP2008andTBSYNC_OK; nevercloseout/deletephysicalTPUviaeitheroldregistryafterhandoff. Artifacts static-zero-worker-verification.json/static-zero-speed-verification.json/static-zero-state.json andnogates-temporary-final-registry.json.


## Original gated12/all18: authorized2800 review closeout

Stoppedaftercommon2800review, finalcommittedCP2970(nonwrite12)/2909(all18). Nodes+queues explicitlyverifiedabsent, finalTBsyncaftercloseoutbothSYNC_OK. At1k/2k/2800 parentgaps12-.009198/-.003750/+.0007348 andall18-.012516/-.0033438/-.0001364. MHA-reference ratios12:1.01574/1.01173/.997179, all18:1.02141/1.01046/1.000524. 12crossedbehind2400 and stayednegative; all18extraadvantagecontracted tonearzero. Earlyconvergenceaccelerationdidnotbecomesustainedgain. Bets1.12/1.15wereoveroptimistic; thisisnotafullterminalmeasurement. Zero-static/no-read-gates successorcontinues; bothclosedRUNsremainregisteredhistoricalbaselines, availablethrough2800.

UE5av5p16 P0 both: nonwrite12READY2026-10-15 09:32:57–11:21:51UTC(1h48m54s), all18READY09:32:36–11:21:54UTC(1h49m18s); nozonechange. Fullsummarygated-2800-closeout-summary.json, remote logs/closeout-20261015T112447Z.json. Fullreview200-stepseries report-review2800.json, finalTBproof gated-2800-final-tb-sync.log.

Parameter-flowinterpretation: larger total hiddenwidth isnotlargerordinaryMLP parametercount. Gatedreaders replace392400ordinaryMLPkernelparamswith392464readerparams perselectedlayer; ungatedreplaces374400with373248. Write-downsharing insteadrefundsitsremovedkernel intocompleteordinaryMLPchannels. These aredifferentcapacitytrades; assess theirconvergenceindependently.

## Zero-static ungated reader: first1k

Actual1103, CP1000 committed. Versus gatedall18 at200/400/600/800/1000 gaps+.053885/-.001054/-.000918/-.004377/-.004948, MHA-reference ratios.965557/1.000679/1.000916/1.005881/1.008287. Turns ahead400, recent benefit increases. Versus originalsingle.5 gaps-.102414/-.068162/-.035078/-.022458/-.017464, ratios1.072728/1.045893/1.036237/1.030926/1.029876: total advantage stillcontracts, so lack-of-late-benefit not yet fixed. All18closedbaseline remainsavailablethrough2800; reference seriesbeyonditsstop must remainmissing. AllhealthOFF speed+1.18563percent(.4629875 vs.4575625). Pre-run new/all18 terminalbet1.05[.98,1.12] kept; continue2k–2800. Fullreport report-common1000-static-zero.json.

## New ungated static-zero nonwrite12 arm (2026-10-15)

User requested the same zero-static/no-read-gates design only on twelve non-MLP-writing layers. Class `BamMediumPropMShiftReadG050MLPReadNonWriteLayersNoReadGatesStaticZeroTruePile` inherits the active zero-static all18 arm, overrides location='nonwrite' and widths[3898,3676,3898]. Six MLP-write layers retain the original no-reader width3676; all chosen layers keep8x75memory channels, Wg/W2, post-attention-written M, nonzero .006 dynamic addresses, zero learnable static addresses. No model implementation change. Expected parameters432097856, originalsingle.5-13824 due local integer rounding(-1152/selectedlayer); versusoldgated12 -14592. Reader373248/selectedlayer, no health stats. Forwardextra420864FLOPs/token(+.04575% dominantfullmodel); no extra attention-M materialization on the six excludedwrite layers.

Pre-run bet versusoldgated12 MHA-reference terminalgain1.05[.98,1.12], speed+1%[0,3%]; expect near newungated18 effectiveness with lower runtime overhead, not guaranteed. Keep originalsingle.5 and newzero-static18 as additional direct baselines, each MHA-reference. Oldgated12 comparison available only through2800; do not truncate remaining primary comparisons later. Full13500schedule/T4096/global128/TruePile4096/CP200/latest2/keep0/TB500/allhealthOFF.

Operational audit current mainc5e02ad8: train/train_compile/max_utils/maxtext_utils/optimizers/checkpointing/summary_writer/base.yml/pyconfig.py byte-identical. TBrotate500/begin_step present, health compute guard and checkpoint retention already current. Preserve frozen model lineage. Reuse familyworktree `/data0/xd/mediumprop-mlp-bam-read`, branch `codex/mediumprop-mlp-bam-read`. Newspotv5p16TPU310251 inUE5a, currenttask recent leases allstable/P0; no new backup-zone policy. RetainedFLEX llm-jax-v6e-1-0/EW4a verifiedACTIVE/FLEX_START (expires2026-10-16T01:56:30Z), borrowAOTonly, neveradopt/delete. FocusedCPU3 tests full/localbudgets+selectedorder+scangradcausality, plus5summarywriter; parent initializer and module unchanged/alreadytested. Runtime andFIRST_STEP evidence pending.

### Nonwrite12 zero-static startup verified

Frozen runtimeae19f928c0de0f5ba53d40b22e421d3a728f83e2, UE5a xd-v5p-16-310251-maxtext. CPU8pass(3focused+5summarywriter), SEALED_CONFIG/AOT/FIRST_STEP8verified; actualworker148leaves432097856parameters, widths[3898,3676,3898], locationnonwrite, staticzero, gatesFalse, all16healthflagsFalse. T4096/global128/perdev16, correctzone-localTruePile4096, CP200/latest2/keep0/TB500 verified. No resume/newfrom0. Fullconfigandshapes nonwrite-static-zero-worker-verification.json, launch/AOT/CPU state undernonwrite-static-zero-state.json and localparallel-launchdirectory. RetainedFLEX untouched.

Canonical20–99 all80points .4709125step/s, versus samehealthOFF gated12 .468775 +.455976%, versuszero-staticungated18 .4629875 +1.711709%; forecast+1%[0,3%] withinrange. Allcomparisonshealthmatched; no oldhealthON timing. Three persistent MHA-reference gain triples new/oldgated12, new/zero-static18, new/originalsingle.5 registered. Firstnormal1kreport, groupwith18arm; no earlylossverdict.

## Ungated zero-static group: all18 common2k

All18actual2040/CP2000; newnonwrite12actual321/CP200 verifiedcommitted. At1k/1.2k/1.4k/1.6k/1.8k/2k versusgatedall18 gaps-.0049478/-.0044452/-.0033188/-.0043708/-.0055308/-.0016652, MHA-reference ratios1.008287/1.009029/1.007386/1.011389/1.015752/1.005154. Latest5mean-.00386616/R1.009742. Versusoriginalsingle.5 gaps-.0174636/-.0151864/-.015461/-.0099098/-.0083402/-.005009, ratios1.029876/1.031536/1.035364/1.026200/1.023945/1.015666; last5mean-.01078128/R1.026542. Totalgainpeaked1.4k thencontracts3windows, removalofgates/zerostatic hasnotreversedoldlateproblem. Stillmodestlybetterthangatedall18. Continue2800review; noearlyeffectverdictfornew12, firstformal1k. Speedunchanged canonical+1.18563percentagainsthealthOFFold18. Fullnativehistory report-common2000-static-zero.json.

## New reader group: all18 3k / nonwrite12 first1k

Actual3054/1065 CP3000/1000committed. All18 originalsingle gaps2k/2.2k/2.4k/2.6k/2.8k/3k -.005009/-.0070356/-.0050196/-.0057454/-.0050662/-.0036138; MHA-reference ratios1.015666/1.023045/1.017504/1.021068/1.019447/1.014456, last5R1.019104. Oldgated18 lastcommon2800 gap-.0049298/R1.018914; unavailableafter2800, notinterpolated. Improvementstillmodestbutdidnotvanishlikeoldgatedarms; continue4k.

New12at200/400/600/800/1000 oldgated12gaps-.0041004/-.0100522/-.0042524/-.0054968/-.0052888, ratios1.002767/1.006570/1.004273/1.007417/1.008908; new18gaps+.0245064/+.0133832/+.0035572/+.0019982/+.002977, ratios.983777/.991385/.996454/.997331/.995055. Originalsingle gaps-.0779078/-.0547788/-.0315206/-.0204596/-.0144866, ratios1.055325/1.036882/1.032562/1.028174/1.024783. Old18/new18new12effectbothimprove~.005versusownoldgated1k, cannotattributecombinedchange solelytogateremoval. New12losesslightlyto18butfaster1.71%, continue2k. FullnativeMHAgapsandallratios report-common3000-1000-static-zero.json.

Betobservabilitycorrection: preregisteredterminalnew/oldgated1.05 cannotbeverifiedat13500 becauseoldgated12/all18stopped~2800. Preservehistoricalbets, nevercallcommon2800ratios terminalverification. Subsequentterminalcomparisonsusecompletedoriginalsingle.5 andactivezero-static12/18 peers. No retroactive numerical pre-run bet.

New12firstpreemption2026-10-15T12:32:05Z, firstREADY12:01:02–12:32:05(31m03s), CP731committed. Resumed732 sameae19f92/AOT/13500schedule, CP800then1000committed. First31minleasepartofsameUE5acorrelatedeventasoriginal-init/sharedwriteA, notanindependentzonefailuresample.

## All18 4k / new12 preempted atCP1956

All18CP4000committed actual4053. Originalsingle3k/3.2k/3.4k/3.6k/3.8k/4k gaps-.0036138/-.002998/-.003742/-.0047234/-.0036068/-.0015946, MHA-reference1.014456/1.012496/1.015716/1.020766/1.016233/1.007336. Last5gap-.00333296/R1.014509 versus3kR1.019104, extracapabilitystillcontracts. Improvementoveroldgateddoesnotyetalterearly-gain/late-contractionpattern. Continue nextreview, notcalledbreakthrough.

New12secondpreemptionCP1956 before2000window, oldruntime/scheduleautomaticallyrecovering; lastcomplete1800. Originalsingle1k/1.2k/1.4k/1.6k/1.8kgaps-.0144866/-.0135726/-.0140698/-.008169/-.0042818, ratios1.024783/1.028185/1.032182/1.021598/1.012293. New18gaps+.002977/+.0016138/+.0013912/+.0017408/+.0040584, ratios.995055/.996751/.996927/.995515/.988621. Doesnotyetcatch18; continue2kafterrecovery. LastMHAgap-.3525884, oldgated12gap-.0026612/R1.007605 at1800. No invented2000window. Fullsource report-common4000-preemption1326.json.

## New12 common2k after second recovery

CP2000committed actual2043, all18actual4422/CP4400(lastnormalreport4k). At1k/1.2k/1.4k/1.6k/1.8k/2k originalsingle gaps-.0144866/-.0135726/-.0140698/-.008169/-.0042818/-.0056258, MHA-reference1.024783/1.028185/1.032182/1.021598/1.012293/1.017595. New18gaps+.002977/+.0016138/+.0013912/+.0017408/+.0040584/-.0006168, ratios.995055/.996751/.996927/.995515/.988621/1.001899. Firstsinglewindowturnsahead2k, notstableovertake; ifnearparitypersists12's1.71%timingedgecouldmatter. Oldgated12at2k gap-.0018758/R1.005799. Continue3k. Source report-common2000-nonwrite-static-zero.json. Actualresume1957 fromCP1956 verified; full13500schedule/AOTretained. Newlydeployedwait-stepcached2kwindowinthesameSSHsnapshot, noTBrepairneededforthiswindow.
