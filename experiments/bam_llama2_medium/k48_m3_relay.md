# K48 shared-rank4 M3 anchor relay

Implementation: `/data0/xd/llf-parameter-matched`, `codex/llf-parameter-matched`.
Parent `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer` (73f2e77):24layers,LLF8,M48x32/C8,QK48concat,NoPE48/RoPE16,
sharedrank4QK/staticQK,sharedC8VO independentreadgates,columnonly,MLP3050/3050/3045.

New runs:
- `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3`: one tanh anchor coefficient/layer, QKVO shared.
- `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3QKVO`: two coefficients/layer, QK versus VO/FetchedO.

Capture differentiable M after first LLFblock; firstblockunmodified,outside7-blockscan.
Each subsequentlayercomputes tanh(xW+b), zero kernel/bias. Reads useM+sA;
writes retain originalM. StaticanddynamicQKread the sameQKmixture; VOsharingremains.
FetchedO mixes source-position-scaled anchors using destinationlayercompression.
Noadditionalfetchcontraction or persistentautoregressiveM-cache; extraactivationcarry.
Map original8-block initialization into firstblock+7blocks beforeoptimizerinit.

Actualtotals parent411885440;single411906965(+21525,.02053W_Q,.00523%);
dual411928490(+43050,.04106W_Q,.01045%). KeepMLPwidths; differencesnegligible.
Readhealth968retained; relay147/294additionalmetrics (21layers*7stats*1/2arms).
Matchedrelay-onlyspeedcontrolnotplanned; reportmeasuredspeedwithhealthcounts.

Active bet vsparent:single+.002[-.003,+.007].
Speedbeforetelemetrysingle~-1%,dual-1..-2%. HistoryK32fullM3-.00673,partialM3+.02339@3200,
K64Vonly-.00342terminal,threegates+.00394@5000; independentgatesarenotaguarantee.

Plan13500,checkpoint200,report1000steps,review2800. Bothcompareparent;dualalsocomparessingle.
FreshUE5av5p16trainers`xd-v5p-16-k48-mrelay-m3-maxtext`and
`xd-v5p-16-k48-mrelay-m3-qkvo-maxtext`;UC1a/EW4bbackups.
AOTv6eEW4aprimary,UC1a/UE5abackups. ExistingXLjointandAllLocalremainmonitored.

Validation:57 BAMregression testsPASS377.692s; anchorcapture, parentparameter/zero-outputmapping, nonzerogateQK/VO/fetchroutingandunmodifiedwrite-source checksPASS. Actual fullsizeparams/shardingPASS; fulltraintrace968readhealth+147/294relayhealthPASS. Artifacts `/data0/xd/k48-mrelay-{audit.json,audit.log,regression.log,mapping-test.log,routing-test.log,trace.log}`.

Single launch verified 2026-09-22 UTC: runtime35d4878, UE5a v5p16, AOTloaded, step0 LR0 and FIRST_STEP19. Steps10–14 .6310/s versus parent .6378/s (-1.07%); health1115 versus968. Dual launched UE5a04:53:58 UTC; AOTloaded, step0 LR0 and FIRST_STEP4 verified. Steps10–14 .6248/s (-2.04% vs parent; -.98% vs single), health1262. Both registered exactruntime35d4878 and intendedcomparators. Singlecheckpoint400committed.

## Learned scalar amplitude follow-up

RUN `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3LearnedScale`, same worktree/branch; fresh UE5a v5p16 `xd-v5p-16-k48-mrelay-m3-scale-maxtext`, UC1a/EW4b backups.
Read `M + s_l*tanh(xW+b)*A`, one unrestricted scalar per destination layer initialized1; zero-init dynamic gate and parent initialization mapping retained. First LLF block unchanged. Scalar named `m_relay_amplitude_scale` follows existing no-weight-decay scale rule. Adds21 parameters versus single relay, no MLP adjustment.
Health adds per-layer amplitude_scale, effective_scale_mean, effective_scale_abs_mean; delta_over_m and mixed_over_m use actual scaled contribution. Original scale_mean remains raw tanh gate, preserving comparisons. Total968read+210relay=1178 BAM scalars.
Direct baselines single relay and originalK48. Bet terminal -.001 versus single (range-.004..+.002), speed approximately unchanged before additional63health metrics. Plan13500/ckpt200/report1000/review2800.

Scale validation:57 BAMregressionsPASS380.393s;3relaytestsPASS143.834s (parentmapping,init1/zeroidentity,nonunitamplitude/readrouting/write-source). Actualparams411906986,shardingoverhead.16691%;fulltraintrace968+210healthPASS. Scalar storedshape(1,) for param_scan_axis1 compatibility. Artifacts `/data0/xd/relay-amplitude-{tests.log,regression.log,audit.json,trace.log}`.

Scale launch: runtimee46038cd851182e3fc7fe47a17cc8c1efa5a06f9, UE5a05:56:13 UTC, AOTloaded/from0LR0/FIRST1 verified. Steps10–14 .6318/s (+.13% versus single .6310; -.94% versus originalK48 .6378), health1178 versus1115/968; approximate speedparity, not matchedtelemetry.

## VO-only relay ablation

RUN `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3VOOnly`, same worktree/branch. FreshUE5av5p16 `xd-v5p-16-k48-mrelay-m3-vo-only-maxtext`,UC1a/EW4bbackups.
Single zero-init tanh gate per affected layer; only LocalVO and FetchedO read `M+gA`; static/dynamic LocalQK read originalM. No learned amplitude; unchangedMLP,params411906965(equal single),health968+147=1115. Source-token gate beforeFetchedOcompression unchanged.
Comparators single relay, dual relay, originalK48. Bet terminal-.002 versus single [-.005,+.002], speedapproximatelyunchanged; tests whether allowing QK to add/subtract anchor is counterproductive. Plan13500/ckpt200/report1000/review2800.

VO-only validation:57regressionsPASS385.333s;3relaytestsPASS164.887s covering all4variants, nonzeroQKoriginalM/VOmixedM and unchangedwrite-source. Fullshapeparams411906965 and shardingPASS;fulltraintrace968+147healthPASS. Artifacts `/data0/xd/relay-voonly-{tests.log,regression.log,audit.json,trace.log}`.

Dual relay stopped2937, finalcheckpointcommitted, TPU/queueabsent, TBsyncOK. Through2800 vsK48 last5-.000130[-.001974,+.001437], earlygainlost and latest+.001437; vssingle last5+.003452[+.002746,+.004266], consistentlyworseafter400. No efficiencygain. User-authorized2800reviewcriterionmet. Source-token negativeQKgates(especiallyL9/L12)were realandstrengthening, notaninitializationfailure.

VOOnlylaunch: runtime862bdb315b247583a0e5ad62b10b7f91c40b6eb9,UE5a06:32:12UTC,AOTloaded/from0LR0/FIRST0 verified. Steps10–14 .6288/s (-.35% versus single .6310 matched1115health; +.64% versus dual .6248 health1262; -1.41% versus originalK48 .6378 health968). Allthreecomparatorsregistered.
