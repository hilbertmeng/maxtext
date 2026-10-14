# MediumProp read050 projection sharing

User-authorized three arms, switched before launch from original SOTA to completed
single M-shift read .5. Latest-main fork: 0f7f6d10a2760c205699cf34f9834c892b4926d0.
Worktree /data0/xd/mediumprop-projection-sharing, branch codex/mediumprop-projection-sharing.
Parent model operation restored exactly from 52cf0b3: immutable previous-token M,
columnwise sigmoid .5; mixed M only feeds reads, original M receives writes.
No KV shift, no triple read gates, no address-derived gates.

| arm / RUN suffix after BamMediumPropMShiftReadG050 | bank | MLP sharing | MLP widths | expected params | residue vs432111104 | TPU |
|---|---|---|---|---:|---:|---|
| ReadAttentionSharedR256TruePile | R256 | private R256 | 3887/3762/3887 |432111680|+576|xd-v5p-16-310187-maxtext|
| ReadAttentionSharedR384TruePile | R384 | private R256 |3772/3646/3772|432121472|+10368|xd-v5p-16-310188-maxtext|
| ReadAttentionMLPSharedR384TruePile |R384| shared only6 write layers |3794/3669/3794|432118688|+7584|xd-v5p-16-310189-maxtext|

The third arm's remaining12 layers follow arm2. Full actual parameter-tree counts
matched all three expected totals in the pinned CPU shape probe. Shared reader/attention input yields one D->R GEMM; MLP
recomputes the same weights on its post-attention residual. Distinct up projections
and all existing read/write gates, RMS and bias conventions remain.
Projection-only deltas vsread050: -3.84 W_Q / +1.3568 W_Q / +.3498667 W_Q,
where W_Q=1200². R384 head expansion exceeds down savings in arms2/3, so those
MLPs shrink. Integer-width residuals are recorded above, not claimed exact matching.

Formal training: same13500 schedule, T4096, dataset TruePile4096, scan True,
checkpoint200, max_to_keep2, keep_period0, TB rotation500. Retained FLEX compiler
llm-jax-v6e-1-0 in europe-west4-a is borrowed under its AOT lock; never deleted.
Training-only SPOT v5p16 region us-east5-a, chosen from current owned multi-hour
healthy leases and main's latest regional history. Launcher verifies IDs absent.
Direct comparisons read050 and original SOTA; gain reference MHA,200-step windows,
agent1000-step reports. No new scalar-stat families; omit inherited32-column
M-shift means/stds while retaining14 summary health metrics/layer.

Pre-run forecast after switching parent: terminalloss vs read050 +.003 / -.001 / +.001.
Speed forecast .460 / .460 / .460 step/s vs measuredread050 .458350 (~+.36%).
Arm2 most promising; sharing regularization may help, arm1 capacity compression and
arm3 conflicting read/write tasks are the expected risks. These are bets, not results.

Infrastructure audit: starts latest main0f7f6d10; train.py writer.begin_step precedes
scalars, max_utils.get_summary_writer constructs rotating writer, resolved TB500;
latest checkpoint/JIT-AOT/optimizer paths retained. Runtime writer regressions,
new budget/reuse/direct-deferred/scan/gradient/causality tests and full47 BAM suite
are required CPU gate, cached once per sealed commit across the3 launchers.

Checkpoint diagnosis originalSOTA13500 64e3fc3 on retainedFLEX: joint native-GELU
unit selection and refitted up heads, same6 MLP-write layers,128 unique sequences
96fit/32held,64 sampledpositions/sequence. Heldloss deltas read+attention R384
+.00220165 (SEM.00033778), attention+MLP R384 +.00339578 (.00043283),
all3 R384 +.01056610 (.00109684). This supports train-time validation but does not
prove read050 compatibility. All diagnostic results use capture-free paired forwards;
no checkpoint writes. Artifacts gs://newproject-1-llm_base_models_us-central1/log/bam_diagnostics/mediumprop-r256-sharing/sota13500-20261013/cohort128-r384/.

Status: exact parameter trees and full47 BAM regressions passed; new scan/gradient/causality checks passed. First preparation blocked by a test-only tracer capture bug; no training started. Corrected host callback observes actual post-attention input under remat; resealing and retrying all three gates.


All3 actualFIRST_STEP andstep14 verified at17:07UTC, runtimec9a2a360.
Final trainers310190/310191/310189; failed-gate187/188 nodes and queues allNOT_FOUND,
retainedFLEX untouched. FullsealedCPU:5 new tests+writer5+full47 passed.
Actual worker params432111680/432121472/432118688 matchfullshapeprobes; AOT loaded.
Canonical20–99 .464425/.464400/.465800step/s (80samples each),
vsRead050 .458350: +1.3254/+1.3200/+1.6254%, vsoriginalSOTA .478725:
-2.9871/-2.9923/-2.6999%. Speedbet .460 actualwithin1.0–1.26% faster.
NativeMHA-referenced gains vsRead050 andoriginalSOTA configured incommonreport;
loss200stride andagent1000 cadence. No effect claim fromstartup loss.


##1000-step review

Loss-parent sequences200/400/600/800/1000: R256 -.106711/-.046332/-.027766/
-.021484/-.015709; pairR384 -.100772/-.012836/-.004243/+.001666/+.001904;
tripleR384 -.153697/-.047373/-.029917/-.022979/-.018805.
MHA-referencegain/read050: R2561.075779/1.031195/1.028684/1.029584/1.026874;
pairR3841.071562/1.008642/1.004383/.997705/.996743;
tripleR3841.109146/1.031896/1.030906/1.031644/1.032170.
OriginalSOTAgainendpoint1.027508/.997359/1.032808.
Triple maintains~3.1–3.2% parent-normalized gain from400 while absolutegapshrinks.
Revise terminalbet parentgap R256-.004,pairR384no_positive,tripleR384-.005.
InitialmostoptimisticpairR384 forecast currentlywrong; no finalceiling claimyet.
Verifiedactualties, rank and6-layerMLP scope before interpretinginverseordering.
TBactual252newsummarytags each/no columnstats, allfinite, .5 initialgate.
At1000 meanpreviousshare.261946/.247136/.242091,parent.241360; nearclosed
fractions.144239/.124538/.147902,parent.153473; mixed/currentM_RMS
.925534/.920501/.925862,parent.921455. No obviousgatecollapse explainsloser.
Widthsconfoundattribution: R256MLP[3887,3762,3887] iswiderthantriple
[3794,3669,3794] yettriplebetter. Do notattributeallgainstoMLPrefund.
Historyalsoimportant: originalreader-onlyR384 rank increase repaid97MLPunits,
worktreefull-m-read-gelu128 runtime8dc92ee, stopped3359 withlast5SOTAgap+.003311.
This arguescapacityallocation cannotbeignored; jointsharing benefits are not
monotonicinrank. Continue all3; nextagentreport2000,do notrush1000stop.
Artifacts: mediumprop-projection-sharing/report-1000-compact.json and
shift-health-0-500-1000.json; nativeoriginal report-1000.json.


##3000 review and dominatedpairR384 stop

Parentgap/Gain: R256-.006123/1.024492, pairR384+.002778/.988889,
tripleR384-.008300/1.033201; SOTAgain1.059648/1.022823/1.068656.
Latest5parentgapsR256-.008892/-.008754/-.007988/-.007718/-.006123;
pairR384+.001005/+.001627/+.001028/+.001479/+.002778;
tripleR384-.011493/-.010699/-.010168/-.008646/-.008300.
PairR384 last5allloseparent anddeficitdeepens; triple dominateslossby.011078
at3000 withslightlyfastercanonical .4658 vs.4644 andfewerparams. Agentjudgment
stop pairR384, preserveCP/TB andresourceabsence. R256+triplekeeptraining.
Artifacts pair-3000.json (leadingmark-reported lines followednativeJSON),
triple-3000.json underlocaldiagnosticsroot.
Earlierreport mean typo correctedimmediately: triplelatest1200–2000mean1.036484
vsstandard200–1000mean1.047152 (startup200high1.109146); stable400–1000mean
1.031654. Do notclaimstandardfive-windowmeanincreased.

PairR384 closeout verified3140/CP3140, no failures, node/queueabsent,TBSYNCOK.
OneREADYleaseUE5a17:03:36–19:01:33UTC,1h57m57s,P0.

2026-10-13 combination arms: reuse codex/mediumprop-projection-sharing, audited against latest main 0f7f6d10 (unchanged infrastructure; only intentional M-shift health collector difference). TB rotation500 retained. Ranks by static three-layer scan block [256,384,256] or [256,256,256]; MLP sharing only offset1. Mixed MLP widths [3871,3746,3871], expected432119360 (+8256 vs originalSOTA); all256 widths [3915,3791,3915], expected432104480 (-6624). All new savings refunded to MLP. New runtime shape routing only, focused exact-budget/ties/scan-gradient/causality and writer tests; previous full47 passed parent c9a2a360. Direct controls: pairR256, tripleR384, singleRead050, originalSOTA; MHA gain reference. Bets terminal versus singleRead050: mixed -.006 to -.008, all256 -.005 to -.007. Mixed speed near pairR256, all256 slightly faster. Planned13500; UE5a spot v5p16 IDs310193/310194 chosen from recent long leases; retained FLEX llm-jax-v6e-1-0 EW4a compile-only, no lifecycle ownership.

2026-10-14 projection sharing final: pairR256 and tripleR384 both completed13500, runtime c9a2a360, direct final5 gap triple-minus-pair -.000141, effectively tied. SingleRead050 gaps -.003499/-.003640; originalSOTA gain ratios1.100491/1.101467; speed+.30% triple vs pair. Initial predictions +.003/+.001 vs single were wrong. Follow-up mixed[256,384,256] and all256 a445bbc8 stopped2859/2861, all CP/resource/TB verified: latest2800 normalized gain vs pair .981786/.965099, no recovery and slower, so combining layerwise did not improve. Six-write-layer R384 capacity improves over all256, but reducing other12 layers to256 also loses vs allR384. Cannot infer per-layer substitutability from whole-model results. XLsingle at10000 crossscale20%1.051370 vsMedium1.032471; known Medium terminal1.076290 is forecast anchor, expect likely crossing unless XL rebounds.

2026-10-14 failure mechanism review (user challenged insufficient analysis): full sharedR256 removes MLP-private256 from six layers; read/attention use pre-attention normalized residual whereas MLP writes use post-attention normalized residual. Hypothesis: excessive tying across purposes/input distributions, recoverable partly by384, not proven gradient conflict. Mixed failure disproves inferring layerwise substitutability from whole-model pairR256-vs-pairR384 comparison. Init M RMS four arms1.25-1.26; @1000 pairR2563.442/tripleR3843.486/mixed3.247/all2562.889, gates live; may reflect write strength/coherence/use, not causal evidence. CPU rank-selector identity audit: original pairR256 and tripleR384 versus identical per-block ranks produce bitwise-equal params/loss/grads (tiny scanned fixture). Artifact /data0/xd/bam_diagnostics/mediumprop-projection-sharing/rank-selector-identity-audit.log. Remaining rank/MLP budget/init confounds explicit. Proposed, NOT LAUNCHED: six write layers use shared128 + read/attention-private128 + MLP-private128 (union384, eachconsumer256), anchored to successful pairR256 and refund saved parameters toMLP. Discriminating diagnostics: consumer-specific Down gradient alignment plus MLP write increment/currentM alignment and functional readout contribution. No new training authorized by this discussion.
