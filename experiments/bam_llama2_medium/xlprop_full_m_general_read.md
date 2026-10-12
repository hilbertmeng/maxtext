# XLProp shared full-M generalized read transfer

RUN: `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile`.
Direct baseline: `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile`.
Medium source: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile`.
Runtime worktree `/data0/xd/mediumprop-full-m-read-gelu128`, branch `codex/mediumprop-full-m-read-gelu128`, commit `b83223f78145d44c5222f584a330ac06070a500b`.
Owned spot trainer `xd-v5p-32-310118-maxtext`, UE5a primary; no training FLEX_START request.
Compiler `llm-jax-v6e-1-0` EW4a verified FLEX_START/ACTIVE and idle, compiler-only, never reclaimed.

D1920,28x20head96,M96x40,QK72+independentRoPE24,noQKNorm,TruePile4096.
Replace shared-rank4QK/C10VO with full-M dynamicQ/K/VO keys: shared1920->400GELU down, three separate400->20x40 up; VO shares dynamic keys/pre-bias, V/O gates and static keys independent. R400 scales Medium256 by complete output20x40/(16x32), not input width alone.
Generalized effective read address: dynamic gate times.1RMSNorm(key+zero pre-bias) plus static key times static gate/.99. Static gates1.1sigmoid, zero kernels, physical initial opening.99. Dynamic gates.05; .1sqrt40=.2sqrt10 matches compressed-parent per-coordinate read amplitude under isotropic independent-M assumptions. StaticV originalGaussian, staticQ/K/Ozero.
Attention/embedding addressR400 and every-third independentMLP addressR384 unchanged; write-only generalization OFF. Writers zero-based1,4,7,10,13,16,19,22,25. Existing27+1block-scan schedule retained.

Exact tree1432386360, MHA1432398720, difference-12360(-.000863%). MLP[6140,5952,6140]x9+terminal6140; original[6294,6106,6294]+terminal6294. Before repayment new reader parameters+24783360=6.722917W_Q across28layers, W_Q=1920^2. Nearest integer per-layer repayment, no hardware rounding.

Preserve generic/concat/MLP write health and generalized static gates mean/std/threshold fractions, pre-bias/key RMS, effective static/dynamic key ratios/cosines. Actual token/head static gate adaptivity and same-token/head V/O differences use paired checkpoint probes if needed; pooled std alone does not establish token adaptivity. No RMTHealthDefaults inheritance (this is vector+matrix BAM).
PureJAX/Splash/SEQ_MINOR; M contractions retain inherited dot/dot_btn. Formal plan50000, LR2e-4, loss windows500, reports~2000steps; checkpoint250, keep4000,latest2. Directloss baseline only standardrank4; report MHA-relative and Mudd-relative gain ratios on exact common windows and align relative progress for Medium transfer.
Terminal bets vs directparent: loss-.010, matched-health speed-5%. Historical parent speed.347UE5av5p32, runtime/health differ because new generalized stats add work; label startup comparisons appropriately rather than claiming fully paired speed.

Local targeted CPU tree/layout, 3+1 scan/tail forward/active gradients, parent-equivalent generalized initialization and health export checks run concurrently with AOT and spot prequeue through launch_train_parallel.py. Artifacts `/data0/xd/bam_diagnostics/xlprop-general-read/`; run manifest `run.json`.

## Shared-read-only control

User requests independent attribution of full-M shared readers and generalized reads. Added `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu400TruePile`, same worktree/branch, runtime3425c55, ownedspotv5p32`xd-v5p-32-310119-maxtext`UE5a. Only disable generalized column read, restoreMLP[6167,5979,6167]x9+terminal6167. No pre-bias/static gates are instantiated. Exact expected count1432370680 (MHA-28040). SameR400/fullM96x40/readscale.1/staticinitialization/writeconfig/Splash/SEQ_MINOR; generic andconcathealth remainON, only generalizedextra metrics are absent. No unrelated genericCPU suite; focused fulltree/sharding and active reader forward/gradients, with unchanged27+1tail already checked by generalized launch gate.
Bets: shared-only minusstandard-.006loss/-1%speed; generalized minusshared-.004loss/-4%speed. Direct gaps use the available common horizon through32000; bothplans50000, laterMHA/Mudd comparisons remain available.

Cross-scale ratios use each scale's repairedTruePileMHA M, standardrank4 S, sharedfull-M R and generalized G: shared=(M-R)/(M-S), generalized=(M-G)/(M-R), total=(M-G)/(M-S)=shared*generalized. Thus standardrank4 is used at both scales rather than mixing MediumDirectC8 and XLrank4. Alignrelativeprogress step/plannedtotal for cross-scale interpretation; absolute same-step acrossscales is a separate display. RetainXL Mudd-relativegain ratios to MHA-Mudd. Pair generalized G against newlylaunched R aswellas S inmonitoring.

## Startup verified

Both loaded their exact topology/schedule AOT and reached actualFIRST_STEP (generalized1,shared0). Registered sourceb83223f/3425c55 and worker logs confirm R400, correct widths, generalizedTrue/False, generic+concathealthON, and UE5aTruePile4096 replica. CPU tree/sharding, active gradients and generalized284metrics passed. BothUE5a spotv5p32; no backup candidates, no preemptions atstartup. RetainedFLEX_START compiler untouched.
20-99 mean generalized.382575,shared.3941875; generalized/shared-2.945932%. Generalized extra statistics are included, so this is formal training speed, not a health-OFF architectural estimate. Historicalstandard.347 gives+10.252%/+13.599%, but olderC256 versus currentSplash/SEQ_MINOR means runtime-unmatched. Earlier-5%/-1% forecasts against the old timing cannot isolate read cost.
Medium reference finalfive12600-13400: shared/standard1.098670, generalized/shared1.036512, generalized/standard1.138787. Each per-window total is exactly the product of its two component ratios; separately averaged ratios need not multiply exactly. Full series in `medium-reference-ratios.json`.
Startup registry/evidence/manifests under `/data0/xd/bam_diagnostics/xlprop-general-read/`. Proceed with ordinary XL500windows/~2000stepreports, targeted liveness checks~10minutes, review10000; retain explicit registered directstandard and G-sharedcomparisons. Relative-progress comparisons use the completed Medium ratio curves.

## Closeout（2026-10-12）

User stopped generalized at33002 and shared at37253. Localcloseout entry verified committed checkpoints, deletion of both TPU/queued resources, and SYNC_OK on both. Artifacts `/data0/xd/bam_diagnostics/xlprop-general-read/closeout-final.json`, `gaps-final.json`; remote summary `logs/closeout-20261012T092332Z.json`.

Through last common standard32000, generalized−standard last5−.013130 and shared−standard−.008669; early improvements settled near these levels. Generalized−shared through32500 last5−.004666; an early~−.006 benefit eased but remained. VersusMudd generalized through32500 last5−.039778, gainratio1.546; shared through37000 last5−.033901, gainratio1.452. Do not compare those two terminal ratios at different progress as a component effect. At matched32000: shared/standard1.085, generalized/shared1.043, total/standard1.132, total/Mudd1.542; matchedMedium8600–8800 ranges1.091–1.094,1.031,1.125–1.128,1.646–1.681. The read improvements transfer proportionally; they have not closed the broader XL versusMedium gap againstMudd. Cold loss bets−.010/−.006 were surpassed at the available common horizon; remove bets from exp.py conclusions.

Initial20–99 kernel speeds generalized.3826/shared.3942 (−2.95%); final750 actual .2564/.3472. Whole-event GCS uploads and different event ages impair wall timing; standardC256 versus newSplash/SEQ_MINOR also prevents an architectural speed attribution. All retained health values finite.

Both UE5a SPOT v5p-32, no zone switch. Generalized: oneREADY lease2026-10-11 05:13:37→2026-10-12 09:19:32 (28h05m55), no preemptions. Shared: 05:18:52→16:02:44 (10h43m52, preempted), 16:09:24→2026-10-12 09:19:36 (17h10m12, userstop), one preemption.
