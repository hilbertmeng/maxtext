# XLProp single read-only M-shift .5

Family worktree /data0/xd/mediumprop-m-shift, branchcodex/mediumprop-m-shift,
latest-main fork0f7f6d10 unchanged. Parent actual architecture
BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile,
historical runtimeb83223f, D1920,L28,K96,V40,sharedreadR400, attentionwriteR400,
MLPwriteR384 at9layers, shared VO generalized column reads, original D-input gates.
NewRUN BamXLPropSharedR400GeneralReadMShiftReadG050TruePile. Uses already-verified
single read050 mechanism with one v40 sigmoid previous-token gate: zero1920x40
weights and zero40bias, initial .5. Mix immutable layer-input states, packed-segment
and position boundaries protected, readonly; carry and writes retain original M.
No KV/triple/two-lag or projection-bank sharing. Five summaries/layer,140 new health
scalars total; no per-column stats. Parent health otherwise retained.

28gates cost2151520=.5836372W_Q (W_Q=1920²); kernel2150400,bias1120.
RefundMLP parent[6140,5952,6140]+terminal6140 to[6127,5938,6127]+terminal6126.
374total width units *3*1920=2154240; newtotal1432383640 vsparent1432386360,
residue-2720(-.00019% parent), vsMHA1432398720 residue-15080(-.00105%).
Full actual parameter trees verify both totals; nine3-layer scanned blocks and
separateterminal layer contribute28gate kernels, including terminal read gate.
Parent optimizer rules retained; exclude ONLY new M_shift_gate_b0 from WD to match
Medium donor gatebias convention. Do not broadly exclude existing XL gate_b0 names.

Infrastructure audit latest committed main0f7f6d10: max_utils/maxtext_utils,
train_compile/optimizers/checkpointing/base.yml diffempty. train.py onlyfamily health
formatter additions, TB begin_step and rotating writer retained. ExplicitTB500,
checkpoint250,keep_period0,max_to_keep2,original50000 schedule,lr2e-4,warmup1%,
T4096, batch8/device,TruePile4096,scanTrue,Splash and pureJAX core.
CPU scope focused because model/shift/optimizer code unchanged: fullparameter
budget,v40mix/rawcarry/boundaries,3+terminalscan finite gradients and causality,
compacthealth initialization,writer5,stagealignment tests. PriorfamilyfullBAM47
passed; no unnecessary repeated all-arm training tests for this config-only port.
Sealedconfig andAOT on borrowedFLEXllm-jax-v6e-1-0EW4a,formalSPOTv5p32UE5a
TPU xd-v5p-32-310192-maxtext (launcher verifiesabsent),directbaselineparentSOTA,
MHA referenceLlama2XLPropTruePileMHA. Native500-step windows,agent1000-stepreports.

Bet within XL: mid/late loss-parent -.008; MHA-referenced gainmultiple1.06–1.08;
speed~.367step/s (~-4% vs historicalparent .3826, TBrotation operationalfix means
latehistorical .2564 is not a valid architecture throughput anchor).
Crossscale only compares gainmultiples `(MHA-MShift)/(MHA-parentSOTA)` at exact
schedule fractions, Medium13500 vsXL50000, never crossscale raw loss gaps.
Family scripts/report_mshift_cross_scale.py reads nativecached10-step points;
2% phases mapMedium270 toXL1000, five exact ±25step points/no interpolation,
separate per-scale MHA andSOTA. Missingraw windows staymissing. Mediumcompleted13500;
XLparentstopped33002 so matched parent comparison supported only~66% schedule.
Do not invent later parent loss or silently continue use fixedearlier checkpointloss.
Validation checks equalstage ratios with differentabsolute gains, missingpoint,
nonpositive referencegain. No changes to commonreport script.

Status: development checks, not yet AOT/FIRST_STEP.


ActualFIRST_STEP andstep14 verified17:22UTC, workerparams1432383640 matchshape
probe, compiled AOTloaded, runtimeeeec67ce551aa84ca578c22bef361e5a5ee5d491.
UE5a SPOTv5p32 xd-v5p-32-310192-maxtext. Canonical20–99 .3646625step/s
(80samples) vs historicalparent .3826: -4.6883%, speedbet .367 missedby-.64%.
Parentlate .2564 is omitted as TBwhole-event slowdown; health/logging operational
lineage differs, no same-VM pure-architecture speed claim.
MHA-referenced gain specification persisted; family crossscale script deployed
matchingSHA256472bdaa93fa20d2afba39a4fab260c017311fa5579614085ca130bc9ee2f56e7,
firstpending report executes correctly. Native500 windows,agent1000 reporting.


XL2000 review: ownparentgaps500/1000/1500/2000 -.048231/-.036913/-.028322/
-.025337; MHA-referencegainmultiples1.040713/1.079065/1.087184/1.097893.
Exactcrossscale2%: Medium270 .978969 vsXL1000 1.079065;4% Medium540 .989668
vsXL2000 1.097893. XL strongeratbothobservedearlystages, no finalscalinglaw
claim. Crossscale tables onlyratios; ownXL gap remainsseparate.
Numerical-lineageaudit b83223f→0f7f6d10: train_compile/optimizers/maxtext_utils
unchanged; train.py onlywriter.begin_step; max_utils rotatingwriteroperational
fix. OldMHAe30c1b8 andparentb832 AOT both calltrain.create_model_optimizer,
so no knownmissingWD-control issue inthesecomparisons.
