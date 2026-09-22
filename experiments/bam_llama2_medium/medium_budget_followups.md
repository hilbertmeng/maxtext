# Medium P_loc and joint-QK budget followups

Implementation worktree `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
Base: `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer`.
All runs retain M48x32/C8, NoPE48/RoPE16, col-only, static QK, VO shared C8 independent gates,
P_loc pre-RMS bias/RMS and generic/BAM968 health. Plan13500,checkpoint200, report1000 with200-step windows.
All three initialized model totals must equal411885440. No hardware rounding.

| RUN suffix | Change | MLP LLF widths | Direct baselines | Bet vs original K48 |
|---|---|---|---|---|
| PLocSlice512Linear | x[:512] ->512 linear |3093/3093/3087|K48,Slice384Linear|+.0003 [-.0015,+.002]|
| PLocSlice768Linear | x[:768] ->512 linear |3050/3050/3045|K48,Slice384Linear,Slice512Linear|—|
| QKJointGelu128 | x1024 ->128 GELU ->256; gates unchanged |3082/3082/3077|K48,JointGelu256|-.001 [-.003,+.002]|

Slice512 saves131072 P_loc weights/layer vs R256; compensate128 MLP width units perLLFblock.
Slice768 equals originalP_loc393216weights. Joint128 saves98304QKweights/layer vsrank4;
add32MLPwidth/layer. Joint128 betvsJoint256-.002[-.004,+.001] at maturecommonsteps.
Slice speed expectedflat; Joint128flat to+1% vsK48 .6378/s matchedhealth.

Trainer UE5a primary,UC1a/EW4b backups; exact-runtimev5p16 AOT onEW4a compilerprimary,
UC1a/UE5a backups. Start Slice512/768 as new runs; user explicitly authorized hotreplacement
of active MediumJoint256 by Joint128 after readiness, keeping oldcheckpoint. Allnewrunsstart0.
Old XL runs remain under continuous monitoring.

Owned resources planned: `xd-v5p-16-k48-ploc-slice512-linear-maxtext`,
`xd-v5p-16-k48-ploc-slice768-linear-maxtext`; Joint128 inherits
`xd-v5p-16-k48-ploc-slice384-linear-maxtext` at handoff.

Full run names:
- `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice512Linear`
- `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerPLocSlice768Linear`
- `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerQKJointGelu128`

Runtime e3791a14 (full hash in `/data0/xd/medium-budget-followups-runtime.txt`). Pinned BAM suite57PASS384.454s; actual initialized totals all411885440; shardingoverhead.1668503%<2%; fulltraintrace968metrics allPASS. Artifacts `/data0/xd/medium-budget-followups-{tests.log,audit.json,health-trace.log}`. Three exact AOT preparations submitted independently; readiness launchers gated on ready+cleanup success.

JointGelu128 launch verified: from0, FIRST70, AOTloaded, exacte3791a1. Steps10–14 .6286/s (-1.44%vsK48.6378,+.06%vsJoint256.6282),20–24 .6306. Matchedgeneric/BAM968. OldJoint256paused7245 at23:31:02UTC; checkpointretained,TB SYNC_OK. Evidence `/data0/xd/joint128-start-verified.json`.

Slice768 launch verified from0, FIRST64,AOTloaded,e3791a1. Steps10–14 .6376/s(-.03%vsK48,+.28%vsSlice384),20–24 .6386; matched968health. Evidence `/data0/xd/slice768-start-verified.json`.

Allthree AOT states ready with cleanup_failures=[]: e3791a1-0b3d626d/457d6d22/8c56630e. Slice512 prolonged EW4a provisioning handled by retaining primary and adding UC1a/UE5a candidates; primary eventually compiled, allthree compiler candidates released. FormalSlice512 submitted23:48:56UTC.

Slice512 launch verified from0,FIRST56,AOTloaded,e3791a1. Steps10–14 and20–24 .6356/s(-.34%vsK48,-.03%vsSlice384),matched968health. Evidence `/data0/xd/slice512-start-verified.json`. Allthree nowtrainingUE5a,report1000,review2800.

Slice768 stopped2913 after2800 review: vsK48 last5+.007642, vsSlice384+.005049; vsSlice512 through2200+.003204, no sustained catch-up. Same total params/M-cache, speed-.03%/+.28%/+.31% respectively, matched968health. Checkpoint2913 committed; TPU/queue absent, finalTB SYNC_OK. No preemptions.
Joint128 continues after2800 review: vsK48 last5+.007187, vsJoint256+.008224; both deficits narrowing, preserve later-MLP-budget test.
