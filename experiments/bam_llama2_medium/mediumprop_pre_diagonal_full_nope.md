# Negative diagonal logits plus full NoPE on independent2:1/separateVO

RUN:BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile.
Direct parent:BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeTruePile.
Keep pre-softmax diagonal suppression and exact-zero empty-history AV output.
Retain all75 M-derived QK coordinates instead of57; append18 RoPE =>QK93/V75.
Scale remains1/sqrt75,noQKNorm,M75x32. No new parameters or width changes:
432112192,body2510/1128,tail3904,scan17 single layers+explicit terminal.

Hot-replace user-selected GeneralWrite trainer xd-v5p-16-310372-maxtext/UE5a;
new RUN starts from0 under its own prefix with13500 total steps. Old GeneralWrite
checkpoint/TB/loss/leases retained; this is an architecture replacement, not a
numerical backend parity test or continuation of GeneralWrite parameters.
Borrow retained FLEX_START llm-jax-v6e-1-0/EW4a/worker0 for exact v5p16 AOT.
User-owned compiler and other active pre-softmax arm remain untouched.

Worktree:/data0/xd/mediumprop-attention-diagonal-zero
Branch:codex/mediumprop-attention-diagonal-zero
Reuse the inheritance family's implementation location. Infrastructure audit
against current main15c02109:zero differences in train.py,max_utils.py,
checkpointing.py,train_compile.py,optimizers.py,maxtext_utils.py. All operational
fixes,TB rotation500 and begin_step retained. No attention/kernel source change
from sealed55ea52c:combine its existing strict mask with already validated full
NoPE Pallas read. Targeted gate checks exact parameter tree and native/Pallas+
real Splash QK93/V75 full-model logits/all gradients; prior58 shared checks stay
applicable to unchanged attention/kernel code. Exact config seal/AOT/FIRST_STEP
still required. Generic healthON,internal/BAM OFF,tunedSplash,Pallasv8t48MiB,
fullremat,T4096/global128/per-device16,CP200/latest2,13500 steps.

Pre-run terminal MHA-normalized gain bet relative to direct pre-softmax parent:
1.025 (range1.00..1.05). Standalone fullNoPE delivered~1.029 vs its old parent;
expect substantial but incomplete transfer, since suppressing self attention can
change the value of extra QK coordinates. Expected speed-1..2% vs pre parent.
compare_runs contains only direct pre parent; gain triple common reference is
BamMHAMediumPropC256TruePile. Existing standalone fullNoPE evidence remains in
its archived native reports for interaction analysis, without adding extra gap
baselines to the routine figure. Report2k,cumulative plots from600.


## Verified hot replacement

Runtime5a3d55b4065e3a6fe49af095762a6a71ea834c7f. Parameter-tree and full-model
QK93/V75 Pallas+Splash/native checks passed; max parameter-gradient error4.17e-6.
Exact v5p16/s13500 AOT on the retained FLEX passed; manifest checked by hot-switch
entrypoint. Held only old auto-controller to prevent horizon cleanup while its
worker kept training; GeneralWrite completed13500. At14:18:09UTC handed the same
READY xd-v5p-16-310372-maxtext to new controllerID310375, no node/queue deletion.
OldCP13500 committed and old process absent on both workers before new launch.
New FIRST_STEP1,loss10.851450:starts from scratch under new prefix. Direct parent
and MHA gain reference registered; TruePile4096 UE5a path verified.

Startup20..99 all80-step harmonic rate .686768 vs pre parent .694718 (-1.14%),
median .691,matching speed prediction. Same generic-healthON/internal+BAM OFF,
per-device16,T4096. Worker confirms Loaded compiled function. Artifact root:
/data0/xd/bam_diagnostics/mediumprop-pre-diagonal-full-nope/;see first-step.log,
new-registration-proof.json,handoff-journal.json,startup-speed.json and cpu.log.
