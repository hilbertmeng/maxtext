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
