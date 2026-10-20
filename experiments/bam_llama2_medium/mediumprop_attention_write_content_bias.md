# MediumProp BAM attention write-content bias

RUN: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroAttnWriteContentBiasTruePile`.
Parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile`. Worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
Runtime `e2a387408439f995929aab09a9062c61e565323a`. Trainer `xd-v5p-16-2910122-maxtext`, UE5a.
Owned spot compiler queue: primaryEW4a, backupsUC1a/UE5a through `prepare_train_aot.py`;
user-retained compiler is busy and is not borrowed or managed by this task.

Per-layer attention content bias[16,75] starts at zero and is added before
existing per-head content RMSNorm. Both ordinary attention writes and deferred
attention factors used by sparse MLP writes use this same change. W_O's vector
residual output, MLP content/address/gate, embedding and all read paths are unchanged.
A separate affine module scope preserves existing parameter RNG streams.

Static Q/K/V/O reads all start at zero through inheritance; dynamic reads unchanged.
Additional21600 parameters=.015W_Q total; MLP[3901,3774,3901] unchanged.
Total432117728, still3472 below MHA. Health adds three bias amplitude scalars per
layer (parameter bias RMS, raw attention content RMS, their ratio),54 total.
Otherwise inherit generic+concat+write metrics, TruePile4096 and parent's checkpoint policy.

Bet versus direct LocalVStaticZero parent: terminal gap-.005, speed within1%.
Plan13500, loss windows200, reports1000, reviews2800/5000.
CPU budget/export, zero-bias parity and RNG parity, live bias gradient and normal/
deferred write equivalence plus affected legacy sparse-write regression gate launch.
CPU and owned AOT preparation run concurrently; formal trainer follows both gates.

Monitoring explicitly retains both the direct LocalVStaticZero parent and original standard BAM SOTA via launcher COMPARE_RUNS; the latter remains the direction benchmark.

Startup verified: e2a3874 source, same-zone UE5a TruePile4096, compiled function loaded, FIRST_STEP4 and continued through39. All54 new bias scalars are present; layer1 bias RMS at30 is .000261 with finite raw-content ratio .0681. Owned spot AOT and all queue candidates have completed cleanup.

Steady step20–99:80 samples, mean .5190625step/s, median .521; +.12% vs direct parent .5184375, −.18% vs standard .520.

Closeout: hot replaced at2208 after committed checkpoint; new owner FIRST_STEP verified. Last5(1200-2000) vs V-zero -.003832, vs standard +.090495; small benefit did not repair zero-V deficit. Local closeout wrapper used after marking registry stopped, preserving the new trainer TPU.

READY leases (UTC); preemptions 0:
- 2026-10-08T09:49:15Z → 2026-10-08T11:04:10Z, 4495s, us-east5-a, hot_switch_run_boundary.
