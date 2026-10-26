# Full NoPE on the completed independent-VO Medium SOTA

## Question and intervention

Revalidate retaining all75 M-derived Q/K coordinates on the completed
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile`
SOTA (runtime5dea945), rather than the older sparse-write/shared-VO architecture.
Candidate:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysQK75TruePile`.
Only change `bam_local_qk_col_output_dim` and `bam_partial_rope_nope_dim` from57 to75:
Q/K become75NoPE+18RoPE=93, V stays75, M stays75x32. Keep1/sqrt75 logit scaling,
noQKNorm, all initialization, scan and writer schedules unchanged.

The direct comparator is the completed independent-VO parent only. Gain ratios use
`BamMHAMediumPropC256TruePile` as the common reference through the registry's native
three-run calculation. Plot both loss gap and ratio from600, using200-step windows.

## Budget and kernel

No trainable parameter is added; complete parameter-tree shapes/dtypes must match
the parent exactly:432112192 parameters. Body L0..L16 ordinary/private widths2510/1128,
L17 ordinary3904; no final attention/private writer. Seventeen single scanned body
layers and one explicit terminal layer; no per-layer width mechanism.

Pallas v8t contracts M once at75 data coordinates; append18 standard RoPE coordinates
outside the fused read. This preserves the independent V/O fused reverse and M-shift
and avoids extending V/O or padding the M state to93. Both forward and every input/
parameter gradient are checked against native JAX, including real75x32 M, QK93/V75,
independent V/O keys, static gates, shift gates and tile/document boundaries.

Formal settings:TruePile4096, global128/per-device16, v5p16 BF16,13500 steps,
full remat, tuned Splash1024/fused backward, Pallasv8t48MiB/fused inputs;
training health ON, internal/BAM health OFF. CP200/latest2, TB rotation500.

## Ownership and preparation

Worktree:/data0/xd/mediumprop-independent-vo-qk75
Branch:codex/mediumprop-independent-vo-qk75
Planned training TPU:xd-v5p-16-310370-maxtext, us-east5-a.
Region choice:recent same-size Medium leases in UE5a completed without preemptions.
Borrow retained FLEX_START llm-jax-v6e-1-0/europe-west4-a/worker0 only for AOT;
actual QR ACTIVE/FLEX_START and idle worker verified2026-10-26. Never delete it.

Infrastructure audit against main25d0baf:fresh worktree from that commit; no changes
in train.py,max_utils.py,checkpointing.py,train_compile.py,optimizers.py or
maxtext_utils.py relative to parent5dea945. Rotating writer construction in
max_utils.py and begin_step before scalars in train.py are present; inherited
rotation500 is checked. Add summary_writer regressions to the launch gate.
Main's later independent-VO extensions support hybrid heads; with zero ordinary
V heads, the parent parameter tree/initialization/read behavior is retained.

## Pre-run prediction

Terminal ratio relative to completed independent-VO parent:1.02, expected1.00..1.04;
rough terminal loss gap-.0033 (using the parent's MHA-relative advantage~.164).
Speed-1%..-3%, point estimate-2%. Historical sparse-write QK75 gain1.035 is supporting
evidence, not assumed to transfer unchanged to this stronger architecture.

CPU shared-core regression plus candidate budget/native/Pallas forward/all-gradient
and writer tests must pass; sealed runtime config, exact v5p16 AOT and FIRST_STEP are
required before reporting successful launch. This document will retain lifecycle
milestones and final conclusions, not intermediate monitoring reports.

## Launch verification

Runtime:c5ca75f2d1b38e734c1a5f76f535dc1babfb0f4f (metadata commits do not change it).
CPU:3 candidate checks,5 summary-writer checks,47 shared BAM checks and2 parent
independent-VO regressions all passed. Full-model worst parameter-gradient relative
error4.04e-6; full75x32 kernel check includes QK93,V75 and tile-boundary reset gates.
Exact v5p16/13500 AOT compiled on retained FLEX; compiler lifecycle preserved.
Training xd-v5p-16-310370-maxtext/UE5a FIRST_STEP confirmed atstep3, loss10.835516;
this startup throughput is not the stable-speed estimate. RUN data path verified as
`gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096`.
Native gain-ratio monitoring registered for candidate/independent-VO parent/MHA;
compare_runs contains only that direct parent. Task-local monitor state/artifacts:
`/data0/xd/bam_diagnostics/mediumprop-independent-vo-qk75/`.

Startup speed measured from all80 logged steps20..99 (harmonic mean of per-step
rates):.6430688995step/s, median.647. Parent same window:.6995475194, median.704;
relative speed-8.0736%. Actual AOT load confirmed in the worker log. This exceeds
the pre-run1..3% slowdown prediction; retain the result as a failed speed prediction.
Do not infer the hardware bottleneck from logical QK FLOPs alone. Raw proof:
`/data0/xd/bam_diagnostics/mediumprop-independent-vo-qk75/speed.json`.

## Kernel handoff at CP3800

Current runtime:f8d886683827c744d20258c7c6dc22e67ac20dc1, source branch
claude/qk75-profile; local worktree:/data0/xd/mediumprop-independent-vo-qk75-fast,
metadata branch:codex/mediumprop-independent-vo-qk75-fast. User-provided optimization
writes QK93 directly from the Pallas read kernel and splits its cotangent internally;
removes the external concat/relayout, preserving model/configuration/parameter tree.
Exact effective-config seal and AOT manifest/CRC validated for v5p16,s13500. The
handoff-provided candidate checks passed;47 shared CPU regressions replayed here and passed.

Same RUN, same retained xd-v5p-16-310370-maxtext/UE5a. The distinct-RUN hot-switch
entrypoint refuses identical names, so used the Hot Retrain procedure with ownership
checks and a task journal. Periodic CP3800 committed; old raw loss retained through3861
(61 post-checkpoint steps), then controller stopped and both workers quiesced. Graceful
exit committed CP3863; backed up its root data cursor and intentionally restored the
CP3800 cursor so new state3801 and data replay the same old post-anchor batches.
No permanent restore-step override; subsequent checkpoints advance the normal cursor.
New source/AOT verified, FIRST_STEP bound beyond old last3861 returned new3863;
actual first completed step3801. Registry commit/AOT updated; comparisons, ratio
reference, report cursor2000 and original READY lease retained. No TPU deletion/creation.

Paired3801..3820: mean absolute loss difference2.575e-5, max6.3e-5.
Full61-step overlap3801..3861: mean absolute3.17705e-5, max1.05e-4, signed mean2.29508e-6;
exact data-cursor and LR match. This verifies post-switch versus post-anchor old outputs,
rather than comparing adjacent unrelated batches across the handoff.

Same-TPU80-step timing: c5 steps3700..3779 harmonic mean.63965847 (median.646),
f8 steps3821..3900 harmonic mean.69004099 (median.694):+7.8765% versus old kernel,
-1.3590% versus parent's measured.69954752. Formal batch16/device and all health,
scan/Splash/remat settings unchanged. Kernel optimization removes most of the initial
8% timing deficit; do not attribute that original deficit solely to wider QK arithmetic.

Proofs:/data0/xd/bam_diagnostics/mediumprop-independent-vo-qk75/kernel-switch/
(overlap-validation.json,speed-comparison.json,old-final.log,new-worker0.log,cursors,
FIRST_STEP and handoff journal). Remote journal:
/home/lishengping/xd/projects/logs/hot-switch/qk75-c5ca75f-f8d8866/journal.json.
Task handoff source:/home/xd/projects/xd_tpu_scripts/qk75_kernel_handoff_20261026.py,
deployed matchingSHA256714c798d417e9b59f6eba2c9a4bcd1f165454b72e76067a4ed44fb5fbef11ad3.
The original c5 runtime worktree is preserved for reproduction; regular loss reports
stay in task artifacts and chat, not in this ledger.

## Main implementation and configuration-only Pythia transfer

User requested the verified complete-NoPE implementation in main. Ported the
optimized f8d8866 reader (direct wider Q/K outputs and in-kernel cotangent split),
not the superseded c5 external-concatenation implementation. Native model support
was already present; the historical sealed Medium RUN continues on f8d8866.

Added:
BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysQK80
as a child of:
BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeys.
Only retained NoPE widths change60->80: QK100/V80, M80x64, RoPE20.
All R640 projection ranks, single-layer scan31+tail1, serial GELU, private writes,
initialization, health and training recipe inherit. Body ordinary/private MLPs
7808/2944; terminal9984; all remain128-aligned. No new per-layer width mechanism.

This parent uses complete-head QKNorm. The optimized pre-RoPE QKNorm path now
creates scales and divides the RMS statistic using the actual concatenated width
100 rather than the value-head width80. Its standard-branch scale/RoPE precomputation,
local sum-of-squares reduction and post-read elementwise finish are retained.
Attention scaling remains1/sqrt80 as in the parent; both NoPE and RoPE coordinates
are normalized before RoPE. The fused reader handles both QK93/V75 and QK100/V80.

Full abstract trees: parent2,775,189,600; child2,775,190,880. Exactly1,280 additional
parameters (32layers*2arms*20norm coordinates), .0001953125W_Q or .0000461% of parent.
All other parameter shapes/dtypes are identical. This negligible norm cost is not
charged to MLPs. No formal2.8B training RUN, TPU allocation or target AOT requested.
CPU Pallas interpretation verifies math/gradients, not target-TPU memory or throughput.

Validation entry:scripts/check_independent_vo_qk75.sh. It checks Medium's exact
parameter-tree equality and full-model/native-vs-Pallas gradients, direct QK93 kernel
all-input gradients and tile boundaries, and Pythia's actual full-budget trees plus
QK100/full-QKNorm model gradients. Worst full-model relative gradient discrepancy:
Medium4.04e-6; Pythia5.27e-6. Five targeted checks, five summary-writer checks and
47 shared regressions and3 existing Pallas QKNorm compatibility checks all passed
in the pinned CPU environment (60checks total). The production RUNs remain
independently monitored; integration does not hot-switch their source.

Main-merge validation logs:/data0/xd/bam_diagnostics/full-nope-main-merge/
