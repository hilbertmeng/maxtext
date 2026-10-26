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
