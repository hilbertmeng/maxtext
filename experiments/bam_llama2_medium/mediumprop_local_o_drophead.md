# Per-token O-only DropHead, p=.2/.4

Implementation worktree: `/data0/xd/mediumprop-attention-diagonal-zero`;
branch `codex/mediumprop-attention-diagonal-zero`. Main `exp.py` also registers both classes.

Parent: `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile`
(completed13.5k runtime5a3d55b), strongest completed MediumProp combination.
No change to M75x32/allR256, QK93/V75, diagonal-logit suppression,
M-shift .5, independent VO keys, 2:1 MLP allocation or initialization.

| RUN | p | Owned trainer | Zone |
|---|---|---|---|
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODropHeadP020TruePile` | .2 | `xd-v5p-16-310379-maxtext` | us-east5-a |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODropHeadP040TruePile` | .4 | `xd-v5p-16-310380-maxtext` | us-east5-a |

Each sample/token/head independently keeps its complete static+dynamic O read with
probability1-p and rescales kept O by1/(1-p). AV stays active. Both W_O residual
output and attention M-write content receive AV+maskedO. No new parameters:
432112192, ordinary/private/tail widths2510/1128/3904, unchanged M cache.
The already verified post-read Pallas frontend performs masking; no read-kernel ABI edit.

Formal runtime: Pallasv8t/48MiB/fused inputs, tunedSplash, scan17+explicit tail1,
T4096/globalbatch128/per-device16, planned13500, CP200/TBrotation500,
generic training healthON/internal+BAM OFF. Borrow `llm-jax-v6e-1-0` FLEX_START
in europe-west4-a for serialized AOTs; never own/delete the compiler.
UE5a chosen from repeated recent5.5h uninterrupted comparable leases.

CPU gate: new p=.4 sampling/config and four-way parameter budget, existing sparse
train-probe tests and TB rotation tests. Existing per-token native/Pallas whole-model
forward/all-gradient and BF16/remat checks remain applicable: model/helper source
unchanged from main283bbb31, only class launch settings/probability changed.
Infrastructure audit against main283bbb31: train/train_compile/checkpointing/max_utils/
optimizers/summary_writer match; current sparse eval_loss pipeline and rotation retained.

Comparisons: only direct parent in compare_runs; common gain reference
`BamMHAMediumPropC256TruePile`. Console `loss` remains the optimization objective;
`eval_loss` is the same training batch/old parameters with dropout disabled,
measured only at 200*k+[-20,-10,0,10,20], k>=1. Registry reads eval_loss for these
new RUNs and ordinary loss for historical controls; never mixes unmeasured steps.
2k public reports combine both arms on common gap/gain axes, start600; native stride200.

Before-run terminal gain bets versus parent (MHA reference): p=.2 R1.015
(range.99..1.04), p=.4 R1.005 (range.97..1.04); favor .2. Expected throughput
~.66-.68step/s versus parent's .686768, to be checked after startup.
13.5k planned; standard2800/5000 trend reviews. No inference-cache or parameter cost.
