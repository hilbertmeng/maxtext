# MediumProp standard-arm versus full-head QKNorm

User authorized2026-10-21. New formal RUN:
`BamMediumPropMShiftReadG050StandardQKNormTruePile`.
Formal runtime `2f6466ba18472c4ea3886534cf44bfc65e1ac394`;
AOT load and FIRST_STEP verified2026-10-21 (global batch128, norm-after-concat
False, all health OFF, TB500, actual432112328 parameters). Completed13500 on
2026-10-21; final results and forecast assessment follow below.
Worktree `/data0/xd/mediumprop-qknorm`, branch `codex/mediumprop-qknorm`;
new TPU identity `xd-v5p-16-310260-maxtext`, primary `us-east5-a`.
Retained FLEX compilers checked first: none READY in EW4a/UC1a/UE5a.
Use the managed spot AOT workflow, preserving all user-owned retained resources.
No extra formal model arms are authorized by this experiment.

## Design and baselines

Original MediumProp TruePile4096 (not Pythia), D1200/L18/N16/head75,
read-only single-gate M-shift .5, shared R256 within Q/K/VO reads,
independent attention/MLP write downs. Normalize only vector-projected18 Q/K
coordinates before their RoPE; concatenate the M-derived57 afterwards.
MLP widths `[3802,3676,3802]`, initialization, optimizer and data order unchanged.
T4096, global batch128 (16/device on v5p-16), 13500 steps, LR3e-4,
warmup135, cosine floor.1. Original epsilon1e-6 and effective norm scale1.
Only648 learned norm scales (0.00045 W_Q total) are added; no MLP repayment
for this negligible parameter change. Checkpoints200/latest2/keep0/finalforced,
TB rotation500; generic, internal and BAM health OFF as in the agreed plan.
Model-level tuned Splash tiles1024/compute512/fused backward enabled.

Direct references (all completed13500):

- `BamMediumPropMShiftReadG050FullQKNormTruePile`, runtime03c8ade:
  full75 Q/K norm after M reads and gates, before partialRoPE18.
  Historical final5 loss versus no-QKNorm is-.0109042.
- `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile`, runtime52cf0b3:
  no QKNorm. Historical health ON, so its old speed is not a timing control.
- `BamMHAMediumPropC256TruePile`, runtime0d31a3d: common gain reference.
- `BamMHAMediumPropC256QKNormTruePile`, runtime03c8ade: strengthened MHA
  control available for second-order QKNorm comparisons; no new MHA run.

Mudd was subsequently removed at the user's request: it has no QKNorm and is
not the relevant control for this question. Original MHA remains only the common
gain reference, not a separate direct-loss comparison.

Native loss stride200, agent reports1000. Register gain triples for standard18
versus full75 and versus no-QKNorm using the original MHA reference; also
retain full75 versus no-QKNorm with that same reference. Do not substitute Mudd.
The existing paired-QKNorm interaction reporter can use the new BAM arm and
the same three historical controls for the second-order series.

The matched timing probe is separate from formal training: four norm-placement
and remat full/minimal arms on one spotv6e-1, per-device batch16/T4096,
source6750b251, wrapper and note in the implementation worktree. Report timing
hardware explicitly; do not compare those absolute speeds with formal v5p-16.

Prior: standard18 minus full75 terminal loss+.005, range[-.005,+.015];
matched-health throughput within1%. This is a mechanism prior, not an extrapolation.
Equivalent explicit no-QKNorm prior: terminal loss-.006, range[-.016,+.004],
MHA-referenced gain multiplier about1.04. Expected to retain roughly half the
observed full75 improvement; beating full75 would falsify the claimed benefit
of constraining the amplitude of the M-derived Q/K coordinates.
Runtime shape/initialization/causality tests explicitly check the norm width
and RoPE order; agreement between two implementations alone is insufficient.
For speed, compare the same all-health-OFF/tuned-Splash setup for both norm
placements, not historical measurements with other kernel or health settings.

## Final result, 2026-10-21

Standard18 fails to retain the benefit of full-head QKNorm. Against no QKNorm,
the modest early advantage vanishes around3k; after the final sign reversal at
3.8k, every native200-step window is worse through13.4k. Training to completion
confirms this was not merely slower convergence.

All loss points use five common raw samples in a +/-25 window. Ratios are from
the shared registry, with original MHA as the common reference. Final five
windows are12600/12800/13000/13200/13400:

| Comparator | Final5 mean gap (standard18 minus comparator) | Range | Mean gain multiplier |
|---|---:|---:|---:|
| Full75 QKNorm | +.01299896 | +.0126354 to +.0137142 | .92194879 |
| No QKNorm BAM | +.00209472 | +.0015134 to +.0025742 | .98654200 |
| MHA QKNorm | -.12881012 | -.1299094 to -.1276328 | not registered |

The full75 gap shrinks from+.038144 at1k to+.016912 at5k and+.013294 at10k,
then plateaus near+.013. No-QKNorm gap moves from-.006243 at1k to+.001204
at5k, +.001625 at10k and+.002095 at the final5 mean. Its corresponding gain
multiplier moves1.01068 -> .99396 -> .98999 -> .98654. Full75-referenced ratios
remain around.92 in late training despite a shrinking absolute gap.

The prior +.005 versus full75 was too optimistic (outcome+.013, inside the
stated upper bound+.015). More decisively, the -.006 /1.04 prior versus no
QKNorm was wrong: the measured mean is+.002095 /.986542. Normalizing only the
standard arm is not a useful approximation here. These results support making
the M-derived and standard Q/K coordinates share the normalization denominator;
they do not separately identify logit-scale and branch-balance mechanisms.
The currently training ungated-static-read arm tests a different issue: whether
those gates add value after full-head normalization. No extra experiment is
authorized by this closeout.

Formal throughput ~.541 step/s with all health OFF and tuned Splash. Historical
full75 used older Splash tiles, and no-QKNorm had health ON: their formal speeds
are not matched timing controls. The v6e matched norm-placement probe found
standard18+2.22%; that is a v6e result, not a measured v5p16 speedup.

The checkpoint-preserving full/minimal/full segment is documented in
`mediumprop_qknorm_timing.md`: only steps5682–5856 used minimal; all other
steps used the original full runtime, same data and original13500-step LR
schedule. One UE5a preemption forced checkpoint12283; resumed at12284.
Final checkpoint13500 is committed; the TPU and queued resource are verified
absent; TensorBoard SYNC_OK verified. Complete READY leases (UTC), all on
v5p16/us-east5-a:

| Start | End | Duration | Exit |
|---|---|---|---|
| 2026-10-21 06:33:20 | 2026-10-21 13:05:36 | 6h32m16s | preempted |
| 2026-10-21 13:12:03 | 2026-10-21 13:55:59 | 43m56s | complete13500 |

Artifacts: `/data0/xd/bam_diagnostics/mediumprop-qknorm/standard-final-report.json`
(complete native200 trajectories), `standard-final-trends.png`/`.svg`,
`standard-final-leases.txt`, and `standard-final-tb-sync.log` in the same directory.

## Mainline implementation

Full-head BAM QKNorm is now implemented on both the pure JAX and fused Pallas
paths. For concat-QK BAM with `qk_norm=True`, the default
`bam_qk_norm_after_concat=True` normalizes concatenated M-derived and standard
coordinates, then applies RoPE to the standard tail only. Both full-head and
standard-arm-only implementations normalize **before** RoPE. MHA is unchanged.
The explicit False flag reproduces this experiment's standard18 placement.
With QKNorm disabled, the existing computation is unchanged.

The Pallas read receives unrotated standard Q/K in full-head mode. Its folded
Q scale is disabled, then full-head norm, partial RoPE and the usual
`1/sqrt(head_dim)` Q scaling run after the read. Normalizing an already scaled
or rotated query would change the model; neither is used.

`BamPythia2p8BSota` explicitly selects full80 norm before RoPE20, inherited by
IndependentR768 and Pallas variants. Norm scales increase by3840 parameters
from the previous standard20 scope; MLP widths stay128-aligned and unchanged
(negligible budget exception). Existing standard20 checkpoints retain their
explicit False scope when resumed; current formal training remains pinned to
its registered runtime, not this mainline change.

Regression coverage: `MaxText/tests/bam_full_qknorm_test.py` checks width,
pre-RoPE ordering, nonuniform learned scales, formal budgets, finite/nonzero
scan gradients and causality; `bam_pallas_qknorm_test.py` checks logits and all
parameter gradients against pure JAX for v7 and v8t with single-gate M-shift,
plus standard-arm compatibility. The shared BAM suite is also a merge gate.

## Infrastructure audit before sealing

Reviewed main `e094303f` against existing worktree `809f5b5b` (and original
model runtime03c8ade). The worktree model sources under `MaxText/layers/`,
optimizer, AOT train_compile, max_utils and summary_writer are unchanged from
03c8ade. In particular TB rotating-writer construction and `begin_step` before
scalars already exist; resolved rotation500 is positive. Keep these paths.

Main-only changes inspected individually:

- `train.py`: compact M-shift health decoding, multi-tail metric decoding,
  scanned Pythia activation reporting, explicit health-default access. All
  are inactive for this single-carry original model with all health OFF;
  retain historical numerical/training path.
- `checkpointing.py`: additive exact-step retention options; the old path
  already normalizes keep_period0 toNone and supports final saves. No fixed
  milestone request for this RUN; retain legacy latest2 retention.
- `pyconfig.py`/base defaults: Pythia validation, global-batch resolver and
  keys for newer circuits. None is needed for this fixed original v5p-16
  batch; preserve historical defaults and explicitly declare experiment flags.
- No main-only changes to optimizer WD, AOT optimizer propagation, summary
  writer or training-step math require porting for this RUN.

Thus no model/infrastructure files are transplanted wholesale from main.
Validation includes standard18 norm placement, unchanged other parameter shapes
and seeded initial values, scanned finite/nonzero gradients, causality, existing
full75 placement/scales and the actual summary-writer rotation regression tests.
Sealed-runtime consistency and actual AOT load/FIRST_STEP remain launch gates.
