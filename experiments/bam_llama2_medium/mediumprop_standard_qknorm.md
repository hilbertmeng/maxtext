# MediumProp standard-arm versus full-head QKNorm

User authorized2026-10-21. New formal RUN:
`BamMediumPropMShiftReadG050StandardQKNormTruePile`.
Formal runtime `2f6466ba18472c4ea3886534cf44bfc65e1ac394`;
AOT load and FIRST_STEP verified2026-10-21 (global batch128, norm-after-concat
False, all health OFF, TB500, actual432112328 parameters). Model loss/forecast
results remain to be measured.
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
- `MuddLlama2MediumPropTruePile`, runtime4fee490: user-added direct baseline,
  same TruePile4096 and13500-step plan. Include BAM/Mudd gain trajectories
  with `BamMHAMediumPropC256TruePile` as the common reference. Added after
  runtime sealing through launch COMPARE_RUNS/registry metadata; no model change
  or AOT invalidation is required.

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
