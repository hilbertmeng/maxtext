# MediumProp standard-arm versus full-head QKNorm

User authorized2026-10-21. New formal RUN:
`BamMediumPropMShiftReadG050StandardQKNormTruePile`.
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

Native loss stride200, agent reports1000. Register gain triples for standard18
versus full75 and versus no-QKNorm using the original MHA reference; also
retain full75 versus no-QKNorm with that same reference. Do not substitute Mudd.
The existing paired-QKNorm interaction reporter can use the new BAM arm and
the same three historical controls for the second-order series.

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

## Compatible minimal-remat continuation

User requests continuation to the full13500-step endpoint, even though the
standard18 normalization comparison has become clear, to measure its final gap.
User also authorizes switching this same RUN to minimal remat after the completed
v5p-8 timing pair (+3.89%, per-device16). Preserve the RUN/checkpoint/output/data
identities, original13500-step optimizer/LR schedule and all health-OFF settings.

The new runtime changes only remat from full to minimal; compare_runs also records
Mudd explicitly, matching the five already registered live comparisons. Runtime
2f6466ba18472c4ea3886534cf44bfc65e1ac394 is the pre-switch full-remat source.
No model/optimizer implementation or parameter shapes change. Compile exact
v5p-16/s13500 AOT before stopping the old trainer; require committed final save,
all old workers absent, preserved data cursor, then a strictly newer FIRST_STEP
and a subsequent periodic checkpoint. Keep the existing training TPU throughout.

Prior for formal v5p-16: +3--5% throughput (.541 to approximately.56 step/s),
subject to target verification. The paired v5p-8 result is documented in
`mediumprop_qknorm_timing.md`; it is not the formal-v5p-16 result.


Target verification contradicted the transfer prior: on the same formal v5p-16,
full5620--5679 averaged.5394333 step/s (median.541), minimal5702--5761 averaged
.4397667 (median.441), 60 samples each: minimal -18.4762%. Actual configs differ
only in remat, report metadata and executable path; batch128/per-device16,
Splash1024/fused, health OFF, data cursor and13500-step LR schedule match.
Minimal resumed correctly at5682 from committed5681; no checkpoint/data reset.
The v5p-8 +3.89% result therefore does not transfer to this target. Perform a
same-machine return to the original full executable to distinguish an executable
regression from a changed machine condition; do not assert a specific HLO cause
without profiling. Preserve both executables and logs. Full13.5k loss training
continues, as explicitly requested by the user.
