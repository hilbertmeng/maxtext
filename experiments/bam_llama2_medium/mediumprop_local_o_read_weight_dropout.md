# Per-token O read-weight dropout p=.2

RUN: `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOReadWeightDropoutP020TruePile`.
Runtime `265562203ab613f633c3e142314a4ebd5d1aae4a`, branch `claude/o-read-dropout-pallas`,
worktree `/data0/xd/o-read-dropout-pallas`. Main exp.py retains its native JAX reference
(`bam_pallas_core=False`); the sealed formal RUN enables the optimized Pallas adapter.
No model-source merge is implied by this ledger entry.

Parent/only compare_runs: `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile`.
Gain reference: `BamMHAMediumPropC256TruePile`. This controls the same strongest completed
2:1/independent-VO/negative-diagonal/full-NoPE architecture, not a different dropout family.
Existing DropHead p=.2 results can supply the head-versus-address-noise contrast with
identical dropout-disabled training-batch loss semantics.

An independently sampled inverted Bernoulli(.8)/.8 mask of shape [B,T,N,32] multiplies
complete static+dynamic O address weights after their normalization, inside the v8t
read. No post-mask renormalization. AV is retained; vector W_O and attention M writes
both use AV+masked O. Evaluation disables dropout. Same-batch CE probes use old,
pre-update parameters, at 200*k+[-20,-10,0,10,20], k>=1; the training objective remains
separate. This is training-batch fitting evidence, not held-out generalization.

No parameter/budget/cache changes:432112192 parameters, ordinary/private/tail MLP
2510/1128/3904, M75x32/allR256, QK93/V75. No inference-time dropout compute.
T4096/global128/per-device16, scan17+explicit tail1, full remat, tuned Splash,
Pallasv8t48MiB/fusedinputs, generic healthON/internal+BAM healthOFF, CP200/TBrotation500.

User-directed plan:13500 steps; explicitly no2800/5000 loss-based stopping points.
Public reports every2000 steps, native windows200, cumulative gap/gain plots from600.
Before-run terminal gain bet1.01 [.98,1.04]; favor gentler address noise over whole-head
noise, with reduced confidence after both negative DropHead results. Expected speed
.677step/s (-1.4% vs parent.686768).

Owned trainer: `xd-v5p-16-310381-maxtext`, us-east5-a, v5p-16 spot, no backup queue.
Registered2026-10-27 04:29:51UTC; READY first observed04:37:15UTC, AOT-loaded and step0
verified. Stable startup20..99 harmonic speed.681282078step/s, parent.686768097
(-.7988%). This startup timing excludes sparse evaluation overhead.

Launch gates: SEALED_CONFIG_OK; existing exact v5p16/s13500 AOT+manifest; CPU10 O
regularization checks (including full-model native/Pallas forward/all-gradient
maxrel6.520921e-6),3 sparse-eval checks and5 TB-rotation checks. Infrastructure audit
against main3e5d18dd: train/train_compile/checkpointing/max_utils/optimizers/pyconfig/
summary_writer/base.yml match; rotation500 and writer.begin_step retained. Negative-
diagonal function is unchanged from db290e9; only unused post-softmax-zero support
differs outside the adapter. No new AOT/diagnostic TPU was requested for this launch.

AOT state on tpu-ag: `/home/lishengping/xd/projects/aot_runs/2655622-016a0355.json`.
Artifact: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/2655622/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOReadWeightDropoutP020TruePile.pickle`.
Local launch/gate/speed evidence and live monitor state:
`/data0/xd/bam_diagnostics/mediumprop-local-o-read-weight-dropout/`.

## Final closeout, 2026-10-27

Stopped at the user's request before the13500-step endpoint. Checkpoints and local
TensorBoard copies are secured; all corresponding nodes and queued resources were
independently verified absent. No zone switches or passive candidates.

`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOReadWeightDropoutP020TruePile`: stopped8076, P2; last five complete windows through8000: gap+0.00715464, MHA-reference gain0.96351483.

The brief2k recovery did not persist; the fitting deficit plateaued around+.006..008.
There was no sustained late recovery by8k. Terminal1.01 bet remains unmeasured.

Full chronological READY leases (UTC; manual stop distinguished from preemption):

```text
BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOReadWeightDropoutP020TruePile: preemptions=2 ready_leases=3
01    2h49m12s  us-east5-a  xd-v5p-16-310381-maxtext  2026-10-27T04:37:15Z -> 2026-10-27T07:26:27Z  preempted
02      40m16s  us-east5-a  xd-v5p-16-310381-maxtext  2026-10-27T07:33:58Z -> 2026-10-27T08:14:14Z  preempted
03       7m27s  us-east5-a  xd-v5p-16-310381-maxtext  2026-10-27T08:22:48Z -> 2026-10-27T08:30:15Z  run_stop
```

Final exact common-step reports, teardown/CP proof and TB marker:
`/data0/xd/bam_diagnostics/mediumprop-local-o-scale-050-100/final-three-{report,metrics,proof}.json`;
TB log `/home/xd/.local/state/maxtext-tensorboard-sync/closeout-20261027T083012689716Z.log`.
