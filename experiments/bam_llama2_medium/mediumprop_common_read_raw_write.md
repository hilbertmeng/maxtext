# Common V/O read scale and amplitude-preserving writes

Runtime worktree `/data0/xd/mediumprop-common-read-raw-write`, branch `codex/mediumprop-common-read-raw-write`, forked from latest main `f4b117c8`. Parent:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile` (bare strongest combo, no dynamic-localO gamma).

Three configurations append `CommonReadScaleRawWriteTruePile`, `CommonReadScaleRawAttentionWriteTruePile`, or `CommonReadScaleRawMLPWriteTruePile` to the parent prefix before `TruePile`. All keep the same common read scale and raw embedding write. The two controls retain small-MLP or attention content RMSNorm, respectively. All compare to the bare SOTA parent in a shared native loss-gap/gain plot; MHA is the common gain reference. Pairwise differences in that plot identify the benefit of restoring either normalization; no extra full factorial or recalibration run is authorized.

Each actual layer learns one direct signed scalar `common_vo_read_scale`, initialized to `1/sqrt(1+2*l)` for l=0..17. Both the COMPLETE V and localO reads (dynamic and static, all heads) are multiplied before context attention and construction of attention write contents. Q/K are unchanged at a fixed M/input. Residual output and attention write both consume the scaled content; no second scale in the write kernel. The final layer still has its coefficient while its attention/small-MLP writers remain absent.

The scalar is stored as length one to allow the existing parameter scan axis1. It is initialized from the actual scanned layer index, not a reused static block index. Native/Pallas whole-model tests, explicit 18-layer initialization, and a single-layer signed-scaling test cover complete V/O plus residual/write gradients and Q/K invariance. Static V/O keys are perturbed nonzero in the semantic check; coefficients0,-.5,2 test exactly-once scaling and direct signed parameterization.

For each writer, a positive fixed divisor replaces content RMSNorm while address RMSNorm and the existing gate stay unchanged:

| Writer | Fixed denominator |
|---|---:|
| Embedding | .0016052833 |
| Attention | .1946289063 |
| Small MLP | .0045444682 |

`None` retains that writer's original content RMSNorm. Division uses the calibrated FP32 constant and rounds the result to model dtype; the denominator is never recomputed from learned read scales. The v8t writer takes per-group static divisors, with corresponding content AND gate derivatives. Read kernel unchanged; common scaling is in XLA. No new matrix products or write-kernel VMEM buffers.

These constants come from the user's side calibration at model snapshot `f4b117c8f89bb444729a1db598416071a9e5687f`, formal dimensions/T4096, seed0, native reads/writes with Splash, formal scan, eps1e-6, 128 real Pile sequences (64 calibration/64 heldout). It iterates the coupled network and matches each writer's full outer-product mean energy, including gates/head cross terms; it does not enforce equal output amplitudes or equal per-layer write energy. The attention single-pass denominator~.24 is not used. Heldout writer energy ratios .99983759/.97742353/.99492518.

Source/results: `/tmp/bam-write-rms-calibration-s2pchq_6/`, `results/calibration-seed0-128.json`, `results/read-parts-summary.json`; GCS `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/side-write-rms-s2pchq_6/`. The control variants intentionally keep these same denominators; their initial whole-network energies are not independently refitted. Training inherits the parent's seed9876 and all optimizer/data settings for comparison fairness; the constants retain their documented seed0 calibration provenance.

Context write contents are dominated by localV aggregated over context rather than localO at initialization. The read scalar controls BOTH paths. Even after theoretical read scaling, context/V RMS grows from~.22 atL1 to~.76 atL17; preserving this enhancement/cancellation information is the experiment's purpose. Raw attention writes also shift energy budget from weak early to strong late layers. Effects cannot be attributed solely to token/head amplitude retention.

Parameters432112210 (parent432112192+18=.0000125 W_Q), MLP2510/1128/tail3904 unchanged. Existing scale weight-decay exemption retained. Generic training health ON, internal activation/BAM-specific collections OFF. Eighteen signed parameter values are emitted under `parameter_values/.../common_vo_read_scale`, alongside existing L2 and raw-gradient norms; no extra activation capture.

Latest infrastructure inherited from f4b117c8; verify operational changes before sealing. Tuned Splash, Pallas v8t read/write48MiB, full remat, scan17 single layers+explicit tail, T4096/batch16 perchip, CP200/latest2, TB rotation500, plan13500. Borrow retained idle FLEX0 (EW4a) for AOT only, never adopt or delete it. Current user region priority UE5a, EW4b backup.

Pre-launch terminal bets (MHA-referenced gain relative to bare parent): all-raw1.03, attention-only1.025, small-MLP-only1.005. Prefer the first two because they retain context aggregation amplitude. Expected speed near parent.686768step/s (roughly.68–.70). Main concern is learned feedback gain late in training beyond initialization calibration; inspect the signed layer-scale profiles and loss, not just norms.

## Fourth arm: attention-write-only depth scaling

`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75AttentionWriteScaleRawWriteTruePile` keeps the parent's complete V/localO reads and vector update at fixed M. The same directly learned `1/sqrt(1+2*l)` coefficient multiplies only the attention write contents before the fixed denominator; embedding/attention/private-MLP content RMSNorms are all removed, with the same three denominators. No recalibration or extra scaling on private-MLP/embedding writes. There are17 live attention-write coefficients; no dead L17 coefficient. Parameters432112209, same MLP widths. Signed values are logged under `parameter_values/.../attention_write_scale`. Changing M naturally changes later vector updates; exact fixed-M branch alignment is not a claim of identical whole-network trajectory.

This arm joins the same bare-parent comparison group. It separates write-amplitude retention from attenuation of the vector update. Its pre-launch terminal gain bet is1.025 versus the original all-raw/common-read1.03: common read scaling may also improve the vector branch's learning scale. Expected speed near the parent, potentially slightly faster than scaling both complete reads outside the kernel.

In the small-MLP-only raw control, retained attention content RMSNorm largely cancels a positive common scalar on the attention-write path (epsilon excepted), while the vector residual stays scaled. This is an intended mechanism difference when interpreting the controls.

First three actual runtime `ddbb56389cf129aa0bddd9b06c5285c091a15d2a`, UE5a trainers310408/310409/310410. All have FIRST0 and exact source/AOT/data/13500 schedule verified. Startup20..99:.673725/.6717375/.679300 step/s versus bareparent.686768 (−1.90%/−2.19%/−1.09%). Speed is modestly below the prior, with no extra GEMMs; the kernel-external scale/reduction overhead has not been separately profiled. First three signed-health logs each contain all18 actual initial values matching1/sqrt(1+2*l), finite through250/220/140. All AOTs were serialized by the shared retained-worker lock; FLEX0 preserved. Actual proofs under `/data0/xd/bam_diagnostics/mediumprop-common-read-raw-write/`. Fourth arm is a later sealed runtime28eb60f1 and has its own subtree `write-only/`.
