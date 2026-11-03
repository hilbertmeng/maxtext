# Unit-init H: whole-M read pre-norm, with/without common V/O scale

Worktree `/data0/xd/mediumprop-unit-h-m-prenorm`, branch `codex/mediumprop-unit-h-m-prenorm`, forked from latest main43f3ea5c. Model parent is the completed UnitH RUN (runtime aa7d8ec), not an adapter descendant:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneTruePile`.
Its full13500-step same-stage reference is available through13400. MHA denominator `BamMHAMediumPropC256TruePile`; both new RUNs compare to UnitH on one plot/report group. Original depth-initialized H remains distinct.

New configurations:
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormTruePile`
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormNoReadScaleTruePile`

Apply read-only M-shift first. The mixed matrix [B,T,75,32] is converted to FP32, normalized by one per-token RMS over the75x32 real matrix entries with epsilon1e-6, and cast back to the activation dtype. All dynamic/static Q/K/V/localO consumers share this view. Raw M remains the carried state and writer accumulation base; M-shift remains read-only. Existing layer attention content RMSNorm, raw small-MLP content (divisor1), calibrated embedding content (divisor.0016052833), gates and addresses remain unchanged. L17 still has no attention/private-MLP writer.

Retained arm keeps18 independent signed common V/O scales initialized1, still applied after the complete V/localO reads. Dropped arm removes those parameters entirely, not fixing them in the optimizer tree. No extra norm gain/bias. Ordinary/private/tail widths remain2510/1128/3904. Retained432112210 params; dropped432112192 (18 fewer, negligible, no MLP refund). Added parameters0 W_Q, unchanged inference M-cache. Whole-M RMS normalization adds a reduction and elementwise scaling; the initial unfused runtime timing also includes disabling the parent's Pallas read/write core, so it is not the intrinsic cost of pre-norm.

Pure-JAX BAM read/write reference uses `bam_pallas_core=False`, `bam_pallas_write=False`; tuned Splash remains enabled. Existing core dispatch couples fused read/write; no temporary split-dispatch mechanism is introduced. Pallas adaptation is delegated to Claude by the user; existing write math can be retained. See `bam_llama2_medium/m_prenorm_pallas_handoff.md` for the compact contract.

Both use single-layer scan first17 plus explicit tail, remat full, T4096/global128/per-device16, normal Adam/LR13500 plan, Medium checkpoint200/latest2, force-final checkpoint and TB rotation500. Generic training health ON, internal NN and BAM activation health OFF. Retained signed read-scale health remains available; no new per-column metrics. Existing eval settings are inherited from UnitH, not silently changed.

Infrastructure audit: branch is directly from current main43f3ea5c; inherited rotating writer construction/begin_step, checkpoint and optimizer/AOT fixes. CPU summary-writer rotation/resume5 tests passed. Current shared47 BAM tests passed. Targeted checks cover full formal budgets; kept/dropped initial-function/all-common-gradient equivalence with active s_l gradients; flattened-vector oracle for logits/all parameter gradients; shift/segment-boundary ordering plus raw-carry preservation; BF16 forward/backward and zero-matrix finite norm/Jacobian. CPU artifacts under `/data0/xd/bam_diagnostics/mediumprop-unit-h-m-prenorm/`.

Pre-training bets (not results), relative to UnitH with MHA gain reference: retained terminal multiplier1.010, dropped.990; uncertain broad ranges.97-1.04 and.95-1.03, respectively. Favor retained because unit-norm M need not supply optimal V/O strength at every depth, while token-adaptive magnitude control may help the raw-write feedback. Old BAM/RMT normalization losses are a negative prior, not a verdict on this architecture. Unfused startup expected.55-.60step/s versus UnitH.6761 (roughly11%-19% slower); later fused speed must be assessed independently.

Planned formal slots310530/310531, v5p-16, primaryUE5a/backupEW4b selected from this cohort's long completed P0 leases. No training is claimed until source/config/AOT/data/parameter/FIRST_STEP validation. Fixed2k reporting,4k/8k review recommendations require user confirmation before a stop; full training plan13500, native windows200. No intermediate loss diary in exp.py.


## Initial-amplitude calculation

Both arms have identical initial functions, but neither matches the raw-M UnitH read strength. Independent Gaussian approximation (embedding/weight std.006,D1200,R256,N16,gate.1,write epsilon1e-6,content divisor.0016052833): projected content RMS.00124708, post-divisor content.776858, raw address.0000598597, post-address-norm.0597527, seed-M RMS.0185677. Independent adjacent-token half mixing reduces RMS to.0131294, predicting a read multiplier75.9451 (boundary/no-shift53.7789). This changes first-layer Q/K/V/O M-derived components, not the standard RoPE projections.

Component-only validation uses the real embedding writer at formal dimensions with1024 independent Gaussian token embeddings: seed RMS.0185339, shifted RMS.0131199, multiplier median76.2812/P10-P90=71.2637-81.4902. It is not a full-model same-seed/Pile amplitude measurement; the component's RNG path is independent. Script and raw result: `/data0/xd/bam_diagnostics/mediumprop-unit-h-m-prenorm/initial_seed_scale.py` and `initial-seed-scale.json`. Full context-attention residual updates do not inherit a single76x factor: Q/K change attention weights, and each source V has its own matrix RMS divisor. Subsequent layers also receive modified M and vector states. This sizable initial branch-scale change must be considered alongside removal of token-dependent M magnitude feedback.


## Verified native startup

Runtime ee4fa9d4416730e798d2595350141996af0c59b4. Both AOT manifests/seals and all4 worker Git hashes match. Actual FIRST_STEP0, Loaded compiled function, UE5a truepile4096 replica and parameter counts432112210/432112192 verified. Kept/dropped startup20..99 harmonic speeds.484319/.489683step/s (vs historical matched-health Pallas UnitH.676113: -28.37%/-27.57%, separate training TPUs). The.55-.60 unfused speed prior was too optimistic. These timings bundle native read/write with M normalization; do not attribute the full slowdown to the normalization reduction or claim a speed advantage from removing18 scales. Pallas adaptation is pending with Claude; no redundant speed-only profile is launched for this temporary native path.

Trainer310530/310531, both UE5a v5p-16, source immutable. One native report group `mediumprop-unit-h-m-prenorm`, fixed2k cadence,4k/8k reviews requiring user confirmation for a stop, UnitH comparator and MHA gain denominator. Watcher PID2868788 on tpu-ag owns only this group. Per-run startup proofs, native registry snapshots and launch artifacts are in the task diagnostic directory. The user-owned FLEX compiler is retained.


## Added Q/K scale arm

User subsequently requested a third RUN **on the retained-V/O arm only**, explicitly choosing one shared Q/K scalar per layer (not independent Q and K scales). Class `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormQKScaleTruePile`, direct comparator `...MPreNormTruePile`. Existing flag `bam_common_qk_read_scale=True` stores18 signed scalars initialized1, multiplying complete dynamic+static local Q/K after the normalized M read and before concatenating with the unscaled standard/RoPE Q/K. V/O s_l remains independently learnable. Total432112228 (+18 over retained arm); same2510/1128/3904 widths, no MLP repayment. Initial forward at scalar1 equals the retained arm; QK logits scale quadratically with a common scalar when considering the NoPE-NoPE term, while the RoPE contribution is unchanged.

The old QK-scale option required full-head QKNorm. Relax validation to also allow no-QKNorm concatenated heads; when QKNorm is enabled it must still be full-head. Native path already scales local Q/K in this location. A small XLA epilogue completes the existing Pallas bridge when QKNorm is off, multiplying only the local prefixes; new M pre-norm Pallas support remains Claude's task. This does not change either running native control at ee4fa9d.

Third-arm pre-training bet: terminal MHA-referenced gain1.015 relative to its direct retained-pre-norm parent (broad.98-1.05); native throughput within1% of that parent's startup.484319. Extra Q/K freedom can undo inappropriate per-layer changes in the NoPE/RoPE amplitude balance introduced by whole-M normalization, while keeping token-adaptive magnitude control. It does not simply restore the parent token-specific scale. Formal trainer slot310532, sameUE5a/EW4b policy and13500-step plan. All three remain one report group; third-arm curves use their actual direct parent, distinct from the first two UnitH comparisons.


Third-arm startup verified: runtime0ec940cc4070cecd041ba2feaaf856b8c849a75a, FIRST0 and all2 worker hashes, exact AOT/data/432112228 parameters match. Native20..99 harmonic speed.481833step/s, -.51% against retained-parent.484319 (separate TPUs, descriptive timing). Third native group member compares to retained pre-norm+V/O, not the no-V/O arm. All3 native RUNs now train; original2 remainee4fa9d. No parameter-tree or optimizer change is applied to those controls.


## Verified Pallas runtime handoffs

First two RUNs switched on their same UE5a v5p-16 TPUs from native ee4fa9d to fe49cc5ba0d3ff3cff3f7daccdf85f6befb620d1 (branch `claude/mediumprop-m-prenorm-pallas`), replaying checkpoint2000. Native continued through2050/2057 before stopping; complete loss/cache and committed final checkpoints were preserved. Restore used the anchor checkpoint's own data cursor, not a latest-step approximation. Same post-checkpoint steps2001..2030 have mean absolute loss deltas.00003697/.00011807 and maxima.000108/.000484. Both passed; parameter trees and original13500-step LR schedule remain unchanged.

All worker sources and AOT loads verified. Fused runtime enables Pallas core/write/packed keys, named Splash residual and `minimal_save_splash` remat; reference configs above remain native ledger records. At checkpoint-free steps2220..2299, kept/dropped harmonic speeds.626105/.628259step/s, respectively7.40%/7.08% below matched-health UnitH.676113. Separate training TPUs; this combines kernel/remat changes and pre-norm, not an isolated normalization timing. Initial.72-.75 extrapolation from v5p-8 did not hold on v5p-16. The initial<.70 speed-stop directive briefly paused both; user explicitly revoked it. Resumed the same READY TPUs at checkpoints2151/2150, first completed steps2152/2151; neither lease was deleted or counted as a preemption. Continue training despite this speed difference; no repeated mechanical speed stop.

QK-scale RUN switched native0ec940c to18966e45620ff1aa67e10d8d4e04a44cd454e2a5 on its same TPU310532, replaying checkpoint1400 after preserving native post-checkpoint steps. Complete source/AOT/cursor and 30-step overlap proof: `kernel-switch/310532/validation-summary.json` in the task diagnostic directory. Pallas math folds its local Q/K scalar into the read path. All three continue with their existing scientific comparators. Latest user direction: only4k/8k interim reports, no early-stop decisions at either point, otherwise quiet training through the13500-step plan. Claude may provide another optimized AOT; apply any such handed-off hot switch with the same overlap validation.

Operational proofs are under `/data0/xd/bam_diagnostics/mediumprop-unit-h-m-prenorm/kernel-switch/`. Routine plots/loss trends stay in report artifacts/chat rather than intermediate exp.py conclusions.


## Full-remat speed correction

Claude's second handoff f8d07cd3c441fbb15e2cd01572a6fb24e617e6b9 changes only effective remat/saved-Splash-residual settings for these RUNs: `minimal_save_splash` becomes inherited `full`, residual name is cleared. Packed pre-norm/QK-fold core, math, parameters, data and13500-step schedule remain unchanged. Effective inherited-config delta, exact AOT manifests/checksums and pushed commit verified before stopping. Same-RUN/same-TPU overlap anchors: retained3200, dropped3600, QK2600. Old runs retained at least30 post-anchor steps; first new steps3201/3601/2601 and each data cursor verified. All-worker Git sources and AOT loads match f8d07cd.

Matched post-anchor30-step loss max absolute differences are.000061/.000058/.000126 (retained/dropped/QK). Checkpoint-free80-step harmonic speeds.680525/.683427/.677145step/s; all pass the new.66 floor. Relative to historical matched-health UnitH.676113, retained/dropped are+.65%/+1.08% on separate training TPUs; QK is-.50% against the retained arm. These descriptive formal timings are consistent with Claude's paired v5p-16 remat profile. Do not extrapolate the opposite v5p-8 remat ranking to the formal topology. Proofs: `/data0/xd/bam_diagnostics/mediumprop-unit-h-m-prenorm/kernel-switch-full-remat/`. Continue quiet training; only4k/8k interim reports, no effect-based early stopping.


## Terminal result: all three completed13500

All final checkpoints/data cursors13500 and node+queued-resource absence independently verified. TensorBoard synchronized for all three (`final/tensorboard-sync.log`); the local fallback timer had not triggered, so this task performed the explicit repair sync. Final runtimef8d07cd; no4k/8k effect-based stop was executed. All native loss/gain windows extend through13400; final5 means use12600,12800,13000,13200,13400.

| RUN alias | Direct comparator | Final5 gap | Final5 MHA-ref multiplier | Pre-training prior |
|---|---|---:|---:|---:|
| Pre-norm + V/O scale | UnitH | -0.00048112 | 1.00271179 | 1.010 |
| Pre-norm, no V/O scale | UnitH | -0.00024528 | 1.00137702 | .990 |
| Pre-norm + V/O + Q/K scale | Pre-norm + V/O scale | -0.00057508 | 1.00321137 | 1.015 |

Interpretation: M pre-norm produces a clear early advantage, but most of it disappears by the middle of training and does not develop into a material late advantage. Keeping/removing the18 V/O scales ends near parity; deleting them does not establish a useful improvement. Q/K scale has a small mostly persistent incremental benefit over the kept arm, including the terminal windows. Both positive priors were too optimistic; the negative prior for the dropped arm was also wrong, although its small positive result does not support a breakthrough claim. On this evidence, do not prioritize M pre-norm as a2.8B default. The target2.8B has full-head QKNorm, so the small Q/K-scale result is not directly portable evidence. No new experiment is started by this conclusion.

The remaining Q/K benefit is consistent with correcting the changed NoPE/RoPE amplitude balance after normalizing M, but this is a mechanism hypothesis rather than a causal diagnosis. The reliable observation is strong early benefit versus very small terminal benefit; initial loss alone overstated the architectural gain.

Final cumulative plot: `/home/xd/.codex/visualizations/2026/10/12/01a12f6f-6eac-7373-9064-16a8d4349af6/m-prenorm-three-final-20261102-215716-009346.png`. Native reports, exact parameter-tree/source handoffs, resource-absence checks and checkpoints: `/data0/xd/bam_diagnostics/mediumprop-unit-h-m-prenorm/final/`.

Complete READY leases (all v5p-16 in UE5a; no zone switches):

| RUN alias / TPU | Start UTC | End UTC | Duration | End reason |
|---|---|---|---:|---|
| Pre-norm + V/O scale / xd-v5p-16-310530-maxtext | 2026-11-02 15:10:40 | 2026-11-02 16:43:51 | 1h33m11s | service preemption |
| Pre-norm + V/O scale / xd-v5p-16-310530-maxtext | 2026-11-02 16:49:51 | 2026-11-02 18:15:46 | 1h25m55s | service preemption |
| Pre-norm, no V/O scale / xd-v5p-16-310531-maxtext | 2026-11-02 15:11:05 | 2026-11-02 21:25:34 | 6h14m29s | clean completion |
| Pre-norm + V/O scale / xd-v5p-16-310530-maxtext | 2026-11-02 18:21:52 | 2026-11-02 21:48:35 | 3h26m43s | clean completion |
| Pre-norm + V/O + Q/K scale / xd-v5p-16-310532-maxtext | 2026-11-02 15:44:51 | 2026-11-02 21:53:53 | 6h09m02s | clean completion |

Kernel handoffs and the revoked speed pause retained the same physical TPU leases; neither counts as a preemption. Retained arm P2, dropped P0, QK P0. User-owned FLEX compiler remains outside this task's lifecycle ownership and is retained.


## Main integration and2.8B FastKernel ports

Integration worktree `/home/xd/projects/maxtext`, branch `refactor-bam`; no new RUN or TPU assignment.

Integrated the trained f8d07cd native/Pallas whole-M read pre-norm and local-Q/K scale implementation into main, including the forward and backward kernels, zero-matrix/shift-boundary behavior and raw-M carry. The three Medium classes now resolve exactly to their trained f8d07cd effective attributes instead of native-only ledger stubs. Existing training outcomes/hashes are retained.

| New configuration | Configuration parent | Actual history used by compare_runs | Plan | Q/K logit multiplier |
|---|---|---|---:|---|
| `BamPythia2p8BHUnit300BFastKernelMPreNormQKScale` | `BamPythia2p8BHUnit300BFastKernel` | `BamPythia2p8BHUnit300B` |143000|1/sqrt(100)|
| `BamPythia2p8BHUnitShortTrainFastKernelMPreNormQKScale` | `BamPythia2p8BHUnitShortTrainFastKernel` | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne` |28000|1/sqrt(80)|

The FastKernel names are configuration names without independent training registries; the long plan previously ran under its original UnitH RUN and the short plan has the recorded original UnitH history. These are the corresponding same-math baselines, not unrelated ancestors. All other recipe attributes inherit from the selected configuration parent.

Both new classes keep32 unit-initialized signed common V/O scales and add32 unit-initialized signed common Q/K scales (one per layer shared by Q and K). Whole-M read RMS is computed in FP32 over the80x64 logical entries **after** M-shift; all dynamic/static M reads use this normalized view. Carry/writes still use raw M. Epsilon inherits the parent's normalization epsilon1e-5 (the Medium runs use1e-6); no new epsilon mechanism is introduced. The Q/K scalar multiplies only the local80-dimensional read before the complete100-dimensional QKNorm; the standard20-dimensional projections do not receive this multiplier. V/O scaling still multiplies the complete V/localO reads.

Added parameters32, total2775190944 versus parent2775190912 (0.00000488 W_Q total). Ordinary/private/tail GELU widths7808/2944/9984 remain unchanged and128-aligned; no MLP repayment is needed for32 scalars. First31 layers retain single-layer scan and the explicit final layer. Inference M-cache, original initialization, all existing write settings, optimizer/LR plan, health defaults and matched FastKernel controls remain inherited. Signed values and raw-gradient norms for both scale families remain available through generic health; scale WD remains zero.

The existing fused-QKNorm path rejected a common Q/K scalar. Preserve fusion by using v8t's existing per-arm dynamic scales and scaled static Q/K weights, so the read's Q/K outputs already include the scalar when the fused norm computes its full-head RMS. This includes the scalar's square in the denominator and its complete gradient, including zero/negative coefficients; standard/RoPE inputs remain unchanged. No separate XLA norm pass or further kernel ABI change beyond the imported pre-norm/per-arm-scale support is required. Keep2.8B's packed keys, fused full-head QKNorm, saved Splash residuals, minimal_save_splash remat and63MiB VMEM setting; do not transplant Medium's full-remat preference to this different model.

This task writes/merges code only: no2.8B RUN is started and no new target AOT is prepared. CPU interpret checks validate numerical paths; target-topology TPU compilation/VMEM confirmation remains part of the normal future AOT launch gate. Source/config comparison and CPU artifacts are under `/data0/xd/bam_diagnostics/mediumprop-unit-h-m-prenorm/merge-main-*20261103/`.


Integration validation:72 unique pinned-CPU checks passed:47 shared BAM,5 native pre-norm,3 local-Q/K scale,6 Pallas pre-norm,5 existing Pythia Q/K-scale regressions,4 new2.8B checks (both recipes/budget/health, F32 full gradients and BF16 training loss), plus2 existing adapter BF16 training-loss checks after repairing their shared fixture.

The old BF16 training-objective fixture mistakenly treated Transformer's returned per-token cross entropies as logits and applied softmax again. Its training-mode probe now uses actual next-token targets/masks and averages those cross entropies once. BF16 and F32 references use the same representable parameter point. The precision probe is one unpacked sequence; packed integer segment cotangents expose a separate CPU Splash-interpret issue, while kernel boundary/pre-norm tests cover read-shift behavior independently. Existing random-VJP regression behavior/strict bounds are retained.

For the2.8B pre-norm combination's deliberately zero/negative Q/K-scale stress case: complete-gradient relativeL2 versus F32 is native.27263005 / Pallas.07068237; the same Pallas model without the new Q/K scalar is.06889777. Thus the new scalar increases relative-L2 error by.00178460 (0.17846 percentage points, about2.6% relative) over its same-kernel control, and remains within the pre-existing25% relative-control allowance. A fixed5% cap calibrated on a different reduced model would reject the matched control itself, so real-training probes accept the larger of the ordinary5% allowance and1.25 times measured same-kernel control error; the measured values remain explicit. Mean native/Pallas training-CE difference.00455213, full token-loss relativeL2.01084635; F32 outputs/all gradients satisfy the existing tight oracle bounds. These are stress-fixture results, not a training-trajectory or TPU-throughput claim.

Validation manifest and exact metrics: `/data0/xd/bam_diagnostics/mediumprop-unit-h-m-prenorm/merge-main-20261103/validation.json`. No new AOT or training RUN was launched.


## 2.8B Q/K-scale VMEM fix and speed (claude, branch claude/pythia-prenorm-qkscale-vmem)

The per-arm [N,V] Q/K scale port at 1675b1b doubled the fused reverse's per-c scale-gradient scratch (2NP to 4NP rows, 2 MiB); the v5p-256 AOT failed with `Scoped allocation with size 63.71M and limit 63.00M`. CPU interpret checks cannot detect this.

Fix: common Q/K and V/O scalars are passed per head ([N], constant over V). Both v8t kernels fold such scales into the [R,T] gate once; the gate, key and scale gradients follow from the c-summed accumulator after the loop. There is no per-c scratch or multiply. Per-coordinate scales (for example a dynamic-O gamma) keep the per-c path. Math is unchanged: kernel per-head vs broadcast outputs and all gradients agree to 1e-5. The formal 2.8B BF16 probe is unchanged (Pallas .0707, native .2726, CE delta .00455).

v5p-8 paired profiles (8/device, 32 layers):

| arm | 2cecc14 (per-c rows) | bb7cedf (gate fold) |
|---|---:|---:|
| FastKernel parent | .6870 | .7040 |
| parent without V/O scale | | .7030 |
| + whole-M pre-norm | .6760 | .6900 |
| + pre-norm + Q/K scale (formal) | .6700 | .6880 |

The formal class runs 2.3% slower than FastKernel. Pre-norm accounts for 2.0% (read fwd 74.0 to 93.3 ms, bwd 69.4 to 78.5 ms), Q/K scale for 0.3%, and the V/O scale for about 0. Folding the existing V/O scale into the gate also makes the parent configuration faster than its per-c implementation. v5p-256 AOT_READY at bb7cedf for s143000 and s28000.

