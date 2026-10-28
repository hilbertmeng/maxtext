# Independent ordinary/private MLP input mixtures

Worktree: /data0/xd/mediumprop-mlp-input-mix
Branch: codex/mediumprop-mlp-input-mix
Artifacts: /data0/xd/bam_diagnostics/mediumprop-mlp-input-mix
Fresh fork of mainc7382488; its training/checkpoint/TB infrastructure is inherited.
Direct parent: BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile
Gain reference: BamMHAMediumPropC256TruePile

| RUN | Gate parameterization/initial opening | Private write-control input | TPU ID | Terminal ratio bet |
| --- | --- | --- | ---: | ---: |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG099WriteControlHTruePile` | sigmoid .99 | original norm(h) | 310392 | 1.005 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG099WriteControlMixTruePile` | sigmoid .99 | mixed private norm | 310393 | 1.010 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG100Scale2WriteControlHTruePile` | 2sigmoid 1 | original norm(h) | 310394 | 1.020 |
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75MLPInputMixG100Scale2WriteControlMixTruePile` | 2sigmoid 1 | mixed private norm | 310395 | 1.025 |

## Contract

Let x be the pre-attention residual, a the complete attention output including
MHA+localO after the original W_O, and h=x+a. One zero-initialized D-by2 projection
of the existing MLPnorm(h) generates two scalar residual gates per token. Ordinary input
is N(g_x*x+a); private input is N(g_mx*x+a). Both use the same existing
MLP RMSNorm parameter set/epsilon/statistics dtype, evaluated on different inputs.
The outer identity residual remains h; no mixture changes this vector residual.
Terminal has only an ordinary MLP, so only D-by1 plus one bias are allocated.

For sigmoid gate range1, kernels zero and biaslog99 initialize each residual gate to.99;
initial inputs slightly favor attention versus parent (BF16 gate quantization applies).
For range2, kernels/biases zero initialize gates toexact1; original shared-seed
parameters and initial function are tested against parent, for both writer controls.
Writer-control variantH retains N(h) for private dynamic address and write gate;
variantMix reuses N(private mixture), without another norm. Attention write controls,
contents and all readers remain the parent architecture. These are independent-writer
experiments, not children of the separate shared-general writer RUN.

The briefly proposed six-gate split of MHA/localO was cancelled by the user before
implementation, because independently projecting those components adds W_O compute.
All four final configurations use two gates and no extra attention projection.

No Pallas kernel/interface change: routing is outside the v8t read/write kernels.
Verify all new backward paths by full-model native/Pallas logits/all-gradient tests
for each of the four configurations, using perturbed learned gate parameters.

## Budget and runtime

Extra17*(1200*2+2)+(1200+1)=42,035 parameters, .00973% of432,112,192.
Use the user's explicit no-repayment instruction: MLP widths2510/1128/final3904 unchanged,
which also preserves the exact-unit variant's initial parent function. Total432,154,227.
All four parameter-tree shapes identical; no unused final private gate and no new
norm scales. Single-layer scan17 plus explicit final layer, no terminal M writes.

T4096/global128/per-device16, v5p16, full13500 optimizer/LR/AOT schedule. CP200/latest2,
TB rotation500, tuned Splash1024/fused backward, Pallasv8t48MiB/fused inputs.
Generic training healthON, internal/BAM metricsOFF except targeted input-mixture health.
Only normalized coefficient fraction (not activation energy) for attention, 1/(g_x+1) mean/std for each branch/layer is recorded:
70 new scalars total, no per-head/column dumps and no redundant common-scale metrics.
Existing MHA-reference ratios come from native run_registry; no local gain arithmetic.

Expected startup throughput .66..69step/s versus direct parent's.686768; no separate
profile TPU requested. Terminal ratio bets in table are priors, not measured effects.
Mixed write controls may align content/address/gate specialization; zero-logit range2
may learn ratios faster than near-saturated sigmoid. Neither is guaranteed to help.

## Launch gate and monitoring

Use one exact-source CPU gate for all four: focused budget/real-input/shared-norm/
initial-parent-function/health-export checks, all four full-model Pallas gradient
comparisons, full BAM regressions and summary-writer rotation tests. Shared receipt
is keyed to the sealed commit and source hashes. Compile four exact v5p16/s13500 AOTs
on the verified retained FLEX compiler through its worker lock (serial compiles).
Launch distinct user-owned SPOT training prequeues; preparation failures release only
owned training resources, never the borrowed compiler.

Only5000-step public review/stop decision; no routine progress reports or repeated
agent logtails. Mechanical health/recovery stays active. Review each run against the
direct parent using the complete cumulative trajectory from600, not rank-only selection.
If several beat baseline, retain all credible winners. Surviving runs keep plan13500.
The shared-general writer RUN has the same quiet5000-review reporting policy, with
slightly more patience because its larger ordinary MLP may benefit later.

Final user simplification: only x is gated; complete attention outputa is fixed.
The two parameterizations now differ in range as well as learning dynamics: sigmoid
can only attenuate x, while2sigmoid can amplify x and relatively suppress attention.
The original four-gate implementation was replaced before sealing/launch.
