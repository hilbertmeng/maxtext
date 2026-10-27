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

## Verified startup

Sealed runtime33e6bc1806f2c49619f529a28cba7eedd4a3c76e on the branch/worktree above.
Implementation remains isolated; main carries the configuration ledger and rejects
unsupported input-mix execution rather than silently running the parent.
All four exact v5p16/s13500 AOTs, actual worker Git/data, FIRST_STEP0 and step100 verified.
One commit/source-keyed CPU gate passed60 tests: focused4, full-model Pallas4,
fullBAM47 and summary-writer5; shared receipt reused by the four launchers.
The borrowed FLEX compiler was retained, four training prequeues are separately owned.

| TPU / gate / control | Startup20..99 step/s |
| --- | ---: |
| 310392 / sigmoid.99 / H | .669775 |
| 310393 / sigmoid.99 / Mix | .659025 |
| 310394 / 2sigmoid1 / H | .668988 |
| 310395 / 2sigmoid1 / Mix | .663125 |

Direct parent.686768; these include70 targeted health scalars absent in the parent,
so they are descriptive startup comparisons rather than isolated operator timings.
Exact-unit parent equality is tested in F32; BF16 compiler reductions can give small
first-step differences and do not establish bitwise trajectory equivalence.
Launch/source/timing/CPU receipts: startup-registries.json, startup-ID.json,
startup-summary.json, cpu-pass.json and launch-results.json under the artifact root.

## Common5k review and closeout

All four stopped after the authorized common5000 review. Every arm gained early,
but reversed by~2.2k and developed a sustained deficit; no credible winner or
recovering arm was discarded. Wider/faster2sigmoid did not help overall, contrary
to the prior. All positive terminal bets lacked support through5k; a13500 result
was not measured. Performance has no compensating parameter/cache gain and
startup throughput is slower than parent. Sources and native5k report/figure are
saved in the artifact root; no hand-computed gain definition is used.

| Gate / write controls | Committed stop | 4200..5000 mean gap | Mean native R |
| --- | ---: | ---: | ---: |
| sigmoid.99 / H | 5279 | +0.00326460 | 0.98567077 |
| sigmoid.99 / Mix | 5160 | +0.00440108 | 0.98068689 |
| 2sigmoid1 / H | 5175 | +0.00840096 | 0.96318144 |
| 2sigmoid1 / Mix | 5225 | +0.00418648 | 0.98164625 |

The strongest interaction is within2sigmoid: feeding writer controls the same
mixed private input reduces the deficit considerably; within sigmoid.99 it does
not help. This supports content/control alignment when changing input strongly,
not a universal benefit from mixed controls.

Targeted health confirms gates learned, not stuck at zero kernels: at5k ordinary
MLP coefficient-share alpha=1/(1+g_x) has layer medians.712/.737/.776/.786
in the table order, with middle layers commonly.7..88. The private branch medians
are.688/.635/.749/.709. These are normalized scalar coefficient fractions, NOT
activation-energy ratios or causal attributions; a=the complete original attention
output. The nominal unit parent corresponds to.5. All full layer mean/std snapshots
at0/1k/2k/4k/5k are retained in gate-health-review5000.json,70 tags per RUN.

A concrete follow-up hypothesis, not an executed experiment: preserve ordinary
MLP inputN(h), and change only the dedicated M-MLP input together with its write
controls. This preserves the residual-processing path while allowing the memory
updater to specialize. The current four arms changed both, so they cannot identify
which branch causes the shared deficit; middle-layer ordinary-input attention
bias is evidence for prioritizing this distinction, not proof of causality.

All four checkpoint commit markers independently verified. Node and queued-resource
absence independently checked for all eight resources; final TB sync has four
SYNC_OK entries. Each used v5p16/UE5a throughout with P0; no region switches or
passive candidates. Full leases UTC, each ends with the manual review stop:

| Gate / controls | READY start | End | Duration |
| --- | --- | --- | --- |
| sigmoid.99 / H | 2026-10-27T16:16:22Z | 2026-10-27T18:32:15Z | 2h15m53s |
| sigmoid.99 / Mix | 2026-10-27T16:17:19Z | 2026-10-27T18:32:17Z | 2h14m58s |
| 2sigmoid1 / H | 2026-10-27T16:19:07Z | 2026-10-27T18:32:20Z | 2h13m13s |
| 2sigmoid1 / Mix | 2026-10-27T16:16:24Z | 2026-10-27T18:32:23Z | 2h15m59s |

Proofs: final-registries.json, final-leases.json, final-resource_proofs.json,
final-checkpoint-markers.json and final-closeout_summary.json. Final sync receipt:
/home/xd/.local/state/maxtext-tensorboard-sync/closeout-20261027T183211876378Z.log.
Implementation remains isolated; no merge of the failed architecture is implied.
