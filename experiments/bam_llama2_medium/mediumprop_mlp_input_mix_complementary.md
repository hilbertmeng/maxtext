# Complementary residual-attention gate, one MLP branch at a time

Worktree: /data0/xd/mediumprop-mlp-input-mix
Branch: codex/mediumprop-mlp-input-mix
Runtime:0244976dc73c24d9b0b67d57a6746b0960b8c130
Artifacts: /data0/xd/bam_diagnostics/mediumprop-mlp-input-mix-complementary
Main retains ledger-only configs; input-mix runtime implementation is isolated.

## Exact change and comparisons

Original h=x+a and the outer residual stay unchanged. A single per-token gate
from the existing MLPnorm(h) is g=2sigmoid(z), zero kernel and bias. The selected
MLP consumes Norm(g*x+(2-g)*a), reusing the existing norm parameters and epsilon.
Only the ordinary MLP changes in one arm; only the dedicated MLP and all its write
controls change in the other. The latter has no unused terminal memory gate.
Initial model function and all shared initial parameters equal the no-gate SOTA.
The coefficient attention share is (2-g)/2, now able to range from0 to1.

Common prefix:
BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75

| New suffix | Intended TPU | Direct extra control |
| --- | --- | --- |
| MLPInputMixComplementaryOrdinaryOnlyTruePile | xd-v5p-16-310398-maxtext | MLPInputMixG099OrdinaryOnlyTruePile |
| MLPInputMixComplementaryMemoryOnlyWriteControlMixTruePile | xd-v5p-16-310399-maxtext | MLPInputMixG099MemoryOnlyWriteControlMixTruePile |

Both retain the full-training no-gate direction SOTA prefix+TruePile. All native
gain ratios use BamMHAMediumPropC256TruePile as reference. Do not extrapolate the
stopped ordinary .99 control beyond its actual mature endpoint. Store both
comparisons; routine public overlay keeps only two new curves versus no-gate SOTA
unless a direct .99 comparison is needed to answer the gate-parameterization question.

Prelaunch terminal native gain bets vs no-gate SOTA: ordinary1.00, memory1.01.
Speed prior .67..69step/s, close to corresponding .99 gates; no kernel ABI changes.
Extra21618/20417 gate parameters remain explicitly exempt from repayment. Widths
2510/1128/final3904, totals432133810/432132609 unchanged from respective .99 arms.
Only36/34 targeted mean/std attention-share health scalars; generic training healthON,
internal NN/BAM statisticsOFF, no new per-head or coordinate statistics.

## Runtime and launch gates

UE5a v5p16 selected from recent comparable long leases, with no passive zones.
T4096/global128/per-device16, plan13500, full original LR schedule, CP200/latest2,
TBrotation500, scan17+explicit terminal, full remat, tunedSplash1024/fused backward,
Pallasv8t read+write/fused inputs48MiB. Pallas kernels remain the verified parent.

Main infrastructure audited through66f2d96c: no main-only operational changes since
c7382488 in train/compile/checkpoint/optimizer/config/summary-writer dependencies.
train.py tree differs because main lacks this family's exporter; no upstream fix
is omitted. Preserve numerical ancestry and the complete inherited TB rotation.

Pinned CPU gate8 checks: formal full-model budgets, actual branch inputs/control
routing and gradients, actual scan health export, parent initial function/parameters,
perturbed full-model JAX/Pallas logits and all gradients for both new arms, and the
original two-gate routing/health regressions. Exact sealed config check also required.
Retained user-owned FLEX_START llm-jax-v6e-1-0 EW4a worker0 is ACTIVE and idle;
borrow for two serial AOTs through per-worker lock, never adopt/delete the compiler.
CPU/prequeue preparation proceeds independently; launch only after all gates pass.

Report together every2000 steps with cumulative native loss-gap/gain plots from600.
Keep5000 as a decision review, not an automatic endpoint. Retain every credible
baseline-beating arm rather than forcing winner-only selection. Existing memory-only
.99 run continues on its own full13500 schedule.

## Verified startup

Both launchers exited0 after actual FIRST_STEP. Independent step100 verification
confirmed first completed step0, exact0244976d worker Git, exact v5p16/s13500 AOT
loaded, actual zone-local TruePile4096 data argument, total13500 training/LR schedule,
and80 timing samples from steps20..99. Both native MHA-reference triples are
registered against no-gate SOTA and the corresponding sigmoid.99 single-gate control.
No compiler resource was created or deleted; retained-worker locking serialized AOTs.

| Variant | TPU | Startup20..99 step/s | vs sigmoid.99 | vs no-gate SOTA |
| --- | --- | ---: | ---: | ---: |
| ordinary only | 310398 | 0.677525 | +0.17% | -1.35% |
| memory only + mixed controls | 310399 | 0.669950 | -0.50% | -2.45% |

These are descriptive formal-training startup windows from different leases, not
paired operator timings. Corresponding sigmoid controls share36/34 targeted health
scalars; no-gate SOTA lacks these extra summaries. The small timing differences do
not justify another profiling TPU. Receipts: startup-ID.json, startup-registries.json,
startup-summary.json, step100-ID.txt, cpu-pass.json, launch-results.json, and sealed
runtime/source hashes. Main stores ledger classes; implementation remains isolated.
