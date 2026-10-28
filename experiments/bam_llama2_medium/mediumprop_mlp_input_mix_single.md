# One residual gate: ordinary versus dedicated memory MLP

Worktree: /data0/xd/mediumprop-mlp-input-mix
Branch: codex/mediumprop-mlp-input-mix
Artifacts: /data0/xd/bam_diagnostics/mediumprop-mlp-input-mix-single
Implementation parent: sealed33e6bc1806f2c49619f529a28cba7eedd4a3c76e.
Main infrastructure audited through3d933e27: no main-only changes since branchpoint
c7382488 in train/compile/checkpoint/optimizer/config/summary-writer paths.
Main's347126a5 input-mix rejection is a ledger-only guard, not a runtime fix to port.
Preserve original numerical lineage and operational TBrotation500/CP200/latest2.

Common prefix: BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75
New suffixes:
- MLPInputMixG099OrdinaryOnlyTruePile
- MLPInputMixG099MemoryOnlyWriteControlMixTruePile

## Question and exact contract

The user requests two fresh training runs from the sigmoid.99/mixed-control
experiment: replace two per-token residual gates with one, acting only on the
ordinary MLP or only on the memory MLP. The gate uses original MLPnorm(h), h=x+a,
with zero kernel and biaslog99, g=sigmoid(z). Complete attention outputa includes
MHA+localO after the existing W_O; no additional attention projection is introduced.
Outer identity residual staysh, all input normalizations reuse the existing MLPnorm
parameters/epsilon/dtype while computing their own statistics.

Ordinary-only: ordinary inputN(g*x+a); private MLP content/address/write gate all
use originalN(h). The sole gate is present in all18 layers, including terminal.
Memory-only: ordinary inputN(h); private content/address/write gate all use
N(g*x+a). Gates exist only at17 live private writers; terminal has no dummy gate.
No changes to M-shift, readers, attention writes, kernel algorithms, or final dead writes.

The requested no-repayment exception remains: widths2510/1128/final3904 unchanged.
Ordinary-only extra18*(1200+1)=21618, total432133810 (.0150125 W_Q).
Memory-only extra17*(1200+1)=20417, total432132609 (.01417847 W_Q).
These sum to the two-gate parent's42035 parameters, not an MLP-budget change.

Controls in compare_runs for each:
1. Direction SOTA without MLP input gates: prefix+TruePile, complete13500.
2. Direct two-gate sigmoid.99/mixed-control parent:
   prefix+MLPInputMixG099WriteControlMixTruePile, stopped5160 (mature5000).
The latter isolates removing one branch's gate; it cannot replace the stronger
full-training SOTA baseline. Gain reference remains BamMHAMediumPropC256TruePile,
with native run_registry triples registered against both controls.

Prelaunch terminal MHA-reference R bets versus no-gate SOTA: ordinary-only.99,
memory-only1.01. Better memory-only is a hypothesis from preserving the residual
processor while adapting the dedicated memory updater; paired health observations
do not establish causality. Startup prior.675..695step/s versus parent.686768.

## Runtime and gates

T4096/global128/per-device16, v5p16, plan13500 original LR/AOT schedule. UE5a chosen
from recent long comparable leases in regional history; SPOT only, no passive zones.
TPUs xd-v5p-16-310396-maxtext and xd-v5p-16-310397-maxtext, task-owned new prequeues.
Single-layer scan17+explicit final; full remat; tunedSplash1024/fused backward;
Pallasv8t48MiB read+write/fused inputs. Gate routing stays outside the kernels;
verify full-model native/Pallas forward and every gradient with perturbed learned gates.
Generic training healthON, internal/BAM statsOFF except36/34 gate coefficient-share
mean/std scalars (ordinary18/private17 layers). Do not export missing branches or
invent a terminal private health record. No per-head/column statistics.

Reuse verified idle user-owned FLEX_START llm-jax-v6e-1-0 EW4a worker0 for AOT,
with actual resource-type/process proof. Borrow without lifecycle ownership;
per-worker lock serializes exact-config AOTs while CPU/prequeues proceed concurrently.
Shared commit/source-keyed CPU gate: focused6 including backward through private M,
new full-model Pallas2, shared BAM47 and summary-writer5. Preparation failures release
only newly owned training prequeues, never the retained compiler.

After actual AOT/FIRST_STEP/data/source/step100 verification, only5000-step review,
quiet intermediate waits; retain all credible baseline beaters. Surviving runs may
continue through13500 and close out with cumulative native gap/ratio figures from600.
