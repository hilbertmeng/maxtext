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

After verified startup, report the two runs together every2000 steps, per the user
updated cadence. Keep the5000-step decision review and all credible baseline beaters.
Use native200-step windows and cumulative gap/ratio plots from600 against both
registered comparators; no change to the full13500 training schedule.

## Verified startup

Runtime2f36e3ac65837e5a16085c5b97095c0f2a682fa6, clean pushed worktree above.
Both parallel launchers exited0 after actualFIRST_STEP0; exact v5p16/s13500 AOT,
worker Git/data path and step100 gates independently verified. Actual worker
configuration logs confirm steps and full LR schedule13500. Source/manifest and
CPU gate did not change between preparation and launch.

CPU60 PASS once, receipt reused across both launchers: focused6/newPallas2/fullBAM47/
summary-writer5. The old missing branch selector falls back to both, with historical
input-routing/shared-norm/outer-residual/initial-parent-function/health coverage.
The two retained-worker AOTs serialized without adopting compiler lifecycle ownership.

| Variant | TPU | Startup20..99 step/s | vs two-gate parent | vs no-gate SOTA |
| --- | --- | ---: | ---: | ---: |
| ordinary only | 310396 | 0.676375 | +2.63% | -1.51% |
| memory only + mixed controls | 310397 | 0.673325 | +2.17% | -1.96% |

Each timing uses80 training steps20..99, excluding the compilation/first-step period.
Parent generic health matches; targeted gate statistics differ70 versus36/34, so
these are descriptive startup comparisons, not isolated operator cost measurements.
No unrelated timing/profile TPU was created. The slightly low memory-only startup
relative to the.675 lower prior does not justify kernel profiling or a restart.

Both native MHA-reference gain triples registered against both saved comparators.
The two-gate parent's mature endpoint is5000, so it cannot supply comparisons beyond
that point; the no-gate SOTA provides the full13500 reference. Do not interpolate it.
Updated main infrastructure audit37d53edf confirms f3f52892/f06b0eb2 add only exp.py
fork classes, with no later operational fixes to port. See infrastructure-audit.json.
Main keeps ledger-only configurations and its existing unsupported-runtime guard;
this is not a merge of the experimental routing implementation.

Receipts: cpu-pass.json, startup-registries.json, startup-ID.json,
startup-summary.json, launch-results.json and step100-ID.txt under the artifact root.

## Ordinary-only closeout after5k review

Ordinary-only stopped5229/13500, P1. The early advantage reversed by2.2k;
subsequent loss deficits persisted without recovery. Final five native windows
4400..5200 average gap+0.00413664, native MHA-reference R0.98163805 versus
no-gate SOTA. No compensating throughput/parameter/cache benefit. Terminal
R bet.99 was unsupported through the measured stop; no full endpoint was measured.

The latest user report policy retains only the two experiment curves against
no-gate SOTA; live registry comparators/triples were narrowed accordingly. Historical
two-gate controls and all their cached losses remain available, but are not current
published baselines. The memory-only experiment remains active under its full plan;
its result will be recorded at closeout rather than adding live progress diaries.

CP5229 committed, no observed lost tail, final TB SYNC_OK, node and queued-resource
absence independently verified. v5p16 remained UE5a with no passive candidates or
zone switch. Complete READY leases UTC (final end is manual stop request):

| # | Start | End | Duration | Reason |
| --- | --- | --- | --- | --- |
| 1 | 2026-10-28T02:41:43Z | 2026-10-28T03:20:50Z | 0h39m07s | preempted |
| 2 | 2026-10-28T03:26:45Z | 2026-10-28T05:06:49Z | 1h40m04s | run_stop |

Proofs: ordinary-final-proof.json, ordinary-final-registry.json, ordinary-final-leases.json,
ordinary-final-pending.json, ordinary-final-node.txt, ordinary-final-queue.txt,
closeout-ordinary-summary.json; TB receipt closeout-20261028T050646163288Z.log.
