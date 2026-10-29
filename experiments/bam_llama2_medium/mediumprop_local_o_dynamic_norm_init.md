# Dynamic local-O RMSNorm scale initialization pair

RUNs share the prefix
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75`:

| Suffix | Effective initial gamma | Runtime / branch | Trainer | Startup step/s, 20..99 |
|---|---:|---|---|---:|
| LocalODynamicNormScale050TruePile | .5 | 2662b397ff307eb3792563e043c766fb5b1252aa / codex/local-o-dynamic-norm-init | xd-v5p-16-310384-maxtext | .6897875 |
| LocalODynamicNormScaleTruePile | 1 | 881f99174e3401af211301af468a2e98ea226491 / claude/o-dyn-norm-scale-pallas | xd-v5p-16-310385-maxtext | .6911875 |

Both trainers are in us-east5-a, plan13500. Parent startup speed is .6867681;
these initial timing differences are within about1%, so they establish no material
speed penalty. Both exact v5p16/s13500 AOTs loaded and FIRST_STEP gates passed.
Startup provisioning was preempted once for each arm and recovered automatically.
These are startup records; live state belongs in registry/private artifacts.

Direct comparator for both is the same prefix plus `TruePile`, the completed
M75x32/allR256 independent-VO/negative-diagonal/full-NoPE 2:1 parent.
MHA gain reference is `BamMHAMediumPropC256TruePile`.
Before-run terminal R bets: init.5=1.01, init1=1.005.
Reports every2k, cumulative gap/gain plots from600 with both lines overlaid;
no2800/5000 loss-stop reviews, endpoint13500.

Each layer learns gamma[16,32] multiplying only the dynamic O address after
RMSNorm and dynamic gating. Static O, V, Q and K are unchanged. Gamma=1
preserves the initial parent function. Effective gamma uses the configured
constant directly, without a +1 offset. Dynamic sigmoid is bounded by1;
the static read gate uses1.1*sigmoid. Static O has no address RMSNorm.
There is no total-localO scalar and no dropout in this pair.

9216 additional parameters give432121408 total (+.00213%); negligible overhead
is not refunded. Ordinary/private/tail MLP2510/1128/3904 remain unchanged.
Existing scale WD exemption covers gamma. Generic parameter/gradient health ON,
internal and BAM activation statistics OFF; no per-column statistics added.
T4096/global128/per-device16, scan17+tail1, fullremat, tunedSplash,
v8t48MiB/fusedinputs, Pallas read+write, CP200/TBrotation500.
Dataset resolves to the us-east5 TruePile4096 replica.

Runtime881f991 is the supplied Pallas implementation in
`/data0/xd/o-dyn-norm-scale-pallas`. The .5 arm uses the isolated worktree
`/data0/xd/local-o-dynamic-norm-init`; its only change over881f991 is the
initializer subclass. Main exp.py classes retain native reference flags
PallasFalse; formal runtime commits explicitly enable read and write kernels.
The presence of these ledger classes does not imply that the Pallas adapter
has been merged into main.

Infrastructure audit against mainc049b9d: train/train_compile/checkpointing,
optimizer/maxtext utilities, summary writer, pyconfig and runtime defaults
match; TBrotation500 and begin_step remain intact. Focused launch gates passed:
effective initializer/parent identity and five TB writer regression checks.
The supplied runtime additionally passed kernel forward/all-gradient checks
over shared/independent VO, masks and tile64/256; whole-model scan native vs
Pallas maxrel1e-5, and dropout combination4.6e-6.

The borrowed FLEX_START compiler llm-jax-v6e-1-0, europe-west4-a worker0,
was verified idle and retained. No lifecycle ownership was adopted.
AOTs are under the common compiled_trainsteps bucket, runtime881f991 and
2662b39 respectively, jax081-i0ae3f58-c17f538a/v5p-16/s13500.
Immutable startup/config/AOT evidence and live monitor cursors:
`/data0/xd/bam_diagnostics/mediumprop-dynamic-o-norm-init-050-100/`.

## Init1 closeout and init.1 hot replacement

Init1 stopped at committed9025 on2026-10-27 10:24:29UTC at the user's request.
The initial near-parity period gave way to a stable deficit from~2.6k; last five
complete windows8200..9000 have gap+.00193832 and MHA-reference gain.98970219
against the direct M32 parent. The positive1.005 prior was unsupported through
stop; the13500 endpoint was not reached. Parameter norms/gradients were finite
and active. No inference-cache or parameter-budget gain offsets the loss deficit.

CP9025 commit marker and stopped registry were independently verified. TensorBoard
SYNC_OK is in closeout-20261027T103446809189Z.log. The v5p16 resource was retained
and transferred to init.1; no node/queue deletion was requested. All-worker old
process exit and checkpoint commit preceded the fresh launch. Old leases, UTC:

| Start | End | Duration | Exit |
|---|---|---:|---|
| 2026-10-27 06:29:10 | 2026-10-27 06:34:12 | 5m02s | preempted |
| 2026-10-27 06:39:26 | 2026-10-27 10:24:29 | 3h45m03s | hot-switch boundary |

New RUN: same common prefix plus `LocalODynamicNormScale010TruePile`.
Runtime34d61c204b29b7fc47e29048869d77124c3cd089, same isolated worktree
`/data0/xd/local-o-dynamic-norm-init`, branch `codex/local-o-dynamic-norm-init`.
Its only model change over init.5 is effective gamma initialization.1. Pallas
read+write still use the verified881f991 kernel; the later7161995 speed variant
was not introduced. Main's configuration remains a native reference.

Logical controller310386 uses physical xd-v5p-16-310385-maxtext in UE5a. The
new model trained from actualstep0, CE10.852695; exact AOT load and FIRST_STEP
were verified. Startup20..99 mean.687325step/s,80 samples, near init.5.
CP400 was committed during startup verification. Init.5 continues unchanged at
runtime2662b397 on its own trainer.

Params432121408 and MLP2510/1128/tail3904 are unchanged. The direct controls are
init.5 and the original M32 parent; both gain triples use original MHA. Before-run
terminal bet versus init.5 is1.005; speed expected unchanged within~1%.
Plan13500, native200 windows and public2k reports; tunedSplash/v8t48/fusedinputs,
generic healthON/internal+BAM OFF, CP200/TB500. No numerical kernel hot-switch
comparison is required for a fresh initialization experiment.

Infrastructure audit against maindf3f267c found no changes in train/compile,
checkpoint/optimizer/config/summary-writer defaults. Focused initializer/parent
identity and five summary-writer tests passed. Source/config seal passed before
AOT on the verified idle FLEX_START llm-jax-v6e-1-0 EW4a worker0, retained.
AOT state: tpu-ag `/home/lishengping/xd/projects/aot_runs/34d61c2-ce6479c6.json`.
Fresh-source/AOT/FIRST_STEP, old closeout, ownership and report artifacts:
`/data0/xd/bam_diagnostics/mediumprop-dynamic-o-norm-init-010-hot/`.

## Init .5 completed 13500 (2026-10-27)

Runtime2662b39, v5p-16 xd-v5p-16-310384-maxtext, UE5a throughout, P3.
Committed CP13500, independent node/queue absence checks and final TensorBoard SYNC_OK verified.
Final5(12600..13400) vs the registered strongest M32 parent: gap-.00169960,
MHA-reference gain ratio1.00966719, matching the prelaunch1.01 bet. Initial deficit
turned positive around1.6k; a small gain persisted over middle/late training to the end.
Parameters432121408 and widths2510/1128/tail3904 unchanged; startup.689788step/s.

Complete18-layer health profiles are retained, not only first/last/median summaries.
At13450, parameter RMS spans 0.492544..0.516574; early layers often fall below.5 while
deep layers rise, and the layer median remains.499830. All18 raw-gradient tags are
finite and nonzero. These are RMS values, not signed means or actual O/context output ratios.
Compared with the failed total-O scalar controls, this result is consistent with retaining
the independently learned static-O contribution while moderating dynamic O. It does not
establish a monotonic benefit from ever-smaller initial scales.

Final loss/gain figure: `dynamic-o-norm-13400-5600-20261027-124910-848722.png`.
Full layer-health figure: `dynamic-o-norm-full-layer-health-20261027-125158-512099.png`.
Source snapshots/provenance and final resource/lease/TB proof live in
`/data0/xd/bam_diagnostics/mediumprop-dynamic-o-norm-init-010-hot/`.

All READY leases, UTC (final endpoint is clean completion):

| Start | End | Duration | Exit |
| --- | --- | --- | --- |
| 2026-10-27 06:29:36 | 2026-10-27 06:35:51 | 6m15s | preempted |
| 2026-10-27 06:42:35 | 2026-10-27 07:02:31 | 19m56s | preempted |
| 2026-10-27 07:10:53 | 2026-10-27 07:26:47 | 15m54s | preempted |
| 2026-10-27 07:34:04 | 2026-10-27 12:49:25 | 5h15m21s | run_stop |

## Init .1 completed 13500 (2026-10-27)

Runtime34d61c2, physical310385/logical310386 in UE5a, P0. CP13500 marker,
independent node/queued-resource absence and final TensorBoard SYNC_OK verified.
The large initial deficit versus init.5 disappeared around4k; final5(12600..13400)
gap-.00002128 and MHA-reference ratio1.00012024 show terminal parity, missing the
1.005 prior. Versus the original strongest M32 parent, final5 gap-.00172088 and
ratio1.00978469 nearly match init.5's gain; reducing initialization further does
not add meaningful terminal benefit here.

At13450, the full18-layer gamma RMS range is.107835(L1)..153956(L16),
median.123247, L0.118693 and L17.142541; every raw-gradient norm is finite/nonzero.
The learned values remain far below init.5 despite tied loss. These parameter norms
are not actual localO/context activation ratios. Params432121408 and MLP widths
2510/1128/tail3904 remain unchanged; startup.687325step/s.

One complete READY lease:2026-10-27 10:25:18..16:01:54UTC,5h36m36s, clean completion.
The resource assignment started10:25:15. Final native reports, full-layer health,
CP/TB and independent resource/lease proof: the010-hot artifact root above,
`tenth-final-metrics.json`, `scale-health-13400-13400.json`,
`tenth-final-resource-proof.json` and `tenth-final-leases.json`.

## Per-head scale configuration

Configuration-only addition on main `/home/xd/projects/maxtext`, `refactor-bam`:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050PerHeadTruePile`.
Direct comparator is the completed coordinate-wise init.5 class above. The new
`bam_local_o_dynamic_norm_scale_per_head=True` stores one scalar per layer/head,
shape[16] instead of[16,32], initialized.5. It scales dynamic O only after address
RMSNorm; static O, V and other reads are separate. Each head can change amplitude,
not its relative address coordinates.

18*16=288 scales versus9216: total432112480, comparator-8928 (-.002066%,
-.0062 W_Q). This negligible difference leaves ordinary/private/tail MLP widths
2510/1128/3904 unchanged. Existing scale WD exemption and generic parameter/
raw-gradient norms remain; offline RMS uses sqrt16 per layer instead of sqrt512.

Broadcast at the native/Pallas interface preserves the existing[N,V] kernel ABI.
The existing coordinate-gradient VJP is summed over V for each scalar; no kernel
source change or Claude handoff is required. Pallas read+write, tuned Splash and
generic healthON/internal metricsOFF are enabled on the new class. The old
coordinate classes retain their original parameter shapes and defaults.

Pinned CPU verification: all56 checks pass (3 new granularity/actual-budget/
full-model Pallas checks,6 existing coordinate-scale checks,47 shared BAM
regressions). Perturbed per-head values match tied coordinate values exactly
in native forward; scalar gradients equal the sum over coordinate gradients.
Full scan native/Pallas worst relative parameter-gradient errors2.01e-6 plain
and4.52e-5 with O read-weight dropout; head-scale errors5.94e-7/5.44e-7.
Receipts: `/data0/xd/bam_diagnostics/local-o-head-scale/validation.json` and the
focused/coordinate-regression/shared-regression subdirectories.
