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
