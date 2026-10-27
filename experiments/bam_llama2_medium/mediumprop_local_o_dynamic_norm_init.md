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
