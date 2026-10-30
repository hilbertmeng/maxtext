# Add dynamic V gamma1 to the original dynamic O gamma.5 baseline

User-authorized RUN; hot-replace complete-localO-head.5 only after CPU/AOT ready.
Worktree `/data0/xd/mediumprop-common-read-raw-write`, branch
`codex/mediumprop-common-read-raw-write`; main exp.py is ledger-only.

RUN `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050LocalVDynamicNormScale100TruePile`.
Parent/comparator `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile`.
Gain reference `BamMHAMediumPropC256TruePile`. Fresh step0,13500 total steps;
new RUN id310434, retain physical UE5a v5p16 xd-v5p-16-310432-maxtext from
complete-localO-head.5 at the verified handoff. No checkpoint-tree reuse.

Keep original O dynamic address gamma[N16,V32] initialized.5. Add independent
V dynamic address gamma[N16,V32] initialized1, directly learned/signed,
after address RMSNorm and before M read. Static V/O reads unchanged, no H
external scale, no complete-O head scalar, no depth-decay initialization.
Q/K and standard attention self masking unchanged. Preserve the original
normalized embedding/attention/private-write recipe (distinct from H's raw
embedding/private-write recipe). MLP2510/1128/tail3904 unchanged; last
attention/private writers absent. Added9216 parameters, total432130624,
+.00213% or .0064 W_Q total versus parent, negligible, no MLP repayment.
Gamma uses existing scale weight-decay exemption.

Motivation: initialization probes showed context aggregation of localV
rather than localO dominates attention write content. The extra control
addresses the primary feedback path while preserving the already-positive
O gamma.5 modification. Old failed integrated V/O depth-gamma runs replaced
H's full-read scaling and used different initializers/write norms; they do
not isolate this local increment. Medium experiments prepare2.8B migration;
value/output scales are not canceled by QKNorm.

Existing native and v8t Pallas dynamic V/O gamma implementations are reused;
read+write enabled, no kernel adaptation or new ABI. Focused CPU scope:
formal parameter/config/initialV-neutrality, same shared initial weights
without copying; scanned native/Pallas full F32/BF16 model gradients with
perturbed V/O gammas, static paths and standard attention intact. Parent
shared paths/full47 regressions already verified; no production math edit.
BF16 uses the existing F32-reference parity gate. Full-shape exact sealed
AOT required before retaining-TPU handoff.

Prior: terminal MHA-reference gain vs original O gamma.5=1.015 [.995,1.04].
Expected speed within1% of parent .689788step/s; record matched health and
cross-lease limitation. General training healthON, internal/BAMOFF, tuned
Splash, full remat,17-layer scan+tail, T4096/B16 perdevice/global128,
CP200/latest2/TBrotate500 inherited. Existing generic parameter/raw-gradient
norms cover both9216-element gamma fields; report all18-layer V/O parameter
RMS (not activation contribution ratios), without per-column activation tags.
Audit through main d667c525: only experiment metadata/docs changed since
preceding runtime infrastructure audit; no omitted operational fixes.
Artifacts `/data0/xd/bam_diagnostics/mediumprop-common-read-raw-write/dynamic-v1-o05/`.
