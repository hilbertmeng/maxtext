# H: localO gate suppresses the same head's context output

User-authorized fresh13500-step MediumProp experiment. Parent/comparator H:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile`.
Child adds `ContextLocalOComplement` before `TruePile`. RUNid310436.
Family `/data0/xd/mediumprop-common-read-raw-write`, branch
`codex/mediumprop-common-read-raw-write`; main ledger-only. UE5a v5p16,
EW4b backup, retained FLEX v6e AOT; independent new RUN/GCS prefix.

Per token/head, reuse the sigmoid dynamic O read gate g (not read-key scale,
static gate, or depth scale). After Splash AV, before shared W_O and before
attention-write construction, context becomes (1-g)*context. Dynamic O still
uses g, static O is untouched. H's full V/O read scale still applies exactly
once upstream. Both residual update and attention M write consume the changed
sum. Self-diagonal masking, ordinary/memory SwiGLU, and normalized attention
write content remain unchanged. No parameters or initialization copying;
total432112210, widths2510/1128/tail3904. This is separate from the new H
no-extra-memory-write-gate experiment; that flag remains enabled here.

No Pallas/Splash kernel or ABI change: reuse the already-computed O logits
outside the read kernel and apply an elementwise context multiplier after
Splash. Native shares the existing gate through an optional return value.
Crucial backward requirement: O logits receive both dynamic-read and context
suppression cotangents; no stop_gradient on the gate. Unit tests cover an
independent parent reference for residual+M-write and all gradients, scanned
F32/BF16 native/Pallas models, parameter identity, budget and scan-tail health.

Minimal explanatory activation health: only gate mean/std per layer,36 tags
for18 layers, no per-head/column dump or broader internal metrics. train.py
changes solely export those two compact statistics, including scan17+tail;
this also requires the health decoder to respect its explicit final tail.
Generic training healthON, internal/BAM concat and M-shift statisticsOFF.
Tuned Splash/full remat/Pallas read+write/T4096/B16/global128/CP200/TBrotate500.
Infrastructure audited against main09f5fdcf; previous source differences are
family feature defaults/signed-scale metrics, no missing operational fixes.

Prior: terminal gain vs H with MHA reference1.015 [.985,1.045]; speed within1%.
Interpret gate evolution jointly with loss: parameter-free competition can
reduce simultaneous dependence on context and direct M, but static O is an
independent surviving path; this is not a convex mixture of full MHA/localO.

Matched second RUNid310437: append `SelfAttention` before `TruePile`, inherit
the competition class and set only `bam_attention_negative_diagonal=False`.
Same432112210 parameters and widths; direct comparator is the masked
competition RUN. Both export the same36 gate mean/std metrics. Existing
ordinary causal Splash includes the diagonal, including segment-start self
values; no kernel edit. Config/shape, explicit segmented Splash self-row
reference and full F32/BF16 native/Pallas gradient checks gate launch.
Prior vs masked competition: terminal gain1.005 [.985,1.035], speed unchanged.
This isolates restoring self attention under competition; there is no claim
to a complete two-factor interaction estimate without an unmasked H control.

Launches verified: masked runtime d123e5bf on310436, unmasked runtime8e15b747 on retained physical310429 with RUNid310437. Both FIRST_STEP0/exactAOT/source/data/13500 schedule and periodicCP200 verified. Startup20..99: masked.6729step/s (+.452%H), unmasked.672875step/s (-.0037%masked). No kernel edits. Masked4 focused tests+47 BAM regressions, unmasked3 targeted tests passed; full F32/BF16 model gradients included. Actual36-tag TensorBoard verification follows startup artifact collection.
