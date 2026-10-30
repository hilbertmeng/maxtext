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

Reporting correction requested by user: all competing RUN curves use the
common H comparator and MHA gain reference. SelfAttention still inherits
the masked competition architecture, but no longer plots that sibling as
its comparator. Genealogy and the displayed common benchmark are distinct.

Third authorized arm RUNid310438: masked competition plus original
dynamic-O address gamma[16,32] initialized.5. Append
`LocalODynamicNormScale050` before `TruePile`. This scales only the dynamic
O normalized address, not static O, not V, and not the external 1-g context
multiplier. +9216 parameters, total432121426; widths unchanged. Existing
Pallas gamma ABI reused, no kernel edit. Keep the36 mean/std gate metrics
and generic parameter/gradient norms for gamma. Test static O untouched,
context suppression unchanged, uncopied shared initial weights, full-model
F32/BF16 native/Pallas gradients with perturbed gamma. No production-code
change since parent d123e5bf, so focused configuration-combination CPU gate.
Terminal gain prior relative H1.01 [.99,1.04], speed within1% of masked
competition. Read old completed H+gamma.5 as mechanistic context; ordinary
group plots contain only these three RUNs against H.

Matched second RUNid310437: append `SelfAttention` before `TruePile`, inherit
the competition class and set only `bam_attention_negative_diagonal=False`.
Same432112210 parameters and widths; genealogical parent is the masked
competition RUN; routine direct comparator was switched to H per user instruction. Both export the same36 gate mean/std metrics. Existing
ordinary causal Splash includes the diagonal, including segment-start self
values; no kernel edit. Config/shape, explicit segmented Splash self-row
reference and full F32/BF16 native/Pallas gradient checks gate launch.
Prior vs masked competition: terminal gain1.005 [.985,1.035], speed unchanged.
This isolates restoring self attention under competition; there is no claim
to a complete two-factor interaction estimate without an unmasked H control.

Launches verified: masked runtime d123e5bf on310436, unmasked runtime8e15b747 on retained physical310429 with RUNid310437. Both FIRST_STEP0/exactAOT/source/data/13500 schedule and periodicCP200 verified. Startup20..99: masked.6729step/s (+.452%H), unmasked.672875step/s (-.0037%masked). No kernel edits. Masked4 focused tests+47 BAM regressions, unmasked3 targeted tests passed; full F32/BF16 model gradients included. Actual TensorBoard verified36 finite gate mean/std tags per RUN across all18 layers; initial mean.0503309965. Both actual18 initial signed H depth scales verified. Evidence: context-local-o-complement/initial-health/proof.json under the diagnostic artifact root.

Gamma masked arm6355ba82 startup verified: FIRST0/exact source/AOT/13500 schedule/data/CP200. Speed20..99 .6695375step/s, -.500%masked sibling / -.050%H. All18 gamma initial RMS.5 and signed H depth coefficients verified,36 actual gate TB tags and finite nonzero gamma gradients.

Self-attention control closeout (2026-10-30): stopped8347/13500 at8k review.
CP8347 committed, all TensorBoard synced, node and queued resource independently
absent; P0. READY12:15:19→15:46:02UTC (3h30m43s), assignment12:15:16UTC.
Relative H, early positive gap grew to~.008 then narrowed. At8.2k gap+.0022998,
R.988413; latest5 mean gap+.00304916/R.984974 (native MHA reference).
The8k/8.2k rebound was retained in the conclusion; full13500 outcome is untested.
Scientific result through this endpoint: restoring self attention did not beat H
or the masked competition sibling; no material parameter/cache/speed benefit.
Closeout artifacts: context-local-o-complement-self-attention/closeout under
`/data0/xd/bam_diagnostics/mediumprop-common-read-raw-write/`.

Masked competition plus gamma.5 closeout (2026-10-30): stopped8546/13500 at
registered8k review. CP8546 committed, TB SYNC_OK, node+QR independently absent;
P0. Assignment13:23:10UTC; READY13:23:13→17:02:59UTC (3h39m46s).
Large initial deficit narrowed by4k, but relative gains then plateaued near.975
through8.4k. Final gap+.005592/R.971845; latest5 mean+.00504008/R.97492135.
Intermittent late catch-up was retained in the decision; full13500 outcome is
unmeasured. No observed benefit from combining this gamma with H competition.
Atmatched8k, gamma layerRMS.472-.521 and dynamic gate layermeans.057-.439;
selected compact metrics finite, whole18-layer profiles saved in gamma-review8k
artifacts. This is not evidence against the separately validated2.8B gamma.2.
Artifacts: context-local-o-complement-gamma050/closeout and
report-groups/gamma-review8k-20261030-165549 under the diagnostic family root.
