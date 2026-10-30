# H: remove the redundant output write gate of the memory SwiGLU

User-authorized training, fresh13500 steps. Primary question: the small MLP
already has a fine-grained SwiGLU W_g; does an additional token/head sigmoid
gate on its write output help, or duplicate/restrict that control?
This is not primarily an initialization-amplitude experiment. Its consequence
is nonetheless explicit: previous initial write gain~.1 becomes exactly1.
No fixed compensation, copied initialization, or added normalization.

Parent/comparator is completed H:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile`.
Child adds suffix `NoMLPWriteGate` before `TruePile`.
Worktree `/data0/xd/mediumprop-common-read-raw-write`, branch
`codex/mediumprop-common-read-raw-write`; main exp.py remains ledger-only.
RUNid310435, UE5a v5p16 with EW4b backup; not reusing a stopped run's prefix.

Remove the head gate's D*16 weights and16 biases from each of17 writers.
Saved326672 parameters, refund17*5*3600=306000 to ordinary MLPs:
ordinary2515, memory1128, tail3904; total432091538, H-20672 (-.004784%).
The per-layer unused1216-parameter rounding remainder is recorded rather
than introducing special tail redistribution. H's18 learned full V/O read
scales, embedding fixed divisor.0016052833, raw memory content divisor1,
normalized attention content, address RMSNorm, attention/embedding gates
and absent terminal writers all remain as in H. SwiGLU W_g remains trained.

Native writes use ones. Existing v8t Pallas writer receives constant+inf
logits, sigmoid=1 exactly and zero unused logit cotangent; no gate parameters
or kernel ABI edit. Validate with independent ungated mixed-writer reference,
multi-batch/multi-tile full gradients including finite zero logit derivative,
full scanned F32/BF16 models, retained W_g gradients, parameter tree/budget,
and existing default-on BAM regression plus TB rotation tests.

Infrastructure audit against main09f5fdcf: train.py/base.yml differences are
this family's previously introduced signed scale metrics/feature defaults,
not later omitted operational fixes. TB writer begin_step/rotation500,
checkpoint/schedule/optimizer infrastructure unchanged from reviewed main.
Generic training healthON; internal/BAM activation healthOFF. No new tags:
existing memory-MLP parameter/raw-gradient norms and signed H read scales
cover the relevant paths. Tuned Splash, Pallas read+write, full remat,
scan17+tail, T4096/B16 per device/global128, CP200/latest2.

Prior: MHA-reference terminal gain relative H1.025 [.99,1.06], steady speed
within1% of H (.669875step/s reference, separate matched-settings lease).
Primary evidence is loss/gain trajectory vs H; learned read-scale and
memory-MLP norm trajectories help interpret stronger ungated write feedback.

Launch ed83fadb: exact AOT/source/data/13500 schedule, FIRST_STEP0, periodicCP200 verified. All18 actual initial H read scalars matched depth formula; memory SwiGLU W_g raw gradients finite and nonzero. Startup20..99 .6737875step/s, +.584%H (matched settings, separate leases). Full5 focused tests and47 BAM regressions passed; F32 all-gradient maxrelative1.93e-6, BF16 fused gradient relative L2 vs F32 1.445%.
