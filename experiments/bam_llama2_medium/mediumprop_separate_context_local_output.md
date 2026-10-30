# Separate context/localO output bases and context-only attention M write

User-authorized new Medium RUN, derived from H. Runtime worktree
`/data0/xd/mediumprop-common-read-raw-write`, branch
`codex/mediumprop-common-read-raw-write`; main exp.py is ledger-only.

RUN:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedSeparateContextLocalOutputTruePile`.
Comparator H:
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile`.
Gain reference `BamMHAMediumPropC256TruePile`.

Keep H's learned full V/O read scalar at each layer. Context attention restores
causal self-attention, including the first token. Let c be its context output
and o the scaled complete localO read. Update the residual by W_context(c) +
W_local(o), using independently sampled output matrices from the same original
initializer distribution. No copied kernels, no parent-function matching.
Only c supplies attention write content; retain the original content RMSNorm,
addresses, gates and existing v8t write kernel. Small-MLP writes and embedding
writes are H's originals. localO can still affect M indirectly via the residual
and small MLP; M->localV->context->M remains. The experiment removes direct
localO attention feedback rather than all M feedback.

D=1200, 16*75=D. Each extra bias-free output projection costs D^2=1 W_Q;
repay locally by400 ordinary SwiGLU hidden units (3D*400=D^2).
Ordinary widths2110 for the first17 layers, tail3504; small MLP1128 unchanged.
Total432112210, exactly H; last attention and small-MLP writers remain absent.
Extra dense forward FLOPs equal the dense FLOPs removed from the ordinary MLP;
there is one more output matmul invocation, so timing still needs measurement.

Pallas read and write kernels remain unchanged. Split output projection and
context-only factor selection are XLA operations between them; both native
and fused paths use the same semantics. Formal tuned Splash, T4096/B16 per
core/global128, full remat, scan17+tail1, CP200/latest2, TBrotate500 preserved.
Generic health ON, internal/BAM activation health OFF; retain H's18 signed
scale parameter summaries. Do not add internal activations for this launch.

Before launch assess the current depth1 integrated-gamma RUN for a retained-TPU
hot replacement, as requested. Finish CPU/AOT first and compare its latest
cumulative gap/gain trajectory against H and the depth2 sibling before deciding.
Do not stop it during preparation. Final handoff uses the native journaled
hot_switch_run workflow with a new RUN and new parameter tree, from step0.
Compiler is the existing guaranteed FLEX llm-jax-v6e-1-0, retained/never deleted.
Artifacts: `/data0/xd/bam_diagnostics/mediumprop-common-read-raw-write/separate-context-local-output/`.

Pre-run prior: terminal gain vsH1.015, plausible range.98--1.05; startup speed
within~2% of H. Independent residual bases and removal of direct localO feedback
may help; restored self-attention and a400-unit ordinary-MLP reduction may hurt.
The first experiment intentionally tests the integrated routing change rather
than attributing each effect separately.

Validation: exact formal parameter tree432112210; independent initial output
kernels; explicit output einsum reference; localO-key intervention changes the
residual but leaves attention M write identical; first-token context nonzero;
F32 scanned native/Pallas full-gradient maxrel1.68e-6; BF16 maxleaf5.85%, global
relative gradient L2 1.15%. Full47 CPU BAM regressions passed. Same audited
runtime infrastructure as H/coarse-scale arms, checked through main d007a1fd;
no omitted logging/checkpoint/AOT/optimizer operational changes.
