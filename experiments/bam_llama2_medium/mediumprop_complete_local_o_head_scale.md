# Complete localO read scale per layer/head, initialized .5

User-confirmed raw strongest-combination base, no H scale and no V scaling.
Worktree `/data0/xd/mediumprop-common-read-raw-write`, branch
`codex/mediumprop-common-read-raw-write`; main exp.py remains ledger-only.
RUN `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalOReadScalePerHead050TruePile`.
Planned13500 on UE5a v5p16 `xd-v5p-16-310432-maxtext`, EW4b recovery backup.

Remove the learned coordinate dynamic-O address norm gamma. Instead create
`o_read_scale[N]`, directly learned signed, all .5, independently at each
actual layer, including the final layer. Multiply COMPLETE dynamic+static
localO output after the existing read kernel and before adding context or
constructing attention write content. Other reads remain unchanged. Retain
all parent normalization, writers, tuned Splash, full remat, scan17+tail1.
Both Pallas read/write enabled, kernels unchanged, gradient through existing
XLA post-read epilog. Ordinary/small/tail MLP2510/1128/3904 unchanged.
432112480 parameters, versus432121408 coordinate-gamma parent: -8928,
-.0062 W_Q total, negligible; no MLP reallocation.

Compare only the original coordinate gamma.5 parent, using
BamMHAMediumPropC256TruePile as the gain reference. The unsuccessful historical
dynamic-head gamma control is excluded from live comparisons and plots.
Full-O scaling additionally controls the static read, so its effect differs
from scaling only the dynamic address even at the same head granularity.
Prior vs coordinate gamma.5: terminal R1.005 [.99,1.02], speed within~2%.
No target2.8B training requested. This tests the simpler amplitude-control
alternative before choosing a target architecture.

Generic healthON/internal+BAMOFF. Existing coefficient norm/raw-gradient
summaries plus18 signed head means; derive head dispersion from RMS/mean.
No per-head activation summaries. CP200/latest2 and TBrotate500 retained.
Use retained guaranteed FLEX llm-jax-v6e-1-0/EW4a for AOT without lifecycle
ownership; formal spot TPU prequeue independent of compiler.

CPU coverage: exact formal budget/config/WD/health; signed asymmetric
independent static+dynamic output reference with all gradients; full-model
scanned F32 native/Pallas; BF16 with same-parameter F32 reference when the
existing fused F32 norm backward differs from native BF16 rounding. Keep
full-gradient relative L2 below3%, unchanged mixed per-leaf bound and the new
O coefficient's pure relative bound; any exception to native parity must be
at least as close to F32 truth. Parent fa0a53c full47 regressions already pass;
this change only adds an opt-in post-read O scalar path, with no kernel,
normalization, optimizer or scan changes. No repeat of unrelated full suite.
All operational fixes audited through main d368f5f8; no omitted runtime fixes.
Artifacts `/data0/xd/bam_diagnostics/mediumprop-common-read-raw-write/complete-local-o-head-scale/`.
