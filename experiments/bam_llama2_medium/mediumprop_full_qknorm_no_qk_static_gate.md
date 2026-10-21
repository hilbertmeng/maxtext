# Full-head QKNorm without static-Q/K-read gates

Authorized2026-10-21. RUN
`BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile`.
Implementation: `/data0/xd/mediumprop-qknorm`, `codex/mediumprop-qknorm`;
main exp.py is ledger only. Parent formal runtime03c8ade, full75 QKNorm,
original MediumProp TruePile4096/single read-only M-shift.5.

Remove only generalized static Q/K read gate kernels and biases. Retain static
Q/K address vectors, dynamic Q/K reads and gates, V/O static/dynamic gates,
full-head RMS QKNorm before partialRoPE18, independent attention/MLP writes.
Parent gate is1.1*sigmoid(Wx+b), W=0, b=log(9), opening.99;
its effective read multiplier divides by.99 and therefore starts at1.
Removing the gate preserves initial read amplitude, up to BF16 rounding.
The sigmoid argument initially corresponds to.9, not.99; it is not a completely
saturated plain sigmoid. No new health metrics; inherit health OFF.

D1200/L18/N16: saves18*2*(1200*16+16)=691776 params (.4804 W_Q).
User explicitly requests no MLP reimbursement: retain[3802,3676,3802].
Total431422604 versus parent432114380, saving691776 (.4804 W_Q, .1601%).
This isolates gate removal with identical remaining parameter shapes; initial
operator parity is checked with shared seeded parameters.
T4096/global128/per-device16/v5p16, full remat,13500 steps, inherited original
optimizer/LR/data order. CP200/latest2/keep0/finalforced; TB rotate500.
Splash q/kv1024,kv_compute512,q_dkv/kv_dkv/kv_dkv_compute1024,fused backward.
User requests this tuning for future experiments as well.

Direct baselines: full75 QKNorm parent; no-QKNorm single.5;
MHA and MHA QKNorm; Mudd. Original MHA is the common gain reference.
Register gain triples versus full75, noQK and Mudd. Native stride200,
report1000. Historical full75 uses512/nonfused Splash, so raw speed differences
are not pure gate-removal effects. Prior for matched-kernel/health throughput:
+0.5 to+1.5%; with tuning, expect speed near the tuned standard18 formal control
(.54 step/s), rather than the parent's historical.4743.
Terminal prior versus full75 loss-.002, range[-.006,+.003];
MHA-referenced gain versus full75 about1.01. This tests whether learned static
Q/K read suppression helps after full-head normalization, not whether the
initial static M-read amplitude should be increased.

Infrastructure audit: reviewed main9428eff0 against worktree885dc779 and
historical parent03c8ade. Summary writer, max_utils, optimizer and AOT math
are unchanged and include bounded GCS rotation with begin_step. Main-only
train changes concern disabled health / new Pythia multi-tail decoders;
checkpoint changes add optional exact-step retention, not requested here.
Preserve legacy latest2/keep0 normalization and finalforced saves. No relevant
operational fix is omitted; do not merge unrelated model changes. CPU gate:
exact removed tree/budget, fixed-shape initial parity, scanned finite active
gradients and causality, full75 placement, summary-writer rotation regressions.
Pallas generalized static gates honor the same removal (ones for Q/K,
V/O computed normally); formal path stays historical JAX, not Pallas.

TPU ownership planned: xd-v5p-16-310270-maxtext, primary us-east5-a,
backups europe-west4-b/us-central1-a. Recent same-shape formal UE5a lease
has no preemptions through9k; EW4b diagnostics also stable. Retained FLEX
compilers checked: no READY host in EW4a/UC1a/UE5a; use managed AOT queues.
