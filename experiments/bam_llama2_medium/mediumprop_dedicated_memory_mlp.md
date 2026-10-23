# Per-layer dedicated memory MLPs

Implementation worktree `/data0/xd/mediumprop-dedicated-m-mlp`, branch
`codex/mediumprop-dedicated-m-mlp`; forked current `refactor-bam` at781a86a6.
Parent: `BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile`.

Separate ordinary residual SwiGLU and M-content SwiGLU, both consuming the same
post-attention normalized residual. The memory branch never enters the vector residual;
its output is reshaped to16x75. Preserve the parent's independent R256 GELU addresses,
per-head write gates (initial .1), attention writes, generalized reads and read-only M-shift .5.
No added write-amplitude scaling. M-cache75x32 is unchanged.
L0–L16 write memory; L17 has no memory branch/address/gate because it has no downstream reader.
The parent's otherwise unused L17 attention writer is preserved for a controlled comparison.

| RUN | Ordinary widths | Memory widths | Ordinary params | Memory params | MLP write controls | Total |
|---|---|---|---:|---:|---:|---:|
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile` | all3200 | L0–15:511, L16:505 |207360000|31251600|7786000|432113280|
| `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile` | L0–16:2535, L17:3806 | L0–16:1140 |168843600|69768000|7786000|432113280|

Both totals are parent+1600, MHA−7920; respectively +.001111/−.0055 W_Q (D²=1440000).
Ratio2:1 includes address/gate parameters: a body ordinary branch9126000 versus
memory4104000+controls458000=4562000, ratio2.000438. L17's freed update budget goes to
its ordinary MLP, so the aggregate ratio need not equal2. Exact integer width residuals
are included in the totals above. Body16 layers scan as8 identical two-layer blocks;
L16 and L17 compile separately for the last writer/terminal budget exceptions.

Both classes use the original T4096/global128/per-device16/13500 schedule, BF16,
full remat, scan, tuned Splash1024/fused backward and Pallas v8t/48MiB/fused inputs.
Generic training health ON; internal activations and BAM-specific health OFF;
checkpoint200/latest2/keep0, TensorBoard rotation500.
Infrastructure audit: source is the latest main781a86a6, preserving current train/AOT/WD,
checkpoint and complete rotating-writer path; validate summary-writer regression with CPU gate.
No model/numerical initialization changes beyond introducing a separately initialized branch.

Direct comparison: original single .5 parent only; matched MHA is the gain-ratio reference.
Report early each1000 steps with200-step loss windows; review2800/5000, plan13500.
Pre-run terminal gain-ratio bets: MHAWidth1.01, Ratio2to1 1.04 versus parent;
throughput bets −3%/−5%. Historical parent timings have different health/kernel settings,
so formal speed claims require a matched timing control.

Resource ownership: new v5p-16 IDs310317/310318, us-east5-a; no passive candidates.
Retained FLEX compiler llm-jax-v6e-1-0/europe-west4-a worker0 is borrowed, never released.
Independent launch preparations share its worker-level AOT lock (serial compiles).
Preparation artifacts: `/data0/xd/bam_diagnostics/mediumprop-dedicated-m-mlp`.

Sealed runtime18a9d9cf5e912ffbc5549e65746f261b722c28ec; both FIRST_STEP and step14 gates passed.
Startup throughput~.707step/s, with the declared metric/kernel settings. CPU gate:47 shared
BAM regressions +4 dedicated-path checks +5 summary-writer tests, all passed.
Both AOTs compiled serially on the retained worker and launched on their assigned TPUs.
