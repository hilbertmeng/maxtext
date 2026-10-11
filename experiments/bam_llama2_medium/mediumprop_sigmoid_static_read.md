# MediumProp generalized static read: plain sigmoid gates

Implementation worktree: `/data0/xd/mediumprop-full-m-read-gelu128`.
Branch: `codex/mediumprop-full-m-read-gelu128`; sealed runtime `25cd9df96b96de8478c29028320330199ee59c76`.
RUN: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadSigmoidStaticGateTruePile`.
Training resource: `xd-v5p-16-310120-maxtext`, UE5a SPOT; compiler borrowed from FLEX_START `llm-jax-v6e-1-0`, EW4a, never lifecycle-owned.

Direct parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile` (runtime `64e3fc3`, completed13500).

Only change: for Q/K/V/O static read gates, initialize zero gate kernel and bias `log(.99/.01)`;
compute `gs = sigmoid(logit)` and `static_read *= gs` without any compensation.
Old gates used `1.1*sigmoid(logit)` with initial raw opening .99 and divided read scale by .99.
Dynamic keys, pre-RMS biases, shared R256 GELU reader, independent destination gates, M75x32,
QK57+RoPE18, writing, initialization RNG sequence and all health statistics are inherited unchanged.
Generalized write gate code retains the old behavior.

Both variants have432111104 parameters; MLP widths remain `[3813,3686,3813]` repeated6times.
Parameter delta:0W_Q. Initial static V read is .99 times the parent; zero-initialized static Q/K/O remain zero.
Gate derivative at initialization changes from .1 effective derivative to .0099, so this is also a comparison of gate learning behavior.

Bet at13500: loss +.001 relative to the old generalized read; matched-health speed unchanged.
Normal Medium windows200, reports about1000steps; monitor relative direct-parent loss, gain factor over shared R256,
static V/O gate opening and static/dynamic read ratio. No gate-benefit inference from initial read amplitude alone.

Focused CPU gate checks the exact full parameter/sharding layout, actual static-read amplitude and saturation,
unchanged write gates, finite forward/backward, nonzero new gate gradients and exported health.
CPU checks, retained AOT, and training-only queue run concurrently; FIRST_STEP remains the startup gate.
Artifacts: `/data0/xd/bam_diagnostics/mediumprop-sigmoid-static-read/`.
