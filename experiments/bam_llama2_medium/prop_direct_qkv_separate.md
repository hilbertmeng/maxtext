# MediumProp DirectC8 independent Q/K/VO compression

Worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
RUN `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionTruePile`.
Owned trainer: `xd-v5p-16-1009-maxtext`, UE5a preferred; retained EW4a non-preemptible `llm-jax-v6e-1-0` is borrowed for AOT only, never recycled.

Parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile`, runtime `3f72aac`.
M75×32, C8, 18 all-local layers, QK57+RoPE18, independent MLP write every third layer; V/O dynamic keys shared, gates independent.
Q and K each get their own 32×8 compression. Existing compression remains for V/O. New matrices clone the same layer's original matrix, preserving the complete initial forward pass. Full-M static reads, keys, gates and all read/write scales are unchanged.

Extra parameters: 18×2×32×8=9216=.0064 W_Q (W_Q=1200²). Total432103040; parent432093824, MHA432121200. MLP widths remain [3901,3774,3901]. Extra compression MACs per token/layer: 2×75×32×8=38400=.02667 W_Q. Resume-only checkpoint retention: last2, no permanent accumulation.

Bet: terminal loss −.003 vs DirectC8; speed −1% to −3% vs parent's UE5a .508 step/s with identical generic/concat health. Plan13500 steps, r200 windows, reporting every~1000 steps, first review2800.

Focused CPU gate: full parameter count, old parameters and initial output bitwise equality, finite consumed gradients for both new projections, gradient sum conservation at original shared projection; small scanned model.

Launch: runtime `e6f1ef91bc92b51a53fbd926413ca1ebcc111b17`, FIRST_STEP verified, AOT loaded, UE5a data path verified. Stable speed .499 step/s (median observed steps26–99), −1.8% vs parent .508. CPU gates passed, initialization RNG draws avoided using a cloned params variable. Local preparation logs: `/home/xd/.local/state/maxtext-parallel-launch/BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8SeparateQKVProjectionTruePile-20261009T134021Z`.
