# Original Medium T2048 matrix-value BAM control

Implementation worktree: `/data0/xd/medium-t2048-k64-embed-qk48`;
branch: `codex/medium-t2048-k64-embed-qk48`.
RUN: `BamMediumT2048K64EmbedVOnlyQK48`.
Hot-switch source: `RMTMediumPropT4096TruePileK48EmbedUnembedDirect32NoOLLF`
on its UE5a v5p-16 after AOT and CPU validation.

This maps the MediumProp QK57 architecture to the original Medium geometry:
24 layers, D1024, 16 heads of 64, T2048, batch32, eight LLF blocks,
M64x32/C8, matrix QK48 plus independent RoPE16. It preserves embedding
matrix seeding, matrix-only L values, static V/O reads, dynamic C8 VO reads,
and fetchedO in F layers. P_loc GELU bottleneck stays R256.

The L and F MLP widths are 3366 and 3025. The full parameter tree has
411,623,824 parameters, 7,568 above `Llama2Medium` (0.00184%). Every L/F
layer has the same 12,786,640 parameters. Full-size shape and train-step
tracing pass. The focused CPU test verifies the 64x32 seed, QK48+RoPE16,
absence of L W_V, and presence of F W_V.

Direct loss comparisons use original T2048 Pile records and
`Llama2Medium`, the historical fetchedO K48 shared-rank4 BAM, and
`RMTMediumT2048AllLocalK48EmbedUnembedDirect32`.
Pre-run bets: versus K48 BAM, final loss gap -0.010 to -0.020; versus
all-local dynamic RMT, near parity to a modest win for BAM. Expected steady
speed ~0.60 step/s versus K48 0.638 and RMT 0.392; recheck on the retained TPU.
