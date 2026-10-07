# MediumProp QK57 final-block independent LocalVO reads

RUN: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLastBlockSeparateVOTruePile`.
Direct parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`, actual runtime23c692b.
Implementation worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.

Keep M75×32/C8, QK57+RoPE18, W_O, existing independent static V/O reads and gates,
and independent MLP→M writes at zero-based1/4/7/10/13/16.
Only layers15/16/17 gain their own zero-initialized V dynamic C8 read key instead
of borrowing O's dynamic read. Both reads share the original C8 compressed M.
Their key normalization, scales and gate recipes stay unchanged.

The first five blocks scan under `layers`; the final three layers execute under
`final_block` with actual static indices15/16/17 and the continuing vector/M carry.
This avoids allocating dormant independent keys in earlier layers. Prefix widths
stay3901/3774/3901; final widths3858/3731/3858 repay the added153600 parameters
per layer using43 exact MLP units. Added BAM parameters460800=.32W_Q;
MLP reduction464400=.3225W_Q. Expected432092528 total parameters, parent−3600.
The split changes the initialization RNG tree; initialization methods and numerical
hyperparameters stay the same, but this is not an identical-initial-tensor control.

Focused CPU gates verify parent budget preservation, prefix/final parameter scopes,
all18 health layer indices, six MLP writers, actual QK75/V75 forward dimensions,
finite gradients, and consumption of all three independent V keys.
Inherited read/gate/write health remains on; add final-block ungated V/O read RMS
and cosine to distinguish whether the learned reads diverge.

Pre-run bet: final13500 latest5 gap−.002 vs direct QK57 parent; speed.516 vs.520
step/s (−.8%) on UE5a v5p-16. Normal reports~1000 steps; review2800/5000.
