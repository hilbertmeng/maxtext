# Medium/XL B normalized-write and zero embedding seed combination

User-directed2026-10-01. Worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`.
Both formal trainers queued only in UE5a. Borrow verified idle STANDARD/guaranteed
EW4a `llm-jax-v6e-1-0` for serial AOTs; never enroll it in cleanup.

Medium RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero`,
trainer `xd-v5p-16-2910012-maxtext`,13500 steps. Derive from trained SeedZero,
add only shared normalized attention/MLP write contents. MLP4100;431888672 params.
Keep full-M per-sublayer learned scales, raw-M vector RMSNorm, NoO, full-row
writes, tail32 C8 reads, Direct32 unembedding, normalized shared embedding
contents and zero-initialized learnable embedding address. Layer static write
addresses retain original Gaussian initialization.

XL RUN `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit`,
trainer `xd-v5p-32-2910013-maxtext`,50000 steps. Derive from failed XL normalized
combination, change only embedding static address initialization to zero.
MLP6643;1432453720 params. M60x96,C10,R384,tail40 proxy,RoPE24; pure JAX and
ordinary layer scan on both scales. No new matrix/vector gain or clipping.

Checkpoint **save** intervals remain200/250. Explicit **retention** overrides
inherited keep_period0/max_to_keep2: Medium retains every1000, XL every4000,
and each retains two recent checkpoints. Full state includes params,
optimizer and step; Pile progress cur_files is written alongside each save.
No special checkpoint steps. Exact terminal checkpoint retained on clean exit.

Medium baselines: SeedZero, SharedWriteNorm, B, originalNoO. Bet terminal
versus SeedZero-.002; speed.373step/s flat. Review2800/5000; routine1000-step
batches. XL baselines: B, old combination, TruePileMHA, Mudd. Bet versus B
-.008@17500; speed.307step/s flat. Initial200/400 windows then500-step windows;
reports about2000 steps. Critical outcome: whether Mudd-relative MHA gain
keeps decaying late, not just early absolute loss gain or finite training.

Keep existing dynamic/static read amplitudes, write ratios/cosines by layer,
read/write gates, embedding amplitudes/gate, raw matrix proxy/tail RMS and
full-M learned-scale statistics. Do not add Gram/SVD to every training step;
retained checkpoints support paired offline final/penultimate-layer probes.

Targeted CPU gates: both full parameter trees and effective flag scope,
zero seed/nonzero layer addresses, learned scales initializedone, finite
scanned gradients including seed/layer scale/address, dynamic-health presence,
shared layer content normalization equation and amplitude invariance.
