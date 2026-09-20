# Shared C8 with K64 and fixed QK32 concat

Implementation: `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
RUN `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64TruncateMLPPerLayer`.
TPU ID `qkstatic-vo-c8-k64`, expected name `xd-v5p-16-qkstatic-vo-c8-k64-maxtext`.
Direct baseline `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer`
(5d535d3, UE5a .6890 steps/s, generic + concat health ON).

Raw M32x32 becomes64x32; compressed M32x8 becomes64x8. All row reads remain pruned.
Q/K retain the shared rank4 dynamic basis and separate zero-init, ungated static keys.
Both dynamic and static column results retain only their first32 coordinates.
Q/K remain BAM NoPE32 concatenated with standard RoPE32; standard projections, QKNorm,
RoPE frequencies and sqrt(head_dim64) attention scale stay unchanged.
LocalV/O use one shared C8 read including the original shared gate, now covering all64
head coordinates. FetchedO also uses all64. Standard V remains64. Write data is the full
O64; its RMS now normalizes over64 instead of32 (a real mathematical change).

No new parameters: both models411598464; MLP[2879,2879,2874]. Static keys remain32x16,
compressed read keys remain8x16, P_loc and write gates unchanged, write RMS has no scale.
Additional0 W_Q; raw and compressed M storage double. K-dependent write/compression/VO
contractions and Fetched M transport grow; standard attention and MLP do not.
Extra K coordinates are not directly read by Q/K, but can affect later layers through
V/O, residuals and write normalization. No added learned projection or V concat.

Validation: actual full24-layer parameter audit, full train-step/health trace, pinned BAM
suite, targeted same-parameter K32/K64 QK+RoPE equivalence, nonzero static/dynamic reads,
extra-coordinate interventions, full64 V/O and writeback, gradients and FetchedO checks.
Artifacts `/data0/xd/vo-c8-k64-{audit.json,audit.log,trace.log,tests.log}`.

Bet vs direct SharedC8: final loss gap -.004, speed -5%. Health settings/count unchanged.
Train from scratch,13500 steps,checkpoint200. Formal primaryUE5a,backupsUC1a/EW4b;
AOT primaryEW4a,backupsUC1a/UE5a. No hot replacement of another active experiment.

## K64 QK48 follow-up

RUN `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48TruncateMLPPerLayer`,
TPU ID `qkstatic-vo-c8-k64-qk48`. Direct baseline is the K64 QK32 RUN above.
Keep M64x32/C8 and full64 V/O reads/write. Expand dynamic/static BAM QK from32 to48;
shrink standard Q/K from32 to16, with NoPE48/RoPE16 (regenerated16-dimensional RoPE).
This is a combined QK-allocation, RoPE and MLP-allocation experiment.

Each layer saves2*1024*16*16=524288 standard Q/K weights (0.5 W_Q); all24 save12 W_Q.
Return the budget to each layer's MLP, rounded to the nearest integer width against
the original MHA per-layer target12847104. L widths2879→3050, F2874→3045 (+171 each).
Each width costs3072 parameters; resulting total411623040, +6784 (+.001648%) vs MHA,
and +24576 vs K64QK32 due to integer rounding. No hardware-friendly rounding.
L layers are64 below their target, F layers976 above. Cache unchanged vs K64QK32.
Bet vs K64QK32: finalgap-.004, speed-1%; same920 generic+concat health scalars.
Artifact prefix `/data0/xd/vo-c8-k64-qk48-`; same acquisition and training schedule.
