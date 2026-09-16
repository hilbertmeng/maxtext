# LLF first-block shared LocalV/LocalO rows

- Experiment: `BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly`
- Direct baseline: `BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B`
- Worktree: `/data0/xd/llf-first-block-shared-row`
- Branch: `codex/llf-first-block-shared-row`

The first `L,L,F` block is explicit. Its two L layers retain the parent's shared LocalV/LocalO
row answer and independent rank-4-B LocalV column. The remaining seven `L,L,F` blocks stay in
block scan; their L layers retain LocalQ/K rows and LocalV/LocalO columns but contain no
LocalV/LocalO row parameters or contractions. Every F layer keeps the parent's fetchO row read.
The M carry, write path, optimizer, schedule, and history-M cache are unchanged.
This removes 15,655,360 parameters (`3.733 W_Q`) across the 14 affected L layers.

## LocalV-only column experiment

`BamMediumIndependentLLFBAlignedRowLocalVColOnlyRank4B` is the exact Medium analogue of
`BamXLSharedBasisLocalVColOnlyRank4CFp32`: every L layer retains LocalV's independent rank-4-B
full-M column, while LocalV no longer consumes the shared LocalO row answer. LocalO, LocalQ/K,
all F reads, M carry and M-cache are unchanged. It removes only the LocalV row destination gates:
524,544 parameters (`0.125 W_Q`) across 16 L layers.

Pre-run bet versus RowShared: late dloss center `-0.0003`, likely `[-0.0015,+0.001]`; throughput
tied within `0..+0.3%`. XL ColOnly is slightly better than RowShared, while Medium QKVColOnly
and RowShared evidence both place the isolated LocalV-row value near zero.

Pre-run bet versus the parent: late dloss center `+0.0015`, likely `[0,+0.004]`, with throughput
`+2..3%`. The bet reflects the full row-removal result (~`+0.007`) and the strong first-block
concentration seen in frozen selective-retention diagnostics, while allowing residual value in
later LocalO/LocalV rows.

## Runtime

- code commit: `46daaf3`
- AOT artifact: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/46daaf3/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly.pickle`
- TPU / FIRST_STEP: `xd-v5p-16-llf-first-block-vo-row-maxtext`, UE5a, step 4
- speed: `.698 step/s` near step 183 (`+2.11%` vs matched-health BAlignedRow `.6836`)

## O-row GELU-LoRA variant

`BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu`
inherits the same row placement, then replaces every retained O-row key projection with
`D→256→nK` GELU: the two L layers in block 0 and all eight F layers. Later L layers remain
column-only and instantiate no row bottleneck. This removes another 3,932,160 parameters,
for 19,587,520 (`4.670 W_Q`) fewer parameters than the original RowShared parent.

Pre-run bet versus FirstBlockOnly: late dloss center `+0.0005`, likely `[-0.001,+0.002]`,
and throughput `-1.5..-2%`. The prior all-layer O-row GELU run ended near `+0.002`, with its
F-only arm loss-neutral and its L-only arm around `+0.006`; retaining only the two earliest L
rows should remove most of that L sensitivity, while the extra nonlinear projections retain
the previously observed throughput cost.

Runtime: `46daaf3`, UE5a `xd-v5p-16-llf-first-block-vo-row-gelu-maxtext`, AOT loaded and
FIRST_STEP 1. It measured `.699 step/s` near step 89 (`+2.25%` vs BAlignedRow, `+.14%` vs
FirstBlockOnly), so the speed bet was wrong: at ten retained row readers the projection FLOP
saving outweighed the GELU overhead.

## All-L LocalV/O column-only variant

`BamMediumIndependentLLFBAlignedRowLocalVOColOnly` keeps the whole eight-block scan. All 16 L
layers drop the shared LocalV/O row projection, gate, decoder and contraction; every F layer
keeps the parent's linear fetchO row. LocalQ/K and independent rank-4-B LocalV columns are
unchanged. It removes 17,891,840 parameters (`4.266 W_Q`) versus the RowShared parent.

Pre-run bet versus RowShared: late dloss center `+0.003`, likely `[+0.001,+0.006]`, and
throughput `+2.5..3.5%`. Relative to FirstBlockOnly, the bet is roughly `+0.0015` from removing
the two earliest L row reads, with a smaller additional speed gain.

Runtime: `01a601e`, UE5a `xd-v5p-16-llf-local-vo-col-only-maxtext`, AOT loaded and
FIRST_STEP 3; `.703 step/s` near step 43 (`+2.84%` vs BAlignedRow, `+.72%` vs
FirstBlockOnly). This matches the throughput bet.
