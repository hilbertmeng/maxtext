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

Pre-run bet versus the parent: late dloss center `+0.0015`, likely `[0,+0.004]`, with throughput
`+2..3%`. The bet reflects the full row-removal result (~`+0.007`) and the strong first-block
concentration seen in frozen selective-retention diagnostics, while allowing residual value in
later LocalO/LocalV rows.

## Runtime

- code commit:
- AOT artifact:
- TPU / FIRST_STEP:
- speed:

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
