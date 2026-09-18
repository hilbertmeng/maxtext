# XL DirectC8 all-column parameter match and K128

Implementation: `codex/xl-directc8-all-col-k128`,
`/data0/xd/xl-directc8-all-col-k128`.

`BamXLSharedBasisQKDirectC8MLPPerLayerColOnly` keeps the historical DirectC8
LocalQ/K column mechanism, removes every LocalQ/K/V/O and fetched-O row read,
and keeps all column reads. Its unrestricted integer L/L/F SwiGLU widths are
5178/5178/5243. The shaped tree has 1,420,900,224 parameters, 20,608 fewer than
`BamMHALlama2XLHead16x128C256PartialRoPE`; this is the closest point on the
eight-repeated-block integer-channel lattice.

`BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE`
changes only raw M from 64x32 to 128x32. C8 remains fixed. LocalQ/K read the
compressed 128x8 state, truncate the 128-dimensional column answer to the first
96 coordinates, and leave the final 32 Q/K coordinates for RoPE. K64 and K128
have identical full parameter trees, including identical per-role counts.

Validation artifacts:

- `/data0/xd/xl-directc8-all-col-k128-budget.json`: MHA/K64/K128 full shaped
  parameter counts.
- `/data0/xd/xl-directc8-all-col-k128-trainshape.log`: both full 24-layer
  scan train-step shapes.
- `/data0/xd/xl-directc8-all-col-k128-tests.log`: BAM regression tests; the
  pre-existing fetched-read test receiver was updated for the compact arm spec.

The direct C8 numerical test checks the contraction independently for both K64
and K128 and verifies active gradients. K128 doubles the k extent of raw and
compressed M, so it increases state/cache and BAM FLOPs without adding parameters.
