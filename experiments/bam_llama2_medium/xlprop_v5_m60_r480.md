# XL V5 SiTU M96x60 / R480 configuration

Configuration: `BamXLPropV5SiTU47M60R480`.
Parent: `BamXLPropV5FastKernelFullRematSiTU47`.
Implementation/worktree: `/home/xd/projects/maxtext`, `refactor-bam`.
Configuration only: no training RUN, TPU allocation, AOT or speed measurement.

Keep L28 / D1920 / 20 heads of 96, ordinary SiTU 4/7, private SwiGLU,
whole-M pre-norm, unit-initialized learned QK/VO scales, full NoPE96 +
RoPE24, Prop training/initialization and the parent's full-remat execution.
The only architectural changes are M96x40 -> M96x60 and all four ranks
400 -> 480: shared read down, attention-write down, memory-write address
down, and embedding-write down. Read/attention-write/memory-write down
projections remain independent. Inference matrix storage increases 50%.

## Parameter matching

Match the parent total, including embedding/unembedding, retaining a
roughly 2:1 ordinary/complete-memory branch ratio in the first 27 layers.
The complete memory branch includes its address projections and write gates.

| Quantity | Parent | M60/R480 |
|---|---:|---:|
|Ordinary MLP width, L0--26|4048|3859|
|Private memory MLP width, L0--26|1827|1662|
|Terminal ordinary MLP width, L27|6277|5993|
|Ordinary / complete-memory parameters per regular layer|2.001287:1|2.000644:1|
|Total parameters|1,432,396,996|1,432,395,236|

Without MLP repayment the expansion costs 56,688,160 parameters, or 15.37765
W_Q units (W_Q=1920^2). Repayment is distributed across regular layers;
the terminal width handles its own larger read plus the embedding change
and integer rounding. The final difference is -1760 parameters versus
parent (-.000477 W_Q), or -3484 versus the matched MHA reference.
Regular-layer totals differ by only +180 parameters each. The new ordinary
branch has 22,227,840 parameters; its complete memory branch has 11,110,340.
Terminal attention and memory writers remain disabled. Retain one-layer
scan over L0--26 with an explicit L27; no per-layer-width mechanism is added.
The direct comparator is the SiTU parent.

## Pallas verification

Keep packed v8t reads/writes, fused inputs and 63 MiB scoped VMEM. Both
40 and 60 address coordinates pad to 64 in the kernel, while K96 and the
120-dimensional Q/K output remain unchanged. R480 projections execute
outside the read/write kernel; no new kernel interface is required.

Focused checks in the pinned CPU environment passed:

- Full-size parameter-tree counts for the actual parent and new named class,
  all four ranks, SiTU limits, scan/write placement and the branch ratio.
- New V60 / K96 / 20-head read against an independent reference, all gradients.
- Whole-block packed versus unpacked 20-head keys at V60, all gradients.
- Reduced model retaining V60 / K96: packed Pallas read+write versus native,
  forward and every parameter gradient; maximum relative gradient error 7.7e-6.

The last three reuse the existing XL controls with the new address shape;
no unrelated regression suite was repeated. TPU VMEM allocation and speed
remain unmeasured until a target-topology compilation/profile is requested.
Evidence and reproducible scripts:
`/data0/xd/bam_diagnostics/xlprop-v5-m60-r480-config-20261116/`:
`budget_probe.py`, `budget_verify.py`, `budget-verified.json`,
`kernel_check.py`, `kernel-check-r1/`.
