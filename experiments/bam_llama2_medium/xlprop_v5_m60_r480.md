# XL V5 SiTU M96x60 / R480 configuration

Configuration: `BamXLPropV5SiTU47M60R480`.
Parent: `BamXLPropV5FastKernelFullRematSiTU47`.
Implementation/worktree: `/home/xd/projects/maxtext`, `refactor-bam`.
Formal RUN uses the configuration name; TPU `xd-v5p-32-311046-maxtext`.
Runtime `b5f52719f058c051888b296c2d26b79788e6000f`, initially UC1a.

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

Keep packed v8t reads/writes, fused inputs and 63 MiB scoped VMEM.
K96 and the 120-dimensional Q/K output remain unchanged. V60 is supported
by the existing kernel, but address-coordinate loops, scratch buffers and
I/O still use actual V; the earlier claim that both V40 and V60 uniformly
pad to64 was incorrect. R480 projections execute outside the read/write
kernel; no new kernel interface is required.

Focused checks in the pinned CPU environment passed:

- Full-size parameter-tree counts for the actual parent and new named class,
  all four ranks, SiTU limits, scan/write placement and the branch ratio.
- New V60 / K96 / 20-head read against an independent reference, all gradients.
- Whole-block packed versus unpacked 20-head keys at V60, all gradients.
- Reduced model retaining V60 / K96: packed Pallas read+write versus native,
  forward and every parameter gradient; maximum relative gradient error 7.7e-6.

The last three reuse the existing XL controls with the new address shape;
no unrelated regression suite was repeated. The exact v5p-32 AOT subsequently compiled and loaded on the formal workers.
No separate profile TPU was allocated.
Evidence and reproducible scripts:
`/data0/xd/bam_diagnostics/xlprop-v5-m60-r480-config-20261116/`:
`budget_probe.py`, `budget_verify.py`, `budget-verified.json`,
`kernel_check.py`, `kernel-check-r1/`.


## Formal launch

Train the confirmed XL budget: batch128 (8/device), T4096 TruePile4097,
seed9876, 50,000 steps =26,214,400,000 tokens; LR endpoint50,000.
Policy is UE5a primary, UC1a backup after300 seconds, EW4b excluded.
Keep the parent's full remat/tuned Splash and packed v8t read/write.
Generic training health ON, internal activations OFF, TBrotation500;
CP250/latest2/forced final, with no extra periodic permanent retention.

Recipe/optimizer/initialization/geometry audit against the actual completed
parent runtime `bd181bc` found no unintended differences in the selected
recipe fields; source and configuration now use the supported main V5
implementation. A clean detached Git clone at `/tmp/maxtext-local-aot-b5f52719`
was used only for compilation; the experiment implementation stays in main,
and unrelated working-tree edits were not included. The sealed-config gate
passed. The runtime already includes rotating writer construction and
begin_step before scalar writes; five summary-writer regressions passed.
The earlier full-size shape/count and three targeted numerical checks were
reused because no numerical source changed after those gates.

The exact v5p-32/s50000 AOT compiled on the ordinary CPU host in the
worker-matched pinned environment and was verified with its manifest:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/b5f5271/jax081-i0ae3f58-c5aae279/v5p-32/s50000/BamXLPropV5SiTU47M60R480.pickle`.
All four formal workers loaded it at the exact runtime hash, using the
registered UC1a TruePile4096 replica and global batch128. Native FIRST_STEP,
step14 and step100 gates passed. Median steps55--99 across all four workers
was .467 step/s, -10.2% versus the parent's recorded .520 (unpaired).
This is slower than the -5% pre-run speed bet; architecture effects remain
the purpose of training, not a performance elimination criterion.

Pre-run final MHA-referenced gain bet versus the SiTU parent:1.03,
plausible range1.01--1.06. Use only the direct parent as the plotted loss
comparator; MHA `Llama2XLPropTruePileMHA` supplies the gain denominator.
Group `xlprop-v5-m60-r480`, cadence2000, no review steps. Preserve500-step
native windows and plot cumulative loss gaps/gain multipliers; regular
health uses raw/clipped gradient norms in the same RUN color. Continue
through the full plan; no performance-based elimination.

Local preparation/startup/monitoring artifacts remain under the directory
above. Remote task:
`/home/lishengping/xd/projects/logs/xlprop-v5-m60-r480-20261116/`.
`launch-plan.json`, `recipe-audit.json`, the CPU AOT state/manifest pointer,
`startup-proofs.json` and `startup-accepted.json` record the immutable plan
and actual launch acceptance. The registry is authoritative for live state.


## Execution-cost clarification

The sealed runtime's `_tm_slabs`, `_tm_fill_keys` and reverse coordinate
passes iterate over `range(cdim)`, where cdim is40 or60. Forward scratch
includes `(cdim * tile, rows)`; reverse has several further cdim-dependent
buffers. Thus kernel VPU work and scratch also grow, independently of any
MXU instruction alignment. `_to_tokens` explicitly pads its row slab to128
for an XLU transpose; that is not a uniform64-column M representation.

Read/address-up matrix sizes change400x800 to480x1200 (+80%); their
logical output volume increases50%. Keeping total parameters constant by
shrinking MLPs preserves neither operation shapes nor memory traffic.
These source facts explain why equal-alignment intuition underestimated
cost. The observed -10.2% startup throughput falls below the -5% prior;
source inspection alone does not quantify each phase's measured contribution.
No kernel, model setting, AOT or running training process was changed by
this clarification.
