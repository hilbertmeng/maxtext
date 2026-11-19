# Medium V5 SwiGLU: rebalance actual attention and MLP budgets

Prepared on main `refactor-bam`, `/home/xd/projects/maxtext`. These are new
configuration controls; historical trained configurations remain unchanged.
The user subsequently authorized both formal training RUNs on 2026-11-18;
accepted launch ownership is recorded below.

The fixed target remains 432,121,200 total parameters, including both untied
50,432-token tables. Retain 18/24 layers, 16 heads, 32 M columns, all four
address ranks256, SwiGLU in both MLPs, V5 M pre-norm and unit-initialized
QK/VO scales. Change D and the MLP widths to move the actual attention:MLP
split toward 1:2. M rows follow the head dimension, as in the preceding
fixed-total depth study; this is not a fixed-M-element-count control.

## Accounting

Attention includes all self-attention parameters: standard Q/K/O, local M
reads, M-shift, attention M-write content/address/gates, learned read scales,
and the attention input norm. Complete MLP includes the ordinary and memory
MLPs, memory-write address/gate projections and its input norm. The two token
tables, embedding-to-M module and final norm are separate fixed overhead,
counted in the total budget but not assigned to either layer branch.

At fixed D, changing only MLP widths cannot satisfy both total budget and the
attention:MLP split: the total available MLP budget is already determined.
Search D in multiples of16 with the existing even RoPE rule
`2 * floor(head_dim / 8)`. Select the closest split, preserve approximately
2:1 ordinary:complete memory parameters in each writer layer, and use the
existing explicit terminal MLP width for integer-budget rounding. No new
per-layer-width mechanism is introduced.

| Depth | Original attention:MLP | New D / head / M | Ordinary / memory / terminal widths | New attention:MLP | Actual total / target residual |
|---|---|---|---|---|---|
|18|1:3.780931|1520 /95 /95x32|1501 /631 /2269|1:2.031172|432119828 /-1372|
|24|1:3.462208|1312 /82 /82x32|1404 /578 /2126|1:2.020615|432120128 /-1072|

The 16-head/even-RoPE geometry prevents an exact1:2 split at the exact budget;
the reported ratios are full-model sums, including the write-free terminal.
The nonterminal ordinary:complete memory ratios are1.999930 and1.999711.

| Depth | Attention parameters | Complete MLP parameters | Separate overhead |
|---|---:|---:|---:|
|18|91037268|184912320|156170240|
|24|98514848|199060592|134544688|

## Configuration and validation

- `BamMediumPropV5L18TotalBudgetAttentionMLP1to2SwiGLUTruePile`
- `BamMediumPropV5L24TotalBudgetAttentionMLP1to2SwiGLUTruePile`

The 18-layer architectural parent is `BamMediumPropV5`; it is a configuration
alias, not a registered RUN. Its direct trained comparator is the completed
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormQKScaleTruePile`.
The 24-layer direct comparator is the completed
`BamMediumPropV5L24TotalBudgetSwiGLUTruePile`. Retain the existing same-depth
MHA budget controls as the MHA references.

Single-layer scan retains each uniform L-1 prefix; attention and memory M
writes stay disabled in the explicit terminal layer. RoPE22/full-NoPE95 and
RoPE20/full-NoPE82 give Q/K widths117 and102; both remain in the existing
128-wide Pallas read tiles. Packed keys, fused inputs and v8t read/write
kernels remain enabled, VMEM48MiB. No kernel implementation change is needed.
Embedding content divisor follows the prior cross-width sqrt(D/1200) rule.

Full-size abstract parameter trees and integer-search artifacts are under
`/data0/xd/bam_diagnostics/medium-depth-bam-attention-mlp-20261118/`.
The regression entry is `MaxText/tests/mediumprop_total_budget_depth_test.py`:
it verifies actual branch sums, the full budget, all four ranks, scan/terminal
writes, SwiGLU, and native/Pallas full-model gradients for the new head widths.
Both formal v5p-16/s13500 AOT artifacts are available under runtime3c7579d's
GCS compiled-trainstep prefix; their manifests and actual loads were verified.

Validation: both selected full-size abstract trees reproduce the table above.
The budget regression and native/Pallas gradient regression passed (two tests,
225.8 seconds, pinned CPU environment). The latter preserves each new head/RoPE
geometry while reducing heads, MLP widths, ranks, vocabulary and sequence length;
all104 model gradient leaves agree, worst relative errors4.87e-5 for95/22 and
6.08e-6 for82/20. Signed QK/VO-scale perturbations exercise their folded paths.

## Accepted training launch

Runtime `3c7579dcb894a62546a3eb2163a063688af7b94e`, main `refactor-bam`;
source worktree `/home/xd/projects/maxtext`, controlling task
`codex-xl-v5-situ47`. The RUN names are the configuration names above.

| Depth | Owned trainer | Active zone | Startup median, steps20--99 | Historical direct parent | Speed delta |
|---|---|---|---:|---:|---:|
|18|`xd-v5p-16-311180-maxtext`|UC1a|.746 step/s|.677145|+10.2%|
|24|`xd-v5p-16-311181-maxtext`|UC1a|.606 step/s|.546|+11.0%|

These timings are unpaired, with generic training health enabled and internal
activation/BAM sow statistics disabled. Startup is faster than the .72/.57
prior. Both workers in each pod independently confirm runtime source and loaded
AOT. Actual FIRST_STEP is0; committed checkpoint200 and both data cursors verify
seed9876, global batch128, two infeed hosts and the UC1a TruePile4096 replica.
The losing UE5a candidates were released after native first-step acceptance.

UE5a remains primary, UC1a backup after300 seconds, EW4b excluded. Borrowed
FLEX `llm-jax-v6e-1-0` in EW4a is not owned by training cleanup; its native worker
lock serialized the two AOT jobs. Each trainer was submitted independently once
its AOT/manifest was ready. No separate experimental worktree was created.

Group `mediumprop-depth-bam-attention-mlp` has a fixed2k public calendar and no
performance-elimination reviews; both RUNs retain the original13500-step plan.
Native comparison triples use each depth's MHA reference and preserve the
same-depth original BAM and stronger post-norm/fixed-router Mudd controls.
Reports show both parent effects and absolute18-to24 depth gains. The allocation
interaction is `(new/old gain at24L)/(new/old gain at18L)`; interpret it alongside
the two parent effects, since harming the shallow model also steepens a curve.
Terminal parent-gain priors are .98 at18L and1.00 at24L, interaction about1.02.

Task artifacts: `launch-plan.json`, `startup-proofs.json`, `startup-accepted.json`
and `checkpoint200-proofs.json` in the local artifact root. The native registry,
training controllers and shared watch are on tpu-ag; the XL sparse-write RUN
continues independently under the same controlling task.

## Completed full-budget results (2026-11-19 closeout)

Both RUNs completed 13,500 updates at runtime3c7579d, without preemption.
They trained in UC1a after the preparation assignment moved from UE5a. Steady
speeds were approximately .739/.600 step/s for L18/L24; the startup deltas
above remain unpaired historical comparisons. Final committed checkpoint13500
and its seed9876/global128/two-infeed-host cursor passed verification. Local
TensorBoards returned SYNC_OK. Independent node and queued-resource checks
in UE5a, UC1a and EW4b were absent for both identities.

Final means use the same five native windows12600,12800,13000,13200,13400:

| Depth | Gap vs original same-depth BAM | MHA-reference parent multiplier | Gap vs post-norm/fixed-router Mudd | BAM/Mudd multiplier |
|---|---:|---:|---:|---:|
|18|+.00352616|.98031007|-.08385096|1.91484423|
|24|-.00093504|1.00514435|-.07821768|1.74115538|

The L18 early lead disappeared around5k and became a growing small deficit.
The L24 early lead also shrank, but retained a small terminal advantage.
Parent-gain priors .98/1.00 were close; neither model demonstrated a large
absolute improvement from moving parameters toward attention.

Matched-depth MHA advantages are .17551412/.18376228. Their native paired
G24/G18 ratio averages1.04702122. Reallocation interaction averages1.02533273
versus the pre-run1.02 prior. Absolute BAM loss improves .00982152 from18 to24,
whereas MHA improves .00157336, leaving .00824816 additional BAM depth gain.
The old allocation's corresponding additional gain is .00378696. Thus the
allocation steepens BAM's depth response, but the decomposition matters:
+.00352616 harm at18 and -.00093504 improvement at24 together explain the
.00446120 change. Do not interpret this as a large absolute24-layer win or an
isolated improvement in memory iteration.

Native data/figure: `/data0/xd/bam_diagnostics/medium-depth-bam-attention-mlp-20261118/final-check-20261119/`.
Independent closeout proofs and every chronological READY lease are in
`/data0/xd/bam_diagnostics/research-closeout-20261119/{proofs.json,resource-history.md}`;
the same assignments and leases are in `experiments/tpu_region_preemption_history.md`.
Both registry groups are archived after final comparison delivery. Source
metadata is a later commit; the runtime remains3c7579d.
