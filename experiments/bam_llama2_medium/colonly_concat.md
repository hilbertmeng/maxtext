# ColOnly projection substitution

Implementation: `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
Direct loss/speed baseline: `BamMediumIndependentLLFMLPPerLayerColOnly`
(runtime `2ca927c`, UE5a .7354 step/s, generic health ON / BAM OFF).
Both runs retain D1024, H16, head64, 24-layer LLF block scan, M32x32/C8,
all row reads removed, read gates .05 with scales /10, original WD, optimizer, 13,500-step schedule.
MHA parameter target is 411,616,256; widths are independently rounded to the
nearest integer per layer, with no hardware-friendly rounding.

| RUN | Changed standard projections | MLP L/L/F | Parameters | Delta vs MHA | TPU ID |
|---|---|---|---:|---:|---|
| BamMediumIndependentLLFColOnlyVConcatMLPPerLayer | L V: 1024→16×32 | L0=2650; then2703/2703/2596 | 411607312 | -8944 (-.00217%) | colonly-vconcat |
| BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer | all Q/K: 1024→16×32 | 2810/2810/2874 | 411592576 | -23680 (-.00575%) | colonly-qkconcat-r4 |

V arm: L0 keeps full standard V64 and removes LocalV/LocalO, with its own MLP2650.
The first LLF block is peeled; the remaining seven blocks are scanned.
Subsequent L layers concatenate standard V32 first and full gated LocalV column32 second.
QK retain original additive injection and full RoPE. F-layer V stays full64.
Keeping standard V in the first32 leaves a fresh write-data source even when M=0.
LocalO/FetchedO and W_O are unchanged.

QK arm: concatenate gated BAM32 first and standard QK32 second; rotate only
standard32 using the existing partial-RoPE convention. The attention divisor
remains sqrt(64). V/O stay additive. Four raw column bases are shared between Q/K;
head mixing and per-head gates remain independent, effective-key Gram routing
in fp32, matching the semantics of historical
`BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis` (`c664e82`).
That XL run tied separate rank4 at late matched steps while saving parameters;
it is a precedent for shared bases, not a direct loss baseline for concatenation.

QK shared column-key kernels use regular nonzero Dense initialization; key
biases stay zero. All read gates start at .05; common key scale .2 and LocalV
scale .1 preserve the old .005×2 / .005×1 initial amplitude, respectively.
This changes the gate operating point, not initial BAM strength; sigmoid read
amplitude ceilings are ten times lower. With zero Q and K read
keys, concatenated BAM attention scores and both gradients would be identically
zero. Additive injection avoided that trap via standard/BAM cross terms.
No additional trainable parameters are introduced by the initialization.
BAM parameter counts per L/F: V arm 871410/674674; QK arm1066064/869328.
QK rank4 sharing adds194654 parameters/layer over parent rank1, while narrowing
Q and K saves1048576/layer. V narrowing saves524288 per L layer.
M-cache size is unchanged in both arms.

Validation: pinned CPU BAM suite including concatenation projection shapes,
first-layer write with zero M, finite nonzero shared-basis loss gradient,
shared column cache vs uncached reads, BAM/rotary-half placement, and F V width.
Actual full-model parameter-tree audits: `/data0/xd/concat-audit.json`.
Unit log: `/data0/xd/concat-unit.log`.

Pre-run predictions vs parent: V gap+.006 and speed-1%; QK gap-.003 and speed-2%
(lower confidence for QK). These are bets, not measured results.
Training primary UE5a; passive UC1a/EW4b after the standard queue timeout.
AOT compilers EW4a, then UC1a/UE5a. TPU names are
`xd-v5p-16-colonly-vconcat-maxtext` and `xd-v5p-16-colonly-qkconcat-r4-maxtext`.

Targeted health ON in both runs: each read gate mean/std/low/high fractions,
BAM/standard per-coordinate RMS, and QK-concat score RMS/ratio from16 evenly
spaced tokens of one sequence (causal, row-centered). Per-layer export supports
both peeled first-block and ordinary block scan. Existing broad BAM health
flags stay OFF. Trace validation: `/data0/xd/concat-train-trace.log`.
Historical ColOnly speed has BAM health OFF and is not a matched-health timing.
