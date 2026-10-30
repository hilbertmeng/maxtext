# Pythia 2.8B H: context/localO complementary gate

Configuration:
`BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedContextLocalOComplement`.

Direct parent and comparison:
`BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixed`.

Transfer the Medium H masked-context competition from runtime `d123e5b`.
Reuse the existing per-token/head dynamic localO gate `g`: Splash context is
multiplied by `1-g` before combining with localO, shared W_O, and attention M writing.
Dynamic localO retains its original gate; static O and its independent gate remain
unchanged. Continue masking attention's self diagonal. No dynamic-address gamma.

The worktree starts from main `032b75db`, including Claude's optimized v8t H
read: common V/O scale is folded into static read weights and post-RMS dynamic
weights. This transfer does not edit either Pallas kernel. The context multiplier
uses existing O logits outside the read kernel, so both gradient paths add without
another gate projection. Native reads optionally return those same logits.

The 2.8B parent recipe remains unchanged: 32 layers, M80x64, R640,
full-head QKNorm, serial GELU/LayerNorm, ordinary/memory/tail widths7808/2944/9984,
minimal remat, v8t read+write with60MiB VMEM, single-layer scan31+tail1,
and no final attention/memory-MLP writes. All original batch/schedule/data settings
are inherited. No extra parameters:2775190912 total, identical parameter tree to H.

Generic health remains ON, internal/BAM bulk statistics OFF. Export just gate
mean/std per layer (64 scalars); existing generic health keeps signed32 common
read scales. The shared health decoder handles31 scanned layers and the separate
tail. No per-address-column metrics are added.

Implementation/validation worktree:
`/data0/xd/pythia-h-context-complement`, branch `codex/pythia-h-context-complement`.
CPU artifacts: `/data0/xd/bam_diagnostics/pythia-h-context-complement/`.
RUN/TPU: none; this delivery creates a configuration, not a training launch or AOT.

Validation: six targeted CPU checks passed in the pinned environment, including
formal target parameter-tree equality, an independent H intervention reference
with all gradients, native/Pallas FP32 and BF16 all-gradient checks, incompatible
config rejection, and real scan+tail health export (formal synthetic export64 tags).
FP32 worst leaf max-relative gradient error was3.124e-6. Against FP32 truth,
BF16 global relative gradient L2 was2.168% native versus1.727% fused.
The optimized common-scale folding path is exercised with perturbed learned scales.
