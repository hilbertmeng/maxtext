# XL V5 sparse MLP writes: normalization versus vector-path isolation

Worktree `/home/xd/.codex/worktrees/xl-v5-sparse-mlp/maxtext`, branch
`codex/xlprop-v5-sparse-mlp`, based on the actual completed full-remat runtime
`d54c4c5f1339b01e406932ec28be0e398722b918`. Preserve its V5 model and v8t
packed read/write kernels; do not restore the old SOTA read architecture.
This is a root-cause experiment for repeated gradient/loss spikes, not SOTA
elimination. Both fresh RUNs use the confirmed 50,000-step XL budget.

| RUN | MLP writes | Embedding content | Writer-MLP content | Writer MLP enters vector residual |
|---|---|---|---|---|
| `BamXLPropV5EveryThirdMLPWriteNorm` | L1/L4/.../L25 | per-head RMSNorm | per-head RMSNorm | yes |
| `BamXLPropV5EveryThirdMLPWriteRawMemoryOnly` | L1/L4/.../L25 | original V5 fixed divisor .00203054060713255 | raw, divisor 1 | no |

The original SOTA's zero-based middle-of-block write positions were checked
at runtime eeec67c: `(layer+1)%3 == 2`, not L2/L5/etc. Replace every-layer
private memory MLPs with one ordinary SwiGLU MLP at each layer. Nine identical
three-layer blocks scan together, followed by explicit L27. Attention writes
remain enabled at L0--L26; both terminal writers remain disabled. Only sparse
writer layers in the second arm lose their direct MLP vector update. The
attention residual and every other layer's ordinary MLP remain active.

User clarified that the second arm retains V5's embedding divisor. Setting it
to one would reduce embedding-write amplitude by approximately 492.5 times,
introducing an unrelated intervention; that draft was corrected before AOT
or formal training. The first arm restores embedding and MLP content RMSNorm;
attention content normalization, write gates/addresses, whole-M pre-norm,
unit-initialized V/O and Q/K scales, NoPE/RoPE geometry, ranks and all training
hyperparameters remain those of the full-remat parent.

Both arms have block MLP widths 6071/5875/6071 and terminal width 6272.
Full abstract initialization counts 1,432,399,516 parameters in each, versus
1,432,396,996 in the V5 parent and the MHA target 1,432,398,720: deltas +2520
and +796 respectively. The complete parameter trees of the two new arms match.
Removing 18 MLP address/gate projection sets saves 20,289,960 parameters;
refund locally to the non-writing MLPs, then close integer residue at L27.
No per-layer-width mechanism is introduced.

Pallas interface is unchanged. Non-writing layers pass attention factors;
writing layers also pass the ordinary MLP's content, gate logits and private
R400 address. Existing content divisors None versus 1 already implement the
normalization choices. The residual-off switch acts only after collecting
write factors; it changes no kernel math or parameter tree.

## Infrastructure and numerical gates

Audited main 2083fb47 against d54c4c5. Training/AOT/checkpoint/optimizer,
summary writers and max_utils operational paths are unchanged; train.py's
only later commit is V5 cleanup 847de6bb, removing legacy health formatters.
Retain the old compatible block-scan health formatter instead of copying the
single-layer-only refactor. TB rotation 500, begin_step, checkpoint retention
and terminal-save behavior are present. Re-run summary-writer regressions.
Keep generic training health ON and internal activation/BAM sow statistics OFF.

Focused gates: exact full-sized parameter trees and schedule/geometry contract;
four-layer native/Pallas scanned forward and every parameter gradient for both
arms, including mixed attention-only/attention-plus-MLP writes and packed reads;
a direct sparse-layer test proving zero MLP-parameter gradient from the vector
output and nonzero gradient from the matrix output. Exact v5p-32 AOT separately
gates formal shapes, sharding and VMEM. Preserve full remat; do not use saved
Splash residuals. Use the retained v8t implementation, 63 MiB VMEM and tuned
Splash. Local CPU AOT uses the verified current worker environment; no compiler
TPU or healthy trainer is borrowed.

## Prior and monitoring

Prediction before training: both arms reduce severe/persistent early loss
spikes versus the parent. The raw memory-only arm should more strongly interrupt
the ordinary-MLP residual amplification chain. The normalized arm may still spike
because its wider MLPs continue to update the vector stream. These two arms change
more than one factor relative to each other: do not label their loss difference
as a pure normalization effect. If both improve stability, their common sparse
architecture is a candidate explanation, not an isolated causal proof.

Terminal MHA-referenced gain priors versus V5: normalized 1.01 (range .98--1.04),
raw memory-only 1.025 (range .99--1.06). Speed prior approximately .54 step/s,
about +4% versus historical .518, plausible +0% to +10%, unpaired. Stability is
primary: compare same-step raw/clipped norms, actual recorded loss, loss-excess
burden and recovery duration, with the parent and the two new arms together.
Direct loss/gain baseline is `BamXLPropV5FastKernelFullRemat`; use XL MHA only
as the gain denominator. Start with fixed 2k reports, no loss-based elimination;
keep existing XL M60/R480 capacity monitoring active.

## Sealed preparation

Sealed implementation commit `0345fc9ef920fdb68f264397d84746c9b95dca5a`.
Four targeted tests passed, including both scanned native/Pallas all-gradient
controls (193 parameter leaves each) and vector-path isolation. Five summary
writer regressions passed. Formal budget and sealed effective-config checks pass.
The first draft test used a logits-shaped cotangent against the model's token-loss
output; the harness was corrected to that output shape before the successful run.
No kernel math changed.

Owned formal trainer names: normalized RUN `xd-v5p-32-311060-maxtext`; raw
memory-only RUN `xd-v5p-32-311061-maxtext`. UE5a primary, UC1a backup after
300 seconds, EW4b excluded. No retained compiler is owned by these runs.
Artifact/preparation root:
`/data0/xd/bam_diagnostics/xlprop-v5-sparse-mlp-20261117/`.

The initial CPU AOT wrapper rejected a linked worktree because its checkout
preflight requires a .git directory. Compile from the exact clean detached
clone `/tmp/maxtext-xl-v5-sparse-aot-0345` instead; runtime commit, config seal
and verified environment are unchanged. Independent compiles use disjoint
CPU sets 0--15 and 16--31. Start each ready arm without waiting for the other.
