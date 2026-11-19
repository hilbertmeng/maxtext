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

## Verified historical stability premise

The old XL single-.5 SOTA `BamXLPropSharedR400GeneralReadMShiftReadG050TruePile`
has all 1,941 ten-step raw-gradient samples from 600 through 20,000. Maximum
raw norm is 2.062887 at step 620; counts above 10 and 100 are both zero. Its
runtime eeec67c emitted raw and clipped norms, so this is not missing health
telemetry. Evidence: `old-sota-gradient-check.json` under the task artifact
root. The full-remat V5 comparator has genuine large gradient/loss episodes
at approximately 4.14k, 7.91k and 10.1k; retain its original history only
through 8,750 and exclude the discarded FSS-damaged continuation.

The raw memory-only arm also preserves the aggregate vector-MLP budget almost
exactly: 665,568,000 versus the V5 parent's 665,700,480 parameters (-0.0199%).
The complete matrix-MLP branch is 314,704,980 versus 314,569,980 (+135,000).
Their net +2520 is the already recorded rounding difference. Thus this arm
mainly moves ordinary/memory specialization from parallel branches within each
layer to separate layer positions; it does not simply remove one third of the
ordinary residual-MLP parameter budget. The normalized arm instead lets the
large writer MLP enter both streams. Keep that distinction in causal analysis.

## Accepted startup

Both RUNs reached FIRST_STEP and continued beyond their startup gates in UC1a,
using the original regional race with UE5a primary and UC1a backup. Each pod's
four workers independently report the exact 0345fc9 source and loaded AOT;
all four have global batch 128 and the UC1a TruePile4096 replica. Normalized
startup capture spans steps 0--81, raw memory-only capture 2--102 (bounded
last-100-step excerpts). Median recorded speed after step 20 is .539 step/s
for each, about +4.1% versus historical V5 .518, unpaired. Runtime hashes in
main exp.py refer to this accepted implementation, not this metadata commit.

Proofs: `startup-proofs.json`, `startup-accepted.json`, per-RUN first-step
gates on tpu-ag, and `stream-budget-comparison.json` under the task root.
Monitoring group `xlprop-v5-sparse-mlp` uses a fixed 2k calendar, no elimination
reviews, both arms against the full-remat V5 parent. The independently monitored
capacity RUN subsequently completed its 50k plan.
`report_group.py` renders native cumulative gap/gain lines and aligned actual
loss plus raw/clipped gradients, with the parent's valid original prefix.

## Normalized arm: user-requested closeout, 2026-11-18

`BamXLPropV5EveryThirdMLPWriteNorm` stopped at committed checkpoint 24,421
on the user's explicit instruction. Relative to `BamXLPropV5FastKernelFullRemat`,
the last five mature 22k--24k windows average loss gap +.0032614 and
MHA-referenced gain .97695. Early parent gradient/loss episodes did not recur,
but normal-phase fitting remained worse; the terminal prior 1.01 is not tested
by this early stop. The parent's persistent 35.65k--36.93k disturbance was not
reached and remains unanswered for this arm.

Runtime remains `0345fc9`, v5p-32, approximately .534--.539 step/s, about +4%
versus the historical .518 parent timing (unpaired). The complete active-zone
path was UE5a -> UC1a -> UE5a -> UC1a -> UE5a -> UC1a -> UE5a; native records
contain nine preemptions and ten formal READY leases. Every assignment and
READY interval is recorded in `experiments/tpu_region_preemption_history.md`.
The user-stop boundary is 00:21:56 UTC; resource absence was verified at
00:25:01 UTC. Checkpoint 24,421, its seed9876/global128/four-host cursor,
local TensorBoard `SYNC_OK`, and both zones' TPU/queue deletion are verified.
Evidence is under `norm-final/` in the task artifact root; final cumulative
curves and native report data are in `norm_closeout/`.

The raw memory-only arm continues to 50k, including its planned check of the
parent's long disturbance window. This stop does not establish a pure content-
normalization effect: the two sparse arms also differ in their vector-residual
updates, as described above.

## Raw memory-only arm: completed 50,000 (2026-11-19 closeout)

`BamXLPropV5EveryThirdMLPWriteRawMemoryOnly` completed the original50k schedule
at runtime0345fc9, v5p-32/311061. Native P9, ten chronological READY leases;
active assignment path UE5a -> UC1a -> UE5a -> UC1a -> UE5a -> UC1a -> UE5a
-> UC1a -> UE5a. The initial UE5a stint was preparation before verified training.
Steady speed approximately .536 step/s, about+4% versus historical full-remat
V5 .518 (unpaired, matching generic-health settings). Final checkpoint50000,
seed9876/global128/four-infeed-host cursor, local TB SYNC_OK and all regional
node/queued-resource absence checks passed.

Relative to `BamXLPropV5FastKernelFullRemat`, small normal-phase advantages
were punctuated by large temporary gains during parent loss disturbances.
The parent's prolonged35.65k-36.93k disturbance did not recur in this arm.
After37k the loss lead narrowed; final five common windows47500--49500 give
mean gap-.00199104 and MHA-reference gain1.01623523. The1.025 prior was mildly
optimistic, within its .99--1.06 range. No50000-centered window is claimed:
there are no post-end samples to complete it.

Recorded ten-step raw-gradient samples from600 onward peak at1.27485, with
none above10; initial training transients are excluded from that statement.
The matched170-sample proof over35400--37100 contains40 parent raw-gradient
values above10 versus zero here, parent max345624.9 versus .22379. Same-grid
loss max3.27189 versus2.09173. Sparse raw writes can therefore remain stable
without restoring write-content normalization, while maintaining a small
terminal fitting advantage. Sparsity and omitting the writer-MLP vector update
are still coupled: their individual causal contributions are not identified.

Native final loss/gain/health data:
`/data0/xd/bam_diagnostics/xlprop-v5-sparse-mlp-20261117/final50000/`.
Critical-window proof: `risk37100/matched-grid-summary.json` in that root.
Independent checkpoint/resource proofs and all leases:
`/data0/xd/bam_diagnostics/research-closeout-20261119/`.
The shared regional history records all nine assignments and ten READY leases;
no retained compiler was deleted as part of training closeout.
