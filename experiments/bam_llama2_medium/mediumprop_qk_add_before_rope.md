# MediumProp full M QK add-before-RoPE

Runtime worktree: `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
RUN: `BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile`.
Direct baseline: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` (`23c692b`).

Use full M75 static plus gated dynamic LocalQ/K read. Add the standard vector QK18 projection to coordinates 57:75 before rotating those 18 coordinates once. Keep 0:57 NoPE; total attention width75 and sqrt75 logit divisor. QKNorm remains disabled. All other parent settings, M75×32/C8, sparse private MLP writes, scan, initialization, health and MLP widths [3901,3774,3901] remain unchanged. Parameter tree and total432096128 match the parent, so no MLP adjustment.

`matrix_qk_scores` measures NoPE-prefix score RMS versus combined RoPE-tail score RMS (including matrix/vector cross terms). Existing read-amplitude metrics still measure the original unrotated arms. This is not the previous QK93 concatenation.

Bet: terminal13500 loss gap -.003 versus QK57; steady speed unchanged from parent .520 step/s. Normal1000-step reports/200-step windows, review2800/5000. UE5a primary, spot trainer. Tail-block independent VO stays running through its2800 review; no hot switch.

Runtime237f3e5; UE5a xd-v5p-16-2910113-maxtext. AOT loaded/FIRST_STEP9 verified, zone-local TruePile. 20–99 median .516 step/s vs parent .520 (-0.77%),80 samples; health settings match.

Stopped at committed checkpoint11604 on2026-10-07T22:57:07Z. Last five mature
windows10600-11400:gap+.0065756/+.0064320/+.0065898/+.0062914/+.0059476;
mean+.00636728,range[+.0059476,+.0065898]. The200-step deficit+.110935
shrinks quickly initially, but settles near+.006 from2200 onward. Review points
2800 and5000 were+.006054 and+.006529. There is no sustained approach tozero.
5200 has no complete common window and is omitted rather than interpolated.

The terminal bet-.003 was wrong. Keeping the independent18-dimensional RoPE
channel is better in this configuration than mixing matrix and vector QK in
that channel before RoPE. This result does not isolate the RoPE transform
from the extra matrix signal and its cross terms. Late10400-11400 raw-gradient
median.214 versus parent.221, no nonfinite metrics or sustained gradient burst;
the deficit is not explained by training instability.

Official local closeout completed: checkpoint11604 committed, loss cached through
11602, TPU/queue verified absent, localTB syncSYNC_OK. UE5a v5p-16,4preemptions,
5READY leases; full chronology is in experiments/tpu_region_preemption_history.md.
