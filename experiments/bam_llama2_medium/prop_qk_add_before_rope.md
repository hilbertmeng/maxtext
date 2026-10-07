# MediumProp full M QK add-before-RoPE

Runtime worktree: `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
RUN: `BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile`.
Direct baseline: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` (`23c692b`).

Use full M75 static plus gated dynamic LocalQ/K read. Add the standard vector QK18 projection to coordinates 57:75 before rotating those 18 coordinates once. Keep 0:57 NoPE; total attention width75 and sqrt75 logit divisor. QKNorm remains disabled. All other parent settings, M75×32/C8, sparse private MLP writes, scan, initialization, health and MLP widths [3901,3774,3901] remain unchanged. Parameter tree and total432096128 match the parent, so no MLP adjustment.

`matrix_qk_scores` measures NoPE-prefix score RMS versus combined RoPE-tail score RMS (including matrix/vector cross terms). Existing read-amplitude metrics still measure the original unrotated arms. This is not the previous QK93 concatenation.

Bet: terminal13500 loss gap -.003 versus QK57; steady speed unchanged from parent .520 step/s. Normal1000-step reports/200-step windows, review2800/5000. UE5a primary, spot trainer. Tail-block independent VO stays running through its2800 review; no hot switch.
