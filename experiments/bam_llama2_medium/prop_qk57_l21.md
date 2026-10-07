# MediumProp original QK57,21 layers

Runtime worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
RUN `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdL21TruePile`.
Direct parent `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` (`23c692b`).

Retain M75×32/C8, full-M shared-rank4 LocalQK, static QK reads, QK57+independent vector RoPE18 (QKNorm off), no W_V, normal W_O and shared dynamic VO keys with independent gates. Increase18 to21 layers,seven three-layer blocks. Private R256 GELU MLP writes at zero-based layers1/4/7/10/13/16/19. Uniform MLP width3177, total432088816 vs parent's432096128 (-7312); MHA432121200 (-32384). No QK93 extension or add-before-RoPE change.

The original MLP widths [3901,3774,3901] refund42681600=29.64W_Q when restored to3200. Three additional layers at3200 overshoot the parent by1731488=1.20242W_Q. Uniform3177 (0.71875% below3200) permits21 nearly matched layers.

Bet13500: loss -.005 vs parent; speed .48 step/s vs .520 (-7.69%). Same basic and concat health. Normal1000-step reports,200-step gap windows; review2800/5000. UE5a spot trainer; AOT borrowed from retained FLEX_START llm-jax-v6e-1-0 without lifecycle ownership. CPU checks focus on full exact budget, full train-step graph and seven writer-health positions; shared architecture already passed finite consumed-gradient checks.
