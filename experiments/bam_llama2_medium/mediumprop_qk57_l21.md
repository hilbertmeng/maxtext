# MediumProp original QK57,21 layers

Runtime worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
RUN `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdL21TruePile`.
Direct parent `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile` (`23c692b`).

Retain M75×32/C8, full-M shared-rank4 LocalQK, static QK reads, QK57+independent vector RoPE18 (QKNorm off), no W_V, normal W_O and shared dynamic VO keys with independent gates. Increase18 to21 layers,seven three-layer blocks. Private R256 GELU MLP writes at zero-based layers1/4/7/10/13/16/19. Uniform MLP width3177, total432088816 vs parent's432096128 (-7312); MHA432121200 (-32384). No QK93 extension or add-before-RoPE change.

The original MLP widths [3901,3774,3901] refund42681600=29.64W_Q when restored to3200. Three additional layers at3200 overshoot the parent by1731488=1.20242W_Q. Uniform3177 (0.71875% below3200) permits21 nearly matched layers.

Bet13500: loss -.005 vs parent; speed .48 step/s vs .520 (-7.69%). Same basic and concat health. Normal1000-step reports,200-step gap windows; review2800/5000. UE5a spot trainer; AOT borrowed from retained FLEX_START llm-jax-v6e-1-0 without lifecycle ownership. CPU checks focus on full exact budget, full train-step graph and seven writer-health positions; shared architecture already passed finite consumed-gradient checks.

Runtime e10dfc6; UE5a xd-v5p-16-2910114-maxtext. AOT loaded/FIRST_STEP1, local TruePile path confirmed. Steady20–99 median .466 step/s (80 samples), vs parent .520 -10.38%; below speed wager .480 by2.92%. Parameter/runtime ledger audit passed.

Completed13500 (2026-10-08UTC). Final12600..13400 gaps -.008951,-.009025,-.009886,-.010253,-.009406;
mean-.009504,range[-.010253,-.008951]. Initial lead shrinks from-.062 at200 to~-.009 by1000,
then remains there through completion, without late erosion. Bet-.005 understated depth gain by~.0045;
observed .466 vsspeedbet.480. Closeout wrapper already_closed,finalcheckpoint13500,zero loststeps,TB SYNC_OK;
node and queuedresource absence verified.0preemptions,UE5a only,READY lease2026-10-07 15:55:25 ->2026-10-08 00:23:47 (8h28m22s).
