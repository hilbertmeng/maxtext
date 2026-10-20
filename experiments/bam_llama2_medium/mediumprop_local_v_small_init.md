# MediumProp small static V seed on B

RUN: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdVOReadNormalStaticVSmallTruePile`.
Direct parent B: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroVOReadNormalTruePile`.
Direction benchmark: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`.
Worktree `/data0/xd/mediumprop-qk75-sparse`, branch `codex/mediumprop-qk75-sparse`.
Runtime `2fa62ee9b3ad9b20c60c89078e59e957ea387242`.
Hot replacement target: content-bias trainer `xd-v5p-16-2910122-maxtext`, UE5a.

Only treatment versus B: static V Gaussian std .1/sqrt32, rather than zero.
Shared dynamic VO key regular initialized, V/O scales1, V/O gates .1/.05;
QK scale .2 and static Q/K/O zero remain B's settings. No content bias.
MLP[3901,3774,3901], total432096128 unchanged. Fresh step0 RUN.

Bet final vs B -.005; speed unchanged within1%. Plan13500, loss windows200,
reports1000, reviews2800/5000, inherited generic+concat+write health.
Focused full-budget, same-seed leaf parity, exact one-tenth Gaussian and finite
scanned forward/gradient check passed; sealed effective config verified.
B and this RUN have separate exact AOT artifacts. Compiler owned spot v6e;
occupied retained flex compiler untouched. Content-bias run remains alive
until this new RUN's preparation is complete.

Artifacts `/data0/xd/bam_diagnostics/mediumprop-static-v-small-launch`.
Main exp.py is ledger-only for this implementation.

At10:58UTC the retained EW4a FLEX_START llm-jax-v6e-1-0 was rechecked idle/HEALTHY.
Both spot preparers canceled with successful cleanup of all six candidates.
AOTs moved to isolated artifacts on the retained worker, serialized by worker lock;
this path owns no TPU lifecycle and does not modify its original checkout.

Startup verified: exact runtime2fa62ee, compiled function loaded, FIRST_STEP6, step0 dataset origin and zone-local UE5a path. UE5a v5p-16 .5192step/s (20-99), +.17% vs B / -.16% vs standard; matched health. Initial V/O gates .09947/.05033; captured BAM metrics finite.
