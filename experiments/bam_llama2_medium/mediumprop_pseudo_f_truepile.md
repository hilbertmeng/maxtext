# MediumProp pseudo F: reopen the MLP-to-M content path without fetch

Worktree `/data0/xd/mediumprop-k75-embed`, branch `codex/mediumprop-k75-embed`. RUN `BamMediumPropK75EmbedVOnlyQK57PseudoFTruePile`; UE5a trainer `xd-v5p-16-2910018-maxtext`. Retained EW4a FLEX_START compiler `llm-jax-v6e-1-0` is borrowed only, excluded from cleanup.

AllLocal control uses matrix-derived V in all18 layers. Every third pseudo F restores W_V x, removes its matrix V reads/gate, and retains both static full-M32->16 and dynamically gated C8 local O. No fetched M or fetch_head_mix is computed. QK57+RoPE18, M75x32/C8, embedding-seeded M, L-layer recipes, write rules and TruePile4096 are unchanged. Six3-layer blocks permit unequal MLP widths.

Per pseudo F, W_V adds1440000, static V removes512 and V gate removes19216: net1420272=.9863 W_Q. L3901/pseudoF3507 are nearest widths to the common MHA budget; total432102560, +11232 versus AllLocal432091328 and -4224 versus originalLLF432106784. Focused CPU checks verify all four exact parameter budgets, no fetch projection, only pseudo F W_V, static local O in all layers and scanned forward/gradient/actual train-health export.

Four architecture cells: AllLocal(matrix V/local O), MatrixV LLF(matrix V/fetched O), pseudo F(vector V/local O), originalLLF(vector V/fetched O). This tests the MLP->x->V->attention->M content route from both opening and closing directions, including its MLP parameter opportunity cost.

Bet terminal pseudoF-AllLocal -.020, pseudoF-originalLLF +.008; speed .525step/s (~-.4% vs AllLocal.5271, flat versus originalLLF.525). Direct baselines also include the new MatrixVLLF arm. Plan13500, checkpoint200/persistent1000/latest2, pureJAX. Review2800/5000; preserve mechanistic ablation value even if it loses to originalLLF.
