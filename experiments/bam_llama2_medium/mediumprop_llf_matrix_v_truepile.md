# MediumProp LLF with matrix-only F-layer V

Worktree `/data0/xd/mediumprop-k75-embed`, branch `codex/mediumprop-k75-embed`.
RUN `BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile`; owned trainer `xd-v5p-16-2910017-maxtext`, UE5a. Retained FLEX_START compiler `llm-jax-v6e-1-0`, EW4a: borrowed only, never released.

Direct controls: `BamMediumPropK75EmbedVOnlyQK57TruePile` and `BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile`. TruePile4096, D1200/H16x75, M75x32/C8, QK57+RoPE18, embedding-seeded M, six LLF blocks.

Only F-layer V changes: remove the full W_V; use a full-M static32->16 read plus dynamically gated C8 local-M read. V shares the existing fetched-O W_R projection and compression, with its own gate; fetched O remains a dynamic read of the unchanged fetched state, without a static projection. F has no local O. Compression is evaluated once for local V and fetch. All L layers are unchanged.

Each F removes 1,440,000 parameters and adds 19,728 (static512 + gate19,200 + gate bias16), net saving1,420,272 (.9863 W_Q). F MLP3502->3897; L3901. New total432,117,152 versus original LLF432,106,784; difference10,368 (.0024%). New gates/static V use the L recipe's initialization and normalization. PureJAX only. Checkpoints every200; persistent1000; latest2; plan13500.

Bet terminal: new minus AllLocal -.015, new minus original LLF +.013. Speed .523step/s versus original LLF .525 (-.4%) and AllLocal .5271 (-.8%). The hypothesis is that restoring free W_V accounts for an important portion of the original LLF benefit; retained fetched O should still help.

Focused CPU checks cover exact full budgets and unchanged parent's parameter tree, no W_V anywhere, static V but no static O in F, scanned forward gradients with an explicitly awakened zero-init W_R, and actual train-health export for F local V and fetched O. Normal training/basic+concat health stays enabled. Preparation is CPU/AOT/training-prequeue in parallel, gated before launch.
