# MediumProp sparse MLP write content projection

Worktree: `/data0/xd/mediumprop-qk75-sparse`; branch `codex/mediumprop-qk75-sparse`.
Parent: `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`, runtime `23c692b7eaa56b356945281972840fab2b973681`.
Both arms are **18-layer V32/C8, QK57+RoPE18**, writing at TB layers 1/4/7/10/13/16.

| RUN | MLP widths | write content before existing per-head RMSNorm | total parameters | TPU |
|---|---|---|---:|---|
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdWOTransposeTruePile` | 3901/3774/3901 | y W_Oᵀ, same layer W_O and live gradients |432096128|xd-v5p-16-2910117-maxtext|
| `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdSeparateProjectionTruePile` |3901/3374/3901|y P, independent learned D×D per write layer|432096128|xd-v5p-16-2910118-maxtext|

No change to the residual branch: original y enters x; projected content only enters M. Address, sigmoid write gate, content RMSNorm, attention write and single carry decay retained. W_Oᵀ is an adjoint, not an inverse. Independent kernels use exactly the W_O Gaussian/fan-in initialization convention, not identity; 6×D²=6W_Q are repaid with400 MLP units per write layer.

Direct baseline for both is the original18-layer parent; independent also compares shared transpose. Generic and existing BAM concat/write health retained. TruePile4096 data routed to trainer zone. SPOT v5p-16 UE5a; user-owned FLEX_START compiler borrowed in EW4a, never lifecycle-owned or deleted.

Frozen prelaunch bets at13500: shared−parent −.004, .512 step/s vs parent .520 (−1.5%); independent−parent −.002, independent−shared +.002, .520step/s (flat). Extra content GEMM costs1W_Q per write layer; independent pays it out of smaller MLP.

Focused CPU: exact full count/tree sharing, complete train-step graph with all writer health keys; actual adjoint/identity/gradient algebra; scanned finite forward/backward with projection parameters consumed. Normal Medium1000-step reports,200-step windows; reviews2800/5000.
