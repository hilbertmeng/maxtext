# MediumProp sparse MLP residual projection

Worktree `/data0/xd/mediumprop-qk75-sparse`; branch `codex/mediumprop-qk75-sparse`.
Parent `BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile`, runtime `23c692b7eaa56b356945281972840fab2b973681`.
L18, D1200, M75x32/C8, QK57+RoPE18; TB writer layers1/4/7/10/13/16.

MLP output y splits into two branches. M uses the original y reshaped to16x75, with the parent's per-head content RMSNorm, independent GELUR256 address, write gate and carry decay. Only the vector branch becomes y_head W_O or y_head P. Both are forward head-to-embed maps (not transposes); no post-projection normalization. All other twelve layers retain the parent's ordinary MLP residual addition.

| RUN | residual map | MLP widths | parameters | owned TPU / zone |
|---|---|---|---:|---|
| BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualWOTruePile | actual same-layer W_O, live gradient |3901/3774/3901|432096128|xd-v5p-16-2910119-maxtext / UE5a|
| BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdResidualProjectionTruePile | independent16x75x1200 Gaussian map, same W_O fan-in convention |3901/3374/3901|432096128|xd-v5p-16-2910120-maxtext / UE5a|

Independent adds6W_Q=8640000 parameters and repays400 MLP units at each writer; main GEMM FLOPs unchanged. Shared adds no parameters, keeps MLP widths, and adds one D² multiply per writer. Parameter sharing does not eliminate that multiply.

Direct baselines: both original18-layer parent; shared also old write-side W_O transpose; independent also shared residual and old independent write-side projection. Generic+concat+MLP-write health retained, no new timing-only health toggles. Normal Medium1000-step reports with200-step windows;2800/5000 review, initial launch plan13500.

Frozen prelaunch bets: shared-parent -.003, .512step/s vs .520 (-1.5%); independent-parent -.002, independent-shared +.001, .520step/s (flat). Shared rationale: attention and MLP head content can use one learned head-to-residual coordinate map, without narrowing MLP. This is a prediction, not a verified mechanism.

Borrow user-owned FLEX_START v6e-1 llm-jax-v6e-1-0 in EW4a serially for the two AOTs. Never delete/reinstall/enroll it in automatic cleanup. Formal trainers are SPOT v5p-16 in UE5a, selected from current matching leases; TruePile4096 uses that zone's replica. CPU checks, AOT, and trainer prequeue run concurrently; both launchers share a checked sealed-source CPU receipt.

Focused checks: exact full parameter counts and writer health train-step graph; actual W_O forward contraction and live gradient (not adjoint); changing the independent residual kernel leaves this layer's M output exactly unchanged; scanned finite forward/backward with projection parameters consumed; invalid-option guards.
