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

Sealed runtime d296d0eea0d8fb74665f542d7a8e92abe7b03df8. Focused five-check CPU gate passed49.83s, two disjoint8-core groups; both launchers share `/data0/xd/bam_diagnostics/mediumprop-mlp-residual-projection-launch/cpu-receipt.json`. Earlier failures were numerical test fixtures only: small residual subtraction and equivalent float32 reduction order; corrected tests amplify the isolated signal and use exact small-integer algebra probes. No architecture/init change, no training started on rejected sources. Both owned pending trainer queues were verified deleted on failed gates; retained compiler remained untouched.

Both passed FIRST_STEP and loaded the AOT executable on d296d0e, with zone-local UE5a TruePile4096. Initial periodic checkpoint200 committed. Median steps20–99: shared .521 (+.2% vs parent .520); independent .515 (-1.0%). No nonfinite health scalars in initial TB capture. Continue normal1000-step monitoring.

2026-10-08 update: Shared residual WO through7000: parent deficit slowly narrowing, last5+.001852 [+.001395,+.002322], continue. Independent residual stopped7195: parent +.003046 [+.002451,+.003757] at7000; vs shared +.001193 / old independent write+.000642. Initial gain reversed; slower than both alternatives. Resources absent, CP7195 committed, TB SYNC_OK. Bet parent -.002 failed; independent/shared +.001 is close to observed +.001193.

2026-10-08 final shared residual: stopped8578. Last5@8400 vsparent +.001798 [+.001111,+.002686], deficit slowly narrowed but recent catch-up slowed; old write-side shared -.002401@5000. Fits800-8400 exponent+plateau terminal+.002046 / power+plateau+.001639; recent4000 linear+.000016, recent5000 linear+.001218. Forecasts do not establish a useful terminal win; stopped, no parameter reduction and measured speed+.2%. Artifacts in `/data0/xd/bam_diagnostics/mediumprop-mlp-residual-projection-launch/shared/extrapolation-8400.{json,png}`. Resources released, CP8578 committed, TB SYNC_OK.
