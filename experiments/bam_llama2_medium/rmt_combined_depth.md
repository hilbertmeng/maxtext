# Dynamic RMT boundary combination: MLP capacity versus depth

Implementation: `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`.
RUN: `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32L22`.
Parent: `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32` (`be5491f`).
Main exp.py is ledger-only; no RMT implementation is merged into main.

Retain dynamic embedding + Direct32 unembedding, all-local matrix attention,
QK57 + independent RoPE18, VectorNorm, middle tail32 C8 reads and full48
writes, health metrics, optimizer, data and total13500-step schedule.
Depth18->22; uniform MLP4078->3212. Use native layer scan, not the inherited
historical LLL block scan. No new forward implementation is necessary.

Standard MHA MLP width3200 would release56894400 params (39.510 W_Q).
Each full dynamic RMT layer at3200 costs13976912 (9.706189 W_Q), including
2456912 non-MLP params. Four layers fit. With22 layers and MLP3212, total
432083008,38192 below MHA432121200 and36352 below parent432119360.
Do not use hardware rounding. Uniform3213 would exceed MHA by41008.

Direct loss/speed baseline: parent combination only. Pre-run13500 bet:
loss-.008, steady step/s-15% vs parent .3777115. The test asks whether extra
matrix updates outweigh the MLP capacity they displace; retain boundaries.

CPU gate is targeted: actual full-shape parameter-tree audit, small native
layer-scan forward/finite gradients including the last layer, all per-layer
and boundary health export, and the verified parent's combined-boundary
initialization/gradient regression. Two bounded CPU groups run independently;
no full BAM suite or unrelated port/profile tests. AOT and UE5a trainer queue
run concurrently; any CPU failure blocks formal training.

Training TPU: `xd-v5p-16-2609268-maxtext`, UE5a; no backup-zone migration.
Borrowed AOT host: EW4a `llm-jax-v6e-1-1`, verified FLEX_START+idle; never
adopted into cleanup. AOT/CPU/queue are managed by launch_train_parallel.py.

Startup: runtime `a8d5bcdb93f2b78031f6d00f86c4d08718a9ae60`, pushed.
CPU full-shape audit passed; two targeted checks passed in71.25s in parallel
(groups52.23/71.25s), before AOT completion. Formal runtime check passed.
AOT verified on the borrowed FLEX_START compiler; UE5a prequeue READY
2026-09-27T00:29:56Z (registry first observed00:30:07Z), FIRST_STEP0 at
~00:33:12Z, finite loss10.852458. Retained compiler ownership unchanged.
Artifacts: `/data0/xd/rmt-combined-l22-launch.log`; preparation/cpu/aot/queue
logs under `/home/xd/.local/state/maxtext-parallel-launch/` (L22 prefix).

Actual AOT load and step100 verified. Steady steps20-99 (80 samples,
inverse mean rounded log latency): 0.3190472 step/s vs parent 0.3777115,
-15.53%. Same inherited generic/layer-health and both boundary-health
flags; layer count increases18->22. Speed prediction(-15%) matches closely.
Evidence: `/data0/xd/rmt-combined-l22-compiled-loaded.log`,
`/data0/xd/rmt-combined-l22-speed.json`, `/data0/xd/rmt-combined-l22-step100.log`.

Review snapshot2026-09-27: actual~3300, committed checkpoint3200. Same-step
L22-parent gap +.030313@1000 -> +.014344@1600 -> +.008998@2000 ->
+.006331@2800 -> +.003987@3200. Latest5+.006164 (.003987..+.008158).
Continue beyond2800: the deficit is still closing materially; no stop.
The brief800-1000 slowdown did not persist. Final-loss bet remains unresolved.

Report refresh: at actual3480/checkpoint3400, gap+.004508, slightly above
+.003987@3200. Latest5+.005434 (.003987..+.006943); still closing over
1000-step windows. Continue; no sign crossing yet.
