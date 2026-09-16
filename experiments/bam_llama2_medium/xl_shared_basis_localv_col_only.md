# XL SharedBasis LocalV column only

- Experiment: `BamXLSharedBasisLocalVColOnlyRank4CFp32`
- Direct baselines: `BamXLSharedBasisLocalVRowSharedColRank4CFp32` and
  `BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis`
- Worktree: `/data0/xd/xl-shared-basis-row-only`
- Branch: `codex/xl-shared-basis-row-only`
- Target: spot EW4b `v5p-32`, 50,000 steps, scan/AOT

The experiment removes LocalV's row branch entirely. LocalV keeps its independent full-M
64x32 rank-4 effective-key (C-fp32) column read. LocalO and its compressed-C8 row read remain
unchanged, but that row answer is no longer added to V. Q/K SharedBasis, F, write, optimizer,
schedule, and M-cache are unchanged.

This separates two conclusions: RowShared tying the original baseline shows that the independent
full-M LocalV row read can be replaced; ColOnly tests whether LocalV needs any row signal at all.
Relative to RowShared it removes only the LocalV row gate and row-answer add, because the LocalO
read is still required. The parameter saving is 32,784 weights per L layer (`0.00782 W_Q`), or
524,544 weights across 16 L layers (`0.125 W_Q`).

Pre-run bet versus RowShared: dloss center `+0.0003`, likely `[-0.001,+0.002]` by 10k; throughput
tied. A tie would support pure-column LocalV; a positive gap would measure the value of the shared
LocalO row signal.

## Runtime

- code commit: `c742660d68c48e8e93fe7fe3b839006567450664`
- AOT artifact: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/c742660/jax081-i0ae3f58-c17f538a/v5p-32/s50000/BamXLSharedBasisLocalVColOnlyRank4CFp32.pickle`
- TPU / FIRST_STEP: `xd-v5p-32-xl-localv-col-only-maxtext` in EW4b / step 2
- speed: `0.5527 steps/s` over steps 10-19
