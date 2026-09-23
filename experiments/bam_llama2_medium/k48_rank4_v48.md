# K48 shared-rank4 V48

- Worktree `/data0/xd/k48v48-rank4`; branch `codex/k48v48-rank4`, from `f2b26dad`.
- RUN `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerV48`; direct baseline `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer`.
- Trainer `xd-v5p-16-k48-rank4-v48-maxtext`: UE5a primary; UC1a/EW4b backups.
- Raw M48x32 → M48x48; C8, QK48+RoPE16, shared rank4, independent VO gates,
  P_loc GELU256 and inherited initialization/scales unchanged.
- +132032 BAM parameters/layer (.125916 W_Q); MLP -43/layer to 3007/3007/3002.
  Total 411883904, -1536 vs parent; compressed fetched cache remains 48x8.
- Full13500 schedule, checkpoint200; review2800, report batches~1000 steps (200-step windows).
- Generic training health and 968 BAM read-health scalars retained to match baseline.

Launch verified: runtime `768fc805d98954ab30be42b6095560438d2fc768`, UE5a.
AOT loaded and FIRST_STEP confirmed; step10–14 mean .6130 steps/s,
-3.89% vs same-health K48 .6378. Parameter/shape audit and nonzero L/F forward
and gradient checks passed; runtime attention source inherits the 46-test validated
merge. Evidence: `/data0/xd/k48v48-launch-evidence.txt`,
`/data0/xd/k48v48-traintrace.log`, `/data0/xd/k48v48-forward.log`.

Stopped at2891: early loss advantage eroded to near zero; last5 through2800
mean -.001117, last3 -.000399 vs K48. Speed -3.89%, fetched cache unchanged.
Checkpoint2891 committed, TPU/queue absent, TensorBoard synced.
