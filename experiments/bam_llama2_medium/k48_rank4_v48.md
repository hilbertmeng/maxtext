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
- Bet: final gap -.003, plausible -.006..+.002; 65% chance of a lower loss.
  Speed -1..-4% vs UE5a .6378 steps/s. Benchmark judgment must distinguish a
  small incremental gain from a substantial improvement over the existing ~-.1 MHA gap.
