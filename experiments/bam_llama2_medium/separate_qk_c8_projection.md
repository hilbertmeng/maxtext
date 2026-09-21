# Separate QK C8 compression: Medium K48 and K64

Implementation worktree `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
The two runs isolate whether sharing the address compression between QK and VO/FetchedO causes the K48 DirectC8 regression.

Each layer adds a 32×8 `local_qk_c8_projection`, shared by Q and K only.
LocalVO and FetchedO retain `abs_v_cache_projection`. The new projection copies the latter at initialization;
existing parameters and initial outputs are exactly preserved. Static Q/K continue reading full M.
MLP widths stay 3050/3050/3045, per explicit user instruction. Added parameters: 6144 total
(0.00149% of parent, 0.005859 W_Q total; 0.000244 W_Q per layer).
Total parameters 411888512; M cache unchanged. Gates, scales, WD, data, schedule and health unchanged.

- RUN `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8SeparateQKProjectionMLPPerLayer`
  Direct baseline `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48DirectC8MLPPerLayer`; ID `qkstatic-k48-directc8-separate-qk-proj`.
- RUN `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8SeparateQKProjectionMLPPerLayer`
  Direct baseline `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48DirectC8MLPPerLayer`; ID `qkstatic-k64-directc8-separate-qk-proj`.

K48 replaces the existing Medium K48 DirectC8 combination on its allocated TPU, restarting at step0 under a new RUN.
K64 receives a separate trainer. UE5a primary, UC1a/EW4b backup after5minutes without capacity.
Plan13500, checkpoint200, review2800, report1000; generic healthON plus968BAM metrics.
Compiler UC1a primary; EW4a/UE5a backups.

Validation: pinned BAM suite54 existing tests pass; new test passes K48/K64 × L/F,
checking exact preservation of initial parameters/output, projection isolation, and finite/nonzero gradients
once the zero-initialized O read key has begun learning. Both full-size parameter/sharding audits pass;
actual train-step traces produce968health scalars. Audit artifacts `/data0/xd/separate-qk-{audit,trace,focused}.log`.

## Launch verification

Both use sealed runtime `27acf149c5223b29666680777eec8070cf77fa76`, exact v5p-16/s13500 AOT.
K48 took over `xd-v5p-16-qkstatic-vo-c8-ig-k48-qk48-directc8-maxtext` in UE5a after the parent
paused with committed4891; first0, steps10–14 .6324 (−.41% vsparent.6350).
K64 runs on `xd-v5p-16-qkstatic-k64-directc8-separate-qk-proj-maxtext`, UE5a;
first0, stable20–24 .6162 (−1.25% vsparent.6240). Steps12/13 had a timing-outlier pair;
their arithmetic speed mean is not used. Both controller/registry commits and comparisons verified;
AOT loaded and968BAM health metrics retained. All compiler candidates released.
Raw launch verification `/data0/xd/separate-qk-k{48,64}-launch.json`.

K48 monitoring also compares the original non-DirectC8 shared-rank4 baseline
`BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer`,
per user request: distinguish repairing the DirectC8 regression from a net gain beyond rank4.
Speed .6324 vs .6378 (−.85%), matched health. This is a monitoring-only update; runtime unchanged.

K64 monitoring also includes the original shared-rank4 baseline
`BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK64QK48TruncateMLPPerLayer`,
completed13500, runtime8c188b0. Speed .6162 vs .6320 (-2.50%), matched968 health.

## Closeout

K48 stopped4961; last5 through4800: -.006718 vs shared-P DirectC8,
+.002357 vs original K48 rank4. Sharing the QK/O compression explains much of the
DirectC8 regression, but separating it still did not beat rank4; speed -.85% vs rank4.
K64 stopped4172; last5 through4000: +.005187 vs shared-P DirectC8,
+.007086 vs K64 rank4. Its extra deficit versus DirectC8 flattened around+.005;
no net gain versus rank4, speed -2.50%. Both retain the same cache and MLP widths.
Both final checkpoints committed; TPU/queues absent; TensorBoard SYNC_OK.
Raw reports `/data0/xd/k48-split-final.md`, `/data0/xd/k64-split-final.md`;
closeout `/data0/xd/medium-split-closeout-summary.json`.
