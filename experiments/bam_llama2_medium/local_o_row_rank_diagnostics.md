# LLF BAlignedRow: choosing dynamic O row rank

Status: spectral stage complete on128/128 sequences; group causal rank sweep running.
No retraining rank recommendation yet.

## Scope and reproducibility

- Model: `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow`.
- Original training source: `77401da6f83a5aa6ddd61994e028c3c694221518`.
- Checkpoint: `gs://newproject-1-llm_projects_us-east5/log/BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow/checkpoints/13500/items`;
  `commit_success.txt` verified before acquisition.
- Branch/worktree: `codex/llf-o-row-rank`, `/data0/xd/llf-o-row-rank`, based on original runtime.
- Capture runtime: `cbb1042ad642b522e7b089b796d3ac5323ecf215`.
- Runner: `experiments/bam_llama2_medium/run_local_o_row_rank_probe.sh`;
  capture/statistics: `local_o_row_rank_probe.py`; aggregation: `analyze_local_o_row_rank.py`.
- TPU candidate: `xd-v6e-localo-rowrank-ewa4a-0912`, EW4a v6e-1.
- Cohort: fixed128 Pile T2048, seed9876, all valid positions, existing
  `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`.
  File and sequence hashes are verified and recorded in metadata.
- Artifacts: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/llf-o-row-rank-13500`;
  local `/data0/xd/bam_diagnostics/llf-o-row-rank-13500`.

## Questions

1. At each token/layer, how many dynamic bases preserve actual row-key directions,
   amplitude-weighted keys, and native row outputs? Retain full rank1–16 curves,
   not a preselected rank4 verdict. Native output has only8 coordinates.
2. How do raw, RMS-normalized, and gated spectra differ? Keep signed and absolute
   head correlations separately. L layers read local compressed M, F layers read
   actual fetched M. Do not concatenate positions before computing local rank.
3. Does key-optimal truncation discard directions irrelevant to M, or distort
   readouts despite high key energy? Measure actual output error and output-optimal
   energy bounds; all16 destination heads and their W_O blocks remain present.
4. Follow spectral elbows with same-batch causal rank sweeps at individual layers
   and L/F/all-layer groups. Keep rank-as-oracle separate from trainable A(x)/H(x)
   realizability and from C-routing normalization/gate constraints.

## Resource and validation plan

TPU runs batch1 forward/capture. CPU uses at most16 layer tasks with BLAS thread1,
at most two activation batches in flight; subsequent TPU inference overlaps CPU
SVD. Benchmark four serial versus parallel layers and assert agreement; capture
must reproduce ordinary forward loss. Save per-sequence statistics atomically,
incrementally upload directly from worker to GCS. No activation vectors are retained.
Record CPU count, throughput and pipeline latency; adjust concurrency based on
measurements. L0 zero-energy quantities are undefined, not evidence for rank0.

## Complete spectral results

| Layer group | Output energy r2 | r4 | r6 | Key-SVD read error r4 | r8 |
|---|---:|---:|---:|---:|---:|
| Local (exclude zero-M L0) | .83880 | .96306 | .99331 | .24181 | .08049 |
| Fetch | .83280 | .95981 | .99249 | .22998 | .07836 |

Local aggregation includes all15 nonzero Local layers; Fetch includes8 layers.
Energy is squared Frobenius singular-value energy, averaged per-token then per-sequence;
read error first energy-weights tokens within a sequence, then averages sequences/layers.
The two columns therefore have deliberately different weighting: both are retained
in raw per-sequence files, not silently presented as the same statistic.

Key rank4 retains only~64–65% energy; key rank11 is needed for~95% on average.
Output rank4 retains~96%, rank6~99.3%, but this is an M-dependent oracle, not
proof that cheap A(x)/H(x) projections will learn the useful subspace.
L/F averages are similar. Deep layers tend to be less compressible; e.g. F20
output rank4 retains .9431 versus F5 .9743. Layerwise/full-rank curves are in
`/data0/xd/bam_diagnostics/llf-o-row-rank-13500/summary.md` and aggregate.npz.

After normalizing every head to unit norm, output rank4 still retains~90–95%
in most layers (L1~99.45%), versus~50–57% for keys. Thus low output rank is
not solely a consequence of head-amplitude imbalance. Output signed mean cosine
is near zero while mean absolute cosine is~.43–.48 outside L1; both signs matter.
The layerwise correlation table is included in summary.md, using all128 samples.

Capture versus ordinary forward is exactly equal on sample0. Worker44 CPUs/172GiB;
16 threads, BLAS1, two in-flight samples; four-layer serial2.64s versus parallel.92s
(2.87x). Stable two-sample pipeline latency~6.4s, throughput~one sample/3.2s,
RSS~8.4GiB; observed process CPU~5–6 cores, not full44-core utilization.

## Causal follow-up

Runtime `0633d909d53a718c773b7e08ae1ee39c10178b15`, same worker/checkpoint/cohort.
`ORANK_STAGE=ablation ORANK_SCOPE=groups ORANK_RANKS=1,2,3,4,5,6,7,8,12,16`
with the same launcher. Key-optimal and output-optimal oracle truncation preserve
the original head destinations. Runtime layer/rank selection shares one compile;
inactive layers skip eigensolves. No-op forward delta is exactly zero on sample0.
Rank16 keys/rank8 output are explicit no-op controls. Full per-sequence loss deltas
and scenario ordering are saved in ablation_groups_*.npz/json.

Report incremental paired results at 32/64/96/128 sequences, rather than waiting
for all128. Use `summarize_local_o_row_ablation.py ARTIFACT_DIR --limit 32`
(increase the limit at subsequent milestones). It checks exact no-op controls
and reports descriptive sample-level uncertainty, not corrected significance.
After the first32, prioritize unresolved ranks/layer groups rather than blindly
extending the full54-scenario grid. The running grid costs~34s/sequence;
this is TPU intervention-forward work, unlike the CPU-heavy spectral stage.

### First32 causal results

Positive means higher loss. Each intervention applies at all token positions in
the selected layer group, then executes the remaining network normally.

| Layers | Key-optimal r4 | Output-optimal r4 | Output-optimal r6 |
|---|---:|---:|---:|
| Local | +.010554 | +.000600 | +.000215 |
| Fetch | +.007260 | +.000508 | +.000105 |
| All | +.017956 | +.000907 | +.000290 |

All-layer output-r4/r6 descriptive mean ±1.96 SE is ±.000469/±.000317.
Key-r4 has a positive gap on all32 sequences; output-r4 on23/32.
The complete first32 rank1–16 table is `causal_first32.md` in the artifact directory.
This supports useful low-rank *outputs*, not low-rank fitting of the original keys.
It does not establish a learned rank4/6 reader's retraining loss.

The broad scan was ended after this milestone. Focused continuation runtime
`595cfff33c15e8a69fc11606a651611b09096d6d` uses the unchanged intervention function,
`ORANK_TAG=groups_focus ORANK_START=32 ORANK_STOP=128 ORANK_RANKS=4,6`.
Twelve scenarios replace54. `summarize_local_o_row_ablation.py --focus` merges
matching scenarios across the two batches and verifies cohort/checkpoint identity.

### First64 update

| Layers | Key-optimal r4 | Output-optimal r4 | Output-optimal r6 |
|---|---:|---:|---:|
| Local | +.010681 | +.000632 | +.000291 |
| Fetch | +.007687 | +.000630 | +.000080 |
| All | +.019126 | +.001204 | +.000283 |

All-layer output-r4/r6 mean ±1.96 SE: ±.000380/±.000233. Rank6 is not established
as exactly loss-free. Three overlap sequences (32–34) have exactly identical
baseline and matched scenario gaps between broad/focused executions.
Focused throughput is~8.5s/sequence versus34s before (~4x faster).

For a prospective dynamic row reader, projection dimensions are R*(32+16) versus
16*32=512; contraction MACs are R*32*8+16*R*8 versus16*32*8=4096.
Both decrease62.5% at R4,43.75% at R6,25% at R8, before Gram/routing overhead.
These are operator estimates, not whole-model throughput predictions.

## Final result (user-ended scan)

Full spectra:128 sequences. Broad causal grid: first32 reported (35 completed).
Focused r4/r6 causal grid:109 contiguous sequences after merging. User ended
further measurement because the conclusion was clear; no claim of128 causal samples.

| Layers | Key-optimal r4 | Key-optimal r6 | Output-optimal r4 | Output-optimal r6 |
|---|---:|---:|---:|---:|
| Local | +.009706 | +.005054 | +.000630 | +.000120 |
| Fetch | +.007344 | +.003459 | +.000576 | +.000027 |
| All | +.017893 | +.008877 | +.001118 | +.000140 |

All-layer output-r4/r6 ±1.96 SE: ±.000315/±.000174. R4 leaves a small loss;
R6's mean is small and its descriptive interval includes zero. Rank2/3 have
larger all-layer output-oracle loss (+.00882/+.00320 on the first32).
Thus dynamic rank4 is a plausible efficiency experiment, not a diagnosed loss-free
replacement. C-fp32 does not itself guarantee that x-generated bases recover the
M-dependent output-optimal subspace. L/F output losses are comparable at r4;
do not assign layer-specific ranks based solely on spectral energy.

`causal_final.md` retains the complete focused table and uncertainty.
User approved training L/F O-row rank4 C-fp32 on BAlignedRow in a separate worktree:
`codex/llf-o-row-rank-training`, `/data0/xd/llf-o-row-rank-training`, based on the
original BAlignedRow runtime77401da6. No other read arm changes.

Workflow lesson: report32-sequence batches and narrow settled branches promptly.
The initial54-scenario grid cost34s/sample; narrowing to12 cut this to8.5s/sample.
Three overlapping samples verified exact losses across the runner change.
All scripts and per-sequence artifacts retained. On2026-09-12 the exact diagnostic
TPU `xd-v6e-localo-rowrank-ewa4a-0912` and its queued-resource were deleted and
verified absent. This does not refer to subsequent training/AOT resources.
