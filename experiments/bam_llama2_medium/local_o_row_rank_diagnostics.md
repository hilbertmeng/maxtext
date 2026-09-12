# LLF BAlignedRow: choosing dynamic O row rank

Status: acquiring v6e for spectral stage; no rank recommendation yet.

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
