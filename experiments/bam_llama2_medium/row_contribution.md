# Q/K/V/O row-read contribution in BAlignedRow

Status: implementation/validation; no measured contribution yet.

- Model: `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow`.
- Training source: `77401da6f83a5aa6ddd61994e028c3c694221518`.
- Checkpoint: `gs://newproject-1-llm_projects_us-east5/log/BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow/checkpoints/13500/items`.
- Worktree `/data0/xd/llf-row-contribution`; branch `codex/llf-row-contribution`.
- Use exactly first64 sequences of existing Pile T2048 seed9876 128-sequence cohort;
  record hashes of every batch field and checkpoint identity. No new sampling.
- Four diagnostic v6e-1 workers each own a disjoint16-sequence range. Prefer EW4a;
  candidate backups UC1a/UE5a. Names begin `xd-v6e-rowko-`; no training TPU is used.
- GCS artifact root `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/llf-row-contribution-13500-0914`,
  separate worker directories; local `/data0/xd/bam_diagnostics/llf-row-contribution-13500-0914`.

## Intervention and interpretation

At every token, multiply only the final row-output coordinates by a runtime scale,
leaving column output, RMS normalization, read gates and mixing unchanged. For
Local Q/K/V this is after `_read_local`; for LocalO/fetched O it is after
`_read_fetched_m`, before injection into the residual/attention path. Coordinates
32:64 are the row footprint (V/O use8 active compressed coordinates plus zeros).
Q/K's downstream RoPE runs normally. All downstream attention, writes and losses
are recomputed: this is whole-network necessity at the trained checkpoint, not
frozen downstream activation attribution or retraining benefit. Parameter trees
and production code are unchanged. No raw `sow` activation capture.

141 unique named scenarios, one dynamic shape:
-16 complete Q/K/V/O knockout coalitions, including all-on and all-off;
-5 half-amplitude controls (each path alone and all paths together);
-88 single-layer/path knockouts (Q/K/O at24 layers, V at16 Local layers);
-32 LLF-unit/path knockouts.
Layer0 and nonexistent F-layer V are explicit no-op controls. All-on is checked
against ordinary forward; batched variants also checked against serial variants.
Up to8 variants execute together with `vmap`; warm throughput is measured.
On baseline mismatch, use serial variants rather than change the comparison.

Exact four-player Shapley allocates `loss(all rows off)-loss(native)` across
paths, averaging marginal knockout effects over every coalition. It sums to the
total by construction, but is a context-averaged allocation, not a unique causal
truth. Report path-only knockouts alongside it. Single-layer knockout values need
not sum to path-only knockout; LLF knockouts expose local interactions. Report
per-sequence paired loss gaps and descriptive uncertainty; do not call every
small layer ordering statistically established.

Row-key kernel costs alone across the model: Q=.75 W_Q, K=.75, V=2, O=12;
W_Q=1024². They total15.5 W_Q (~16.25M weights), before row-related gates/mixing
or contractions. This is a row-projection denominator, not all BAM overhead.
No speed saving is inferred from multiplying an already computed row by zero.

## Reproduction

Runner `experiments/bam_llama2_medium/row_contribution.py`, launcher
`run_row_contribution.sh`, summarizer `summarize_row_contribution.py`.
Each worker launches with `ROW_START=0/16/32/48`, `ROW_STOP=16/32/48/64`,
`ROW_STAGE=all`, `ROW_VARIANT_BATCH=8`, distinct `ROW_OUTPUT` and `ROW_GCS`.
See final metadata for exact diagnostic commit, cohort hashes and resolved config.
First32 and final64 aggregate results will be reported before closeout.
