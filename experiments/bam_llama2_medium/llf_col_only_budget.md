# All-row removal at the PerLayer parameter budget

RUN: `BamMediumIndependentLLFMLPPerLayerColOnly`.
Implementation: `codex/llf-parameter-matched`, `/data0/xd/llf-parameter-matched`.
Parent: `BamMediumIndependentLLFBAlignedRowMLPPerLayer`, runtime `94e13f9`.

Question: at fixed parameter budget, are BAM row reads more valuable than MLP
capacity? A positive RUN-minus-PerLayer loss gap supports row reads; a larger
gap is stronger evidence of their value, not a failed throughput optimization.
Prediction: final gap +.010; throughput +4% (secondary, low confidence).

Remove Q/K row reads in all layers, V row reads in Local layers and O row reads
in both Local/fetch layers. Delete corresponding key, bias, gate and head-mix
parameter slots. Delete the now-useless stored O row decoder (already unused in
the parent's Direct mode). Keep all column reads, their gates/routing/scales,
M writing, M32x32 carry and C8 compression. M-cache size is unchanged.
LocalQ/K remain rank1 legacy, LocalV rank4 B with scale1; other read scales2.
The aligned-row projection is not invoked because there is no LocalV row output.

Compact Local parameters are expanded with constant zero row slots only at the
existing read-helper boundary, preserving the tested normalization algebra.
The static column-only selector skips every row-M contraction. These zeros are
not trainable parameters. Column head-mix initialization selects the original
bilateral initializer's column slice rather than changing the random draw.

## Exact parameter accounting

SwiGLU has three bias-free matrices; one extra channel costs 3D=3072 parameters.
No hardware-alignment rounding is used. Integer channel counts cannot represent
the final per-layer residue; use the largest width within the parent budget.

| Layer | Original MLP | Deleted parameters | Returned MLP channels | New MLP | Budget residue |
|---|---:|---:|---:|---:|---:|
| L (16 layers) | 2256 | 858338 | 279 | 2535 | -1250 |
| F (8 layers) | 2400 | 645202 | 210 | 2610 | -82 |

Total removed 18,895,024; returned 18,874,368. Total model parameters 412,081,840
versus PerLayer 412,102,496: -20,656 (0.0050% of total; 0.0067% excluding the
unchanged embedding/output projections). The model still has 465,584 more
parameters than clean MHA. This is integer-granularity near matching, not exact
matching and not an optimized hardware configuration. No dummy parameters added.

Comparisons: PerLayer (allocation effect), clean MHA (remaining equal-budget BAM
advantage). Both use clean WD, generic health ON, BAM-specific health OFF.
Train from scratch with 24-layer LLF block-scan/AOT, 13500-step original schedule,
checkpoint200 and forced final checkpoint. Formal primary UE5a, EW4b backup
after300s; compiler primary EW4a, backups UC1a/UE5a after300s.

## Reproduction and checks

`experiments/bam_llama2_medium/audit_matched_mlp.py` counts real shaped parameter
trees; JSON: `/data0/xd/llf-col-only-final-audit.json`. A pre-MLP-reinvestment
audit is at `/data0/xd/llf-col-only-param-audit.json`. Use the pinned CPU environment:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=MaxText \
 /data0/xd/conda/envs/maxtext-cpu/bin/python \
 experiments/bam_llama2_medium/audit_matched_mlp.py \
 BamMediumIndependentLLFBAlignedRowMLPPerLayer \
 BamMediumIndependentLLFMLPPerLayerColOnly \
 --output /data0/xd/llf-col-only-final-audit.json
```

`MaxText/tests/bam_col_only_budget_test.py` compares compact parameter modules
with mapped bilateral parameters and disabled row reads, for both L/F paths,
using nonzero read keys; checks parameter shapes and finite nonzero gradients.
The full-model shape audit exercises the 8-block scan with non-aligned MLP widths.

Runtime prepared at `2ca927c1a76011a247303efccbb3afd5b868ffd2`.
Validation: 15 local-module/scan tests and 43 base BAM regressions passed.
Logs: `/data0/xd/llf-col-only-tests.log`, `/data0/xd/llf-col-only-regression.log`.
AOT orchestration state:
`tpu-ag:/home/lishengping/xd/projects/aot_runs/2ca927c-79f7102d.json`.
Region evidence was rechecked on 2026-09-15: UE5a had substantial Medium churn
on September14 and correlated Anchor recoveries on September15. Recent EW4b
long leases concern older v5p-32 runs, not a simultaneous v5p-16 comparison.
UE5a remains primary with EW4b staged backup; do not interpret that choice as
proof of context-independent regional superiority.
