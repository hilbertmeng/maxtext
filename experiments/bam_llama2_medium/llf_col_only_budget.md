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

## Pure row removal with the original MLP

`BamMediumIndependentLLFBAlignedRowColOnly` keeps BAlignedRow's 2816 MLP channels
in every layer, while reusing the exact compact column-only implementation above.
It removes all Q/K/V/O row reads, not just fetched O. Write, column reads, C8,
LLF schedule, key scales and clean WD rules remain unchanged. No parameters are
reinvested. Shape audit counts 430,956,208 parameters, down 18,895,024 from
BAlignedRow's 449,851,232; LocalV key scale remains 1.0.

Comparisons: BAlignedRow (pure deletion penalty), PerLayerColOnly (MLP capacity
with rows absent). Prediction: final gap +.008
vs BAlignedRow, throughput +5% vs matched generic-health-ON .6836 steps/s.
The prediction is uncertain: the old V2 fetched-O-only ablation cost +.0108,
and linearly transferring the MLP-budget result is not established.
Use 13,500 steps, scan+AOT, checkpoint200; generic health ON/BAM health OFF.
Do not early-stop merely because the ablation has a positive gap at2800.

Audit artifact: `/data0/xd/llf-baligned-col-only-audit.json`;
test log: `/data0/xd/llf-baligned-col-only-tests.log`.

### Row parameter fractions

Count all BAM additions over the same D1024 MHA (411,616,256 total), including
stored decoder parameters, consistently with the deletion audit above.
V2 shape audit: `/data0/xd/v2-row-param-audit.json`.

| Model | BAM additions | All row-specific parameters | Fraction |
|---|---:|---:|---:|
| `BamLlama2MediumV2` | 31,677,024 | 15,484,848 | 48.88% |
| `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow` | 38,234,976 | 18,895,024 | 49.42% |

V2 row parameters per layer: O=1024*16*32 +1024*16 +16 +16*8*32
=544,784; Q/K combined=2*(1024*(32+1+16)+32+1)=100,418.
Across24 layers O rows are13,074,816 (41.28% of BAM additions), Q/K rows2,410,032
(7.61%). The historical FetchedColOnly ablates only O outputs, leaving Q/K rows;
its implementation masks the row contraction rather than necessarily removing
every row parameter. Stored C8-to32 decoders are unused by Direct injection;
they remain counted here solely for consistency with the actual parameter trees.

## O-only row removal

`BamMediumIndependentLLFBAlignedRowOColOnly` removes O-row parameters and reads in
both L (LocalO) and F (fetchO) layers. All Q/K/V row and column reads, O column
reads, writes, M compression and MLP2816 remain unchanged. It inherits the same
clean training settings; only O read-arm compaction is new. Comparisons are
BAlignedRow and the all-row-removal RUN, with no MHA comparator.
Prediction: final gap +.006 vs BAlignedRow, speed +3%; these are hypotheses, not
an additive attribution of pathway value.

Parameter audit: 436,776,416, down13,074,816 (34.20% of BAlignedRow BAM additions).
JSON `/data0/xd/llf-o-col-only-audit.json`; mapped nonzero forward and gradient
checks for both L/F in `MaxText/tests/bam_col_only_budget_test.py`, log
`/data0/xd/llf-o-col-only-tests.log`. Q/K/V parameter trees stay intact.

All-row RUN launch verified at runtime4c67f28 on UE5a with `Loaded compiled
function!` and FIRST_STEP. Steps10–14 mean .7264 steps/s: +6.26% vs matched-health
BAlignedRow .6836, -1.22% vs PerLayerColOnly .7354. Generic health ON/BAM OFF,
checkpoint200; the AOT compiler was released by prepare_train_aot.py.

O-only RUN launch verified at runtime3dc60d3 on UE5a: AOT loaded, step0 and
step14 completed. Steps10–14 mean .703 steps/s, +2.84% vs matched-health
BAlignedRow .6836 (prediction +3%). Generic health ON/BAM OFF, checkpoint200.
Its compiler cleanup finished before FIRST_STEP. Both parameter-removal runs
retain the 13,500-step schedule; no early stopping decision follows from speed.

## Complete the QKV-row × O-row factorial

`BamMediumIndependentLLFBAlignedRowQKVColOnly` removes only LocalQ/K/V row
parameters and contractions. L LocalO and F fetchO remain bilateral, MLP2816
and writes remain unchanged. Disable the unused LocalV aligned-row projection.
Same worktree/branch as above, scan+AOT, 13,500 steps, checkpoint200,
generic health ON/BAM health OFF. Direct comparisons: BAlignedRow and ColOnly.
Prediction vs BAlignedRow: final gap +.002 and speed +3%.

| QKV rows | O rows retained | O rows removed |
|---|---|---|
| retained | BAlignedRow | OColOnly |
| removed | QKVColOnly | ColOnly |

Compare BAlignedRow−QKVColOnly with OColOnly−ColOnly at common steps to test
whether QKV-row value depends on O rows. A positive second difference alone
does not establish that QKV rows are harmful in the intact model. Keep the full
schedule rather than stopping at2800 merely because an ablation worsens loss.
The compact local layout preserves the original column head-mix initializer
slice; parameter shape changes can still alter compiled numerics, so this is
not a claim of bitwise identical whole-training initialization/trajectory.
Validation: `MaxText/tests/bam_col_only_budget_test.py` tests both L/F with
nonzero reads, mapped parameters and finite gradients; audit via
`experiments/bam_llama2_medium/audit_matched_mlp.py`.
Artifacts: `/data0/xd/llf-qkv-col-only-tests.log`,
`/data0/xd/llf-qkv-col-only-audit.json`.
