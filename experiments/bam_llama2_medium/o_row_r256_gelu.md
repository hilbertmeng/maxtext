# BAlignedRow O row-key GELU bottlenecks

Implementation: `/data0/xd/llf-o-row-lora`, branch `codex/llf-o-row-lora`,
rooted at `bff38c30` (historical BAlignedRow family, not cleaned main).

All arms compare directly with
`BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow`.
They keep LocalQ/K rank1 legacy, LocalV rank4 B with aligned row and key scale 1,
all gates, column keys, M compression, P_loc, optimizer/WD and the LLF schedule.
Only the O **row-key projection** changes from `1024→16×32` to
`1024→256→16×32`, GELU between projections. No new bias. Down uses the existing
normal .006 initializer; up is zero initialized like the old read-key projection.
Column keys remain zero-initialized linear `1024→16×8`.
This is projection factorization, not a four-basis dynamic readout.

| Configuration | Changed layers | Parameters saved | Predicted final gap | Predicted throughput |
|---|---|---:|---:|---:|
| BamMediumIndependentLLFBAlignedRowFetchORowR256Gelu | 8 F | 1,048,576 | +.001 | +.2% |
| BamMediumIndependentLLFBAlignedRowLocalORowR256Gelu | 16 L | 2,097,152 | +.001 | +.3% |
| BamMediumIndependentLLFBAlignedRowAllORowR256Gelu | 24 L/F | 3,145,728 | +.002 | +.5% |

Savings per changed layer: `131072 = .125 W_Q`, with `W_Q=1024²`.
Expected total parameter counts: 448802656 / 447754080 / 446705504 respectively.
No M-cache reduction. Down gradients start at zero because up starts at zero;
up receives gradients immediately. This intentionally changes the row-key parameterization.

Training: block-scan + v6e-precompiled AOT, v5p-16, 13500 total steps,
checkpoint every 200, final checkpoint enabled. Generic health ON, BAM sow OFF.
Matched timing reference: `BamMediumIndependentLLFBAlignedRowGenericHealthSpeed`,
runtime `e8aca6b`, UE5a .6836 steps/s (generic ON/BAM OFF).
Trainer primary UE5a, EW4b added after 300s without capacity;
compiler primary EW4a, staged UC1a/UE5a backups.

Validation entrypoints:

```bash
bash /home/xd/projects/maxtext/.claude/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh /data0/xd/llf-o-row-lora
JAX_PLATFORMS=cpu PYTHONPATH=MaxText:MaxText/tests /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_o_row_lora_test.py
```

Prepare each exact training executable using tpu-ag `prepare_train_aot.py EXP COMMIT v5p-16 13500`;
launch with `run_exp_xd.sh` only after verified AOT readiness. Runtime hashes and actual
speed are recorded in the configuration comments after FIRST_STEP.
