# BAlignedRow O row-key GELU bottlenecks

Implementation: `/data0/xd/llf-o-row-lora`, branch `codex/llf-o-row-lora`,
rooted at `bff38c30` (historical BAlignedRow family, not cleaned main).
Runtime: `6bdfe0d90cec8a1dc4affb2ea58193f031f0fa55`.

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

47/47 common BAM tests passed. The six arm×layer placement/gradient cases passed;
the all-O Local layer also exactly matches baseline initial output and all unrelated
parameter initializations. The new optional-config default was corrected locally
before runtime sealing (the config wrapper returns None for absent properties).

```bash
bash /home/xd/projects/maxtext/.claude/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh /data0/xd/llf-o-row-lora
JAX_PLATFORMS=cpu PYTHONPATH=MaxText:MaxText/tests /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_o_row_lora_test.py
```

Prepare each exact training executable using tpu-ag `prepare_train_aot.py EXP COMMIT v5p-16 13500`;
launch with `run_exp_xd.sh` only after verified AOT readiness. Runtime hashes and actual
speed are recorded in the configuration comments after FIRST_STEP.

## Launch verification (2026-09-14 UTC)

All three loaded their exact v6e-produced executable and passed step14 on UE5a;
step0 loss is identically 10.843424. Parameter counts match the predictions above.
All compiler candidates were deleted (`AOT_CLEANUP_DONE` for each arm).

| Arm (full configuration names above) | TPU | steps10–14 mean | vs .6836 |
|---|---|---:|---:|
| FetchO-only | xd-v5p-16-llf-orow-r256-fetch-maxtext | .6768 | −.99% |
| LocalO-only | xd-v5p-16-llf-orow-r256-local-maxtext | .6766 | −1.02% |
| All-O | xd-v5p-16-llf-orow-r256-all-maxtext | .6698 | −2.02% |

Raw speed samples: Fetch [.679,.667,.680,.679,.679],
Local [.676,.678,.676,.676,.677], All [.673,.660,.671,.673,.672].
The predicted small speedup was not observed; loss/parameter-efficiency remains the question.
Splitting one projection into two dependent GEMMs plus GELU adds launch/intermediate/backward
work despite lower FLOPs; this is a candidate explanation, not a measured kernel attribution.
Generic/BAM health, region, AOT/scan and actual parameter counts have been checked;
no fresh XPlane was requested, so kernel-level causes remain unverified (`!?` in ledger).

Orchestration artifacts on tpu-ag under `/home/lishengping/xd/projects/`:
`aot_runs/6bdfe0d-{6a2d641d,e38f4af4,cc4efed9}.json` (Fetch/Local/All),
`logs/aot-orow-r256-{Fetch,Local,All}.log`,
`logs/launch-orow-r256-{fetch,local,all}.log`, and `run_registry/<full-class>.json`.
AOT GCS prefix:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/6bdfe0d/jax081-i0ae3f58-c17f538a/v5p-16/s13500/`.
Each artifact is `<full-class>.pickle` with its manifest.
Training data resolved to UE5-local
`gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord`.
This task performs startup verification only; continued monitoring follows the user's separate ownership.
