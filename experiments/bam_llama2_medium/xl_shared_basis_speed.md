# XL shared Q/K rank4 basis: matched-health throughput

Implementation: `codex/local-read-gram`, `/data0/xd/local-read-gram`.
Diagnostic runtime: `e05b537dbaa36828f6eb62b39f8a6a28c90f9ef7`.

Target: `BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasisNoHealthProfile`,
derived from `BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis` by disabling
generic training health. All BAM-specific health flags are also false.
The formal run retains generic health ON and is not interrupted.

Historical timing baseline: `BamXLIndependentLLFLocalQKVCFp32AlignedRow`, runtime
`6977fa0`, EW4b v5p-32, all-health-OFF, scan/AOT, 0.5504 steps/s at steps 10–14.
Reuse this measurement; this is not a same-commit pair.

Matched settings: EW4b v5p-32, T2048, per-device batch 16, scan, AOT,
50000-step schedule, `wd_mults=[]`. Target checkpoint period remains 250 in the
compiled configuration; the diagnostic runner disables actual checkpoint writes.

Pre-test prediction: 0.554 steps/s, +0.7% vs baseline; expected range 0 to +1.5%.

## Reproduction

AOT prepared through `prepare_train_aot.py`, state
`tpu-ag:/home/lishengping/xd/projects/aot_runs/e05b537-90ee16bc.json`.
Artifact root:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/e05b537/jax081-i0ae3f58-c17f538a/v5p-32/s50000`.

Target TPU: `xd-v5p-32-qkshared-nohealth-0912`, `europe-west4-b`.
Use `run_profile_matrix.sh`, `PROFILE_STEPS=50000`, `PROFILE_DONE_STEP=33`,
trace steps 10–14. Read both early and trace-free final steps; exclude explicit
profiler disturbances. XPlane travels worker → GCS → local workstation.

## Result

Trace-free steps 28–33: `.549 .549 .549 .549 .549 .548` steps/s;
mean **.548833**, versus historical Rank2 .5504: **−0.285%**, essentially tied.
Steps 4–9 and 16–32 are all .549; steps 11–14 are .548. Profiler startup at
step10 (.493) and export at step15 (.026) are excluded. Final steps37–42 all
.549 corroborate the stable result. The +.7% prediction was optimistic.

This removes the health-setting confound; it does not isolate all intervening
commits. No meaningful Rank2 speed advantage is established. Sharing saves work
relative to independent Rank4, but Rank4 mixing/normalization remains; this is a
plausible cost balance, not an operator-level causal measurement.

Runner manifest on tpu-ag:
`logs/profile-matrix-e05b537-xl-shared-nohealth-20260912T110344Z-876139.tsv`.
Runner log: `logs/profile-xl-shared-nohealth-0912.log`.
GCS: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/e05b537/xl-shared-nohealth/`.
Local artifacts: `/data0/xd/bam_diagnostics/xl-shared-nohealth-0912`.
AOT `Loaded compiled function!` and FIRST_STEP verified; matrix completed with
primary XPlane uploaded directly from worker. Compiler cleanup completed.
