# BAlignedRow generic-health timing reference

Purpose: supply a reusable speed baseline for normal training (generic health ON,
BAM-specific sow OFF), replacing comparisons against historical all-health-OFF timing.

- Baseline: `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow`.
- Speed class: `BamMediumIndependentLLFBAlignedRowGenericHealthSpeed`.
- Runtime: `e8aca6b60833b4fb129d545e29d2af566890b5f8`, branch
  `codex/llf-o-row-rank-training`, worktree `/data0/xd/llf-o-row-rank-training`.
- Original baseline implementation: `77401da6`; training/compiler/optimizer/fusion
  files unchanged. New O-row-rank path disabled for this baseline.
- Only resolved configuration change besides name: `record_training_health_metrics=True`.
  BAM health flags OFF; profiler OFF; block-scan ON; total schedule 13500;
  checkpoint period 200. Smoke invocation disables checkpoint I/O only.
- TPU: `xd-v5p-16-baligned-health-0912`, UE5a v5p-16; standalone, outside auto-train.
- AOT compiled on v6e-1, with the exact target topology/schedule.
- Dataset: `gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord`.
- Reused runner: `/data0/xd/local-read-gram/experiments/bam_llama2_medium/run_xl_read_simplification_timing.py`;
  manifest: `baligned_generic_health_speed.json` in this directory.
- Remote result: `tpu-ag:/home/lishengping/xd/projects/logs/baligned-generic-health-speed.json`.

Command on tpu-ag:

```bash
python3 /home/lishengping/xd/projects/run_xl_read_simplification_timing.py \
  --tpu xd-v5p-16-baligned-health-0912 --zone us-east5-a \
  --commit e8aca6b60833b4fb129d545e29d2af566890b5f8 \
  --exp BamMediumIndependentLLFBAlignedRowGenericHealthSpeed --steps 13500 \
  --manifest /home/lishengping/xd/projects/baligned_generic_health_speed.json \
  --output /home/lishengping/xd/projects/logs/baligned-generic-health-speed.json
```

Measured steps10–14: .683/.684/.683/.684/.684, mean **.6836 steps/s**.
Historical all-health-OFF baseline: .6930 steps/s; generic health costs 1.36% throughput.
New O-row rank4 formal run: .6782 steps/s, generic ON/BAM OFF, **-0.79%** versus
this matched-health reference (not the previously incomparable -2.14%).
Local result: `/data0/xd/bam_diagnostics/baligned-generic-health-0912/baligned-generic-health-speed.json`.
