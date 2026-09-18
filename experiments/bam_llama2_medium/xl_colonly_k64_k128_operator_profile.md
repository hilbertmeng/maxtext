# XL ColOnly K64/K128 operator matrix and paired main profile

Status: acquisition / profiling; no measured winner yet.

Training configurations:
- `BamXLSharedBasisQKDirectC8MLPPerLayerColOnly`
- `BamXLSharedBasisQKDirectC8MLPPerLayerColOnlyK128QK96TruncatePartialRoPE`

Training runtime `4fb2021`; profile runtime `3d59d64aa54de26ba3fe3b61cbc4321552fceaa0`.
Implementation branch `codex/xl-directc8-all-col-k128`, worktree
`/data0/xd/xl-directc8-all-col-k128`. Training stays unchanged.

## Matrix

Classes `BamXLK{64,128}OperatorW{M,D}R{M,D}S{M,D}` exhaust all 16 settings.
W is `bam_write_outer_implementation` (`mul_reduce`/`dot`), R is
`bam_read_implementation` (`mul_reduce_btn`/`dot_btn`), S is
`bam_local_second_implementation` (`mul_reduce`/`dot`). R applies to all
M read contractions. Alpha head mixing remains its existing dot setting.

Screening: EW4a v6e-1, six layers / two LLF blocks, per-device batch 1,
T2048, unchanged D2048/H16/head128, C8 and per-role MLP widths.
Generic health ON, BAM sow OFF, full remat, block scan. Trace steps 10–14,
100-step no-checkpoint ceiling; stop after trace collector verification.
`profile_periodically_period=-1`: one trace per arm.

Two EW4a TPUs; each measures K64/K128 WMRMSM as a shared control.
A then measures W=M arms; B then measures W=D arms. Re-pair marginal
winners on one VM. Full suffix classes restore 24 layers and per-device
batch 16 for final v5p-32 AOT confirmation. Include both original controls
and each model's winner in one same-zone final matrix.

## Resources and runners

- A: `xd-v6e-1-xl-k128-ops-europe-west4-a`; initial allocation entered
  maintenance before FIRST_STEP; verified released, replaced, now matrix A running.
- B: `xd-v6e-1-xl-k-ops-b-ew4a`; matrix B running, K64 control trace verified.
- Backup queues: `xd-v6e-1-xl-k128-ops-us-central1-a`,
  `xd-v6e-1-xl-k128-ops-us-east5-a`; both verified deleted after first trace.
- Launcher: `/data0/xd/xl-ops-launch.sh`, copied to
  `tpu-ag:/home/lishengping/xd/projects/logs/xl-ops-launch.sh`.
- Authoritative matrix runner: `/home/xd/projects/xd_tpu_scripts/run_profile_matrix.sh`.
- Orchestration logs: tpu-ag `logs/xl-k-ops-{a,b}.log` and profile manifests.
- Artifact prefix: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/3d59d64/`.
- Planned local artifacts: `/data0/xd/bam_diagnostics/xl-colonly-k-operators/`.

## Validation and analysis

32 six/full-layer profile classes imported and checked for k/layer schedule consistency.
43 BAM attention tests passed after fixing the profile mixin MRO; log
`/data0/xd/xl-k128-operators-tests-fixed.log`.

Compare initial loss trajectories and gradient-related health metrics across implementations;
changes in reduction order need not be bit-identical. Profiles are measurements, not new training runs.
Main tables follow `bam_exp_memo.md`: theoretical W_Q, device step and scoped times,
XPlane FLOPs and bytes, forward vs backward/recompute where distinguishable.
Exclude scan while-parent double counting; report overlapping scopes explicitly.
Reuse `experiments/bam_llama2_medium/analyze_bam_xplane.py`, extending classification
for DirectC8 QK and independent LocalV if necessary. Parse profiles locally only.

## Initial paired controls (provisional, v6e B=1 / 6 layers)

Same B VM, mean of complete kernel-covered device steps in the step10–14 trace; figures are
not full-v5p training timings. Generic health ON, BAM sow OFF, all three
operators mul_reduce. Existing scope classification needs DirectC8/LocalV
fusion review before treating the table as exhaustive additive attribution.

| Configuration | Device step ms | Compiled XLA TF | Compiled XLA GB |
|---|---:|---:|---:|
| BamXLK64OperatorWMRMSM | 61.580 | 6.31662 | 73.973 |
| BamXLK128OperatorWMRMSM | 71.041 | 6.43658 | 82.286 |

K128 step time +15.36%. Initial named-scope deltas: LocalQK +4.50 ms,
write M +2.61 ms, O reads +1.03 ms, temporal fetch approximately unchanged.
These are scope observations, not yet causal attribution of the difference.
Raw artifacts and summary JSON/TXT:
`/data0/xd/bam_diagnostics/xl-colonly-k-operators/b-k{64,128}-mmm*`.
