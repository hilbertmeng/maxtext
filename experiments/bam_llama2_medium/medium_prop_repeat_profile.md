# MediumProp same-AOT repeat

Question: formal MediumProp BAM throughput varies widely while its MHA control is
stable. Separate host/runtime variation from persistent executable overhead before
attributing the full loss to proportional dimensions.

Main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`; runtime
`e2946d70b8da4c954ee360a63dc3a5b3e1b34765`.
Diagnostic-only TPU: `xd-v5p-16-mediumprop-repeat`, us-east5-a, v5p-16.
Initially diagnostic-only. After both profiles and the GCS-write repeat, this TPU
was reassigned to the formal BAM RUN to replace its abnormally slow original resource.
Its lifecycle is now owned by auto-train; do NOT separately delete it as a diagnostic.

Arms on one VM:
- `BamMHAMediumPropC256`, basic training health ON.
- `BamLlama2MediumPropK57SharedRank4MLPPerLayer`, basic+concat health ON.

Reuse both formal AOT artifacts under
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/e2946d7/jax081-i0ae3f58-c17f538a/v5p-16/s13500/`.
Preserve13500-step schedule, batch16, sequence4096, optimizer and Pile dataset region.
Override only unique diagnostic output/RUN names, disable checkpoints, enable
XPlane10–14, and stop after99 plus verified trace upload. The original health
asymmetry is intentional here: this tests exact executable reproducibility across
machines, not a health-isolated architecture speed comparison.

Runner: `/data0/xd/run_mediumprop_repeat_profile.sh`, mirrored to tpu-ag.
Worker wrapper: `/data0/xd/run_train_smoke_mediumprop.sh`, selects the formal UE5a data.
GCS trace root:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/e2946d7/mediumprop-repeat/`.

Interpretation: steady normal throughput on the repeat VM implicates the original
host/runtime/data path; reproducing the slowdown permits kernel/layout attribution
from XPlane. Fixed padding overhead alone cannot explain eightfold single-step
variation. Neither outcome alone attributes the remaining gap specifically to health.

## Results and repair

Exact AOT, steps20–99 harmonic means: MHA .727073, BAM .563686 steps/s (-22.47%).
A third BAM repeat retained formal GCS TensorBoard writes and reached .561441
(-.40% vs local-write repeat), excluding logging destination as the cause of the
original resource's .026–.034 steps/s degradation. Early losses matched the original
run to logged precision. Main-host CPU, RAM and disk showed no pressure; TPU warnings
were startup-only. The underlying resource fault is not localized further.

XPlane device-step: MHA1366.329ms, BAM1747.015ms (+27.86%); model FLOPs114.27894/
116.06666TF (+1.56%); bytes1217.468/1683.502GB (+38.28%). Health asymmetry is retained.
LocalQK97.993ms and write93.278ms; these are observed costs, not isolated regressions.
Raw traces, parsed profiles, final logs and speeds.json:
`/data0/xd/bam_diagnostics/mediumprop-repeat/`.

Original formal TPU `xd-v5p-16-mediumprop-k57-maxtext` was gracefully stopped at
committed942, then the SAME RUN/runtime/AOT resumed on `xd-v5p-16-mediumprop-repeat`
at ~.564 steps/s. The replacement was subsequently preempted, but committed1001
before teardown; auto-train resumes from1001. No architecture/parameter change.
Repair orchestration/journal: `/data0/xd/repair_mediumprop_resource.py`, mirrored to
`tpu-ag:/home/lishengping/xd/projects/repair_mediumprop_resource.py`, with journal
`logs/mediumprop-resource-repair.json`. Old-resource release requested after1001
was verified. The formal registry remains authoritative for later resource changes.
