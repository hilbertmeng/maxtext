# MediumProp same-AOT repeat

Question: formal MediumProp BAM throughput varies widely while its MHA control is
stable. Separate host/runtime variation from persistent executable overhead before
attributing the full loss to proportional dimensions.

Main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`; runtime
`e2946d70b8da4c954ee360a63dc3a5b3e1b34765`.
Diagnostic-only TPU: `xd-v5p-16-mediumprop-repeat`, us-east5-a, v5p-16.
No training registry entry; remove this exact resource after artifacts are collected.
Formal training resources are unchanged.

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
