# Matched QKNorm placement and remat timing

Formal loss experiment continues unchanged at runtime2f6466b on
`xd-v5p-16-310260-maxtext`/us-east5-a, T4096/global128, full remat.

This separate timing probe uses one owned spotv6e-1 in europe-west4-a,
`xd-v6e-1-qknorm-pair-20261021-ew4a`, with the same full18-layer MediumProp
model and per-device batch16/T4096 for all four arms. Global batch16 and
unsharded parameters differ from formal v5p-16; report these as paired v6e
timings, not direct formal-v5p throughput measurements.

| Arm | QKNorm | remat |
|---|---|---|
| BamMediumPropMShiftReadG050StandardQKNormTruePile | standard18 | full |
| BamMediumPropMShiftReadG050FullQKNormTruePile | full75 | full |
| BamMediumPropMShiftReadG050StandardQKNormMinimalProfile | standard18 | minimal |
| BamMediumPropMShiftReadG050FullQKNormMinimalProfile | full75 | minimal |

The existing smoke runner disables checkpoints and uses distinct profile output
prefixes, never the formal RUN's checkpoint/log prefix. The wrapper
`experiments/bam_llama2_medium/profile_mediumprop_qknorm.sh` gives both norm
placements the same tuned Splash1024/compute512/fused-backward configuration
and profiler schedule. Generic/internal/BAM health remain OFF in all four arms.
Seed, data, initialization, widths, WD/LR schedule, precision and scan are shared.

Matrix:200-step bound, trace10--14, collect one trace per arm, stop after99.
Use steady20--99 log throughput plus the primary device trace; collector verifies
nonempty GCS traces before moving between arms. All artifacts go worker→GCS→local;
tpu-ag handles orchestration only. Release this exact diagnostic TPU after
verified artifact collection. Do not interrupt the formal training TPU.

Prior: placement alone within1% throughput; minimal +3--8% if it fits memory.
Timing-only classes do not alter formal source/runtime or automatically switch
the ongoing training run. A formal remat change requires a matching AOT and
checkpoint/cursor-preserving restart after measurement.
