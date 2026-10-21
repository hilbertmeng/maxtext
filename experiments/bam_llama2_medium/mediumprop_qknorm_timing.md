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
The v6e matrix's two full-remat arms completed, but standard18/minimal exceeded
the v6e memory limit during compilation. This does not establish v5p feasibility.
At the user's direction, the v5p-8 follow-up tests only standard18 QKNorm,
full versus minimal remat. Keep per-device batch16/T4096/BF16 and all health OFF;
global batch64 on four devices. Do not reduce per-device batch or repeat norm-placement timings.
The two v5p-8 profiles use source aa1c98016c044fd2b1e26f35ecfba7c438c04915
and 200-step AOTs; the formal RUN stays at2f6466b/full.
The v6e is reused solely as a CPU AOT compiler for the v5p-8 matrix, then released.
The full75 v5p profiles explicitly inherit the same tuned Splash settings, so
compiled execution does not depend on unrecorded environment overrides.

Timing-only classes do not alter formal source/runtime or automatically switch
the ongoing training run. A formal remat change requires a matching AOT and
checkpoint/cursor-preserving restart after measurement.

V6e artifacts were collected before release:
`/data0/xd/bam_diagnostics/mediumprop-qknorm/timing-20261021/` contains both
primary raw XPlanes/JSONs, all three worker logs (including minimal's compile OOM),
steady-speed JSON and raw-XPlane analysis. Full remat only: standard18 .4278875
step/s versus full75 .4185875 (+2.22%), window20--99/80 samples. Primary device
step means2330.840 versus2384.148ms agree. This supersedes the placement-within1%
prior for this v6e measurement; no v5p placement conclusion is claimed.

V5p AOT root: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/qknorm_timing/aa1c980/v5p-8/s200`;
compile log collected locally as `v5p8-compile.log`. Both sourceaa1c980 AOTs
passed `AOT_MATRIX_OK`, topologyv5p-8/steps200. Owned spot candidates:
`xd-v5p-8-qknorm-remat-20261021-ue5a` (us-east5-a) and
`xd-v5p-8-qknorm-remat-20261021-uc1a` (us-central1-a).
The winner will run both arms; release the loser only after verified trace success.
Orchestration wrapper `/tmp/qknorm-v5p8-profile-20261021.sh` on tpu-ag invokes the
authoritative immutable matrix runner with regional TruePile4096 routing.
