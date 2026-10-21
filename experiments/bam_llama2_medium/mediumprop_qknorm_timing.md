# MediumProp QKNorm remat timing

Completed 2026-10-21. The user narrowed the v5p test to **standard18 QKNorm,
full versus minimal remat**; no full75 QKNorm arm was run on v5p.

## Result

| Configuration | remat | mean step/s,20--99 | all-device XPlane mean step ms |
|---|---|---:|---:|
| BamMediumPropMShiftReadG050StandardQKNormTruePile | full | .5934375 | 1678.044 |
| BamMediumPropMShiftReadG050StandardQKNormMinimalProfile | minimal | .6165000 | 1615.377 |

Minimal is **3.8863% faster** by stable log throughput and **3.8795% faster** by
raw XPlane, within the +3--8% prior. Each stable window has80 samples, ranges
.593--.594 and .616--.617 step/s. Each trace covers five steps10--14 on eight
execution cores (40 enclosing step spans); leaf overlap is negligible and less
than0.4ms of the first core's step is unattributed. Minimal fits at batch16.

Both arms used the same spot v5p-8 VM in us-central1-a, T4096/BF16,
per_device_batch_size16 (matching formal training), four JAX devices/global64,
all generic/internal/BAM health OFF, and tuned Splash1024/compute512/fused
backward. The effective configs differ only in remat and run/artifact identities.
The smaller device count changes global batch from the formal v5p-16's128;
per-device batch was not changed to compensate.

## Reproduction and artifacts

- Implementation: `/data0/xd/mediumprop-qknorm`, branch `codex/mediumprop-qknorm`.
- Sealed profile source: `aa1c98016c044fd2b1e26f35ecfba7c438c04915`.
- Both AOTs: topologyv5p-8/steps200, `AOT_MATRIX_OK` and actual
  `Loaded compiled function!` plus first step verified.
- AOT root: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/qknorm_timing/aa1c980/v5p-8/s200`.
- Matrix: `20261021T072322Z-2459746`; authoritative
  `/home/xd/projects/xd_tpu_scripts/run_profile_matrix.sh`, immutable runner SHA256
  `a27edeb0ae50b4fbb920713743374d32a81a860614ad0f7fcd767423f5452716`.
  Wrapper on tpu-ag: `/tmp/qknorm-v5p8-profile-20261021.sh`; regional data resolved
  by `dataset_paths.py` with TruePile4096 and the sealed source.
- Runner settings: PROFILE_STEPS200, DONE_STEP99, SKIP10, PERIOD1000, DURATION5,
  TRACE_COUNT1, AOT_ROOT as above; no checkpoints. Fixed timing window20--99
  excludes initialization/profiling. Collector verifies GCS traces before moving
  to the next arm and kills only each exact diagnostic train process.
- GCS artifacts: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/aa1c980/qknorm-remat/`.
- Local artifacts: `/data0/xd/bam_diagnostics/mediumprop-qknorm/timing-v5p8-20261021/`:
  both complete worker logs, both primary raw XPlanes/JSONs, `steady-speeds.json`,
  `raw-xplane-analysis.json`, `matrix.tsv` and `matrix.log`.
  Parser: `experiments/bam_llama2_medium/analyze_rmt_pallas_profiles.py`, raw PB
  preferred over potentially truncated trace JSON. Artifacts travel worker→GCS→local.
- Winner `xd-v5p-8-qknorm-remat-20261021-uc1a`/us-central1-a and backup
  `xd-v5p-8-qknorm-remat-20261021-ue5a`/us-east5-a were both verified absent
  (node and queued resource) after required artifact collection. Backup was
  released once the winner's first step and primary trace were verified.

## Earlier v6e probe

The preceding v6e-1/32GB probe completed full-remat standard18 and full75 arms,
but standard18/minimal failed compilation with memory exhaustion. That is not a
v5p feasibility result. Per-device batch16 was preserved when moving to v5p-8.
The user then removed norm-placement timing from the v5p scope.

V6e full-remat pair: standard18 .4278875 versus full75 .4185875 step/s (+2.22%),
window20--99; device step means2330.840 versus2384.148ms agree. This supersedes
the placement-within1% prior only on that v6e measurement. Both used tuned
Splash and health OFF, source6750b25, common smoke wrapper
`experiments/bam_llama2_medium/profile_mediumprop_qknorm.sh`.

All three worker logs (including minimal's failed compile), both primary
XPlanes/JSONs, speed summary and raw trace analysis are retained under
`/data0/xd/bam_diagnostics/mediumprop-qknorm/timing-20261021/`.
The owned `xd-v6e-1-qknorm-pair-20261021-ew4a`/europe-west4-a was reused only
as a CPU compiler for the two v5p AOTs, then verified absent after compile-log
collection. User-retained compilers were not modified or released.

Formal loss training remains runtime2f6466b on
`xd-v5p-16-310260-maxtext`/us-east5-a, full remat/T4096/global128. This timing
probe does not automatically change its runtime or resume/checkpoint state.
