# RMT write contraction and health overhead

Worktree`/data0/xd/rmt-k48-dynamic`, branch`codex/rmt-k48-dynamic`. All paired full-layer arms use one allocated EW4b v5p-16, `xd-v5p-16-rmt-write-lowering-europe-west4-b`, batch/sequence/model/schedule unchanged. Generic training health is ON throughout; RMT-specific health is explicitly varied.

## Conclusions

- Gram health explains most of the combined single-outer regression, but removing it does not create a speed advantage. With eight write stats removed, single .405 versus matched original .409.
- Pure dynamic slowdown nearly disappears when only its eight write stats are removed: .408 versus .409; all read/gate/M health remains. With all RMT health OFF it ties original .413.
- Mul_reduce is slower than dot even with RMT health OFF: pure .395 versus .413 (−4.4%); single .398 versus .410 (−2.9%). Swapping dot operands/output layout also fails. Keep the dot runtime.
- A subsequent equation-preserving row-health/carry-layout/RMS-reuse repair has recovered pure speed to .407 versus the original .405 on the same profile TPU; detailed paired controls and artifact accounting are maintained in [rmt_dynamic_write_speed_repair.md](rmt_dynamic_write_speed_repair.md). No FLOP-share speed ceiling is justified.
- The isolated write microbenchmark speedup did not transfer to the whole step. Same math can lower into different fusion/layout costs; full-step matched controls settle the speed claim.

## Canonical measurements

RMT health ON has41 scalars/layer; write-stat-OFF keeps33 read/gate/M scalars/layer. Health-OFF retains generic training health. Step milliseconds are means over enclosing complete jit-train spans across execution cores; category attribution uses one complete core and reconciles its leaves with wall time.

### 5e627eb: rmt_write_lowering

| Configuration class | step/s | XPlane step ms |
|---|---:|---:|
| `RMTMediumPropK48DynamicFull48RoPE18VectorNorm` | 0.405 | 2438.474 |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite` | 0.401 | 2465.671 |
| `RMTVectorNormDynamicOnlyWriteMulReduceProfile` | 0.391 | 2528.944 |
| `RMTVectorNormDynamicOnlyWriteTransposedDotProfile` | 0.401 | 2466.078 |
| `RMTVectorNormNoExtraHealthProfile` | 0.413 | 2400.487 |
| `RMTVectorNormDynamicOnlyWriteNoExtraHealthProfile` | 0.413 | 2400.206 |

Artifacts: `/data0/xd/bam_diagnostics/rmt-single-outer-write/write_lowering_profile`; GCS `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/5e627eb/rmt_write_lowering`.

### fbba660: rmt_write_health

| Configuration class | step/s | XPlane step ms |
|---|---:|---:|
| `RMTMediumPropK48DynamicFull48RoPE18VectorNorm` | 0.405 | 2438.715 |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite` | 0.390 | 2535.879 |
| `RMTVectorNormNoWriteHealthProfile` | 0.409 | 2419.381 |
| `RMTVectorNormSingleOuterNoWriteHealthProfile` | 0.405 | 2437.755 |

Artifacts: `/data0/xd/bam_diagnostics/rmt-single-outer-write/write_health_profile`; GCS `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/fbba660/rmt_write_health`.

### 9ceb2f3: rmt_dynamic_write_health

| Configuration class | step/s | XPlane step ms |
|---|---:|---:|
| `RMTVectorNormNoWriteHealthProfile` | 0.409 | 2418.863 |
| `RMTVectorNormDynamicOnlyNoWriteHealthProfile` | 0.408 | 2422.765 |
| `RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite` | 0.401 | 2465.819 |

Artifacts: `/data0/xd/bam_diagnostics/rmt-single-outer-write/selective_write_health_profile`; GCS `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/9ceb2f3/rmt_dynamic_write_health`.

### 4854be8: rmt_mul_no_health

| Configuration class | step/s | XPlane step ms |
|---|---:|---:|
| `RMTVectorNormNoExtraHealthProfile` | 0.413 | 2400.075 |
| `RMTVectorNormDynamicOnlyWriteNoExtraHealthProfile` | 0.413 | 2400.179 |
| `RMTVectorNormDynamicOnlyMulReduceNoExtraHealthProfile` | 0.395 | 2510.035 |

Artifacts: `/data0/xd/bam_diagnostics/rmt-single-outer-write/mul_no_health_profile`; GCS `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/4854be8/rmt_mul_no_health`.

### 651d942: rmt_single_mul_no_health

| Configuration class | step/s | XPlane step ms |
|---|---:|---:|
| `RMTVectorNormNoExtraHealthProfile` | 0.413 | 2400.149 |
| `RMTVectorNormSingleOuterNoExtraHealthProfile` | 0.410 | 2415.914 |
| `RMTVectorNormSingleOuterMulReduceNoExtraHealthProfile` | 0.398 | 2490.914 |

Artifacts: `/data0/xd/bam_diagnostics/rmt-single-outer-write/single_mul_no_health_profile`; GCS `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/651d942/rmt_single_mul_no_health`.

## Attribution and validation

Original three-arm matched profile: single +97.25ms; pure +27.21ms. On the first complete core, single convolution−38.52ms, loop fusion+95.80ms, formatting+26.17ms, elementwise+14.24ms. Pure convolution−34.66ms, loop fusion+21.76ms, formatting+21.92ms, elementwise+10.27ms, slice+8.10ms. No new mathematical normalization was introduced.

Single write health materializes FP32 per-token16×16 Gram tensors; the helper costs about72ms directly and also changes fusion/layout. Pure has no write-health Gram: its logging consumers compare dynamic writes to the pre-update residual. QK rank4 normalization Gram remains in both and is not the removed write-statistic Gram.

Earlier microbenchmark health-OFF acceleration was never deployed. Single formal runtime b352efc at1054 retained health and merely replaced its write-health dot-Grams with fused multiply/reduce; this did not deliver the full-step speed gain. Do not conflate it with removing Gram logging.

Pinned CPU gates:17 RMT and47 BAM tests;18 RMT tests after adding selective health. Test-only f914335 fixes miniature geometry resolved from inherited config instead of spelling. Targeted schema/init check covers original, single and pure with33 stats, exactly preserved params/initialization. FP32 values/gradients and BF16 normwise rounding bounds cover contraction alternatives. Sealed full runtime-config checks precede each exact AOT.

Authoritative runner `/home/xd/projects/xd_tpu_scripts/run_profile_matrix.sh`, sealed SHA e66422df6a6e9aa2d676cd0f7c3b9af42519dfe2b95a9d1b3a00a1434be1ab3c. Each matrix uses one sealed source and ready AOT manifests, schedule13500 and trace10–14; collection verifies primary XPlane before stopping the exact arm. Parser `experiments/bam_llama2_medium/analyze_rmt_write_profiles.py` (613e64d) avoids counting scan/while containers twice and validates leaf accounting.

## Resources and formal RUNs

Retained compiler EW4a `llm-jax-v6e-1-0` is STANDARD/guaranteed; `llm-jax-v6e-1-1` is FLEX_START. Actual types and idle workers verified; pinned environments reused, never enrolled in cleanup. Independent AOT groups borrow worker0 through prepare_train_aot_on_worker.py. CPU gates and profile acquisition ran concurrently with compilation.

Owned passive UC1a `xd-v5p-16-rmt-write-lowering-us-central1-a` released after the first verified trace; node/queue absent. All five matrices and19 nonempty primary XPlanes plus19 trace JSONs verified locally. Winner node and queued resource verified absent after artifact collection. Release log `/data0/xd/bam_diagnostics/rmt-single-outer-write/write_lowering_resource_closeout.log`. Both retained compilers preserved. No auto-train on profile TPUs. Large artifacts flow worker→GCS→local, never through tpu-ag.

Remote following scripts: logs/profile_health_after_lowering.sh, profile_selective_health_following.sh, profile_mul_no_health_following.sh, profile_single_mul_no_health_following.sh. Each waits for the previous immutable runner to exit and verifies its trace count plus next AOT readiness.

Single formal RUN b352efc stopped3046 following2800 review; pure formal RUN a82d5fd stopped2869 following2800 review. Neither receives the slower mul_reduce or health-OFF diagnostic runtime. Formal closeout conclusions remain in main exp.py and their experiment documents.
