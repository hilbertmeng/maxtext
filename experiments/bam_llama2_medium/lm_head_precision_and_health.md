# LM-head precision controls and direct logit health

Implementation worktree `/home/xd/projects/maxtext`, branch `refactor-bam`.
These are shared runtime/configuration changes, not a new training experiment.
The paused `BamPythia2p8BHUnit300B` RUN is not resumed by this change.

## Configuration contract

| Configuration attribute | Default | Behavior |
|---|---|---|
| `cast_logits_to_fp32` | `True` | Restore the final `.astype(float32)` in `OutputHead.project_logits`, after projection and any tied-head normalization/soft cap. |
| `logits_fp32_output_bf16_backward` | `False` | BF16 projection operands, FP32 accumulator output, custom VJP with BF16 logits cotangent and BF16 backward dot operands. |
| `record_logits_health_metrics` | `False` | Record only the two main-head logit statistics below, independently of generic/internal training health. |
| `logits_health_sample_size` | `128` | Fixed stratified global token positions per microbatch; positive integer. |

For example, an experiment class can opt in without changing its parameter tree:

```python
class MyExperiment(ExistingExperiment):
    cast_logits_to_fp32 = True
    logits_fp32_output_bf16_backward = True
    record_logits_health_metrics = True
    logits_health_sample_size = 128
```

The restored cast is on the common projection path used by chunked CE, direct
logit APIs and MTP. The new accumulator mode works with both independent and
tied embedding heads. It requires `dtype=bfloat16` and rejects simultaneous
`logits_dot_in_fp32=True`, which requests a different precision policy.

| Accumulator mode | Final cast | Logits dtype |
|---|---|---|
| OFF | OFF | Original projection dtype, normally BF16 |
| OFF | ON | BF16 projection output upcast to FP32 |
| ON | Either | FP32 accumulator output; disabling an upcast does not downcast it |

BF16 operands retain their existing precision. The helper respects the global
matmul precision policy. The custom VJP casts the incoming FP32 CE cotangent to
BF16 before both `dH` and `dW` dots. This preserves the original backward dtype
strategy; it is not a claim of fully FP32 backward arithmetic or zero speed cost.
No parameter, optimizer-state or checkpoint-tree change is introduced.

## Restored-default compatibility

`cast_logits_to_fp32=True` was still declared in `base.yml` but was unused in
the refactored output-head path. Restoring it intentionally changes newly
compiled configurations that retain that default. Existing workers/AOTs sealed
at old commits keep their old numerical behavior. To reproduce full BF16 head/CE
behavior on this source, explicitly disable both `cast_logits_to_fp32` and
`logits_fp32_output_bf16_backward`; use fresh AOTs for any runtime/config change.
Historical config dumps containing `cast_logits_to_fp32=True` are insufficient
to establish CE precision without checking their actual runtime source.

Regression tests now exercise both cast states through the actual output head
and its CE backward: a high-confidence target derivative lost by BF16 CE must
survive the restored FP32 CE path. The single common projection implementation
prevents chunked/direct entry points from silently dropping the option again.

## Two health statistics

For each sampled valid token, take the vocabulary mean `c=mean(z)` and centered
standard deviation `s=sqrt(mean((z-c)^2))`, using FP32 statistics even when the
observed logits are BF16. Export:

- `logits/common_offset_abs_p95`: P95 of `abs(c)`.
- `logits/centered_std_median`: median of `s`.

Compute from the existing projection output, with stopped gradients and no
additional head matmul. The optional path runs only for mutable intermediates,
outside initialization, and only on the main head. MTP still receives the
precision fix but contributes no duplicate health samples. Evaluation uses
separate `evaluation/logits/...` names.

Sampling is deterministic, stratified over the logical global batch, and
independent of loss chunk boundaries. Jitter within strata avoids selecting
only sequence starts when batch/sequence lengths are powers of two. Padding is
excluded. These are quantiles of the bounded sample, not exact quantiles of
every training token. No valid sampled tokens yields NaN, making missing
evidence visible rather than reporting a misleading zero.

Pool row summaries across all loss chunks and gradient-accumulation microbatches
before computing quantiles. Never average local/device or microbatch P95s.
The GA reducer preserves these samples while retaining existing sum semantics
for every other auxiliary leaf. Only two scalars leave the compiled train step.
Record them at the existing TensorBoard cadence; no new per-layer/column series
or independent logging timer is added. Keeping the flag OFF removes the sampling
and statistics path entirely.

## Validation

CPU artifacts:
`/data0/xd/bam_diagnostics/logits-precision-implementation-20261103/`.

- `MaxText/tests/logits_test.py`: 11 focused regressions covering projection/CE
  precision, the controlled backward, tied heads, padding, chunk pooling,
  sampling, GA quantiles, MTP/init exclusion, parameter-tree and gradient neutrality.
- `MaxText/tests/logits_sharding_test.py`: two tests on four forced CPU devices,
  partitioning both batch and vocabulary; global quantiles agree with NumPy.
- `MaxText/tests/logits_training_test.py`: two real Pythia Transformer train-step
  tests, ordinary and accumulated. Health exports work with both generic/internal
  health OFF; toggling health leaves loss and updated parameters exactly equal.
- Full BAM CPU suite: 47 tests in four bounded groups, all passed.

The exact pushed runtime `2826e91b69ec208389ad8d585b689fa12a95720a` also passed
all 11 output-head tests and both real train-step tests on the retained
FLEX_START `llm-jax-v6e-1-0`, EW4a, worker 0, with `JAX_PLATFORMS=tpu`.
The compiler was borrowed under its existing AOT worker lock; the lock was
released and the compiler retained. No training RUN was started or resumed.
TPU logs/results are mirrored under the local artifact root's `tpu/` directory;
GCS source: `gs://newproject-1-llm_projects_us-east5/log/diagnostics/xd/logits_precision_implementation_20261103/`.

The separate whole-repository config-default lint still finds two pre-existing
undeclared defaults: `bam_content_adapter_activation` in `bam_content_adapter.py`
and `bam_memory_controls_from_hidden` in `fusion.py`. Both source references and
missing declarations were verified in the unchanged HEAD baseline. They are
outside this patch; the new options are declared and independently validated.

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=MaxText \
  /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/logits_test.py
env JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 PYTHONPATH=MaxText \
  /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/logits_sharding_test.py
env JAX_PLATFORMS=cpu PYTHONPATH=MaxText \
  /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/logits_training_test.py
```

Research motivation and frozen-checkpoint evidence:
[H unit CE/logit-offset diagnostics](pythia2p8_h_ce_precision.md).
