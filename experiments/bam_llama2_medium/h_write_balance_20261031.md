# H memory-MLP / attention write balance — 2026-10-31

## Question and result

Does removing private memory-MLP content normalization make its actual M writes persistently too weak, and does the effect explain H's different behavior between Medium and Pythia 2.8B?

**At 2.8B 10k the simple persistent-write-deficit explanation is not supported by these magnitude measurements.** H's median layer content RMS ratio before the write gate is .302786, but the matrix-increment ratio after gates, address normalization and summing heads is .789952. The old normalized control at the same step has .798181. These are medians over writing layers, not fractions of total write energy or a causal attribution to loss. The profiles differ: H's last four writing layers are 1.3549/1.3152/1.4321/1.3050, above attention. The late-layer median (layers16–30) is .9508 in H vs .7640 in control. Across all layers weighted by each writer's matrix energy, the ratio is .9831 in H vs1.1292 in control (the latter includes its large first-layer MLP/attention ratio4.2787). Thus one summary cannot substitute for the layer profiles.

This revises the earlier inference from pre-gate content alone: training can compensate through write gates and content growth. It does not show equal write directions, information quality or loss value, nor identify why dynamic-O gamma hurts H. The old control differs in several H switches and has dynamic-O gamma; it is a recipe reference, not an isolated normalization ablation. A useful next causal probe would preserve write directions and replace only layerwise writer amplitudes on the same held-out sequences, rather than infer utility from RMS.

## Exact experiment identities

| Label | Full configuration class | Training runtime | Checkpoint steps | Native plan |
|---|---|---|---|---|
| medium-H-common2048 | `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile` | `d0ed1c80f7543c60f8595a32e058efc677404d22` | 0,13500 | 13500 |
| Pythia28-H | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixed` | `9c11469da97a0ade00530448736e55024bfd6e99` | 0,2000,4000,6000,8000,10000 | 28000 |
| Pythia28-normalized-control | `BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80LocalODynamicNormScale020` | `5c3a995444897bd8a148a2e888a8c0e1d6721c6b` | 0,10000 | 28000 |

All checkpoint roots are in `jobs.json`, under `gs://newproject-1-llm_projects_us-east5/log/<FULL_CLASS>/checkpoints/<STEP>/items`. Step0 reconstructs the exact canonical training initializer and seed via the matching runtime's `max_utils.init_initial_state`; it is not a trained checkpoint. Training checkpointers, optimizer state and formal RUNs were not modified.

Implementation is on `refactor-bam` in `/home/xd/projects/maxtext` (no new experiment worktree). Successful diagnostic runner commit: `53a328517cb50034fec20fba9bb6bff5142d477e`. Each exact training runtime was loaded from a separately verified detached worker worktree. Reusable scripts:

- `experiments/bam_llama2_medium/h_write_balance_probe.py`
- `experiments/bam_llama2_medium/h_write_balance_cohort.py`

Local artifact root: `/data0/xd/bam_diagnostics/h-write-balance-20261031/`.
GCS archive: `gs://newproject-1-llm_base_models_us-central1/diagnostics/h-write-balance-20261031-4d0b199`.
Borrowed, explicitly user-authorized FLEX: `llm-jax-v6e-1-0`, `europe-west4-a`, worker0. The shared worker lock was acquired; the probe processes exited, lock released, FLEX verified ACTIVE and retained. No TPU was created or deleted. No training RUN was launched or changed.

## Cohort and execution

32 fixed-seed Pile validation sequences, seed20261031. Shuffle1024 source records before packing4097-token chunks and select32 without replacement. Source: `gs://newproject-1-common_datasets_europe-west4/pythia_model_test/pile_test/val_with_eos.tfrecord`. Exact packing order, chunk IDs and sequence hashes are in `cohort/cohort.json`; tokens in `cohort/cohort.npz`.

All main comparisons use identical first2048 input tokens per sequence; Medium13500 also runs the same sequences at its native4096. The latter changes formal-write ratios by at most3.67% per layer, without changing the conclusion. The main comparison is not matched in training progress: Medium13500 is100%, 2.8B10000 is35.7%. Medium intermediate checkpoints were not retained, so none are synthesized.

**Batch1, 32 forward calls per checkpoint.** No backward or optimizer execution. Exact runtime fused Pallas writes and read path are preserved. The instrumented forward returns compact per-sequence statistics; no giant activation dump is transferred. Params are explicit JIT arguments, compiled function reused across same-shaped checkpoints.

Kernel overrides used: token tile128 throughout, scoped VMEM14MiB for Medium and40MiB for Pythia N32/V64. Default14 failed the latter's final register/stack allocation. A64-token tile is invalid for this token-minor layout; the runner now validates128 alignment. Historical optional kernel knobs are set after config parsing, and config output dirs are precreated. Post-run runner default VMEM was raised to40 for this v6e probe; exact successful measurements above explicitly used14/40 as recorded in result JSONs.

Reproduction command on the borrowed worker (set EXP/RUNTIME/STEPS/ROOT/VMEM from jobs.json):

```bash
HARDWARE=tpu JAX_PLATFORMS=tpu /home/lishengping/miniconda3/bin/python \
  /tmp/h-write-balance-20261031/source/experiments/bam_llama2_medium/h_write_balance_probe.py \
  --runtime-dir "$RUNTIME" --exp "$EXP" --checkpoint-root "$ROOT" \
  --steps "$STEPS" --cohort /tmp/h-write-balance-20261031/cohort.npz \
  --output "$OUTPUT" --length 2048 --sequences 32 --tile 128 --vmem-mib "$VMEM"
```

## Metrics and validation

Per layer, ratio = `sqrt(mean(MLP mean-square) / mean(attention mean-square))`, pooling the same sequences, tokens, heads and coordinates. It is not the average of tokenwise ratios. Four stages:

1. Content after the actual RMSNorm or fixed divisor, before the write gate.
2. Content multiplied by its actual sigmoid write gate.
3. Separately formed matrix increment after address RMSNorm and head summation, with original factor BF16 rounding and FP32 matrix accumulation.
4. Actual BF16 stored marginal: full original-kernel write minus leave-one-writer original-kernel write. These conditional marginals are not additive; at initialization BF16 rounding can increase the marginal RMS relative to the standalone increment, without implying more useful signal.

Pinned CPU checks passed: factor-statistics check, scan capture, and elementwise identical logits for tiny original vs instrumented model. Original full-write output is returned unchanged; extra leave-one-writer kernel calls only supply health statistics. All11 result sets are finite and preserve the32 cohort hashes; NPZ shape is `[32, writing_layers, 2, 9]`. `artifact-audit.json` stores file hashes. Final layer has no M write and is shaded in the figures. Plot confidence bands are pointwise sequence-bootstrap95% intervals, not a training-seed uncertainty estimate.

## Initialization theory and observed trajectories

Independent .006 Gaussian projections, unit-normalized input, and the actual nonlinearities predict raw private-MLP output RMS .0044203 for Medium SwiGLU (D1200,width1128) and .0530498 for Pythia GELU (D2560,width2944). GELU is approximately x/2 near zero; SwiGLU has an additional small projected factor, approximately xy/2. Actual0k content ratios have medians .0044334 and .0529425, respectively, confirming the scale prediction. This twelvefold initial difference includes dimensions and activation; it is not a clean model-size-only effect.

| Model / step | Content before gate | Gated content | Formal matrix increment | Actual BF16 marginal |
|---|---:|---:|---:|---:|
| 2.8B H / 0 | 0.052942 | 0.052766 | 0.052716 | 0.053718 |
| 2.8B H / 2000 | 0.209008 | 0.673655 | 0.670151 | 0.670260 |
| 2.8B H / 4000 | 0.244576 | 0.793200 | 0.778521 | 0.778574 |
| 2.8B H / 6000 | 0.283035 | 0.806864 | 0.793354 | 0.793394 |
| 2.8B H / 8000 | 0.298010 | 0.791861 | 0.788275 | 0.788304 |
| 2.8B H / 10000 | 0.302786 | 0.779280 | 0.789952 | 0.789971 |
| Medium H / 0 | 0.004433 | 0.004435 | 0.004429 | 0.007085 |
| Medium H / 13500 | 0.222737 | 0.522796 | 0.631294 | 0.631315 |
| 2.8B control / 0 | 0.999820 | 0.998586 | 1.000575 | 1.000580 |
| 2.8B control / 10000 | 1.000003 | 0.750910 | 0.798181 | 0.798189 |

The table contains medians over writing layers. Control pre-gate RMS ratio is near1 by construction; it is not identically1 at every layer because finite epsilon and its fixed content divisor can matter. After gates, even that control is not constrained to ratio1. In trained checkpoints formal increment and actual BF16 marginal almost coincide, so storage rounding does not explain a large trained-stage write deficit.

## Timing / bottleneck

Pythia2k: params-only restore64.67s, first forward including compilation11.33s, all32 forwards13.95s. Pythia10k: restore42.88s, all32 forwards2.71s, total45.60s. Batching32 at once was not attempted; batch1 conservatively bounded probe memory. The original log prints model tracing messages at both0k and2k but not4k/6k/8k/10k:2k did retrace once, rather than every checkpoint recompiling. The original probe did not record parameter PyTree/container/sharding signatures, so the exact cause of the0k→restored transition is not established from those logs. The post-run runner now canonicalizes both paths to ordinary dictionaries and single-device sharding, logs parameter signatures, and counts forward traces. This is a reproducibility fix; completed measurements were not rerun or represented as using the new instrumentation. With a hot executable, checkpoint restoration dominates, so increasing batch would save only a small fraction of total checkpoint time. Preparation/configuration/VMEM debugging is separate overhead and must not be described as forward computation time.

## Figures

- `h-write-balance-profiles-20261031-024124.png` / `.svg`:0k plus checkpoint trajectories, pre-gate and actual matrix write ratios, both scales, old control10k.
- `h-write-balance-gate-bridge-20261031-024124.png` / `.svg`: before gate → gated content → matrix increment → BF16 stored marginal, latest available stages.
- Rendering script: local artifact root `plot_results.py`; summary and source hashes: `figures.json`.


## Medium normalized-control extension

User requested the missing right-column old-normalized references. Two exact historical models were probed at0 and13500, on the same32 T2048 sequences:

| Label | Full configuration class | Training runtime | Diagnostic runtime | VMEM |
|---|---|---|---|---|
| medium-normalized-control | `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile` | `5a3d55b4065e3a6fe49af095762a6a71ea834c7f` | `12014ba95bf66f3efa8c2bdc13ad86052bed9890` | 14MiB |
| medium-normalized-gamma050 | `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile` | `2662b397ff307eb3792563e043c766fb5b1252aa` | `12014ba95bf66f3efa8c2bdc13ad86052bed9890` | 40MiB |

The runner retains compatibility with these older normalized writers by only passing `content_divisors` to leave-one-writer calls when that API exists. Formal forward output remains the original write result. No training or checkpoint changes.

| Model, step13500 | Pre-gate content ratio | Gated content ratio | Formal matrix ratio | Stored BF16 marginal ratio |
|---|---:|---:|---:|---:|
| H | 0.222737 | 0.522796 | 0.631294 | 0.631315 |
| Old normalized | 0.999997 | 0.608859 | 0.743541 | 0.743573 |
| Old normalized + dynamic O .5 | 0.999996 | 0.602827 | 0.703135 | 0.703186 |

These are medians over writing layers, not global energy ratios or loss effects. Medium H's trained write ratio is moderately below both normalized recipes, much less than its pre-gate ratio suggests. The0k normalized pre-gate median is .9745 rather than exactly1 due to finite RMSNorm epsilon on small initial MLP contents; trained values are approximately1.

The plain control succeeded at14MiB. Reusing14MiB for the historical dynamic-O kernel caused scoped VMEM allocation36.08M vs limit34.69M at first forward; rerun with40MiB succeeded. Use the runner's40MiB v6e default rather than infer budget adequacy from a different model/kernel. `medium-controls-jobs-attempt1.json` preserves the first settings, `medium-controls-jobs.json` the successful settings. Initial-source/configuration preparation is not per-checkpoint inference cost.

GCS extension prefix: `gs://newproject-1-llm_base_models_us-central1/diagnostics/h-write-balance-20261031-4d0b199/medium-controls-12014ba`. Local `medium-controls-audit.json` verifies all four finite32-sequence result sets against the original cohort hashes. Plot source and figure paths are updated in local `plot_results.py` / `figures.json`. The same borrowed FLEX was released from the worker lock, retained ACTIVE, and never deleted; there were no other diagnostic resources.

### Why0k to restored weights retraced

The supplementary CPU abstract-parameter comparison on the FLEX loaded no full checkpoint arrays: the initial model and saved metadata have identical parameter-tree structure, shapes and dtypes. It identified one weakly typed initial leaf, `params/decoder/layers/block/self_attention/gw_b0`. Initialization has `weak_type=True`, whereas restoration requests ordinary strongly typed FP32 arrays. This extra JAX signature component explains a possible cache miss despite identical printed shape/dtype. A focused pinned-CPU test confirmed that changing only weak typing triggers a second trace; a BF16-plus-bias example also changed the promoted output dtype from BF16 to FP32. Consequently weak typing is recorded, not forcibly removed: preserving the exact0k numerical semantics takes precedence over avoiding a single compile. Trained checkpoints still reuse their common signature. Evidence: `check-param-signatures-r3.log` and the runner's `PARAM_SIGNATURE`/`FORWARD_TRACE` output; later runner metadata also records weak-type paths.
