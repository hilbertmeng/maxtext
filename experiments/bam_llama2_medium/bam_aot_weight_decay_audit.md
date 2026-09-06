# AOT optimizer-rule omission (2026-09-06)

Runtime: `bef8312`, branch `codex/bam-nonnegative-mix`.
RUN: `BamLlama2MediumV2C256RmsGeluAlphaMix` (v5p-16, UE5a, scan+AOT).

`MaxText/train_compile.py:get_shaped_inputs` constructs `get_optimizer(config,
learning_rate_schedule)` without `wd_tree`. Ordinary
`train.setup_mesh_and_model` constructs `get_wd_tree(config, params_shape)` and
passes it to the optimizer. `adam_pax` falls back to uniform weight decay when
the tree is absent. This is a semantic optimizer difference, not merely a
compiler-rounding difference.

The configured `.*scale$` exclusion matches `fetch_mix_scale`, but the compiled
optimizer never receives it. With zero gradients, initial scale .25, learning
rate .0003, and configured decay .1:

| Optimizer construction | Scale after one update |
|---|---:|
| Ordinary train, actual rule tree | .2500000000 |
| Current AOT, omitted rule tree | .2499925047 |

Reproduction: [audit_aot_weight_decay.py](audit_aot_weight_decay.py), run from
this worktree with `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=MaxText
/data0/xd/conda/envs/maxtext-cpu/bin/python
experiments/bam_llama2_medium/audit_aot_weight_decay.py`.
It uses the actual rule builder and optimizer, rather than a regex-only test.

Observed `bam/fetch_route/layer_000/mix_scale_over_init` at 200/400/600 is
.996048/.990092/.984179. Across all layers, mean ratios are
1.002974/1.026930/1.049928; the scale is learning, but its evolution includes
unintended decay. Do not describe this RUN as no-decay.

The omission also affects other configured exemptions (bias and norm/gate
scales). Historical AOT/JIT comparisons need a runtime-commit audit before
attributing their differences to floating-point compilation. The size and sign
of the resulting **loss** effect are not established by this zero-gradient test.
No running optimizer or AOT executable has been changed as part of this audit.

## Introduction and detection failure

`8b35f19` (2025-02-21) introduced the custom weight-decay tree in ordinary
`train.setup_mesh_and_model`, changing its optimizer construction to pass the
tree. `c0ab7d1` (2025-03-04) subsequently evolved this support. The AOT call still
comes from the original two-argument optimizer API (`0fee3204`, 2024-01-13).
The omission was dormant in the compile-only path until that path was used for
training executables. Adopting XAOT without reviewing both constructors exposed
the latent divergence across formal experiments.

Existing `train_compile_test.py` largely checks successful compilation, not
matching updates. Both cases in `aot_hlo_identical_test.py` are skipped. The
recent mix-scale regression checks only regex membership; it does not exercise
the AOT-created optimizer. This is duplicated setup logic plus missing semantic
integration coverage, not a TPU resource/rounding problem.

## Systematic audit checklist (in progress)

| Contract | Evidence / status |
|---|---|
| Weight decay and parameter exclusions | **Confirmed mismatch**, zero-gradient reproduction above |
| Adam moments, epsilon, gradient clipping, accumulation | Same `train_step`/optimizer implementation; full-update paired test pending |
| Step and learning-rate schedule | Both use `create_learning_rate_schedule`; AOT closure bakes config, counter is live optimizer state; boundary/resume tests pending |
| Parameter initialization and RNG | Actual initialization/restoration remains in ordinary `setup_train_loop`; AOT seed-0 values are shape-only. Both entrypoints set `unsafe_rbg`; environment overrides still require audit |
| Input values, masks and loss normalization | AOT takes runtime batch values, same `train_step`; shape/config and dtype parity tests pending |
| Static config vs runtime config | Compiler uses overrides for topology/output/checkpointing; semantic config manifest and loader rejection tests pending |
| State/serialization contract | Loader reconstructs tree **shapes**, not equivalence of the baked optimizer; serialize/load update test pending |
| Shardings, precision, remat, scan, topology | Shared signature helper; compiler flags and resolved target config still require paired audit |
| Historical exposure | Audit each runtime commit and executable provenance; no blanket claim that all AOT experiments have identical exposure |

Required acceptance: identical params, optimizer state, step, batch, RNG and
semantic config must give matching forward loss, raw/clipped gradients, moments,
parameter deltas and next counters across native JIT and saved/loaded AOT. Include
nonzero decay with excluded leaves, nonzero gradients, warmup/end-of-schedule
steps, and restored nonzero moment state. A successful compilation or matching
first forward loss is insufficient. Preserve old executables and RUN hashes;
do not silently replace a running trajectory while auditing.

## Repair and controlled rerun

Both entrypoints now call `train.create_model_optimizer`; the AOT path uses the
same parameter-shape rule tree as ordinary training. CPU regression
`MaxText/tests/aot_optimizer_contract_test.py` calls the real AOT setup entrypoint
with a small model fixture, then serializes/reloads its compiled optimizer update.
With and without exclusions, parameter values and optimizer states match JIT
exactly at counters 0/1/199/200/201/2800/13499/13500, including nonzero moments.
Zero-gradient excluded scale/bias leaves remain unchanged while the kernel
decays. Both tests pass. This validates the optimizer contract, not full-model
cross-topology equivalence; the remaining checklist stays open.

`BamLlama2MediumV2C256RmsGeluAlphaMixWDFix` inherits the original GELU model and
trains from scratch with the corrected optimizer. Its primary comparator is
`BamLlama2MediumV2C256RmsGeluAlphaMix`. This changes **all configured exclusions**,
not only `fetch_mix_scale`; any loss effect must be interpreted at that scope.
The original RUN retains its `bef8312` executable unchanged.
