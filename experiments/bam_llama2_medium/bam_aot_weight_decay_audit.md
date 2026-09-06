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
