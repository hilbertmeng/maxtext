# Pythia architecture

`exp_class=PythiaArchitecture` selects the GPT-NeoX parallel-residual circuit.
It is an architecture recipe, not a submitted training experiment or a
parameter-matched Medium/XL configuration. It inherits runtime defaults from
`Common`; data, optimizer and model dimensions remain independently selectable.

Each layer computes

```
x_next = x + Attention(LayerNorm_a(x)) + MLP(LayerNorm_m(x))
```

The two affine LayerNorms are independent (epsilon `1e-5`). The final vector
also passes through affine LayerNorm. Attention is global causal MHA with Q/K/V
and output biases; QKNorm is disabled. The MLP has one exact GELU activation,
input/output biases, and defaults to width `4 * emb_dim`. Input embedding and
unembedding are independent. Dropout is zero.

RoPE affects the first 25% of each Q/K head, with split-half rotation and base
10000. `pythia_rotary_dim=-1` derives that width from
`pythia_rotary_fraction=0.25`; the result must be an exact even integer. For
non-NeoX head sizes such as 75, select an explicit even prefix (for example
`pythia_rotary_dim=18`) rather than silently rounding. Such an override is an
intentional architectural variant.

Initialization follows the original NeoX training recipe rather than the
Hugging Face configuration's generic `initializer_range`: embedding, Q/K/V,
MLP input and vocabulary head use normal standard deviation `sqrt(2/(5D))`;
attention output and MLP output use `2/(L*sqrt(D))`. Biases start at zero and
LayerNorm scales at one. Exact GELU matches the Hugging Face Pythia reference;
this does not claim bitwise identity to historical fused bias-GELU kernels.

## Configuration and budget

Select `base_emb_dim`, `base_num_decoder_layers`, `base_num_query_heads`,
`base_num_kv_heads`, and `head_dim` explicitly when defining an experiment.
Heads must match (MHA), with `num_query_heads * head_dim == emb_dim`.
`base_mlp_dim=-1` derives 4D after dimension scaling; a positive explicit MLP
width is respected, including when matching a parameter budget. Biases and
LayerNorm parameters must be included in that budget. No published Pythia
model size, tokenizer, data order or optimizer schedule is implicitly imported.

`PythiaDecoderLayer` reuses the existing attention backend, scan/rematerialization,
chunked LM loss and checkpoint path. The new partial-RoPE, output-bias and
output-initialization options default to the existing behavior for other
architectures. BAM/Mudd/RMT/deep embeddings and pipeline combinations are not
part of this recipe and incompatible settings are rejected.

## Validation

Run the tests in the project's pinned JAX CPU environment:

```sh
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$PWD" \
  --test-file MaxText/tests/pythia_test.py --jobs 1
.agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh "$PWD"
```

`PYTHIA_REFERENCE_PYTHON` selects a separate interpreter with PyTorch and
Transformers. Locally the reference uses torch 2.9.1 and Transformers
4.40.0.dev0 (`GPTNeoXForCausalLM`, eager attention); no reference packages are
added to the pinned JAX environment. Without that interpreter only the external
reference test is skipped. The reference compares same-weight FP32 logits,
token losses and every parameter gradient, including nonzero biases and
nonuniform LayerNorm scales.

Other checks cover initializer roles, invalid configurations, scan/unscanned
same-weight agreement, causal masking, checkpoint save/restore, BF16 Adam
updates with full rematerialization, fused/unfused projections and cached
autoregressive decoding. The existing 47-test BAM suite is the regression gate.
TPU Splash execution and production AOT compilation still need validation at
the eventual selected model dimensions before launching a training experiment.

## Sources

- [Pythia 410M training configuration](https://github.com/EleutherAI/pythia/blob/main/models/410M/pythia-410m.yml)
- [NeoX small/wang initializers](https://github.com/EleutherAI/gpt-neox/blob/v1.0/megatron/model/init_functions.py)
- [Hugging Face GPT-NeoX reference](https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py)

Implementation worktree: `/data0/xd/pythia-mainline`, branch
`codex/pythia-mainline`, based on main `0f7f6d10`. No RUN or TPU was created for
this migration.
