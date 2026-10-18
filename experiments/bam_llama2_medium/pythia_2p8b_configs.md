# Pythia-2.8B / BAM configuration pair

Implementation: main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`.
No RUN, TPU, AOT submission, or training result belongs to these configurations.

## Reference and scope

- [Official training configuration](https://github.com/EleutherAI/pythia/blob/main/models/2.8B/pythia-2.8b.yml)
- [Released model configuration](https://huggingface.co/EleutherAI/pythia-2.8b/blob/main/config.json)
- [NeoX small_init and wang_init](https://github.com/EleutherAI/gpt-neox/blob/v1.0/megatron/model/init_functions.py)

`Pythia2p8B` follows the official architecture, initialization, and scalar optimizer/schedule
settings. The user explicitly chose BF16 instead of the original FP16 with dynamic loss scaling.
This is a MaxText implementation, not a reproduction of NeoX/DeepSpeed execution, checkpoint
cadence, or the official data order. The available pretokenized Pile pipeline is retained.

The reference architecture is 32 layers, D2560, 32 heads of width80, GELU MLP10240,
affine LayerNorm epsilon1e-5, 20 rotary coordinates/head, untied input/output vocabulary50304,
and parallel residual. Both classes use pad_id=-1: official EOS0 stays supervised instead of
being masked as padding by the Pile converter. The optimizer uses LR1.6e-4, Adam betas .9/.95, epsilon1e-8,
WD.1, clipping1, warmup1%, cosine decay to .1 of peak, and143000 optimizer steps.
The global batch is1024 sequences of2048 tokens:2097152 tokens/step.

`target_train_batch_size=1024` resolves per-device batch from the target topology and gradient
accumulation. Changing `gradient_accumulation_steps` changes the microbatch without changing
this global budget. It must divide1024. Set the target to null to use legacy per-device sizing.
This field defaults to null for existing experiments.

## BAM circuit and initialization

`BamPythia2p8BSota` transfers the architecture of XL single-gate read.5 SOTA
(`BamXLPropSharedR400GeneralReadMShiftReadG050TruePile`, historical runtime `eeec67c`),
with read/write ranks reduced from400 to384, not the later read/attention-write sharing variants.
The name denotes the transferred circuit;
no2.8B loss improvement has been measured.

- M80x40; NoPE60 plus standard RoPE20 coordinates/head;32 embedding-write heads.
- Shared GELU R384 inside Q/K/VO generalized reads. Attention-write R384 and MLP-write R384
  remain independent. Every-third-layer MLP writing keeps the XL offset: zero-based1,4,...,28.
  The layer31 MLP write is omitted: final M has no downstream consumer.
- One dynamic v40 M-shift gate, read-only mixing with previous-token M, initially .5.
- Serial residual, GELU, affine LayerNorm. No QKNorm or KV-shift.
- Thirty-two layers:10 scanned three-layer blocks, then two explicit tail layers carrying
  both residual and M. Tail widths are12416 and13254; the last layer absorbs the budget remainder.
- Generic and BAM-specific health statistics are OFF.

The initialization repair follows the Medium serial Pythia experiment (`eaa95f8`):

| Role | Initialization |
|---|---|
| MHA reference token embedding | non-truncated normal, std .0125 = sqrt(2/(5D)) |
| BAM token embedding | non-truncated normal, std .006 |
| BAM random projections, including first M write | original BAM normal, std .006 |
| Standard Q/K/V, MLP input, unembedding | Pythia small_init, std .0125 |
| Standard attention output and MLP output | Pythia wang_init, std 2/(32 sqrt(2560)) |
| Semantic zero kernels, gate biases, learned norm scales | preserve their existing semantic initializers |

Only BAM write RMS epsilon is1e-6; decoder/final LayerNorm stays1e-5 and read-key epsilon
stays1e-4. The initial embedding write has no added input compensation factor.

## Parameter accounting

| Configuration | MLP widths | Parameters |
|---|---|---:|
| Pythia2p8B |10240 every layer|2,775,208,960|
| BamPythia2p8BSota |[12416,12160,12416]; tail widths12416/13254|2,775,209,638|

There are22 non-write and10 write MLPs. R400→384 frees5,304,320 parameters (including the
embedding-write bottleneck), which refill MLP. The first31 layers use128-aligned widths;
`bam_last_mlp_dim=13254` overrides only the last layer, after the regular tail-width override.
Each extra GELU MLP channel costs2D+1=5121 parameters. The nearest integer last width leaves
678 parameters above MHA (.0000244%). Full parameter trees are checked with `jax.eval_shape`; full weights are not allocated.

## Validation

`MaxText/tests/pythia_2p8b_test.py` checks resolved reference settings, topology/accumulation
batch invariance, full parameter accounting, initialization roles and first-write epsilon,
scanned/unscanned forward parity with two tails, and BF16 gradients plus an optimizer update.
Existing serial Pythia/Mudd and XL single-tail checks cover compatibility.

The R384 revision checks full parameter accounting, separate tail widths, BF16 forward/backward
and an update, plus legacy XL tail behavior and config-default lint. No TPU training or AOT
validation was performed.
