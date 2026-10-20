# Pythia-2.8B / BAM configuration pair

Implementation: main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`.
No formal training RUN has been launched. Earlier timing/alignment runs used the old
M80x40, independent R384 circuit; their measurements do not validate the current revision.

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
with the user-requested M80x64, R1024 read/write projections, read/attention-write down sharing
and QKNorm revision (2026-10-20).
The name denotes the transferred circuit;
no2.8B loss improvement has been measured.

- M80x64; NoPE60 plus standard RoPE20 coordinates/head;32 embedding-write heads.
- Shared GELU R1024 inside Q/K/VO generalized reads and attention writing: one down kernel
  and one activated feature tensor, with independent up projections. MLP-write R1024 remains
  independent; embedding-write bottleneck is also1024.
  Every-third-layer MLP writing keeps the XL offset: zero-based1,4,...,28.
  The layer31 MLP write is omitted: final M has no downstream consumer.
- One dynamic v64 M-shift gate, read-only mixing with previous-token M, initially .5.
- Serial residual, GELU, affine LayerNorm. QKNorm enabled; no KV-shift.
- Thirty-two layers:10 scanned three-layer blocks, then two explicit tail layers carrying
  both residual and M. Tail widths are11008 and11648; the final budget adjustment is128-aligned.
- Generic and BAM-specific health statistics are OFF.

The configuration declares the scan schedule explicitly:

```python
scan_layers = True
scan_layers_unroll = 1
bam_pair_scan = True
bam_local_fetch_block_size = 3
bam_mlp_write_every = 3
bam_mlp_write_offset = 2
bam_final_local_layer_count = 2
```

The32-layer computation is `scan(block_of_3, length=10) -> tail30 -> tail31`.
Each layer has independent parameters; `(residual, M)` carries across blocks and both tails.
Every layer uses local Q/K/V/O reads; `fetch_2` is a legacy module name for the third slot.
The middle slot writes M from its MLP. Tail30 uses MLP11008, tail31 uses MLP11648;
the final MLP write is disabled because no later layer reads its M.
The MHA reference uses an ordinary32-layer scan with parallel Pythia residual.

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
| BamPythia2p8BSota |[11008,10112,11008]; tail widths11008/11648|2,775,039,456|

There are22 non-write and10 write MLPs. The new circuit at the old MLP widths would have
3,039,559,590 parameters, up264,349,952 from the old independent-R384, M80x40 circuit.
This count includes the user's enabled QKNorm. Sharing eliminates all32 independent
attention-write down kernels (32*2560*1024 =83,886,080 parameters versus independent R1024).

An independent MLP write now costs4,802,592 parameters;896 fewer hidden channels in its
MLP refund4,588,416, the nearest128-channel increment. Use21 non-write MLPs of11008,
10 write MLPs of10112, and the last non-write MLP of11648. All32 layers are128-aligned.
`bam_last_mlp_dim=11648` overrides only the last layer, after the regular tail-width override.
The non-MLP subtotal is1,013,661,280 and the MLP subtotal is1,761,378,176.
Each extra GELU MLP channel costs2D+1=5121 parameters. Rounding the near-exact last width11681
to11648 leaves169,504 parameters below MHA (-.00611%, -.0258643 W_Q with W_Q=2560^2).
Full parameter trees are checked with `jax.eval_shape`; full weights are not allocated.

## Validation

`MaxText/tests/pythia_2p8b_test.py` checks resolved reference settings, topology/accumulation
batch invariance, full parameter accounting, initialization roles and first-write epsilon,
scanned/unscanned forward parity with two tails, and BF16 gradients plus an optimizer update.
Shared read/write forward and gradients are compared against independent down projections
whose weights are tied explicitly; shared-down gradients must equal the sum of both branches.
Existing serial Pythia/Mudd and XL single-tail checks cover compatibility.

The R1024 revision also uses the tiny end-to-end Pythia v8t test to compare pure JAX with
the interpreted fused-input Pallas path, including logits and all gradients. No full-size TPU
training, AOT or throughput validation has been performed for this revision.
