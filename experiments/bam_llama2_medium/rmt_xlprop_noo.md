# XLProp TruePile dynamic RMT NoO configuration

Runtime worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`.
Main `refactor-bam` retains the configuration ledger. RUN:
`RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO`.

Inherits `Llama2XLProp` first for the XL backbone/training schedule and
`RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO` for the RMT recipe.
Direct loss control: `Llama2XLPropTruePileMHA`.

| Setting | MediumProp NoO | XLProp NoO |
|---|---:|---:|
| Layers / heads / head dimension | 18 / 16 / 75 | 28 / 20 / 96 |
| Flattened proxy dimension | 1200 | 1920 |
| RMT matrix, address rows x content columns | 48 x 75 | 60 x 96 |
| Proxy rows / dynamic-read rows | first16 / tail32 | first20 / tail40 |
| VO and MLP compressed read state | 75 x 8 | 96 x 10 |
| Dynamic write address GELU bottleneck | 256 | 384 |
| Matrix-derived QK / independently projected RoPE QK | 57 / 18 | 72 / 24 |
| Direct dynamic unembedding key per head | 32 | 40 |
| Uniform MLP width | 4078 | 6622 |

The row/head ratios stay exactly proportional: 48/16 = 60/20 = 3 and
32/8 = 40/10 = 4. Write-address output width scales from16*48=768 to20*60=1200,
so exact proportional scaling gives256*1200/768=400. The selected bottleneck
is R384, the nearest multiple of128. It applies to attention,
MLP and embedding dynamic writes. The matrix aspect ratio is approximately
preserved (48/75=.64;60/96=.625), subject to the integer head/row constraint.

Static QKV and MLP reads use the full matrix; dynamic QK, V and MLP reads use
the tail. QK retains shared rank4, V/MLP use separate C10 compression, and
unembedding reads the tail40 directly without compression. Attention and MLP
keep both static and full-row dynamic writes, including pre-RMS address bias.
Proxy vectors use VectorNorm; the full matrix still has final normalization.
NoO removes the extra dynamic attention O read. All28 layers use direct layer
scan; there is no fetchedO. The inherited pure-JAX path is intended.

Training settings inherit XLProp: TruePile4096, batch8/device, 50,000 steps,
learning rate2e-4, checkpoint250, and the same optimizer/WD rules as XLProp MHA.
The launcher resolves `DATASET_VARIANT=truepile4096` to the training-zone replica.

## Parameter audit

Full-size `jax.eval_shape` parameter-tree audit, with B1/T4 solely for avoiding
activation allocation. No training was run. The audit temporarily generalized
K/C/R in the imported RMT module in memory; the on-disk runtime was not changed.
The same audit reproduced the Medium NoO count431,773,472 exactly.

| Configuration | Total parameters | Difference from XLProp MHA |
|---|---:|---:|
| XLProp MHA, MLP5120 | 1,432,398,720 | 0 |
| XL NoO R384, MLP6621 | 1,432,269,400 | -129,320 |
| XL NoO R384, MLP6622 | 1,432,430,680 | +31,960 (+0.002231%) |

R400 to R384 saves2,845,440 parameters across attention, MLP and embedding writes.
One uniform MLP-width increment costs28*3*1920=161,280 parameters. Thus6622
is the nearest uniform integer width, without hardware rounding.

## Launch plan

UE5a v5p-32, owned TPU `xd-v5p-32-rmtxlprop-noo-maxtext`, 50,000 steps.
Retained FLEX_START `llm-jax-v6e-1-1` in EW4a is borrowed for AOT only, never reclaimed.
Pure-JAX runtime generalizes K/C/R and head-dependent tail padding and health labels.
Focused checks cover full parameter counts, scanned forward/gradient, boundary initialization,
XL health layout, and the existing Medium NoO path. No Pallas paths are enabled.

Bet: terminal RUN-minus-MHA loss -.10; speed .30 step/s (~44% slower than .54 MHA).
Normal generic health and inherited RMT health remain enabled; timing against MHA
must disclose the extra RMT health overhead. XL review at10k, then based on full trend.
