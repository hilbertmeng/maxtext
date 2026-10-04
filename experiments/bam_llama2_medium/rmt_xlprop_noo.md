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


## Medium pre-norm/raw-MLP/attention-content-bias transfer (2026-10-05 JST)

Runtime worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`;
RUN `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias`.
UE5a v5p-32 owned TPU `xd-v5p-32-2910064-maxtext`; 50k automatic endpoint,
10k first regular decision, agent reports about every2k with500-step gap windows.
First200/400 windows also check early scale/clipping/carry health.

Transfer the complete promising Medium recipe: QKV static-read zero initialization;
shared normalized embedding content with learnable zero seed; no layer M pre-norm;
MLP summed static/dynamic read vector has learned pre-RMSNorm, static/dynamic MLP
writes share raw output; attention static/dynamic writes share per-head RMS-normalized
content after a new zero-initialized content bias. Embedding, final matrix norm and
Direct40 unembedding remain the verified XL counterparts. No static read biases,
MLP content bias, fetchedO, block scan, Pallas or fused write/read paths.
M60x96,20 heads,C10,proxy20/tail40,write-address R384,RoPE24 stay XL-proportional.

Relative to XL QKV-zero/SeedZero: input norm adds28*1920=53,760 parameters and
attention content bias adds28*20*96=53,760, together107,520=.029167 W_Q (D1920).
Its existing MLP6643 budget was267,560 below MHA. New MLP6644 adds161,280:
final1,432,399,960 versus MHA1,432,398,720 (+1240), nearest integer uniform width.
Focused CPU checks validate this full tree, XL row/head/C ratios, all-layer health,
zero-bias initial-logit/common-parameter parity and finite nonzero bias/norm gradients.

Borrow verified idle retained FLEX_START `llm-jax-v6e-1-0` EW4a for AOT only;
CPU/AOT/new training prequeue run through official local parallel launcher.
TruePile4096 is declared on the class; launcher resolves UE5a-local data.
Inherits `RMTHealthDefaults` first: all carry/dynamic/write/stability statistics,
checkpoint250, permanent checkpoints every2000, latest2 otherwise.

Direct baselines: BAM independent third-layer MLP writes, BAM LLF, Mudd, TruePile MHA.
Track gain/Mudd, gain/LLF and gain/independent as well as direct signed gaps.
Bets at17500: versus BAM independent-.005, LLF-.012421 using its observed-.007421
independent-to-LLF gap; MHA gain .117562 / Mudd gain .080251 =1.465x.
Speed .320step/s; historical independent.347 and LLF.359 are unmatched-health references.
Key failure criterion: renewed late gain decay despite controlled MLP input/output scale.


Startup verified at sealed runtime874fd5112324788673b4b889b476fa61e3aa3577.
Focused CPU two checks passed27.748s; AOT loaded on actual v5p-32 and FIRST_STEP observed.
Worker config and CheckpointManagerOptions independently confirm keep_period2000,
max_to_keep2, checkpoint250, all-layer RMTHealthDefaults and preNorm/raw-write flags.
UE5a-local TruePile path and exact source/AOT commit checked. Initial speed median20-50
.316step/s (single53-step .290 is not the median), versus .320bet and historical independent
.347 (-8.9%, unmatched extra RMT health). All28 carry/input/stability metrics exist and are finite.
Initial raw gradient norm0/20/50 is506.7/7529.3/260.2; clipping coefficient50 is.00384.
Embedding address-bias energy fraction50 .432 versus Medium donor.902. This is successful
startup, not evidence that training health improved; explicitly review200/400 recovery.
