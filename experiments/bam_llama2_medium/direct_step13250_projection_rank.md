# Direct step-13250 projection-rank diagnostic

Checkpoint: `BamLlama2MediumV1CompressAbsV8Direct/13250`. Four shuffled Pile-eval batches
(32 sequences each) were split 64/64 for fitting and held-out evaluation; token activations
were sampled every 32 positions. Projection outputs are flattened jointly over all 16 heads.
Layer 0 `W_R` is exactly zero because it reads the initially empty M stream, so `W_R`
summaries below use layers 1–23.

## P_loc: D -> 16x32

Median over 24 layers (brackets are layer min/max):

| spectrum | r95 | energy@64 | energy@128 | energy@256 |
|---|---:|---:|---:|---:|
| kernel `[1024,512]` | 344 [217,398] | .494 [.292,.637] | .690 [.506,.876] | .884 [.790,.965] |
| centered output activation | 207 [133,343] | .807 [.497,.869] | .897 [.686,.946] | .970 [.884,.990] |

Held-out activation-PCA affine reconstruction after the actual per-head write RMS:

| rank | median relative RMS error | median cosine | same-batch held-out dloss |
|---:|---:|---:|---:|
| 64 | .397 | .921 | +.04093 |
| 128 | .266 | .964 | +.01342 (64/64 sequences worse) |
| 256 | .147 | .989 | +.00286 (57/64 worse) |

Weight-SVD rank-128 is worse than activation-PCA rank-128: held-out dloss `+.01731`.
Thus the proposed `D -> D/8 -> nv` linear bottleneck is not supported as a lossless
factorization of this checkpoint. From-scratch adaptation and the nonlinear arm may still
change the result, but rank 128 is a real capacity ablation rather than redundant parameters.
Layer 23 is the hardest layer (activation r95=343, rank-128 energy=.686, post-RMS error=.631).

## Fetch W_R, row and column separated

| kernel | output dim | median r95 | energy at 1/4 dim | energy at 1/2 dim |
|---|---:|---:|---:|---:|
| row key | 512 | 384 [255,392] | rank128=.557 [.529,.807] | rank256=.820 [.804,.951] |
| column key | 128 | 105 [94,108] | rank32=.596 [.515,.741] | rank64=.798 [.769,.875] |

Centered held-out runtime-key activations:

| stage | row r95/512 | row energy in fitted rank128 | column r95/128 | column energy in fitted rank32 |
|---|---:|---:|---:|---:|
| pre-RMS | 315 | .696 | 75 | .842 |
| post-RMS, pre-gate | 324 | .687 | 77 | .797 |
| post-gate | 328 | .634 | 71 | .837 |

The kernels themselves are broadly full-rank on both sides. Runtime column keys are more
compressible than runtime row keys, but neither supports a symmetric 4x bottleneck as a
near-lossless default: post-gate rank128 retains only .634 of row-key energy, whereas rank32
retains .837 for column keys. Any W_R factorization should therefore be asymmetric and tested
with paired loss ablations rather than inferred from kernel rank alone.

Raw report: `/data0/xd/bam_diagnostics/bam_projection_rank_direct13250/projection_rank.json`
(SHA-256 `504ad3817d4fddbc08bbe696c32e1490965ab476f15f28f4d55fae227fa06b18`).
