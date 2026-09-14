# Within-LLF row-key projection sharing

Model: **BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow**, checkpoint **13500**.

## Conclusion

The trained row-key matrices do not support direct cross-layer tying. There is
modest shared input-subspace structure, stronger for O than V, but not enough to
make a high-fidelity shared low-rank parameterization economical. This conclusion
concerns weights at this checkpoint; it is not a causal loss test or evidence that
training with sharing from initialization cannot succeed.

Analyze all eight LLF units, with unit b containing layers (3b,3b+1,3b+2).
V exists in the two L layers only. O uses LocalO in L and fetched O in F.
Layer 0's row-key matrices are exactly zero because its input M is zero; it is
excluded from pooled comparisons. Unit 0 is separately retained below and in JSON.

## Matrix extraction and methods

- Scan axis is dimension 1 of each kernel, with length 8.
- V: each L `W_local_packed/kernel` is [1024,8,612]. Q and K each occupy
  64 basis + 2 gate + 32 mixing columns. V basis is packed columns 196:452,
  reshaped [1024,4,64], with its first 32 coordinates per basis being the row
  key. Thus the studied matrix is **1024×128** per layer.
- O: `W_R/kernel` is [1024,8,16,1,40]. First 32 of each head's 40 coordinates
  generate the row key; the final 8 generate the column key. Studied matrix is
  **1024×512** in each L/F layer. No column projections are mixed into either study.
- Biases, gates, head mixing, read normalization, M contractions, W_O and
  activations are outside the sharing intervention. V biases were restored for
  inspection but remain layer-specific; O projection is bias-free.
- Raw cosine/Pearson compare corresponding matrix entries. Direct tying uses
  the least-squares optimal arithmetic mean, reporting aggregate relative
  Frobenius reconstruction error.
- Input subspaces use left singular vectors. Top-r overlap is
  `||U_a[:,:r]^T U_b[:,:r]||_F²/r`. r=64 is summarized; r16/r32 also saved.
  Eight independently shuffled input-coordinate controls per pair preserve each
  matrix's spectrum and all within-matrix output correlations. Seed 9876.
- Uncentered linear CKA compares `W W^T`; it is saved with its shuffled controls.
  Procrustes cosine permits arbitrary orthogonal output rotation and is only a
  geometric oracle. Rotating K coordinates independently changes interaction with
  M; it is not a legal drop-in substitution.
- Shared input factorization: `W_l ≈ A B_l`, with A [1024,r] shared only within
  one unit, and separate B_l. Truncated SVD of `[W_1 W_2 ...]` is the optimal
  aggregate squared-weight-error solution at rank r. Every layer still applies A
  to its own input; sharing parameters does not imply reuse of one activation.
- Parameter cost includes **both** A and every B_l: `r*(1024+n*d)`, versus
  original `n*1024*d`. Break-even ranks are V-LL **204.8**, O-LL **512**, O-LLF
  **614.4**. Original costs per unit are respectively .25, 1.0, 1.5 W_Q,
  with W_Q=1024². Savings quoted below apply to these projections only.
- Same-budget control uses independent layer SVDs, optimally allocating singular
  components, optionally retaining entire dense matrices. Layers need not receive
  equal budgets. At budget 1 the dense control is exactly lossless.
- Repeat joint SVD after equalizing each nonzero layer's Frobenius norm. Every
  reported 95%-energy rank for units 1–7 is unchanged: the result is not caused by
  one layer's larger matrix norm. Energy is squared singular-value energy;
  95% energy still corresponds to 22.4% relative weight error, not 95% loss fidelity.

## Correlation, units 1–7

All entries describe **BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow**.
Values are means across the seven units; shuffled values additionally average
8 controls. These are geometric comparisons, not independent-sample p-values.

| Path / pair | Matrix cosine | Top-64 input overlap | Shuffled overlap | CKA | Shuffled CKA |
|---|---:|---:|---:|---:|---:|
| V L0→L1 | .00116 | .07944 | .06254 | .12600 | .10380 |
| O L0→L1 | .00385 | .12601 | .06267 | .37812 | .29764 |
| O L1→F2 | .00208 | .12128 | .06265 | .36556 | .29554 |

V matrix cosine ranges -.0128 to +.0159; O across all within-unit pairs ranges
-.0078 to +.0200. Mean-tying errors are ~.70–.71. Consequently an apparently
moderate O CKA does not mean that the matrices can be replaced by their mean.
O overlap weakens in later units: L0→L1 top-64 overlap is .164 at unit1 and .084
at unit7, versus ~.063 shuffled. V falls from .089 to .079 (minimum .070 at unit6).

## Joint rank and parameter economics

Model in every row: **BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow**.

| Unit | Layers | V-LL rank95 | O-LL rank95 | O-LLF rank95 |
|---|---|---:|---:|---:|
| 0, degenerate first layer | 0,1,2 | 64 | 242 | 482 |
| 1 | 3,4,5 | 214 | 580 | 685 |
| 2 | 6,7,8 | 216 | 593 | 699 |
| 3 | 9,10,11 | 216 | 595 | 693 |
| 4 | 12,13,14 | 216 | 600 | 704 |
| 5 | 15,16,17 | 217 | 597 | 706 |
| 6 | 18,19,20 | 219 | 609 | 718 |
| 7 | 21,22,23 | 216 | 613 | 721 |

For units1–7, 95% joint energy costs **1.045–1.069×** original V parameters,
**1.133–1.197×** O-LL, and **1.115–1.174×** O-LLF. Even this tolerance does not
reduce parameter count. Individual V matrices already need rank113–115 of128
for95%; individual O matrices need rank368–394 of512. Unit0 is special:
its V result is simply the nonzero layer1 spectrum, not cross-layer redundancy.

| Sharing scope / rank | Parameters / original | Mean retained energy | Independent same-budget control |
|---|---:|---:|---:|
| V-LL r64 | .3125 | .4635 | .4431 |
| V-LL r128 | .6250 | .7370 | .7280 |
| V-LL r192 | .9375 | .9067 | .9449 |
| O-LL r256 | .5000 | .6661 | .6558 |
| O-LL r384 | .7500 | .8180 | .8281 |
| O-LLF r256 | .4167 | .6013 | .5882 |
| O-LLF r384 | .6250 | .7545 | .7465 |
| O-LLF r512 | .8333 | .8582 | .8880 |

Low-budget sharing buys only ~1–2 percentage points of retained weight energy
relative to independent compression. At milder compression the independent/dense
control is better. There is no strong structural reason to prioritize sharing
these full projections over independent low-rank compression. If a sharing
experiment is still desired, O has more overlap than V, and early units are the
more plausible location; keep a layer-specific residual and compare against an
equal-parameter independent-rank control. Loss benefit, optimal rank and actual
speed must be measured separately. No training was launched by this diagnosis.

## Reproduction and artifacts

- Training/runtime source: `77401da6f83a5aa6ddd61994e028c3c694221518`.
- Diagnostic worktree `/data0/xd/llf-cross-layer-row`, branch `codex/llf-cross-layer-row`,
  created at the exact training commit; model code unchanged.
- Checkpoint: `gs://newproject-1-llm_projects_us-east5/log/BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow/checkpoints/13500/items`.
  Both step and items `commit_success.txt` verified (150/156 bytes).
- Diagnostic runner commit: `72f55275` (training/model code remains `77401da6`).
- Runner: `experiments/bam_llama2_medium/cross_layer_row_parameters.py`;
  figure: `plot_cross_layer_row_parameters.py`. JSON records runner SHA256.
- CPU-only Orbax partial restore of eight leaves, explicitly overriding saved TPU
  sharding with SingleDeviceSharding on CPU; no model/optimizer full restore.
  Existing gcloud ADC is reused without printing credential contents.
- No input cohort, sequence hashes or eval overrides: this is a weight-only study.
  Restored leaf shapes and SHA256 digests are in `restore.json`; all finite.
- Local artifacts: `/data0/xd/bam_diagnostics/llf-cross-layer-row-13500/`:
  `parameters.npz` (~105MiB), `restore.json`, `analysis.json`, logs, `sharing.png/pdf`.
- Initial restore plus analysis88.3s; expanded cached analysis36.6s with BLAS4.
  No TPU acquired, no remote worker, no resource lease or cleanup required.
- Synthetic identity, zero-matrix and full-rank reconstruction checks passed.
  Equal-budget independent/dense control verified to reach exactly1 at full budget.
- Initial CPU restore lacked ADC; runner now locates the already-authenticated
  gcloud account's existing ADC when standard ADC is absent. No credential changes.

```bash
cd /data0/xd/llf-cross-layer-row
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
 /data0/xd/conda/envs/maxtext-cpu/bin/python \
 experiments/bam_llama2_medium/cross_layer_row_parameters.py \
 --output /data0/xd/bam_diagnostics/llf-cross-layer-row-13500
/home/xd/miniconda3/envs/tune/bin/python \
 experiments/bam_llama2_medium/plot_cross_layer_row_parameters.py \
 /data0/xd/bam_diagnostics/llf-cross-layer-row-13500
```
