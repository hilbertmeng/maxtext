# K48 P_loc_up shared-address rank diagnostic

Source RUN: `BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer`.
Runtime73f2e77163de5f19d9ead2aa5d80220af4d6b2ee; committed checkpoint13400.
Checkpoint: `gs://newproject-1-llm_projects_us-east5/log/BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer/checkpoints/13400/items`.
Read-only CPU Orbax partial restore of six leaves (three LLF roles' kernels and biases).
Saved kernel shape256x8x16x32, bias16x8x32: block-scan axis1 is moved to the front.
Each layer's kernel256x16x32 unfolds to4096x32, measuring a shared address basis across all heads.
Bias is excluded from compression, matching PLocHeadC8/PLocHeadC8Gelu.
Energy is squared singular value, not singular-value sum. No input cohort: this is a weight-only diagnostic.

Effective writes L0–22 retain47.3994% weight energy in the best shared rank8 approximation
(range40.5989–70.5838%). Achieving90% requires18–27 directions. Independent per-head rank8
retains58.4864% on average, but permits different bases and therefore is not the fifth arm's constraint.
The final F layer23 write has no downstream M consumer (decoder keeps final hidden state only),
so its27.83% energy retention is reported but excluded from effective-write aggregates.

This rejects the explanation that the trained write-address up-projection is already nearly
shared-rank8. It does not determine train-from-scratch results or actual activation-weighted error.
If W_up,n≈T_n A, the old coefficients remain GELU(xW_down)T_n; the fifth arm substitutes xW_n,
and the sixth GELU(xW_n). Low shared output rank alone would not justify those substitutions.

| Layer | Role | Shared rank8 energy | Rank for90% | Rank for95% | Per-head rank8 energy |
|---:|---|---:|---:|---:|---:|
| 0 | local_0 | 55.71% | 23 | 27 | 59.90% |
| 1 | local_1 | 50.65% | 24 | 28 | 63.52% |
| 2 | fetch_2 | 46.14% | 25 | 28 | 54.22% |
| 3 | local_0 | 44.67% | 26 | 29 | 55.51% |
| 4 | local_1 | 48.09% | 25 | 28 | 60.16% |
| 5 | fetch_2 | 46.58% | 25 | 29 | 59.56% |
| 6 | local_0 | 45.39% | 26 | 29 | 59.48% |
| 7 | local_1 | 45.44% | 26 | 29 | 59.75% |
| 8 | fetch_2 | 43.66% | 26 | 29 | 54.52% |
| 9 | local_0 | 47.88% | 25 | 29 | 58.72% |
| 10 | local_1 | 50.67% | 24 | 28 | 64.33% |
| 11 | fetch_2 | 46.46% | 26 | 29 | 58.18% |
| 12 | local_0 | 43.58% | 25 | 28 | 56.77% |
| 13 | local_1 | 40.67% | 26 | 29 | 55.02% |
| 14 | fetch_2 | 41.63% | 27 | 29 | 51.77% |
| 15 | local_0 | 43.78% | 26 | 29 | 59.37% |
| 16 | local_1 | 40.60% | 26 | 29 | 58.04% |
| 17 | fetch_2 | 45.16% | 26 | 29 | 54.80% |
| 18 | local_0 | 43.18% | 26 | 29 | 54.41% |
| 19 | local_1 | 45.18% | 25 | 28 | 55.44% |
| 20 | fetch_2 | 50.09% | 23 | 27 | 57.05% |
| 21 | local_0 | 54.39% | 21 | 25 | 60.84% |
| 22 | local_1 | 70.58% | 18 | 23 | 73.82% |
| 23 | fetch_2 | 27.83% | 29 | 31 | 37.31% |

Reproduction/artifacts: `/data0/xd/bam_diagnostics/k48-ploc-up-rank-s13400/` contains `analyze.py`,
`ploc_up_weights.npz`, `report.json`, and `analysis.log`. Run with pinned
`/data0/xd/conda/envs/maxtext-cpu/bin/python`, `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=`,
`OPENBLAS_NUM_THREADS=1`. Cached NPZ avoids repeated GCS reads; fresh restore uses existing
GCS ADC credentials, explicit single-CPU-device sharding, and partial_restore=True.
No training checkpoint or process was changed; no TPU was allocated for this diagnostic.
