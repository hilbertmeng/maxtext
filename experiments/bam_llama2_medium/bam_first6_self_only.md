# First-six-layer self-only causal probe

## Question and intervention

Can layers 0–5 discard cross-token BAM fetch while retaining the existing compressed
self read? For these layers only, scale the **non-diagonal mixed-alpha entries** by
lambda=1, 0.5, or 0; retain diagonal-one. Layers 6–23 retain lambda=1. Both row and
column use the resulting matrix. Recompute the complete downstream network with
normal RMS denominators, unchanged parameters, and dropout disabled.

This is a finite checkpoint intervention, not frozen-final-RMS residual IG and not a
retraining result. Self-only still reads the original K×C compressed M. Reading
uncompressed local M is a separate, untested architecture change.

## Reproduction

Branch `codex/bam-first6-self-only`; runtime commit
`79a182cb5f1f5a23996c71c8a66b621e39839201`, based on the validated residual-attribution
branch at `c02671d`. No main-branch model code was changed.

| Model | Full configuration class | Checkpoint | Trainer commit | Runtime / microbatch |
|---|---|---|---|---|
| Medium | `BamLlama2MediumV2` | `gs://newproject-1-llm_base_models_us-central1/log/BamLlama2MediumV2/checkpoints/13250/items` | `1afd942` | non-scan JIT / 2 |
| XL | `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2` | `gs://newproject-1-llm_projects_europe-west4/log/BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2/checkpoints/49720/items` | `aef0d97411a1725386ebba1aeae1bf4acb1bb79e` | layer-scan JIT / 1 |

Both use the exact same 128 unique Pile-eval sequences, T=2048:

- Cohort: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`
- SHA-256: `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`
- Driver: `experiments/bam_llama2_medium/first6_self_only.py`
- Launcher: `bash experiments/bam_llama2_medium/run_first6_self_only.sh medium` (or `xl`).
- Analysis: `python experiments/bam_llama2_medium/analyze_first6_self_only.py RESULT_DIR HISTORICAL_RESIDUAL_DIR`.
- Semantic tests: `experiments/bam_llama2_medium/first6_self_only_test.py` (pinned CPU environment; both tests pass).
- Model hook: optional `causal_ablation/cross_scale` collection in `attentions.py`; `models.py` slices this collection along layer-scan axis 0. It is not a parameter and never enters the checkpoint.
- Machines: `xd-v6e-self6-medium-ew4a`, `xd-v6e-self6-xl-ew4a`, both spot v6e-1 in `europe-west4-a`.
- Environment: JAX/jaxlib 0.8.1, Flax 0.12.1, NumPy 2.1.3, Optax 0.2.6.

Raw artifacts (each batch retains sample hashes, paired sample/token losses, valid masks):

- `/data0/xd/bam_diagnostics/bam-first6-self-only-medium-79a182c/`
- `/data0/xd/bam_diagnostics/bam-first6-self-only-xl-79a182c/`
- GCS: append those directory basenames to `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/`.
- `summary.json` is the runtime summary; `paired_analysis.json` also verifies historical cohort order and baseline drift.
- Runtime source archive: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/sources/bam-first6-self-only-source.tar.gz`, SHA-256 `0856a3c59f7e13267d657a96802b15f64900c5fc79a9c5ea3662b4c341f5fbf5`.

## Results

Delta is intervention loss minus **this run's lambda=1 loss**, paired per sequence.
The confidence intervals are approximate 95% intervals across 128 paired sequences,
not uncertainty bounds for future retraining.

| Model | Cross retained | Mean delta ±95% CI | Median delta | P95 delta | Samples harmed |
|---|---:|---:|---:|---:|---:|
| Medium | 50% | +.013591 ±.001248 | +.013310 | +.027334 | 127/128 |
| Medium | 0%: self-only | +.141351 ±.008203 | +.134185 | +.230249 | 128/128 |
| XL | 50% | +.019434 ±.001490 | +.018301 | +.035258 | 126/128 |
| XL | 0%: self-only | +.521456 ±.047210 | +.461377 | +.992876 | 128/128 |

Matched baseline mean losses: Medium 2.40079695, XL 2.09222695. Even the least
harmed sample under complete removal worsens by .05950 (Medium) / .11548 (XL).
Complete removal is 10.40× / 26.83× more harmful than half removal, respectively.

### Validation and numerical boundary

- In each model's first batch, lambda=1 and a separately compiled **unmodified**
  forward have exactly equal per-token CE (maximum difference 0).
- CPU tests verify lambda=0 equals local M, lambda=.5 follows the intended linear
  fetch interpolation, and layer scan sees `[0]*6+[1]*18`, not a broadcast zero.
- All 128 sequence hashes and their order match the prior residual-attribution run.
- Against the historical capture-enabled forward, baseline mean drift is +.0000332
  (Medium) / -.0001005 (XL); maximum per-sequence error .0020795 / .0024567.
  Thus historical baseline reproduction is **not bit-exact**. Capture-enabled and
  capture-disabled compiled graphs differ; the precise source of this small drift
  was not isolated. It is far smaller than the measured intervention, and every
  reported delta uses the new matched baseline rather than the historical loss.

## Interpretation

The earlier residual IG assigned layers 0–5 only 1.09% of Medium's total signed
cross contribution and -0.11% of XL's. Their large causal removal penalties reject
using that direct attribution as a pruning-safety score. Early cross fetch has
substantial downstream utility and/or downstream dependence: its contributions
can reappear in later MHA, MLP, and BAM terms rather than in the early residual term.
This probe does not identify which downstream route mediates the dependence.

XL is more sensitive despite having less early direct cross attribution. Complete
removal costs 3.69× the Medium delta, whereas half removal costs 1.43×. The strong
nonlinearity also cautions against extrapolating from small perturbations.

This rejects a low-damage **checkpoint replacement** of the entire first-six-layer
block. It does not prove a from-scratch self-only architecture cannot adapt, nor does
it measure the user's proposed full-M self read. The previous speculative retraining
bet near +.003 is unsupported by this evidence; +.141/.521 are not replacement
forecasts for retraining either. The next discriminating probe would remove each
early layer independently and compare their sum with the joint intervention before
selecting a sparse schedule. No such additional probe or retraining was launched.

## Workflow audit

Actual driver time, including restore/setup, compilation, controls and all samples:
Medium 154.3s (setup 98.4s), XL 45.6s (setup 13.0s). This diagnostic has only forward
passes; it does not repeat the ten-node IG calculation.

The retained older diagnostic nodes were already PREEMPTED; new candidates were
requested. One installation interface mismatch was fixed at its source:
`/home/xd/projects/xd_tpu_scripts/create_standalone_tpu.sh` now accepts a GCS source
archive and passes it directly to the installer rather than treating it as a local
file and SCPing it. Shell validation and real installs succeeded; deployed/local
SHA-256 both `82f5ca08ab56ece718fdb502cc6e8b63ab97222278219d2752fdb166ca5585ea`.
Artifacts were uploaded worker→GCS and pulled GCS→local, with no raw-data relay
through tpu-ag. The reusable driver, analyzer and tests are retained in Git.
