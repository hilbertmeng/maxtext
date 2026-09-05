# Row-cross causal sensitivity and signed mixed alpha

2026-09-05. Layer numbers are zero-based. Fixed, identical 128 Pile sequences.

## Main findings

1. **XL layer 11's negative direct attribution is not evidence of a dispensable path.**
   Removing its row-cross read increases full-network loss by **+.01558 ± .00193**
   (95% paired-sample mean CI half-width); all 128 sequences worsen. Retaining .5 or
   1.5 also worsens mean loss. Among the four tested scales, the trained scale 1 is best.
2. **The negative-alpha part accounts for the negative direct attribution, but is
   causally important.** At XL L11 it carries 84.93% of absolute cross-alpha mass,
   contributes −.29105% of final-residual direct IG, and removing it costs +.01672.
   Removing positive alpha has no resolved mean effect (+.00014 ± .00017).
3. **Negative alpha is not synonymous with negative utility.** Medium L8/L9's
   positive direct row-cross attribution also comes predominantly from negative alpha.
   Removing that part costs +.01832 / +.06181. Thus the Medium/XL difference is about
   what the signed read vectors do in context, not a universally bad negative route.
4. The measurements establish a distinction between final-readout direct contribution
   and downstream causal dependence. They do **not yet locate** which subsequent
   attention/MLP/M interactions make XL L11 necessary, nor prove a particular
   cross-token mediation circuit. No retraining or pruning recommendation follows.

## Models and reproduction

| Label | Full configuration class | Checkpoint | Trainer commit | Diagnostic TPU |
|---|---|---:|---|---|
| Medium | `BamLlama2MediumV2` | 13250 | `1afd942` | `xd-v6e-rowcross-xl-ew4a`, europe-west4-a |
| XL | `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2` | 49720 | `aef0d97411a1725386ebba1aeae1bf4acb1bb79e` | `xd-v6e-rowcross-med-uc1a`, us-central1-a |

The TPU names reflect acquisition slots, not the model finally assigned to them.
Both are standalone spot v6e-1. XL's node was preempted **after all artifacts were
uploaded and retrieved**. Medium's node is retained for follow-up; neither is managed
by auto-train. Checkpoints were restored read-only and no training took place.

- Branch: `codex/bam-row-cross-sign`; runtime commit
  `edbb6b77d6d45d6023f1b84d83d55c08aa57d854`.
- Runner: [row_cross_sign.py](row_cross_sign.py);
  [launch script](run_row_cross_sign.sh);
  [host analysis](analyze_row_cross_sign.py);
  [targeted tests](row_cross_sign_test.py).
- Instrumentation is isolated in this branch's `MaxText/layers/attentions.py` and
  `MaxText/layers/models.py`, not merged into production.
- Run `bash experiments/bam_llama2_medium/run_row_cross_sign.sh medium` or `xl`
  from the runtime commit in the standard JAX 0.8.1 / Flax 0.12.1 TPU environment.
- Medium checkpoint:
  `gs://newproject-1-llm_base_models_us-central1/log/BamLlama2MediumV2/checkpoints/13250/items`.
- XL checkpoint:
  `gs://newproject-1-llm_projects_europe-west4/log/BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2/checkpoints/49720/items`.
- Cohort: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`;
  SHA256 `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`.
  T=2048, seed=9876, 128 unique sequences; Medium microbatch 2, XL microbatch 1.

Raw results are under `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/`
and `/data0/xd/bam_diagnostics/`, in directories
`bam-row-cross-sign-medium-edbb6b7` and `bam-row-cross-sign-xl-edbb6b7`.
Each contains `summary.json`, `baseline_check.npz`, cohort, per-batch token/sequence
loss and per-sequence/layer sign/energy/IG statistics. `analysis.json` is computed locally.
There are 64+64 Medium and 128+128 XL loss/metric batch files, covering all 128 samples
exactly once. No residual-width vectors are retained.

## Intervention and attribution semantics

At the chosen layer, retain self-read and the entire col read. Split only off-diagonal
mixed alpha into `alpha_positive=max(alpha,0)` and `alpha_negative=min(alpha,0)`.
Fetch/read their matrices with the same trained row keys and gates. Map through the
same per-head W_O blocks for direct residual attribution.

- Uniform scale: `row_self + lambda * (row_total - row_self)`, lambda=0/.5/1/1.5.
- Signed intervention: `row_total + (p-1)*row_positive + (n-1)*row_negative`,
  `(p,n)=(0,1)` or `(1,0)`.
- Preserve the `(1,1)` local endpoint bit-for-bit; all downstream layers recompute.
  Parameter values, tokens, self-read and col-side settings are unchanged.
- Float arithmetic leaves a small decomposition closure residual in the signed arms;
  it is retained rather than silently assigned to either sign. Mean residual norm /
  total-row norm is .25–.31% Medium, .35–.60% XL across L6–11.

`V` below is the prior **direct final-residual** IG: ten Gauss–Legendre nodes on
`h(alpha)=alpha*h_final`, freezing final RMS denominator at its unscaled value.
Positive V denotes loss reduction. Per-sequence contributions are normalized by that
sequence's total contribution, then averaged. It is not full-network path IG.
Energy E is mean token `||z||/||h_final||`, then mean over sequences, not squared energy.
Row means K/data-key read producing the V/address-side result; col is the opposite.

## Causal loss: each model against its own unmodified same-graph baseline

Positive delta means intervention harms loss. Values are sequence-mean paired deltas.

| Model, layer | retain 0 | retain .5 | retain 1 | retain 1.5 | remove positive alpha | remove negative alpha |
|---|---:|---:|---:|---:|---:|---:|
| XL, 11 | +.015583 | +.004297 | 0 | +.002794 | +.000136 | +.016719 |
| XL, 6 | +.025691 | +.003390 | 0 | +.000926 | +.004055 | +.037471 |
| XL, 10 | +.000952 | +.000205 | 0 | +.000325 | +.000546 | +.001863 |
| Medium, 8 | +.018136 | +.003800 | 0 | +.003898 | +.000005 | +.018324 |
| Medium, 9 | +.060938 | +.009507 | 0 | +.004614 | +.000651 | +.061808 |
| Medium, 11 | +.000763 | +.000098 | 0 | +.000185 | +.000605 | +.000155 |
| Medium, 6 | +.024623 | +.006900 | 0 | +.003878 | +.000071 | +.025007 |

XL L11: zero/.5/1.5/negative-removal harm 128/114/110/128 of 128 sequences.
Medium L9: zero and negative-removal harm all 128.
Full per-arm CIs, medians and counts are retained in `analysis.json`.
Sub-milliloss effects, especially positive-alpha removal, warrant caution given bf16
and multiple exploratory comparisons; the large effects do not depend on such claims.

## Sign decomposition across layers 6–11

Count is negative entries / valid off-diagonal entries, including exact zeros in the
denominator. Mass is `sum(abs(alpha_negative))/sum(abs(alpha_cross))`.
V values are **percent of total direct contribution**, not loss deltas.

| Model | Layer | Negative count % | Negative mass % | V self % | V positive-cross % | V negative-cross % | V cross total % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medium | 6 | 30.03 | 81.83 | −.29251 | −.00948 | +.20426 | +.19478 |
| Medium | 7 | 81.51 | 69.93 | +.03609 | +.00893 | +.00325 | +.01218 |
| Medium | 8 | 59.81 | 85.59 | −.39546 | +.00194 | +.24760 | +.24955 |
| Medium | 9 | 48.73 | 60.89 | −.31660 | +.00559 | +.12929 | +.13489 |
| Medium | 10 | 68.94 | 30.94 | +.19645 | +.06313 | −.00405 | +.05908 |
| Medium | 11 | 61.24 | 26.61 | +.19388 | +.07054 | −.00156 | +.06898 |
| XL | 6 | 93.88 | 81.95 | −.23463 | −.04017 | +.20364 | +.16348 |
| XL | 7 | 61.41 | 46.94 | +.06656 | +.01253 | −.02174 | −.00920 |
| XL | 8 | 80.02 | 33.92 | +.09116 | +.02939 | −.02625 | +.00314 |
| XL | 9 | 77.47 | 38.49 | +.01501 | +.00575 | −.00030 | +.00544 |
| XL | 10 | 81.86 | 80.20 | −.39369 | −.04120 | +.11808 | +.07688 |
| XL | 11 | 44.79 | 84.93 | +.62219 | +.02426 | −.29105 | −.26680 |

Cross total is measured as total-row V minus self V, not by assuming exact bf16 closure.

Three concrete distinctions:

- **XL L11:** fewer negative entries than positive, yet negative entries carry most
  mass and energy. Positive/negative cross E=.00153/.01727; self E=.02767.
  Their residual-space cosine averages −.602. The negative part opposes positive
  final-readout contribution but remains essential to the network's computation.
- **Medium L8/L9:** negative-cross E=.01780/.01866 and positive direct V=.24760%/.12929%.
  These are not absent negative routes; their output aligns differently with the loss
  direction. Medium L8's positive/negative vector cosine is +.043, illustrating that
  opposite coefficient signs need not even produce opposing residual vectors.
- **XL L8:** cross total V is near zero because +.02939% and −.02625% cancel.
  **XL L9:** both signed contributions are small. These are distinct reasons for a
  small net attribution and should not be conflated with XL L11's strong negative term.

As a descriptive sample-level check, XL L11 negative-part direct V correlates with its
removal loss at r=−.591: more negative direct attribution accompanies *greater* causal
harm on removal. Medium L8 has r=+.789, L9 +.408. Negative mass fraction alone is a poor
universal predictor (XL L11 r=−.249, Medium L8 +.024, L9 +.652). These are within-cohort
associations, not causal effect decompositions.

## Validation, numerical limits, and timing

- All-one local helper is bit-exact in bf16; signed/uniform controls and scanned vs
  non-scanned layer addressing pass the two targeted CPU tests.
- Standard BAM suite: 34/35 pass. The remaining pre-existing MLP-write shape test
  accesses `.shape` on a `LogicallyPartitioned` wrapper; reproduced unchanged in the
  pre-diagnostic worktree. It is not a new causal-probe failure.
- Instrumented vs separately compiled original first-batch mean loss drift:
  Medium −.000238, XL +.000607. Max token difference .125 in both; token RMS error
  .02688/.03746. Original .05 max-token threshold rejected XL; the revised runner
  retains raw token errors and checks aggregate loss drift <.001, since bf16 token
  quantization itself can exceed .05. This is a tolerance change, **not a claim of
  bit-exact full-network equivalence**.
- Across all 128, baseline vs historical saved losses: Medium +.0000171 ± .0001066,
  XL −.0000570 ± .0001493. Captured-statistics forward vs intervention baseline mean
  drift is −.00000305/−.00000100. Historical per-layer total-row normalized IG mean
  drift is at most 4.2e−7/2.9e−7, preserving the observed attribution pattern.
- Restore/setup + compilation + all arms/IG: Medium 245.1s (setup 97.0s), XL 111.3s
  (setup 26.2s). Resource installation/recovery and source transfer dominated elapsed
  workflow time. Standalone launchers now reject missing source archive/commit before
  acquisition; the skill example passes both. Source upload used a non-composite GCS
  object, compatible with the fresh worker's gsutil integrity check. Artifacts traveled
  TPU → GCS → workstation, not through tpu-ag.

## What this resolves, and next discriminating question

The XL L11 anomaly is reproducible and is largely a **negative-alpha-derived negative
direct term with positive full-network necessity**. Medium supplies a concrete opposite
case: negative-alpha-derived positive direct terms with positive necessity.

To locate the mechanism rather than rename it, a next experiment could selectively
restore downstream MHA/MLP outputs or M states after the L11 intervention and measure
which restoration rescues loss. This would distinguish same-token representation
correction from later cross-token mediation. The current data do not identify that
mediator and do not settle whether C8→C32 behavior originates here.
