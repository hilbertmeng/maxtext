# Whole-M read RMS rescale diagnostic

Artifact: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/mnorm_rescale/20260901T030641Z`

The historical whole-M read normalization is

\[
M_h=M/\sqrt{\operatorname{mean}(M^2)+\epsilon}.
\]

At initialization, the unnormalized matrix RMS rises from `0.4116` at layer 1
to `1.9473` at layer 23. Unit normalization therefore amplifies the earliest
matrix by `2.43x`, while attenuating late matrices. The calibrated diagnostic
uses a shared fixed scale `s=0.411624`:

\[
M_h=sM/\sqrt{\operatorname{mean}(M^2)+\epsilon}.
\]

## Full v5p-16, batch 256, steps 0--30

`W_R` is the RMS across the 24 per-layer `W_R/kernel` gradient L2 norms, matching
the historical TensorBoard diagnostic.

| step | raw grad: unit | raw grad: none | raw grad: calibrated | `W_R`: unit | `W_R`: none | `W_R`: calibrated |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 5.3186 | 6.1008 | 3.7945 | .8347 | 1.0339 | .3436 |
| 10 | 2.5239 | 2.6756 | 2.4725 | .1292 | .2130 | .0729 |
| 20 | 13.7962 | 2.4476 | 2.4404 | .4314 | .0844 | .0190 |
| 30 | 2.2620 | 2.2564 | 2.2893 | .0361 | .0323 | .0168 |

The calibrated scale removes the historical step-20 raw-gradient spike and
reduces every layer's step-20 `W_R` gradient below NoMNorm (`0.12--0.44x` for
layers 1--23). This supports the hypothesis that unit whole-M RMS creates the
spike by amplifying small early-layer matrices, rather than merely by changing
their depth trend.

It is not yet an effectiveness result: calibrated loss is `+.0668` versus
NoMNorm at step 30, consistent with over-suppressing the BAM read while keeping
the scale fixed. The next trainable design is one positive scalar per layer,
initialized to `0.411624`, so normalization removes instantaneous M-amplitude
dependence while training can recover useful layer-specific read strength.

The v6e batch-1 diagnostic establishes the same initialization mechanism but
does not reproduce the full-batch step-20 spike; clipping conclusions must use
the v5p-16 result above.
