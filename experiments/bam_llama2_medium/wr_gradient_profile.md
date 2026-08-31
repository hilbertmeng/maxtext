# Fetched `W_R` initialization-gradient profile

Full-24 Medium, one identical Pile-eval batch (`b1debb...8387`), step 0. Historical
V1/V2 commits are `03367ac`/`1afd942`. Reproduce with
`medium_v1_v2_gradient_profile.py` and `run_medium_v1_v2_gradient_profile.sh`.

## Baseline

| config | raw grad norm | clip multiplier | `W_R` mean `|grad|` | `W_R` share of grad² | `W_R/Q` | `W_R/V` | `W_R/MLP` |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | 43.120 | .0232 | .004815 | 92.73% | 2593x | 5.74x | 64.0x |
| V2 | 24.323 | .0411 | .002917 | 77.15% | 1571x | 3.47x | 38.7x |

Both baselines have identical initial loss (`10.8443365`) and ordinary MHA
gradients. Layer 0 has zero `W_R` gradient because its incoming M is zero; the
gradient becomes much larger and generally grows with the accumulated M in
later layers.

For `r = 2 sigmoid(g) RMSNorm(W_R x)`, zero `W_R`, epsilon `1e-4`, and gate
opening `.005`, the zero-point Jacobian is `2*.005/sqrt(1e-4) = 1`. This makes
zero-valued `W_R` parameters dominate global clipping.

## Same-batch initialization ablations

| config | V1 loss delta | V1 raw / `W_R` grad² | V2 loss delta | V2 raw / `W_R` grad² |
|---|---:|---:|---:|---:|
| zero `W_R`, gate .005 | 0 | 43.120 / 92.73% | 0 | 24.323 / 77.15% |
| zero `W_R`, gate .0005 | 0 | 12.337 / 11.19% | 0 | 11.819 / 3.23% |
| normal(.006) `W_R`, gate .005 | +.00224 | 15.601 / 2.11% | +.00836 | 12.957 / .80% |
| normal(.006) `W_R`, gate .0005 | +.00526 | 11.650 / .04% | -.00287 | 11.658 / .01% |

The single-batch loss sign of nonzero initialization is not a capability
result. The robust profile result is that either a 10x smaller zero-point
Jacobian or regular `W_R` initialization removes its domination of global
clipping; their training behavior needs paired from-scratch runs.

## Mitigation tradeoffs

- Lower initial gate opening: exact original forward; simplest way to reduce
  the zero-point Jacobian. It retains the zero parameter/gradient-scale
  mismatch and may delay fetched-read learning.
- Regular `W_R` initialization: gives a finite, ordinary parameter scale and
  reduces the RMSNorm Jacobian after projection. It activates a small random
  fetched read at initialization, so loss impact must be trained, not inferred
  from one batch.
- Regular init plus a smaller gate: healthiest initial gradient geometry, but
  combines the two behavioral changes and should follow the single-factor runs.
- Separate `W_R` gradient clipping/scaling: preserves the forward exactly and
  prevents `W_R` from suppressing other groups under global clipping. It is an
  optimizer-specific fix and does not repair the singular zero-parameter
  parameterization, so it is lower priority than the two architectural controls.
- A zero-initialized outer LayerScale with regular `W_R` would preserve an exact
  zero read and initially train only the LayerScale. It is principled but adds a
  staged-learning mechanism; test only if the simpler controls fail.

Artifacts: `diagnostics/medium_v1_wr_init_ablation.json` and
`diagnostics/medium_v2_wr_init_ablation.json`.
