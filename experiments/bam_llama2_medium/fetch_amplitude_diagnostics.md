# Fetched-read amplitude diagnostics

Fixed 32-sequence Pile-eval cohort at step 1,400. All three runs use identical
sequence hashes. Layer 0 is excluded below because it has no historical `M` to
read and its fetched key remains at the zero point.

| RUN | `a` row/col | sigmoid gate row/col | effective coefficient row/col | `||y_bam||/||y_std||` |
|---|---:|---:|---:|---:|
| C8 `a0=.05657` | .06937/.07698 | .5376/.6323 | .01326/.01780 | 1.875 |
| C8 `a0=.025` | .04984/.05758 | .6765/.7194 | .01188/.01514 | 1.404 |
| native C32 `a0=.025` | .05577/.05404 | .7028/.7362 | .00693/.00721 | 1.825 |

`post_gate_key_rms` agrees with the effective coefficient to within 0.04%, so
the RMS direction normalization is behaving as designed. No learned `a` is
negative. The lower-`a0` runs compensate by increasing both `a` and sigmoid
gate opening; thus `a` and the routing gate are not optimization-independent.
C32 keeps roughly half C8's per-coordinate coefficient but recovers almost all
of the high-amplitude C8 readout energy through its wider read. At this point
the late-layer `||y_bam||/||y_std||` means are 3.372, 2.413, and 3.099 in table
order; fetched-read strength remains strongly depth-dependent.

Initial TensorBoard `raw_grad_norm > 1` fractions through step 200 were 85.7%,
71.4%, and 52.4% in table order. The explicit scale therefore improves early
clipping behavior, especially at C32, but the learned scale/gate subsequently
opens to compensate. The loss trajectory is still needed to decide whether
that healthier initialization changes the final optimum rather than only its
arrival time.

## Early-gradient re-audit

For a zero-initialized fetched key projection `z=W_R x`, the zero-point
Jacobian of the RMS-gated key is

\[
J_0=\frac{a\,p}{\sqrt{C\epsilon}},\qquad p=\operatorname{sigmoid}(b_0).
\]

V2 has no explicit `a`, and uses `2*p/sqrt(epsilon)=1` with `p=.005` and
`epsilon=1e-4`. The first amplitude implementation used `p=.5`, so
`a=.05657` also gives `J0=1`; it was an exact initial-strength control, not a
gradient-suppression arm. `a=.025` gives `J0=.442` for C8. The later gate-.005
parameterization reproduces the same values with `a=5.65685` and `a=2.5`.

| early metric | V2-strength control | C8 low-a | C32 low-a |
|---|---:|---:|---:|
| step-0 global raw grad | 4.337 | 3.582 | 3.583 |
| step-0 fetched `W_R` grad L2 | 2.713 | 1.174 | 1.176 |
| `W_R` fraction of raw-grad squared | 39.1% | 10.7% | 10.8% |
| raw grad excluding fetched `W_R` | 3.384 | 3.385 | 3.385 |

Thus low `a` did suppress the initial fetched-`W_R` gradient by the intended
factor and removed most of its large global-gradient contribution. It did
**not** eliminate gradient clipping: the unchanged non-`W_R` gradient alone
was already over the threshold. Nor did it normalize `W_R` per parameter: at
step 0 its gradient RMS was still roughly `543x` the standard `W_Q` gradient
RMS for C8 low-a, versus `1254x` in the V2-strength control.

The suppression is also not durable. `a` and the dynamic gate enter only via
their product in the forward coefficient, but are not optimizer-equivalent.
The original C8/C32 low-a checkpoints increased both learned `a` and gate
opening by step 1,400; their effective coefficients rose above the initial
value. Even fixing `a` leaves the dynamic gate free to compensate. Two C32
arms with the same initial coefficient but `p=.5,a=.025` versus
`p=.005,a=2.5` had nearly identical step-0 `W_R` gradients (`1.176/1.184`),
yet differed by `4.8x` at step 50 (`.030/.145`), directly demonstrating the
amplitude/gate-initialization interaction.

Finally, `1/sqrt(C)` normalizes read amplitude, not the global `W_R` gradient.
At equal low-a settings C8 and C32 had essentially identical total step-0
`W_R` gradient; C32 merely distributed it over more parameters. Adam further
weakens raw-gradient scaling as an update-control mechanism because a constant
gradient scale largely cancels in `m/sqrt(v)`.

- Runner: `fetch_amplitude_diagnostics.py`
- Runner commit: `24bc4ea`
- Raw reports: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/fetch_amplitude_1400/`
- Local mirror: `/data0/xd/bam_diagnostics/fetch_amplitude_1400/`
