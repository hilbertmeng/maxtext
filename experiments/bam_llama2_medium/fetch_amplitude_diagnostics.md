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

- Runner: `fetch_amplitude_diagnostics.py`
- Runner commit: `24bc4ea`
- Raw reports: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/fetch_amplitude_1400/`
- Local mirror: `/data0/xd/bam_diagnostics/fetch_amplitude_1400/`
