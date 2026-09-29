# Dynamic RMT XL startup: cross-scale normalization and backward-path diagnosis

Worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`.
Diagnostic runtime `6c081ce6ed6c8d1749ff5fdfa8c56aec4fdc4aa0`.
Borrowed hosts: EW4a `llm-jax-v6e-1-1` (FLEX_START; XL28) and EW4a `llm-jax-v6e-1-0`
(STANDARD, user-reserved; Medium18, Medium28, XL18), each verified idle before use.
Both remain user-owned; this task has no cleanup/lifecycle ownership of them.
The newly requested UE5a `xd-v5p-32-rmtnorm-diag` was cancelled and both resources
verified absent. All model changes remain in the diagnostic worktree, not main.

## Question and controls

Medium's successful VectorNorm training is the essential counterexample: removal
of full-M pre-norm cannot by itself explain why XL fails. Compare the full shapes
at identical fixed Pile sequences, independent of warmup and optimizer schedules.
Native initialization cross: Medium18, Medium28, XL18, XL28. Changing depth also
changes the intended static-write initialization `1/sqrt(2L)`; native-depth results
must not be presented as a fixed-weight truncation experiment.

Model families:
- `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO`: D1200,H16,M48x75,R256,C8.
- `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO`: D1920,H20,M60x96,R384,C10.

At native Medium18 and XL28, compare current VectorNorm, original matrix pre-norm,
and both norms. Also stop layer-write address or content gradients (or both),
without changing forward outputs or embedding write gradients. Gate gradients
remain present even in `stop_both`. These are causal backward-path interventions,
not proposed trainable architectures. Common parameters and data are held exactly
fixed between interventions; extra matrix gains are identity-initialized.

`MaxText/rmt_norm_probe.py` uses only_eval, no restore/update/checkpoint save, four
B1/T4096 Pile batches, seed from the training class, scalar-only custom-VJP taps.
It records M before attention, after attention, after MLP, normalized read inputs,
and raw/normalized dynamic-write addresses/content. Gradient norms use RMS as well
as L2, avoiding misleading comparisons due solely to different parameter counts.
Small CPU checks verified finite outputs/gradients and identical forward losses
for all three stop-gradient interventions. No full generic test suite was rerun.

## Predictions before inspecting full-size probes

Assumptions: independent zero-mean initializer weights with sigma=.006; small-input
GELU(z) approximately z/2; embedding RMS=.006; gates approximately.1; epsilon=1e-6.
Let H=heads,D=flattened proxy width,R=address hidden width,L=depth.
For embedding address and content:

- raw address RMS `sigma_e * sigma_w^2 * sqrt(DR)/2`;
- raw content RMS `sigma_e * sigma_w * sqrt(D)`;
- normalized RMS `s/sqrt(s^2+epsilon)`;
- dynamic seed RMS approximately `.1*sqrt(H)*address_norm_rms*content_norm_rms`;
- combine independent static seed (RMS .006) in quadrature.

| Quantity | Medium | XL |
|---|---:|---:|
| Predicted raw embedding address RMS | .00005986 | .00009273 |
| Predicted raw embedding content RMS | .0012471 | .0015774 |
| Predicted seed M RMS | .0195881 | .0353894 |
| Historical measured seed M RMS | .0194431 | .0353172 |

The initial scale difference is quantitatively predicted, not evidence of a coding
bug. At fixed isotropic M-cotangent RMS, embedding geometry/RMS derivatives predict
only about1.527x bias gradient L2 growth; the historical bias norm ratio is31.423x.
Thus the bulk requires different upstream M gradients and/or directional coherence.
Historical measurements alone do not hold input cohorts fixed; the new probe does.

For approximately independent, fully normalized layer writes, dynamic M increments
have variance approximately H*g^2; unlike static writes they lack 1/sqrt(2L)
scaling. After two writes per layer, M RMS is approximately g*sqrt(2LH):2.40 vs3.35.
At the last attention input (2L-2 writes), prediction2.3324 vs3.2863; historical
measurement2.4288 vs3.6584. Correlations, gates, static writes and imperfect RMS
normalization account for deviations from this deliberately simple model.

With VectorNorm, scaling M by c approximately leaves proxy-generated keys/gates
and independent RoPE projections unchanged, while matrix-derived QK scales by c.
Its contribution to attention logits therefore scales by c^2. This is a concrete
cross-depth and cross-scale imbalance; it is not yet proof of the gradient root cause.

Small-signal SwiGLU MLP output RMS is approximately
`(D*sqrt(F)*sigma_w^3/2) * input_RMS^2`: coefficients .008276 (Medium) and .016874 (XL).
The subsequent per-head content RMSNorm has tangent gain up to
`1/sqrt(content_RMS^2+epsilon)`, so its operating regime should be checked explicitly.

## Results

Same four B1/T4096 Pile sequences (identical shard hashes across arms), native
initialization, seed 9876. Embedding address pre-RMS bias gradient L2, mean of 4:

| Case | Medium18 | Medium28 | XL18 | XL28 |
|---|---:|---:|---:|---:|
| VectorNorm (trained config) | 6,688 | 73,310 | 13,720 | 258,500 |
| Matrix pre-norm | 1,667 | 2,248 | 1,313 | 1,752 |
| Both norms | 1,686 | 2,259 | 1,314 | 1,738 |
| VectorNorm, stop layer-write address grad | 1,099 | 11,000 | 2,786 | 44,020 |
| VectorNorm, stop layer-write content grad | 124.8 | 184.7 | 139.6 | 279.7 |
| VectorNorm, stop both | 4.41 | 4.99 | 6.38 | 9.65 |

Backward M cotangent RMS, last-layer output to layer-0 output (VectorNorm / matrix):

| Arm | VectorNorm growth | per layer | Matrix growth | per layer |
|---|---:|---:|---:|---:|
| Medium18 | 265 | 1.388 | 30.3 | 1.222 |
| Medium28 | 3,790 | 1.357 | 55.3 | 1.160 |
| XL18 | 590 | 1.455 | 29.0 | 1.219 |
| XL28 | 14,700 | 1.427 | 52.2 | 1.158 |

- With VectorNorm the per-layer backward gain stays ~1.4 at every depth, so the
  embedding cotangent grows exponentially in L. Matrix pre-norm makes the per-layer
  gain decay with depth and leaves the bias gradient nearly depth/width invariant.
- Depth is the dominant cross-scale factor: Medium18→Medium28 is 11.0x, Medium18→XL18
  is 2.05x. XL28/Medium18 = 38.7x, consistent with the 31.4x training step-0 raw-norm ratio
  (Medium NoO 701.6 vs XL NoO 22,061.8).
- The amplified path is the layer dynamic-write content backward (stop-content: ~900x
  reduction at XL28); the address path is secondary. The final factor is the embedding
  address RMSNorm in its epsilon regime (raw RMS 9.2e-5 vs sqrt(eps)=1e-3), giving the
  pre-RMS bias ~1000x tangent gain.
- Training telemetry: Medium raw norm fell 701.6→31.2→5.86 by steps 0/50/100; XL stayed
  22,062→17,307→8,504→5,381→7,655 at 0/50/100/200/250. Medium escapes clip domination;
  XL does not, and its clipped downstream updates fall below Adam epsilon (above).

`stop_attn_content`/`stop_mlp_content` are listed in the runner but rejected by the model
at this commit; attention-vs-MLP content attribution is not yet measured.

## Artifacts

Worker root `/tmp/rmt-norm-diagnostic`, isolated source `/tmp/rmt-norm-code-6c081ce6`.
GCS `gs://newproject-1-llm_projects_europe-west4/diagnostics/rmt-norm-6c081ce6/`;
local `/data0/xd/bam_diagnostics/rmt-norm-6c081ce6/` (per-arm `results/`, `probe.log`,
launcher `run_norm_probe_matrix.sh`).
This is an initialization/localization study, not a new formal training RUN.

## Paired initialization results (2026-09-29)

Four identical B1/T4096 Pile batches, exact cohort hashes verified across all four
native-initialization configurations. Values below are mean embedding address-bias
L2 gradient norms, not global norms or training-loss improvements.

| Model | Depth | VectorNorm | Full-M norm | Both norms |
|---|---:|---:|---:|---:|
| RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO |18|6688.128|1666.675|1685.829|
| RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO |28|73313.613|—|—|
| RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO |18|13719.160|—|—|
| RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO |28|258515.543|1752.075|1737.909|

Depth18→28 multiplies this gradient by10.96 in Medium and18.84 in XL. Native depth
also changes static-write initialization and scanned random draws; this is not a
fixed-weight prefix comparison. The result implicates accumulation with depth,
not embedding parameter count alone.

For XL28, stopping gradients through layer dynamic-write addresses gives44015.185;
stopping through contents gives279.732; stopping both gives9.652. Forward values
remain mathematically identical (small bf16 compiler differences occur); gate
backprop remains enabled. These are localization tools, not candidate training
rules. Medium18 gives1099.474,124.759,4.409 respectively.

Actual loss-cotangent RMS ratios across XL28 late attention sublayers are1.3–1.4;
full-M read pre-normalization reduces them to about1.02. The MLP differences are
much smaller. Raw residual M still grows with full-M norm: it stabilizes read
inputs, not the carried matrix. Ratios are directional VJP measurements, not
Jacobian spectral norms.

Follow-up commit `0ad214af` separates attention/MLP content stops, QK/V stops,
and normalizes only matrix-derived QK coordinates, retaining independent RoPE
coordinates. Focused CPU checks pass: stop interventions preserve forward loss
exactly; every case has finite gradients. Running on the same reserved FLEX_START
`llm-jax-v6e-1-1`, without changing formal training or enabling Pallas.

Completed first-wave artifacts are additionally archived at
`gs://newproject-1-llm_projects_europe-west4/diagnostics/rmt-norm-6c081ce6-reserved1-complete/`
and `/data0/xd/bam_diagnostics/rmt-norm-6c081ce6-reserved1-complete/`.
Use this immutable completed snapshot: another session is reusing the original
prefix on reserved0. Targeted follow-up uses a separate `rmt-norm-0ad214af` prefix.

## Attention localization: `0ad214af`

| XL28 intervention | Mean embedding bias gradient L2 |
|---|---:|
| VectorNorm baseline |258515.543|
| Stop attention dynamic-write content gradient |1366.032|
| Stop MLP dynamic-write content gradient |60364.664|
| Stop QK gradient |856.158|
| Stop V gradient |116552.951|
| Divide matrix-derived QK by per-token full-M RMS |1800.066|

The QK-only intervention preserves independent RoPE projections, V, MLP reads,
write normalization, residual M and final normalization. All gradients remain
active. Its144x reduction nearly matches full-M norm's148x reduction. At the
last attention layer the actual VJP RMS gain falls from1.3856 to1.0153. Thus the
large gradient is localized upstream of the embedding bias, to the repeated
unnormalized matrix-QK/attention/dynamic-content-write chain. This establishes
an initialization mechanism, not yet restored optimization or final-loss quality.

A useful differential identity is `dy_QK = sum_j a_j (v_j-y) ds_j`. Scaling M by c
scales matrix-derived logits as c²; V and y scale as c for fixed attention weights.
The content RMSNorm cancels the latter amplitude but does not cancel QK score
sensitivity. Locally, with attention distribution and directions held fixed,
QK-mediated content derivatives can scale as c, whereas the direct V-mediated
term scales as1/c. Softmax distribution changes prevent a universal monotonic
bound; the observed VJP gains and interventions supply the empirical test.

Medium18 gives875.035 with QK gradients stopped,3167.412 with V gradients stopped,
and1788.339 with QK-only RMS scaling. Thus QK scale control brings Medium and XL
bias gradient norms to nearly the same level, despite the38.65x original gap.

Commit `ee68acaa` additionally detaches the per-token normalization denominator
in backward while preserving its forward values, separating scale control from
the RMSNorm radial-gradient projection. It completed after the targeted probe on
the same reserved worker using its lock. No training architecture changes have
been merged into main or restarted as a formal RUN.


Detached-denominator control: mean bias gradient1804.108 versus1800.066 with
ordinary denominator differentiation (+.225%); every paired forward loss is
exactly equal. The reduction is overwhelmingly from controlling QK amplitude,
not the radial projection in the normalization Jacobian.

All probes completed; reserved FLEX_START worker is idle and retained. Targeted
and scale-control results, complete cohort metadata and scalar gradient taps are
saved under `/data0/xd/bam_diagnostics/rmt-norm-0ad214af/` and
`/data0/xd/bam_diagnostics/rmt-norm-ee68acaa/`, mirrored under the same diagnostic
prefixes in the EW4 GCS bucket. Exact launch scripts are saved in those directories.

Recommended next validation is QK-only scale stabilization with the original
optimizer, embedding bias and remaining VectorNorm architecture retained. Judge
its early training against normal XLProp MHA, not only the pathological original.
Initialization diagnosis alone does not establish healthy optimization or final
loss competitiveness. Do not treat gradient-stop probes as training remedies.
