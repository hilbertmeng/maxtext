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

## MediumProp QK-zero / no M pre-norm control (2026-10-03)

User-authorized RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKZeroInit`, based on `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm`. Runtime `8d9394089df060f5d8ea4f50404498aab488a71a`, worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`; pure JAX, layer scan. Static Q/K slices of qkv_key zero-init; V uses bit-identical original Gaussian draw. Remove attention/MLP M read pre-norm and its scales; retain both VectorNorm and final_matrix_norm. SharedWriteNorm unchanged for layer attention/MLP; original embedding content projection and writes unchanged. MLP4078 unchanged;431773472 parameters,129600 fewer (0.09 W_Q) than parent. Extra absolute write/carry health enabled; historical speed comparison is consequently not perfectly matched. Checkpoints permanent every1000 plus latest2.

Targeted CPU checks: full budget/scope, exact unchanged initializers outside QK, scanned forward/finite gradients, nonzero Q/K gradients, SharedWriteNorm regression;5 passed. Bet final13500 loss−parent -.006; speed .380 vs parent~.375. Direct loss baselines parent and original NoO. Use former H TPU `xd-v5p-16-2910039-maxtext` UE5a after hot handoff; compiler `llm-jax-v6e-1-0` EW4a verified FLEX_START, borrowed without cleanup ownership.

### QKV-zero paired arm

User additionally authorized `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInit`: static V zero-init added to QK-zero, no other architecture/MLP/health change. Runtime6a1edf49bcf12c4aee70b095d10aec20d68727a4, same family worktree/branch. Parameters431773472,MLP4078; direct baseline QKZeroInit only. Bet13500 +.002 vs QK-zero, same~.383step/s. Launch ID2910042, UE5a `xd-v5p-16-2910042-maxtext`; borrowed EW4a FLEX_START compiler `llm-jax-v6e-1-0`. Parallel launcher gates focused CPU checks, AOT and training prequeue before FIRST_STEP.

QK-zero parent successfully started on retained TPU: AOT loaded, FIRST_STEP3, speed10–14 .3828step/s (+2.1% vs historical SharedWriteNorm .375, extra write-scale health). Step20 all18 layer health finite; final-layer attention M RMS6.686,MLP M RMS6.879; static Q/K RMS .01184/.01238 vs dynamic .4011/.4224. Final-layer attn and MLP write/carry ratios .0843/.0912. These are startup observations, not a stability conclusion.

QKV-zero CPU gate exposed an incorrect test expectation: dynamic V key_kernel is already zero-initialized, so zero static V means exactly V=0 initially and attention Q/K gradients must vanish. Static and dynamic V gradients are nonzero. A targeted real scanned-model update verifies Q/K gradients become nonzero in every tested layer after V learns; all gradients/health finite. Corrected test, no model change. Sealed runtime now1ecbd16c8768c92f37e838f14c08845864e5abb3. The first failed CPU gate released the owned training queue and did not launch training or delete the retained compiler.


### QKV-zero embedding compositions (2026-10-03)

Runtime `5e0812f`, worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`. Parent `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInit`; append `EmbedSeedZero` / `EmbedScale0` for the two new RUNs. Both share per-head normalized embedding contents, remove the independent embedding content projection and refund its 1 W_Q to MLP4078→4100. Total431759072, parent−14400 (−.01W_Q). SeedZero retains learnable seed_key with zero initialization, scale1; Scale0 uses original seed initialization but fixed scale0. No matrix pre-norm; VectorNorm/final norm, QKV-zero and layer SharedWriteNorm unchanged.

SeedZero will hot-replace strict BAM H on UE5a `xd-v5p-16-2910043-maxtext` (launcher2910044); Scale0 queues new UE5a `xd-v5p-16-2910045-maxtext`. Borrow EW4a FLEX_START `llm-jax-v6e-1-0` for serial AOT only, never delete it. Both plan13500, checkpoints200, keep1000+latest2; review2800/5000, reports1000. SeedZero compares QKV-zero; Scale0 compares QKV-zero and SeedZero. Bets terminal −.004/−.002 vs parent, speed~.383 vs .382.

Focused CPU tests passed (2 tests31.235s): full budgets/scopes, matching initial effective seed/output, finite forward/two-update gradients, learnable zero-seed gradient versus permanently zero Scale0 seed gradient, QK learning unlocked after first V update. Sealed config checked for both RUNs. The parallel Scale0 launch reuses this exact-commit CPU proof; it does not rerun the same tests.

Launch verified: SeedZero reached21+, AOT loaded and UE5a-local TruePile path; startup10–14 mean .3826step/s (parent .382, flat). Scale0 reached5 then14+, AOT loaded and local path; startup .3798 (−.6% vs parent, −.7% vs SeedZero). Both20-step dynamic read/write health finite. Strict H closed at2926 with committed checkpoint and local TB sync; its TPU transferred. Artifacts: `/data0/xd/bam_diagnostics/rmt-readnorm-launch/qkv-seedzero-*`, `qkv-scale0-*`.


### 2026-10-03: SeedZero final static read zero initialization

RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroFinalReadZero`; worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`, runtime `a08a7b10`; UE5a `xd-v5p-16-2910047-maxtext`. Only `rmt_final_read_key_zero_init=True`; parent SeedZero, all other model and health settings inherited. Parameters431759072, MLP4100 unchanged. Bet13500 parent gap-.003, speed~.382 unchanged. Reviews2800/5000. Targeted CPU budget/parent-parameter equivalence and staged gradient-unlock checks run concurrently with borrowed FLEX_START AOT and training prequeue.

Launch verified: CPU2 targeted tests26.966s; sealed config passed; AOT loaded; UE5a-local dataset; FIRST_STEP5 and step36 observed. Steps10-14 mean.3826step/s vs SeedZero~.382 (flat). Initial loss stayed nearuniform, began dropping afterstep24 and reached10.2211 at36. Retained FLEX_START compiler remains excluded from cleanup.


### 2026-10-04: Q/V/MLP static-read bias

RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBias`, parent `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero`. Runtime `ef379716`, worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`; pure JAX. Per-layer zero-initialized Q bias16x57 (NoPE only), V/MLP biases16x75, no K bias, no gate/norm. Added59616=.0414W_Q, total431818688; MLP4100 unchanged. Bet13500 -.003, speed flat. Checkpoints200, permanent1000+latest2; reviews2800/5000, regular1000-step reports.

CPU targeted2 tests passed: full budget, exact parent-parameter and initial-loss equivalence, bias gradients after two updates. Initial shape assertion used wrong layer axis; corrected to configured param_scan_axis. AOT verified on retained EW4a FLEX_START llm-jax-v6e-1-0 (no lifecycle ownership). User authorized FinalReadZero hot replacement; old committed2427, new on its UE5a xd-v5p-16-2910047-maxtext, launcher2910048. Extra owned prequeue xd-v5p-16-2910048-maxtext deletion requested. New AOT loaded, step18+ verified, correct zone-local TruePile; steps10-14 mean.3808 vs parent.3826 (-.5%).

FinalReadZero closeout: gap+.710305@200 -> +.056261@2400; last5+.069377, still improving at user stop2427. No speed gain. Single UE5a lease2026-10-03T14:10:54Z to16:00:22Z,1h49m28s,zero preemptions. Checkpoint committed, TPU transferred, TB incrementally synced. Artifacts `/data0/xd/bam_diagnostics/rmt-readnorm-launch/static-read-bias-*`, `finalread-final-report.txt`, `finalread-leases.txt`.

### 2026-10-04 Medium final VectorNorm skip

User-directed run: `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroFinalVectorSkip`.
Parent is current Medium QKV-zero/SharedWriteNorm/EmbedSeedZero; this is not an XL launch.
Runtime `4975460`, worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`;
TPU `xd-v5p-16-2910049-maxtext`, UE5a. AOT borrowed FLEX_START `llm-jax-v6e-1-0` EW4a, never owned for cleanup.
Add the existing `unembedding_vector_norm` output directly to final static+dynamic reads.
The proxy is the final, post-scan, final-matrix-normalized first16 rows; no layer-input proxy reuse.
No added parameters, gate, scale, or MLP changes (431759072 params, MLP4100).
Focused CPU checks prove same parameter initialization, exact output addition and finite gradients;
sealed config and AOT passed. FIRST_STEP37, actual AOT load and local dataset verified;
startup .3804 step/s vs parent .3826 (-.6%), finite initial health.
Bet: terminal parent gap -.003, steady speed flat. Reviews2800/5000, normal1000-step reports;
checkpoint200, permanent1000/latest2. Launch artifacts in
`/data0/xd/bam_diagnostics/rmt-readnorm-launch/final-vector-skip-launch.log`.

### 2026-10-04 XL QKV-zero / SharedWriteNorm / EmbedSeedZero

RUN `RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero`.
Runtime `671a0f25`, worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`;
owned TPU `xd-v5p-32-2910050-maxtext`, UE5a. Borrowed compiler `llm-jax-v6e-1-0` FLEX_START EW4a.
Parent is the 17547-step paused XL learned-scale/shared-write/embedding-SeedZero combination
(the user's historical “SWN”, not the earlier 5271-step failed combination).
Transfer Medium static-QKV-zero and no layer-M-pre-norm; retain vector/final norms, R384,
MLP6643 and shared normalized layer/embedding writes. No final vector skip.
1432131160 parameters; removed layer gains account for322560 fewer than parent.

User monitoring plan archived at `/data0/xd/bam_diagnostics/rmt-readnorm-launch/xl-qkvzero-seedzero-user-plan.txt`.
Selected zero-based output layers1/7/14/21/25/27 record global-batch/token mean energy fraction
and raw RMS after MLP addition (`M_output`/`diag_M_out`). Existing attn carry atlayer+1
provides previous-layer output RMS, but does not cover final output.
Existing write/boundary health retained. New gradient diagnostics export both norm and energy
fractions for embedding-address bias and LM head, plus clipping coefficient.
Checkpoint250, latest2, permanent every2000 throughout (user simplified retention).
Initial aec0fc2 attempt failed before training because training Orbax0.12.4 removed the CPU-supported should_keep_fn.
Runtime671a0f2 restores the original checkpoint manager and keep_period=2000. Global carry mean test distinguishes from averaging
per-example fractions; scanned finite-gradient/zero-init/export and full-budget checks passed.

Early same-step loss reports50/100/200/300/500 vs MHA/B; then500-step windows, reports every2000.
At500 stop/diagnose if still aboveMHA, NaN, or raw gradients persistently >10x B.
Health reports at500/4000/8000/10000/17500 and anomalies. At10000 discuss stop if gain/Mudd<1.38
or prior2k decline>.02. At17500 gain/Mudd>=1.25 is partial recovery, not LLF superiority.
Historical supplied numbers are reference hypotheses until verified against the archived runs.
MHA is the gain origin, not a gain-ratio denominator (zero denominator); report its loss gap.
The shared-component hypothesis requires measured mediation, not merely a stable run that loses gain.
Bet adopts proposed scale:70% stable; conditional gain/Mudd1.40 at10k,1.25 at17.5k; speed~.31
vs parent.307 with new statistics not matched.


XL stage0 passed at500 (runtime671a0f2): gap vs B -.033769, MHA -1.419849; new run crossed B at350.
Raw gradient norms at0/100/200/300/500: 6104.53/36.85/55.03/18.87/3.26;
B:230.16/54.91/80.90/29.01/6.42. Initial amplification is real, but did not persist.
Embedding address bias gradient energy fraction at100/200/300/500:
new .9685/.9846/.8877/.7163; B .9972/.9974/.9947/.9797.
Medium analogous SeedZero fractions .3053/.0948/.0318/.0105; it also peaked early (.8204 at50).
Do not infer XL-specific failure merely from early bias dominance. Medium/XL warmups135/500
also preclude treating equal early steps as equal optimizer progress; warmup cannot explain step0.
Final M output RMS100/200/300/500:8.36/7.95/8.11/8.16; middle shared-energy fraction:.250/.154/.109/.066.
Evidence: `/data0/xd/bam_diagnostics/rmt-readnorm-launch/compare-early-embed-grad.json`,
`xl-qkvzero-seedzero-health-500.json`, `xl500-report.txt`.
User replaced the paused SeedZero comparison with running
`BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile`; retain B/MHA/Mudd/LLF.
Report full gain ratios including LLF and independent MLP writes, not only Mudd.

FinalVectorSkip stopped2092 at user direction: initial negative gap reversed by600,
then +.004–.006 over1000–2000; last5mean+.004829 vs SeedZero, no parameter/speed benefit.
Checkpoint committed, node/queue absent, TB synced; main exp.py and regional lease ledger updated.


### 2026-10-04 Medium MLP front16 writes, full48 every third layer

RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPFront16FullEveryThird`.
Parent is completed13500 Medium QKV-zero/SWN/EmbedSeedZero. Runtime2e89bd2d,
worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`; UE5a TPU
`xd-v5p-16-2910051-maxtext`; borrow EW4a FLEX_START llm-jax-v6e-1-0 without cleanup ownership.
Question: mimic BAM's distinction between vector-local MLP updates and periodic whole-M writes,
not static-vs-dynamic sparsity. Both static and dynamic MLP writes use rows[16,48,16] in each
three-layer block; full rows at zero indices1/4/7/10/13/16. Attention and all reads unchanged.
Restricted addresses normalize their actual16 coordinates; no full48 projection followed by masking.
Removed per restricted layer: address_up256x512, address bias512, static key512 =132096;
12 layers save1585152=1.1008W_Q. Refund per-layer MLP4100->4137 at those layers;
widths[4137,4100,4137], total431772320 vs parent431759072 (+13248,+.0031%).
Block scan is needed for heterogeneous parameter shapes; homogeneous parent remains layer scan.
Focused tests: full budget/shapes, untouched attention and full-write parent equivalence,
exact absence of tail-row MLP updates, six-layer scan forward/backward and per-layer health export.
Initial3 tests passed41.582s; strengthened tail-health assertions included in final launch CPU gate.
Bet13500 parent gap-.004, speed+.5%; reviews2800/5000; checkpoint200/permanent1000/latest2.
Launcher logs `/data0/xd/bam_diagnostics/rmt-readnorm-launch/mlp-prefix-launch.log`.
