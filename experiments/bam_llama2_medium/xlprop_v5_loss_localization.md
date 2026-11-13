# XLProp V5 FullRemat loss-spike parameter localization

2026-11-12. Frozen full-batch interventions for **BamXLPropV5FastKernelFullRemat**.

**Current result:** early value/readout changes propagate into later routing and residual-write direction changes. Early W_O/V rollbacks have strong conditional loss effects. Sampled weak-M write geometry also changes, but its causal role remains untested. This is not a demonstrated first-token sink mechanism.

## Reproduction and validation

- Immutable model runtime: `d54c4c5f1339b01e406932ec28be0e398722b918`; original v5p-32 / JAX0.8.1 / 50000-step compiled train step.
- Probe helper: `31eea4ab`, branch `mqy/xl-v5-loss-localization`; isolated worktree `.auto-agent/worktrees/xl-v5-loss-localization` in the owner workspace.
- Old parameters: CP35635; peak parameters: CP35675; exact original training batch35676 (128 sequences ×4096 tokens), same RNG and optimizer background. Every returned update is discarded.
- Old CP: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-a35650-20261109/ASpikeReplayTail35599/checkpoints/35635/items`.
- Peak diagnostic copy and outputs: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-lossloc-20261112-r2/`.
- Successful four workers: `t1v-n-d490e65a-w-{0,1,2,3}`. All exit0; all37 case records match across workers. All16 global input-shard indices/hashes match the prior exact-peak reproduction. Physical worker assignment differs, so comparison is by global shard index.
- Peak, old, and whole-attention controls reproduce historical full-batch loss/raw values below. Three focused CPU selector/plan/upload-namespace checks passed.
- Scope of attention rollback: families classified `attention_*`, excluding common read scales. L0–26 are slices on param scan axis1; L27 is the independent final layer. Layer rollback includes the whole classified attention module, not only localV.
- Successful owned resource: `xd-v5p-32-lossloc-1112-r3-uc` / us-central1-a. Race loser: `xd-v5p-32-lossloc-1112-r3-ew` / europe-west4-b. Both nodes and both queued resources independently verified NOT_FOUND after completion; protected compiler and formal training TPUs untouched.
- Local raw results: `training_reports/xlprop-v5-loss-localization-20261112/retry3/loss-localization.json`; successful and earlier failed worker logs under sibling `retry2/workers/`. Verification files in retry3.

## Controls and combinations

Revert: selected families from CP35635 in the CP35675 background. Introduce: selected families from CP35675 in the CP35635 background. localO is the BAM local read, distinct from attention output projection W_O.

| Case | Full-batch loss | Raw gradient norm |
|---|---:|---:|
| peak_control | 2.595792532 | 809415 |
| old_control | 1.977011561 | 0.215274 |
| revert_V_WO | 2.044545412 | 14229.2 |
| introduce_V_WO | 1.996144414 | 72.1775 |
| revert_O_WO | 2.063884020 | 2077.73 |
| introduce_O_WO | 1.990609646 | 2424.25 |
| revert_V_O_WO | 2.025623083 | 184.563 |
| introduce_V_O_WO | 2.007433414 | 450.331 |
| revert_all_attention | 2.002737761 | 1642.82 |

Full peak-minus-old loss difference is 0.618780971. Joint V/O/W_O rollback removes 92.14% of this difference, but introducing those new families into the old background adds only 0.030421853 loss (4.92% of the difference). Thus these parameter changes have a strong effect conditional on the peak background; they do not reproduce the full spike by themselves.

Loss and gradient rescue differ: V+W_O rollback yields lower loss but larger raw norm than O+W_O rollback; joint V/O/W_O raw norm remains far above the old control despite near-recovered loss.

## Per-layer whole-attention rollback

Recovery percentage = (peak loss − intervened loss)/(peak loss − old loss). These are overlapping conditional effects, not additive causal shares.

| Layer (zero-based) | Loss | Raw norm | Conditional loss recovery |
|---:|---:|---:|---:|
| L3 | 2.156707048 | 8204.72 | 70.96% |
| L0 | 2.311893940 | 6987.91 | 45.88% |
| L1 | 2.360633373 | 21551.1 | 38.00% |
| L5 | 2.381184101 | 11958.1 | 34.68% |
| L2 | 2.387402773 | 10975.1 | 33.68% |
| L4 | 2.446456909 | 12955 | 24.13% |
| L6 | 2.554530621 | 117797 | 6.67% |
| L20 | 2.588751793 | 221647 | 1.14% |
| L15 | 2.590075016 | 160864 | 0.92% |
| L14 | 2.590093851 | 21746.9 | 0.92% |
| L9 | 2.590234280 | 104246 | 0.90% |
| L18 | 2.590409040 | 439481 | 0.87% |
| L16 | 2.591058493 | 362136 | 0.77% |
| L17 | 2.591363668 | 127745 | 0.72% |
| L11 | 2.591914177 | 17368.3 | 0.63% |
| L19 | 2.592149973 | 324058 | 0.59% |
| L13 | 2.592799902 | 206453 | 0.48% |
| L21 | 2.593751431 | 987832 | 0.33% |
| L7 | 2.593960047 | 223378 | 0.30% |
| L22 | 2.594107389 | 584594 | 0.27% |
| L8 | 2.594302177 | 118790 | 0.24% |
| L24 | 2.594404221 | 859404 | 0.22% |
| L12 | 2.594569683 | 4225.27 | 0.20% |
| L23 | 2.594664812 | 859160 | 0.18% |
| L27 | 2.594689608 | 817933 | 0.18% |
| L25 | 2.594697237 | 802781 | 0.18% |
| L26 | 2.594835997 | 805296 | 0.15% |
| L10 | 2.606604576 | 17399.3 | -1.75% |

L0–L5 are the top six layers for loss rescue; L3 removes about71% of the peak-minus-old difference by itself. L6 is substantially weaker; every later positive effect is at most about1.14%. This is a conditional sensitivity ranking, not proof of chronological onset or uniqueness.

L12 rollback reduces raw norm from809414.56 to4225.27 while loss remains2.59457. L10 rollback reduces raw norm to17399.33 but worsens loss to2.60660. Gradient-norm reduction alone is therefore unsuitable for ranking the source of the loss spike.

## Next decision

First split L3 whole-attention rollback into localV, localO, W_O and routing/write controls; retain L0 as a second candidate and a late-layer negative control. Then patch only the actual sink-position V at fixed same-layer Q/K routing, with old-V, zero-V and ordinary-position controls. If first-position intervention does not recover loss, downgrade the sink hypothesis. Existing SwiGLU gain evidence concerns the downstream amplifier; this experiment identifies an upstream parameter-sensitive region, not a sink circuit or retraining fix. No finer TPU sweep launched at this checkpoint.


## Attention-sink causal follow-up (2026-11-12, r4)

The tested single-layer first-position Value mechanism is not the main explanation for the full-batch loss spike. This does not rule out sink routing changes, interactions across layers, or earlier onset mechanisms. All interventions below use the same frozen batch 35676 and unchanged historical model.

Original AOT parameter splits (23 cases) precede an instrumented full-remat objective (21 cases). The latter reproduces peak/old loss exactly; raw norm relative errors are 0.06655%/0.01055%. Four workers agree on every case metric; all 16 global input-shard hashes match the previous 37-case run. The original-AOT L3 all-attention rollback also reproduces exactly. Returned updates are discarded.

L3 localV rollback across the entire sequence restores 34.51% of the peak-minus-old loss increment; W_O restores 39.73%, their V/O/W_O group 63.29%, all attention 70.96%. In contrast, replacing only its first V with the old localV read on the **same peak upstream input** restores 0.119%; the full-old-checkpoint first V restores 0.916%, and zero first V restores 1.840%. Ordinary position128 zeroing restores 0.204%. These conditional effects are not additive causal shares.

The first-source attention mass at L3 averaged across the batch, 20 heads and six sampled query positions decreases from 0.08328 (old) to 0.06008 (peak). Mean per-head native V0 RMS rises 0.02236 → 0.02863, while the sampled mean p(q,0) × RMS(V0) changes only 0.000867 → 0.000903. Native V0 RMS is captured **before** each clamp. Routing statistics are FP32 reconstruction on queries16/128/512/1220/2048/4095 with segment and strict causal masks, not exact Splash probabilities or a uniform-token average. They do not prove that every head/token is stable. Actual causal forward calls retain original Splash.

Large raw-norm reductions alone are nonspecific: at L12, zero first V changes loss 2.595793 → 2.595800 while raw norm drops 808876 → 32580; zero position128 likewise leaves loss unchanged and reduces raw to 27685. Freezing first V to its native value preserves forward loss and leaves raw near baseline, separating simple gradient disconnection from forward operating-point perturbation. This supports sensitivity of the downstream gradient amplification chain without identifying the loss trigger.

Best current inference: sequence-wide L3 content/readout changes move downstream FFNs to a harmful operating point; first-position V alone is insufficient. The next discriminating test would restore **routing** while retaining peak V/readout, and compare prefix/interior position groups for the L3 localV/W_O effect. No additional diagnostic resource is launched after this stage.

### Original-AOT cases

| Case | Loss | Raw grad norm |
|---|---:|---:|
| peak_control | 2.595792532 | 809414.562 |
| old_control | 1.977011561 | 0.215273544 |
| revert_L03_V | 2.382264376 | 25777.3359 |
| revert_L03_O | 2.483150721 | 12025.8105 |
| revert_L03_WO | 2.349965572 | 18941.8457 |
| revert_L03_QK | 2.502486706 | 33645.4844 |
| revert_L03_write | 2.568997145 | 118829.203 |
| revert_L03_V_O_WO | 2.204149723 | 6380.45605 |
| revert_L03_all | 2.156707048 | 8204.71582 |
| revert_L00_V | 2.557567120 | 26816.1504 |
| revert_L00_O | 2.582391739 | 20524.6855 |
| revert_L00_WO | 2.353046656 | 40392.6055 |
| revert_L00_QK | 2.590943336 | 73159.2656 |
| revert_L00_write | 2.561372757 | 53406.6641 |
| revert_L00_V_O_WO | 2.331946135 | 4085.66357 |
| revert_L00_all | 2.311893940 | 6987.91162 |
| revert_L12_V | 2.594961405 | 52116.2188 |
| revert_L12_O | 2.595736265 | 12615.6445 |
| revert_L12_WO | 2.595643520 | 92454.0703 |
| revert_L12_QK | 2.596091509 | 8735.31543 |
| revert_L12_write | 2.595366716 | 41095.5625 |
| revert_L12_V_O_WO | 2.594688892 | 13134.8174 |
| revert_L12_all | 2.594569683 | 4225.27393 |

### Instrumented sink cases

Clamp modes: native, zero, old donor, or frozen current value. Same-layer Q/K is unchanged; later layers can respond. Raw gradients refer to the clamped graph, with frozen-current identity as a derivative control.

| Case | Loss | Raw grad norm |
|---|---:|---:|
| peak_control | 2.595792532 | 808875.938 |
| old_control | 1.977011561 | 0.215296254 |
| L00_old_native | 1.977011561 | 0.215296254 |
| L12_old_native | 1.977011561 | 0.215296254 |
| L03_old_localV_parameters_same_upstream | 2.382264376 | 25739.5625 |
| L03_old_localV_first_V_same_upstream | 2.595057726 | 10968.0098 |
| L03_native | 2.595792532 | 808875.938 |
| L03_freeze_first_V | 2.595792532 | 807451.25 |
| L03_zero_first_V | 2.584409475 | 213010.703 |
| L03_old_first_V | 2.590123177 | 29245.4707 |
| L03_zero_position128_V | 2.594530821 | 45269.1562 |
| L00_native | 2.595792532 | 808875.938 |
| L00_freeze_first_V | 2.595792532 | 808766.125 |
| L00_zero_first_V | 2.586647272 | 10191.2725 |
| L00_old_first_V | 2.595410824 | 96396.6875 |
| L00_zero_position128_V | 2.595140696 | 427599.312 |
| L12_native | 2.595792532 | 808875.938 |
| L12_freeze_first_V | 2.595792532 | 808879.438 |
| L12_zero_first_V | 2.595799923 | 32580.2246 |
| L12_old_first_V | 2.595819712 | 23810.7207 |
| L12_zero_position128_V | 2.595814943 | 27685.4199 |

### Reproduction and closeout

Runner `xlprop_sink_probe.py` and focused tests are committed at `1d176fbdfcf217e85375f939eaa29d30e7149392`; helper `xlprop_loss_localization_probe.py` is unchanged. Local evidence: `training_reports/xlprop-v5-loss-localization-20261112/sink-r4/{loss-localization.json,sink-localization.json,verification.json,resource-cleanup-verification.json,workers/}`. GCS: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-lossloc-20261112-r4/`. Worker wrapper retains original d54 model/AOT, original reference CP35635 and existing owned r2 copy of CP35675.

Resources `xd-v5p-32-sink-1112-r4-uc` / `-ew` both released; independent node and queue queries return actual NOT_FOUND for all four objects. A transient loser cleanup error was resolved by final cleanup. No HC training resource or retained compiler was touched.

## Routing and joint-prefix follow-up (sink-r5, 2026-11-12)

Same frozen batch128×4096 at step35676, CP35635/35675, historical d54 model and original AOT. All27 original-AOT cases and33 instrumented full-batch cases completed on four workers. Case metrics agree across workers (excluding timing), all16 global input-shard hashes match r4, peak/old loss exactly reproduce and raw-gradient fidelity gates pass (0.06655% /0.01055% relative error). Helper commit `f45520a1`. Three CPU tests cover group layer selection, token range, untouched layers and frozen-identity derivatives.

The leading explanation remains full-sequence content/readout changes in early attention layers, with a smaller prefix-mediated contribution. Joint first-token V ablations rescue4.37% (L0–5) or5.68% (all28 layers); prefix8 ablations rescue13.46%/16.08%, versus2.31%/2.07% for interior8 controls. Restoring L0–5 V/W_O parameters rescues86.12%, all attention90.18%; Q/K rollback rescues27.67% (L0–5) or30.27% (whole model). These conditional effects overlap and must not be added.

This supports prefix sensitivity but does not establish attention-sink instability as the principal cause. V ablation is a strong content intervention, not restoration of healthy routing. Q/K parameter rollback leaves common read scales and current upstream state; it is not an exact old attention-map clamp. Multi-layer interventions can change downstream routing. Frozen-current V controls preserve loss exactly while changing derivatives; ordinary-position ablations can sharply reduce raw gradients with little loss rescue. Thus raw norm reduction is not evidence of identifying the loss trigger. No direct temporal-onset or training-stabilization experiment was done.

Loss rescue = (peak − intervention)/(peak − old), peak2.595792532, old1.977011561, increment0.618780971. L0 is zero-based.

| Intervention | Loss | Raw grad norm | Loss-increment rescue |
|---|---:|---:|---:|
| revert_L00_05_QK | 2.424562693 | 25519.084 | 27.672% |
| revert_L00_05_V_WO | 2.062868595 | 280.32642 | 86.125% |
| revert_L00_05_all | 2.037791491 | 226.64597 | 90.177% |
| revert_all_QK | 2.408491373 | 13759.761 | 30.269% |
| L00_05_zero_first_V | 2.568723917 | 51092.781 | 4.375% |
| L00_05_freeze_first_V | 2.595792532 | 807307.62 | 0.000% |
| L00_05_zero_position128_V | 2.594747782 | 60295.949 | 0.169% |
| L00_05_zero_prefix8_V | 2.512530565 | 22378.119 | 13.456% |
| L00_05_freeze_prefix8_V | 2.595792532 | 804278.81 | 0.000% |
| L00_05_zero_interior8_V | 2.581478596 | 150714.81 | 2.313% |
| L00_27_zero_first_V | 2.560631990 | 23284.121 | 5.682% |
| L00_27_freeze_first_V | 2.595792532 | 807355.38 | 0.000% |
| L00_27_zero_position128_V | 2.595017433 | 119304.95 | 0.125% |
| L00_27_zero_prefix8_V | 2.496270418 | 80188.844 | 16.084% |
| L00_27_freeze_prefix8_V | 2.595792532 | 804144 | 0.000% |
| L00_27_zero_interior8_V | 2.583007336 | 11292.156 | 2.066% |

Group `first_value_rms_by_head` is native preclamp L3; group sampled-routing statistics are zero placeholders, not measurements. No interpretation uses those group placeholder statistics. Single-layer statistics retain the six-query FP32 reconstruction limitation from r4.

Artifacts: `training_reports/xlprop-v5-loss-localization-20261112/sink-r5/{loss-localization.json,sink-localization.json,verification.json,resource-cleanup-verification.json,workers/}`; GCS `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-lossloc-20261112-r5/`. Original r2 diagnostic CP copy reused read-only; updates discarded. Controller completed all60 cases; EW resource released23:27:12UTC, UC23:34:54UTC. Independent exact node/queue describes returned real NOT_FOUND for all4 objects. No HC training or retained compiler was touched.

Next discriminating test would patch matched old routing while retaining peak values, or test early-onset states. This stage ends here; no new resources are started automatically.


## Pattern/write follow-up r6: execution failure, no new inference (2026-11-13)

The original frozen peak executable reproduced loss2.595793/raw809414.5625 at step35676.
The new capture helper then failed before its7-case localization/capture stage: it imported
`layers.bam_kernels`, a later module name absent from immutable runtime d54c4c5.
The correct historical module is `layers.bam_pallas`. This was a probe implementation error,
not a model numerical failure. No `results/loss-localization.json`, `pattern-write.json`, or
pattern/write NPZ was produced; r6 supplies no new causal or residual-energy evidence.

The helper now uses a historical runtime import function and an explicit import/signature
regression against a detached d54 source checkout. All5 focused CPU tests passed in6.202s;
TPU forward fidelity remains untested for this corrected capture. Existing r6 runner/manifest
retain the exact failed helper identity; do not overwrite/reuse them as a successful run.
All four exact node/queue describes returned NOT_FOUND at04:27UTC despite the controller's
prior UC delete-call timeout. The fixed acquisition deadline03:32:37UTC has elapsed; no new
resources were requested and no deadline was extended.

Separately, ChatServer wakeups stopped because the accumulated reason exceeded the server's
length limit. The reason was replaced by a205-character pointer to the canonical monitor;
status API independently confirmed enabled/on, ten-minute interval, next04:37:14UTC.


## Ongoing mechanism investigation (owner room764, 2026-11-13)

The owner explicitly asks for continued analysis until a reasonable explanation
and a concrete fault location are reached. r7 is active under its existing bounded
acquisition/execution budgets; it is not the scientific stopping point.

Current best explanation is a value/readout-driven operating-point shift into
latent expansive ordinary-SwiGLU directions. This joins, but does not conflate,
two observations: early-attention rollback rescues loss; fixed-boundary local
VJPs identify the ordinary MLP amplifier. New attention/value parameters can
activate large gradients even on old MLP weights; small parameter-norm changes
and a healthy whole-M RMS do not identify the sensitive direction. First-position
V ablation is much weaker than whole early V/W_O rollback. None of this yet
identifies the earliest offending write or establishes a universal sink circuit.

Competing explanations to discriminate with matched r7 captures:

1. **Routing shift:** changed per-head weights, not merely changed mean sink mass,
   dominate the reconstructed context delta. Must compare native Splash context
   error before interpreting FP32 probabilities.
2. **Content/projection shift:** at fixed old P or fixed W_O, value-context and
   write-direction changes dominate; ordinary residual magnitude may stay small.
   Old-basis low-energy M components could change while total energy stays healthy.
3. **Numerical loss of weak writes:** BF16 addition removes a material directional
   component. This requires measured rounding/reference errors, not an inference
   from weak amplitude alone.

The host analyzer `analyze_pattern_writes.py` keeps per-head TV/JS/entropy/prefix
mass, exact algebraic P/V and input/W_O decompositions, native reconstruction
errors, residual write ratios/cosines/energy growth, and sampled M singular-space
write energies. M weak spaces are complements of the old matrix's 99%-energy
subspace (including its null space); both current and fixed-old bases are reported.
Signed component projections may exceed1 or be negative and are not causal shares.
Three synthetic mathematical tests check empty causal rows, disjoint distributions,
null/strong singular subspaces, and canceling decomposition components.

If captures nominate a specific path, next diagnostic should patch that component
in the peak model against healthy and equal-size/random-direction controls, then
measure both full-batch loss and downstream ordinary-MLP inputs/gains at L6/L13/L14.
This must include the healthy-to-onset boundary when claiming a trigger: comparing
CP35635 and35675 on batch35676 alone is not chronological-onset evidence.

Completion means a specific layer/path or direction supported by matched forward
measurements and a targeted causal rescue/control, connecting early writes to the
known downstream amplifier. A raw-gradient reduction, a large mean-route change,
a weak-energy correlation, or successful script execution is insufficient. Report
any remaining scope limits; do not force a unique explanation unsupported by data.
No architecture training, old Medium restart, borrowed training TPU, or protected
compiler use is authorized by this diagnostic follow-up.


## r7: early readout changes propagate through depth (2026-11-13)

Seven original-AOT cases completed; six forward captures passed their per-case
absolute-loss fidelity bound1e-3 (largest error0.00062704). Peak/old original AOT
losses remain2.595792532/1.977011561. The TPU was preempted before the last
all-attention forward capture and worker closeout. Four available process0 input
hashes match the earlier16-shard reference; all-worker exit codes and a fresh
independent16-shard comparison are unavailable, and are not claimed.
All four exact resource node/queue objects independently returned NOT_FOUND.

Original-AOT first-six-layer rollbacks:

| Family | Loss | Conditional recovery of peak-minus-old loss |
|---|---:|---:|
| QK |2.424562693|27.67%|
| V |2.262374640|53.88%|
| W_O |2.124925375|76.10%|
| V+W_O |2.062868595|86.13%|
| all attention |2.037791491|90.18%|

Sampled geometry uses two fixed sequences, twelve queries per sequence, all20
heads and all4096 source keys. Reconstructed FP32 contexts differ from native
Splash contexts by at most0.212% across the six layers/old-peak controls. This
supports interpreting the following sampled probability comparisons; they do
not describe every query of the full batch.

| Layer | Mean route TV, peak vs old | Write cosine, old vs peak | Full-batch write RMS, old -> peak |
|---|---:|---:|---:|
|L0|.0285|.9993|.17334 -> .18400|
|L1|.0717|.9776|.04750 -> .04668|
|L2|.1426|.8937|.04975 -> .03638|
|L3|.1548|.9067|.04995 -> .05368|
|L4|.2767|.7967|.06715 -> .07479|
|L5|.3460|.6501|.08464 -> .09923|

Thus weights/patterns are not stable, and write magnitude does not uniformly
explode or collapse. At L5, QK rollback leaves mean route TV.3258, W_O rollback
reduces it to.1726, and V+W_O rollback to.1143. L0 routing is exactly unchanged
by its downstream V/W_O rollback, as expected; later routing responds through
changed upstream states. This is evidence for a depth feedback path, not proof
that routing changes themselves cause the loss peak.

Algebraic fixed-old-P decomposition assigns about94.8% of L0 context-delta
projection to changed V tensors, falling to51.3% at L5 as route changes grow.
At fixed head inputs, direct W_O-kernel changes account for only0.5--14.2% of
observed old-to-peak write-delta projection, despite their strong conditional
loss effect. These are order-dependent signed vector projections, not causal
shares. The V tensors and head inputs include upstream-state/local-read changes,
not only the local V parameter subset. Together with the rollout controls,
these observations favor a small early readout perturbation being propagated
and amplified over a large isolated W_O-output amplitude error.

At the peak-dominant sequence104/query1220, L5 attention writes have total
squared norm255.58 ->271.62, while energy outside the old M content subspace
retaining99% of its energy rises49.14 ->100.68 (fraction19.23% ->37.07%). The
basis is held fixed at the old state; it includes the old null space. This
localized weak-component increase is real in these captured tensors but remains
correlation; the second sampled sequence does not show the same doubling.
Attention+memory-write reference errors against actual M deltas are at most1.622%.

A measurement issue was caught during host cross-check: full-batch JIT addition
rounding fields report near-zero errors, inconsistent with materialized BF16
snapshots. Those two fields are excluded from analysis; compiler fusion or
reassociation is a possible explanation, not independently established. Host
recomputation on samples gives addition error relative to write norm0.17--2.58%
and rounded-away coordinate fractions0.17--0.87%. These measurements neither
establish BF16 as the cause nor justify calling all weak writes lost.

Artifacts: `training_reports/xlprop-v5-loss-localization-20261112/pattern-r7/`:
`loss-localization.json`, `pattern-write.json`, six named NPZs,
`paired-pattern-write-analysis.json`, `verification.json`,
`resource-cleanup-verification.json`, `pattern-write-summary.png`. Host analyzer
excludes invalid full-batch rounding fields and reports sampled host values.

Next discriminating test: separate **old write direction with current amplitude**
from **old amplitude with current direction**, with whole-write replacement and
no-op controls, across L0--5 and L3 alone. Restore donor writes on the TPU and
keep them sharded (six full-batch vectors are large; do not download or replicate
all of them). Patch the ordinary attention output while preserving same-layer
pre-W_O M-write factors, and measure the downstream MLP operating points/gains.
If only direction rescue works, localize the subspace next. A weak-M hypothesis
requires its own targeted component-patch/control. Current data do not complete
that causal bridge; continue the owner-authorized investigation.


### Direction/amplitude follow-up design (r8)

Runner `xlprop_write_direction_probe.py` captures full old and peak ordinary
attention writes for L0--5, retaining both as explicitly sharded TPU arguments
(about24GB globally if BF16, not replicated or downloaded). Frozen peak reference
writes are a forward-identity gate before any interpretation. At selected layers,
patch either old direction x frozen-peak norm, frozen-peak direction x old norm,
whole old write, or seeded-coordinate-permuted old direction x frozen-peak norm.
The references stay fixed across interventions; later native inputs cannot silently
change the amplitude reference. Apply independently to L0--5 and L3 alone.

Two original-AOT controls precede11 forward cases (old, peak, frozen-peak, four
patch modes x two layer scopes). Old/peak/frozen identity loss errors must be
below1e-3. Donor directions must be nonzero at valid tokens. Same-layer pre-W_O
M-write factors remain untouched; downstream states respond. Export only sampled
writes and normalized ordinary-MLP inputs at L6/L13/L14, not full donors. This
stage makes no instrumented gradient/gain claim. Captured inputs can nominate
which downstream fixed-boundary VJP to check next.

Three pinned CPU checks passed in5.364s against the historical d54 source:
mode identity, separate norm/direction and seeded-permutation norm preservation,
finite BF16 zero fallbacks, and historical MLP hook import/signature. These are
preparation checks; forward fidelity and causal loss outcomes remain pending.

### 2026-11-13 direction/amplitude r8: ordinary write content is the stronger causal lever

Frozen batch35676 / checkpoints35635 and35675 / unchanged d54 model. Full-batch first-six ordinary attention outputs were captured once under both checkpoints and held sharded on16 chips. At selected layers, patches substitute old direction at peak norm, peak direction at old norm, full old vector, or seed9876 coordinate-permuted old direction at peak norm. Same-layer pre-W_O M-write factors remain unchanged; downstream computation responds. This is an activation-clamping experiment, not an isolated local-weight substitution.

Original AOT old/peak losses1.977011561/2.595792532 and raw gradients.2152735/809414.56 reproduced. Instrumented old/peak/frozen-peak loss errors <=.000402451; frozen-peak identity equals instrumented peak exactly. All4workers exit0 and scientific metrics agree (timings and process-local shard identities excluded from equality); all16 actual input hashes match r5. Both exact diagnostic nodes and queues independently NOT_FOUND after controller cleanup. Local artifacts: training_reports/xlprop-v5-loss-localization-20261112/direction-r8/{write-direction.json,loss-localization.json,verification.json,analyze.py,direction-amplitude-summary.png}; eleven small sampled NPZs retained.

|Scope (zero-based)|Old direction, peak norm|Old amplitude, peak direction|Whole old write|Permuted old direction, peak norm|
|---|---:|---:|---:|---:|
|L0–5 loss|2.037339|2.348651|2.025789|5.946329|
|L0–5 loss increment rescued|90.25%|39.94%|92.12%|−541.47%|
|L3 loss|2.105986|2.520068|2.111349|2.172365|
|L3 loss increment rescued|79.16%|12.24%|78.29%|68.43%|

Sampled actual norm errors <=.228%, and direction-only cosine with healthy vector >=.999998 on average, verify direction/amplitude separation at2sequences×12queries (not a full-batch maximum claim). In L0–5 direction patches, mean sampled normalized ordinary-MLP input distance to old, divided by unpatched peak-to-old distance, falls to.227/.186/.185 at L6/L13/L14. Amplitude-only leaves.837/.843/.843. Whole-old leaves.081/.121/.124; permuted direction worsens to1.688/1.432/1.465. Thus early ordinary write geometry causally changes downstream input geometry and peak loss; amplitude contributes but is insufficient in this comparison. No patched VJP/gain measurement was made.

Cautions that change interpretation: L3 permuted direction itself rescues68.43%, so generic disruption of the abnormal propagation path competes with uniquely restoring a healthy content direction. Six-layer permutation is catastrophic and not a small matched-distance null; one coordinate permutation does not establish all random controls. Cross-layer clamping bypasses feedback and uses healthy activations computed with the full healthy network, so these percentages neither identify a single W_O parameter nor prove the temporal onset trigger. Effects are conditional and overlapping, not additive. Weak-M energy remains observational; ordinary-write rescue with same-layer M factors preserved does not rule out downstream M interactions.

Next highest-value causal bridge (room764 continuing authorization): use bounded follow-up to connect the restored L6/L13/L14 normalized inputs to known expansive ordinary-SwiGLU units/gate-derivative paths; measure unit activation and local fixed-boundary gain, with an original-AOT/identity fidelity gate. Distinguish healthy-direction restoration from a generic break in propagation using partial direction interpolation and equal-size perturbation controls at L3, before assigning a specific harmful component. Actual onset chronology requires separate35636 batch. Research goal remains open; no new diagnostic allocation was launched in this finite reporting turn.

### 2026-11-13 r9 matched-angle controls: direction-specific loss rescue; internal-gain identity fails

All23 cases complete, all4 worker exits0/scientific metrics identical (excluding timing and process-local shard lists); all16 input hashes match r8. Unclamped old and peak losses exactly match original AOT; raw-gradient relative errors.1450%/.1152%. Frozen-peak loss differs.00061202 (<.1% of the original loss increment). Peak-norm-preserving first-six rotations toward the healthy direction by25/50/100% of each token's angle rescue37.78/65.04/90.25%; three equal-angle/displacement random tangents rescue6.57/21.03/5.91%. Equal-size rotation away worsens loss to3.67743 (−174.80% rescue). L3 corresponding toward rescues31.43/53.38/79.11%, matched random12.87/17.65/10.88%, away loss3.31566. This strengthens the direction-specific conditional loss effect beyond r8's unmatched large permutation. Actual sampled random norm error <=.216%; displacement max discrepancy <=.311% of mean healthy displacement. Not a full-batch maximum claim.

**Do not use this run to attribute MLP gain recovery to healthy-direction restoration.** Unclamped peak at seq104/q1220 reproduces ordinary-MLP L13/L14 gains27.688/11.838; frozen-peak identity itself yields.945/.811 despite nearly unchanged batch loss. Sample ordinary write differences are1.09% relative L2, and dominant L13/L14 normalized inputs differ173.9%/162.3%. Thus local state/cotangent identity is not validated by loss-only fidelity. The donors originate in a forward-only compiled capture, whereas measured cotangents originate in another full-gradient graph; changed materialization/lowering is a competing explanation that must be tested. Do not explain the identity failure as the intended model intervention or suppress it behind aggregate fidelity. Local outputs/gains in this run remain diagnostic conditional data. `bridge-r9/verification.json` explicitly sets internal identity false and blocks gain-causal interpretation.

Selected exact r9 UC TPU is intentionally retained under owner room778; unused EW competitor released. Follow-up same-graph donor capture (`xlprop_write_bridge_identity_probe.py`) exports donors from the very full-gradient compiled program used for interventions, preserves their batch sharding, never gathers them to host, and gates frozen identity on sampled write equality, per-layer input/output/gate/up relative errors<.5% and input/output-VJP errors<5%. Six CPU tests PASS9.480s, including same-graph frozen identity preserving downstream cotangents while upstream parameter gradients change. Actual TPU gates remain pending. Use retained owned TPU only; no new race. The weak-M component has still not been selectively intervened on, and ordinary residual weak-basis calibration remains necessary. Research remains open.

### 2026-11-13 r10 repairs internal identity; localized MLP gain is not a sufficient loss proxy

Same-full-gradient-graph donors resolve the r9 failure: original old/peak forward losses match AOT, raw errors.1450%/.1152%; frozen-peak loss, sampled writes, all sampled L6/L13/L14 normalized inputs/outputs/gate/up preactivations and input/output cotangents are **bitwise identical** to unclamped peak. All23 cases and4workers complete with identical scientific metrics; all16 input hashes match r8. Sampled norm mismatch remains<.215%; matched BF16 displacement discrepancy reaches.4473% of mean healthy displacement (the inherited r9.4% host check exceeded; actual values retained and reported with.5% tolerance, not silently called exact matching).

First-six healthy rotations25/50/100% rescue37.78/65.02/90.27%; matched random6.26/20.91/6.06%; away−174.82%. At the previously dominant seq104/q1220, peak/frozen L13/L14 ordinary-MLP VJP gains are27.688/11.838 and healthy-direction gains.652/.639. However random directions also reduce those local gains to about.697–1.076/.644–.842, and away reduces them to.548/.510 while worsening loss. Thus genuine local gain suppression is neither sufficient for healthy loss recovery nor a global spike explanation; unsampled gradient-peak relocation remains possible. These are actual conditional network cotangents; both local input and upstream direction vary. The weak-M contribution is still untested.

Next bounded probe `xlprop_weak_memory_probe.py` directly changes analytic attention-write weak components, or actual total M-update weak components, relative to each healthy sampled M's99%-energy content basis. It targets24positions (same2sequences×12queries) at L3/L5/L0..5; full-batch loss is measured but intervention is sparse. Weak removal/healthy restoration keep the current strong component fixed before BF16 readdition; matched-norm strong and permuted-subspace removal controls plus whole-attention removal/restoration separate component specificity. Native combined-write reference error and actual strong-subspace leakage must accompany interpretation. Whole-batch ordinary residual weak-basis analysis remains separate and pending. Four CPU projection/energy/strong-preservation/matching/zero tests PASS; original-AOT forward/raw and exact no-op M-state gates precede27cases. Reuse owner-retained exact TPU; no acquisition/deletion. Sparse negative results do not exclude unmodified tokens.

### 2026-11-13 r11: sparse weak-M intervention is effective but barely changes batch loss

All27 cases complete;4worker exits0/scientific JSON agrees and16input hashes match r10. Original old/peak loss exact; raw-gradient errors.2490%/.1243%. Native total-write delta versus summed analytic attention/memory reference errors.3424%/.3367%. Same-program no-op M tensors/references/raw gradient are bitwise identical. Native M arrays were serialized as NumPy void2 by the BF16 custom dtype; analysis decodes only known M fields as ml_dtypes.bfloat16, verifies byte roundtrip, and independently reproduces the reference errors exactly. Original archives remain unchanged.

For the24sampled positions at L3/L5/L0..5, attention-weak removal/healthy restoration rescues at most.03422%/.02312% of the full-batch loss increment. Weak intervention on the actual total M update reaches at most.05891%; even replacing/removing the complete sampled attention write reaches only.11163%. Matched strong/random interventions are similarly tiny. Weak corrections actually execute: aggregate actual/intended correction norm ratios.99995–1.00038; strong-subspace leakage is.369–1.393% of intended correction norm. At the previously observed L5/seq104/q1220 point, correction norm ratios are about.9997–1.0003 and strong leakage about.53–.92%. Thus the tiny effect is not explained by the weak correction simply disappearing in BF16. These positions do not explain the whole batch loss peak under tested interventions. This is not a whole-batch negative result and does not exclude other tokens or the ordinary residual stream's weak directions. Raw gradients often fall greatly under these local edits while loss remains almost fixed, again separating a local gradient peak from global loss damage.

Artifacts `weak-r11/{weak-memory.json,loss-localization.json,verification.json,analyze.py}` plus27sampled NPZs. Retain exact owned UC TPU under room778. Next priority is full-batch **ordinary** residual-write projection against healthy uncentered second-moment energy bases, rather than repeating sparse M controls. CPU utility `ordinary_weak_projection.py` provides95%/99% nested bases with separate held-out energy accounting and frozen-peak weak removal/healthy restoration/matched strong/random controls;4tests PASS2.075s. Backend integration/calibration and full-batch causal run remain pending. Same-full-gradient donor identity and actual BF16 component leakage must remain gates.
