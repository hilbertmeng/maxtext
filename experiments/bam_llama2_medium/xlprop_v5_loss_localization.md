# XLProp V5 FullRemat loss-spike parameter localization

2026-11-12. Frozen full-batch interventions for **BamXLPropV5FastKernelFullRemat**.

**Result:** early attention parameters dominate conditional loss rescue; zero-based L3 (the fourth attention layer) is strongest. This does not yet identify its localV branch or establish an attention-sink mechanism.

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
