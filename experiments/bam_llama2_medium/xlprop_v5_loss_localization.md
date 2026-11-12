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
