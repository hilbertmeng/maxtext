# Frozen old MUDD43k/53k softmax precision

Question: change only old PaxML lm_head softmax from FP32 to BF16; compare immediate loss difference with same-step new serial−old gap.
Runtime native `eb94e5a52b7fe606069f746b40e06649a5de0c3d`; CUDA reference `8d04573e` adds verified local immutable captures only. Worktree `/home/mengqy/Projects/Bam/maxtext-diag-mudd-origin-gpu`, branch `mqy/diag-mudd-origin-gpu-20261102`.
No training or checkpoint writes. Native owned Spot v6e-1 in europe-west4-a; physicalGPU3 UUIDcbac5000 CUDA reference. Active training resources untouched. Shared owned ledger is in BAM HUnit report directory.

Old checkpoints `gs://newproject-1-llm_projects_europe-west4/log/PileMUDDPythia3BPlusOcdbtv6e/checkpoints/checkpoint_000{43000,53000}/state` (see manifest for literal full URI), actual state exactly label. lm_head shape2560x50432.
Immutable original H from native capture `5a0a03f84b387e61cd743457654d9bcc6c955256`,32x2048 matched training-source texts. Same BF16 H/W output projection is separately materialized before every CE branch; only probability calculation changes.

## Implementations and validation

1. Original Pax softmax: cast BF16 logits FP32; log_softmax; FP32 one-hot/reduction.
2. Remove exactly `logits.astype(jnp.float32)` from real Praxis FullSoftmax; compiled manual expression verified against real original/cast-removal modules. Preserve FP32 one-hot/reduction.
3. MaxText actual training custom-VJP forward, not its different eval-only primal expression.
4. Force BF16 log-probs to an independent physical array; separately renormalize with FP32 logsumexp to diagnose probability-mass rounding. These are controls, not the one-line intervention.

CPU and CUDA small tests pass actual Praxis and actual MaxText train-forward comparisons. Native worker runs the same real-module test before probes.
All64 cases must pass input hashes, state/checkpoint/source identity, exact shape2048, FP32 CE storage, finite values, metadata mean recomputation. FP32 replay mean difference from original native archive<1e−4; every text<.005. No silent tolerance widening. Bootstrap32 texts5000 times, no token independence assumption.43k baseline gap CI crosses0, ratios are descriptive point estimates.

Comparison gap uses original native BF16 full networks with standardized FP32 CE on the SAME cohort:43k .0037365034;53k .0094131082. It is not an instantaneous or smoothed training-curve gap. Frozen forward interventions cannot determine long-term BF16 backward/training effects.

Artifacts native `gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/mudd_spectral_20261102/softmax_precision_v1/`; CUDA `softmax_precision_gpu_v2/`. Reports `training_reports/mudd_spectral_20261102/softmax_precision` in shared Bam workspace; source probe and actual-module test versioned under MaxText.
Report URL http://172.29.142.228:8873/shorttrain/no-self-readout/mudd-spectral/softmax-precision/

## Native TPU results

|checkpoint|FP32 CE|BF16 CE|BF16−FP32 (95% text CI)|serial−old FP32 gap|Δ/gap|
|---|---:|---:|---:|---:|---:|
|43000|2.232380672|2.232197172|-0.000183500 [-0.000206029,-0.000160798]|+0.003736503|-4.911%|
|53000|2.212832854|2.212646568|-0.000186286 [-0.000207022,-0.000166770]|+0.009413108|-1.979%|

Both losses decrease by about.000185, opposite the positive model gap; the effect is almost unchanged43k→53k. This does not explain gap growth of.005677. Actual MaxText train-forward CE equals one-line Praxis cast-removal per token on all64 cases. GPU gives−.00029248/−.00018595; native TPU authoritative. Strict BF16 log-probs+FP32 renormalization has delta−.0000220/+.0000645, showing normalization rounding contributes to the small raw decrease.

CPU, CUDA, and nativeTPU real-module oracle tests PASS.64 cases independently verified. FP32 replay drift is+.00000631/−.00000235 at43k/53k, well within pre-registered1e−4 cohort gate.

All9 owned candidate TPU nodes and queues in the shared BAM/MUDD/softmax ledger absent independently verified; state verified_resources_released. Old12 MUDD-prefix resources also absent, separate audit. Diagnostic GPU processes exited. Serial reference labels43k/53k have internal43001/53001; old labels and internal steps equal.
