# Pythia 2.8B output-head gradient and precision diagnostics

Implementation: `/home/xd/projects/maxtext`, `refactor-bam`. These are read-only
checkpoint probes, not changes to the training RUN. The existing FLEX compiler
`llm-jax-v6e-1-0`, `europe-west4-a`, worker 0 was borrowed under its AOT worker
lock; no TPU lifecycle ownership was adopted. All jobs finished, released that
lock and retained the compiler.

## Identities and measurement scope

| Arm | Full configuration / runtime | Checkpoints |
|---|---|---|
| Long H | `BamPythia2p8BHUnit300BFastKernel`, logical RUN `BamPythia2p8BHUnit300B`, `16ee2385da477c6967c09545dd8bbaa32e0eed66` | 33000, 41000, precision at 43000 |
| Short H | `BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne`, `6c5dbfe9df446b1c6f6421616f7af97156c7f9a5` | 8000, 16000, 21549 |
| Old BAM | `BamPythia2p8BSotaIndependentR768Pallas`, `1ac11833cf6a62cb5a6b0f9ae58e2f4c0520f551` | 33000, 37000 |
| Serial Mudd | `MuddPythia3BSerialMLPKeepActivations`, actual RUN `MuddPythia3BSerialMLP`, `9cdcbd02d93437b3e795a93cb787e608e39a9fff` | existing matched-cohort probes at 33000, 43000, 53000 |

Use the same 32 Pile texts, T2048, seed 20261021; cohort SHA256
`78566a141f1bafeab9873dc5dfaf9aff059fe137ea7fb3a0e088dec20f1bbbbe`.
The cohort and per-sequence hashes are retained with each result. Source:
`gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/cross_scale_20261025/read_gradient/cohort32.npz`.

`head_gradient_amplitude_probe.py` at diagnostic commit `f88bab481ed568c20d0840e57a8d9db0d2075e38`
captures native BF16 hidden states **after** final LayerNorm, with Pallas/Splash
disabled and highest matmul precision. Restored arrays are explicit JIT inputs;
same-shape checkpoints reuse one trace. It then computes the FP32 conditional
head gradient using BF16-effective head weights. This 32-text mean-gradient
norm must not be equated with formal global-batch-1024 training records.

## Input amplitude versus recorded training gradients

| Arm | Early actual post-LN RMS | Late actual post-LN RMS | Change |
|---|---:|---:|---:|
| Long H, 33k to 41k | 1.84153 | 2.30264 | +25.04% |
| Old BAM, 33k to 37k | 1.68311 | 1.84891 | +9.85% |
| Serial Mudd, 33k to 53k | 1.77314 | 2.85161 | +60.82% |
| Short H, 8k to 21.549k | 1.26429 | 1.47194 | +16.42% |

Short H's intermediate 16k RMS is 1.36997. Nevertheless its formal training
head-gradient median falls from .05565 near 8k to .04210 near 21.549k (-24.35%).
Long H's corresponding formal medians rise from .06425 near 33k to .08551
near 41k (+33.10%). Dividing these formal medians by independently measured
cohort RMS gives approximate changes of -35.02% and +6.45%, respectively.
These are cross-sample proxies, not an exact factorization of historical
training batches. The short and long schedules also represent different
relative training stages and use different Q/K logit multipliers.

Thus increasing input amplitude is a real amplifier but is **not sufficient**
to explain the direction of the formal head-gradient trend. The conditional
32-text mean is largely dominated by per-text variance; its almost flat Short
H norm cannot substitute for the decreasing batch-1024 training norm.
Prediction-error/feature correlation and minibatch cancellation matter.

LayerNorm produces `h = gamma * z + beta`, not unit-RMS h. Recovered Long H z
mean square is .9999845 at 33k and .9999909 at 41k: normalization itself works.
An exact six-order Shapley decomposition of the increase in **h mean square**
assigns Long H's increase 74.05% to gamma, 2.90% to beta and 23.05% to channel
moments. For Short H the corresponding shares are 116.10%, 5.17%, -21.27%:
feature redistribution partly offsets affine amplification. These percentages
describe input energy, not causal shares of head-gradient growth.

Parallel Mudd is a useful negative control: the existing matched 33k/43k/53k
probes show post-LN RMS 1.14664/1.13547/1.12543 and conditional head gradients
.18015/.17427/.17312. It does not show the same growth. Original official MHA
and parallel Mudd logs lack leaf-level head-gradient histories; do not invent
them from global gradient norms.

## 43k frozen-head precision comparison

Runner: `head_precision_probe.py`, diagnostic commit
`efc871fba1987cc4e064d59ec6b5f40ab372709b`; two focused CPU tests pass.
Restore original FP32 master `params.params.decoder.lm_head.logits_dense`
from `gs://newproject-1-llm_projects_us-east5/log/BamPythia2p8BHUnit300B/checkpoints/43000/items`.
All arms use identical captured BF16 post-LN hidden values and targets.
FP32 projection uses highest precision; only the head and CE precision change.
The primary CE uses the sealed runtime's `_cross_entropy_with_logits_fwd`,
the custom-VJP forward actually executed under training differentiation.

| Unembedding | Softmax / CE | Mean loss | Delta versus full FP32 |
|---|---|---:|---:|
| BF16 | BF16 | 2.210837837 | +.001995541 |
| BF16 | FP32 | 2.211083505 | +.002241209 |
| FP32 | BF16 | 2.211046617 | +.002204321 |
| FP32 | FP32 | 2.208842296 | 0 |

The paired 32-text bootstrap 95% interval for full BF16 minus full FP32 is
[.001550279, .002449683]. Single-stage upgrades do not yield the same result;
rounding/cancellation is not additive. This establishes a numerical difference
on one frozen checkpoint, not the loss benefit of retraining in FP32.

An important reporting trap: the same CE function's undifferentiated primal
uses `logits - logsumexp(logits)`, whereas its training forward uses max-shift
then `shifted - log(sum(exp(shifted)))`. They agree in real arithmetic but not
BF16. Here plain BF16 evaluation gives 2.199764118, lower than full FP32 by
.009078179, a rounding artifact. Do not compare that value to the formal
training path and call the difference a model improvement.

## Reproduction and artifacts

Local artifact root: `/data0/xd/bam_diagnostics/h-unit-300b-launch-20261101`.
GCS root: `gs://newproject-1-llm_projects_us-east5/log/diagnostics/xd/h_head_amplitude_20261102/`.
Per-checkpoint captures are under `h/`, `old_bam/`, `h_shorttrain/`, and
`h_precision_43000/43000/`; each has `complete.json`, sequence records and
`head-input-NNN.npz`. Head master weights were read on the TPU VM; no full
checkpoint was downloaded to the workstation.

Key local result files: `head-precision-43000.json`,
`actual-training-head-gradient-amplitude-proxy.json`,
`final-ln-amplitude-decomposition.json`, `native-head-replay.json`,
`head-input-rms-ratio-bootstrap.json`, `parallel-mudd-mha-head-gradient-evidence.json`.
The replay controller/protocol and logs are retained in the same root.

```bash
python experiments/bam_llama2_medium/head_precision_probe.py \
  --runtime-dir /tmp/maxtext-offline-aot-16ee2385da47 \
  --inputs /tmp/xd-h-head-amplitude-20261102/h_precision_43000/43000 \
  --checkpoint-items gs://newproject-1-llm_projects_us-east5/log/BamPythia2p8BHUnit300B/checkpoints/43000/items \
  --output /tmp/xd-h-head-amplitude-20261102/head-precision-43000.json
```
