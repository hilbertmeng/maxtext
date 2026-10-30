# Pythia2.8B strongest combination: dynamic local-O gamma .2

User confirmed configuration/code only; no2.8B training or AOT resource request.
Parent: BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80
New: BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80LocalODynamicNormScale020
Development: /data0/xd/local-o-dynamic-norm-init, branch codex/local-o-dynamic-norm-init.
Artifacts: /data0/xd/bam_diagnostics/pythia2p8-dynamic-o-scale020.

The parent is the trained R640, independent-VO, strict-history negative-diagonal,
full80 NoPE+20 RoPE combination. Full100-D QKNorm before RoPE, GELU serial MLP,
embedding.006/BAM.006 initialization and ordinary transformer Pythia initialization
all inherit. Body31 layers single-layer scan plus explicitL31, ordinary7808/private2944,
terminal9984, minimal remat, tunedSplash, Pallasv8t read/write and60MiB scopedVMEM.
Original28000-step LR/optimizer/batch/data/checkpoint plan inherits without a restore.

Change only learned dynamic-O address scale initialization to.2. Each layer has
32heads*64coordinates, after address RMSNorm/dynamic gating and before adding staticO.
StaticO, Q/K/V, existing dynamic/static sigmoid gates and M shifts remain unchanged.
The scale is unconstrained, not sigmoid-reparameterized and not a1+gamma offset.

Added32*32*64=65536 parameters (.01 W_Q total, .00236% of parent), exempt from MLP
repayment. Exact total2775256416 versus parent2775190880. All MLP widths unchanged
and128-aligned; no per-layer width override. Existing scale WD exclusion applies.

Health producer is the existing record_training_health_metrics=True path, with
record_internal_nn_metrics=False and broad BAM statisticsOFF. It produces32 gamma
parameter L2 norms plus32 raw-gradient L2 norms. Body31 and explicit tail are covered.
For readable layer RMS divide bysqrt(2048)=45.254834, not Medium sqrt(512).
Gamma-init.2 therefore has initial layer L2 norm9.050967 and RMS.2. Parameter RMS
is not a signed mean, actual O/context output ratio or causal attribution.

Pallas support reuses the trained881f991 implementation and its2655622 O-mask
prerequisite, keeping parent behavior when scale/mask are disabled. This ports the
existing dynamic-O forward and dgamma VJP, not a new kernel algorithm. Main's old
Medium reference configurations retain their native flags; running Medium single-gate
experiments remain sealed at2f36e3a and are not hot-switched.

Validation: exact full-model abstract parameter budget; actual32-layer generic
parameter/raw-gradient health export and WD exclusion;32-head M80x64/QK100 attention
native versus Pallas outputs and all gradients with packed segments, M-shift and
nonzero staticO/perturbed gamma (max relative gradient error2.53e-6); full47 BAM
regressions. Extra finite-BF16/validation checks are saved alongside these receipts.
The geometry test respects native[K,V] versus fused[V,K] carry layout and the fused
attention's deferred-writer return. Formal configuration/health checks passed directly;
geometry fixture setup was corrected before its successful result.

A future training launch must compile this new class's target-topology AOT; the
parent AOT has a different parameter tree and is not reused. No target-TPU AOT or
hardware timing claim is made by these CPU interpret checks.

## 2026-10-30: frozen 16k readout diagnostics

Diagnostic source `45048e29610097c5a6dc3e78047022864176175b`, branch `mqy/diag-norm020-20261030`, worktree `/home/mengqy/Projects/Bam/maxtext-diag-norm020`.
Norm020 training source `5c3a995444897bd8a148a2e888a8c0e1d6721c6b`; Full training source `44141a7c22c395255a6d0d58fff8e103d9a3fe87`. Restore parameters only from each class's `gs://newproject-1-llm_projects_us-east5/log/CLASS/checkpoints/STEP/items`.

Reuse the original32 fixed Pile training-source texts, seed20261021, first2048 inputs/next-token targets, batch4. RMS uses64 fixed query positions>=64; CE scores all2048 targets. No training, optimizer restore or checkpoint writes. All three checkpoints use native BF16 logits with explicitly FP32 log-softmax/CE. The older report used BF16-rounded CE; Full12k separately replays that score against its historical capture (max mean-per-text drift0.00195). Do not combine the abandoned non-`fp32` artifacts with this run.

155 interventions/checkpoint: global branch deletion; whole32-head deletion of each layer's MHA_o or M_o; separate vector-only/write-only controls; predeclared joint bands1–8/9–16/17–24/25–32/17–32. Combined changes vector output and current direct attention M-write content; vector-only preserves that call's computed direct attention write. Downstream computation and FFN writes stay live. L32 has no attention M-write; combined/vector losses match exactly. Ablation deltas use each chunk's same-graph neutral;4000 paired whole-text bootstrap resamples. Single-layer effects are nonadditive.

| Configuration class | Step | Native CE | Post-W_o O/A L17–24 | O/A L25–32 | Mean single-layer MHA deletion ΔCE L17–24 | L25–32 |
|---|---:|---:|---:|---:|---:|---:|
| BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80 | 12000 | 2.32125 | 1.187 | 2.861 | 0.01498 | 0.00369 |
| BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80 | 16000 | 2.27966 | 1.195 | 3.057 | 0.01543 | 0.00367 |
| BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80LocalODynamicNormScale020 | 16000 | 2.27390 | 0.866 | 1.784 | 0.02010 | 0.00973 |

Partial middle/upper-layer recovery is supported by both Full12k and same-step Full16k controls. Largest increases are atL17,L24–26,L28. Final layers remain weak; L32 MHA deletion is0.00402 versus0.00476/0.00486 in Full12k/16k. Vector-only mean MHA deletion also rises: L17–24 Full16k0.00556→Norm0.00778; L25–32 0.00250→0.00447. Joint vector-only removal across17–32 rises0.21355→0.39016 (paired difference95%CI[0.16060,0.19293]).

M_o still matters despite smaller RMS. ForL25–32 combined joint removal, Full16k0.59691→Norm0.68064; vector-only0.57794→0.53497. The difference between these scopes is not an additive write-channel attribution. Norm's native CE is0.00576 lower on these32 texts than Full16k (paired95%CI[-0.01107,-0.00049]); this is not held-out validation or training-seed uncertainty. Learned gamma global RMS0.19946/mean0.19742; independent CPU partial restore matches captured coordinates. This is an address-coordinate scale, not a fixed0.2 output multiplier.

Completed on task-owned Spot v6e-1 workers in europe-west4-a: r2 produced Full12k and Full16k capture; r4 produced Norm16k capture/cases0–7; r5 resumed Norm and produced Full16k cases. Two physical preemptions were recovered with immutable per8-case GCS chunks. The final manifest names the final worker, not the sole producer of all artifacts. All13 exact owned nodes/queues (including unused candidates) audited absent. No on-demand TPU was used. No writes under ag2 `/home/lishengping/xd/`.

Artifacts and invocation:

- Canonical raw prefix: `gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/local_o_norm020_20261030/fp32`; manifests/configs at its root, `full/{12000,16000}` and `norm020/16000` each contain capture plus20 chunk pairs and completion marker. `report/` is the complete HTML export; `scripts/` stores task helper snapshots.
- Local raw: `/data0/mengqy/bam_diagnostics/local_o_norm020_20261030`; analysis/verification/scripts: `/home/mengqy/Projects/Bam/training_reports/local_o_norm020_20261030`.
- Runner: `MaxText/norm020_probe.py --manifest /tmp/norm020_job/arms.json --family norm020 --training-config /tmp/norm020_job/norm020.train.config.txt --cohort-reference /tmp/norm020_job/reference.json --output /tmp/bam_stream_probe/norm020_results --gcs-output PREFIX`; repeat with `--family full` and matching training config. Run from source root with `JAX_PLATFORMS=tpu PYTHONPATH=MaxText`; detached `worker_jobs.py` stages exact inputs and thread settings.
- Pinned JAX/jaxlib0.8.1, Flax0.12.1, Optax0.2.6, NumPy2.1.3. Tiny native BF16 full-QK/gamma/.2 CPU checks pass; explicit FP32 CE matches an independent NumPy reference. All465 interventions verified locally, identical cohort hashes, shifted targets, resumed chunk baselines, finite arrays, first-token no-self zero and independent gamma restore. Native/neutral drift<0.00168 nats per-text; projection relative RMS<0.00240.
- HTML: `http://172.29.142.228:8873/shorttrain/no-self-readout/norm020/`; all32-layer tables support model/scope switching. Five PNG/PDF figures include an upper16-layer zoom. Desktop1440/mobile390 browser checks pass, including selected-model numeric cells and all images.
