# H long-run head-health comparison

## Scope and cadence

Existing online health records, including output-head raw-gradient norm and global
raw_grad_norm median/P95, remain paired against the previous H at every public
report (3k, 8k, 13k, ...). Only frozen-cohort checkpoint probes use the 3k, 13k,
23k, ... calendar. These are distinct measurement/report schedules.

Current RUN: `BamPythia2p8BHUnit300BFastKernelMPreNormQKScale`, runtime
`5a48faaa36989deb1c24747c6314b49b0d8b649f`, trainer
`xd-v5p-256-311103-maxtext`, `us-east5-a`, main workspace
`/home/xd/projects/maxtext`, branch `refactor-bam`.
Previous logical RUN: `BamPythia2p8BHUnit300B`; diagnostic configuration
`BamPythia2p8BHUnit300BFastKernel`, source
`16ee2385da477c6967c09545dd8bbaa32e0eed66`. Its FastKernel changes preserve
the parameter tree and architecture; the diagnostic disables Pallas/Splash
and uses the same native JAX BF16 backbone protocol for both arms.

Official `Pythia_2.8B_2022` has no recorded training-gradient history, globally
or per head. Both W&B segments (`1c9jr6vp`, steps 1–48000; `12vuw5ef`,
47001–143000) set `log_grad_norm=False`. Their full history-key inventories
contain only training loss, learning rate and loss scale among training
metrics; gradient-related timer names are not gradient magnitudes. The local
`official-mha-gradient-availability.json` retains the audited source hashes.
Do not fill the MHA line with another run or checkpoint-probe gradients.

## Frozen-cohort protocol

Reuse the established 32 Pile texts, T2048, seed 20261021, all 65536 token
positions. Cohort SHA256:
`78566a141f1bafeab9873dc5dfaf9aff059fe137ea7fb3a0e088dec20f1bbbbe`.
Source:
`gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/cross_scale_20261025/read_gradient/cohort32.npz`.
Sequence hashes match exactly between the two 3k probes. These are fixed
diagnostic texts, not a reconstruction of historical global-batch-1024 steps.

Primary statistics use BF16-effective post-final-LayerNorm features and head
weights, with FP32 matrix-accumulator output in both arms: P95 absolute
per-token vocabulary-mean logit, and median within-token vocabulary std.
This measures representation geometry at a common precision; it is not a
comparison of each runtime's differently rounded training losses. It must not
be equated with online 128-token-sample quantiles over different training batches.

Runner: `head_health_checkpoint_probe.py`, initial measurement source `5ff4499`.
Only compact moments leave the TPU; the ignored model head/CE is eliminated by
JIT. Each 3k arm restored in about 82 seconds, compiled its forward once, and
processed 32 texts by about 94 seconds including restore. Total was about
169 seconds because the first version unnecessarily compiled a large TPU
quantile sort. Runner `7252f22` moves already-host-resident compact quantiles
to NumPy, checked against production quantile semantics in the pinned CPU env.
The first version's auxiliary BF16/F32 cast-roundtrip control retained excess
precision inside fusion and is excluded. The reported FP32 accumulator metrics
are unaffected. The corrected runner forces BF16 mantissa rounding explicitly.

Reserved compiler `llm-jax-v6e-1-0`, `europe-west4-a`, worker 0, was borrowed
under its existing AOT worker lock. No TPU lifecycle ownership was adopted.
The user narrowed probe scope while it ran: after both 3k results were safely
published, the extra old 44k/44221 work was cancelled. The resulting SIGTERM in
the coordinator log records intentional scope reduction, not a failed 3k result.

## Results and limits

| Frozen checkpoint | Previous H offset P95 | New H offset P95 | Previous H class std median | New H class std median |
|---|---:|---:|---:|---:|
| 3000 | 4.16691 | 4.04348 | 2.46848 | 2.44954 |
| 13000 | 19.61511 | pending | 2.66868 | pending |
| 23000 | 54.10717 | pending | 2.71214 | pending |
| 33000 | 120.76044 | pending | 2.75896 | pending |
| 43000 | 207.72593 | pending | 2.77637 | pending |

Old 13k–43k rows reuse the same-cohort full-token data in
`pythia2p8_h_ce_precision.md`; they were not rerun. The new/old 3k offset
ratio is .97038, class-spread ratio .99233. Actual post-final-LN RMS is
1.18859 versus 1.18653. These early differences are small; they do not
establish improvement in the old run's late offset/gradient-growth regime.

Formal training head-gradient medians, four consecutive 500-step windows
starting at 1k: new .12971/.09613/.08220/.07400; previous H
.13769/.08986/.07506/.07574. The early head-gradient trajectories are close
even though the new run's global gradient norm is lower. A smaller global
norm alone does not demonstrate a repaired output-head gradient.

## Artifacts and reproduction

Local:
`/data0/xd/bam_diagnostics/h-prenorm-qkscale-fp32-launch-20261103/`.
GCS:
`gs://newproject-1-llm_projects_us-east5/log/diagnostics/xd/h_prenorm_qkscale_fp32_head_health_20261103/`.
The local protocol, scoped worker/borrow controllers, `paired-head-health/`,
`paired-head-health-summary.json`, and `old-head-health-reused.json` retain
identities, per-sequence hashes, source paths and primary statistics.
No full checkpoint was downloaded to the workstation. Training continues
under its original sealed runtime.
