# H long-run head-health comparison

## Scope and cadence

Existing online health records, including output-head raw-gradient norm and global
raw_grad_norm median/P95, remain paired against the previous H at every public
report (3k, 8k, 13k, ...). Only frozen-cohort checkpoint probes use the 3k, 13k,
23k, ... calendar. These are distinct measurement/report schedules.

Completed RUN: `BamPythia2p8BHUnit300BFastKernelMPreNormQKScale`, runtime
`5a48faaa36989deb1c24747c6314b49b0d8b649f`, trainer
`xd-v5p-256-311103-maxtext`, final zone `us-central1-a`, main workspace
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
comparison of each runtime's differently rounded training losses. Online
runs using the same sampling and window definitions can be compared across
training batches; identical batches are not a prerequisite. The fixed cohort
reduces comparison noise and supplies the missing old-run measurements here. It must not
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

## Final results and limits

The RUN completed 143000 updates. Each row below is an actual restored-checkpoint
measurement on the identical cohort. The old RUN ends at 43k; absent old values
remain absent. New 143k matched 133k runtime, runner/module hashes, parameter
count, sequence hashes, all 65536 positions and single-forward protocol; restore
plus measurement took 108.83 seconds on the retained FLEX worker.

| Checkpoint | Old offset P95 | New offset P95 | Old class std | New class std | Old post-ln_f RMS | New post-ln_f RMS |
|---|---:|---:|---:|---:|---:|---:|
| 3000 | 4.16691 | 4.04348 | 2.46848 | 2.44954 | 1.18653 | 1.18859 |
| 13000 | 19.61511 | 4.70240 | 2.66868 | 2.61735 | 1.32817 | 1.28927 |
| 23000 | 54.10717 | 4.94314 | 2.71214 | 2.61908 | 1.47613 | 1.29873 |
| 33000 | 120.76044 | 5.12062 | 2.75896 | 2.64078 | 1.84153 | 1.33077 |
| 43000 | 207.72593 | 5.28501 | 2.77637 | 2.65497 | 2.42650 | 1.36536 |
| 53000 | — | 5.39617 | — | 2.66604 | — | 1.42299 |
| 63000 | — | 5.51140 | — | 2.67987 | — | 1.49065 |
| 73000 | — | 5.58190 | — | 2.70033 | — | 1.57507 |
| 83000 | — | 5.66319 | — | 2.71604 | — | 1.65320 |
| 93000 | — | 5.72199 | — | 2.72423 | — | 1.73295 |
| 103000 | — | 5.77462 | — | 2.74175 | — | 1.81918 |
| 113000 | — | 5.78069 | — | 2.74721 | — | 1.89591 |
| 123000 | — | 5.78723 | — | 2.74906 | — | 1.95697 |
| 133000 | — | 5.80262 | — | 2.75387 | — | 2.01425 |
| 143000 | — | 5.80668 | — | 2.75184 | — | 2.05662 |

At 43k, offset P95 is 5.28501 versus 207.72593 (-97.46%), while class spread differs
only -4.37% and post-ln_f RMS -43.73%. The much larger relative common-offset
reduction cannot be explained by amplitude shrinkage alone. By 143k, offset
P95 is 5.80668 and class std 2.75184; late common-mode growth has essentially
flattened. Actual ln_f output RMS still rises, but its 10k increments decelerate:
103k–113k 4.22%, 113k–123k 3.22%, 123k–133k 2.93%, 133k–143k 2.10%.
The earlier accelerating interval does not describe the terminal trajectory.

Formal training head-gradient median/P95 by consecutive 1k windows 138k–143k:
.05516/.05819, .05524/.05856, .05595/.05885, .05616/.05992,
.05579/.05945. Global gradient medians over those windows are
.12096, .12007, .12159, .12126, .12258. Both retain a mild upward drift, with no
old-style output-head surge. Online offset and class-spread windows are flat.
The old head-gradient rise after 30k and its common-logit drift do not recur in
the combined recipe. These measurements do not isolate the effects of M
pre-norm, learned scales, and FP32 logits from one another.

## Artifacts and reproduction

Artifact root:
`/data0/xd/bam_diagnostics/h-prenorm-qkscale-fp32-launch-20261103/`.
GCS:
`gs://newproject-1-llm_projects_us-east5/log/diagnostics/xd/h_prenorm_qkscale_fp32_head_health_20261103/`.
`paired-head-health-summary.json`, per-step health/protocol JSON, and the final
three-panel figure provenance retain all measured values and source hashes.
`online-health-143000.json` and the cumulative-health manifests contain the
training-batch gradient/logit histories. Sealed event files are fingerprint
cached; report generation does not revalidate every old file.

Later fixed-cohort probes 123k/133k/143k borrowed the retained FLEX worker under
its AOT lock; they did not own TPU lifecycle and left the worker retained.
Some earlier probes used task-owned spot diagnostics; service interruptions
of those workers must not be described as FLEX preemption. No full checkpoint
was downloaded locally. Training/resource closeout and full comparisons:
[h_prenorm_qkscale_fp32_300b.md](h_prenorm_qkscale_fp32_300b.md).
