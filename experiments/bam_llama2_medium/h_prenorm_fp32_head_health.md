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

## Results and limits

| Frozen checkpoint | Previous H offset P95 | New H offset P95 | Previous H class std median | New H class std median |
|---|---:|---:|---:|---:|
| 3000 | 4.16691 | 4.04348 | 2.46848 | 2.44954 |
| 13000 | 19.61511 | 4.70240 | 2.66868 | 2.61735 |
| 23000 | 54.10717 | 4.94314 | 2.71214 | 2.61908 |
| 33000 | 120.76044 | 5.12062 | 2.75896 | 2.64078 |
| 43000 | 207.72593 | pending | 2.77637 | pending |

Old 13k–43k rows reuse the same-cohort full-token data in
`pythia2p8_h_ce_precision.md`; they were not rerun. The new/old 3k offset
ratio is .97038, class-spread ratio .99233. Actual post-final-LN RMS is
1.18859 versus 1.18653. These early differences are small; they do not
establish improvement in the old run's late offset/gradient-growth regime.

At 13k the common-offset ratio is .23973 (-76.03%), while the class-spread
ratio is .98077 (-1.92%). Actual post-final-LN RMS is 1.28927 versus old
1.32817 (-2.93%). Overall amplitude reduction cannot explain the much larger
offset suppression. The offset/class-spread ratio is 1.80 versus old 7.35.
This establishes early common-mode improvement for the combined recipe; it
does not yet establish improved post-30k loss convergence. The 13k probe
(`7252f22`) took 89.42 seconds, traced the forward once, and released the FLEX
worker lock. Its explicit BF16 rounding control differs from the accumulator
as expected; primary results remain on the common FP32-accumulation path.

At 23k, common offset is 4.94314 versus old 54.10717 (-90.86%); class spread
is 2.61908 versus 2.71214 (-3.43%). Actual post-final-LN RMS is 1.29873
versus 1.47613 (-12.02%). New 13k–23k RMS grows only .73%, while old grows
11.14%; new class spread is nearly unchanged. The same-cohort sample hashes
match, and the probe traced once and completed in 89.56 seconds.
Same-stage formal head-gradient median is .05010 versus .05775. Its roughly
13% reduction is close in scale to the roughly 12% cohort-RMS reduction; this
is a clue about amplitude, not a same-batch gradient factorization or proof
of lower gradient noise. The checkpoint-health comparison is rendered as
one figure with three side-by-side subplots, preserving the measured 3k anchor.

At 33k, common offset is 5.12062 versus old 120.76044 (-95.76%); class spread
is 2.64078 versus 2.75896 (-4.28%). Actual post-final-LN RMS is 1.33077
versus 1.84153 (-27.74%). New 23k–33k RMS grows 2.47%, versus 24.75% for
old H. Same-stage formal head-gradient median/P95 is .04899/.05813 versus
old .06303/.08020; the new head gradient remains nearly flat while the old
head rises. This extends the evidence into the old run's post-30k drift
regime. These measurements establish suppression of common-mode growth and
much slower feature-amplitude growth for the combined recipe, not an
isolated causal attribution to FP32 logits. The probe used all 65536 tokens,
matched all 32 sequence hashes, traced its forward once, completed in 94.85
seconds, and released the reserved FLEX worker lock.

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
