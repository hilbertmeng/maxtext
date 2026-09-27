# Joint headwise MLP and matrix scan-carry profile

Implementation: `/data0/xd/rmt-k48-dynamic`, branch `codex/rmt-k48-dynamic`.
Runtime: `d1b1f895a94ef3da401719b47b464d9e6fa50ca4` (pushed).
This is a speed-only, opt-in implementation. Formal MHABudget/LLF trainers
remain on immutable runtime140dd4b.

Question: does preserving the MLP input/output head/value axes improve
whole-step layout, and does it interact with transposing the block-scan carry?

All arms use full18-layer MHABudget: D1200,16x75 heads, T4096, M48x75,
MLP4118, matched generic and41 RMT health metrics, same step13500 schedule,
checkpoint disabled. Parameter count432112752 for every arm.

| Configuration | Headwise MLP | Transposed carry |
|---|---|---|
| RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget | false | false |
| RMTVectorNormMHABudgetTransposedCarryProfile | false | true |
| RMTVectorNormMHABudgetHeadwiseMLPProfile | true | false |
| RMTVectorNormMHABudgetHeadwiseMLPTransposedCarryProfile | true | true |

Headwise W1/Wg[16,75,I] contract the last two activation axes; W2[I,16,75]
produces both axes directly. Initializer fan-in remains1200. Logical sharding
uses embed on the head axis and no logical axis on value, retaining the old
embed partition semantics. This changes parameter shape, so a future live
checkpoint conversion must transpose the scan parameter axis before
reshaping each weight and optimizer leaf. No hot switch is part of this probe.

Transposed scan carry changes only the interlayer ABI from[B,T,48,75] to
[B,T,75,48]; layer computation still consumes canonical M. Mathematical
operations and parameter count remain identical.

CPU tests verify identical initialized weights, forward values, input/weight
gradients (float32 and BF16), and full block scan with converted kernels,
including LLF and combined carry. Full-size abstract parameter audits passed.

Pre-run speed bets versus original: carry-only0%; headwise+0.5%; both+1%.
Prior original VectorNorm carry-only had no benefit; MHABudget's wider MLP
may change fusion. Reshape removal alone does not imply a speedup. Compare
steady steps20-99 and all-device XPlane steps10-14 on one v5p-16 VM; inspect
full-M copies, vector copies, MLP dot layout, and residual-add fusion.

Runner: `.agents/skills/tpu-diagnostics/scripts/run_profile_matrix.sh`.
Compile all four sealed arms on retained EW4a v6e-1 compilers, one job per
host; retain their TPU lifecycle. Short standalone diagnostic v5p-16 candidates
may race EW4b/UC1a. Delete only owned diagnostic resources after verified GCS
artifacts are downloaded locally. Artifact root:
`/data0/xd/bam_diagnostics/rmt-headwise-carry`.

Acquired diagnostic candidates:
`xd-v5p-16-rmt-headwise-europe-west4-b` (EW4b),
`xd-v5p-16-rmt-headwise-us-central1-a` (UC1a).
Compilers: retained EW4a `llm-jax-v6e-1-0` STANDARD and
`llm-jax-v6e-1-1` FLEX_START; one offline CPU compilation per host.
Authoritative runner is `/home/xd/projects/xd_tpu_scripts/run_profile_matrix.sh`,
deployed on tpu-ag with matching SHA256
`e66422df6a6e9aa2d676cd0f7c3b9af42519dfe2b95a9d1b3a00a1434be1ab3c`.

Paired target: EW4b `xd-v5p-16-rmt-headwise-europe-west4-b`.
Matrix ID `headwise-20260926T1210`, label `rmt_headwise_carry`, steps13500,
trace10-14, stop after99. GCS prefix:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/d1b1f89/rmt_headwise_carry`.
UC1a backup was released after the baseline trace/steady window completed;
both its TPU and queued resource were verified absent.

## Results

All four arms completed on the same EW4b v5p-16. Stable speed uses the same
20-99 window (80 samples), inverse mean latency from rounded logs. Device
ms below use the eight primary-worker cores' first fully kernel-covered step.
Only one complete detailed step per core survives the coverage check; later
trace markers without complete leaves are excluded from operator attribution.

| Configuration | Stable step/s | vs original | Device ms | Device throughput vs original |
|---|---:|---:|---:|---:|
| RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget | 0.379875 | +0.000% | 2599.46 | +0.000% |
| RMTVectorNormMHABudgetTransposedCarryProfile | 0.378050 | -0.480% | 2611.94 | -0.478% |
| RMTVectorNormMHABudgetHeadwiseMLPProfile | 0.379950 | +0.020% | 2598.58 | +0.034% |
| RMTVectorNormMHABudgetHeadwiseMLPTransposedCarryProfile | 0.378025 | -0.487% | 2611.17 | -0.448% |

No practical improvement: retain the original runtime in formal training.
The headwise +0.5% and combined +1% bets failed. Headwise reduced data-format
kernel time by13.6ms but convolution fusion grew20.2ms, including about13.2ms
in W2 backward scopes. Carry-only reduced formatting1.5ms but added14.7ms
loop-fusion work. Its scan-buffer update is2.37->17.20ms on the first core.
Source labels can move between transpose/add scopes; those renamings cancel
and are not new operators. Joint layout does not recover this overhead.

The useful local result is the real reduction in data formatting. A focused
next probe could keep W1/Wg headwise and restore W2's flat kernel to test
whether that saving survives without the W2 backward penalty. The present
results do not establish a benefit for this untested hybrid. No additional
training experiment or live checkpoint conversion was performed.

Canonical original fine-grained table is in main `bam_exp_memo.md`, section
**RMT MHABudget full-layer main profile (2026-09-26)**. Parameters432112752;
per-layer block parameters17,281,712. Forward block theory15.99553 W_Q, versus
matched MHA15.62667 (+2.36%), excluding LM head and lower-order elementwise
work. Matrix-flow reads/writes/vector norms/health consume1102.03ms (42.39%);
SwiGLU440.32ms (16.94%). All copies288.48ms (11.10%) overlap those scopes.

Reproduce attribution with `analyze_rmt_main_profile.py TRACE --source
MaxText/layers/rmt.py --output OUTPUT`; render audited theory/table with
`render_rmt_mha_budget_main_profile.py ARTIFACT_ROOT`. Sources are in the
implementation worktree's `experiments/bam_llama2_medium/`. The original
source must match runtime d1b1f89 for health-source attribution.

All four primary XPlanes and trace JSONs were verified nonempty in GCS and
downloaded directly to the local artifact root. Both diagnostic TPUs and
queued resources were deleted and verified absent. The two retained compiler
TPUs remain outside cleanup; the three formal140dd4b training runs continue.
