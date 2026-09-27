# RMT attention-write / MLP-read scheduling

**Retain the original implementation.** Merging reads alone changes device throughput
by -0.18%; mapped chunks lose 11–27%; static-unrolled 512/1024 lose 4.04%/12.30%.
The pre-run +5% best-case bet was wrong. This rules out these JAX scheduling
implementations as optimizations, not the possibility of a fused write/read kernel.

Model: `RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32`.
18 layers, six 3-layer block scans, D1200, 16 heads × 75, MLP4078,
432,119,360 parameters, T4096, configured per-device batch16. Generic training
health ON; extra RMT dynamic health OFF. All measurements use the same EW4b
v5p-16, dataset and 13,500-step schedule, with XPlane steps10–14 and stable logs
after profiling. This is a scheduling diagnostic, not a retraining experiment.

Implementation: `/data0/xd/rmt-write-read-chunks`, branch `codex/rmt-write-read-chunks`.
Parent `ffb40f2d`; mapped/merged runtime `3617fbb019504b966c1fdf680aea1f42699d9a50`;
static-unrolled runtime `49884422f695029bc8cc0d844a6232068d6d7a5d`.
Unrelated concurrent main-worktree/Pallas edits are not included.

## Changes tested

- `rmt_mlp_merge_reads`: combine the full48→16 static MLP read and zero-padded
  tail32→8 compression into one48→24 contraction, preserving trainable parameters.
- `rmt_write_read_chunk_size`: 0 (no chunks),64,128,256,512,1024,2048.
  One unit performs attention static/dynamic writes and residual additions,
  static/C8 reads, first16-row vector RMS, and the dynamic MLP read/key/gate.
  Attention's existing query chunk256 is unchanged. Write-address projections
  and MLP dense layers stay outside the unit. Updated M still crosses the layer
  boundary; no algebraic reassociation of the two outer writes is introduced.
- `rmt_write_read_chunk_unroll`: a Python loop with static slices/concatenation,
  tested at512/1024, replacing `lax.map` while retaining the same arithmetic.
  This follow-up tests whether scan bookkeeping explains the mapped regression.

## Paired results

The original14-arm sweep and repeat control:

| Configuration | Device step (ms) | Stable step/s | Device throughput vs paired control |
|---|---:|---:|---:|
| `RMTWriteReadControlProfile` | 2579.69 | 0.3848 | +0.00% |
| `RMTWriteReadSplitC512Profile` | 2922.27 | 0.3398 | -11.72% |
| `RMTWriteReadSplitC1024Profile` | 2900.55 | 0.3425 | -11.06% |
| `RMTWriteReadSplitC256Profile` | 2970.08 | 0.3343 | -13.14% |
| `RMTWriteReadSplitC2048Profile` | 2906.01 | 0.3420 | -11.23% |
| `RMTWriteReadMergedProfile` | 2584.47 | 0.3833 | -0.18% |
| `RMTWriteReadMergedC512Profile` | 2922.90 | 0.3400 | -11.74% |
| `RMTWriteReadMergedC1024Profile` | 2898.98 | 0.3420 | -11.01% |
| `RMTWriteReadMergedC256Profile` | 2973.23 | 0.3338 | -13.24% |
| `RMTWriteReadMergedC2048Profile` | 2909.37 | 0.3410 | -11.33% |
| `RMTWriteReadSplitC128Profile` | 3082.81 | 0.3220 | -16.32% |
| `RMTWriteReadMergedC128Profile` | 3084.24 | 0.3220 | -16.36% |
| `RMTWriteReadSplitC64Profile` | 3544.77 | 0.2800 | -27.23% |
| `RMTWriteReadMergedC64Profile` | 3528.75 | 0.2820 | -26.89% |
| `RMTWriteReadControlProfile` (repeat) | 2579.91 | 0.3845 | -0.01% |

Static-unrolled follow-up, including a fresh same-commit control:

| Configuration | Device step (ms) | Stable step/s | Device throughput vs paired control |
|---|---:|---:|---:|
| `RMTWriteReadControlProfile` | 2579.95 | 0.3845 | +0.00% |
| `RMTWriteReadUnrolledC512Profile` | 2688.54 | 0.3690 | -4.04% |
| `RMTWriteReadUnrolledC1024Profile` | 2941.76 | 0.3370 | -12.30% |

Stable log throughput and device timing agree on the direction. The repeated
original exactly reproduces same-step losses. No candidate is credited with a
training-loss improvement from these short initialization profiles.

## Kernel evidence

Use raw `.xplane.pb`, not trace JSON: small chunks exceed the approximately
one-million-event JSON cap. `analyze_rmt_write_profiles.py` reads the pinned
protobuf definition (`XPLANE_PROTO` overrides its location), excludes nested
while containers, and checks leaf overlap/coverage. Below are disjoint kernel
categories on the first complete core; the result tables average train-step
spans over all devices represented in the primary trace.

| Kernel category (ms) | Original | Merged/no chunk | Mapped1024 | Mapped64 | Unrolled512 | Unrolled1024 |
|---|---:|---:|---:|---:|---:|---:|
| convolution fusion | 1356.65 | 1342.45 | 1358.33 | 1525.66 | 1435.10 | 1704.98 |
| data formatting | 299.19 | 303.17 | 421.46 | 532.38 | 316.21 | 311.64 |
| loop fusion | 614.44 | 627.48 | 712.81 | 787.22 | 643.03 | 635.40 |
| dynamic-update-slice | 162.15 | 162.41 | 189.87 | 315.78 | 142.34 | 142.31 |
| broadcast | 16.83 | 17.15 | 74.77 | 102.23 | 15.21 | 15.22 |
| collective-permute-start | 28.15 | 28.11 | 34.11 | 140.68 | 30.88 | 27.77 |
| collective-permute-done | 1.85 | 1.85 | 2.11 | 30.50 | 3.09 | 2.04 |

Mapped1024 adds roughly123ms of data formatting,98ms of elementwise fusion,
58ms of broadcast and28ms of output updates; matrix-multiply kernel time barely
changes. At64, matrix multiplication and communication also become slower.
Merging alone saves about14ms in matrix-multiply kernels, which is offset by
extra elementwise/layout work. Merely enclosing operations in a token chunk
does not establish on-chip producer/consumer fusion.

Static unrolling removes most mapped-loop bookkeeping, but still increases
convolution-fusion time by about 78 ms at 512 and 348 ms at 1024. Thus mapped
loop overhead explains only part of the failure. A future fused kernel must
prove reduced HBM traffic without sacrificing projection/GEMM efficiency; merely
changing chunk size or concatenating projections did not accomplish this.

## FLOPs and numerical checks

One W_Q projection at D1200 is1,440,000 MAC/token. Separate linear reads cost
`(48*16+32*8)*75=76,800` MAC; merged costs `48*24*75=86,400`: an extra9,600 MAC,
0.006667 W_Q per layer (+12.5% for these contractions before zero elimination).
Chunking changes no parameters and removes no nominal FLOPs. Its intended
benefit is reduced memory traffic, which must be measured rather than assumed.

Full18-layer shape audits retain432,119,360 parameters. Focused per-layer CPU
checks preserve exact parameter names/seeded values and exercise active dynamic
keys, health outputs and all parameter gradients. At D1200/head75/MLP4078,
FP32 mapped-gradient relative L2 error is below3e-7; BF16 is about0.24%.
Unrolled split reads reduce the tested BF16 error to0.0144%. BF16 training
trajectories are nevertheless not bit-identical. Early loss differences have
both signs; these diagnostics establish no final loss benefit or penalty.

**Unrolled1024 additionally fails the training-equivalence screen:** step30 loss
10.852290 vs control 9.484537 (512: 9.867630); accuracy remains near random.
Its step0 loss already differs (10.852971 vs 10.852016), while repeated original
controls reproduce exactly. Data token counts, schedule and sealed configurations
match; logs contain no numerical exception. The focused checks use four tokens
and one layer, even at full width, and do not validate the full18-layer TPU
training trajectory. A compiler/layout-dependent numerical issue remains
unresolved; do not explain it away as harmless BF16 noise or use this variant
for training. Its throughput is reported only as a failed candidate. Full logs
and early loss samples are retained in `train-logs/` and `unroll/loss-audit.json`.

## Reproduction and artifacts

- Local artifacts: `/data0/xd/bam_diagnostics/rmt-write-read-chunks`;
  mapped `summary.json`, `verified_artifacts.json`, `table.md`, `throughput.png`,
  raw `profiles/`, CPU logs under `cpu/`; follow-up under `unroll/`.
- AOT root: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/rmt-write-read-chunks/aot/{3617fbb,4988442}`.
- Primary profiles: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/3617fbb/rmt_wr_chunks`
  and `.../4988442/rmt_wr_unroll`.
- Authoritative runner: `/home/xd/projects/xd_tpu_scripts/run_profile_matrix.sh`;
  `AOT_ROOT` as above, `PROFILE_STEPS=13500`, `PROFILE_DONE_STEP=30`.
  `summarize_rmt_write_read_profiles.py ROOT` regenerates the table/plot.
- Target: `xd-v5p-16-rmt-wr-chunks-ew4b`, europe-west4-b. Temporary compilers:
  `xd-v6e-1-rmt-wr-chunks-ew4a` and `xd-v6e-1-rmt-wr-chunks-b-ew4a`, europe-west4-a.
  Both reserved compilers and all formal training RUNs were untouched.

## Execution audit

Concurrent AOT processes on one host conflicted on libtpu's process lock despite
CPU affinity; subsequent jobs use one compiler per worker. Spot compiler
preemptions did not interrupt the target's original sweep. The unrolled follow-up
compiled on idle target workers0/1 after the first matrix completed.
An unguarded120-second preflight was corrected in `xd_tpu_scripts` commit
`e5947dd`: SSH preflight and compiler upload now have lifecycle guards and use
internal IP. Mocked preemption tests verify that neither failure proceeds to
compilation. Raw XPlane parsing fixes the misleading partial JSON attribution.
