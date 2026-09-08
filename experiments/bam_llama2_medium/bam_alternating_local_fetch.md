# Alternating LocalO / fetched-M read

Code branch: `codex/bam-alternating-local-fetch`; worktree:
`/data0/xd/bam-alternating-local-fetch`. Main repository contains ledger-only configuration records; runtime changes stay in this worktree.
Runtime commit: `a77952e98e28eb8508c7c8a9ec16982c37f9b72d`.
AOT root: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/local-fetch/a77952e98e28eb8508c7c8a9ec16982c37f9b72d/`.
Each `<CLASS>.pickle.manifest.json` records the source, compiler/environment hashes and topology.

## Contract

- Full-24 Medium V2 C256, float32_logits=False, original fixed RMS-gate amplitude.
  Current CleanControl supplies the common optimizer; no GELU/learned mix-scale changes.
- Layer numbering starts at zero: LocalO, fetch, LocalO, fetch, ... . Every layer keeps
  LocalQK, attention-output M write, and the standard MLP. Fetch layers keep diagonal-one.
- LocalO reads incoming local M before this layer's write. The compression factor changes
  only LocalO's read view/output width, not the full matrix carried between layers.
- Independent LocalV is rank 2, bilateral, full-M, legacy signed head/rank RMS mixing,
  with zero-init keys/pre-RMS bias and .005 gates. It is present only in LocalO layers.
  Its key/gate/head-mix projections are packed, and its read is injected into V before AV.
- SharedRead uses the compressed or full LocalO per-head read once with ungated normalized keys;
  independent O/V row/column gates route that answer to both destinations. It is not a
  semantics-preserving implementation of the independently parameterized rank-2 LocalV.
- Scan uses 12 static two-layer blocks, with separate local/fetch parameter subtrees.
  Real runtime layer indices are 2*i / 2*i+1. No conditional dispatch is used.

## Throughput matrix

Every entry uses full layers, the same commit, v5p-16 and one zone, identical batch/T,
13,500-step schedule and enabled asynchronous checkpoints every 200 steps (including a
forced final checkpoint). Capture steps 10–14, then stop the standalone measurement process;
the short measurement does not change the training endpoint or checkpoint configuration.
Disable all BAM and generic training-health metrics in both measurements and formal training;
loss/accuracy/performance logging and necessary gradient clipping remain enabled.
Compile 12 configuration-specific AOTs in parallel on v6e-1, targeting v5p-16; reuse the
chosen scan/non-scan executable for each formal RUN. Measure log throughput and XPlane
device step separately.
Compare each column to its own control; additionally compare scan/non-scan retention for
each variant. The baseline uses conventional single-layer scan; alternating variants use
pair scan, so cross-row deltas include the necessary scan-body change.

Log speed is the median of steps 10–14 (robust to starting the profiler at step 10);
device time is the mean of the eight devices' captured train steps. All measurements use
`xd-v5p-16-local-fetch-ue5a`, `us-east5-a`.

| Variant | Non-scan class | Scan class | Non-scan step/s / ms | Scan step/s / ms |
|---|---|---|---|---|
| V2 control | BamLlama2MediumV2C256LocalFetchControlNonScan | BamLlama2MediumV2C256LocalFetchControlScan | 0.683 / 1450.4 | 0.673 / 1478.6 |
| compressed LocalO | BamLlama2MediumV2C256LocalFetchC8NonScan | BamLlama2MediumV2C256LocalFetchC8Scan | 0.714 / 1389.2 | 0.705 / 1410.8 |
| compressed LocalO + rank2 LocalV | BamLlama2MediumV2C256LocalFetchC8LocalVNonScan | BamLlama2MediumV2C256LocalFetchC8LocalVScan | 0.699 / 1418.2 | 0.691 / 1440.0 |
| full LocalO | BamLlama2MediumV2C256LocalFetchFullNonScan | BamLlama2MediumV2C256LocalFetchFullScan | 0.699 / 1419.0 | 0.689 / 1444.1 |
| full LocalO + rank2 LocalV (profile only) | BamLlama2MediumV2C256LocalFetchFullLocalVNonScan | BamLlama2MediumV2C256LocalFetchFullLocalVScan | 0.685 / 1448.3 | 0.675 / 1473.7 |
| compressed LocalO/V shared read | BamLlama2MediumV2C256LocalFetchC8SharedReadNonScan | BamLlama2MediumV2C256LocalFetchC8SharedReadScan | 0.707 / 1403.9 | 0.699 / 1423.3 |

Provisional bets versus matched all-fetch control: compressed LocalO +4–7% throughput;
independent LocalV consumes part of that saving; shared read should recover most of its
projection/contraction overhead. Full LocalO adds read-key width and contraction traffic,
so neither its speed gain nor its accuracy gain is assured. These are predictions, not results.

All 12 target logs confirm `Loaded compiled function!`; both standalone groups completed.
Non-scan log throughput improves only 1.1–1.5% over scan, versus 7–11x AOT preparation time.
Choose scan for formal training. The user replaced full-M + independent LocalV with full-M
shared LocalO/V read; retain the former's profile evidence, but do not train it. The five formal
variants are C8, C8LocalV, Full, C8SharedRead and FullSharedRead. For the last one, prepare only
the formal scan AOT and measure steps 10–14 during training (no additional standalone profile).
Its class is `BamLlama2MediumV2C256LocalFetchFullSharedReadScan`; it reuses the existing shared
runtime with `bam_local_o_compress_v=False`. All formal runs compare to
`BamLlama2MediumV2C256ScanAotCleanControl`; C8LocalV/C8SharedRead additionally compare to C8,
FullSharedRead to Full, and Full to C8, separating the changes within this family.

## AOT preparation time

Matched v6e-1 EW4a successful attempts, runtime `a77952e`. Time below starts at train-step
tracing and ends when the serialized executable is created in GCS: it includes lowering,
compilation, serialization and upload, but excludes queue/install/retry/cleanup. It is not
pure XLA compile time. Start = the `train_compile.py:132` tracing-warning timestamp minus its
reported tracing duration; end = `gsutil stat` object's `Creation time`.
Full configuration names are in the throughput matrix above.

| Variant | Scan seconds | Non-scan seconds | Non-scan / scan |
|---|---:|---:|---:|
| Control | 33.0 | 360.7 | 10.9× |
| compressed LocalO | 46.8 | 337.2 | 7.2× |
| compressed LocalO + rank2 LocalV | 49.6 | 394.1 | 7.9× |
| full LocalO | 46.7 | 357.0 | 7.7× |
| full LocalO + rank2 LocalV | 50.5 | 384.6 | 7.6× |
| compressed LocalO/V shared read | 48.8 | 374.6 | 7.7× |

Raw compiler logs: `tpu-ag:/home/lishengping/xd/projects/logs/local-fetch-aot-a77952e98e28eb8508c7c8a9ec16982c37f9b72d/<CLASS>.log`.
The much shorter scan critical path motivates independent compile lanes and pipelined target
measurements; cleanup completion must not gate artifact consumption.

## Validation

Pinned local suite: `.claude/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh WORKTREE`.
New tests: `MaxText/tests/bam_local_fetch_test.py`; five full attention module forward/gradient
checks (six after adding FullSharedRead), two-block scan parameter/layout checks, shared output-gate scale equivalence, and both
full train-step signatures without health outputs. All four tests pass (83.8 s); the existing
57-test BAM suite also passes (184.4 s). Target FIRST_STEP and device profiling remain
required. Diagnostic lifecycle is standalone; auto-train does not own or delete this profile TPU.

AOT batch entry (on tpu-ag):
`.claude/skills/tpu-diagnostics/scripts/prepare_local_fetch_aot.sh COMMIT`.
`prepare_local_fetch_aot.py --per-lane N` reserves independent scan/non-scan concurrency
(this matrix uses four per lane). Existing preparers are adopted; a verified artifact frees its compile slot
immediately while candidate cleanup continues. Each six-arm group emits `AOT_GROUP_READY`
independently, so scan measurements can start while non-scan compilation continues.
Each preparer retains its multi-zone candidates until its artifact verifies; logs/manifests persist.
The compiled smoke entry inherits checkpoint settings and enables XPlane explicitly.
On the installed standalone v5p-16, launch each ready group via tpu-ag's
`profile_local_fetch_matrix.sh TPU ZONE COMMIT Scan|NonScan`; it fixes the 13,500-step
schedule, trace window and zone-local output bucket before calling `run_profile_matrix.sh`.
`run_local_fetch_pipeline.sh COMMIT TPU ZONE` waits for the verified scan group, requests and
installs the standalone target, then measures scan while non-scan compilation continues.
It measures non-scan on that same target when ready and retains it after the matrix completes.

## Evidence and formal launch

Scan matrix: `20260907T094351Z-2562631`; non-scan matrix: `20260907T095637Z-2572369`.
GCS traces: `gs://newproject-1-llm_projects_us-east5/log/diagnostics/local-fetch/a77952e/`.
Local artifacts: `/data0/xd/bam_diagnostics/local-fetch-a77952e/<VARIANT><LAYOUT>/`.
Device-step extraction: `python3 experiments/bam_llama2_medium/analyze_bam_xplane.py --steps-only 'TRACE_GLOB'`.
The fast mode counts each device train step once, without summing overlapping HLO scopes.

FullSharedRead runtime/config commit: `3210379c20cd80b792816a849d81fbf7f4b9f6ad`.
Only configuration/test/report tooling changed since the original matrix runtime; attention,
model, optimizer and training runtime files are unchanged. Its six-variant forward/gradient
module test passed in 63.3 seconds. Formal scan AOT preparation uses `prepare_train_aot.py`.
C8, C8LocalV, Full and C8SharedRead formal launches submitted to UE5a at 10:21 UTC,
2026-09-07; C8 reuses the standalone profile TPU and reached step 18 by 10:23 UTC.
The other three require their own FIRST_STEP verification. FullSharedRead waits for AOT_READY
before target allocation. Training uses the measured original AOTs for the four existing arms.

## Mainline promotion and LLLF

The LocalO, independent rank-2 LocalV, shared LocalO/LocalV and static multi-layer scan
implementation is promoted to `/home/xd/projects/maxtext`, branch `refactor-bam`.
Historical runtime hashes remain in `MaxText/exp.py`; new runs use the mainline implementation.

`BamLlama2MediumV2C256LocalFetchC8SharedReadLLLFScan` changes LLF to LLLF:
six static four-layer blocks, compressed C8 LocalO/LocalV share one read on each local layer.
All 24 layers still write M and read LocalQK. Compare runs:
`BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan` and
`BamLlama2MediumV2C256ScanAotCleanControl`.
Before-launch prediction versus LLF: final gap -.001 to +.002, throughput +0–2%.
Use scan+AOT, 13,500 steps, checkpoint every 200 steps, health metrics disabled;
UE5a selected from the recent uninterrupted same-model leases.

Runtime `2980161dd2678a2ce0835777312554b27ec574c3`; local validation: 5 LocalFetch
tests (including LLLF) and 57 BAM tests pass. AOT prepared by tpu-ag
`prepare_train_aot.py EXP COMMIT v5p-16 13500`, artifact:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/2980161/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamLlama2MediumV2C256LocalFetchC8SharedReadLLLFScan.pickle`.
Formal TPU `xd-v5p-16-shared-lllf`, UE5a; launch 2026-09-07 23:43 UTC;
compiled function loaded and FIRST_STEP verified. Steps 10–14 average .7122 steps/s:
+0.9% vs shared LLF (.706), +5.8% vs matched health-off Clean (.673), within prediction.
Registry: tpu-ag `run_registry/BamLlama2MediumV2C256LocalFetchC8SharedReadLLLFScan.json`.

LLF versus Clean, means of the existing identical ±25-step/10-stride windows:

| Steps | Independent LocalV | Shared LocalV |
|---|---:|---:|
| 2000–3800 | -.01272 | -.00823 |
| 4000–5800 | -.01020 | -.00788 |
| 6000–7800 | -.00921 | -.00847 |
| 8000–9800 | -.00840 | -.00770 |
| 10000–11800 | -.00801 | -.00780 |
| 12000–13400 | -.00800 | -.00776 |

Shared LLF enters a long plateau near 2.8k–3k; independent LLF continues narrowing
until roughly 10k. At matched relative progress these correspond to about 11k and 37k
of a 50k XL run, not a guarantee that stabilization transfers across scales.

### LLLF shared–independent–shared LocalV

`BamLlama2MediumV2C256LocalFetchC8SharedIndependentSharedLLLFScan` retains LLLF,
but uses `bam_local_o_v_mode = ['shared', 'rank2', 'shared', 'none'] * 6`.
Only the middle LocalV uses an independent rank-2 full-M read; its LocalO still uses C8.
The other two LocalV branches reuse their compressed LocalO read. This tests whether sparse
independent reads improve quality without paying for them at every local layer.
Direct compare: `BamLlama2MediumV2C256LocalFetchC8SharedReadLLLFScan`.
Prediction: final gap -.0015 to 0, throughput -0.3–1%; no guaranteed gain.
Same mainline, scan+AOT, health-off, 13,500 steps/checkpoint 200, preferred zone UE5a.
Runtime `b7eb1d235c61dc36246138e65c01479d430b6789`; 5 LocalFetch regression tests pass.
`prepare_train_aot.py` produced and verified
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/b7eb1d2/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamLlama2MediumV2C256LocalFetchC8SharedIndependentSharedLLLFScan.pickle`.
All compiler candidates were released. Formal TPU `xd-v5p-16-sis-lllf`, UE5a;
registered 2026-09-08 00:10 UTC, compiled function loaded and FIRST_STEP verified.
Steps 10–14: .7072 steps/s, -0.7% vs all-shared LLLF (.7122), within prediction.
