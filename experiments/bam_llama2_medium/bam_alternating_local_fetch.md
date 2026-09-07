# Alternating LocalO / fetched-M read

Code branch: `codex/bam-alternating-local-fetch`; worktree:
`/data0/xd/bam-alternating-local-fetch`. Main repository remains unchanged.
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
- SharedRead uses the compressed LocalO per-head read once with ungated normalized keys;
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

| Variant | Non-scan class | Scan class | Non-scan speed | Scan speed |
|---|---|---|---|---|
| V2 control | BamLlama2MediumV2C256LocalFetchControlNonScan | BamLlama2MediumV2C256LocalFetchControlScan | pending | pending |
| compressed LocalO | BamLlama2MediumV2C256LocalFetchC8NonScan | BamLlama2MediumV2C256LocalFetchC8Scan | pending | pending |
| compressed LocalO + rank2 LocalV | BamLlama2MediumV2C256LocalFetchC8LocalVNonScan | BamLlama2MediumV2C256LocalFetchC8LocalVScan | pending | pending |
| full LocalO | BamLlama2MediumV2C256LocalFetchFullNonScan | BamLlama2MediumV2C256LocalFetchFullScan | pending | pending |
| full LocalO + rank2 LocalV | BamLlama2MediumV2C256LocalFetchFullLocalVNonScan | BamLlama2MediumV2C256LocalFetchFullLocalVScan | pending | pending |
| compressed LocalO/V shared read | BamLlama2MediumV2C256LocalFetchC8SharedReadNonScan | BamLlama2MediumV2C256LocalFetchC8SharedReadScan | pending | pending |

Provisional bets versus matched all-fetch control: compressed LocalO +4–7% throughput;
independent LocalV consumes part of that saving; shared read should recover most of its
projection/contraction overhead. Full LocalO adds read-key width and contraction traffic,
so neither its speed gain nor its accuracy gain is assured. These are predictions, not results.

No formal long training is launched by this speed matrix. Runtime/artifact hashes and measured
results must replace the pending entries before deciding the subsequent training matrix.

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
| compressed LocalO/V shared read | 48.8 | 374.6 | 7.7× |

Raw compiler logs: `tpu-ag:/home/lishengping/xd/projects/logs/local-fetch-aot-a77952e98e28eb8508c7c8a9ec16982c37f9b72d/<CLASS>.log`.
The much shorter scan critical path motivates independent compile lanes and pipelined target
measurements; cleanup completion must not gate artifact consumption.

## Validation

Pinned local suite: `.claude/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh WORKTREE`.
New tests: `MaxText/tests/bam_local_fetch_test.py`; five full attention module forward/gradient
checks, two-block scan parameter/layout checks, shared output-gate scale equivalence, and both
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
