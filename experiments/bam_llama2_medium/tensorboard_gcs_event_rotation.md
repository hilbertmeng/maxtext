# TensorBoard GCS event rotation

2026-10-12。Worktree `/data0/xd/maxtext-tb-rotation`, branch `codex/gcs-tensorboard-event-rotation`.

## Problem and evidence

TensorBoardX `GCSRecordWriter.flush()` copies its full in-memory event buffer and calls `blob.upload_from_string()`. MaxText flushes every `log_period=25`. It retransmits all previous events, so the cost grows with worker lifetime and health metric volume. This does not change model updates, but slows training and invalidates speed comparisons across different event ages.

XL generalized read had a 1167 MB active event near 24000 steps; shared read had a 292 MB active event after a preemption/restart. Over the latest 750 steps, harmonic mean step/s was .312572 vs .374948 (-16.64%); ordinary-step medians were .379/.395. Slow steps recur every25 (G roughly14s, R roughly5-6s). Checkpoint-to-checkpoint wall times agree. The difference is not an architecture-only compute cost.

Artifacts: `/data0/xd/bam_diagnostics/xlprop-general-read/wall-speed-reconciliation.json`, `worker-tb-record-writer-source.txt`, `worker-checkpoint-evidence-24000.json`.

## Fix

`MaxText/summary_writer.py` wraps only GCS writers. At the first metric of each new 500-step segment, it closes/finalizes the previous event and opens a uniquely suffixed event in the same run directory. All metrics for the same step stay together. Ordinary flushes remain every25, preserving health/loss visibility. Local writers are unchanged. `tensorboard_gcs_rotate_period=0` restores the original behavior.

The final upload failure propagates instead of silently discarding data. Filenames keep the `events.out.tfevents.*` prefix, compatible with TensorBoard and the incremental sync script. Existing event files are retained. This bounds the active event by the metric volume of 500 steps; old segments are never repeatedly uploaded by this wrapper. It does not remove device-to-host metric-transfer costs or model health-computation costs.

## Validation

Pinned CPU environment `/data0/xd/conda/envs/maxtext-cpu/bin/python`, `PYTHONPATH=MaxText`, `MaxText/tests/summary_writer_test.py`: five tests pass. Covers bounded simulated GCS uploads, exact scalar history, unique filenames, resume/evaluation boundary behavior, local/disabled compatibility, upload failure propagation, and loading three real TensorBoardX event segments with EventAccumulator as one continuous run (no missing/duplicated loss or health points). Syntax and whitespace checks pass. Actual worker TensorBoardX signature supports `filename_suffix`, `flush`, and `close`.

Model computation, optimizer, checkpoint and RNG configuration are unchanged. Existing trainers still use their sealed runtimes; adopting the fix requires a controlled restart and runtime/AOT bookkeeping. No live trainer has been modified by this change.

## Upstream check (2026-10-12)

Latest releases checked: [MaxText maxtext-v0.2.5](https://github.com/AI-Hypercomputer/maxtext/releases/tag/maxtext-v0.2.5), published 2026-10-02, and [TensorBoardX v2.6.5](https://github.com/lanpa/tensorboardX/releases/tag/v2.6.5), published 2026-04-04. Release and current-main writer paths still have this behavior: [TensorBoardX GCS flush](https://github.com/lanpa/tensorboardX/blob/v2.6.5/tensorboardX/record_writer.py) uploads `upload_buffer.getvalue()` in full; [MaxText metric logger](https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/maxtext-v0.2.5/src/maxtext/common/metric_logger.py) still calls `self.writer.flush()` from training metric handling. The deferred one-entry metric buffer helps avoid waiting on current JAX computation, but does not make GCS uploads incremental. New MaxText's default log period is 100 instead of our 25; fewer explicit flushes mitigate frequency, without bounding the growing event file.

[TensorBoard issue #5655](https://github.com/tensorflow/tensorboard/issues/5655), discussed in April 2022, identified quadratic cumulative cloud transfer when repeatedly updating one file; its closed status does not represent an implemented remedy. [TensorBoardX PR #327](https://github.com/lanpa/tensorboardX/pull/327), merged in 2019, introduced periodic flushing instead of flushing each write, also a frequency mitigation. Upgrading alone therefore does not replace event rotation.

Historical local event audit: ordinary MHA examples have maximum individual events of 27–74 MB; standard Medium/XL BAM examples 208/355 MB; the current generalized XL event reached 1167 MB. These sizes establish different exposure, not historical slowdown percentages. Per-step loss comparisons are unaffected; long-run throughput comparisons require actual wall times and comparable metric volume/event age. Research snapshots and audit JSON: `/data0/xd/bam_diagnostics/xlprop-general-read/upstream-current/`, `upstream-issue-search.json`, `historical-event-size-audit.json`.
