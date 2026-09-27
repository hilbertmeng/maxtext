# RMT attention-write / MLP-read chunk tuning

Worktree `/data0/xd/rmt-write-read-chunks`, branch `codex/rmt-write-read-chunks`,
parent `ffb40f2d`. Main contains unrelated concurrent Pallas changes; none are
included. Model is the completed18-layer combined-boundary RMT, MLP4078,
block scan, RoPE18, all static/dynamic paths intact. No retraining RUN.

Task-owned resources (never use the two occupied retained compilers):
- `xd-v5p-16-rmt-wr-chunks-ew4b`, europe-west4-b: paired full-step target.
- `xd-v6e-1-rmt-wr-chunks-ew4a`, europe-west4-a: spot AOT compiler.
Both are disposable diagnostics and must be deleted after verified artifacts.

Two independent switches:
- `rmt_mlp_merge_reads`: concatenate48->16 static read and zero-padded32->8
  compression into one48->24 contraction. Same parameters, +12.5% nominal
  multiplication count for these two contractions before zero elimination.
- `rmt_write_read_chunk_size`: 0(full sequence),64,128,256,512,1024,2048.
  Fixed-size `lax.map` executes attention's two write contractions/residual
  additions, static/C8 read, updated-first16 vector RMS, dynamic key/gate
  projections and dynamic MLP read as one unit. Chunk size is independent of
  the unchanged attention query chunk256. Address projections stay full-sequence;
  MLP dense projections stay outside this unit. No automatic arithmetic merging
  of the two outer writes, and no single-outer or Pallas code.

First matrix: all14 combinations on one v5p-16 and one sealed commit. Generic
training health ON, extra RMT health OFF for all arms; include optimizer and
backward/remat. Re-pair the best candidate with a repeat control; compare with
formal extra-health settings if a meaningful gain appears. Trace10-14, fixed
original13500-step learning-rate schedule, same Pile dataset/batch.

Pre-run bet: best full-step throughput +5%, likely C512 or C1024; small chunks
may lose to loop/small-GEMM costs. No systematic loss change expected. The
combined linear projection/reduction grouping need not be BF16 bit-identical.

Focused CPU gates: original/optimized parameter trees and seeded values;
nonzero dynamic keys; FP32/BF16 values, all input/parameter gradients and health;
full-size shape/count through actual remat/block scan; sealed config audit.
Full-layer AOTs use the existing orchestration/compiler, not an ad hoc training
entrypoint. Artifacts will live under `/data0/xd/bam_diagnostics/rmt-write-read-chunks`.

Follow-up: the mapped C256/512/1024/2048 arms regress full-step throughput by
11–13%. Raw XPlane attributes the regression mostly to data formatting,
elementwise fusions and chunk-output updates, rather than matrix multiplication.
Test static-unrolled C512/C1024, matching attention's Python query-chunk style,
with a fresh same-commit control. `rmt_write_read_chunk_unroll=True` only changes
scheduling; it retains the same write/read arithmetic and parameters. The initial
+5% bet is currently too optimistic; this follow-up isolates loop organization.

Use raw `.xplane.pb` for kernel attribution: small chunks exceed the trace JSON's
~1,000,000-event cap. `analyze_rmt_write_profiles.py` loads the pinned protobuf
module (override `XPLANE_PROTO` if needed), validates non-overlapping leaf coverage,
and preserves any unattributed time instead of assigning it to a kernel category.
Both small-size and actual D1200/head75/MLP4078 FP32/BF16 CPU checks pass. Optimized
BF16 training trajectories are not bit-identical; early-step loss differences are
recorded separately and must not be interpreted as final loss improvements.
