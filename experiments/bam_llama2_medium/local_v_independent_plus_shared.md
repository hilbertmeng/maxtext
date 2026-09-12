# LLF LocalV: independent rank4-B-AlignedRow + LocalO-shared read

Implementation: `codex/local-read-gram`, `/data0/xd/local-read-gram`.
RUN: `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRowSharedRead`.
Direct loss baseline: `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow` (historical completed13500).

Only L layers change. Keep independent LocalV rank4-B, AlignedRow, Q/K rank1 legacy,
LocalO and all F-layer semantics. Reuse LocalO's ungated compact `(col_K,row_C)` read;
each destination gets its own sigmoid gate. LocalV adds independent + shared outputs
to standard V. LocalO still uses `W_R_gate`; independent V uses its packed gate;
shared V uses appended `W_lv_shared_gate[D,N,2]` and `W_lv_shared_gate_b0[N,2]`.
All three gate pairs initialize to opening .005. There is no extra Read-M contraction,
no shared/independent parameter tying and no replacement of the independent branch.

New gate kernel uses zero initialization and usual kernel WD; its `_gate_b0` skips WD,
as do the baseline's gate biases. Append new parameters without renaming/reseeding existing
ones. A test verifies all common initial parameters exactly. LocalO changes from key-side
gate to equivalent output-side gate for reuse; bf16 rounding need not be bit-identical.
Do not claim exact historical training reproduction from initialization parity alone.

Cost prediction: +.03125 W_Q per L layer for gate projection, +.020833 W_Q averaged over LLF.
M-cache unchanged. Architectural throughput prediction -.5% (roughly 0 to -1%);
the new run also enables generic and dual-branch health, unlike the historical health-off
baseline, so observed speed includes instrumentation and is not a pure architecture pair.
Final loss bet: -.001 vs BAlignedRow, low-to-medium confidence. Do not mistake early transients
for complementarity; retain the full13500 schedule, checkpoint200, block-scan+AOT.

## Health contract

`record_training_health_metrics=True`: raw/global/per-parameter gradient and parameter norms.
`bam_record_local_v_dual_health=True`: compact stop-gradient summaries, no raw vector export.
Other BAM health flags remain off; the existing LocalQK-only exporter is not block-scan aware.
For each actual L layer and row/address vs col/data side:

- independent V, shared V and LocalO gate mean/std, <.05, >.95, and five bins covering [0,1];
- independent/shared/summed read RMS and RMS ratios to the same standard-V coordinate slice;
- read uncentered cosine, centered Pearson, and `2<ind,shared>/(||ind||²+||shared||²)`;
- signed gate correlations (ind/shared, ind/LocalO, shared/LocalO), joint opening >.05/.2/.8;
- each branch's read/standard-V ratio and read-energy fraction conditioned on all five gate bins.

Empty bins have zero numerical summaries and zero population; they are not evidence of measured
zero read strength. Constant gates yield zero centered correlation with guarded denominator;
interpret that together with gate std. Read correlation is pooled over batch/token/head/coordinate,
not a causal redundancy claim. Bin energy ratios condition numerator and denominator on identical
positions/heads. No division by a small individual gate is used.

Historical baseline has no matching BAM captures. Within-run branch comparisons answer whether
the added branch is used/complementary, not a baseline-relative causal claim. Do not restart the
historical baseline or run an extra diagnostic solely to fill those missing metrics.

Capture/reduction: `MaxText/layers/bam_local_v_health.py`; forward hook in `attentions.py`;
TB export hook in `train.py`, tested on actual block-scan and non-scan train-step shapes.
Sync with main training skill `sync_tensorboard_incremental.py RUN`, then:

```bash
/data0/xd/conda/envs/maxtext-cpu/bin/python experiments/bam_llama2_medium/report_local_v_dual_health.py RUN --steps 200,400,600
```

CPU tests: `PYTHONPATH=MaxText:MaxText/tests JAX_PLATFORMS=cpu` with the pinned Python,
`-m unittest bam_local_v_dual_test.DualLocalVTest`, plus the standard BAM entrypoint.
AOT via tpu-ag `prepare_train_aot.py` (EW4a primary); formal v5p-16 UE5a primary,
EW4b backup after five minutes, retain alternate until FIRST_STEP.
# All-health-OFF timing, 2026-09-12

Runtime `b32d691068153815ea4bd7b2485f1b54acea627c`, branch `codex/local-read-gram`;
configuration `BamMediumIndependentLLFLocalVRank4RoutingBAlignedRowSharedReadNoHealthProfile`.
UE5a `xd-v5p-16-dual-nohealth-0912`, full-layer block-scan, v6e-precompiled AOT,
13500-step schedule; generic and BAM-specific health both OFF. No-checkpoint standalone
runner `run_profile_matrix.sh` used `PROFILE_STEPS=13500 PROFILE_DONE_STEP=15`.
Steps10–14: .682/.604/.683/.683/.683; profiler startup disturbed step11.
Trace-free28–33: .684/.684/.684/.684/.684/.683, mean .68383 steps/s,
−1.32% versus historical BAlignedRow .6930 (same zone/health, historical runtime reused).
Formal dual-health-ON .6598 is 3.51% slower than this timing control, not all architecture overhead.

Manifest: `tpu-ag:/home/lishengping/xd/projects/logs/profile-matrix-b32d691-dual-nohealth-20260912T062118Z-612380.tsv`.
Artifacts: `/data0/xd/bam_diagnostics/dual-localv-nohealth-0912`;
GCS prefix `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/b32d691/dual-nohealth/`.
