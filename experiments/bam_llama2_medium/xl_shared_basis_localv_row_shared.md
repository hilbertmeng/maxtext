# XL SharedBasis LocalV row sharing

## Configuration

- Experiment: `BamXLSharedBasisLocalVRowSharedColRank4CFp32`
- Direct baseline: `BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis`
- Worktree: `/data0/xd/xl-shared-basis-row-only`
- Branch: `codex/xl-shared-basis-row-only`
- Target: spot `v5p-32`, `europe-west4-b`, 50,000 steps, scan/AOT

The experiment changes only LocalV on L layers. Its row read reuses LocalO's compressed-C8
row answer and receives an independent LocalV gate. Its column read remains an independent
rank-4 effective-key (C) read of the full 64x32 history matrix. Q/K continue to share their
rank-4 basis; LocalO, F, write, partial RoPE, optimizer, schedule, and history-M cache are the
same as the direct baseline.

For each L layer, the old LocalV projection has 528 basis/gate/mix outputs and the new column
projection plus row gate has 208 outputs. The saving is 320 x 2048 = 655,360 weights, or
0.15625 `W_Q`, per L layer. Across the 16 L layers this is 10,485,760 weights (2.5 `W_Q`).

## Pre-run bet

- Loss: centered on zero relative to SharedBasis; likely dloss in `[-0.002, +0.002]` at
  25k-30k steps.
- Throughput: `0%` to `+1.5%` on a matched EW4b v5p-32 run.
- Expected result: positive if loss ties, because the row projection and full-M row contraction
  are removed while the history-M cache is unchanged.

The prior Medium B-routing experiment completed at a statistical tie with BAlignedRow through
13.4k steps, which is the main empirical basis for the loss prediction. XL differs by using
effective-key C routing and a 64x32 matrix, so the prediction allows a wider loss interval.

## Validation

- Pinned BAM CPU suite: 43 tests passed.
- XL and Medium row-sharing module forward/gradient checks: 2 tests passed.
- Python compilation and `git diff --check`: passed.

## Runtime

- Code commit: `97be64f241ea5ed5596098348e501ab67fcdf4ff`
- RUN registry: `BamXLSharedBasisLocalVRowSharedColRank4CFp32`
- TPU: `xd-v5p-32-xl-localv-row-shared-maxtext`, `europe-west4-b`
- AOT artifact: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/97be64f/jax081-i0ae3f58-c17f538a/v5p-32/s50000/BamXLSharedBasisLocalVRowSharedColRank4CFp32.pickle`
- FIRST_STEP gate: reached step 27 after loading the compiled function.
- Steps 10-14 throughput: about `0.544 steps/s`, tied with SharedBasis's historical
  `0.5444 steps/s` measurement.

The initial lease entered maintenance during startup and left an incomplete step-1 checkpoint.
Auto-train removed that incomplete checkpoint, restored the committed step-0 state, and passed
the first-step gate on the replacement lease.
