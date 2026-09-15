# XL SharedBasis: LocalQK row budget versus MLP

Implementation worktree: `/data0/xd/xl-shared-basis-col-only`, branch
`codex/xl-shared-basis-col-only`, based on the actual SharedBasis runtime `c664e82`.
The main `MaxText/exp.py` is the ledger, not the implementation location.

## Experimental scope

Parent: `BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis`.

| RUN | LocalQK | MLP width | compare_runs |
|---|---|---:|---|
| `BamXLSharedBasisQKColOnlyMLP` | Shared full-M rank4 column bases, independent C-fp32 head mixing/gates; no row parameters | 5642 | Parent |
| `BamXLSharedBasisQKDirectC8MLP` | Independent Q/K per-head 16x8 column keys, same compressed M projection as O; no row parameters or head mixing | 5642 | Parent, first RUN |

Both preserve M64x32, C8, NoPE96/RoPE32, LocalV, LocalO/fetchO and writes.
Pre-RMS column-key bias remains present. The direct arm uses the established O-style
RMS-gated direct read; the first arm preserves the parent's effective-key Gram,
mix-side scaling, and rank expansion order. Initial column head-mix slices are
selected from the original bilateral initializer draw.

The first contrast measures LocalQK row-read parameters against general MLP capacity.
A persistent positive loss gap supports the value of those BAM parameters.
The second contrast compares two nearly parameter-matched column mechanisms; it is
a joint change of compressed versus full read space, Q/K independence, and routing,
not an isolated test of any one of those factors.

Predictions: first minus parent +.006 late loss; second minus first +.004.
Speed is secondary; MLP parameter replacement need not compensate Read-M cost exactly.

## Budget and controls

The shared full-M column projection plus independent head mixing has width
`4*32 + 2*(16*4) = 256`; the direct Q/K column projections have width
`2*(16*8) = 256`. Column gates also have equal total width.

Deleting the shared row basis, two row-mix projections, row gates and their biases
removes 852,256 parameters per layer. A SwiGLU channel costs `3*2048=6144`
parameters: return 138 channels per layer, with no hardware-friendly rounding.
Both arms use the same width by user direction; small bias-count differences do not
change the MLP width. Parent width is 5504, new width is 5642.

Scan+AOT, all-decay (`wd_mults=[]`) exactly as parent; generic health ON, BAM health
OFF; checkpoint250, original 50,000-step schedule. Freeze NoPE width explicitly to96
rather than inferring it from the reduced injection footprint.

## Reproduction and validation

Use the pinned CPU Python `/data0/xd/conda/envs/maxtext-cpu/bin/python` with
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=MaxText` from this worktree.

- `MaxText/tests/bam_qk_col_only_test.py`: mapped nonzero-key column forward and
  parameter/M gradients, direct-arm compact shapes and active gradients.
- `experiments/bam_llama2_medium/audit_xl_qk_col_only.py --output OUTPUT.json`:
  abstract full-24 parameter tree and scan train-step, including generic raw-grad metric.
- Main skill `scripts/run_bam_unit_tests.sh WORKTREE`: baseline regression suite.

Local evidence: `/data0/xd/xl-col-only-budget.json`,
`/data0/xd/xl-qk-col-only-test.log`, `/data0/xd/xl-qk-col-only-regression.log`.

Prepare exact `v5p-32`, 50,000-step AOT using `prepare_train_aot.py` before requesting
formal trainers. Compiler primary EW4a, backups UC1a/UE5a after300s; formal primary
UE5a, backup EW4b after300s, informed by current shared region history. Keep the
alternate until FIRST_STEP. This task verifies launch/initial speed; routine training
monitoring remains with the user's monitoring task unless reassigned.
