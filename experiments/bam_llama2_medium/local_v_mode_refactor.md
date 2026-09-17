# LocalV mode/rank cleanup (2026-09-15)

Worktree: `/home/xd/projects/maxtext`; branch: `refactor-bam`.
This is a local refactor with no training RUN or TPU ownership.

## Configuration

| Layer mode | `bam_local_v_rank` | V read |
| --- | --- | --- |
| No `local_v` token | Any | Disabled |
| Contains `local_v` | `None` | Shared O reader, with its own V gate |
| Contains `local_v` | Positive integer | Independent read of that rank |

`bam_local_o_v_mode`, `bam_local_v_mode`, the module's `local_v_mode`,
and cached `_local_v_mode`/`_local_o` are removed. Retired configuration names
produce a migration error. V rank no longer falls back to Q rank; K rank and
unset routing/bias settings retain their existing Q fallback.
Independent V uses the same `_BamReadArm` path as Q/K and can run without LocalO
or LocalQK. Shared V also works without an O output destination, retaining the
shared reader and its separate V gate.

The configuration inventory covers 541 BAM classes, including historical
ledger entries. Each migrated field was checked against the inventory, and
V selection was checked on all 9,175 layers using previously supported modes.
Unsupported historical modes occur in 111 classes and still require their
recorded implementation; inventory coverage does not establish that those
experiments run on the current implementation.

Only `BamLlama2MediumV2C256LocalFetchC8SharedIndependentSharedLLLFScan` mixed
shared and independent V across layers. Its four-layer block is represented
by `bam_local_v_rank = [None, 2, None, None] * 6`; the fourth layer has no
`local_v` marker. The scan path checks that per-layer settings repeat with
the block.

## Reproduction and validation

The pre-edit source snapshot includes pre-existing uncommitted changes and
is stored at `/data0/xd/bam_diagnostics/local_v_mode_cleanup_20260915/before`.
The underlying HEAD was `ac7fccc6c4876bd5ca1712889f0a690a87cde350`, but comparing
against that commit alone would include unrelated edits. The same artifact
directory contains `manifest.json`, `config_migration.json`, test logs, and
`local_v_refactor.patch`, which isolates this refactor from the saved snapshot.

Tests use the pinned CPU environment from the diagnostics skill:

```bash
.claude/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh /home/xd/projects/maxtext
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText \
  /data0/xd/conda/envs/maxtext-cpu/bin/python \
  experiments/bam_llama2_medium/check_local_v_mode_refactor.py \
  --before /data0/xd/bam_diagnostics/local_v_mode_cleanup_20260915/before
```

Additional test scripts: `bam_config_test.py`, `bam_local_fetch_test.py`,
`bam_local_v_modes_test.py`, `bam_gram_read_test.py`, and
`bam_shared_qk_basis_test.py`, under `MaxText/tests/`.
All passed: 43 core tests, 8 config tests, 7 LocalFetch tests, 5 new LocalV
tests (the script also discovers the 7 imported LocalFetch tests), 3 Gram
tests, and 10 shared-basis tests.

All 12 numerical comparisons passed with maximum absolute error 0. The comparison checks
identical parameter trees and initialization, perturbed-parameter outputs
and M carry, and parameter/input/M gradients with exact array equality.
Coverage includes control, LocalO, independent/shared V, full/compressed M,
the mixed block's four layers, Medium/XL rank-4, and an eight-layer scan.
