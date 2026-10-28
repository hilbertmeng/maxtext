# Direct unmasking from a trained combination checkpoint

Question:can ordinary causal self-attention release additional capability after
strict-history pretraining, and how quickly does the model adapt? Compare two
full-state forks of combinationCP21557. The500-update point is a review only;
keep the original28000-step run and learning-rate schedule. No500-step stop guard.

| New RUN / primary-ledger alias | Actual runtime EXP | Runtime / reused v5p128 AOT |
|---|---|---|
| BamPythia2p8BComboKeepPreMaskResume500 | BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80 |44141a7c22c395255a6d0d58fff8e103d9a3fe87|
| BamPythia2p8BComboUnmaskResume500 | BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysQK80 |59ef09e04368599b143776f4f707ae2c86e2162f|

Main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`; aliases are
ledger-only for these launches because actual workers use the corresponding
original EXP/commit and original compiled executable. No recompilation or compiler
allocation. Parameter trees, full QKNorm100/NoPE80+RoPE20/V80, M80x64, R640 and
ordinary7808/private2944/tail9984 GELU widths remain identical. No new parameters
or MLP reimbursement. Same normal healthON/internal+BAM OFF, minimal remat,
global1024/T2048,64 devices/B16, v5p128; scoped VMEM60MiB on physical64MiB.

The exact source checkpoint is:
`gs://newproject-1-llm_projects_us-east5/log/BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80/checkpoints/21557`.

Server-side copies retain the complete30,344,618,630-byte state, including Adam
and Pile cursor. Destination cursor hashes both equal
`c99a72a9dc181181840d1b44f2dd53a82de7b951bfd72e51f47ac9fafb7d7ea8`.
Publish destination commit markers only after copy/size/cursor verification.
Actual state restoration yields the first logged/pre-update step21558; this is
update0 on diagnostic plots. Review500 is therefore native step22058. Same seed,
files, global batch and cursor preserve the same future batches across the pair.

Historical infrastructure audit against main3d933e27:checkpointing, optimizer,
train_compile, max_utils/maxtext_utils unchanged since source44141a7. Later
train/pyconfig/default changes add optional no-dropout probes and O regularizers;
these are disabled here. Existing rotating summary writer/begin_step and positive
rotation1000 already exist in the sealed source runtimes. Preserve their numerical
lineage. Focused CPU gate verified identical2,775,190,880-parameter trees and
unchanged LR values when the host endpoint varies; actual launches retain28000.
Both workers verified their original HEAD/AOT, full restore and original LR:
warmup560, cosine27440; first LR4.339e-5. No optimizer or warmup restart.

UE5a primary, EW4b staged backup, chosen from prior19-hour P0 v5p128 leases.
Task nodes:xd-v5p-128-unmask500-keep-1028 (ID310385),
xd-v5p-128-unmask500-open-1028 (ID310386). No lifecycle operation touches the
reserved FLEX compiler or other sessions' resources.

Dense paired monitoring starts with the first batch; initial transients are the
measurement target. Raw per-step loss and20-step centers/±5 windows/sample1 are
calculated by run_registry.loss_gap_series; plot the signed unmask-minus-keep gap.
The first point measures the pre-update intervention, subsequent points measure
adaptation and any crossing into net benefit. Reuse report_unmask_resume.py;
source/provenance PNG sidecars and raw JSON remain in task artifacts. Original
MHA/Mudd histories do not cover this continuation sufficiently for synchronous
new gain ratios; do not substitute old losses. Do not write interim losses in
exp.py. User requested close attention within500 and grouped review at500, with
no automatic stop or premature convergence claim.

Artifacts/current monitor:
`/data0/xd/bam_diagnostics/pythia2p8b-unmask-resume500-20261028/`.
