# Shared LLF: LocalO / LocalV gate diagnostic

## Reproduction

- Training RUN: `BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan`.
- Training commit: `f6af33c7d1cb313a8db06bb55aabc133b1b450e5`.
- Read-only checkpoint: `gs://newproject-1-llm_projects_us-east5/log/BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan/checkpoints/13500/items` (committed step 13500).
- Diagnostic branch/worktree: `codex/llf-gate-diagnostics`, `/data0/xd/bam-llf-gate-diagnostics`.
- Runners: `experiments/bam_llama2_medium/{run_local_ov_gate_probe.sh,local_ov_gate_probe.py,analyze_local_ov_gates.py}`.
- Command on the installed worker: `bash experiments/bam_llama2_medium/run_local_ov_gate_probe.sh`.
- Exact diagnostic commit, parameter shapes, cohort SHA256 and per-sequence hashes are saved in `metadata.json` / `parameter_shapes.json`.
- Cohort: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`; all 128 sequences, T2048.
- Artifacts: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/local-ov-gate-final128`; local `/data0/xd/bam_diagnostics/local-ov-gate-final128`.
- TPU candidates: `xd-v6e-1-ovgate-{europe-west4-a,us-central1-a,us-east5-a}`; only the first successful worker executes the full probe.

## Questions and scope

1. O/V gate correlation per local layer, separately row/address and col/data;
   distinguish pooled correlation from within-head centered correlation.
2. Full five-bin joint distribution over [0,1], plus weighting by shared ungated
   read energy (not residual contribution). Keep all-token scalar data and per-sequence moments.
3. Joint gate-kernel singular spectrum and centered runtime-logit spectrum
   (the latter uses every eighth token, explicitly recorded; other statistics use all valid tokens).
4. Whole-network same-batch loss after replacing V gates by O or vice versa,
   independently for each side and each of the 16 local layers, plus all-local-layer controls.
   All downstream paths remain active. These are stronger interventions than a shared
   GELU hidden representation and cannot by themselves prove its retraining benefit.

The production attention implementation is unchanged. Linen interceptors record scalar
gate logits and squared ungated read norms; interventions replace only the selected gate
kernel/bias in a new parameter tree. An untouched no-capture forward must match the capture
baseline within 2e-5 per-sequence loss before accepting results. Tiny CPU scan tests pass.

## Results

Pending diagnostic completion.
