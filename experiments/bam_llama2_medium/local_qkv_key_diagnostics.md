# Independent LLF LocalQ/K/V key sharing probe

Status: preparing inference, no empirical conclusions yet.

## Reproduction

- Model: `BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan`.
- Training commit: `f6af33c7d1cb313a8db06bb55aabc133b1b450e5`.
- Checkpoint: `gs://newproject-1-llm_projects_us-east5/log/BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan/checkpoints/13500/items`.
- Implementation: branch `codex/llf-qkv-key-diagnostics`, worktree `/data0/xd/bam-llf-qkv-keys`.
- Runner: `experiments/bam_llama2_medium/run_local_qkv_key_probe.sh`; Python capture/statistics: `local_qkv_key_probe.py` in this directory. Metadata records actual diagnostic commit.
- Artifacts: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/independent-llf-qkv-keys-13500-v1`.
- Cohort: fixed128 Pile T2048, seed9876, file SHA256 `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`; per-sequence hashes retained.
- Read-only restore, batch4, all valid token positions, 32 resumable batches. Statistics run on worker; artifacts travel worker→GCS→local.

## Questions and boundaries

Q/K have rank1; V has rank2 in the 16 Local layers. All three read full local32×32 M. Row and column spaces are analyzed separately. Raw keys include historical pre-RMS bias; transformed keys include RMS and gates. Head mixing is included in effective-key comparisons.

Per-sequence outputs retain signed/absolute/squared cosine, valid-pair fraction, joint local rank energy, and corresponding quantities after temporal centering. Centering distinguishes common mean directions from input-dependent similarity. Joint SVD uses unit basis rows to avoid dominance by gate amplitude. Its rank is token-local, not proof that a fixed projection can compress across tokens.

Negative cosine is not automatically incompatibility: signed head mixing can absorb a basis sign flip. Conversely, shared subspace is not proof that identical Q/K/V keys or readouts can be tied without loss. This probe measures correlations, not causal necessity or retraining benefit.

Prior references: `v1_step13250_gate_diagnostics.md` (gate-strength correlation, not key-sharing evidence), `bam_fetch_head_rank.md` (native shared-space rank methodology), `local_ov_gate_diagnostics.md` (fixed cohort and restore/capture workflow).
