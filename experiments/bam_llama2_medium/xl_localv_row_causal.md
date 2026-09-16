# XL shared LocalV row and LocalQ/K row causal diagnostics

## Reproduction and scope

- Model: `BamXLSharedBasisLocalVRowSharedColRank4CFp32`.
- Training runtime: `97be64f241ea5ed5596098348e501ab67fcdf4ff`.
- Original checkpoint: `gs://newproject-1-llm_projects_europe-west4/log/BamXLSharedBasisLocalVRowSharedColRank4CFp32/checkpoints/5250/items`.
- Preserved diagnostic copy: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl-localv-row-causal-20260916/checkpoints/5250/items`.
- Worktree/branch: `/data0/xd/xl-localv-row-causal`, `codex/xl-localv-row-causal`, based on exact training runtime. Production model source unchanged.
- Runner: `experiments/bam_llama2_medium/localv_row_causal.py`; launcher `run_localv_row_causal.sh`; aggregate `summarize_localv_row_causal.py`.
- Runtime/runner hashes, checkpoint, all sequence hashes, and scenario ordering are recorded in stage metadata.
- TPU: `xd-v6e-localv-row-causal-ew4a-0916`, EW4a v6e-1, owned only by this diagnosis.
- Fixed 128-sequence Pile T2048 cohort: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`. Use first64 after the user's runtime-budget instruction; all valid tokens, no selected source position.
- Result prefix: `gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl-localv-row-causal-20260916/results`.
- Local artifacts: `/data0/xd/bam_diagnostics/xl-localv-row-causal-5250`.

Invocation on the worker (exact diagnostic commit recorded by runner):

```bash
VROW_STAGE=dose VROW_STOP=32 bash experiments/bam_llama2_medium/run_localv_row_causal.sh
VROW_STAGE=route VROW_STOP=32 bash experiments/bam_llama2_medium/run_localv_row_causal.sh
VROW_STAGE=qk VROW_STOP=32 bash experiments/bam_llama2_medium/run_localv_row_causal.sh
```

Resume with the same output and STOP=64; verified sample files are skipped.

## Questions and intervention definitions

1. Scale gated LocalV row outputs by 0/.5/1/1.5, all L layers and each L separately. LocalO's use of the shared answer remains untouched, as does V column read.
2. Decompose that V change through current-layer MHA into diagonal (same-position source/destination) and off-diagonal transport. Remove each and both, then run the downstream network normally (including M writes). This is a **transport boundary**, not a claim that eventual loss changes occur only at the source/destination positions.
3. For LocalQ/K, separately remove Q row, K row, and both, per-layer and globally. Also test global .5/1.5 doses. Q/K affect attention weights, so the V-linear decomposition is not applied to them.

For bf16, the V difference is measured after the actual rounded addition to standard V. Total AV effect is native AV minus no-row AV with identical alpha. The diagonal uses the rounded V difference; the off-diagonal is its complement and includes contraction rounding. Native endpoints must reproduce the ordinary model, and both-route deletion must reproduce direct row deletion. Float32 synthetic tests verify algebra. These finite-precision conventions and tests are necessary before interpreting small loss gaps.

Paired per-sequence loss is retained; report mean, descriptive ±1.96 SE and fraction positive. Single-layer effects need not sum to all-layer effects. Frozen-checkpoint necessity is not from-scratch ablation benefit. This5250-step snapshot does not establish final-training necessity.

## Results

Initial runtime: `72c264c94ac7119f3b247f0c088a4be04e06b478`. CPU transport/module tests passed; TPU native and inactive-layer no-op passed exactly. Checkpoint restore3.94s; first sample including compile37s, later dose samples~1.42s each for52 scenarios. One TPU suffices; aggregation is small CPU work, contractions/forwards run on TPU.

First32 LocalV paired sequences: all-L deletion +.0388465 (±.0049532 descriptive95% interval half-width), L1-only deletion +.0311338 (±.0045209), both positive in32/32 samples. Global half/amplified1.5 doses +.0058070/+.0035626. This contradicts negligible current-checkpoint necessity, but does not predict the retraining gap. The L1/global ratio is not an additive attribution share. Follow-up tests retain only L1 and jointly remove Q/K to assess compensation.
