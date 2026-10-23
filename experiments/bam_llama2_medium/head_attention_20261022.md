# Pythia BAM / Serial Mudd attention-head diagnostic, 2026-10-22

Canonical report: `/home/mengqy/Projects/Bam/training_reports/bam_mudd_head_attention/README.md`.
GCS: `gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/bam_mudd_heads_20261022/` (raw per-arm data and `comparison_report/`).

Exact training implementations: BAM `1ac11833cf6a62cb5a6b0f9ae58e2f4c0520f551`, Mudd `9cdcbd02d93437b3e795a93cb787e608e39a9fff`. Capture commits: BAM `afb008bbd77e12e7b5ef3779ed86cb5069bf1697`, Mudd `f2bf55e3a275cf93a051973ef6430c5c46c515ec`; immutable manifests in the canonical report. Reusable observer and runner: `MaxText/head_attention_capture.py`, `MaxText/head_attention_probe.py`, `MaxText/make_head_token_categories.py`.

BAM worktree `/home/mengqy/Projects/Bam/maxtext-diag-heads`, branch `mqy/diag-head-attention-20261022`; Mudd worktree `/home/mengqy/Projects/Bam/maxtext-diag-mudd-heads`, branch `mqy/diag-mudd-head-attention-20261022`. No model core/experiment configuration changes. Canonical baseline is `MuddPythia3BSerialMLPKeepActivations`, formal RUN `MuddPythia3BSerialMLP`; not a substitute for old PaxML checkpoints. Both models measured at 13k and 33k on the same 32 seeded Pile sequences (T2048).

Result: no BAM-specific collapse into first-token/punctuation sinks, attention diffusion, or increasing map redundancy. BAM keeps more content-directed heads, but that does not preserve its loss advantage. Attention behavior alone cannot establish functional head contribution; next discriminating test is head-output and final-MLP perturbation with local/M paths separated. Details and values have a single source in the canonical report.

Own resource `xd-v5p8-mqy-heads-1022-ue5a`, us-east5-a Spot v5p-8, FSDP4/batch4; all 128 observations verified, then TPU and queue deleted and independently observed NOT_FOUND. No active diagnostic controllers/resources remain. Formal training/monitoring untouched.

Follow-up 2026-10-23: user supplied an exact per-position definition; see `head_position_20261023.md` and its canonical HTML for the new result. Old content-threshold numbers are not the new activation ratio.
