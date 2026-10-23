# Exact per-position head activation and entropy, 2026-10-23

Canonical local report: `/home/mengqy/Projects/Bam/training_reports/bam_mudd_head_position/README.md`; primary HTML source `/home/mengqy/Projects/Bam/sites/bam-mudd-head-activation-20261023/dist/index.html`. Publication receipt in `site_publication.json` under the local report.

User definition: active iff argmax key not at positions0/1 and token ID not in `[0,15,187]` (actual Pythia BOS/period/newline). Every one of 2048 queries is counted; no query-class filtering, head-wide threshold, or content-mass cutoff. Original attention entropy is computed at every head-position and averaged only over active cells. Normalized entropy divides each cell by ln(causal context length).

Models: `BamPythia2p8BSotaIndependentR768Pallas` and `MuddPythia3BSerialMLPKeepActivations` (RUN `MuddPythia3BSerialMLP`),13k/33k, same32-text seed20261021 cohort. Not old PaxML Mudd or XL. Worktrees/branches remain `/home/mengqy/Projects/Bam/maxtext-diag-heads` / `mqy/diag-head-attention-20261022`, and `/home/mengqy/Projects/Bam/maxtext-diag-mudd-heads` / `mqy/diag-mudd-head-attention-20261022`.

Runtime commits and exact checkpoint URIs are immutable in `arms.json`, `arms_mudd.json`, and raw metadata. New reusable files: `MaxText/head_position_capture.py`, `head_position_probe.py`, `tests/head_position_stats_test.py`, `tests/head_position_capture_test.py`. Historical core model source unchanged; params restored read-only. Full QK statistics use chunked query processing and export only argmax and entropy, preserving native attention output.

BAM activation88.615%→86.838%, Serial Mudd63.998%→54.834%; active normalized entropy.719924→.676555 vs.688372→.607615. Persistent active head-position entropy changes are close (−.041889/−.041089), highlighting population selection. BAM layer1 is a persistent distinction; Mudd has stronger mid/late activity shifts. No causal head-output intervention done. Interpretations/plots live in the HTML, not duplicate ledgers.

CPU dense-reference/chunk and exact-definition tests passed, native tiny BAM XLA/Pallas and historical Mudd preservation tests passed. All128 sequences/268435456 cells verified and same original loss controls as prior round. Own Spot v5p8 `xd-v5p8-mqy-headpos-1023-ue5a` / us-east5-a plus queue deleted and both independently NOT_FOUND. No training or monitor changes.

Raw local `/data0/mengqy/bam_diagnostics/bam_mudd_head_position_20261023/`; GCS `gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/bam_mudd_head_position_20261023/`. Analysis code and full JSON live with the canonical report. Installer now checks a locally Git-derived install manifest rather than accepting a transcribed SHA; runtime arms retain separate immutable commit identities.
