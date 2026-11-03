# HUnit BAM early weak-readout write trajectory

Read-only analysis of BamPythia2p8BHUnit300BFastKernel, RUN BamPythia2p8BHUnit300B.
Checkpoints3k/13k/23k/33k/43k, internal label+1. Unified FastKernel inference for all five; the first three were trained before the26k FastKernel switch.
Same32 fixed Pile-source texts and2048 targets as hunit_write_origin_20261103. Reuse existing final-only native observer, add final-memory RMS statistics; baseline native H/CE replay on every text; all ledgers must close. Fixed43k Wc bottom128 is retrospective, own-checkpoint bottom128 measures an observable metric without a future basis. Final whole-Mo/MHA read removals recompute final FFN and LN, not training interventions.

Question: when do final dynamic Mo, FFN and MHA begin allocating energy to weak readout coordinates? Does raw-memory amplitude grow early enough to motivate calibrated read-view normalization? Expected dynamic Mo weak power grows, but whole-branch necessity need not worsen. Smaller norms alone cannot establish efficacy. No training, no checkpoint writes.

Worktree /home/mengqy/Projects/Bam/maxtext-diag-hunit-origin; branch mqy/diag-hunit-early-20261103. Reports training_reports/bam_hunit_early_weak_20261103 in shared Bam workspace. Owned diagnostic Spot v6e-1 prefix xd-v6e1-mqy-hunearly-1103-, EW4a/UC1a/UE5a candidates, pending backups retained until first useful text, all resources removed after local verification. Maximum3h acquisition/inference,6leases/8generations. All remote files under /home/lishengping/mengqy; no training resources or ag2 protected directory touched.

Runner MaxText/hunit_early_weak_probe.py. Companion head_inventory.py computes BF16-effective Wc bases independently on CPU. Worker outputs go directly to GCS then local verification before resource deletion. Critical failure hypotheses: changed BF16 fusion due observation (pairedH/CE/power gate), future-basis artifact (own basis comparison), wrong early checkpoint/config (state/tree/config/identity checks). Existing arithmetic/native interpreted tests plus added final-memory observer test before sealing.
