# HUnit 3k–43k native v5p trajectory: complete

BamPythia2p8BHUnit300BFastKernel, actual RUN BamPythia2p8BHUnit300B. All5 checkpoints × same32 Pile-source texts ×2048 targets,160 cases independently verified. No training. Spot v5p-8 in us-east5-a,4 independent single-chip batch1 processes. Runtime v2-alpha-tpuv5; native source 43b34723a80d5eacf61302664bb1398e724d716c. Exact task-owned VM and queue independently verified ABSENT; resource-audit.json is authoritative.

| Step | Own Wc bottom5% head-energy fraction | Physical M RMS | Final dynamicMo raw weak RMS | Final FFN raw weak RMS |
|---:|---:|---:|---:|---:|
|3000|6.17%|10.733|0.231|0.180|
|13000|20.58%|13.467|1.402|2.870|
|23000|34.77%|12.297|3.276|5.416|
|33000|54.48%|12.168|8.823|5.503|
|43000|71.60%|12.073|14.204|5.389|

Raw component RMS is the centered residual write projected into fixed43k U, with fullD2560 denominator; attention branches follow Wo, FFN is its direct residual write. Raw same-U is geometric comparison, not SVD of the full raw-space CΓWc operator. Head allocations map through actual final-LN gamma and frozen actual token denominator. Signed inner-product allocations include cross terms; not independent nonnegative energy fractions.

The trajectory already concentrates into contemporaneous weak directions at13k (20.6%) and23k (34.8%);3k is6.2%. DynamicMo overtakes FFN in raw weak writing between23k/33k, while FFN plateaus after23k.23k→43k physicalM RMS drops12.30→12.07 but dynamicMo raw weak RMS grows4.34x. Its pre-Wo total RMS grows1.74x and effective weak-direction gain(postweak/pretotal)2.50x. CommonVO read scalar decreases.959→.930. OverallM amplitude growth alone does not explain the mechanism; read selection, alignment, Wo and finalLN deserve attention. This does not prove whole-M pre-read normalization ineffective or beneficial after retraining.

After finalLN,43k FFN remains47.86% of head weak energy; dynamicMo32.04%,staticMo4.63%. WholeMo removal ΔCE is+.054 at3k andabout+.067–.068 from13k onward. Weakpower/keep after removal changes1.047,1.035,.890,.697,.575. Mo retains useful signals; whole-branch intervention cannot assign usefulness to weak features alone.13k/23k can screen whether a proposed normalization reduces the mechanism while preserving matched loss and useful reads;3k is a baseline, not proof of long-run efficacy. No normalization training has been performed.

Calibration scope: initial cross-v6e H replay failed the original1.5% bound for24/64 texts (max5.64%). These rejected flags remain intact. Weak-power maximum difference.235%, cohort CE differences−.000053/+.000531, component-share maximum difference.011percentagepoints. Gates were not widened or unfavorable texts filtered. Instead all5 points were independently recomputed on v5p; old v6e results stay separate in the parent page. Same-v5p observer checks pass ALL160 cases: maxH difference.0394%, max single-text CE drift.000601, max weakpower relative difference6.32e−6; ledgers close. Full numbers/5000 text-bootstrap intervals in results.json and verification.json.

Checkpoint label maps to internalstep label+1,2,775,190,912params. The first3 checkpoints trained before the26k FastKernel switch; unified inference backend/math isolates learned-parameter trajectories, not a bitwise replay of historical training. Wc uses actual BF16-effective weights and subtracts vocabulary mean; each own basis is recomputed independently on CPU. Fixed43k U is retrospective, own basis is contemporaneously observable. PhysicalM stats are unshifted input memory; effective shifted-read spectrum/key/gate effects were not separately attributed. Both MHA and Mo consume memory; direct FFN writing can still mediate earlier memory influence.

Artifacts/local scripts: shared Bam training_reports/bam_hunit_early_weak_v5p_20261103; GCS gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/bam_hunit_early_weak_v5p_20261103/native_v3; local /data0/mengqy/bam_diagnostics/bam_hunit_early_weak_v5p_20261103/native_v3. Worktree /home/mengqy/Projects/Bam/maxtext-diag-hunit-origin, branch mqy/diag-hunit-early-20261103. Prior v6e/general-image attempts are closed under the sibling report directory; attempt/source guards protect lifecycle and publication. Diagnostic NumPy scalar JSON regression fixed at this runtime; numeric code and within-backend thresholds unchanged.

Report http://172.29.142.228:8873/shorttrain/no-self-readout/hunit-spectral/write-origin/early-trajectory/ . Full PNG/PDF curves and results are served there; static server8873 intentionally remains. All diagnostic TPU jobs are finished and deleted.
