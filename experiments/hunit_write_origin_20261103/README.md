# HUnit BAM: weak-readout residual writes, 2026-11-03

Experiment `BamPythia2p8BHUnit300BFastKernel`; actual RUN `BamPythia2p8BHUnit300B`.
Training source `16ee2385da477c6967c09545dd8bbaa32e0eed66`; native inference source `251e5f95f5915fe978129d34aba7bbabeef43564`.
Worktree `/home/mengqy/Projects/Bam/maxtext-diag-hunit-origin`, branch `mqy/diag-hunit-origin-20261103`.
Checkpoints33000/43000, internal33001/43001, 2,775,190,912params. Same32 Pile training-source texts,2048 targets, fixed43k vocabulary-centered BF16-effective Wc bottom128 basis (D2560,V50304). Not held-out eval or retraining.

## Main result

Direct writes concentrate at layer32:90.035% of43k head weak energy, versus2.791/2.190/1.528% at layers31/30/29. Layer32 explains97.871% of33k→43k head weak-energy growth.

|Final layer component|43k mapped head allocation|43k raw same-U allocation|33k→43k raw weak RMS|
|---|---:|---:|---:|
|Vector FFN|47.859%|21.644%|5.5025→5.3889|
|Dynamic M_o|32.044%|64.627%|8.8203→14.2024|
|Static M_o|4.630%|1.868%|.4887→.6210|
|MHA|5.489%|.978%|.6432→.8291|

Raw same-U weak-energy growth allocation: final M_o87.331% [84.170%,89.393%], dynamic85.421pp/static1.909pp; FFN7.893%. After actual final LN: FFN48.667%, M_o41.791%, MHA7.389%. Final-LN scale/position normalization changes the relative allocation. Raw same-U projection is a coordinate comparison, not SVD of the raw-space complete effective readout operator. Prediction-readout claims use the mapped head measurements.
Final Mo head allocation36.674% [33.901%,39.149%], FFN47.859% [45.146%,51.087%]; paired FFN-minus-Mo11.185pp [5.986,17.136], bootstrap32 texts5000 times. At33k Mo30.008%, FFN46.806%.

43k functional removal (recompute final FFN and final LN): remove Mo → ΔCE+.068408 [.059654,.077086], weak power ratio.575282; remove MHA →+.004554 [.003266,.005825], ratio.892526.33k corresponding ΔCE+.067264/+.004346 and weak-power ratios.696591/.947560.
Shared final VO scalar decreases.94414→.93026; not a rising scalar explanation. No claim yet separating M content, dynamic read keys or Wo changes.

## Metrics and interpretation

For each direct write dx_i, map through actual final-LN gamma and frozen actual token denominator: dh_i=gamma*(dx_i−mean_dim(dx_i))/r_t. Signed allocation E<dh_i U,hU>/E||hU||²; own projected RMS separate. Negative allocations cancel, shares can exceed100%. Bias, attention BF16 rounding and residual-addition rounding close the budget. Different from nonlinear removal or training causality.
Static Mo: extra unchanged native read with dynamic O scale zero. Dynamic=actual total−static, preserving read rounding. Branch Wo is FP32 linear using exact BF16 effective weights; native bias/rounding separate. RMS denominator D=2560.
MHA and Mo both consume M; labels identify direct vector-write consumers, not upstream memory writers. FFN input already contains Mo, so FFN's direct dominance does not rule out Mo as upstream influence. Whole-branch removals affect strong/weak features jointly.

## Fidelity and failed observations

A heavy all-layer internal observer perturbs BF16 fusion; at43k case23 H error3.66% despite tiny CE error. Initial b4 also failed auxiliary projection closure. These are not primary measurements, no sample filtering to hide failed fidelity. Heavy internal data remain exploratory in JSON.
Primary method1: observe only physical layer carry boundaries. Original native H and average CE match exactly on ALL64 cases; all32 net layer increments measured.
Primary method2: keep31-layer prefix untouched, observe final layer only. Maximum H error .03858%/.03672% at33k/43k, mean CE differences tiny; full-vs-focused final allocation discrepancy .00908/.00090 percentage points. Actual numeric checks in verification/results.
Native Pallas/Splash not replaced; strict integer RNE enforces BF16 auxiliary boundaries, every statistic matmul HIGHEST, relative continuity checked instead of exact equality of FP32 dots. Tiny interpreted native model: original/observed loss equality, carry/final-only capture, energy ledger, paired keep/removals all pass. Independent local verifier checks64 text hashes/state/source, finite shapes, original H/CE, budgets, weak-power replay, and paired bootstrap.

## Lifecycle and artifacts

Owned Spot v6e-1 only, prefix `xd-v6e1-mqy-hunorigin-1103-`: EW4a-r1 preempted during install; EW4a-r2 ran all inference; pending UC1a-r1/UE5a-r1 preserved until first completed case. All4 exact nodes/queues independently verified absent. No active training or protected ag2 xd directory writes.
Artifact root `gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/bam_hunit_write_origin_20261103/native_stability_v3`; local `/data0/mengqy/bam_diagnostics/bam_hunit_write_origin_20261103/native_stability_v3`.
Canonical run reports under shared Bam workspace `training_reports/bam_hunit_write_origin_20261103`. Runner `MaxText/hunit_write_origin_probe.py`, observers `MaxText/hunit_write_origin.py`, test `MaxText/tests/hunit_write_origin_test.py`.
Reproduce with worker manifest/training config at sealed source, then `collect_analyze.py` on the report directory and `publish.py`; aggregate JSON/scripts copied here for versioned reproducibility. Per-sequence data stay outside repo.
Report http://172.29.142.228:8873/shorttrain/no-self-readout/hunit-spectral/write-origin/
