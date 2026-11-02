# Serial MUDD53k: origin of Wc weak head gradients

Worktree `/home/mengqy/Projects/Bam/maxtext-diag-mudd-origin-gpu`, branch `mqy/diag-mudd-origin-gpu-20261102`; native runtime `c8cbf94d5c900856949c07bc303a36a28e5f12df`.
Checkpoint `gs://newproject-1-llm_projects_us-east5/log/MuddPythia3BSerialMLP/checkpoints/53000/items`, internal53001, 2,784,962,589params. No training, parameter-only M-route resets preserve a03 original capture graph; Q/K/V and every other parameter unchanged.
Fixed32x2048 cohort and original prediction errors/basis; projected gradient measure is signed aligned component divided by ORIGINAL FULL head gradient squared norm, not new-gradient energy reduction. Different interventions are not additive.

Native keep H exactly reproduces archival32 frames.15 arms/480 frames verified independently (source/checkpoint/token identity, H/power, fixed-error gradient vectors). All data `gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/mudd_spectral_20261102/gradient_origin_native_reuse_v1/serial/53000/`.
Report/manifest/verification in shared workspace `training_reports/mudd_spectral_20261102/gradient_origin/{arms_native.json,verified_native.json,results_native.json}`. Source scripts are versioned in MaxText; verifier and publisher copied here.

## Findings

Archival LN-aware final-router CPU split:33k19.95−1.65=18.30pp;53k62.46−7.21=55.25pp. Baseline includes upstream MUDD; not a plain Transformer.
Native all-body current-only resets remove50.98pp aligned weak-gradient component; all-m53.86pp; final-only−7.21pp.
Group beforeL2–8:+20.18pp;L9–16:+17.58;L17–24:+12.73;L25–32:+9.93. Selected first group fine scan: L2−.7102,L3+1.1798,L4+1.4829,L5−.2274,L6+.1243,L7+1.1808,L8+.4324pp. Not exhaustive global single-layer ranking. Group-vs-singles show strong nonlinear cumulative dependence; cannot sum results.
Loss increases on these interventions: keep native/FP32CE2.22225, allbody4.44932, final2.29534. Direct removal is not recommended. Weak directions are functionally useful.

## Rejected paths and lifecycle

IG observation changed graph beyond H tolerance; rejected. Pure-JAX GPU baseline case12 CE drift.005022>.005, Hrelative.0181; rejected, no layer conclusions taken from it. Native c8 original graph supersedes GPU.
Reused owned Spot `xd-v6e1-mqy-hunitspec-1102-ew4a-r2` sequentially after BAM completed; old prefix `xd-v6e1-mqy-mudorigin-1102-` all12 resources absent verified. Parent shared ledger/final audit in BAM report directory; later frozen softmax probe uses same owner prefix. No writes under ag2 protected xd path; no active training touched.

Report http://172.29.142.228:8873/shorttrain/no-self-readout/mudd-spectral/gradient-origin/
