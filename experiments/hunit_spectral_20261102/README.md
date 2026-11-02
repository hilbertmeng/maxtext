# HUnit BAM output-head diagnostic, 2026-11-02

Experiment `BamPythia2p8BHUnit300BFastKernel`; actual RUN `BamPythia2p8BHUnit300B`.
Training runtime `16ee2385da477c6967c09545dd8bbaa32e0eed66`, checkpoints33000/43000 (internal33001/43001).
Worktree `/home/mengqy/Projects/Bam/maxtext-diag-hunit-gpu`, branch `mqy/diag-hunit-gpu-reference-20261102`.
Native capture `aa12ff2e5a23448872ce683ebc94e1ed820937c7`; strict BF16 effective-weight head postprocess `c5fc8130` (full hash in manifest). Capture and head-analysis commits deliberately distinct.

Primary manifest/report: `training_reports/bam_hunit_spectral_20261102/arms_primary.json`, `results_primary.json`, `verified_primary.json`, `head_precision_controls_primary.json`, `ln_swap.json` in the shared Bam workspace. Artifacts GCS prefix `gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/bam_hunit_spectral_20261102/`.
Native capture `native_tpu_v2`, corrected head stats `native_effective_head_v1`; CUDA full-network reference `gpu_math_reference_v2` and corrected `gpu_effective_head_v1`. Sequence symlinks point to their immutable original captures.

## Findings

Native H RMS1.84129→2.42623; Wc-bottom5% input energy54.4718%→71.5980%; head gradient fraction18.2675%→44.1401%. Weak directions explain94.8582% of H squared-energy growth and86.7601% of fixed-cohort FP32 head-gradient squared-norm growth.
Formal equal1k windows: global raw gradient RMS+4.89%, head+58.71%, other parameters−6.21%; head share13.45%→30.80%. 1918 leaves reproduce global squared norm within6.01e−7.
Frozen-data final-LN swap: final scale+bias explain61.90% of weak-input energy growth with fixed43k basis (CI57.01%–66.51%), residual38.10%. Fixed33k basis agrees. This is functional Shapley, not training causality.
43k head weak-gradient fraction persists with FP32 CE:45.19% versus training customVJP45.29%; strict FP32 readout44.14%.

## Verification and rejected paths

All64 native capture frames verified: checkpoint identity, text hashes, H/final-LN reconstruction, CPU projections, Parseval, native head CE. Strict effective W uses integer RNE rather than an eliminable BF16→FP32 round trip. Original GPU readout used raw FP32 W after compiler cast elimination and is superseded; never report its CG values. CPU tiny test checks actual OutputHead CE and every gradient over two chunks. Primary4096-token config contains one2048-token chunk; independent CUDA head controls reproduce strict CG metrics exactly.

No exact nullspace exists. Wc weak directions differ from raw W and cannot be called useless. Frozen same32 training-source texts (not held-out valid), not formal batch1024. Native forward uses owned Spot v6e-1; strict head recomputation uses physicalGPU3 UUIDcbac5000. All inference only; active training untouched. Owned resource ledger and final absence audit live in the report folder; controller shared sequentially with MUDD origin and frozen softmax probes.

Report: http://172.29.142.228:8873/shorttrain/no-self-readout/hunit-spectral/
