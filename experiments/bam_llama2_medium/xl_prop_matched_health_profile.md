# XLProp C256 basic-health paired profile

Main worktree `/home/xd/projects/maxtext`, branch `refactor-bam`.
Two full28-layer,D1920,20x96,T4096,batch8/device configurations on one v5p-32:

- `BamMHAXLPropC256BasicHealthProfile`: `Llama2XLProp` MHA geometry/MLP5120,
  BAM-MHA control (`bam_layer_modes=none`), full RoPE, C256, layer scan.
- `BamXLPropK72SharedRank4BasicHealthProfile`: formal K72x40/C10,R400,
  MLP5684,9xLLF+finalL, unchanged write/read operators; concat health OFF.

Both generic training health ON, internal/BAM sow OFF, bf16 attention logits,
full remat, no checkpoints,100-step AOT with the original50000-step LR schedule.
Trace steps10–14; use trace-free20–24 for throughput. The MHA control retains its
MHA WD rules; BAM retains all-decay. Ordinary layer scan versus BAM block scan
is recorded; no scan implementation change is part of this diagnostic.
Actual shape audits match formal parameter counts: MHA1,432,398,720,
BAM1,432,412,720. Raw artifacts `/data0/xd/bam_diagnostics/xlprop-matched-health/`.

Compiler: user-authorized retained non-preemptible `llm-jax-v6e-1-0` in EW4a.
Borrow via `prepare_train_aot_on_worker.py`, serialized retained-host lock,
reuse installed environment; never delete/recreate this machine.
Target: independent `xd-v5p-32-xlprop-matched-health` in UE5a after both AOTs ready.
Orchestration: `run_profile_matrix.sh`; no auto-train controller on profile TPU.
Release the diagnostic TPU after verified artifact collection.

Prediction before timing: MHA control .50–.55 steps/s; BAM basic-health .40–.43,
BAM roughly18–25% slower. These are hypotheses, not calibrated measurements.
