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

## Result

Runtime `1517cce01b1899e7c8287e1f637775e97d8a8d28`; same UE5a v5p-32
`xd-v5p-32-xlprop-matched-health`, 2026-09-23. Both arms loaded AOT.

| Configuration | Trace-free step/s (20–24) | vs MHA control | Device-step ms |
|---|---:|---:|---:|
| BamMHAXLPropC256BasicHealthProfile | .5610 | — | 1771.361 |
| BamXLPropK72SharedRank4BasicHealthProfile | .3890 | -30.66% | 2546.169 |

BAM step time +43.74%; XPlane model FLOPs194.31856->195.98080TF (+.855%),
bytes1728.211->2381.272GB (+37.79%). Copy kernels17.224->174.354ms.
BAM read/write scopes total462.534ms; LocalQK141.930ms, write131.872ms,
VO/FetchedO read76.356ms. These source scopes can overlap through fusion;
copy time is a separate classification and must not be added again.
QK-logits scope202.460->308.101ms despite essentially unchanged model FLOPs;
MLP637.007->744.614ms with the widened MLP. These are attribution clues, not
isolated causal measurements of operators.

Formal same-zone speeds: Splash MHA~.548–.550, BAM with concat health~.384.
Thus control MHA is ~2% faster than Splash, while BAM basic-health is only~1.3%
faster than the formal health-enabled run. Neither comparison is a same-VM
single-factor ablation; the decisive .561/.389 pair is same-VM/matched-health.
Extra health and a faster Splash baseline do not explain BAM's large overhead.

Theory script `/data0/xd/bam_diagnostics/xlprop-theory/flops.py`:
ideal causal forward contractions14.13385/14.33057 W_Q per-layer average;
both C25614.26667/14.47161 (+1.4365%). Counts nominal28 writes; removing the
unused final write lowers BAM by.01147 W_Q. LM head adds.93810 to both.

Raw logs `mha.log`, `bam.log`; exact points `speeds.json`; per-device profiles
`mha-profile.json`, `bam-profile.json`, aggregate `profile-summary.json` under
the artifact directory above. Both XPlanes and both JSON traces verified locally.
GCS prefix `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/1517cce/xlprop-matched-health/`.

Cleanup verified: diagnostic TPU and queued resource absent after artifact collection;
no diagnostic preemptions. Borrowed EW4a non-preemptible compiler remained READY.
