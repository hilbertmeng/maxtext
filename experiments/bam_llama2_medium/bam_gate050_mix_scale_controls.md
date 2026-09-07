# Gate050 and mix-scale isolation

Runtime implementation and configuration classes are in the main `refactor-bam`
checkout, not a diagnostic worktree. All four runs use C256, layer-scan, v6e-prepared
AOT, 13,500 total updates, checkpoint period 200, and UE5a formal v5p-16 workers.

| Configuration class | Change | Direct compare_runs |
|---|---|---|
| BamLlama2MediumV2C256ScanAotOldGate050FixedAmplitude | Old all-decay Control; fetched p=.05, fixed a/sqrt(C)=.2; additive, no depth scaling | BamLlama2MediumV2C256ScanAotControl |
| BamLlama2MediumV2C256ScanAotOldMixScaleOnly | Learned scalar per layer, init 1/sqrt(16)=.25; signed RMS mixing without GELU; only fetch_mix_scale skips WD | BamLlama2MediumV2C256ScanAotControl |
| BamLlama2MediumV2C256ScanAotOldGeluMixScaleNoWD | Same as OldMixScaleOnly plus GELU after alpha mixing; original parameters retain old WD | BamLlama2MediumV2C256ScanAotControl; BamLlama2MediumV2C256ScanAotOldMixScaleOnly; BamLlama2MediumV2C256RmsGeluAlphaMix; BamLlama2MediumV2C256RmsGeluAlphaMixWDFix |
| BamLlama2MediumV2C256ScanAotCleanMixScaleOnly | Scale-only with Clean's unchanged WD exemptions | BamLlama2MediumV2C256ScanAotCleanControl; BamLlama2MediumV2C256ScanAotCleanGeluAlphaMix; BamLlama2MediumV2C256ScanAotOldMixScaleOnly |

Old all-decay behavior is expressed as `wd_mults=[]` in the corrected optimizer,
not by bypassing its WD-rule construction. Scale-only and GELU use the same
parameter tree and initial scale; GELU is their only forward difference. Only
non-diagonal routing changes; the fetch diagonal remains one.

## Pre-training expectations

- Old Gate050: near zero or a tiny gain (~-.001, uncertainty about .002).
- Old ScaleOnly: near zero or a small gain; no evidence yet that learned scale
  explains the complete GELU-Clean gain.
- Old GELU with scale-only WD exemption: more promising than the old decayed-scale
  GELU, but a persistent gain over Old Control is uncertain.
- Clean ScaleOnly: could explain part or all of GELU-Clean's ~-.0028 gain; its
  difference from GELU-Clean tests whether the activation adds value.

Monitor cumulative loss gaps and checkpoint commits every 200 steps. Selected TB
health: raw_grad/clipping, W_R gradients, per-layer mix_scale/init and pre-GELU
negative-edge fraction; for Gate050 compare gate distributions, M RMS and
yBAM/ySTD with Old Control. Negative-edge fractions exclude masked/diagonal edges.
Use matching-step RUN/BASE values and long trends, not only final points.

## Launch verification

Runtime `42a1ffa801e52f16c6a525564ef4e1dfd83f914c`; all four v6e AOT jobs
completed in EW4a and released their compiler candidates. All four formal UE5a
workers reported `Loaded compiled function!` and passed registry FIRST_STEP and
step-14 gates. Speed below is the mean of logged steps 10–14; no material speed
anomaly relative to the recent same-code-family ~.65 steps/s runs.

| Class suffix after BamLlama2MediumV2C256ScanAot | TPU | steps/s |
|---|---|---:|
| OldGate050FixedAmplitude | xd-v5p-16-0-maxtext | .6468 |
| OldMixScaleOnly | xd-v5p-16-1-maxtext | .6494 |
| OldGeluMixScaleNoWD | xd-v5p-16-2-maxtext | .6476 |
| CleanMixScaleOnly | xd-v5p-16-3-maxtext | .6480 |

Validation: 57 local BAM tests and six optimizer/AOT-serialization contract tests
passed. Tests verify the scale-only initial weights equal the original 16-head
fixed-scale weights and that only the intended parameter leaves skip decay.
