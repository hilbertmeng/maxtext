# Gate050 and mix-scale isolation

Runtime implementation and configuration classes are in the main `refactor-bam`
checkout, not a diagnostic worktree. All four runs use C256, layer-scan, v6e-prepared
AOT, 13,500 total updates, checkpoint period 200, and UE5a formal v5p-16 workers.

| Configuration class | Change | Direct compare_runs |
|---|---|---|
| BamLlama2MediumV2C256ScanAotOldGate050FixedAmplitude | Old all-decay Control; fetched p=.05, fixed a/sqrt(C)=.2; additive, no depth scaling | BamLlama2MediumV2C256ScanAotControl |
| BamLlama2MediumV2C256ScanAotOldMixScaleOnly | Learned scalar per layer, init 1/sqrt(16)=.25; signed RMS mixing without GELU; only fetch_mix_scale skips WD | BamLlama2MediumV2C256ScanAotControl |
| BamLlama2MediumV2C256ScanAotOldGeluMixScaleNoWD | Same as OldMixScaleOnly plus GELU after alpha mixing; original parameters retain old WD | BamLlama2MediumV2C256ScanAotControl; BamLlama2MediumV2C256ScanAotOldMixScaleOnly; BamLlama2MediumV2C256RmsGeluAlphaMix |
| BamLlama2MediumV2C256ScanAotCleanMixScaleOnly | Scale-only with Clean's unchanged WD exemptions | BamLlama2MediumV2C256ScanAotCleanControl; BamLlama2MediumV2C256ScanAotCleanGeluAlphaMix |

New-GELU versus historical GELU isolates decay on the new mix scale. New-GELU
versus GELU-WDFix and Clean-Scale versus Old-Scale are reserved for occasional WD
interaction analysis, not routine monitoring (comparison set revised after step 2800).

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

Report every 1,000 steps after the 5,000-step report, retaining the full 200-step
gap/r200 series; check resource/checkpoint health between reports. Selected TB
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

## Outcomes

Loss gaps below average the final six comparable 200-step windows, not a stopping-point sample.

| RUN suffix | Stopped/checkpoint | Direct comparisons and trajectory |
|---|---:|---|
| OldGate050FixedAmplitude | 11,357 | vs old Control: +.1083 at 200 shrank to near-zero sign changes after 7k; mean +.00042 over 10200–11200. No durable gain; near-zero pre-training expectation broadly met. |
| OldGeluMixScaleNoWD | 11,445 | vs old Control: +.233 at 200 shrank to a late +.00262 plateau; vs OldMixScaleOnly +.00366 (both 10400–11400). vs historical RmsGeluAlphaMix: early benefit faded, mean -.00004 over 9800–10800. GELU remained harmful on the old optimizer background; exempting only mix-scale WD did not deliver the hoped-for lasting improvement. |
| OldMixScaleOnly | 13,500 | vs old Control: +.0896 at 200 became a sustained small benefit after ~3.4k; ~-.001 around 7k–12k narrowed to mean -.00069 over 12400–13400. A small scale-only gain survives, consistent with the modest-benefit forecast. |
| CleanMixScaleOnly | 13,500 | vs CleanControl: sustained benefit after 600; ~-.004 at 5k–6k narrowed to mean -.00295 over 12400–13400. vs CleanGeluAlphaMix: the early lead steadily vanished into sign changes after 11k, late mean -.00019. Scale-only explains the final GELU-Clean gain within this resolution; GELU added no demonstrated durable benefit. |

Both user stops used one parallel closeout on 2026-09-07: 212.9 seconds, committed
final checkpoints, no lost steps, and both TPU/queue deletions verified. Automatic
TensorBoard sync markers were published. Source summary:
`tpu-ag:/home/lishengping/xd/projects/logs/closeout-20260907T063947Z.json`.
Each had one uninterrupted UE5a lease (~5h10m), zero preemptions and no region
switch; exact UTC intervals are in `experiments/tpu_region_preemption_history.md`.
Both ScaleOnly runs completed with checkpoint 13,500 committed, both TPU/queued resources
verified absent, and automatic TB sync markers published. Old exited at 07:30:49 UTC and its
registry closed after deletion at 07:32:40; Clean exited at 07:34:07 and closed at 07:35:57
on 2026-09-07. Each had one uninterrupted UE5a lease, zero preemptions and zero switches;
the regional ledger retains the full intervals. Runtime launcher logs are
`tpu-ag:/home/lishengping/xd/projects/logs/<FULL_CLASS>.log`.

The learned mix-scale/init ratio (baseline Control is fixed at 1) grew more under Clean's
optimizer background. At steps 2000 / 6000 / 10000 / 13000:

| RUN | L0–7 | L8–15 | L16–23 |
|---|---|---|---|
| OldMixScaleOnly | 1.215 / 1.467 / 1.531 / 1.539 | 1.253 / 1.466 / 1.521 / 1.531 | 1.072 / 1.254 / 1.342 / 1.370 |
| CleanMixScaleOnly | 1.271 / 1.565 / 1.648 / 1.661 | 1.306 / 1.610 / 1.704 / 1.723 | 1.144 / 1.371 / 1.477 / 1.510 |

With diagonal-one, this scale strengthens cross fetch relative to fixed self read, not both
together. Its larger learned adjustment and larger loss benefit under Clean are correlated,
not proof of which WD-exempt parameter group causes the difference. Both ScaleOnly runs exempt
the scale itself from WD, so this pair does not isolate scale WD.
