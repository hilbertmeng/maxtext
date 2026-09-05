# XL row-cross downstream mediation

Status: L11 lifetime/coarse, MHA/V, fine MLP/BAM/M and joint maps complete;
QK routing and serial-path checks in progress. L10 row-self follows L11.
This is checkpoint causal diagnosis, not a validated training modification.

## Reproduction

- Model: `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2`, checkpoint
  `gs://newproject-1-llm_projects_europe-west4/log/BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2/checkpoints/49720/items`.
- Trainer commit: `aef0d97411a1725386ebba1aeae1bf4acb1bb79e`.
- Diagnostic branch: `codex/bam-row-mediation`, worktree `/data0/xd/bam-row-mediation`.
- Scripts: [runner](row_mediation.py), [launcher](run_row_mediation.sh),
  [host analysis](analyze_row_mediation.py), [unit tests](row_mediation_test.py).
- Fixed 128 Pile T2048 sequences, seed9876. Cohort SHA256:
  `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`.
  URI: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz`.
- Runtime commits: `d4add7a` coarse/MHA-V; `2be7629` MHA-V matched self-reference;
  `e0eb3f1` MLP/BAM/M and joint pairs; `03628c6` adds QK routing probes.
- Raw artifacts below are subdirectories of `/data0/xd/bam_diagnostics/`, with
  the same names under `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/`:
  `bam-row-mediation-xl-L11-coarse-all-d4add7a`,
  `bam-row-mediation-xl-L11-fine-mhav-d4add7a`,
  `bam-row-mediation-xl-L11-fine-mhav-2be7629-selfref`.
  Each stores sequence hashes, per-sequence/per-token losses, masks and metadata.

## Interpretation and controls

Deleting L11 row-cross alone raises mean loss by **+.015583**, keeping row-self
and col intact. The source layer's M_out is bit-identical in clean/deleted runs.
Restore clean mediator values into the deleted trajectory (rescue), and reverse
the substitution in the clean trajectory (block). No label-based terminal
residual subtraction is used.

Capture and inference graphs can round differently. Every value-patch estimate
must use a matched self-reference patch: replace the same recipient variable
with its own cached value. The table below subtracts that null intervention.
All 128 clean/deleted controls match exactly between the two MHA/V runs; L11
standard MHA/V, which precede the source deletion, have exactly zero corrected
effects. CI values are paired sequence-level normal 95% half-widths.

## Nearby MHA V: real, but not a complete explanation

| Intervention | Corrected rescue Δloss | Corrected reverse block Δloss |
|---|---:|---:|
| L12 cross-token V | −.005520 ±.000796 | +.006202 ±.001033 |
| L12 same-token V | +.000471 ±.000239 | +.000086 ±.000169 |
| L13 cross-token V | −.001997 ±.000318 | +.000788 ±.000410 |
| L14 cross-token V | −.002602 ±.000357 | +.000718 ±.000272 |
| L15 cross-token V | −.000813 ±.000232 | +.000190 ±.000198 |

The L12 cross-V rescue is about 35% of the deletion loss, not an adequate total
mechanism. Other layers also matter. These rescues are conditional and can
overlap along the same causal chain: summing them is not a mediation partition.

## Additional leads (coarse, 128; not yet self-patch-corrected)

| Patched band/path | Rescue vs deleted recipient | Reverse block vs clean recipient |
|---|---:|---:|
| L14–17 fetched BAM output | −.006239 | +.002811 |
| L18–22 fetched BAM output | −.007897 | +.004818 |
| L18–22 M_out | −.007952 | +.003877 |
| L11–13 MLP output | +.016258 | +.018362 |

Late BAM/M is a substantial candidate, potentially downstream of earlier MHA V.
MLP rescue and blocking both harm: context interactions preclude calling MLP
irrelevant or assigning it a single additive sign. Source-containing full-BAM
restoration is a trivial undo control and is excluded from downstream evidence.

## Joint downstream restoration (128, matched-null corrected)

| L12–23 path(s) | Rescue Δloss | Reverse block Δloss | Rescue / .015583 |
|---|---:|---:|---:|
| Cross-token V | −.010199 | +.009211 | 65.5% |
| Standard MHA output | −.012470 | +.011723 | 80.0% |
| Fetched BAM output | −.011400 | +.012863 | 73.2% |
| Standard MHA + fetched BAM | −.015175 | +.014980 | 97.4% |
| Standard MHA + MLP + fetched BAM + M_out | −.015283 | +.015844 | 98.1% |

These interventions do not restore the deleted source output. The two attention
output families cover most of the deletion loss jointly, but overlap strongly.
The next question is whether early V acts through later BAM/M rather than forming
independent parallel routes. Do not interpret the small *additional* MLP rescue
conditional on both attention families as a standalone measure of MLP importance.

Artifacts: `bam-row-mediation-xl-L11-joint-downstream-e0eb3f1` and its `-selfref`
companion; fine MLP/BAM/M uses `bam-row-mediation-xl-L11-fine-mlpbam-e0eb3f1`
and `-selfref`. All paired clean/deleted controls agree exactly, across all 128.

## Next discriminating checks

1. Finish fine MLP/BAM/M maps with matched self-reference controls.
2. Joint downstream bands L12, L12–13, L12–15 and L12–23: V, standard MHA,
   MLP, BAM read, M state, and combinations; both rescue and block.
3. Separate QK routing into MHA AV versus BAM fetch, and pair QK-only with
   QK+V substitution. Quantify remaining deletion loss instead of stopping at
   the first successful rescue.
4. After L11, repeat for L10 row-self, with its cross/col untouched. L12 is a
   weaker secondary candidate from the [L12–17 screen](bam_row_mediation_plan.md).
