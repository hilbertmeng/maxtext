# XL row-cross downstream mediation

Status: L11 lifetime/coarse, MHA/V, MLP/BAM/M, joint, QK and serial maps complete;
downstream row/col refinement and L10 row-self checks in progress.
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

## MLP and memory paths (128, matched-null corrected)

| Patched band/path | Rescue vs deleted recipient | Reverse block vs clean recipient |
|---|---:|---:|
| L11 MLP output | +.002399 | +.007391 |
| L12 MLP output | +.002536 | +.002711 |
| L13 M_out | −.003605 | +.001437 |
| L18 M_out | −.006229 | +.002686 |
| L20 M_out | −.005926 | +.002567 |
| L23 fetched BAM output | −.004373 | +.001417 |
| L23 MLP output | +.012226 | +.002340 |

Late BAM/M is a substantial candidate, potentially downstream of earlier MHA V.
MLP rescue and blocking both harm: context interactions preclude calling MLP
irrelevant or assigning it a single additive sign. Source-containing full-BAM
restoration is a trivial undo control and is excluded from downstream evidence.
L11 M_out and unused final L23 M_out both have exactly zero corrected effect.

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
The serial test below addresses whether early V acts through later BAM/M.
Do not interpret the small *additional* MLP rescue
conditional on both attention families as a standalone measure of MLP importance.

Artifacts: `bam-row-mediation-xl-L11-joint-downstream-e0eb3f1` and its `-selfref`
companion; fine MLP/BAM/M uses `bam-row-mediation-xl-L11-fine-mlpbam-e0eb3f1`
and `-selfref`. All paired clean/deleted controls agree exactly, across all 128.

## Early V → later BAM/M dependence (128, matched-null corrected)

| Early V intervention | Later clamp | Rescue Δloss | Reverse block Δloss |
|---|---|---:|---:|
| Cross-V L12–15 | none | −.009152 | +.007959 |
| Cross-V L12–15 | BAM output L18–23 | −.004587 | +.003536 |
| Cross-V L12–15 | M_out L18–23 | −.006458 | +.003702 |
| Cross-V L12–17 | none | −.009769 | +.008518 |
| Cross-V L12–17 | BAM output L18–23 | −.004464 | +.003799 |

For rescue, the later clamp retains the deleted trajectory's values; for reverse
blocking it protects the clean trajectory's values. About half of the early-V
effect disappears when later BAM output is fixed. This supports serial dependence,
not a decomposition into independent additive percentages. M-state clamping also
attenuates the effect, but less symmetrically.

Restoring **cross-V + BAM output** jointly through L12–23 rescues −.014294;
reverse blocking costs +.013869. Thus cross-token V plus subsequent BAM accounts
for most of the loss even without restoring the whole standard-MHA output.
Runtime `0d6539f`; artifacts `bam-row-mediation-xl-L11-serial-chain-0d6539f`
and `-selfref`. Both share exact controls over all 128. The expanded QK-probe
graph has a small numerical drift versus the earlier graph: in the first 105
paired sequences, control mean differences were +.000062/+ .000093 and maximum
absolute .001701. Compare effects within each graph's matched-null pair.

## Q/K routing (128, matched-null corrected)

Q/K routing has now completed (128, corrected):

| Layer / routing substitution | Rescue Δloss | Reverse block Δloss |
|---|---:|---:|
| L12 QK → MHA | −.000396 | +.001963 |
| L12 QK → BAM | −.000585 | +.000039 |
| L12 QK → MHA + cross-V | −.006402 | +.007701 |
| L13 QK → MHA | −.002235 | +.000718 |
| L13 QK → BAM | −.001769 | +.000726 |
| L13 QK → MHA + cross-V | −.003987 | +.001537 |

V dominates the immediate L12 response, but L13 routing also matters. Substituting
QK into MHA leaves BAM alpha untouched, and vice versa. These remain conditional
interventions, not additive shares. Artifacts:
`bam-row-mediation-xl-L11-routing-qk-03628c6` and `-selfref`.

## Remaining checks

1. Split downstream fetched BAM output into row and col mediators.
2. Repeat for L10 row-self (now started), with its cross/col untouched. L12 is a
   weaker secondary candidate from the [L12–17 screen](bam_row_mediation_plan.md).
