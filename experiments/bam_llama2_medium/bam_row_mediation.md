# XL row-cross downstream mediation

Status: L11 lifetime/coarse, MHA/V, MLP/BAM/M, joint, QK, serial and row/col maps
complete; L10 row-self checks in progress. Layer indices are zero-based.
This is checkpoint causal diagnosis, not a validated training modification.

## Reproduction

- Model: `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2`, checkpoint
  `gs://newproject-1-llm_projects_europe-west4/log/BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2/checkpoints/49720/items`.
- Trainer commit: `aef0d97411a1725386ebba1aeae1bf4acb1bb79e`.
- Diagnostic branch: `codex/bam-row-mediation`, worktree `/data0/xd/bam-row-mediation`.
- Scripts: [runner](row_mediation.py), [launcher](run_row_mediation.sh),
  [host analysis](analyze_row_mediation.py), [unit tests](row_mediation_test.py),
  [collect-and-analyze](collect_row_mediation.sh). The latter takes an output
  label followed by artifact directory names, transfers at most two concurrently,
  and parses only after every transfer succeeds.
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
Restoring many downstream outputs also reconstructs much of the clean residual
trajectory by construction: 97.4% is an intervention-coverage check, not proof
that an independently manipulable circuit explains exactly 97.4%. The selective
V-edge, routing, side-specific and serial-clamp tests carry the finer mechanistic
evidence. None yet establishes a trainable architecture improvement.
The serial test below addresses whether early V acts through later BAM/M.
Do not interpret the small *additional* MLP rescue
conditional on both attention families as a standalone measure of MLP importance.

Per-sequence remaining deletion cost is `(deleted-clean) + corrected rescue`,
not a ratio of per-sequence effects. For joint MHA+BAM it is **+.000408 ±.000564**,
median +.000611, with 81/128 still positive (5th/95th percentiles
−.001955/+.003690). For cross-V alone it remains **+.005384 ±.001323**, positive
in 124/128. Thus the missing effect is not merely an aggregate-mean artifact;
the broad joint patch largely removes it, but does not recover every sequence.

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

Paired across sequences, the full-output clamp reduces the L12–15 V rescue by
**.004564 ±.000642**, in 122/128 sequences. It reduces the reverse-block cost by
**.004423 ±.000774**, in 117/128. These within-sequence contrasts, reported by
the analyzer as `serial_contrasts`, strengthen the serial-dependence evidence.

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

## Which downstream BAM side cashes out the source signal?

All 128, matched-null corrected. Patch only the selected side, leaving the other
side live; col is the K/data prefix, row the V/address suffix.

| L12–23 fetched output | Rescue Δloss | Reverse block Δloss |
|---|---:|---:|
| Col/data only | −.011008 | +.012282 |
| Row/address only | −.000362 | +.000587 |
| Both | −.011391 | +.012885 |

The source is **row-cross**, but its downstream BAM payoff is predominantly
**col/data readout**. This supports an address/transport → later data-read chain,
not the claim that row paths are globally redundant. These are effects of this
particular source perturbation. Combined with the serial clamps, the evidence
supports early cross-token V feeding later BAM/M and then col readout; QK routing
also contributes. A direct final-residual attribution does not count those
downstream changes back to the originating row contribution.

Runtime `ee578fb`; artifacts `bam-row-mediation-xl-L11-read_sides-sides-ee578fb`
and `-selfref`. All 128 paired clean/deleted controls are exact.

## L10 row-self follow-up: source dose response

Same XL checkpoint/cohort, 128 sequences; keep row-cross and col unchanged.

| Self retention | Δloss vs original | Harmed /128 |
|---|---:|---:|
| 0 | +.014154 ±.001109 | 128 |
| .25 | +.007924 ±.000744 | 127 |
| .5 | +.003656 ±.000471 | 116 |
| 1 | 0 | — |
| 1.5 | +.004462 ±.000523 | 123 |

Negative direct IG does not imply a beneficial deletion here either. Source-layer
M_out remains exactly unchanged. Runtime `ee578fb`; artifacts
`bam-row-mediation-xl-L10-coarse-lifetime-ee578fb-rowself` and its `-selfref`
companion (both complete).

The source-layer MLP interaction differs between the sources. Moving the source
cancellation from pre-MLP to post-MLP changes loss by **−.003101 ±.000445** for
L10 self, versus **+.002443 ±.001058** for L11 cross (paired 128). Thus preserving
the source MLP's clean response helps L10 self under subsequent source removal,
but not L11 cross. This conditional contrast is not the effect of removing MLP
from the model. It warrants fine MLP localization for L10 rather than assuming
the L11 path balance transfers unchanged.

### L10 downstream localization (128, matched-null corrected)

| Path | Rescue Δloss | Reverse block Δloss |
|---|---:|---:|
| L11 cross-token V | −.001110 | +.001616 |
| L12 cross-token V | −.003074 | +.001625 |
| L11–23 cross-token V | −.008834 | +.007834 |
| L11–23 standard MHA | −.011221 | +.010696 |
| L11–23 fetched BAM | −.009356 | +.007851 |
| L11–23 standard MHA + fetched BAM | −.012960 | +.012936 |
| L11–23 standard MHA + MLP + fetched BAM + M_out | −.013471 | +.014390 |

The strongest nearby V rescue moves to L12, two layers after the source. Global
attention-output restoration leaves about +.001195 of the +.014154 deletion
cost; including downstream MLP/M leaves about +.000684. These interventions
exclude source L10 MLP. A separate joint test includes it without restoring the
deleted source BAM output, motivated by the lifetime contrast above. Do not add
the −.003101 lifetime contrast to these rescue estimates across contexts.

Runtime `ee578fb`; artifacts `bam-row-mediation-xl-L10-fine-mhav-ee578fb-rowself`
and `bam-row-mediation-xl-L10-joint-downstream-ee578fb-rowself`, each with its
`-selfref` pair. All paired controls match exactly; source L10 standard MHA/V
effects are exactly zero. The EW4a worker was preempted after both fine-V arms
had uploaded all 128 sequences; no rerun of those completed arms is needed.

### Source MLP is real but overlapping (128, matched-null corrected)

| Restoration / reverse substitution | Rescue Δloss | Reverse block Δloss |
|---|---:|---:|
| L10 MLP only | −.003016 ±.000435 | +.003010 ±.000427 |
| Downstream MHA+BAM, plus L10 MLP | −.013508 | +.013513 |
| All downstream output families, plus L10 MLP | −.013467 | +.014439 |

Isolated L10 MLP helps in 115/128 rescues; reverse blocking harms 114/128.
Conditioned on downstream MHA+BAM restoration, its additional rescue is only
−.000548. With all downstream outputs restored, it adds effectively nothing.
This is consistent with its effect flowing through those descendants, not a
separate additive .003 contribution. Unlike L11 cross, a beneficial source-MLP
response is supported in both intervention directions for L10 self.

Runtime `c2b271d`; phase `source_mlp_joint`, artifacts
`bam-row-mediation-xl-L10-source_mlp_joint-source-c2b271d-rowself` and `-selfref`.
The source BAM output remains deleted in rescue arms; this is not a trivial
restoration of the original intervention. All 128 clean/deleted controls match.

## Medium comparison: downstream dependence is not XL-specific

`BamLlama2MediumV2`, checkpoint 13250, trainer commit `1afd942`, same fixed 128
Pile sequences (microbatch 2). Checkpoint:
`gs://newproject-1-llm_base_models_us-central1/log/BamLlama2MediumV2/checkpoints/13250/items`.
Source L8 row-cross has positive direct IG (+.24955% in the prior signed study),
unlike XL L11 (−.26680%). Current graph deletion cost is +.018124 ±.003180.

| Model/source; all downstream | Cross-V rescue | MHA rescue | BAM rescue | MHA+BAM rescue | Remaining cost after MHA+BAM |
|---|---:|---:|---:|---:|---:|
| Medium L8 | −.008894 | −.012424 | −.014691 | −.017402 | +.000722 ±.000276 |
| XL L11 | −.010199 | −.012470 | −.011400 | −.015175 | +.000408 ±.000564 |

Medium reverse-block costs are respectively +.009122, +.012691, +.012917,
and +.017095. Global cross-V rescues about 49% in Medium versus 65% in XL;
joint MHA+BAM restores about 96% versus 97%. These single-layer examples do not
establish a model-wide scaling law. In particular, opposite direct-IG signs
coexist with strong downstream dependence in both models: final-residual IG is
not a full-network causal-share decomposition. The expectation that Medium
would retain a much larger direct effect is not strongly supported by these
joint patches; its remaining cost is only modestly larger.

Runtime `c2b271d`; artifacts `bam-row-mediation-medium-L8-joint-selected-c2b271d`
and `-selfref`. All paired controls are exact. Medium serial/side-specific checks
are in progress; no architecture change or retraining follows automatically.
