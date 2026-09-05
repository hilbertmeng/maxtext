# Row-cross mediator localization

## Question and limits

Locate a route that can inform an ordinary trainable forward architecture, not
a label-informed terminal-logit correction. No final-residual cancellation arm.
User hypothesis: XL L11 row-cross helps predominantly through MHA V in nearby
(not necessarily immediately following) layers and then other token positions,
rather than through MLP. MLP, BAM read and M-state interventions can falsify it.

Same fixed 128 Pile sequences and frozen checkpoint as `bam_row_cross_sign.md`.
Primary XL: `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2` @49720.
Medium `BamLlama2MediumV2` @13250 can be used as a matched diagnostic reference,
not as a numerical loss baseline for XL.

## Design

1. Baseline and source-layer row-cross deletion, with self and col unchanged.
2. Leave original computation intact up to a chosen boundary; subtract the source
   row-cross residual there; recompute the remaining network. Include source
   attention→MLP boundary, then layer ends through L22, not final L23 output.
   The residual perturbation is computed as the float32 difference of clean vs
   source-deleted post-attention residuals, handling bf16 addition/W_O rounding.
   Check the source M state is bit-identical: in these configurations row head
   coordinates do not directly enter same-layer write U.
3. Bidirectional reference patching at selected layer bands, then individual layers:
   - clean outputs into deleted recipient (rescue);
   - deleted outputs into clean recipient (block);
   - standard MHA output, MLP output, fetched BAM output, M_out state;
   - MHA V substitution on self edges, cross edges, or both.
4. V-edge patches retain the recipient QK/softmax and BAM-fetch routing. They change
   only which V content passes through standard MHA; cross means source != target
   in global token coordinates, including query chunks. This is more specific than
   simply restoring an entire attention output, which mixes routing/content changes.

Reference activations are cached on device for one microbatch, then discarded.
Save all per-sequence/per-token arm losses, hashes and numerical drift; no vectors.
Patch effects are conditional and need not add. Restoration alone is insufficient:
require reverse blocking and localize temporally before proposing an architecture.

Reference capture and patched inference compile as different graphs. For every
rescue/block, also patch the recipient's own cached activation into itself, using
the identical intervention. Subtract this matched self-patch loss, not only the
unpatched recipient loss; source-layer standard MHA/V provide negative controls.
Restoring a band containing the deleted source BAM output is an undo control,
not evidence for a downstream BAM mediator.

One L12 cross-V rescue explains only part of the deletion cost. Complete the
single-layer MHA/V and MLP/BAM/M maps, then jointly patch downstream bands
L12, L12–13, L12–15 and L12–23. Compare cross-V alone, all standard MHA,
MLP, BAM read, M state, and their combinations in both directions with matched
self-reference controls. Sequential and parallel mediators can overlap;
single-site rescue fractions are not disjoint attribution percentages.

Route refinement: cache post-RoPE/scaled Q/K, recompute the donor attention
probabilities with the same masks, and substitute them into MHA AV or BAM
mix/fetch independently. Keep recipient V, mix weights and M unchanged unless
explicitly combined with a V patch. Compare QK-only, V-only and joint QK+V;
do not interpret the numerical difference between whole-MHA and V-only rescues
as an independently additive routing contribution.

Serial-path check: restore early cross-V (L12–13 / L12–15 / L12–17) while
clamping subsequent BAM read or M_out to the deleted recipient's cached values;
reverse the clean/deleted roles for blocking. Compare against the early-V-only
effect, each corrected by its own null patch. Loss of rescue under the downstream
clamp tests whether the earlier V route operates through the later BAM/M route.

Split downstream fetched-output patches by coordinate side (col=data/K prefix,
row=address/V suffix). Patch only the selected side; leave the other side live,
including in joint multi-layer arms. Use the same bidirectional/null controls.

## Parallel work (complete)

- L11 lifetime, fine MHA/V and MLP/BAM/M, joint restoration, QK routing,
  serial clamps and downstream row/col maps are complete (128 each). See
  [the current report](bam_row_mediation.md) for calibrated results and artifacts.
- L10 self coarse, fine MHA/V, MLP/BAM/M, joint/source-MLP and read-sides pairs
  are complete (128 each). Selected Medium L8 joint, serial and read-sides pairs
  are also complete. Runtime `c2b271d` is staged on retained
  `xd-v6e-rowmed-d-ue5a` at `/home/lishengping/xd/projects/row-mediation-c2b271d`.
- `xd-v6e-rowmed-e-ew4a` was preempted after the complete fine MHA/V pair was
  uploaded; data recovered, node/queue verified deleted. Replacement
  `xd-v6e-rowmed-f-ew4a` was also preempted and verified deleted. The fine
  control's last 44 sequences were resumed on UE5a; read-sides completed there
  under the distinct `spare-sides` label. All controls agree exactly across workers.
  The earlier `rowcross-xl-ew4a`, `rowmed-b-ew4a` and `rowmed-c-ew4a` TPUs were
  preempted and deleted; the UC1a raced candidates were also verified deleted.
- Keep using each run's own clean/deleted and self-reference controls; merge by
  sequence hashes. Expanded graphs can drift slightly from previous graphs.
- L12 is a secondary candidate, not grounds to displace the L11 main question.
- L10 row-self follow-up kept row-cross and col unchanged. Its negative direct
  residual attribution did not imply harmful total effect; source MLP was a
  positive mediator, unlike the L11 cross-source conditional response.

## L12–17 screen (complete, 128)

Runtime `3be58866b384f1d4df1fd8049106f765d9858db3`, branch
`codex/bam-row-mediation`, script `row_cross_sign.py` with metric and intervention
layers 12,13,14,15,16,17. Artifacts:
`/data0/xd/bam_diagnostics/bam-row-cross-sign-xl-3be5886`, same directory under
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/`.

| Layer | Mean direct cross V (% total IG) | Delete cross Δloss | 95% CI half-width | Harmed /128 |
|---|---:|---:|---:|---:|
| 12 | −.003112 | +.006173 | .006629 | 114 |
| 13 | +.011887 | +.000790 | .000336 | 81 |
| 14 | +.019189 | +.000063 | .000166 | 68 |
| 15 | +.027051 | +.000040 | .000184 | 76 |
| 16 | +.008649 | +.000278 | .000331 | 72 |
| 17 | +.090006 | +.000462 | .000186 | 88 |

L12 median removal cost +.002276; sample index109 is +.434215 and strongly affects
the mean/normal CI. Removing that sample only as a sensitivity check gives +.002803;
retain it in the official result. Direct V is negative in 76/128, not universal.
Only L12 matches the candidate's mean-sign pattern, and much more weakly than L11.
