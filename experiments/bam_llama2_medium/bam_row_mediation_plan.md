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

## Parallel work

- Existing EW4a TPU: completed XL L12–17 signed/causal screen; next L11 coarse
  lifetime and mediator bands, then MLP/BAM/M fine localization.
- New `xd-v6e-rowmed-b-ew4a`: L11 individual-layer MHA/V localization, L11–23.
- Both use identical code, cohort and their own clean/deleted controls. Merge by
  sequence hashes and check common-control drift before cross-TPU conclusions.
- L12 is a secondary candidate, not grounds to displace the L11 main question.
- After L11, repeat lifetime cuts and bidirectional path localization for **XL
  L10 row-self**, keeping that layer's row-cross and col unchanged. Its negative
  direct residual attribution alone does not establish harmful total effect.

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
