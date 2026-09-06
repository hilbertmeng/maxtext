# XL L11 original-position row consumers

## Purpose and scope

Identify which components need L11's original row-cross/self residual increment,
for how long, and whether the benefit appears at its origin or later predictions.
This refines the whole-component [mediation study](bam_row_mediation.md), with
the goal of designing a normal-forward routing change—not a label-informed
correction to the final residual. [Question/decision plan](bam_row_consumer_positions_plan.md).

The failed Medium RowRelay retained the original residual and added total row
output to the next V only. It does not settle selective cross/self routing or
delivery to several nearby layers. A necessary consumer may also compensate for
harm at the origin rather than transport useful information elsewhere.

## Reproduction

- Model: `BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2`.
- Checkpoint: `gs://newproject-1-llm_projects_europe-west4/log/BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2/checkpoints/49720/items`.
- Trainer commit: `aef0d97411a1725386ebba1aeae1bf4acb1bb79e`.
- Diagnostic branch: `codex/bam-row-mediation`, independent worktree
  `/data0/xd/bam-row-mediation`; validated runtime `53090883874c2d9374b16c0540090072397c4fca`.
- Fixed 128 Pile T2048 cohort: `pile-eval-t2048-seed9876-n128-v1`, SHA256
  `68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661`.
- Scripts: [probe](row_consumer_positions.py), [launcher](run_row_mediation.sh),
  [analysis](analyze_row_consumers.py), [tests](row_consumer_positions_test.py).
- TPU: `xd-v6e-rowcons-cross-ew4a` / `europe-west4-a` for cross;
  `xd-v6e-rowcons-self-uc1a` / `us-central1-a` for self. v6e-1, batch 1, scan.

```bash
DIAGNOSTIC_COMMIT=53090883874c2d9374b16c0540090072397c4fca \
BAM_MEDIATION_PHASE=consumers BAM_MEDIATION_SOURCE=11 \
BAM_MEDIATION_COMPONENT=cross BAM_MEDIATION_LABEL=barrier-all \
BAM_CONSUMER_BARRIER=1 BAM_CONSUMER_SOURCE_MODE=all BAM_MEDIATION_N=128 \
bash experiments/bam_llama2_medium/run_row_mediation.sh xl
```

Use `self` for the paired source component. Current probes cover all valid source
positions; point sampling is retired. Historical artifacts are under
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/` and local
`/data0/xd/bam_diagnostics/`, with prefixes
`bam-row-mediation-xl-L11-consumers-barrier-{point,all}-5309088` (append
`-rowself` for self). Each file retains token losses, source positions, hashes,
scalar checks and source-increment norm, not activation vectors.

## Intervention and controls

At each selected origin, `z` is the actual clean-minus-deleted post-attention
increment. Feed only a selected consumer `RMSNorm(h-z)` instead of `RMSNorm(h)`;
the residual and all other inputs remain untouched. MLP uses its own input.
Q/K projections, LocalQK hidden projections, fetch mix/read/write projections
are separate consumers. V edges split diagonal/outgoing off-diagonal with
unchanged alpha. Write denial does not alter the independently supplied o_head.
Delayed cuts remove the original vector, not all transformed copies.

`point`: one hash-selected origin/example; source and future-distance token losses
remain separate. `all`: every valid origin denies its **own** increment to the
selected component; report global loss, not an origin/future partition. The latter
has greater aggregate signal but is not single-token lineage identification.
Both source components use identical samples/positions and intervention arms.

### Numerical audit

Initial runtime `16434cf` passed zero-increment and causal-prefix controls, but
failed immediate source-cut/deletion equivalence. Its 128-example artifacts are
retained for audit, **not accepted as mechanistic results**.

The one-example boundary audit (`3551b6e`, cross, origin 356) found:

| Check | Fused boundary | Explicit calculation boundary |
|---|---:|---:|
| Unused nonzero reference, token-loss max error | 0 | 0 |
| Same-layer M error | 0 | 0 |
| Standalone captured-residual subtraction error | 0 | 0 |
| Actual post-cut residual max error / differing coordinates | .0078125 / 76 | 0 / 0 |
| MLP input max error | .00390625 | 0 |
| MLP output max error | .0078125 | 0 |
| Immediate-cut vs deletion, token-loss max error | .125 | 0 |

An optimization barrier at the bf16 residual/denial boundary removes this
fusion/rounding sensitivity. All revised samples additionally require exact
unused-reference and immediate-cut controls before proceeding. Null agreement
alone was insufficient. Audit artifacts: `...consumers-boundary{0,1}-3551b6e/`.

## Results

All four sweeps completed 128 examples. Every sample passes the exact zero-z,
unused-reference, immediate-cut/deletion and source-M controls; point probes also
pass unaffected-prefix checks. Cross/self baselines are exactly matched. The
all-origin clean mean is 2.0922732 versus 2.0922704 in the old sign probe; compare
each intervention with its own compiled baseline. Analysis:
`/data0/xd/bam_diagnostics/row-consumers-validated-analysis.json`.

### Original-input consumers: all origins

Entries are same-batch mean Δloss after denying the original L11 increment only
to the named consumer. Positive means the consumer needs that input in the clean
context. These are **not additive attribution shares**.

| Consumer denied | L11 row-cross | L11 row-self |
|---|---:|---:|
| Entire source read deleted | +.015704 ± .001954 | +.068412 ± .005881 |
| L11 MLP | +.007353 ± .000884 | +.032167 ± .003608 |
| L12 cross-token V | +.010551 ± .001774 | +.025554 ± .001850 |
| L13 cross-token V | +.001338 ± .000293 | +.004017 ± .000494 |
| L14 cross-token V | +.000561 ± .000213 | +.001737 ± .000305 |
| L15 cross-token V | +.000604 ± .000580 | +.000889 ± .000531 |
| L12 LocalQK hidden projections | +.001359 ± .000294 | +.000980 ± .000275 |
| Joint L12–15 cross-token V | +.020764 ± .003603 | +.046688 ± .002842 |
| Joint L12–15 self-token V | +.000195 ± .000209 | −.000006 ± .000220 |
| Joint L11–15 MLP | +.011504 ± .001202 | +.071846 ± .009603 |
| Joint all measured L11–15 consumers | +.013921 ± .001651 | +.060636 ± .005057 |

Uncertainty is paired normal-approximation 95% CI half-width across sequences.
Full Q/K, mix/read/write and individual MLP results are in the analysis JSON.

- **L12 cross-token V is the largest measured downstream direct-input consumer**
  for both row-self and row-cross. Self-token V effects are much smaller. Thus the
  same downstream components consume both, not only row-cross.
- Source MLP is also substantial, especially for row-self. Its necessity does not
  distinguish useful transformation from compensation for harmful residual input.
- For cross, joint V denial exceeds deleting the source itself; joint all-consumer
  denial is smaller than V denial alone. This is measured non-additivity, not
  "132% of the contribution explained." The source's direct harm and interactions
  remain when denying only V input.
- These input-denial effects differ from earlier whole-output rescue/block patches:
  their magnitudes cannot be treated as interchangeable mediation percentages.

### Original-increment lifetime

Remove the original vector after the indicated layer's MLP; transformed copies
remain. This independently reproduces the earlier global lifetime pattern after
fixing the explicit bf16 boundary.

| Cut after layer | Cross Δloss | Self Δloss |
|---|---:|---:|
| 11 | +.018209 | +.072643 |
| 12 | +.008683 | +.025918 |
| 13 | +.004704 | +.011379 |
| 14 | +.002775 | +.007035 |
| 15 | +.001761 | +.004247 |
| 16 | +.001014 | +.002462 |
| 17 | +.000683 | +.001529 |
| 22 | +.000454 | +.000381 |

Both increments are most necessary near their creation, but "after L15 no longer
needed" would overstate the result. Absolute self-read necessity is larger.

### Joint source-MLP and V consumers

128-example paired runtime `8c24ec3` repeats the screen arms and adds these joint
denials (same all-position source response):

| Input denial | Cross Δloss | Self Δloss |
|---|---:|---:|
| L12–15 V-cross | +.020764 | +.046688 |
| L11 MLP + L12–15 V-cross | +.015872 | +.041206 |
| L11–15 MLP + L12–15 V-cross | +.010450 | +.046842 |
| L12–15 V-self + V-cross | +.022782 | +.051482 |

Joint V-cross denial grows from L12 alone (+.010551/+.025554 cross/self) through
L12–13 (+.015341/+.038896), L12–14 (+.018361/+.044198), L12–15
(+.020764/+.046688). Neighboring later consumers matter beyond the next layer.
Jointly denying MLP does **not** simply add its isolated harm to V denial.
Analysis: `/data0/xd/bam_diagnostics/row-consumer-interactions-analysis.json`.

### MLP response → same/other-position export (running)

Runtime `2be0d92e5dc0ebc3a70504c6b6a8689779257970`, script
[row_mlp_export.py](row_mlp_export.py), same model/checkpoint/cohort. First deny
the original row increment only to source L11 MLP. Capture its **effective
post-MLP residual change**, including bf16 addition rounding. Then, in an otherwise
clean forward, deny only that MLP response to selected L12–15 consumers. This is
not a donor trajectory generated by deleting the whole row pathway.

The key comparisons are V-self versus V-cross, for row-self, row-cross and both.
The strictly causal off-diagonal V edge sends a source's response only to later
positions, which cannot return to the source. A necessary cross-V export would
therefore rule out a **purely** source-local compensation account of the MLP.
A self-V edge only identifies its first destination; later transport may occur.
Do not convert any of these costs into additive fractions of the total row cost.

Exact per-example controls: deleting the effective MLP response immediately
after its creation reproduces source-MLP input denial at every token loss;
unused-response and zero-response arms reproduce clean; source post-attention
state and same-layer M remain unchanged. No activation vectors are saved.

```bash
DIAGNOSTIC_COMMIT=2be0d92e5dc0ebc3a70504c6b6a8689779257970 \
BAM_MEDIATION_PHASE=mlp_export BAM_MEDIATION_SOURCE=11 \
BAM_MEDIATION_COMPONENT=cross BAM_MEDIATION_LABEL=all \
BAM_CONSUMER_BARRIER=1 BAM_CONSUMER_MLP_EXPORT=1 BAM_CONSUMER_SOURCE_MODE=all \
bash experiments/bam_llama2_medium/run_row_mediation.sh xl
```
Repeat `self` and `both`. Artifact prefixes:
`bam-row-mediation-xl-L11-mlp_export-all-2be0d92` and `-rowself`/`-rowboth`.

### Retired point-source experiment: not a basis for token-lineage conclusions

For one fixed random origin per sequence, deleting cross gives summed token
Δloss **+.00802 ± .18241 per origin** (origin +.01038, later tokens −.00235).
Deleting self gives −.01950 ± .18889. These intervals are too wide to rank paths
or infer a beneficial/harmful future-token effect. They do not invalidate the
all-origin result: collective deletion is a different nonlinear intervention.
The exact numerical controls pass, but the sparse source-selection design is not
accepted for this question. Retain its artifacts for audit, do not rerun or extend
point sampling. The V split is an edge-specific intervention with unchanged alpha,
but it does not by itself locate which later M/col receivers redeem the benefit.

## Next discriminating checks

1. Complete L8–14 whole-network deletion ranking alongside direct IG (L8/9 gap;
   repeat L11 as an anchor). This tells whether L11 is an exceptional trade-off.
2. Separate source-MLP and L12–15 cross-V interactions, rather than adding their
   isolated effects. Validate V edge arithmetic endpoints before interpreting
   sub-milliloss effects.
3. Resolve same-position compensation versus cross-position benefit using
   route-specific controls covering all valid source positions. All-origin loss
   alone cannot provide that decomposition; do not substitute sparse source
   sampling or label a necessary component as beneficial transport without this
   distinction. The present broad result has not resolved later M/col token lineage.

## L8–14 context: direct harm versus whole-network necessity

Same XL configuration/checkpoint/cohort as above. Remove each layer's complete
row-cross contribution, preserve self/col, and recompute all later layers. Reuse
L10/11 from `row_cross_sign.py` runtime `edbb6b7`, L12–14 from `3be5886`; supplement
L8/9 and repeat L11 with runtime `5309088`. The new and old baseline and L11
deletion losses are **exactly equal for every sequence**. This is a whole-path
deletion cost, including downstream and other-token effects, not an isolated
"indirect contribution" obtained by subtracting direct IG.

| Layer | Direct cross V (% total IG) | Deletion Δloss ± 95% CI | Median | Harmed /128 |
|---|---:|---:|---:|---:|
| 8 | +.00314 | +.002429 ± .000710 | +.001914 | 110 |
| 9 | +.00544 | +.000191 ± .000564 | +.000413 | 81 |
| 10 | +.07688 | +.000952 ± .000352 | +.000819 | 93 |
| **11** | **−.26680** | **+.015583 ± .001935** | **+.013802** | **128** |
| 12 | −.003112 | +.006173 ± .006629 | +.002276 | 114 |
| 13 | +.011887 | +.000790 ± .000336 | +.000509 | 81 |
| 14 | +.019189 | +.000063 ± .000166 | +.000058 | 68 |

L11 ranks first by mean and median deletion cost: 2.52× L12's mean and 6.42× L8's.
L12 has one large deletion effect (+.434215, example109); omitting it **only as a
sensitivity check** gives +.002803. Do not claim a precise ranking of the small
L9/10/13/14 effects. L11 is exceptional on both axes: largest direct harm and
largest global necessity. That prioritizes separating its delivery/use from its
persistent direct residual effect, not suppressing the whole read.

**This ranking concerns cross only.** L11 self's direct IG is +.622187%, so its
whole row direct IG is **+.355393%**, not negative. L10's corresponding self/cross/
whole values are −.393694% / +.076884% / −.316811%. Complete self and joint deletion
at every L8–14 layer is therefore necessary before ranking whole-row necessity;
neither direct IG nor the sum of separate deletion costs substitutes for that test.

### Whole-row net effect: completed joint deletion

The primary neighbor comparison is **joint self+cross deletion followed by full
network recomputation**, including same-/other-position and direct/downstream
effects. It is neither direct IG nor the sum of isolated deletions.

| Layer | Delete cross | Delete self | Delete entire row ± 95% CI | Joint − self − cross |
|---|---:|---:|---:|---:|
| 8 | +.002462 | +.005156 | +.008708 ± .000783 | +.001091 |
| 9 | +.000334 | +.001316 | +.001894 ± .000918 | +.000244 |
| 10 | +.001128 | +.014247 | +.009806 ± .000847 | −.005568 |
| **11** | **+.015704** | **+.068412** | **+.021792 ± .002274** | **−.062324** |
| 12 | +.006236 | +.004354 | +.013415 ± .006473 | +.002825 |
| 13 | +.000778 | +.001286 | +.001664 ± .000371 | −.000400 |
| 14 | +.000138 | +.002558 | +.002537 ± .000383 | −.000159 |

L11's **entire-row** deletion cost is largest. Its paired contrast with L12 is
+.008377 ± .006932; 108/128 sequences have the larger cost at L11. L12 uncertainty
retains its extreme example, not silently trimmed. Baseline and L11 cross/self
deletion match the validated consumer sweep **exactly per sequence**.

At L11, removing cross when self is present harms loss +.015704; removing cross
after self has already been removed improves loss **−.046620**. Removing both is
far less harmful than removing self alone. This is strong conditional dependence,
not independent self/cross contribution. Geometric cancellation, local
compensation and downstream cooperation remain distinct candidate explanations;
the table alone does not select one. Joint source-consumer probes and source
geometry are being checked to distinguish them.

Raw: `/data0/xd/bam_diagnostics/bam-row-mediation-xl-L11-neighbors-interactions-8c24ec3-rowself/`
(the suffix identifies the worker pipeline, **not** a self-only measurement).
Analysis: `/data0/xd/bam_diagnostics/row-neighbors-all-analysis.json`. Both raw and
metadata are mirrored under the GCS diagnostic root specified above.

The all-position 22-arm neighbor sweep and added source-MLP/cross-V interaction
sweeps use runtime `8c24ec3fb6aeea26c705a31577622f595d38a81f`. Scripts:
[row_neighbors.py](row_neighbors.py), [analyze_row_neighbors.py](analyze_row_neighbors.py).
Launch neighbors with `BAM_MEDIATION_PHASE=neighbors`, or consumer interactions
with `BAM_MEDIATION_PHASE=consumers BAM_CONSUMER_ARM_SET=interactions`; both use
`BAM_CONSUMER_SOURCE_MODE=all BAM_CONSUMER_BARRIER=1` and the launcher above.

Supplement reproduction:
```bash
DIAGNOSTIC_COMMIT=53090883874c2d9374b16c0540090072397c4fca \
BAM_ROW_SIGN_LAYERS=8,9,11 BAM_ROW_SIGN_METRIC_LAYERS=8,9,11 \
bash experiments/bam_llama2_medium/run_row_cross_sign.sh xl
```
Script [row_cross_sign.py](row_cross_sign.py); launcher
[run_row_cross_sign.sh](run_row_cross_sign.sh). Source branch/worktree and full
model/checkpoint are listed in Reproduction. Raw per-sample/token losses and IG
metrics: `/data0/xd/bam_diagnostics/bam-row-cross-sign-xl-5309088/`, mirrored at
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/bam-row-cross-sign-xl-5309088/`.
