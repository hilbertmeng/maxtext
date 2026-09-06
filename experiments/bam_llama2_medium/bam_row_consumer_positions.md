# XL L11 original-position row consumers

## Current answer and completion status (2026-09-06 UTC)

The main XL L11 cross/self/whole-row consumer, lifetime, and downstream
mediation sweeps are complete on the fixed 128 sequences and all valid origins.
The study is **not closed**: final self controls and all-token own/earlier-origin
counterfactuals remain; the Medium L8 delivery comparison stopped at 94/128 after a nonzero
immediate-cut/deletion check and is not accepted as a completed result.

- L11 is exceptional in **whole-row net necessity**, not only negative direct
  row-cross IG. Self and cross interact strongly; their deletion costs cannot
  be added.
- L12–15 cross-token V is an important first hop. Later standard MHA and
  BAM-col jointly realize much of that hop's benefit. For the isolated L12
  cross-V intervention, descendants cannot causally return to the source token.
- MLP is not dispensable: source-MLP responses also feed cross-token V, and
  retaining V alone fails the normal-forward-compatible delivery test.
- The original vector becomes much less necessary after L15–17, but its
  transformed descendants remain. This is not permission to delete all row
  information there.
- No tested selective-delivery policy yet improves checkpoint loss. The evidence
  supports distributed, interacting consumers, not a proven loss-improving
  replacement for residual injection. See the delivery table below.

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

### MLP response → same/other-position export

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

All three completed 128 sequences and passed every exact control. Same-batch
Δloss, with sequence-level 95% CI half-width:

| Denial | Cross source | Self source | Entire-row source |
|---|---:|---:|---:|
| Original row input to L11 MLP | +.007353 ± .000884 | +.032167 ± .003608 | +.003503 ± .000443 |
| MLP response to L12–15 V-self | +.000072 ± .000127 | +.000261 ± .000165 | −.000009 ± .000133 |
| MLP response to L12–15 V-cross | +.004732 ± .000647 | +.023490 ± .002841 | +.002054 ± .000352 |
| MLP response to L12–15 MLP | +.003819 ± .001211 | +.008200 ± .001280 | +.001883 ± .000333 |
| MLP response to both V-cross and MLP | +.005835 ± .000929 | +.019368 ± .002033 | +.002489 ± .000403 |

The MLP is **not only repairing local harm**: its response is needed on strictly
off-diagonal V edges. Conversely, separate self/cross numbers overstate what can
be assigned to the whole row: its source-MLP dependence is only +.003503, compared
with +.021792 for deleting the entire row. Conditional effects are not additive
contribution fractions. These results refine, rather than replace, the direct
L12–15 V consumer result.

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

1. Test whether the measured self/cross opposition is a difference-reading
   mechanism, using token-level coefficient/geometry comparisons with neighboring
   XL layers and Medium, not a coefficient mean alone.
2. Locate downstream M/col receivers of the specifically isolated early V-cross
   response; distinguish this from the old whole-row-deletion donor trajectory.
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
the table alone does not select one. The following geometric and consumer probes
further constrain these explanations.

### L11 signed routing and self/cross opposition

Write `q_t = 1 + sum_{s != t} alpha_ts`. Algebraically,

```math
\bar M_t=q_t M_t+\sum_{s\ne t}\alpha_{ts}(M_s-M_t).
```

A position-independent common matrix is multiplied by `q_t`; small `q_t` permits
common-component rejection. It does not establish that the suppressed component
is noise, nor that the remaining difference is beneficial by itself.

128 sequences, all valid token positions, runtime
`85dca29786eaaea8bc4b88fbd9ef6b20354f3d8f` on
`xd-v6e-rowcons-path-ue5a` / `us-east5-a`:

| Statistic | Result |
|---|---:|
| Mean `q_t` | −.040015 ± .006991 |
| Fraction `abs(q_t) < .10` / `< .25` | 36.69% / 76.73% |
| Fraction `q_t < 0` | 62.83% |
| Token `q_t` p05 / p50 / p95 | −.3668 / −.06235 / +.3578 |
| Mean residual-space self/cross cosine | −.85113 ± .00726 |
| Fraction with negative cosine | 99.49% |
| Token cosine p05 / p50 / p95 | −.98244 / −.89716 / −.56878 |
| Mean `norm(row) / (norm(self)+norm(cross))` | .38913 ± .01147 |
| Mean self / cross / entire-row residual norm | 16.3779 / 10.3008 / 8.7599 |
| Cross squared-norm fraction parallel to self | 74.99% |
| Mean coefficient in `cross = -beta * self + perpendicular` | beta = .51313 |
| Algebraic `(self+cross)` squared-norm fraction perpendicular to self | 30.09% |

Each reported mean/fraction first averages valid tokens within a sequence, then
averages the 128 sequences. Quantiles pool valid tokens. Geometry uses actual
clean-minus-deletion post-attention residual increments (including W_O and bf16
addition), not private head coordinates. Their additive-closure discrepancy is
1.48% of `norm(self)+norm(cross)`; treat that as numerical granularity, not a new
path. Clean and whole-row deletion token losses match the earlier `8c24ec3`
neighbor sweep **exactly for every token**. All null/boundary/scope checks pass.
The final three rows are per-token scalar projection decompositions, averaged
within sequences then across sequences. Their sum uses the algebraic `self+cross`
rather than the separately rounded whole-row deletion increment. They suggest
both self-amplitude subtraction and an additional context direction; cross is
not merely a scalar self gate. These squared-norm fractions are not loss shares.

Thus opposition is widespread, not merely a cancellation of averages. Coupled
with the conditional deletion results, this supports useful **self-minus-context**
reading as a candidate mechanism. It does not justify labelling negative alpha
as interference or clipping it to zero. Medium L8 also has a small mean total
coefficient (+.10057), despite positive direct row-cross IG: difference reading
alone cannot explain the sign of direct attribution.

Whole-row original-input denial gives +.011063 ± .000894 for joint L12–15
V-cross, versus +.000039 ± .000148 for V-self; source L11 MLP gives
+.003503 ± .000443. The combined self/cross response, not only either isolated
large component, therefore has a needed cross-position export.

The matched 128-sequence geometry controls are now complete:

| Model/layer | Mean `q` | `abs(q)<.25` | Self/cross cosine | Cosine < 0 | `norm(sum)/(norm(self)+norm(cross))` | Sum energy perpendicular to self |
|---|---:|---:|---:|---:|---:|---:|
| XL L8 | 1.58540 | 1.72% | +.19851 | 27.24% | .78166 | 17.70% |
| XL L11 | −.04002 | 76.73% | −.85113 | 99.49% | .38913 | 30.09% |
| XL L12 | 2.25497 | 0% | +.36134 | 13.03% | .83436 | 22.47% |
| Medium L8 | +.10061 | 66.71% | −.82263 | 99.10% | .46714 | 12.83% |

Medium L8 is also a strongly opposed self/cross read, yet its direct row-cross
IG is positive. Thus negative mixing/opposition is **not sufficient** to explain
XL L11's negative direct IG. The larger perpendicular share in XL L11 is a
directional clue, not proof of the causal recipient or a loss contribution.
All means use the same token-then-sequence averaging; the energy column uses
the algebraic sum defined above. XL L8/L11/L12 clean and whole-row deletion
losses match the earlier neighbor sweep token-for-token. Medium has the same
zero-error internal controls, but no corresponding neighbor-sweep anchor.

Models/checkpoints are the XL Rank2 @49,720 and Medium V2 @13,250 listed in
Reproduction. All four use runtime `85dca29786eaaea8bc4b88fbd9ef6b20354f3d8f`.
Additional raw prefixes are
`bam-row-mediation-{xl-L8,xl-L12,medium-L8}-consumers-alpha-geometry-85dca29-rowboth/`;
analyses are `row-route-geometry-{xl8,xl11,xl12,medium8}-analysis.json` under the
local diagnostic root. XL L8 and the resumed Medium sweep completed on
`xd-v6e-rowcons-geometry-ew4a` in `europe-west4-a`; the original XL L11/L12 and
first Medium batches used the UC1a/UE5a workers recorded in their metadata.

Scripts: [probe](row_consumer_positions.py),
[geometry analyzer](analyze_row_route_geometry.py),
[consumer analyzer](analyze_row_consumers.py). Use the consumer launch above with
`BAM_MEDIATION_COMPONENT=both BAM_CONSUMER_ALPHA_GEOMETRY=1`
`BAM_CONSUMER_ARM_SET=interactions BAM_MEDIATION_LABEL=alpha-geometry` and the
runtime hash above. Raw prefix:
`bam-row-mediation-xl-L11-consumers-alpha-geometry-85dca29-rowboth/` under both
the local and GCS diagnostic roots. Local analyses:
`row-route-geometry-xl11-analysis.json`, `row-consumer-both-geometry-analysis.json`.
The raw files retain per-token scalar coefficients/geometry and per-arm losses,
not activation vectors.

#### Early cross-V to later component mediation (cross/self/whole complete)

The L11 **cross** arm is complete on all 128 sequences with runtime `a0177c7`.
Deleting row-cross gives Δloss **+.015582 ± .001915**; denying its input only
to L12 cross-token V gives **+.010552 ± .001761**. In that denied world:

| Restore clean recipient | Δloss vs denied world (95% CI) | Remaining Δloss vs clean |
|---|---:|---:|
| L13–23 fetched col | −.007257 ± .001446 | +.003295 |
| L13–23 M-out | −.006669 ± .001273 | +.003882 |
| L13–23 fetched row | +.000373 ± .000419 | +.010925 |
| L13–23 MLP | +.003736 ± .001401 | +.014287 |

This supports cross-position V export followed by downstream M/col readout,
not primarily another row readout. Restoring col recovers about 69% of this
specific V-denial penalty, **not** 69% of total row-cross necessity. M and col
are overlapping, nonlinear mediation interventions; their effects cannot be
added. MLP restoration alone worsens the denied world, despite MLP input-denial
being harmful in the clean world: necessity and sufficiency depend on context.
No single tested col layer dominates (individual L13–18 restorations recover
.00099–.00166), supporting a distributed downstream pathway.

All seven within-graph audits are exactly zero. Against the older neighbor
graph, mean clean-loss drift is +.000023 and deletion-effect drift −.000122
± .000188; maximum individual token-loss drift is 1.0. Thus the old and new
graphs are **not** tokenwise interchangeable, even though aggregate drift is
small. All reported mediation comparisons use the same new graph.
Analysis: `/data0/xd/bam_diagnostics/row-v-export-cross-analysis.json`;
raw prefix `bam-row-mediation-xl-L11-v_export-L12-a0177c7/` under the diagnostic
GCS/local roots. The completed whole-row and joint L12–15 tests below qualify
how far this isolated-cross result generalizes.

The matched **self** source has also completed all 128 sequences with seven
exact-zero audits. Its row deletion costs +.068428 ± .005829 and L12 V-cross
input denial +.025504 ± .001814. The downstream pattern is similar to cross:

| Restored L13–23 recipient | Self: Δloss vs denied | Self: remaining Δloss vs clean |
|---|---:|---:|
| fetched col | −.014432 ± .001437 | +.011072 |
| M-out | −.012498 ± .001056 | +.013006 |
| fetched row | −.000981 ± .000345 | +.024523 |
| MLP | +.004630 ± .003402 | +.030134 |

Thus col/M mediation is not unique to row-cross. The self MLP restoration has
a near-zero median (−.000495), unlike its positive mean; do not characterize
that worsening as uniform across examples. Self clean/deletion-effect drift
against the old graph is +.000023/+.000016, with per-token maxima 1/.6875.
Analysis: `/data0/xd/bam_diagnostics/row-v-export-self-analysis.json`; raw prefix
is the cross prefix plus `-rowself`.

Runner [row_v_export.py](row_v_export.py) first denies the original L11 row
increment **only to L12 cross-token V edges**. The donor pair therefore differs
at that consumer, not at the original row output everywhere. In both directions,
patch later fetched-col, fetched-row, M-out, and MLP from the opposite donor;
screen individual L13–18 and jointly L13–23, plus L12 M-out/MLP. Repeat for
row-cross, row-self, and their sum; `BAM_V_EXPORT_END=15` additionally tests the
joint L12–15 export if the single-L12 results leave substantial effects.

Use `BAM_MEDIATION_PHASE=v_export BAM_CONSUMER_BARRIER=1` with the standard
launcher and record the sealed diagnostic commit. Require exact token-loss
agreement for the seed graph, unused references, zero-z, both self-patching
worlds, and unchanged source residual/M before accepting any result. All 128
sequences and all valid origins participate. Save token losses/checks, not
reference vectors. This is conditional component mediation: it distinguishes
the first hop's same/other-position edge but does not assign multi-hop effects
to individual origins or turn nonlinear restoration effects into additive shares.

For the isolated first hop there is also a structural position guarantee:
causal V-cross carries source position `s` only to `t>s`. Every later attention
or BAM fetch is causal and every M write/MLP is token-local, so its descendants
cannot return to `s`. A mediated benefit downstream of this isolated hop is
therefore a later-position benefit, not repair of that source position's own
loss. This does not assign nonlinear effects to individual sources when all
origins are perturbed together, and does not cover row paths bypassing that hop.

Runtime `cfa5251` was rejected before collecting intervention results: seed/self
replacement failed numerical endpoint checks. Runtime
`a0177c77890e51e6710e62851d81a6d339cad05f` instantiates only the full-read/M/MLP
patch recipients and materializes their bf16 boundaries in both donor worlds.
All seven within-graph checks then pass exactly. Unlike the earlier geometry
sweep, this graph is not bitwise identical to the historical neighbor sweep;
[analyze_row_v_export.py](analyze_row_v_export.py) quantifies clean/deleted
tokenwise and mean-loss drift against that anchor separately. Do not interpret
small cross-run differences as mediation effects. References remain on device.
The original whole-row UC1a arm was preempted after 65 uploaded sequences;
cross/self completed on UE5a, whose later whole-row retry uploaded 29 before
preemption. Preserve these partial prefixes but do not report them as complete.
Runtime `52a0c190f5aa8f3bc3c7410c0b713e71aef5cdf2` adds validated batch resume
and atomic publication only; the attention/decoder/forward files are unchanged
from `a0177c7`. Four resume tests and all 65 saved batches pass cohort, mask,
shape, finite-value, metadata and exact-control checks. Use
`BAM_MEDIATION_RESUME_GCS` plus the explicitly audited
`BAM_MEDIATION_RESUME_COMMIT` to inherit the saved prefix. The new metadata
records the previous runtime and inherited batch offsets.

Expanded runtime `818a3f0dd8a11a6b63b48f741bd7ca460a35e68a` additionally tests
standard MHA outputs and joint standard-MHA/col/MLP recipients in temporal
bands. This addresses the unaccounted part of V-denial effects without assuming
that col is the whole explanation. Set `BAM_V_EXPORT_RECIPIENT_SET=expanded`;
the standard-output boundary is materialized in both seed and patched graphs,
and all seven exact audits remain required. The prior screen is unchanged when
this flag is absent; eight consumer/unit tests pass.

Expanded results (128 sequences each; seven exact-zero audits) report the
**remaining loss penalty versus clean** after restoring later recipients:

| Source / denied first hop | Denial penalty | Restore MHA | Restore col | MHA + col | MHA + col + MLP |
|---|---:|---:|---:|---:|---:|
| cross / L12 V-cross; restore L13–23 | +.010552 | +.004527 | +.003295 | +.000930 | +.000330 |
| whole / L12 V-cross; restore L13–23 | +.005050 | +.002382 | +.002283 | +.001063 | +.000454 |
| whole / L12–15 V-cross; restore L16–23 | +.010924 | +.006414 | +.005585 | +.002974 | +.001638 |

The first row's last two residuals have 95% CI half-widths .000285/.000221.
The triple restoration recovers about 97%, 91%, and 85% of these **specific
denial penalties**, respectively; these are not additive shares of the total
row contribution. The wider first-hop intervention exposes more distributed
dependency, leaving a larger unaccounted residual. MLP alone does not rescue
these donor worlds, yet helps jointly with MHA/col: single-component signs do
not transfer to joint interventions.

Expanded raw prefixes under the GCS/local diagnostic roots:
`bam-row-mediation-xl-L11-v_export-L12-expanded-818a3f0` (cross; append
`-rowboth` for whole), and
`bam-row-mediation-xl-L11-v_export-L12-15-expanded-818a3f0-rowboth`.
Analyses: `row-v-export-expanded-{cross,both,joint-both}-analysis.json`.
The basic whole-row L12 sweep resumed the validated first 65 examples into
`bam-row-mediation-xl-L11-v_export-L12-52a0c19-rowboth`; all 128 are complete.

#### Selective finite-lifetime delivery: causal feasibility, not future patching

[row_delivery.py](row_delivery.py) keeps the source row increment `z` as a
private carrier. Selected consumers see `h`, the others see `h-z`; after the
selected cutoff layer's MLP, subtract `z` once. The source precedes every
intervention, so `z` can be computed in a normal causal forward without labels,
gradients, clean future activations, or final-layer cancellation. This is a
testable architectural restriction, not a guaranteed improvement.

All three XL sweeps completed 128 sequences with four exact-zero endpoint
checks. Entries are Δloss versus the unmodified checkpoint, cutoff after L17:

| Retained consumers through cutoff | row-cross | row-self | whole row |
|---|---:|---:|---:|
| All (ordinary delayed removal control) | +.000670 | +.001586 | +.000577 |
| Cross-token V only | +.012427 | +.052083 | +.008105 |
| Cross-token V + MLP | +.006017 | +.008724 | +.003159 |
| Cross-token V + MLP + LocalQK | +.004297 | +.006478 | +.002377 |
| Standard MHA Q/K/V + MLP | +.005085 | +.006841 | +.002179 |

For comparison, removing each source outright costs +.015582 / +.068428 /
+.021820. Extending V-only delivery from L12 to L17 barely helps cross
(+.012617 → +.012427): lack of lifetime alone is not its main problem.
Adding MLP is a large improvement, especially for self, but even the broader
tested subsets underperform the matched all-consumer cutoff. This contradicts
the strongest version of “only nearby MHA V needs this information”; it does
not contradict V being an important transport channel. The full row performs
much better than separately restricted self/cross would suggest, again showing
their interaction.

These are collective all-origin interventions, not additive origin-level
attributions. Completed cross/whole no-consumer controls (128 each) quantify
the representation error from repeated bf16 `h-z`: at L12/13/15/17 their loss
differences versus outright deletion are cross −.000048/+.000008/+.000068/+.000075,
whole −.000065/−.000114/−.000115/−.000078. These are much smaller than the
selective policies' .002–.012 penalties. The all-consumer cutoff means exactly
match the previous sweep. Self controls remain in progress.

Runtime `359b559923022700b357073a275b02ed9bfc5627`; launcher
`BAM_MEDIATION_PHASE=delivery BAM_MEDIATION_SOURCE=11
BAM_MEDIATION_COMPONENT=cross|self|both BAM_MEDIATION_LABEL=selective
BAM_CONSUMER_BARRIER=1`, model `xl`. Raw prefix
`bam-row-mediation-xl-L11-delivery-selective-359b559` (append `-rowself` or
`-rowboth`). [Analyzer](analyze_row_delivery.py) validates cohort completeness,
unique sample hashes, finite losses and endpoint checks before aggregation;
outputs `row-delivery-{cross,self,both}-analysis.json` locally. Reproduction
uses the model/checkpoint/cohort specified above.

Runtime `f9091ec6ddb344024e9666ad440b8381c132ed01` adds
`BAM_DELIVERY_CONTROL_ONLY=1` (all/none policies only, unchanged forward) for
the numerical-floor controls. Medium L8/whole uses `BamLlama2MediumV2`, checkpoint
13250, trainer `1afd942`, the same cohort, batch 2/non-scan. Its selective sweep
failed the immediate-cut check at batch offset 94 (token-loss max .125); its
94 completed examples are retained for audit, not used as a full Medium/XL
comparison. Locate the arithmetic discrepancy before rerunning or interpreting
that comparison; do not relax the exact check merely to finish the sweep.

The failure was localized by runtime `cfb7716`, `BAM_CONSUMER_AUDIT=1
BAM_DELIVERY_AUDIT_OFFSET=94`: one post-cut residual coordinate differs by
7.45058e-9, already present in the standalone `clean-(clean-deleted)`
reconstruction. The MLP output then differs in two coordinates (max .0004883),
and one sequence's mean loss differs by .0008876. Source/M scope and unused/zero
controls are exact. This is float32 subtraction losing a tiny coordinate, not
evidence that the selected path failed to be removed. A compensated TwoDiff
representation retains high/low parts of the source increment; the updated
probe will rerun the endpoint and full cohort with exact checks. CPU tests
verify reconstruction across 1024 bf16 pairs spanning widely different scales.
Raw audit: `bam-row-mediation-medium-L8-delivery-audit94-cfb7716-rowboth/audit.json`.

#### Historical SoftmaxMix/RmsMix is not a matched diagonal-one comparison

`BamLlama2MediumRmsGateOnlyDynamicMixFull1` and
`BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1` both retain `local_o`, zero the
fetch diagonal, and use **independent** `W_R`/`W_R_gate` and
`W_Ro`/`W_Ro_gate`. They inherit `bam_share_full_local_read=False`,
`bam_combine_full_local_read=False`, `bam_keep_fetch_diagonal=False`, and
`bam_fetch_diagonal_one=False`. Historical implementation verified in commit
`346bb35`, `MaxText/layers/attentions.py`: mixture/masking 1751–1768,
separate projections 1998–2008 and 2069–2080, routing/read 2280–2368.

Their sum is `Read(cross M, r_fetch) + Read(local M, r_local)`, not
`Read(cross M + local M, r_fetch)`. Only the later shared-key/shared-gate variant
admits the latter equivalence. The reported −.0006 RmsMix/SoftmaxMix difference
at 6200 steps therefore does **not** establish that signed mixing is dispensable
under today's diagonal-one/shared-read architecture. Mixture normalization and
initialization also differ in the old pair (softmax/zero vs unit-L2/regular).
The old registries contain comparisons but no sealed runtime hash; `346bb35`
is the checked-in historical implementation, not a newly inferred launch hash.

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
