# Cross-layer projection sharing: nine active arms


Current authorized scope: **original eight plus the square-R384 conservative thirds
arm, nine total**. The agent incorrectly expanded “add missing controls” to eight
additional MLP toggles. The user challenged the count; IDs310208–310215 are withdrawn
and never trained. Their classes remain only as implementation/test ancestry, not
active experiments. No new formal training resources were created for those eight.
Four owned managed compilers have been released, with node and queue absence verified
on 2026-10-14. Two completed required conservative sharding repair AOTs before release.
Reserved Flex is preserved.

Original direct arms retain f8755258. Four original conservative arms and the new
square bridge use38af4cc1. Original single-device gate missed replicated bridge
parameters; all17 defined configurations now pass the actual eight-chip shard-size
bound, and all9 conservative paths pass forward/gradient/causality checks. Production
tolerance is unchanged. The completed full47 regression from f8755258 still applies
to unchanged model paths.

The square bridge's active comparison set is original ReadAttnR384 conservative
thirds, TripleR512 conservative thirds, and both historical sharing winners. The
withdrawn TripleR384/inner256 arm is not an available training baseline. Accordingly,
only its comparison with TripleR512 holds inner width384 and MLP participation fixed.

Implementation: `/data0/xd/mediumprop-projection-sharing`, branch
`codex/mediumprop-projection-sharing`. Main infrastructure audited against
`0f7f6d10a2760c205699cf34f9834c892b4926d0`, an ancestor of this worktree.
No main-only runtime fixes are missing. Train diff adds compact shift metrics only;
TB rotating writer uses period500 and begin_step before scalar writes.

User-authorized 2026-10-14: two architecture families × conservative/direct ×
3-layer blocks/6-layer contiguous thirds. Global, halves, rank16 adapters excluded.
All runs train from scratch for13500 steps, TruePile4096, batch128, same seed,
optimizer and schedule as their parent. Checkpoints every200, keep2, no permanent
retention. Normal health plus inherited compact M-shift summaries; no new health tags.
Report every1000, native200-step windows; first substantive effect screen at5000,
continue improving contenders rather than infer terminal failure from early loss.

The first axis is NOT a checkpoint conversion or pure rank-only comparison:
ReadAttnR384 retains independent MLP-write R256; TripleR512 includes MLP writes
in the bank. Conservative inner widths are256/384; direct inner widths384/512.

A single decoder parameter has shape [groups,1200,rank], broadcast into scan;
all consumers in its group differentiate into that same leaf. Other layer parameters
remain independent. No fake sharing by equal initialization or duplicated scan leaves.
A uses the parent's normal(.006) initialization independently per group. Conservative
C=C0+delta, with fixed orthonormal DCT columns and zero delta: initial effective Down
has parent marginal variance; only A and delta get normal weight decay. C0 is not a
parameter. C0+delta is formed before one contraction. Direct Up uses parent initializer.
Both affected read and write consumers use the same projection; MLP evaluates its
own post-attention input. No nonlinear feature cache reused across different inputs.

Total target432111104. Integer MLP granularity21600; actual shape-count gate verifies
all entries. W_Q=1200^2=1440000. Savings below include expanded Up costs, bridge costs,
and the shared Down; positive savings refunded to MLP, negative deducted.

## Frozen pre-run bets

These are terminal loss/steady speed hypotheses against each direct parent, not
measured results. Leading ceiling bet: TripleR512/direct/thirds. Conservative expected
safer early convergence; direct may require longer co-adaptation. Diagnostics support
adjacent groups and nonlinear layer specialization, but do not establish train ordering.

| RUN | MLP widths | Parameters | Saved projection W_Q | Terminal gap bet | Speed delta bet |
|---|---|---:|---:|---:|---:|
| `BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeBlockTruePile` | [3903, 3777, 3902] | 432109952 | 0.6912 | -0.003 | -3% |
| `BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile` | [3924, 3798, 3924] | 432109952 | 1.6512 | -0.005 | -3% |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile` | [3857, 3732, 3857] | 432121472 | -1.3568 | +0.002 | +0% |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile` | [3879, 3753, 3878] | 432121472 | -0.3968 | -0.003 | +0% |
| `BamMediumPropMShiftReadG050CrossTripleR512ConservativeBlockTruePile` | [3811, 3685, 3810] | 432108032 | 0.7424 | -0.004 | -5% |
| `BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile` | [3839, 3713, 3839] | 432100832 | 2.0224 | -0.006 | -5% |
| `BamMediumPropMShiftReadG050CrossTripleR512DirectBlockTruePile` | [3786, 3661, 3786] | 432104096 | -0.3499 | -0.003 | -1% |
| `BamMediumPropMShiftReadG050CrossTripleR512DirectThirdsTruePile` | [3815, 3689, 3815] | 432118496 | 0.9301 | -0.007 | -1% |

Both historical sharing winners remain direct baselines for every arm. Integrate gain
triples (this RUN, each old winner, BamMHAMediumPropC256TruePile) in registry reporting;
report sequences of gaps and MHA-reference gain multiples. Include matched-mode scope
and matched-scope mode comparisons as new run windows mature. Training zone UE5a based
on recent8-hour uninterrupted parent leases; no new FLEX training requests. Retained
Flex `llm-jax-v6e-1-0` in EW4a is borrowed for AOT only, never deleted.

## Withdrawn expansion: eight MLP controls never trained

The agent over-expanded the instruction into eight MLP-sharing counterparts
(IDs310208–310215). This expansion was withdrawn; the definitions below are archived,
not an authorized training plan. Unlike the alternative design,
conservative inner rank remains256 for sharedR384 and384 for sharedR512. Private MLP
Down also keeps the native family rank256/384; when shared in direct mode its Up input
expands with the shared bank. Thus the MLP toggle includes this width change in direct
mode, with exact total-parameter compensation; it is not an equal-width weight-tying-only
intervention. Report that boundary rather than attribute every difference to tying alone.
No changes to the first eight's source, initialization, parameter counts, or AOT hash.

| Added RUN | MLP widths | Params | Paired gap bet |
|---|---|---:|---:|
| `BamMediumPropMShiftReadG050CrossTripleR384ConservativeBlockTruePile` | [3932, 3805, 3930] | 432102752 | +0.004 |
| `BamMediumPropMShiftReadG050CrossTripleR384ConservativeThirdsTruePile` | [3953, 3826, 3952] | 432102752 | +0.004 |
| `BamMediumPropMShiftReadG050CrossTripleR384DirectBlockTruePile` | [3880, 3754, 3879] | 432118688 | -0.004 |
| `BamMediumPropMShiftReadG050CrossTripleR384DirectThirdsTruePile` | [3902, 3775, 3900] | 432118688 | -0.004 |
| `BamMediumPropMShiftReadG050CrossReadAttnR512ConservativeBlockTruePile` | [3769, 3642, 3767] | 432108032 | +0.004 |
| `BamMediumPropMShiftReadG050CrossReadAttnR512ConservativeThirdsTruePile` | [3797, 3670, 3796] | 432100832 | +0.004 |
| `BamMediumPropMShiftReadG050CrossReadAttnR512DirectBlockTruePile` | [3750, 3624, 3750] | 432121280 | +0.004 |
| `BamMediumPropMShiftReadG050CrossReadAttnR512DirectThirdsTruePile` | [3779, 3652, 3778] | 432114080 | +0.004 |

## Startup sharding repair and square-bridge arm

The original conservative bridge used replicated axes(None,None), which passed
single-device shape/gradient tests and AOT but failed real eight-chip startup's
unchanged2% parameter-sharding tolerance (R512block:5.94% excess per-chip parameters).
Fix the bridge input axis to `embed`, matching native DenseGeneral down sharding.
Do not relax the guard. Full parameter shape tests now also calculate every leaf's
per-chip shard size on an eight-device FSDP mesh and enforce the production bound.
Original direct configurations are unaffected and keep their sealed runtime.
Conservative configurations must recompile with this fix; preserve training resources.

User also authorized one R384 conservative square384 bridge with shared MLP writes.
Use contiguous thirds, `BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile`,
ID310216. MLP[3860,3734,3860],432116096params, projection savings4257792=2.9568W_Q.
Compare to R384/inner256 and R512/inner384 conservative triple-thirds plus both old
sharing winners. Terminal bet gap−.004 versus inner256, −.005 versus old tripleR384;
expected speed−4% versus old tripleR384. No checkpoint conversion: train from scratch.

Monitoring format: one nine-row table per1000-step milestone, cumulative gap/gain-ratio trends retained compactly. XL remains every2000 steps. All nine registries carry MHA-reference ratios against both historical sharing winners.

User monitoring refinement: no routine TB sync or M-shift gate report for these nine projection-sharing runs. Routine monitoring uses registry loss/gain/speed/checkpoint state; targeted health only for material anomalies, TB archive at closeout.

## Verified launch and speed evidence (2026-10-14)

All nine passed actual v5p16 AOT loading, full parameter-count verification and FIRST_STEP.
Runtime remains f875525 for direct and38af4cc for conservative; later documentation commits do not alter running source.

| ID | Runtime | 80-step mean steps/s | Delta vs historical parent | Sample steps |
|---|---|---:|---:|---|
| 310200 | 38af4cc | 0.463188 | -0.27% | 20–99 |
| 310201 | 38af4cc | 0.460913 | -0.76% | 20–99 |
| 310202 | f875525 | 0.463138 | -0.28% | 60–139 |
| 310203 | f875525 | 0.462150 | -0.49% | 272–351 |
| 310204 | 38af4cc | 0.464487 | -0.28% | 20–99 |
| 310205 | 38af4cc | 0.462262 | -0.76% | 20–99 |
| 310206 | f875525 | 0.464587 | -0.26% | 20–99 |
| 310207 | f875525 | 0.459975 | -1.25% | 135–214 |
| 310216 | 38af4cc | 0.459387 | -1.38% | 20–99 |

Raw startup logs and verified-startups.json/verified-speeds.json reside in `/data0/xd/bam_diagnostics/cross-layer-race-20261014`. Recovery samples skip the first20 steps after resume; checkpoint/I/O outliers remain in each raw80-step mean. No unmatched health-off timing control is used.
