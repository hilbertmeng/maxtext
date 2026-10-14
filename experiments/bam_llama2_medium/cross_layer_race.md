# Cross-layer projection sharing: nine-arm race


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

## Independent parameter-balance re-audit

On user request, re-enumerated complete parameter trees for all nine arms and both historical parents. Every leaf shape matches the actual worker startup dump. MLP count is21600×sum(widths), with shared-bank, bridge, Up expansion and private MLP-down changes accounted separately. Unified target432111104; all nine choose the nearest reachable total at quantum21600 (max absolute error10368=0.0024%). Parent totals differ slightly from the unified target, so parent-relative remainders need not be within half a quantum.

| ID | Non-MLP delta vs parent | MLP delta vs parent | Total delta vs common target |
|---|---:|---:|---:|
| 310200 | -995,328 | +993,600 | -1,152 |
| 310201 | -2,377,728 | +2,376,000 | -1,152 |
| 310202 | +1,953,792 | -1,944,000 | +10,368 |
| 310203 | +571,392 | -561,600 | +10,368 |
| 310204 | -1,069,056 | +1,058,400 | -3,072 |
| 310205 | -2,912,256 | +2,894,400 | -10,272 |
| 310206 | +503,808 | -518,400 | -7,008 |
| 310207 | -1,339,392 | +1,339,200 | +7,392 |
| 310216 | -4,257,792 | +4,255,200 | +4,992 |

Evidence: `parameter-balance-audit.json`, `worker-parameter-census.json`, and reusable `audit_parameter_balance.py` under `/data0/xd/bam_diagnostics/cross-layer-race-20261014`. No training/config change was needed.

Primary reporting comparator unified across all nine arms: `BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile` (old triple-sharedR384), including gaps, MHA-reference gain ratios and throughput delta against0.465800 steps/s. Old ReadAttentionSharedR256 stays registered as auxiliary; configuration inheritance/runtime is unchanged. Historical pre-run bets and parameter-delta audits retain their explicitly named original parents.

User review update at2026-10-14T07:09Z: first elimination review at3000, replacing blanket no-stop-before5000. Evaluate recent1000–1500-step gain-ratio trend, catchup against leading contenders, and speed/other actual gains; stop clearly dominated non-improving arms, retain credible catchup.

Explicit user control for second elimination: at5000, present recommendations with evidence and late-MLP considerations, then wait for explicit user confirmation before any effect-based stop. No automatic stop at the review milestone. Current restored arms continue their resume workflow.

## First review closeout and correction

- `BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile`: CP3341; stopped. oldTripleR384: final5 mean gap+0.008756; last3200 gap+0.007431; oldPairR256: final5 mean gap+0.007002; last3200 gap+0.005892.
- `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile`: CP3551; temporary stop, subsequently resume requested. oldTripleR384: final5 mean gap+0.008416; last3400 gap+0.007022; oldPairR256: final5 mean gap+0.006812; last3400 gap+0.005827.
- `BamMediumPropMShiftReadG050CrossTripleR512ConservativeBlockTruePile`: CP3561; stopped. oldTripleR384: final5 mean gap+0.009660; last3400 gap+0.008978; oldPairR256: final5 mean gap+0.008056; last3400 gap+0.007783.
- `BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile`: CP3418; temporary stop, subsequently resume requested. oldTripleR384: final5 mean gap+0.010400; last3200 gap+0.009624; oldPairR256: final5 mean gap+0.008646; last3200 gap+0.008085; ReadAttnR384ConservativeThirdsTruePile: final5 mean gap+0.004612; last3200 gap+0.004581; TripleR512ConservativeThirdsTruePile: final5 mean gap-0.001275; last3200 gap-0.000979.

The original four-stop decision was inconsistent: faster recent catchup was used to retain conservative512thirds while stronger current/larger-MLP direct384thirds and square384 were stopped. Two resumes restore consistent treatment. Seven contenders remain; direct384block and conservative512block stay stopped. Second elimination at5000 requires explicit user confirmation.

Complete first-stint leases (UTC; v5p16, UE5a):

| RUN | Lease | Zone | Start UTC | End UTC | Duration | End reason |
|---|---|---|---|---|---|---|
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile` | 1 | `us-east5-a` | 2026-10-14 04:51:49 | 2026-10-14 05:05:52 | 14m03s | preempted |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile` | 1 | `us-east5-a` | 2026-10-14 04:59:28 | 2026-10-14 05:06:01 | 6m33s | preempted |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectBlockTruePile` | 2 | `us-east5-a` | 2026-10-14 05:17:09 | 2026-10-14 07:19:54 | 2h2m45s | review stop3341; node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile` | 2 | `us-east5-a` | 2026-10-14 05:16:56 | 2026-10-14 07:19:58 | 2h3m02s | temporary stop, subsequently resume requested; node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossTripleR512ConservativeBlockTruePile` | 1 | `us-east5-a` | 2026-10-14 04:47:42 | 2026-10-14 07:20:01 | 2h32m19s | review stop3561; node/queue absent, TB SYNC_OK |
| `BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile` | 1 | `us-east5-a` | 2026-10-14 05:12:11 | 2026-10-14 07:20:05 | 2h7m54s | temporary stop, subsequently resume requested; node/queue absent, TB SYNC_OK |

Square384 direct-control correction: against ReadAttnR384 conservative thirds, its gaps at1000/2000/3000/3200 are+.008979/+.002069/+.0032346/+.0045814 (last5 mean+.0046118), with4.1472M fewer MLP parameters (-1.6486%), ~0.33% slower and no M-cache reduction. Wider384 bridge/MLP-shared representation has no demonstrated payoff. Restoring based only on the weaker512conthird comparator was incomplete; square384 remains a priority5000 stop candidate subject to user confirmation. Always inspect closest architectural control as well as common SOTA before deciding.

### Second elimination review timing (user clarification)

Conduct the collective review when most other contenders reach 5,000 steps. Square384 need not reach 5,000: it may be proposed for stopping based on available matched-step comparisons and its full trend. This supersedes the mistaken per-arm-only interpretation. Present the proposed elimination list and wait for explicit user confirmation before execution. Square384 remains a priority candidate. Resumed arms203/216 both verified committed checkpoint3600 with expected TruePile data cursor (seed9876, batch128).

At the second review, all R384-family contenders must also be judged against old read+attention sharedR256, alongside common old tripleR384. Report matched-step gap and MHA-reference gain-ratio trends for both, plus speed and MLP budget. Distinguish within-family improvements from overall competitiveness; square384 retains its closest conservative384thirds control. Both baseline/ratio registrations already exist.

User additionally requires MLP absolute/percentage deltas vs oldR256 for the R384-family comparison. Audited oldR256 MLP=249,177,600 ([3887,3762,3887]); retain the original single-read.5 reference column as well.

4k comparison: R384 conservative block/thirds, direct thirds, square384 gaps vs oldR256 = +.0005186/+.0032884/+.0055356/+.0074182; MHA-reference ratios=.997671/.985232/.975141/.966686. Between3k and4k the first3 ratios are flat/slightly worse; square worsens from.972503. Catchup vs oldTripleR384 must not be conflated with improving vs oldR256. Every future R384 summary must show both baseline gap/gain sequences alongside MLP deltas; initial4k omission was corrected in a supplemental table.

### Second review executed after explicit confirmation

- `BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile`: stopped CP5198; BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile: last5000 gap+0.008843, last5mean+0.009043; BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile: last5000 gap+0.007716, last5mean+0.008159. P1, UE5a v5p-16, no zone switch; resources absent and TB SYNC_OK.

```text
BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile: preemptions=1 ready_leases=2
01           -  us-east5-a  xd-v5p-16-310205-maxtext  - -> 2026-10-14T05:10:42Z  preempted
02    3h12m28s  us-east5-a  xd-v5p-16-310205-maxtext  2026-10-14T05:20:13Z -> 2026-10-14T08:32:41Z  run_stop
```

- `BamMediumPropMShiftReadG050CrossTripleR512DirectBlockTruePile`: stopped CP6200; BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile: last6000 gap+0.002614, last5mean+0.003939; BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile: last6000 gap+0.003191, last5mean+0.003501. P0, UE5a v5p-16, no zone switch; resources absent and TB SYNC_OK.

```text
BamMediumPropMShiftReadG050CrossTripleR512DirectBlockTruePile: preemptions=0 ready_leases=1
01    3h47m09s  us-east5-a  xd-v5p-16-310206-maxtext  2026-10-14T04:47:42Z -> 2026-10-14T08:34:51Z  run_stop
```

- `BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile`: stopped CP4960; BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile: last4800 gap+0.007699, last5mean+0.007783; BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile: last4800 gap+0.006972, last5mean+0.007072; BamMediumPropMShiftReadG050CrossReadAttnR384ConservativeThirdsTruePile: last4800 gap+0.003718, last5mean+0.004017; BamMediumPropMShiftReadG050CrossTripleR512ConservativeThirdsTruePile: last4800 gap-0.001126, last5mean-0.001118. P0, UE5a v5p-16, no zone switch; resources absent and TB SYNC_OK.

```text
BamMediumPropMShiftReadG050CrossTripleR384ConservativeInner384ThirdsTruePile: preemptions=0 ready_leases=2
01    2h07m54s  us-east5-a  xd-v5p-16-310216-maxtext  2026-10-14T05:12:11Z -> 2026-10-14T07:20:05Z  run_stop
02      59m02s  us-east5-a  xd-v5p-16-310216-maxtext  2026-10-14T07:33:42Z -> 2026-10-14T08:32:44Z  run_stop
```

Remaining: conservative384 block/thirds, direct384 thirds, direct512 thirds.

### Direct R384 thirds own5k review closeout

User confirmed stop CP5157; all resources absent and TB SYNC_OK. BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile: last5000 gap+0.005535, last5mean+0.005661; BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile: last5000 gap+0.004407, last5mean+0.004777. No zone switch.

```text
BamMediumPropMShiftReadG050CrossReadAttnR384DirectThirdsTruePile: preemptions=2 ready_leases=4
01      14m03s  us-east5-a  xd-v5p-16-310203-maxtext  2026-10-14T04:51:49Z -> 2026-10-14T05:05:52Z  preempted
02    2h03m02s  us-east5-a  xd-v5p-16-310203-maxtext  2026-10-14T05:16:56Z -> 2026-10-14T07:19:58Z  run_stop
03      41m29s  us-east5-a  xd-v5p-16-310203-maxtext  2026-10-14T07:33:20Z -> 2026-10-14T08:14:49Z  preempted
04      24m59s  us-east5-a  xd-v5p-16-310203-maxtext  2026-10-14T08:22:06Z -> 2026-10-14T08:47:05Z  run_stop
```
