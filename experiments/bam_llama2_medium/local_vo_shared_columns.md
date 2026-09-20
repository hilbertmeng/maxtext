# LocalV/LocalO share one gated column read

Worktree `/data0/xd/llf-parameter-matched`, branch `codex/llf-parameter-matched`.
Direct baseline `BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer`
(runtime3351a0b, UE5a .6758 step/s at10-14, generic+concat health ON).
Only L layers change. Keep Q/K concatenation with independent ungated static keys,
standard V64, additive V/O injection in first32, original front32 writeback,
all row reads removed, M32x32/C8, FetchedO, original optimizer/WD and13500-step schedule.
No VO static route and no mixed writeback; this is independent of V-concat experiments.

| RUN | Single shared donor | L/L/F MLP | Total params | Delta vs MHA411616256 | TPU ID |
|---|---|---|---:|---:|---|
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedRank4MLPPerLayer | full M LocalV rank4, scale.1, gate.05 | 2858/2858/2874 | 411612800 | -3456 | qkstatic-vo-shared-r4 |
| BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer | C8 LocalO dynamic, scale.2, gate.05 | 2879/2879/2874 | 411598464 | -17792 | qkstatic-vo-shared-c8 |

The donor's output already includes its gate. Evaluate once, reuse in V and O.
Full-M donor removes LocalO W_R/gate and unused L compression, saving147728/L
(.1408843994140625 W_Q),2363648 total before MLP credit.
C8 donor removes LocalV read/mix/gate projections and biases, saving213136/L
(.2032623291015625 W_Q),3410176 total before MLP credit.
Nearest per-layer integer widths, no hardware rounding. Relative to direct baseline
411617152 params: -4352 and -18688, respectively; cache unchanged.
Existing gate/amplitude health exported for BOTH destinations, with identical shared gate stats.
No additional health scalars beyond the920 of direct QK-static baseline.

Training primary UE5a; backups UC1a/EW4b after5min. AOT EW4a, then UC1a/UE5a.
No old runs stopped as part of launching these arms.
C8 pre-run bet vs QK static: gap+.002, speed+3%.

Validation: actual parameter trees `/data0/xd/vo-shared-audit.json`;
full24-layer train-step trace `/data0/xd/vo-shared-trace.log`;
pinned BAM suite `/data0/xd/vo-shared-tests.log`.
New numerical test checks unused parameters absent, exactly one shared read call,
correct donor call counts, V/O gate identity, active first32/nonzero reads,
finite nonzero donor gradients, and unchanged F V/fetched parameters.

Runtime `5d535d3`; both started on UE5a, AOT loaded and FIRST_STEP verified.
Steps10–14: rank4 .6886 step/s (+1.89% vs QKStatic .6758), C8 .6890 (+1.95%).
All three use matching generic+concat health; both compiler states ready with no cleanup failures.

Rank4 paused at committed3338 for user hot replacement by independent-gate C8.
Gap vs QKStatic: +.103227@200 shrank to +.0162@2800–3200; last5 mean+.016680.
At2800 it remained+.015909 worse than SharedC8 with essentially equal speed and unchanged cache.
