# UnitH raw content writes: embedding amplitude sweep

Runtime worktree `/data0/xd/mediumprop-unit-h-raw-content`, branch `codex/mediumprop-unit-h-raw-content`. Fresh main base and operational audit `a5eeb5f6`: no historical infrastructure replay; writer rotation500, begin_step, checkpoint200/latest2 and source/AOT sealing inherited. Existing v8t raw-content-divisor path reused; no kernel/model-source edits.

Direct comparator is `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneTruePile` (completed13500, historical aa7d8ec). MHA reference `BamMHAMediumPropC256TruePile`.

All three start from scratch, keeping UnitH learned VO coefficients init1, attention/private-MLP content divisor1, normalized addresses and original sigmoid gates. Embedding vector parameters still use .006 initialization; ONLY embedding-to-M content divisor changes: 1 / .0016052833 / .032105666 (raw / parent amplitude / parent amplitude divided20). No M pre-norm, no QKNorm, no adapters. Native initial seed-M scales must be 1:.0016052833 and fixed:small20=20:1 as appropriate; all parameter initializations match across arms.

All have432112210 parameters (no delta), ordinary/private/tail widths2510/1128/3904; first17 single-layer scan plus explicit tail, no final attention/private writes. T4096, formal v5p16, per-chip batch16,13500-step LR plan; tuned Splash, Pallas read+write v8t48MiB, full remat. Generic training health ON, heavy internal/BAM metrics OFF; existing signed18 VO scales retained.

IDs310540/310541/310542; intended owned TPUs `xd-v5p-16-310540-maxtext`, `xd-v5p-16-310541-maxtext`, `xd-v5p-16-310542-maxtext`; primaryUE5a, backupEW4b. Latest comparable Medium leases include clean six-hour completions inUE5a; borrowed user FLEX EW4a compiler has no lifecycle ownership.

Pre-launch bets versus UnitH, matched MHA-reference terminal gain multiplier: raw.98, parent amplitude.99, small20 1.01. Expected speed near parent.6761step/s or slightly faster after attention content RMS removal. Middle seed amplitude is preferred: retain a useful seed without strong initial recycling; predictions are subjective, not diagnostic results.

Registry group `mediumprop-unit-h-raw-content`, default public cadence2000 and review milestones4000/8000; review provides advice and waits for user confirmation before any loss-based stopping. Shared cumulative gap/gain figure from600 against UnitH only. Operational artifacts `/data0/xd/bam_diagnostics/mediumprop-unit-h-raw-content/runtime.json`. No routine progress diary in exp.py.

## Startup verification

Runtime94c0fa1a87ab96126836e0f1bd10fd53965b773c. All three FIRST_STEP, exact source on both workers, loaded AOT, TruePile4096 regional path and original13500 plan verified. Five focused experiment checks (all three full-model CE/all-gradient native/Pallas plus budget and exact seed-M dose), five summary-writer rotation regressions passed; maximum leaf gradient relative discrepancies6.76e-6/3.58e-6/2.24e-6. Parameters432112210 each.

Startup harmonic throughput, steps20..99 (80 points, worker0): raw embedding .681537, parent-amplitude .684309, small20 .690536 step/s. Relative to UnitH .6761125: +.80%/+1.21%/+2.13%; same hardware/batch/health, separate leases, no claim these small between-arm differences are causal. All logged startup losses finite. All three AOTs borrowed the same retained FLEX worker through its serial lock; compiler preserved. Monitor watcher4111317 owns group metadata, routine2000/reviews4000,8000; loss-based stops await user confirmation.

## Uncompensated embedding arm: user stop

Stopped at committed checkpoint4578 on2026-11-03 after the user first retained all three at4k, then explicitly stopped only this arm. The absolute deficit remains near+.10; MHA-reference relative gain continues falling. Final five mature windows3600..4400 average gap+0.09874924, ratio0.59917621 versus UnitH; latest4400 +0.09823820/0.58573265. The terminal.98 bet was overoptimistic relative to this observed prefix;13500 endpoint was not measured. No parameter/cache savings; startup+.80% on a separate matched-health lease.

UE5a v5p16/P0; no migration or passive candidates. Registered active stint05:22:24..07:18:37UTC; complete READY lease05:22:27..07:18:37UTC (1h56m10s), ending at user-stop request. Node and queued resource independently absent after native closeout finished07:21:20UTC. CP4578 committed; local TensorBoard SYNC_OK. Raw-loss/lease/closeout proofs under the task artifact root.

Other two arms remain in training to8k. Group clock transferred to the parent-amplitude arm at the already-published4k phase; normal6k/8k reports preserved. Stopped arm removed from active membership.

## Remaining two arms: user stop near8k

Both were stopped at the user's explicit instruction, after continuing past the4k keep decision. All final checkpoint commits, native loss caches, independent node+queue absence and local TB SYNC_OK are verified. Both are UE5a v5p16/P0, no migration or passive candidates. READY intervals below are UTC and end at stop requests; native resource cleanup finished08:41:41UTC.

| ID | Checkpoint | Mature horizon | Final5 gap vs UnitH | MHA-reference gain | Entire READY lease UTC |
|---|---:|---:|---:|---:|---|
| 310541 | 7813 | 7600 | +0.04507084 | 0.77732788 | 2026-11-03 05:23:36 to 2026-11-03 08:38:53 (3h15m17s) |
| 310542 | 7951 | 7800 | +0.05543088 | 0.72373672 | 2026-11-03 05:22:22 to 2026-11-03 08:38:55 (3h16m33s) |

The parent-amplitude arm remains best among the three raw-attention-write variants. Absolute deficits slowly narrow, while relative gains settle near.78/.72; neither shows sustained recovery toward UnitH. Removing only attention content RMSNorm at the parent's embedding amplitude is already harmful. Decreasing the embedding seed adds a persistent deficit. The terminal.99/1.01 bets were overoptimistic relative to these prefixes; no13500 endpoint was measured. This tests direct RMS removal, including its amplitude change, rather than an energy-matched test of retaining token/head amplitude information.

All three retain432112210 parameters, unchanged widths2510/1128/3904 and unchanged M-cache. Small startup speed improvements do not offset the observed loss deficits. The implementation remains on the sealed runtime branch; no kernel adaptation was needed. Native complete final curves/statistics, lease histories, closeout summaries and the combined final PNG provenance are under `/data0/xd/bam_diagnostics/mediumprop-unit-h-raw-content/`.
