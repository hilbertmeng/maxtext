# UnitH raw content writes: embedding amplitude sweep

Runtime worktree `/data0/xd/mediumprop-unit-h-raw-content`, branch `codex/mediumprop-unit-h-raw-content`. Fresh main base and operational audit `a5eeb5f6`: no historical infrastructure replay; writer rotation500, begin_step, checkpoint200/latest2 and source/AOT sealing inherited. Existing v8t raw-content-divisor path reused; no kernel/model-source edits.

Direct comparator is `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneTruePile` (completed13500, historical aa7d8ec). MHA reference `BamMHAMediumPropC256TruePile`.

All three start from scratch, keeping UnitH learned VO coefficients init1, attention/private-MLP content divisor1, normalized addresses and original sigmoid gates. Embedding vector parameters still use .006 initialization; ONLY embedding-to-M content divisor changes: 1 / .0016052833 / .032105666 (raw / parent amplitude / parent amplitude divided20). No M pre-norm, no QKNorm, no adapters. Native initial seed-M scales must be 1:.0016052833 and fixed:small20=20:1 as appropriate; all parameter initializations match across arms.

All have432112210 parameters (no delta), ordinary/private/tail widths2510/1128/3904; first17 single-layer scan plus explicit tail, no final attention/private writes. T4096, formal v5p16, per-chip batch16,13500-step LR plan; tuned Splash, Pallas read+write v8t48MiB, full remat. Generic training health ON, heavy internal/BAM metrics OFF; existing signed18 VO scales retained.

IDs310540/310541/310542; intended owned TPUs `xd-v5p-16-310540-maxtext`, `xd-v5p-16-310541-maxtext`, `xd-v5p-16-310542-maxtext`; primaryUE5a, backupEW4b. Latest comparable Medium leases include clean six-hour completions inUE5a; borrowed user FLEX EW4a compiler has no lifecycle ownership.

Pre-launch bets versus UnitH, matched MHA-reference terminal gain multiplier: raw.98, parent amplitude.99, small20 1.01. Expected speed near parent.6761step/s or slightly faster after attention content RMS removal. Middle seed amplitude is preferred: retain a useful seed without strong initial recycling; predictions are subjective, not diagnostic results.

Registry group `mediumprop-unit-h-raw-content`, default public cadence2000 and review milestones4000/8000; review provides advice and waits for user confirmation before any loss-based stopping. Shared cumulative gap/gain figure from600 against UnitH only. Operational artifacts `/data0/xd/bam_diagnostics/mediumprop-unit-h-raw-content/runtime.json`. No routine progress diary in exp.py.
