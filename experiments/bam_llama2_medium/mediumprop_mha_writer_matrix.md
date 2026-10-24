# Periodic standard-attention writers on the dedicated2:1 parent

Implementation worktree `/data0/xd/mediumprop-dedicated-m-mlp`, branch
`codex/mediumprop-dedicated-m-mlp`; main `MaxText/exp.py` is ledger-only for this extension.
Parent `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile` (runtime18a9d9c).

The four configuration names share prefix
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriter` and suffix `TruePile`:

| Middle name | Standard-attention layers | Selected ordinary/private widths | L16/L17 ordinary | Total | Delta vs2:1 |
|---|---|---|---|---:|---:|
| Every4KeepMemory | 1,5,9,13,17 | 2029/887 | 3929/3174 |432112080|-1200|
| Every4DropMemory | 1,5,9,13,17 | 3043/0 | 3929/3175 |432112480|-800|
| Every3KeepMemory | 2,5,8,11,14,17 | 2029/887 | 3929/3174 |432112320|-960|
| Every3DropMemory | 2,5,8,11,14,17 | 3043/0 | 3929/3175 |432111920|-1360|

Other productive layers retain2535/1140. Standard-attention sites restore full
1200x1200 Q/K/V, retain the existing output projection and partial RoPE18 coordinates,
and have no Q/K/V/O M read, read key/gate/static parameters or read-only M-shift.
Their attention write retains the parent's content, independent R256 address, gate and
RMS recipe. Keep arms add the independent post-attention private SwiGLU write;
drop arms omit its parameters and operations entirely, without ordinary-MLP write fallback.
Body selected keep budgets: ordinary7304400; complete private3651200 including458000
controls, ratio2.000986. Local remaining-budget rounding costs+240/keep site or-560/drop.
Restoring dense attention net costs2732640 =1.897667 W_Q per productive selected site.

L17 reads no M, so L16 is the final M consumer. Both L16/L17 omit attention writes,
private M FFNs and controls. L16 keeps every BAM read. Every4 scans four repeated
four-layer blocks with two explicit tail layers; Every3 scans five repeated three-layer
blocks with three explicit tail layers. Tail widths repay all rounding residuals.

The writer attention uses ordinary QKV attention and returns raw write factors without
reading/transposing M; outer fusion reuses the unchanged v8t write kernel. Ordinary BAM
sites retain the v8t read core. Empty terminal write groups bypass the write kernel.
Both XLA and Pallas paths support the same parameter tree and mathematical circuit.

Training inherits T4096/global128/per-device16/BF16/full remat/13500 steps, tuned
Splash1024/fused backward, Pallas v8t/48MiB/fused inputs, generic health ON and
internal/BAM metrics OFF. Checkpoint200/latest2, TensorBoard rotation500.
Infrastructure audit against main1bb634ae: main-only commits since the shared ancestor
do not change train/AOT/optimizer/checkpoint/summary-writer/default infrastructure.
Retained compiler llm-jax-v6e-1-0/europe-west4-a: FLEX_START, ACTIVE and idle verified;
borrowed with worker-level serialized AOT, never lifecycle-owned.
Training zone us-east5-a, supported by the current long P0 lease and recent family leases.

Direct comparisons: all vs independent2:1; drop vs keep at each period, Every3Keep vs
Every4Keep, Every3Drop vs Every4Drop. All registered gain multipliers use
`BamMHAMediumPropC256TruePile` as reference. Reports early1000 then2000, full13500 plan;
no scheduled2800/5000 stop reviews, as requested by the user. Cumulative plots begin600 and include gaps plus gain multipliers;
no detailed per-window data tables in human reports.

Pre-run terminal multiplier bets versus independent2:1: Every4Keep1.03,
Every3Keep1.02, Every4Drop1.015, Every3Drop1.00. Throughput prior roughly flat to
slightly faster; verify the actual matched-health steady rates after first steps.
Artifacts `/data0/xd/bam_diagnostics/mediumprop-mha-writer-matrix`.

## Verified launch

Frozen runtime `ad12938bf9be13df9d84c262ce461da25d5ab018`;57 CPU checks passed, including all four
formal budgets, no-M-read/branch-isolation checks, both scan periods/causality/live
writers,47 BAM regressions and5 rotating-summary tests. Complete JAX/Pallas logits
and all-parameter gradients agree; worst relative gradient error4.51619e-6.
All four exact v5p16/13500 AOTs compiled under the same borrowed worker lock;
FLEX compiler preserved. All FIRST_STEP, worker Git/AOT/config and20-99 windows
verified, fresh from0, T4096/global128/per-device16, normal health settings.

- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery4KeepMemoryTruePile`: `xd-v5p-16-310350-maxtext`, us-east5-a; 0.717172step/s20-99, about+1.44% versus independent2:1 startup.
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery4DropMemoryTruePile`: `xd-v5p-16-310351-maxtext`, us-east5-a; 0.736744step/s20-99, about+4.21% versus independent2:1 startup.
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery3KeepMemoryTruePile`: `xd-v5p-16-310352-maxtext`, us-east5-a; 0.726750step/s20-99, about+2.79% versus independent2:1 startup.
- `BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriterEvery3DropMemoryTruePile`: `xd-v5p-16-310353-maxtext`, us-east5-a; 0.737121step/s20-99, about+4.26% versus independent2:1 startup.

Startup/config/queue/AOT proofs and monitoring ownership remain in the artifact directory.

## Closeout (2026-10-24 UTC)

All four stopped on user direction; no full13500 endpoint claim. Every4 slots
were retained and transferred to the new plain-V Every3 pair only after actual
FIRST_STEP. Every3 controls were separately checkpointed/released; both node and
queue independently absent. All final commit markers verified, no lost tail,
all four TensorBoards SYNC_OK. Exact leases are in the regional history.

| Arm | Stop | Final complete window | Last5 gap vs2:1 | Last5 MHA-reference multiplier |
|---|---:|---:|---:|---:|
| Every4 Keep | 4464 | 4400 | +0.02334176 | 0.893151 |
| Every4 Drop | 4847 | 4800 | +0.02524492 | 0.879617 |
| Every3 Keep | 5027 | 5000 | +0.01995744 | 0.903377 |
| Every3 Drop | 4505 | 4400 | +0.01992624 | 0.908829 |

The initial deficits narrowed then flattened around+.02–.025; all four
missed their positive terminal-gain priors through the stop. Every3 was better
than Every4 at matching steps, but still below independent2:1. Every4 private
MLP retention helped slightly; Every3 private retention/drop converged to near
parity, with a small final sign crossing rather than a durable advantage.
Neither small startup speed gains nor equal M-cache offset the loss deficit.
New hypothesis: preserve BAM Q/K/O addressing and replace only V contents.
Artifacts: `/data0/xd/bam_diagnostics/mediumprop-standard-v-writers/`
(`final-mha-summary.json`, closeout/lease/independent-verification proofs).
