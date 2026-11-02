# Unit H: 300B-token training

Configuration: `BamPythia2p8BHUnit300B`.
Implementation worktree `/home/xd/projects/maxtext`, branch `refactor-bam`.
The user authorized launch on 2026-11-01; the runtime is sealed on current main.
The durable orchestration settings are in
[the launch/report plan](pythia2p8_h_unit_300b_plan.json).

## Model identity

Direct architecture parent:
`BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOne`.
Retain its D2560/L32/N32, M80x64, all R640 projections, independent V/O addresses,
negative attention diagonal, full 100-dimensional QKNorm before RoPE20, serial
GELU/LayerNorm and single read-only M-shift .5. Keep ordinary/private MLP widths
7808/2944, terminal 9984, scan 31 plus one explicit tail, Pallas v8t and minimal
remat. Keep the learned V/local O common scale initialized to 1, raw private-MLP
content, fixed embedding divisor .0020825863 and normalized attention-write
content. No dynamic-O gamma, extra Q/K read scalar or private-MLP bias.

Embedding/BAM kernels retain .006 initialization; standard Transformer kernels
retain Pythia initialization. Semantic zeros, gate biases and norm scales remain
unchanged. Total parameters remain 2775190912, exactly equal to the parent;
MLP budgets and inference M-cache are unchanged.

The sole numerical model change is the attention logit multiplier:
`bam_qk_logit_scale_dim='qk_dim'` selects the actual concatenated Q/K width,
currently 1/sqrt(100)=0.1 instead of 1/sqrt(80). Derived architectures follow
their actual Q/K width automatically; no inherited hardcoded 100 remains.
The V/output `head_dim` remains 80; concatenated Q/K width remains 100.
The multiplier is applied after full-head QKNorm, once, in both native JAX
and Pallas paths. Legacy classes keep `None`, preserving their existing scale.
The existing score-health observer uses the same denominator when enabled.

## Long-training plan

Restore the old long RUN's training plan, rather than overriding only `steps`:

| Setting | Value |
|---|---|
| Topology | v5p-256: 128 JAX devices, 32 hosts |
| Global batch / sequence length | 1024 / 2048; accumulation 1 |
| Effective per-device batch | 8 |
| Optimizer updates / LR schedule | 143000 / 143000 |
| Training tokens | 299892736000 (about 300B) |
| Peak LR / warmup | 1.6e-4 / 1430 updates (1%) |
| Decay | Existing affine cosine to 0.1 times peak, exactly as old BAM |
| Optimizer | AdamPax, betas .9/.95, epsilon 1e-8, WD .1; inherited clipping |
| Checkpoints | Every 1000; latest two; keep_period 0; force final |
| Permanent checkpoints | 3000, 13000, ..., 143000 |
| TensorBoard / evaluation | Rotate every 1000; periodic evaluation disabled |
| Health | Generic training ON; internal NN and BAM activation health OFF |

This is a fresh-initialization experiment, not continuation of the 28k short
schedule. Preserve the actual full 143k schedule on any later restart.

FSDP128 audit: 22968832 parameters/device versus ideal 21681179; overhead
5.93904%, caused by 1297792 intentionally replicated small parameters.
Large matrices are correctly sharded and their sharded dimensions divide 128.
Use `sharding_tolerance=.06`, rather than the short-run .05 guard. This is a
validation threshold, with no change to model computation or initialization.
The CPU shape audit shortens sequence length and disables fused kernels only
for metadata inspection; it does not claim target-TPU memory/AOT verification.

## Baselines, coordinates and reporting

Keep these five comparisons:

- `Pythia_2.8B_2022`: common reference for all gain multipliers.
- `Pythia_6.9B_2022_Decay`.
- `PileMUDDPythia3BPlusOcdbtv6e`: historical parallel-MLP Mudd, WD .01.
- `BamPythia2p8BSotaIndependentR768Pallas`: old 300B-plan BAM, runtime 1ac1183,
  paused at checkpoint 37042. Never extend its observed curve past its data.
- `MuddPythia3BSerialMLPKeepActivations`: serial-MLP Mudd, WD .1; actual RUN
  `MuddPythia3BSerialMLP`, runtime 9cdcbd0. Its original every-step snapshot
  reaches native 57669, with mature 500-step windows through 57500.

Use common training-step comparison coordinates. The new RUN, old MaxText BAM,
and serial MaxText Mudd baseline need offset +1; external Pythia/PaxML histories
stay at 0. Official FP16 Pythia iteration counts include overflow-skipped updates,
so the common axis is not strictly a count of successful optimizer updates.
Applying the offset only to the new RUN would recreate the old one-step error
in the new-versus-old BAM comparison. Keep native caches/checkpoint labels
unchanged. Register the gain triples from the JSON plan;
use the shared registry calculations, not a separate gain formula.

User-confirmed long-run cadence: reports at 3k, 8k, 13k, 18k, ... (every 5k).
The first normal report also verifies the permanent 3k checkpoint. Routine reports show the
latest 5k, excluding the initial <2k transient, and retain complete raw caches
for longer-window investigation. Permanent-checkpoint and midpoint reports
alternate. Report final 143k without waiting for nonexistent 143025 samples.
Text/table summaries retain the latest 5k window. The user-selected figure view
instead plots four separate figures, each about 16:10: H/old BAM/both Mudds
loss minus common MHA; H and old BAM gain over parallel Mudd; and H and old BAM
gain over serial Mudd; and H gain over old BAM. All gain figures use the common MHA reference. Each
figure has one shared coordinate system for all its curves; its horizontal
extent follows its own complete data (parallel Mudd 143k in the loss figure,
old BAM about 37k in the gain figures). H ends at its actual progress. Start
all four at 4k. Omit 6.9B from figures while retaining
its registered numerical comparisons. Preserve historical figures; this view
starts with the 18k report. The read-only exporter is `export_full_context.py`
in the launch artifact directory and uses registry arithmetic without changing
any baseline registration/cursor. Mudd's 143k terminal point uses the shared
window helper on the final three real samples, explicitly marked in provenance.
Serial Mudd's all-step snapshot is used directly: its sampled native TB has no
observations at the +1-aligned common grid and must not be interpolated. A
10k-20k alignment check gives loss correlation .99934 at offset +1 versus .68690
at offset 0. Native snapshot/TB agreement is within 5e-7 worker-log rounding.

Reporting lives in registry group `pythia28-h-unit-300b-1101`, not in model math.
The group starts at 3k without advancing any undelivered report cursor.

## Checkpoint rescue audit

Current main includes `CoordinatedCheckpointSaver` in the actual training loop,
with `checkpoint_save_timeout_seconds=1200` passed to Orbax AsyncOptions.
The ordinary persistent-manager path is active for this configuration; emergency
checkpointing is disabled. Previous-save and preemption-save failures coordinate
across ranks, attempt one fresh-directory save of the live state/cursor and exit
resumably after verification. Persistent failure does not report success.

The deployed tpu-ag auto-train, `checkpoint_rescue.py` and incomplete-checkpoint
repair helper match canonical `/home/xd/projects/xd_tpu_scripts` SHA256 hashes.
The outer pending-save trigger is 1300 seconds; rescue has a 600-second upper
bound, after which worker exit must be verified before repair/relaunch.
These are enabled defaults, not flags waiting for an additional patch.

The production recovery module SHA256 is
`b2460321665a54d9252241051df372dce0c525bdc5ce51696f3ee82f27ed9d64`, identical
to the previously validated two-rank real-Orbax/GCS production probe. Those probes
used shortened 1-second local and 10-second GCS timeouts and checked transient,
early-signal, persistent-storage and cursor failures, including restored state
and cursor consistency. Evidence:
`/data0/xd/bam_diagnostics/checkpoint-recovery-implementation-20261031/validation-audit.json`.
The current preparation reruns protocol, retention, rotation and outer-controller
tests; it does not launch a new fault-injection job or claim that this unstarted
143k RUN has already exercised a timeout in production.

Preparation artifacts:
`/data0/xd/bam_diagnostics/h-unit-300b-prep-20261101/`.

## Checks completed

Effective configuration was resolved with the v5p-256 topology hint: 128 target
devices, batch 8/device, global batch 1024, T2048, checkpointing/async saves ON
and emergency checkpointing OFF. No backend compilation was requested. Actual
LR values at warmup, middle, late and final steps exactly match the old BAM
long-plan class. Configuration differences from H are limited to the requested
training plan, baselines, sharding guard and Q/K logit denominator.

The new tests verify the complete parameter count/sharding budget, all intended
configuration overrides and report offsets, validation of the scale option,
bitwise-identical initial parameter trees, and actual attention-boundary scaling
in native JAX and fused Pallas. The full scanned-model native/Pallas forward and
all-parameter gradient check has worst relative leaf error 1.949e-6. The 47 shared
BAM regressions, checkpoint recovery/retention and TB rotation tests, and 22
outer-controller/step-offset tests pass.

The preparation artifact directory retains the resolved configuration, exact
class diff, partition specs, LR samples, rescue audit, and individual test logs.
TPU memory fit, actual loaded AOT, first-step execution and throughput remain
launch-time checks; they are not claimed by this CPU-only preparation.


## Launch identity and pre-run prior

RUN `BamPythia2p8BHUnit300B`, runtime
`fb915b61c7326cd5c680e740c2ff9e4409abd142`, main `refactor-bam`.
Owned training TPU `xd-v5p-256-311101-maxtext`, ID 311101, primary UE5a
(`us-east5-a`), no submitted backup resources. Use the zone-local legacy T2048
Pile replica. The retained `llm-jax-v6e-1-0` compiler in `europe-west4-a`, worker 0,
was verified FLEX_START/guaranteed and idle; this task does not own its lifecycle.

### Kernel runtime handoff at 26k

The logical RUN remains `BamPythia2p8BHUnit300B`. It now uses configuration
`BamPythia2p8BHUnit300BFastKernel` at sealed runtime
`16ee2385da477c6967c09545dd8bbaa32e0eed66`, replacing `fb915b6` on the same
UE5a v5p-256 trainer. Parameters, optimizer, batch, dataset and the original
143000-step LR/checkpoint plan are unchanged. The new AOT manifest identifies
v5p-256/s143000, 206769653 bytes, CRC32C `FlSL4w==`.

An initial handoff restored the SIGTERM checkpoint 26140, which provided no
post-checkpoint overlap with the old implementation. That attempt was stopped
before acceptance. Following the earlier generalized-write kernel handoff,
all training workers were verified absent, the newer committed checkpoint was
retained, and the checkpoint-root cursor was explicitly rewound to the verified
26000 checkpoint's own cursor. The new first step is 26001. Original old-runtime
loss records 26001..26040 were frozen before the registry could replace replayed
steps; no interpolation or neighboring-batch loss comparison is used.

Across those 40 training losses on identical steps and batches, new-minus-old mean is
-0.00004415, RMS difference 0.00009971, maximum absolute difference 0.000218,
and correlation 0.99998850. Matched steps 26021..26040 give harmonic mean speed
0.559396 old versus 0.613275 new step/s, +9.63%, with generic health ON in both.
All 32 workers were independently verified at 16ee238 with `Loaded compiled function!`.
The external checkpoint-rescue trigger/budget remain 1300/600 seconds. Report
cursors remain 23k, next public report 28k; the four-figure plotting policy is unchanged.

Evidence is under `/data0/xd/bam_diagnostics/h-unit-300b-launch-20261101/`:
`fast-kernel-paired-audit.json`, `fast-kernel-overlap-source.json`,
`fast-kernel-all-workers.json`, `fast-kernel-controller-env.json`,
`fast-kernel-handoff-journal.json`, and `fast-kernel-aot-manifest.json`.
The historical method is
`/home/xd/projects/xd_tpu_scripts/general_write_kernel_handoff_20261026.py`.
The exact v5p-256/s143000 AOT and manifest verified, SHA256
`966c5ae9aded84306ee61209e26ffdd181e0de8095da7beceb9af26b77d2aa92`,
220926061 bytes. Full runtime config matches the sealed Git class. Prior CPU
validation remains valid; only the reporting plan changed after sealing.

Pre-run gain-multiplier bets (common reference: original MHA 2.8B):

| Horizon / comparator | Central prior | Expected range |
|---|---:|---:|
| 143k / historical parallel Mudd | 1.80 | 1.45-2.15 |
| 143k / Pythia 6.9B | 1.14 | .92-1.36 |
| Common 36.5k / old BAM long run | 1.40 | 1.20-1.60 |
| Matched-health v5p-256 throughput | .60 step/s | .50-.70 step/s |

The historical terminal Mudd/6.9B gain ratio is .63240268, so the first two
central bets are consistent (1.80 times .63240268 is about 1.14). Unit H's
short-schedule 21.5k ratio versus serial Mudd is 1.65151, with MHA gain .110862;
it supports a useful architectural improvement but is not the long-run parallel
Mudd comparator. The long schedule, comparator architecture, WD and LR differences
preclude direct transfer of that short-run ratio. Expect an initial contraction,
with possible late recovery, rather than a monotonic ratio increase. Old BAM
stopped near 37k; its observed trajectory cannot test a terminal comparison.

The canonical/deployed controller, rescue and AOT wrapper hashes match. Reporting
uses configurable calendar phase 3000 with cadence 5000; generic phase support
passed 17 group/step-offset tests and was deployed from the scripts repository.
No fabricated delivered milestone is used to shift the calendar. The normal 3k
report also checks the first permanent checkpoint; later reports verify retention
through rolling cleanup. The report-plan configuration test passes.

Launch artifacts: `/data0/xd/bam_diagnostics/h-unit-300b-launch-20261101/`.


## Startup verified

All 32 workers check out the sealed runtime and log `Loaded compiled function!`.
Actual step 0 and finite subsequent updates are verified; the registry first-step
waiter reached step 26. Checkpoint 0 has both outer and items commit markers.
Its cursor records seed 9876, global batch 1024, 32 infeed hosts, checkpoint step 0
and step-in-file 1 on the intended UE5a Pile replica. Runtime config confirms
8 examples/device, T2048, Q/K scale policy `qk_dim`, all 143k LR/retention settings,
async saving and Orbax timeout 1200. The controller confirms rescue trigger 1300
and budget 600; the new process has not needed a timeout rescue.

All 20 samples in steps 20-39 give harmonic-mean throughput .561686 step/s,
median .565. This is inside the .50-.70 pre-run range and approximately 4.0%
slower than the old long BAM's historical .585 step/s, with matched generic
health ON/internal NN and BAM health OFF but different architecture and leases.
A better comparison uses the same 20-49 window: old BAM .5938465, new H
.5623226 step/s, a 5.31% speed deficit. The initial comparison to the old late-run
.585 reference understated this matched-window deficit. Later new-run 100-199
steps average .55822 with median .562, so the deficit is not an isolated startup
outlier.

Existing v5p-128 evidence already shows the inherited H recipe cost: old BAM
.337699 versus unit H .316382, a 6.31% deficit with the same batch/remat/health.
Transferring that relative speed to old v5p-256 predicts .556360; observed
.562323 is 1.07% above that estimate. The original .60 central throughput bet
missed this available historical evidence; a wide range does not explain or
excuse the architecture's efficiency cost. These are separate historical leases,
not a paired causal benchmark, but do not indicate an extra long-plan regression.

Historical direct-parent experiments contradict attributing the deficit to MLP
fragmentation or full NoPE QK80 merely from logical operation counts. The 2:1
allocation arm started at .3294-.3295 versus parent about .326 (+1%); later
records show about .325 versus .329 (-1.2%). Its increased MLP/write counts did
not establish a large net cost. QK80 ran at .316706 versus SeparateVO about .317,
essentially unchanged. These are matched-setting historical leases, not paired
causal benchmarks; nevertheless they do not support either change explaining
a 5-6% deficit.

The clearest local slowdown is the SeparateVO transition: about .328 to .317
(-3.4%). That transition bundles independent V/O addresses, three-layer-block
to single-layer scan, retuned MLP widths and a VMEM-budget change. Another
unresolved stage is the old long recipe to the complete-QKNorm ShortTrain/kernel
lineage. These stages, rather than generic MLP-count or Q/K-width arguments,
are the appropriate next attribution targets. The logits denominator change
substitutes a compile-time constant in the existing operation and adds no tensor
pass. The generic Transformer TFLOP estimator omits BAM-specific/small-MLP work,
so its displayed TFLOP/s cannot establish the actual compute difference.
Historical counterexamples and source links are preserved in `speed-review.json`.


Group state confirms the next normal report is 3000, followed by 8000/13000/...,
with all three gain triples and both BAM +1 offsets persisted. No native/group
report cursor has been advanced. Fixed-checkpoint retention after rolling cleanup
remains a later runtime check. `startup-worker.log`, `all-worker-runtime.log`,
`runtime.config.txt`, `checkpoint-0-cursor.json`, `checkpoint-0-markers.txt` and
`startup-speed.json` retain the startup proof in the launch artifact directory.
