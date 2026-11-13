# MediumProp V5 ordinary-MLP SiTU 4/7

RUN: `BamMediumPropV5SiTU47`; branch `codex/mediumprop-v5-situ47`;
worktree `/home/xd/.codex/worktrees/medium-v5-situ47/maxtext`.
Training ownership: `xd-v5p-16-311013-maxtext`, UE5a primary / EW4b and UC1a backups.
On 2026-11-12 the user authorized UC1a for Medium, then separately restored it for XL.
Borrow retained FLEX `llm-jax-v6e-1-0` only for AOT.

## Change and controls

Derive from current-main `BamMediumPropV5`, not XL and not Pythia.
Only ordinary MLPs, including the terminal ordinary MLP, replace SwiGLU by
`(4*tanh(u/4)*sigmoid(u)) * (7*tanh(v/7))`.
Private memory-writing MLPs retain SwiGLU. No parameters, matrix projections,
Pallas interfaces, initialization or optimizer rules change.
Ordinary/private/terminal widths remain 2510/1128/3904; verified total 432,112,228 parameters.
The existing Pallas read/write kernels are reused: activation is in XLA's MLP path.

The trained direct V5 control is the pre-rename RUN
`BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormQKScaleTruePile`
(runtime `f8d07cd`, completed 13,500, ~.677145 step/s).
Use `BamMHAMediumPropC256TruePile` only as the native gain-multiplier reference.
Do not plot a separate MHA loss-gap curve.

## Budget, runtime and workflow audit

From-scratch 13,500 steps and LR endpoint 13,500; T4096, per-device batch 16 on
v5p-16 (8 devices), global batch 128. Total 7,077,888,000 tokens, unchanged
from the same-size V5 control. Original Prop initialization/AdamPax recipe;
full remat and single-layer scan. Generic training health ON, internal
activation/BAM sow statistics OFF; TB rotation 500, checkpoint 200,
resume-only retention 2 and forced final checkpoint.

Forked latest main `b855d5f0`; this already includes the current logging,
checkpoint, optimizer/AOT, parameter-budget and common V5 infrastructure.
Copied only the validated SiTU activation from XL runtime `bd181bc`; XL
model/config and its historical infrastructure were not copied.
CPU gates cover analytic derivatives, fused/unfused MLP full gradients,
formal parameter counts/config, scanned ordinary/private isolation and
initial parameter identity, plus summary-writer rotation. Run the current
full BAM V5 regression suite as well. Seal exact effective configuration
before AOT; launch only after all preparation gates and require FIRST_STEP.

## Prior and monitoring

Prior: final MHA-referenced gain multiplier versus V5 .995, range .985--1.005;
speed within -2% of matched V5. Medium V5 has much less sustained loss-spike
damage to recover than XL, so a net quality gain is not the central prior.
Evaluate actual loss-spike amplitude/duration together with normal-phase CE.
No inference-cache change; parameter delta 0 W_Q.

Fixed public cadence 2,000; reviews 4,000/8,000 give recommendations only,
and require user confirmation before stopping. Otherwise finish 13,500.
Draw cumulative native loss-gap/gain and loss/raw/clipped health series.
XL SiTU was monitored independently and has completed its 50k closeout.

Sealed runtime: `ff2fe4b66be25389bd53a9218efa71f406a9b9e5`.
Main contains the configuration ledger only; executable SiTU code is in the runtime worktree.

## Verified launch

The 20 CPU gates passed (4 SiTU, 5 summary-writer, 11 current V5 regressions).
AOT/manifest: `gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/ff2fe4b/jax081-i0ae3f58-c17f538a/v5p-16/s13500/BamMediumPropV5SiTU47.pickle`.
The obsolete two-option fixture in the first attempt was rejected before training;
the corrected fixture leaves model code unchanged and is sealed in `ff2fe4b`.

UC1a acquired 2026-11-12T04:41:19.662838154Z; actual fresh steps 0--14 captured,
first recovery gate observed step 12 with resume_step=-1. Formal waiter then
verified steps 220/223, ~.685--.688 step/s (recorded parent ~.677145, +1--1.6%).
Worker log confirms `Loaded compiled function!`.
Checkpoint 200 is committed: seed 9876, global batch 128, 2 infeed hosts, UC1a
TruePile b0/b500 cursor at step_in_file=201. Generic health ON/internal OFF.
Existing UE/EW queues were retained during policy change and released by the
native recovery controller only after actual UC1a first-step confirmation.

Artifacts: `/data0/xd/bam_diagnostics/mediumprop-v5-situ47-20261112/`.
Group `mediumprop-v5-situ47` owns 2k reporting and 4k/8k confirmation-only reviews;
native gain triple is configured against the trained V5 control with MHA reference.

## Final result and closeout (2026-11-12 UTC)

Completed the full 13,500-step budget at unchanged runtime `ff2fe4b`.
All 67 native 200-step comparison windows through 13,400 are present.
Against the trained V5 SwiGLU control, the early roughly -.002 to -.003
loss lead disappeared around 3.2k. The middle run stayed near parity;
late windows showed a small positive gap, with no durable recovery.
Final 13.4k gap is +.0005196 and native MHA-referenced gain multiplier
.997080538; the final five windows average +.0004334 / .997589701.
The prior .995 [.985,1.005] was consistent with the result. Steady speed
was about .680 step/s versus the historical parent's .677145 (+.4%);
this is an unpaired training observation, not a matched speed benchmark.
Parameters, inference cache and nominal matrix-operation count are unchanged.

Both Medium trajectories have no XL-like sustained loss-spike episode;
their cumulative raw/clipped gradient and loss plots are close. Thus this
arm establishes a very small normal-phase quality cost, not a demonstrated
Medium stability gain. Together with XL's reduced sustained loss excursions
and slight final quality cost, SiTU 4/7 is supported as a stabilization option,
not a loss-SOTA improvement. Do not tighten caps merely to remove raw spikes.

Checkpoint 13,500 has both `commit_success.txt` and its actual data cursor:
seed 9876, global batch 128, two infeed hosts, UC1a TruePile b13000/b13500,
file index 13, step_in_file 500. TB synchronization completed successfully;
loss coverage ends at 13,499 and raw/clipped health at 13,490.
The registry is complete; independent checks found all six nodes/queues
absent across UC1a/UE5a/EW4b. Group `mediumprop-v5-situ47` is archived and
its owned watcher stopped after final delivery. The retained FLEX compiler
was not deleted. Main retains only the configuration ledger, not SiTU support.

Actual training used UC1a throughout. The initial UE5a assignment was queue
preparation with no verified FIRST_STEP; passive UE5a/EW4b candidates never
became training-zone switches. Two preemptions resumed committed steps 2957
and 3175, with subsequent periodic checkpoint/cursor verification.

| READY lease (UTC, 2026-11-12) | Duration | End |
|---|---:|---|
| 04:44:00--05:59:14 | 1h15m14s | preempted |
| 07:24:39--07:35:45 | 11m06s | preempted |
| 10:57:35--15:19:29 | 4h21m54s | scheduled completion |

Closeout evidence under the artifact directory above: `loss-report-final.json`
(native calculation authority), `registry-final.json`, `leases-final.json`,
`checkpoint13500-proof.txt`, `final-resource-checks.json`, `tb-sync-final.log`,
`plot-gap-final.json`, `plot-grad-final.json`, `plot-health-final.json`, and
`group-final-archived.txt`. The final group delivery is an extra completion
event at 13.4k; its fixed 2k reporting calendar was not shifted to 13.5k.

## Equal-budget depth configurations (not launched)

Keep D1200, 16x75 heads, T4096, the Prop initialization/optimizer and 13,500-step
schedule. Only depth and MLP widths change; BAM keeps M75x32, independent R256,
unit-initialized QK/V/O scaling, M pre-norm and ordinary-only SiTU 4/7.
Retain the same implementation worktree/branch as the completed 18-layer arm.
Configuration and training runtime commit: `64d75057`; sealed effective
attributes match the main ledger for all three classes.
The configurations were initially delivered without AOT or TPU allocation.

| Configuration | Layers | Ordinary MLP | Private MLP | Terminal ordinary MLP | Parameters |
|---|---:|---:|---:|---:|---:|
| `BamMHAMediumPropL12TruePile` | 12 | 5600 | none | 5600 | 432,106,800 |
| `BamMediumPropV5SiTU47L12` | 12 | 4131 | 1937 | 5796 | 432,106,552 |
| `BamMediumPropV5SiTU47L27` | 27 | 1464 | 604 | 1797 | 432,142,342 |

The 27-layer MHA control has 432,142,800 parameters. The 12-layer MHA is
14,400 below the 18-layer MHA (432,121,200), because six layers of norm gains
are removed. BAM matches each same-depth MHA to -248/-458 parameters.
Complete private-MLP cost includes its down/up address projections and write
gates (458,000 fixed parameters per writer layer). The ordinary/private branch
ratios are 2.001238:1 at 12 layers and 2.002127:1 at 27 layers. Thus neither
compression nor expansion refunds all parameters only to the ordinary branch.
All nonterminal BAM layers share one ordinary/private width pair and use
single-layer scan; both terminal M writers stay disabled.

The new MHA derives directly from `Llama2MediumPropL12` and the tuned Splash
mixin, avoiding retired `bam_mha_control`. Standard MHA honors the existing
partial-RoPE settings: first coordinate unrotated, remaining 74 rotated.
An independent analytic RoPE reference verifies the complete attention output
and all parameter/input gradients. No Pythia initialization, bias, LayerNorm
or QKNorm is introduced. Historical 18/27-layer MHA controls remain unchanged
at their recorded C256 runtimes; the new 12-layer MHA uses tuned Splash.

CPU proof: full-size abstract parameter trees for all three new classes,
writer/terminal/scan/kernel policy assertions, and the independent odd-head
MHA gradient check. Artifacts: `depth-config-verification.json` and
`l12-partial-rope-test.log` in the same local artifact directory. Main V5
regressions cover the small standard-MHA partial-RoPE support change:
all 11 passed (`depth-main-v5-regression.log`), alongside the independent
attention-gradient check. `depth-sealed-config.log` records SEALED_CONFIG_OK.

## Depth-family training authorization and resource policy

The user subsequently authorized all three new RUNs for full training, with
UE5a primary and UC1a/EW4b backups, resource-only monitoring and no loss plots
or intermediate stopping reviews. Topology v5p-16, per-device batch 16,
global batch 128, T4096 and 13,500 steps/LR endpoint remain fixed. Total
7,077,888,000 tokens matches the completed Medium reference's actual density;
non-token-embedding/unembedding counts are 311,070,000 (MHA12), 311,069,752
(BAM12), and 311,105,542 (BAM27). These are the same parameter-budget family,
so no new model-scale training-budget assumption is introduced.

Resource ownership: `BamMHAMediumPropL12TruePile` -> `xd-v5p-16-311014-maxtext`;
`BamMediumPropV5SiTU47L12` -> `xd-v5p-16-311015-maxtext`;
`BamMediumPropV5SiTU47L27` -> `xd-v5p-16-311016-maxtext`.
Runtime `64d75057` remains in the same worktree/branch above. Main `a0891fae`
was audited before sealing: training, optimizer, checkpoint, summary-writer
and AOT source files are identical to the runtime. Intentional SiTU support
and configuration differences do not omit operational fixes. TB rotation 500
and its five regression checks remain in the launch gate. Generic training
health stays ON, internal activation/BAM statistics OFF. Borrow verified
FLEX_START `llm-jax-v6e-1-0`/EW4a only for serialized AOT preparation;
training controllers own only the three new spot training identities.

Pre-run priors: MHA12 final gap versus MHA18 +.004 [-.002,+.012]; BAM12
MHA18-referenced gain versus SiTU18 .98 [.94,1.01]; BAM27 corresponding
gain 1.02 [1.00,1.05]. Speed priors on v5p-16: MHA12 about .90 step/s,
BAM12 .78 and BAM27 .55 (roughly +/- .05/.05/.04). The MHA timing baseline
is historical C256, so its kernel difference precludes a pure depth-speed
attribution. Monitoring was initially resource-only; retain these original
priors for final scientific bookkeeping after the later reporting-policy change.

Preparation/resource artifacts:
`/data0/xd/bam_diagnostics/mediumprop-v5-situ-depth-20261112/`.

All three target v5p-16/s13500 AOTs and manifests are ready at the same
`compiled_trainsteps/64d7505/jax081-i0ae3f58-c17f538a/v5p-16/s13500/` root
as the previous Medium artifact layout. The shared launch gate passed all
ten focused checks (four SiTU, one partial-RoPE gradient, five TB rotation).
All three RUNs have actual committed checkpoint/cursor launch evidence:
seed 9876, TruePile4096, global batch 128 and two infeed hosts. Exact launch
proofs are in `initial-committed-cursor-proofs.json`; ongoing lease/queue
observations stay in `resource-status-history.jsonl`, not this ledger.
Resume-bound startup gates (at least 14 updates past the verified source
checkpoint) measured MHA12 .975, BAM12 .844 and BAM27 .522 step/s. These
are unpaired startup observations, not long stable-lease medians. MHA12
and BAM12 have both-worker `Loaded compiled function!` and exact-runtime
hash proofs. Subsequent UC1a resumes verified the same two checks for all
three RUNs, including BAM27; receipts are `*-uc-loaded-aot-proof.txt`.
The next periodic checkpoints also verified UC1a data cursors.
Detailed artifacts stay in the
resource directory rather than adding running loss records to this ledger.
Resource monitoring refreshes committed GCS markers through native
`run_registry.py report-all` as needed, independently of live worker-step
availability. The initial resource-only policy was superseded on 2026-11-13
by grouped 2,000-step reports. Persistent group `mediumprop-v5-situ-depth`
uses BAM27 as its shared clock, reports all three members at their latest
available horizons, and retains no intermediate stopping reviews. Native
loss windows remain 200 steps. Running loss records remain in the registry,
not this ledger.

### MHA12 completed (2026-11-13)

`BamMHAMediumPropL12TruePile` completed 13,500 updates at runtime `64d75057`.
Against the matched-budget 18-layer MHA, its early advantage disappeared
around 4k and became a small, stable late deficit: final 13.4k window
gap +0.0059044, final five-window mean +0.00553692. The prior +0.004
terminal gap, range [-0.002, +0.012], covered this result. Widening the MLP
to 5600 did not fully replace six attention/residual stages.

Training used EW4b then UC1a, about .98 step/s on UC1a, with six registered
preemptions. This timing is not a controlled speed comparison with the
historical C256 MHA18. The full native lease history, one supplemental
pre-training READY interval, and preparation-only candidate observations
are distinguished in the regional history. Final checkpoint 13,500 and its
TruePile4096 cursor were verified (seed 9876, global batch 128, two infeed
hosts), TensorBoard returned SYNC_OK, and all six node/queue identities
across UE5a/UC1a/EW4b were independently verified absent. The retained FLEX
compiler remains untouched. Closeout receipts and native final report are
under the depth-family artifact directory above.

### BAM12 completed (2026-11-13)

`BamMediumPropV5SiTU47L12` completed 13,500 updates at runtime `64d75057`,
using EW4b then UC1a with four registered preemptions. Against V5 SiTU18,
the early deficit grew to about +0.02 by 3k and then stayed near +0.023;
there was no material late catch-up. The final 13.4k gap was +0.0224012;
the final five-window mean was +0.02293956. Against the matched-depth MHA12,
the same means were -0.1609616 and -0.16120424, so BAM retained a large
advantage over MHA despite losing to its 18-layer BAM parent.

The question of depth preference needs independently matched references:
`G_d = L(MHA_d) - L(BAM_d)`, `R_d = G_d/G_18`.
For BAM12, final R12 was 0.9070385 and the final-five mean was 0.9025709.
Its relative advantage therefore lost about 10% after reducing depth,
beyond the shape change's effect on MHA. Native reports now persist this
four-RUN definition through `--paired-gain-ratio`; `gain_difference` is
`G_18-G_d`. Existing three-RUN common-reference ratios remain supported.
The original MHA18-reference prior was 0.98 [0.94, 1.01]; its actual final
ratio was 0.8737665 (last five 0.8715680), decisively outside that range.
Do not silently substitute the later matched-depth definition when judging
that earlier prediction.

Final checkpoint 13,500 and its data cursor were committed and verified,
TensorBoard returned SYNC_OK, and both resource identities in all three
regions were independently absent. The retained FLEX compiler was not
modified. `bam12-final-proof.json`, `bam12-final-report.json` and
`bam12-original-prior-gain-final.json` retain the exact native results and
resource history in the depth-family artifact directory.

### BAM27 completed and depth-family conclusion (2026-11-13)

`BamMediumPropV5SiTU47L27` completed 13,500 updates at runtime `64d75057`.
Against V5 SiTU18, it crossed ahead near 2.2k, reached a roughly -0.006
to -0.007 gap by 5k and retained that advantage through the end. The final
13.4k window was -0.006810526; the final five-window mean was -0.0062574252.
The same final-five means were -0.1962335452 against MHA27 and -0.0197867452
against the historical generalized-read BAM27. Its gain over that old
same-depth BAM was 1.1121326, using MHA27 as reference.

The matched-depth ratio G27/G18 finished at 1.1014690 (final-five mean
1.0986928), compared with BAM12's final-five 0.9025709. Thus increasing
depth from 18 to 27 enhanced this V5 SiTU architecture's advantage over
matched MHA by about 10%; reducing depth to 12 lost about 10%. This was
already visible by 2k and did not reverse in late training. The original
common-MHA18-reference prior, 1.02 [1.00, 1.05], is a different metric:
its actual final-five result was 1.0350252, within the stated range.

The budget allocation helps interpret the shape effect. Ordinary SwiGLU
weights, including the wider terminal layer, are `3D*((L-1)*H+H_terminal)`;
same-depth MHA uses `3D*L*H_MHA`:

| Depth | BAM ordinary MLP parameters | MHA MLP parameters | BAM/MHA ordinary capacity |
|---|---:|---:|---:|
| 12 | 184,453,200 | 241,920,000 | 76.25% |
| 18 | 167,666,400 | 207,360,000 | 80.86% |
| 27 | 143,499,600 | 155,520,000 | 92.27% |

BAM's lower attention-parameter cost becomes more valuable as depth rises,
and its ordinary residual MLP retains a larger fraction of matched MHA's
capacity. More M read/write stages and this allocation effect are both
concrete candidate mechanisms; the experiment establishes the whole
architecture's depth preference, not an isolated causal effect of M
iteration alone. A larger-model depth comparison should retain matched
MHA references and audit this capacity allocation before transferring the
Medium multiplier. The more promising direction is greater depth at fixed
parameter/token budget; the ordinary/private ratio can then be optimized
rather than treating 2:1 as universally optimal.

The last service preemption left committed checkpoint 13,392; the native
controller restored it and completed the remaining 108 updates. After TB
SYNC_OK, 1,283 shared TB/console samples confirmed identical step indices
and a maximum loss difference of 5e-7, explained by console rounding.
Only absent console samples were imported: 66 sample-period-10 points,
including 13,380 and 13,390 needed for the final 13.4k window. All existing
console points were preserved. The native event was regenerated through
13.4k, without interpolation or loss recomputation. Receipts are
`bam27-tb-backfill-proof.json` and `bam27-final-report-before-tb.json`;
`bam27-final-report.json` is the corrected native final report.

Training used EW4b then UC1a, about .52 step/s on UC1a, with seven native
preemptions. This is an unpaired -23% timing difference from SiTU18, not a
controlled kernel comparison. Final checkpoint/cursor 13,500, TB SYNC_OK,
and all six regional node/queue identities were verified. The full lease
and assignment evidence is in `bam27-final-proof.json` and the regional
history. Both 12-layer RUNs were already fully closed; the retained FLEX
compiler was never enrolled in training-resource cleanup.
