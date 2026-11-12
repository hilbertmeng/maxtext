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
Configuration implementation commit: `64d75057`; sealed effective attributes
match the main ledger for all three classes. This is not a trained runtime.
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
attribution. No intermediate metric reports are requested; retain the priors
for final scientific bookkeeping.

Preparation/resource artifacts:
`/data0/xd/bam_diagnostics/mediumprop-v5-situ-depth-20261112/`.
