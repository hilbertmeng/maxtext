# XLProp V5 A-type gradient amplification

Scope: isolate the diffuse amplification that remains after the FSS reverse
numerical failure was removed. Validated frozen points at10100 and at the onset,
peak and tail of the36k wave identify genuine ordinary-SwiGLU amplification;
layer and subpath vary with the operating point. The corrected formal RUN
completed50000; this diagnosis uses independent checkpoint copies/resources.

## Current conclusion: the35.6--36.9k wave

**The main reverse amplifier is the ordinary SwiGLU MLP's aligned, input-dependent
high-gain directions. Parameter updates, especially on the value/readout side,
move subsequent inputs into these directions. This is not the earlier FSS
saved-residual failure, and no single bad update or localO-only cause explains it.**

- Original-AOT replay35636--36950 covers1315 updates: all1315 console losses
  match within six-decimal rounding (maximum4.9913e-7), and all132 common TB raw
  norms match exactly. Within the replay,671 updates have raw>1; last36946.
  Per-update peak809414.5625 at35676 exceeds the10-step sampled345624.90625.
- With pre35636 parameters frozen, all11 selected future batches have raw
  .1969--.2160, including the809k peak batch. Repeated events are parameter-state
  conditioning, not unavoidable bad future batches. M is rebuilt each forward;
  cross-update persistence is carried by parameter/optimizer trajectories.
- At onset, peak and tail, original-mesh loss/gradient controls pass, frozen
  native/Pallas layers agree, and explicit FP32 SwiGLU chain rules reproduce the
  local derivatives. Small memory-MLP gains remain about.29--.43 at tested points.
- One-update rejection is not a cure. Discarding35636 reduces the next84-step
  maximum to200.93 but leaves30 clipped updates; discarding healthy35635 removes
  the35636 spike yet still yields2635.81 and32 clipped updates. Original has74
  clipped updates and maximum809414.56 over the same35637--35720 interval.
- On the frozen peak batch, replacing just localV with its40-update-newer values
  on the otherwise old model raises raw.215→22.10; localO alone gives.436.
  W_O alone gives18.65. Q/K, shared read down, M-shift and attention-write updates
  alone stay below1. These are local parameter-transplant effects, not evidence
  that freezing a family improves a full training run.

Selected local evidence (ordinary MLP / post-attention norm / memory MLP VJP gains):

| Frozen event |Validated layers and directional gains|
|---|---|
|35636 onset|L5:4.077/1.269/.304; L6:7.463/1.001/.292; L7:2.330/.786/.382|
|35676 peak|L10:17.746/.566/.368; L13:27.503/.362/.426; L14:11.840/.323/.327|
|36920 tail|L8:7.894/1.011/.361; L10:2.705/.803/.392; L11:9.905/.755/.382|

New-case artifacts:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-a35650-20261109/`;
local `/data0/xd/bam_diagnostics/xlprop-v5-config-20261108/a35650-replay/`.
The exact controls, interventions and reproducible runners are detailed below.

## Observed event

RUN `BamXLPropV5FastKernelFullRemat`, immutable runtime
`d54c4c5f1339b01e406932ec28be0e398722b918`, full remat/packed keys, original
Prop recipe without QKNorm. At10100: loss2.658856, raw13621.210. L1 maximum
Q/K/V-projection gradient405.212, maximum localO/W_O917.419, ratio.44169.
At10120 raw118.867; by10200 .857159; at10500 .190223. This differs from the
isolated L1 split of B-type spikes. Parameter norms alone do not identify the
exact tensor/operation where amplification begins.

Source measurements:
`/data0/xd/bam_diagnostics/xlprop-v5-config-20261108/claude-replay/full-10100-gradient-spike.json`.
Formal15000 report/artifact remains separate from diagnosis.

## Exact replay and loss-mask isolation

Own diagnostic TPU`xd-v5p-32-xlprop-a-spike-1109`, `us-central1-a`, four workers,
formal global128/per-device8. Source worktree `/home/xd/projects/maxtext`, branch
`refactor-bam`; diagnostic runner902a16f6, model runtime remainsd54c4c5.
The diagnostic v5p-32 node and queue have been released after artifact verification.
Subsequent single-device checks borrow the idle reserved FLEX
`llm-jax-v6e-1-0`, europe-west4-a, through the compiler lock; they own no lifecycle.

Recovered source10000 into an independent prefix, with30-object size/CRC check:
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-a-spike-20261109/ASpikeReplay10000/checkpoints/10000/`.
Original formal data cursor is not rewound. Recovery records/worker launcher:
tpu-ag`logs/xlprop-v5-a-spike-1109/`. The source10000 prefix was restored live as
an extra copy during recovery; keep its generations recorded for eventual cleanup.

Runner:[bounded_aot_replay.py](bounded_aot_replay.py). It executes original train
setup/AOT/RNG/data/schedule, bounding only the host loop10001--10110 and requesting
checkpoint10099. It captures a copy of pre-update state/batch10100 before donation.
The original training step is not recompiled or instrumented.

After replay, restore that frozen state for each probe and change only
`targets_segmentation`, the loss mask. Model inputs/attention masks/forward
computation stay unchanged. Compare the full-mask replay with the captured
unmasked step before trusting isolation. Binary splits first identify a strong
sequence contributor, then a target position, using at most39 AOT calls including
control. Selection is greedy, not a claim of the exhaustive largest token gradient.
Each probe uses mean CE over its selected valid targets; retain total weights,
so its raw norm must not be mistaken for the original global-batch gradient norm.
Gradient additivity and the triangle inequality justify searching child masks
without scanning every sequence/target combination.

Formal AOT:
`gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/d54c4c5/jax081-i0ae3f58-c17f538a/v5p-32/s50000/BamXLPropV5FastKernelFullRemat.pickle`.
Total schedule remains50000. Worker CLI:

```text
--start-step 10001 --end-step 10111 --isolate-step 10100
exp_class=BamXLPropV5FastKernelFullRemat run_name=ASpikeReplay10000
checkpoint_period=10099 upload_loss_tb_period=1
```

Exact launch is preserved in worker.sh and invocation.json. Artifacts upload directly
worker→GCS under the diagnostic root's`results/HOST/`; TB goes to`tb/`.
Only release the owned diagnostic TPU after GCS verification and local download.

## Results

Exact replay reproduces10100 raw13621.2099609375 and loss2.658856;
checkpoint10099 is committed. Original replay completed through10110. The
capture wrapper initially omitted the executable memory-analysis interface,
raising at host-loop closeout before isolation; a transparent attribute proxy
and focused wrapper-contract check fix this. Resume10099 for a one-update
control/capture rather than replaying the first99 updates again.

### Isolated loss contributor

The full-mask frozen-state control reproduces raw13621.2099609375 exactly.
The greedy39-call search selects global sequence98, target2725 (zero based).
Its CE is3.397380352, while its selected-token parameter-gradient norm is
7,997,614,080. The original global denominator is523,977 valid targets, so its
contribution to the global mean gradient has norm15,263.29224. Sequence98's
entire mean gradient, rescaled to the same denominator, has norm15,324.07861.
The remaining token intervals within that sequence have a triangle-inequality
upper bound about165.12 in global units: this token dominates **sequence98**.
Other sequence groups also have strong contributions; these measurements do
not establish that only this token dominates the entire batch. The numerical
additivity bounds are approximate under BF16 rounding.

Captured input SHA256:
`c8b2ddfa3ab67347da4e708e3d0a9b15db92bfdf7c37bd07b9126a7917f7d9de`.
The sequence and all control/split measurements are retained in
`results-isolate10099/t1v-n-e838f9eb-w-2/` under the diagnostic GCS root.
The source checkpoint10099 has a verified committed TrainState and cursor;
it was never written into the formal RUN prefix.

### Single-device fidelity check: not a reproduction

Runner:[xlprop_gradient_localization.py](xlprop_gradient_localization.py),
source commits8c055513/ff14da94, immutable model snapshotd54c4c5. It restores
1,432,396,996 parameter elements, uses the captured sequence and loss token,
and retains layer residual/M cotangents plus selected Q/K/V/context tensors.
The captured input hash matches. However, its selected CE is.148596, versus
3.397380 in the original full-topology executable. Its otherwise normal local
Splash/reference VJP agreement therefore cannot explain or exonerate the
original event.

An additional `--baseline-forward-check` runs **without any probe/interceptor**:

| Execution | Selected-token CE |
|---|---:|
| Original v5p-32 train-step AOT |3.397380352|
| v6e B1 forward only |.148595810|
| v6e B1 selected-bias backward, returned forward |.174560919|
| v6e B8, other rows repeat the target sequence |.138705254|
| Same B8 executable, other rows' input tokens zeroed |.138705254|

Changing unrelated batch rows does not change the target row in this check.
The discrepancy is not introduced solely by the additive probes: even plain
forward and autodiff lowering differ at fixed weights/input. This establishes
numerical sensitivity to execution graph/topology, but does not by itself
identify its first layer or distinguish forward ill-conditioning from another
reverse implementation error. The tiny error of the local VJP is conditional
on the different single-device operating point. Do not present it as a test of
the original amplified point.

Artifacts: `tensor-probe-l15/` and `baseline-forward/` under
`gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-a-spike-20261109/`.
Local diagnosis root:
`/data0/xd/bam_diagnostics/xlprop-v5-config-20261108/a-spike-replay/`.

## Formal-mesh tensor localization and local VJP validation

Runner commitsc96fb7d3/6dffd0a2 add a formal-mesh extension to the bounded
replay and save boundary states/cotangents. Diagnostic node
`xd-v5p-32-xlprop-a-tensors-1109`, UC1a, original global128/per-device8,
16 devices/4 processes. It is now released (node and queue verified absent).
The original parameter sharding, full parameter differentiation and loss mask
are retained. Probe inputs are zero-valued additions for sequence98 only;
recompilation can change BF16 lowering, so loss and parameter-gradient controls
must agree before interpreting the tensors.

| Frozen target2725 | Original AOT | Formal-mesh probe |
|---|---:|---:|
| Selected CE |3.397380352|3.397630930|
| Parameter-gradient norm |7,997,614,080|8,015,838,720|

Forward error.00025058, gradient-norm error.2279%. Saving the boundary tensors
without adding more differentiation paths preserves these exact probe controls.
Adding six whole-network MLP/norm additive probes instead fails the controls
(CE.12919, norm1187.85); those tensors are explicitly rejected for attribution.

### Backward amplification is cumulative

Selected-token residual-state cotangent RMS at layer entry (no batch-mean scaling):

| Entry |L27|L24|L20|L17|L16|L15|L14|L12|L10|L5|L1|L0|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RMS |.00019027|.00246148|.0188863|.125474|.214111|2.18644|9.84360|90.0344|385.370|29,039.2|68,879.5|123,698|

L27→L0 grows about6.50e8-fold; L15 alone has an entry-to-next-entry ratio
10.21. This is not one isolated L1 parameter-gradient explosion. Because the
state contains both residual and M, ratios between state-component norms are
not standalone module operator norms.

### L15: ordinary SwiGLU, not a numerical read/write-kernel explosion

Saved L15 input and L16 input supply the exact layer boundary values; saved L16
cotangents are held fixed as upstream gradients. The standalone layer runner
[xlprop_single_layer_vjp.py](xlprop_single_layer_vjp.py) restores the scanned
weights from10099 and slices layer15. Only this one layer is reevaluated, avoiding
whole-network forward divergence. It compares Pallas/no probes, Pallas/local
MLP probes and native-JAX/local probes. In all three arms, forward relative
errors are below.42% and input-cotangent errors below.56% against the captured
formal-mesh point. Native and Pallas input cotangents differ by.285% (residual)
and.577% (M).

For the Pallas/local-probe arm, directional VJP norm gains are:

| Path |Input/output cotangent RMS ratio|
|---|---:|
| Ordinary SwiGLU MLP |21.782|
| Post-attention input RMSNorm |.51562|
| Dedicated memory MLP |.46905|

The normalized-h gradient RMS is4.663777: ordinary-MLP contribution4.663771,
small-MLP contribution.007408, small-write controls.003192. These are component
norms, not additive scalar energies; the total vector is their sum. Native JAX
repeats the result (ordinary21.801, norm.51562, memory.46910). The product of the
ordinary-MLP and norm gains is about11.23, consistent with the layer's observed
10.21-fold residual-state amplification after the other paths are included.
The ordinary MLP has a large directional Jacobian at this operating point.
This does not claim that every amplifying layer is dominated by the same path.

For completeness, [check_captured_splash.py](check_captured_splash.py) checks the
actual saved L15 Q/K/V and upstream context cotangent against standalone Splash
and an FP32 strict-causal reference. Original-model dQ/dK/dV relative errors are
.297%/.643%/.427%; actual dV is only.000353 of its valid64||G|| Frobenius bound.
Fused and unfused standalone backward both agree with the reference. The final
comparison makes the fused flag a static JIT argument; a mutable config closure
alone does not create a distinct compilation. Use `captured-splash-check-v2/`;
the earlier artifact is superseded.

### Interpretation and next decision

B-type is a confirmed FSS numerical reverse error. At the measured L15 point,
A-type has genuine strong backward amplification through the ordinary MLP,
reproduced by native and Pallas; RMSNorm damps it and the small MLP is not the
main source. This narrows the explanation beyond a generic M feedback story.
It does not establish the cause of the large ordinary-MLP Jacobian, or prove
all other layers numerically correct.

Do not attribute the lack of 2.8B spikes solely to QKNorm: Prop uses SwiGLU and
a different initialization/normalization recipe from Pythia GELU. QKNorm could
still change the learned weights or upstream directions. Nor does the current
large Jacobian prove it arose independently of earlier FSS training: the8750
resume inherits that weight history. No new formal experiment or architecture
change is implied by this read-only diagnosis.

Artifacts under the same diagnostic root: `formal-tensor-probe/`,
`formal-tensor-states/`, `formal-tensor-paths/` (failed fidelity gate),
`captured-splash-check-v2/`, `single-layer-vjp/`. Boundary-state upload and local
copy:27 objects,660,866,086 bytes, all sizes match, four worker exits0. Source
workers→GCS→local; tpu-ag only orchestrated. Borrowed FLEX was retained and its
lock released after each probe. Formal training was never paused or rewound.

Figure and source are archived under the diagnostic GCS root's `figures/` and
`runner/plot_localization.py`. The figure plots complete L0--L27 component
cotangents and entry/next-entry ratios for the selected loss, not the formal
training's batch-averaged raw/clipped history.


## Comparison with the15460 recurrence

The same formal full-remat RUN has another diffuse spike at15460:
raw131.4176, clipped1.00000036, loss2.089097, versus raw.260724 at15440.
L1 Q/K/V-projection maximum3.44177, localO/W_O maximum10.31813, ratio.33357;
there is no isolated L1 Q/K/V split characteristic of B-type.

We compared all recorded per-layer parameter-gradient norms with each event's
nearest pre-spike control (10100/10080 and15460/15440). The following are
**parameter-gradient amplification ratios**, not local tensor VJP gains:

| Layer |10100 attention W_O spike/control|15460 attention W_O spike/control|
|---|---:|---:|
|L26|2.97|1.09|
|L15|139.00|1.27|
|L12|1645.37|9.48|
|L10|3760.77|38.49|
|L5|171490.48|1705.78|
|L1|112963.04|1446.30|

The log layer-profile correlations are.951 for attention W_O and.950 for
ordinary-MLP W2. Both have normal/modestly changed late-layer parameter
gradients and progressive amplification toward shallow layers. Thus15460 fits
the same **diffuse A-type phenotype**, with a weaker and more shallowly
visible amplified region. A monotone profile correlation is not a proof of the
same local cause. In particular, L15's ordinary W2/Wg/W1 gradients at15460 are
only1.11/1.12/1.15 times control; the21.8-fold L15 directional VJP established
at10100 cannot be assigned to15460 from these statistics. A small aggregate
parameter gradient also does not rule out a strongly amplified isolated token
hidden under the healthy batch contribution.

No15460 frozen-state/token isolation or local VJP has been performed, so the
specific layer/branch source remains unconfirmed. Full raw selected-step
parameter statistics and the extraction script are archived as
`a-spike-comparison-10100-15460.json` and `runner/compare_spikes.py` under the
same GCS diagnostic root; local copies are in `a-spike-replay/`.


## Severe recurrence around35680

At the40k reporting milestone, full raw/clipped history identifies a stronger
A-like event in the same immutable full-remat RUN. Recorded raw norm first
exceeds1 at35650 (18.85), peaks35680 at345,624.90625, and has intermittent
above-threshold samples through36930. There are63 recorded points above1 in
35--40k; these are10-step samples, not a count of affected training updates.
The current40000 raw/clipped norms are about.202896.

| Step |Raw norm|Loss|L1 max Q/K/V projection|L1 max localO/W_O|
|---|---:|---:|---:|---:|
|35600|.19006|1.95895|.00356|.00702|
|35650|18.8538|2.05746|1.08903|1.66585|
|35670|84,167.66|2.23076|2555.863|4144.099|
|35680|345,624.91|3.27189|8613.380|6322.173|
|35750|1533.875|2.06499|64.581|105.092|
|36000|15.1950|2.08763|.70910|1.18119|
|36200|.61917|1.87814|.04408|.04852|
|36500|.20231|2.07135|.00447|.00810|

The35680 L1 Q/K/V-to-O parameter-gradient norm ratio is1.362, and both paths
are large, unlike B-type's isolated Q/K/V error. Some of the largest parameter
gradients are the embedding address bias and shallow BAM read gates/keys.
This is an A-type **pattern**; no frozen-point local VJP has established its
specific source. Do not transfer the10100 L15 ordinary-MLP attribution to it.
The loss/gain trajectory has a large temporary deterioration near36k and is
still somewhat below its pre-event gain ratio near40k. The difference from
Mudd's normal late-training catch-up cannot be causally separated without a
counterfactual.

All selected-step raw parameter norms are in `full-35680-gradient-health.json`
under the same diagnostic GCS root. `checkpoint35500-object-metadata.json`
records the source35500 checkpoint's soft-deleted generations, CRC/size metadata
and commit/cursor marker objects. At the40k report the listing was read-only; the subsequent investigation
restores a verified independent copy as described below. Its
listed hard-delete time is2026-11-16 (UTC); verify live/soft-deleted generations
again before restoring. Source:
`gs://newproject-1-llm_base_models_us-central1/log/BamXLPropV5FastKernelFullRemat/checkpoints/35500/`.
The known bounded original-AOT runner can replay from35500 to the onset/peak
within its200-update bound, with the50000-step schedule and full topology;
choose a pre-update capture before inferring a local cause.


## Exact onset replay of the35.65--36.93k wave

Source35500 is recovered into a separate diagnostic prefix, with30 objects,
15,651,280,232 bytes and matching CRC/size. Formal checkpoint/data state is not
rewound. Owned diagnostic TPU `xd-v5p-32-xlprop-a35650-1109`, UC1a, four workers,
formal global128/per-device8, immutable model/AOTd54c4c5. Host helper commits
`a52f8741`, `a8e758e8`, `d7b11bb9` add bounded first-threshold capture, per-update
metrics, sparse host checkpoints and frozen-parameter future-batch controls;
they do not modify the model/optimizer executable.

Replay35501--35700 has20 common10-step records with formal TB: loss **and**
raw gradient norm match exactly at all20 (maximum absolute error0). The first
update with raw norm>1 is **35636**, raw2.197865486, loss1.924464226. Per-update
peak within this first200-update replay is **35676:809414.5625**, larger than
the10-step sampled35680 peak345624.90625. It is not yet a verified maximum
of the complete recurrence through36930.

Original-AOT loss-mask control at35636 matches exactly. Greedy binary mask
isolation selects global sequence9,target1211 (zero based), selected CE
3.699650288 and selected-token parameter-gradient norm1035932.625. The global
mean denominator523956 gives contribution norm1.97714, already close to the
full-batch2.19787; this alone does not establish exclusivity or its local source.
Sequence input SHA256:
`016b7b77bd3077fdb262121600f62db91ff6c239592e78eabc20e307f5c73b04`.

New artifacts (separate from the10100 investigation):

- GCS: `gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xlprop-v5-a35650-20261109/`.
- Local: `/data0/xd/bam_diagnostics/xlprop-v5-config-20261108/a35650-replay/`.
- Original-AOT replay/isolation: `onset/t1v-n-efa3d921-w-1/` (`replay-metrics.jsonl`,
  `captured-step.json`, `isolation.json`, `isolated-sequence.npz`).
- Source control points: `official35500-35700.json`.
- Orchestration/ownership: tpu-ag `logs/xlprop-v5-a35650-1109/`.

The subsequent sections report completed formal-mesh tensor localization,
local VJP/FP32 checks and future-batch controls. The10100 L15 attribution remains
specific to10100; the new points have independent controls and layer selections.


###35636: validated local source and FP32 chain rule

The formal-mesh instrumented point has selected CE3.699650288 (exactly the
original-AOT value) and parameter-gradient norm1045872.375 versus1035932.625
(error.9595%). The saved residual cotangent grows from L27 RMS.00011632 to
L0 RMS23.849, about205022-fold. Strong boundary gains are in L5--7, unlike
the10100 L15 example. At the selected token, M input RMS at L5/L6/L7 is
1.263/1.327/1.460. These saved layer-entry values are not epsilon-scale;
they alone do not measure the later shifted-M/address normalization denominators.

Fixed-boundary local checks on reserved FLEX, diagnostic runnersd5a2ef7a /
b58244aa (model stilld54c4c5):

| Layer |Ordinary MLP directional gain|Post-attention norm gain|Memory MLP gain|Native/Pallas agreement|
|---|---:|---:|---:|---|
|L5|4.07692|1.26864|.30391|Ordinary MLP4.07497 in native|
|L6|7.46277|1.00136|.29180|Ordinary MLP7.46116 in native|
|L7|2.33040|.78558|.38226|Ordinary MLP2.33006 in native|

Local input-cotangent relative errors against the captured formal point are
below1.2%; normalized-h gradient norms are dominated by the ordinary MLP.
These ratios describe the actual upstream direction, not worst-case spectral
norms, and do not by themselves explain every layer of the full network.

Runner[swiglu_local_jacobian.py](swiglu_local_jacobian.py) further evaluates the
explicit SwiGLU chain rule in FP32 at the captured MLP inputs and output
cotangents. It uses both effective BF16-rounded weights and full FP32 weights;
this is a **local** check, not a whole-network FP32 reproduction. The chain
rule and cap/detach counterfactual derivatives are checked against CPU autodiff,
and the power-iteration estimate against an exact small-matrix SVD.

For L6 with effective BF16 weights, FP32 output error is.271%, input-VJP error
.262%, and global directional gain7.47186. At target1211, the gain is7.48515;
its Jacobian spectral lower estimate is11.6896, versus random-direction RMS
gain.57929. Holding the same target upstream gradient fixed but using adjacent
tokens1210/1212 gives gains.76646/.82845. Thus a steep, input-dependent local
direction exists even when typical directions contract.

Four hidden units1291/1751/301/51 contribute respectively.26668/.26459/
.25662/.24041 of the input-gradient projection on its own amplified direction
(sum1.02831; remaining units net-cancel.02831). Their gate preactivations are
7.4436/7.2412/7.0713/7.0748; linear-content preactivations only.2259/.2702/
.2330/.2670. At this point the linear-content derivative branch has gradient
norm1382.23, versus154.19 for differentiation through the gate. A bounded
SiLU derivative does not bound the other branch, which is multiplied by the
unbounded SiLU **value**. Forward output can remain moderate while this
Jacobian is large. No training fix is claimed from this local calculation.

Artifacts: `single-layer-onset/layer-{05,06,07}/` and
`mlp-detail-onset/layer-{05,06}/` under the new35650 diagnostic root.

###35676 peak: a separate fidelity-checked point

A second owned full-topology diagnostic TPU
`xd-v5p-32-xlprop-a35676-1109`, UC1a, reads a CRC-verified independent copy of
checkpoint35675 produced by the exact replay. Original AOT reproduces full
raw809414.5625 and loss2.595792532. Greedy loss-mask isolation selects sequence104,
target1220, CE4.354687691, single-token gradient norm398608236544. Its contribution
to the global mean is760730.325, with denominator523981. Input SHA256:
`492eddc2d58fba0bd32faf217ae270a5f3c6168970cd97f927314167774efd7d`.
The onset sequence is a project XML file; the peak sequence is legal prose.
They are different inputs, not a repeated copy of the same isolated sequence.

Formal-mesh peak tensors pass the original controls: selected CE4.354121208
(error.00056648), norm400494592000 (error.4732%). Residual cotangent RMS grows
from L27.0000789071 to L0 5194782.5, about6.5834e10-fold. Largest boundary gains
include L10 residual11.02, L13 residual9.09 and L14 M9.19; L6 is also retained
for comparison. The saved peak point is therefore not assumed to share the
onset's exact layer attribution. The completed local checks are below.

Artifacts: `peak-isolation/t1v-n-664a732b-w-0/`,
`tensor-peak/t1v-n-664a732b-w-0/` and subsequent `mlp-detail-peak/` under the
same new GCS root. Host ownership/launch files remain in
`tpu-ag:logs/xlprop-v5-a35650-1109/`.


### Peak local validation: ordinary SwiGLU again, different subpaths

| Peak35676 layer |Ordinary MLP VJP gain|Input norm gain|Memory MLP VJP gain|FP32 input-VJP error|
|---|---:|---:|---:|---:|
|L6|2.26168|1.21687|.32471|.242%|
|L10|17.74639|.56591|.36837|.248%|
|L13|27.50281|.36183|.42553|.215%|
|L14|11.83983|.32334|.32739|.172%|

Native/Pallas local checks agree, and the captured full-mesh boundaries are
reproduced within about1.2% in input cotangents. This repeats the ordinary-MLP
source across onset and peak, while excluding a fixed single-layer explanation.
FP32 local Jacobian spectral estimates at L10/L13/L14 are28.146/32.303/13.990.
For L13, the estimated stable rank is only2.264, and the actual upstream
direction receives about85.2% of the maximum estimated gain. These local
Jacobians have a few strong directions; an average/forward RMS misses them.

The microscopic subpath differs. Onset L6 is dominated by the derivative
through the linear-content input multiplied by the SiLU gate **value**. At
peak L14, differentiation through the gate dominates instead: its gradient
norm14375 versus1520 for the linear-content branch. Stopping the gate derivative
reduces the L14 fixed-input gain11.848→1.146 without changing its forward
output; capping SiLU at1 leaves gain11.898. Thus a universal explanation or fix
based only on large SiLU output values is falsified by this second point.
At peak L13 both branches contribute (59883 and124769); capping SiLU at1 lowers
gain27.515→8.498, but detaching the gate still leaves18.950.

At onset L6, removing the identified four units reduces gain7.485→.864;
removing four seeded-random units leaves7.485. The identified gate columns
have pairwise cosine.976--.980, content columns.759--.795, output rows
.926--.938. They are the four largest gate-column norms and four smallest
content/output norms: this is an aligned channel group, not simply unusually
large norms on every weight. Removing units changes the local forward too
(relative change34.5%); this is a mechanistic intervention, not a validated
training recipe. Their actual fixed-input FP32 projection fractions establish
their gradient contribution independently of that forward change.

At peak L14 the top16 units account for73.0% of the gradient projection; the
four leading gate columns have cosine.872--.897 and output rows.698--.744.
Removing those four lowers gain11.848→8.693, while random-four removal leaves
11.848. The peak involves a broader coherent group than onset L6's four units.

Actual captured L6 Splash Q/K/V reverse is also checked against FP32 at the
peak, using identical saved tensors and upstream cotangent. Observed dQ/dK/dV
relative errors are1.195%/1.316%/.883%; actual dV is.0012254 of its64||G||
Frobenius bound. Fused and unfused backward give the same conclusion. This
contrasts with the previous FSS B-type violation; it does not establish that
every attention operation in the full network has been individually tested.

Additional artifacts: `mlp-detail-onset-geometry/layer-06/`, `mlp-detail-peak/`
and`splash-peak/`. The peak diagnostic TPU/node and queue have now both been
verified absent; its44 tensor/metadata objects (1793722630 bytes) were verified
locally first. Reserved FLEX was borrowed through its lock and retained.


## Recurrence controls and the causal limits of a single trigger

Original-AOT long replay35636--36950 has1315 loss matches against registry
console data (absolute error<=4.9913e-7, six-decimal console rounding) and132
exact raw-norm matches against formal TensorBoard. It reproduces the full
wave, including671 raw>1 updates and a final within-replay threshold crossing
at36946. The maximum remains809414.5625 at35676. The initial135 updates
35501--35635 are uniformly quiet: their maximum raw is.238209.

Future-batch test: save each actual future TrainState/batch/RNG during the
unchanged replay, then rerun it with **only parameters** replaced by the frozen
pre35636 values. Future optimizer state/counters/data/RNG remain unchanged;
returned updates are discarded. Actual-state repeat controls match first.

| Step |Actual raw|Frozen-pre35636 raw|
|---|---:|---:|
|35650|18.8538|.21234|
|35676|809414.5625|.21527|
|35680|345624.9063|.21407|
|35750|1533.8752|.20212|
|35800|716.4374|.20232|
|35900|24.9549|.20431|
|36000|15.1950|.21602|
|36200|.61917|.19693|
|36580|8.25201|.21298|
|36920|4.36222|.21143|
|36930|1.11555|.20649|

Two bounded diagnostic branches replay the exact35599 checkpoint/data/AOT:
discard one step's parameter and floating optimizer-moment updates, while
preserving the original advanced Adam/schedule integer counters and state.step.
Counter preservation is tested with actual Optax Adam/schedule states. These
are independent diagnostic copies, not changes to the formal RUN. Prefix loss
and raw norms match exactly through the intervened update (37/36 points).

| Branch |Raw at35636|Max raw35637--35720|Clipped updates in those84 steps|Mean loss in those84 steps|
|---|---:|---:|---:|---:|
|Original|2.19787|809414.5625|74|2.473013|
|Discard35636|2.19787 (computed before rejection)|200.9313|30|1.987817|
|Discard healthy35635|.21080|2635.8091|32|1.992270|

Discarding the first clipped update greatly mitigates this realization but
does not prevent recurrence. A one-step healthy perturbation removes the first
spike yet permits later large spikes. Thus the first selected token/update is
not a unique necessary cause; the trained function is already sensitive to
parameter perturbations and contains latent high-gain directions. Short branch
loss improvements do not establish a final-training-loss benefit.

Artifacts: `long-replay/`, `discard-35636/`, `discard-35635/` and the local
`long-replay-summary.json` / `paired-discard-summary.json`. Diagnostic helper
a0f5237c implements the explicit counter-preserving intervention.

## Parameter families: amplifier location versus trajectory driver

Runner[xlprop_parameter_family_probe.py](xlprop_parameter_family_probe.py),
commits6af6f32a /7f6dc79e /ba811660 /c9ea5bd6. Restore only parameter values
from35635 into the frozen35676 TrainState; the original AOT, optimizer/RNG,
full batch and isolated sequence104/target1220 masks remain fixed. Every test
discards the returned update. The original peak full/isolated controls match
exactly before interpreting transplants. This changes a network's operating
point deliberately; it is not a fidelity-preserving reimplementation test.

Over40 updates, ordinary-MLP parameters change by L2 6.9410 relative to542.509
(1.2794%); the rest changes7.1364 relative to618.156 (1.1545%).

| Ordinary MLP source |Rest source|Full-batch raw|Original target's raw|
|---|---|---:|---:|
|Old35635|Old35635|.21527|9.9217|
|New35675|Old35635|5.08449|9.8432|
|Old35635|New35675|10882.6826|4745.4224|
|New35675|New35675|809414.5625|398608236544|

The old MLP already contains sharp directions. Updating the rest can activate
large gradients without changing those MLP weights. The original selected
peak token becomes healthy in both mixed cases while other tokens dominate
the full-batch gradient. Therefore ordinary-MLP attribution does not imply
that its own recent weight update is the sole cause of the episode.

Only one family moved to its newer values, all other parameters old:

| Family |Parameter elements|Full-batch raw|
|---|---:|---:|
|Attention except common read scales|253544540|4337.1064|
|Ordinary MLP|665700480|5.08449|
|Memory MLP plus address/gate|314569980|.50841|
|Embedding M write|4813620|.37496|
|Common QK/V-O read scales|56|.32878|
|Unembedding|96829440|.22560|
|Layer/final norms|109440|.22136|
|Token embedding|96829440|.20538|

Finer attention transplants (again newer family only, others old):

| Attention subset |Full-batch raw|
|---|---:|
|localV+localO read parameters|37.77321|
|W_O|18.64953|
|localQ/K read parameters|.34132|
|Attention M-write addresses/gates|.29237|
|Standard/RoPE W_Q,K|.22481|
|Shared read down|.22387|
|M-shift|.20534|

Splitting V/O identifies **localV** as the stronger isolated driver:
localV newer alone22.10190, localO alone.43606. This agrees with the emphasis
on the context/value path rather than an automatic localO-only explanation.
It does not make localO irrelevant: reverting only localO on the complete
peak model reduces raw to22474.55; reverting only localV leaves97316.62.
Ranking causes simply by which reversion reduces the peak most would give a
different and misleading answer. Both sufficiency and the changed token
identity must be considered.

The strongest supported interpretation is **latent ordinary-MLP amplification
plus value/readout-driven changes of its input and downstream direction**.
These counterfactuals do not establish a universal single-parameter trigger,
nor that QKNorm would or would not fix retraining. In particular, QK updates
alone do not reproduce this episode's instability in the tested old network.
Unembedding changes alone stay healthy and reverting unembedding leaves
raw807770, so the classifier head is not the main cause here.

At L6, onset and peak have the same four leading units (1291,1751,301,51);
their gate-column norms barely change across the40 updates. Large gradient
changes therefore cannot be inferred from weight-norm growth alone. The
embedding write-address bias receives a large amplified gradient but is not
identified as the originating local amplification operation.

Artifacts: `parameter-families-peak/`, `parameter-families-expanded-peak/`,
`parameter-families-fine-peak/`, `parameter-families-vo-peak/`. Full losses,
isolated-target controls, parameter counts and both transplant directions are
in each master worker's `parameter-family-comparison.json`.

## Tail validation and next discriminating experiment

At36920 original raw4.362222 and isolated sequence88,target1186 CE6.391760349,
selected gradient1562735.375. The full-mesh probe CE differs by1.43e-6 and
parameter-gradient norm by.2638%. Saved residual cotangent grows1.3311e6-fold
from L27 toL0. Selected input SHA256:
`2302092765cf48ecf535a1b1c3ca1096b31efb82821c159b72c2d57a81ec0b4c`.

Frozen local L8/L10/L11 ordinary-MLP gains are7.89358/2.70458/9.90496;
post-attention norm1.01059/.80281/.75513; memory MLP.36086/.39198/.38213.
Native/Pallas agree; explicit FP32 input-VJP errors are.334%/.211%/.232%.
Thus the same ordinary-MLP amplification pattern occurs in onset, peak and
tail, but its strongest layer changes. These sampled local checks do not
claim an exhaustive individual attribution for all671 clipped updates.

The next intervention should preserve the ordinary MLP's gated structure and
smoothly bound both factors (SiTU-GLU), rather than first replacing the entire
activation with GELU. The earlier GELU-first recommendation was premature:
bounded SwiGLU is a smaller change that directly addresses both identified
derivative multipliers. QKNorm remains a separate factor. No new formal
training experiment has been launched; the following local checks inform its
design, not its eventual training result.

Tail artifacts: `tail-isolation/`, `tensor-tail/`, `mlp-detail-tail/`.
Plot runner:[plot_xlprop_spike_replay.py](plot_xlprop_spike_replay.py).
Figure/provenance are backed up under the diagnostic GCS root's `figures/`.
Both owned diagnostic TPU nodes and queued resources are verified absent.
Before releasing the primary node, all44 tail tensor/metadata objects
(1793733409 bytes) and all11 primary-node jobs' four-worker exit codes were
verified locally. Reserved FLEX is retained; its diagnostic jobs have finished
and the borrowed lock is returned. Source checkpoints and evidence remain in GCS.

## Bounded activation research and fixed-boundary comparison

Priority is **SiTU-GLU**. Hard clipping is a published magnitude/reference
comparison, not a change of the research objective. With gate preactivation
`u=W_g x` and up preactivation `v=W_u x`, use

```
h = [beta_g * tanh(u / beta_g) * sigmoid(u)]
    * [beta_u * tanh(v / beta_u)]
```

[Kimi K3, Appendix B](https://arxiv.org/html/2607.24653v1) uses
`beta_g=4, beta_u=25`; both factors are smoothly bounded, with the original
sigmoid and first-order behavior near zero retained. Its claim of better
training behavior than hard clamping is qualitative, not an isolated
quantitative ablation of our architecture. Candidate `2/2` is a stronger
diagnostic contrast selected for the measured operating amplitudes, not a
known optimal training setting.

Published hard bounds differ in location and coverage:

| Released model | Gate factor | Up factor | Coverage/detail |
|---|---|---|---|
| [GPT-OSS](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py) | pre-SiLU upper7 | pre-product +/-7 | sigmoid slope1.702 and up+1 are additional differences; do not silently import these into vanilla SwiGLU |
| [Step-3.5-Flash code](https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/main/modeling_step3p5.py) / [config](https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/main/config.json) | post-SiLU upper7 | +/-7 | routed experts L43/L44 only; L44 shared expert uses16; other published layers disable clipping |
| [DeepSeek V4 code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py) / [paper4.2.3](https://arxiv.org/html/2606.19348v1#S4.SS2.SSS3) | pre-SiLU upper10 | +/-10 | Flash/Pro published configs use10; paper reports improved training stability without performance degradation |

On user-authorized reserved FLEX `llm-jax-v6e-1-0`, worker0, under its shared
worker lock, the unchanged d54c4c5 model restores and boundary fidelity gates
were reused for onset L6 and peak L13/L14. Diagnostic helper commit
`ccf3f013`, source SHA256
`1c6a41b8120c4d5f5bbf67c03d38da175fdf21bd57294ff87bc1a8b7b350b941`.
Only the activation's local FP32 formula is replaced, with **captured inputs
and upstream cotangents fixed**. Effective BF16-rounded weights and original
FP32 weights agree closely; original local VJP errors against captured BF16
are .2622%/.2152%/.1721%. This is not a modified full forward or a loss replay.

Directional input-VJP gain at the selected token:

| Activation | onset35636 L6 | peak35676 L13 | peak35676 L14 |
|---|---:|---:|---:|
| original SwiGLU |7.48515|27.51450|11.84788|
| Step-style post-SiLU/up hard7 |6.76528|27.51450|11.84788|
| DeepSeek-style pre-SiLU/up hard10 |7.48515|27.51450|11.84788|
| vanilla SwiGLU, both preactivation factors hard2 |1.68710|18.35451|11.82301|
| **SiTU4/25** |**3.64992**|**24.27499**|**11.69174**|
| **SiTU2/2** |**1.87312**|**15.49364**|**9.78018**|
| post-product hard2 |7.48515|23.25406|11.89708|

Captured full-sequence layer-output relative changes (L2 delta / original L2)
for SiTU4/25 are2.76%/23.85%/10.87%; SiTU2/2 gives13.53%/48.50%/33.72%.
These sequences come from the spike operating points, **not a healthy
holdout calibration dataset**. Larger changes cannot be interpreted directly
as a training-loss penalty. GPT-OSS's exact activation changes these outputs
by166%/92%/106%, largely a different-function comparison due to its up+1 and
sigmoid slope; its result should not select our threshold.

The smooth cap suppresses all three selected gains; hard7/10 does not affect
the peak's selected points, and hard2 scarcely affects L14. This supports
SiTU as the next minimal architecture trial. Neither tested SiTU setting
removes every local expansive direction. Before selecting a training setting,
inspect the [original u/v distributions across stages and layers](xlprop_v5_swiglu_distributions.md).
Current user direction is distribution first, with4/7/10 as amplitude references
and independent gate/up bounds;2/2 remains a strong diagnostic contrast, not a
default recommendation. The objective is to shorten prolonged spike episodes
while preserving training quality, not eliminate every spike. A future minimal
activation change would affect **only the ordinary MLP**, keeping parameters,
widths and BAM kernels unchanged. A
checkpoint replacement can test immediate suppression; from-scratch or
pre-trigger continuation tests prevention, which a failed immediate hard10
replacement cannot disprove. Do not multiply these isolated layer-gain
ratios to predict a new whole-network raw norm.

For standard SwiGLU, `dh/du=SiLU'(u)*v`, `dh/dv=SiLU(u)`. Bounding both
factors directly controls both multipliers. Bounding only the product does
not: at `u=0, v->infinity`, the product remains0 while `dh/du=v/2` diverges.
Output caps alone therefore do not address the identified mechanism.

[Smooth-SwiGLU](https://arxiv.org/html/2409.12517v1) is related outlier work,
but its function-preserving rescaling targets FP8 quantization and does not
change the exact FP32 Jacobian. Its same-neuron gate/up alignment also must
not be confused with our earlier **across-neuron** column alignment:
top4 same-unit gate/up cosines at onset are .5898--.6191; peak L14 are
-.5078-- -.4289. It is not a matching explanation for every observed point.

All9 activation references and both partial derivatives were checked against
independent CPU JAX autodiff, including hard-clip boundary ties. Each of the
three FLEX jobs exited0. Artifacts (5 files per case) and metadata are under
`bounded-activation-probes/{onset-06,peak-13,peak-14}/`; reproducible worker
protocol is `protocols/flex-bounded-worker.sh`. Source remains
[swiglu_local_jacobian.py](swiglu_local_jacobian.py). FLEX is retained and its
borrowed worker lock returned; no new formal RUN or diagnostic TPU was created.

## 2026-11-18: fixed-boundary L5/L6 attention-path VJPs

Owner requested an actual test of whether L6's SwiGLU amplification is followed
by further multiplicative amplification inside L5 attention. Reused immutable
d54 parameters at CP35635, the original seq9/target1211 loss's saved native
L5/L6 states and cotangents, and its exact4096-token input. Four independent
local-device workers on retained UC1a produced identical reports. This is a
conditional VJP at one known important loss, not a full-batch causal percentage
or a worst-case Jacobian norm.

Artifacts: `training_reports/xl_sink_origin_20261118/attention-vjp-r24/`
(`verification-and-summary.json`, `review_results.py`, all worker reports).
Helper57ad85f8; runtime d54 unchanged. All four workers exit0/complete, helper
and cohort SHAs agree. Maximum original-boundary relative error1.0047%, probe
versus local baseline .4447%, output/read/factor vector closure .3781%; all
preset gates pass. The initial r23 failed during local configuration directory
creation on three workers; another aborted GetSliceInfo during startup. No r23
model measurement exists. The directory repair completes all4 workers without
that backend error. No precision gate was relaxed.

| Actual-direction VJP gain | L5 | L6 |
|---|---:|---:|
| Ordinary SwiGLU |4.07682|7.46300|
| Post-attention norm |1.26864|1.00140|
| Private memory MLP |.30394|.29181|
| W_O: output cotangent to head cotangent |1.66651|1.23673|
| attn_context to normalized ordinary input (M fixed) |.07373|.04487|
| localO to normalized ordinary input (M fixed) |.10599|.08938|
| summed head to normalized ordinary input (M fixed) |.13659|.10133|

Thus L5's ordinary SwiGLU and W_O amplify these actual incoming directions,
while its read/attention mapping back to ordinary input contracts strongly.
This does not establish joint-state contractivity in (residual,M) or at other
inputs. Never call M_input/M_output cotangent ratio a local M gain: residual
output gradients also flow into M. Residual identity bypasses carry the already
amplified post-MLP gradient even when attention's own branch contracts.

L5 localO's input-gradient components, each normalized by the SAME incoming
localO cotangent norm: dynamic address/keynorm/sharedGELU .07042; dynamic gate
.04694; static gate .02286; Mshift/Mnorm .02424. Static address parameter has
no direct ordinary-input derivative. These are vector components, not additive
gain percentages; their vector sum reproduces localO's .10599 gain.

L5 direct attention-M-write head cotangent norm is only.01698 times the total
head cotangent norm; the route via W_O dominates this case. Separating layer
residual-output and M-output cotangents closes within1%. M-output-only gradient
norms divided by full parameter-gradient norms: W_O .01267, static_o_key .02946,
full_m_read_down .02191. Residual-output-only ratios .99893/.99332/.99920.
These are norm ratios, not additive percentages.

Shared down receives QK/V/localO gradient vectors with norms10913/24619/39079
respectively (full51307, singleton loss normalization). Their vector sum closes;
these norms must not be summed as causal shares. Static O parameter gradient
comes through the localO static-address branch. Large parameter gradients are
compatible with contracting ordinary-input VJPs: dB is a sum of g_hidden*x^T,
whereas input propagation multiplies by B^T; d(static_key) sums Mhat*g_head,
whereas the static key itself does not depend on ordinary input.

Separately verified r22 full-batch L6wi1 column energies: four units
1291/1751/301/51 account for98.03427% of squared parameter-gradient norm.
Original1512 AOT controls exact; instrumented full raw error .1641%, selected
wi1 norm error .0791%;16 indexed inputs and seq9 six fields match. This is
parameter-column energy, distinct from the previously measured local input-VJP
projections summing102.8%.

## 2026-11-24: backward-only localO / V cuts (verified)

Owner room2414/2422 requested fixed-forward gradient cuts, local M adjoint
components and read-direction alignment. Runtime stays
`d54c4c5f1339b01e406932ec28be0e398722b918`. The complete attempt is
`training_reports/xl_localov_stopgrad_20261124/attempt-b/`; source helpers are
`xlprop_localov_stopgrad_probe.py` and `xlprop_localov_M_vjp.py` in this branch.
Repair1 helper commit `efd05f2d` is the executed version. Original attempt-a
failed before interventions due donating AOT state reuse; attempt-b repair1
fixed a guard that also matched the explicit terminal's same-named parameter.
Both fixes preserve the numerical gates; the latter had not changed measurements.

Whole-batch experiment: exact original16 input-index/hash pairs for batch35676,
B128/T4096, frozen healthyCP35635 and peakCP35675. Both original serialized AOT
controls reproduced exactly (healthy loss1.9770115613937378/raw.21527354419231415;
peak loss2.595792531967163/raw809414.5625). Autograd baselines are
.2152755212442369 and809418.5271199519. All17 arms per checkpoint preserve the
baseline forward loss exactly; identity gradient-vector fidelity and selected
staticO zero-gradient coordinate checks pass. All4 worker full results and local
VJP reports are exactly equal. `independent-verification.json`, `reviewed-summary.json`
and `verify_completed.py` retain evidence. No retained optimizer updates or
checkpoint saves. Layer indices are zero-based: first6=0..5, L5=sixth layer.

Cuts act on the fused read output before BOTH residual W_O and memory-write
consumers, leaving forward values and read's pass-through M unchanged. V cuts
are at the V-read output, before attention; this is not clipping a standard W_V.

| Backward cut | Healthy norm reduction | Peak norm reduction | Peak remaining norm |
|---|---:|---:|---:|
| localO L5 |1.7245%|−4.9494% (increases)|849480.259|
| localO first6 |5.0970%|2.1839%|791741.978|
| V L5 |2.4977%|17.7723%|665566.437|
| V first6 |7.9804%|48.2382%|418969.742|
| localO+V first6 |13.5446%|63.9368%|291901.931|
| QK first6 |4.1735%|16.4635%|676159.511|

Single V cuts at L3 and L4 reduce peak norm22.5153% and22.4620%, versus
2.2704% and2.0104% healthy. These are separate interventions, not additive shares.
This elevates the V path, especially L3/L4, over localO alone as a spike-gradient
propagation target at this fixed state. It does NOT identify where amplification
was originally created. Even joint first6 OV cuts leave ~292k, far above healthy.

Small net norm change does not mean localO is unimportant. For first6 O cuts,
peak ||g−g_cut||/||g||=.696879 while ||g_cut||/||g||=.978161 and cosine=.752002.
The intervention substantially changes direction/cancellation even though its
net norm changes little. A single L5 O cut increases norm4.95%. Therefore neither
norm-reduction percentages nor component norm² ratios are causal/additive shares.

Separate local probe: native ONSET35636, seq9/target1211, fixed L5/L6 states and
cotangents from the earlier native capture (NOT the peak full-batch state).
QK/V/O M-input adjoints close to the full adjoint with worst relative error.003019,
within the predeclared1% bound. L5 localO vs V M-gradient cosine.0040123;
L6 cosine−.0132749: nearly orthogonal in this conditional probe, not a large
same-direction sum. LocalO's M-adjoint norm exceeds V's here; this is compatible
with a weak effect on whole-batch parameter-gradient norm at a different state.

For actual shifted/whole-RMS-normalized M at positions0/512/1211, O-cotangent
||M g_H||/||g_H|| is1.14–1.64 times the isotropic RMS-direction reference across
L5/L6. The largest singular values are33.65–44.66 vs RMS-direction gain~6.325,
but O gradients do not strongly select the largest singular vector (top1 energy
~.86–3.05%). These sampled measurements show directional preference, not an
orders-of-magnitude alignment amplifier. The geometry calculation isolates the
M contraction, not gates/scales or the full read-parameter Jacobian. V at target
position1211 has zero recorded projected response; do not treat that zero as
an angle/alignment estimate without retaining the corresponding cotangent norm.
This probe lacks a healthy/peak spectrum comparison and cannot rule out strong
alignment at other tokens or the peak.

Current judgment: downstream ordinary MLP remains the demonstrated local
amplifier; V is a stronger measured propagation route for the peak than localO
alone. The proposed localO+V dominant-direction overlap is not established by
these results. Highest-value follow-up is fixed-input V L3/L4 -> downstream MLP
boundary gains, keeping incoming-gradient magnitude separate from unit-cotangent
Jacobian gain. No retraining benefit or stable stop-gradient recipe is claimed.


### 2026-11-24: first-spike full-batch cuts, correcting the measurement point

Owner2431 explicitly requires the FIRST gradient spike, not the later maximum.
The preceding full-batch cut table used step35676; its local M probe alone used
ONSET35636. Those scopes must not be conflated. This follow-up repeats all17
cuts on step35636 with the actual CP35635 state and original B128/T4096 batch.
Runtime/AOT remains d54c4c5; helper83707cb9 adds an explicit onset mode, with no
model changes. Artifacts: `training_reports/xl_localov_stopgrad_20261124/onset/`;
GCS `diagnostics/xl-localov-stopgrad-20261124-onset`. Branch/worktree as above.

All4 workers exit0 and agree exactly. Native original AOT reproduces
loss1.924464225769043/raw2.1978654861450195 exactly; plain autodiff raw2.197876565
is within.001% of native. All16 indexed input hashes match the archived
onset-columns-r22 cohort, all17 cut forward losses equal baseline exactly,
identity gradient fidelity and selected staticO coordinate-zero gates pass.
`review.py` and `verify_completed.py` independently pass. The prior separately
verified ONSET singleton M geometry is reused, not remeasured or relabelled as
full-batch evidence. No optimizer updates are retained and no training resumes.

| Cut (zero-based layers) | First-spike raw | First-spike norm reduction | Late-peak norm reduction |
|---|---:|---:|---:|
| L5 localO |2.508830447| -14.15% | -4.95% |
| First6 localO |2.054205278|6.54%|2.18%|
| L5 V |2.194800033|.14%|17.77%|
| First6 V |1.797203386|18.23%|48.24%|
| First6 localO+V |1.350978237|38.53%|63.94%|
| First6 QK |1.996571469|9.16%|16.46%|
| L3 localO |1.981552000|9.84%|1.12%|
| L3 V |1.977398981|10.03%|22.52%|
| L4 V |2.088074473|5.00%|22.46%|

The late-peak L5 V effect is almost absent at onset. This weakens any claim that
L5 V is the main trigger of the first spike: its stronger peak effect is
consistent with later propagation, not evidence of initiation. L3 O and V have
similar ~10% onset norm effects. First6 V still reduces the norm more than O,
but the relative importance inferred from late-peak norms cannot simply be
transferred to onset.

First6 localO removal changes the gradient by .81427 times baseline norm,
versus .60849 for V, despite its smaller NET norm drop. The remaining-gradient
cosines are .64758 (O) and .79392 (V). L5 O removal increases the norm14.15%:
localO participates in substantial cancellation, not a purely positive amplifier.
Removing first6 O zeros the selected O-address gradients but raises ordinary-MLP
squared gradient norm from .0709 to .1086 times the original GLOBAL squared
norm (~53% higher). Joint O+V cuts similarly leave this MLP energy above its
baseline. These are parameter-family aggregates, not a local Jacobian-gain
measurement, and do not show that the MLP mechanism is cured by cutting reads.

Joint cuts still leave raw1.351>1; they reduce but do not eliminate the first
spike. Direction changes and cancellation prevent additive causal-share claims.
The first-spike data prioritize upstream L3/read-path interactions and the
already demonstrated downstream MLP amplification over an isolated L5-V trigger.
A next localization would hold boundary tensors/cotangents fixed and distinguish
incoming-gradient size from local Jacobian gain; no such new resource is launched
as part of this completed bounded follow-up. Both exact new UC/EW VM/QR resources
were independently fresh NOT_FOUND after controller cleanup.


### 2026-11-24: layer-wise multiples over the normal pre-spike gradients

Owner2442 clarifies that the primary endpoint is the per-layer/module gradient
multiple over normal, not global norm reduction. Reaggregate the SAME completed
first-spike cuts; no new TPU measurement. Normal denominator is the original
native step35635 gradient for each identical parameter group; numerator is
step35636 with each intervention. This is the historical adjacent-batch reference,
not a same-batch healthy-parameter control. Layer indices are zero based.

Use all54 scanned-block parameter leaves per layer; group L2 norms are square
roots of summed squared leaf norms. Every identity group matches the archived
native onset group within1%. `onset/layer-normal-multiples.{json,csv}` retains all17
arms x6layers, whole-block/attention/ordinaryMLP/staticO/staticV/W_O groups.
Attention below includes all self_attention parameters, including write-related
parameters. StaticO/staticV zeroing by their own cuts is mechanical, not evidence
that upstream amplification disappeared.

All block parameters; each entry is cut-gradient norm / normal35635 norm.

|Layer|No cut|First6 O cut|First6 V cut|First6 O+V cut|
|---|---:|---:|---:|---:|
|L0|15.98x|13.68x|13.83x|10.66x|
|L1|17.90x|11.23x|16.43x|8.27x|
|L2|16.66x|17.72x|13.64x|14.03x|
|L3|29.51x|30.69x|20.21x|15.64x|
|L4|18.07x|15.64x|16.17x|11.95x|
|L5|8.75x|6.55x|7.56x|5.00x|

All self_attention parameters; each entry is cut-gradient norm / normal35635 norm.

|Layer|No cut|First6 O cut|First6 V cut|First6 O+V cut|
|---|---:|---:|---:|---:|
|L0|17.38x|8.62x|14.76x|5.69x|
|L1|17.93x|10.27x|16.46x|6.92x|
|L2|16.59x|15.25x|13.10x|9.93x|
|L3|31.25x|32.03x|20.46x|13.52x|
|L4|18.27x|14.32x|15.91x|8.88x|
|L5|9.59x|6.87x|8.13x|4.82x|

Ordinary MLP; each entry is cut-gradient norm / normal35635 norm.

|Layer|No cut|First6 O cut|First6 V cut|First6 O+V cut|
|---|---:|---:|---:|---:|
|L0|14.64x|15.42x|14.62x|15.06x|
|L1|18.38x|23.52x|17.45x|22.80x|
|L2|19.67x|31.22x|18.85x|30.61x|
|L3|22.66x|27.42x|22.51x|27.20x|
|L4|22.47x|26.23x|22.50x|26.27x|
|L5|7.13x|7.13x|7.13x|7.13x|

Attention output W_O; each entry is cut-gradient norm / normal35635 norm.

|Layer|No cut|First6 O cut|First6 V cut|First6 O+V cut|
|---|---:|---:|---:|---:|
|L0|9.26x|10.41x|8.34x|9.03x|
|L1|19.71x|22.17x|17.84x|20.68x|
|L2|17.99x|24.66x|16.89x|23.85x|
|L3|34.94x|44.50x|34.42x|43.91x|
|L4|29.94x|35.32x|30.17x|35.57x|
|L5|16.90x|16.90x|16.90x|16.90x|

Interpretation changes when evaluated at the requested endpoint. Joint cuts
reduce the all-parameter layer multiples substantially but still leave every
first-six block at5.00--15.64x normal. First6 O alone INCREASES whole-block L2/L3
multiples, while directly deleting O-address gradients. Ordinary-MLP L2 grows
19.67x ->31.22x (O cut), or30.61x (joint cut); L3 grows22.66x ->27.42x/27.20x.
W_O L3 grows34.94x ->44.50x/43.91x; L4 grows29.94x ->35.32x/35.57x.
First6 V reduces whole-block and attention-group norms more consistently, but
uncut ordinary-MLP/W_O multiples remain close to their abnormal baseline.

For ANY first6 arm, L5 ordinary MLP stays7.13x and W_O stays16.90x: cuts to
L5 attention read outputs cannot alter gradients of downstream parameterized
operations on that same forward pass. Those unchanged gradients are consistent
with the intervention placement, not a failed cut. Earlier-layer gradients can
change because their backpropagated cotangents traverse the intervened reads.

Thus the completed cuts have not normalized the abnormal uncut early-layer
parameter gradients. Global norm reduction partly reflects deletion of the cut
branches' own parameter gradients and changes in cancellation. It does not show
that localO/V are the principal amplification source, nor that MLP/W_O are
exonerated. The next useful localization is the upstream cotangent arriving at
these anomalous modules versus their local VJP gain, with fixed forward inputs;
no new resource is allocated for this report.


### 2026-11-25: component-level first-six-layer multiples

Reaggregate existing measurements; no new TPU job. Normal denominator remains
the exact parameter norm at native step35635. Numerator uses first-spike35636
with each cut; zero-based layers, original fullbatch loss normalization.
All324 identity leaf norms match native onset within1% (absolute floor1e-7).
Complete324components x17arms are in onset/component-normal-multiples.csv/json;
component_multiples.py reproduces them. Selected below by appreciable absolute
spike gradient or a substantial intervention effect, rather than large ratios
with tiny denominators. O/V cuts mean all first6 layers, not only listed layer.

|Layer|Parameter|No cut|O cut|V cut|O+V cut|
|---|---|---:|---:|---:|---:|
|L0|self_attention/static_o_key|21.9x|0.0x|19.1x|0.0x|
|L0|self_attention/static_v_key|29.9x|33.3x|0.0x|0.0x|
|L0|self_attention/out/kernel|9.3x|10.4x|8.3x|9.0x|
|L0|mlp/wi_0/kernel|16.8x|18.3x|16.3x|17.8x|
|L0|memory_mlp/wi_1/kernel|17.1x|24.4x|4.2x|2.3x|
|L0|post_self_attention_layer_norm/scale|23.7x|25.9x|4.6x|6.8x|
|L1|self_attention/static_o_key|19.2x|0.0x|19.2x|0.0x|
|L1|self_attention/general_v_static_gate/kernel|54.3x|39.8x|0.0x|0.0x|
|L1|self_attention/out/kernel|19.7x|22.2x|17.8x|20.7x|
|L1|mlp/wi_1/kernel|18.0x|22.6x|17.1x|21.9x|
|L1|self_attention/full_m_read_down/kernel|28.4x|23.9x|19.7x|9.9x|
|L1|self_attention/query/kernel|34.9x|26.2x|29.4x|20.3x|
|L2|self_attention/W_lv_gate/kernel|20.2x|26.1x|0.0x|0.0x|
|L2|self_attention/out/kernel|18.0x|24.7x|16.9x|23.9x|
|L2|mlp/wi_0/kernel|20.2x|31.9x|19.3x|31.2x|
|L2|mlp/wi_1/kernel|21.3x|34.9x|20.4x|34.2x|
|L2|mlp/wo/kernel|17.8x|27.5x|17.2x|27.0x|
|L2|self_attention/full_m_read_down/kernel|20.2x|16.0x|15.9x|10.8x|
|L3|self_attention/W_lv_gate/kernel|67.2x|83.6x|0.0x|0.0x|
|L3|self_attention/general_o_static_gate/kernel|74.9x|0.0x|73.2x|0.0x|
|L3|self_attention/out/kernel|34.9x|44.5x|34.4x|43.9x|
|L3|mlp/wi_1/kernel|25.0x|30.2x|24.7x|29.8x|
|L3|self_attention/full_m_read_down/kernel|36.4x|26.3x|31.7x|19.1x|
|L3|self_attention/static_o_key|18.8x|0.0x|18.5x|0.0x|
|L4|self_attention/W_lv_gate/kernel|24.6x|29.6x|0.0x|0.0x|
|L4|self_attention/out/kernel|29.9x|35.3x|30.2x|35.6x|
|L4|mlp/wi_0/kernel|33.3x|39.2x|33.3x|39.3x|
|L4|mlp/wi_1/kernel|18.3x|21.1x|18.1x|21.0x|
|L4|self_attention/full_m_read_down/kernel|17.9x|12.6x|16.7x|6.1x|
|L4|self_attention/static_o_key|19.0x|0.0x|19.1x|0.0x|
|L5|self_attention/out/kernel|16.9x|16.9x|16.9x|16.9x|
|L5|mlp/wi_1/kernel|8.7x|8.7x|8.7x|8.7x|
|L5|self_attention/static_o_key|10.3x|0.0x|10.3x|0.0x|
|L5|self_attention/W_lv_gate/kernel|12.9x|12.9x|0.0x|0.0x|
|L5|self_attention/full_m_read_down/kernel|10.2x|5.5x|8.0x|2.2x|

The most pronounced read-gate multiples occur at L3: W_lv_gate67.2x and
O static gate74.9x. These are not tiny-denominator-only effects: native onset
norms.7014/.3334 versus normal.01043/.004453, respectively. O cut increases
the V gate to83.6x; V cut leaves the O gate73.2x. Their own cuts mechanically
zero them and therefore cannot identify the original amplification source.
L2 ordinary-MLP wi_1 grows21.3x ->34.9x after O cut, with joint cut34.2x;
L3 W_O grows34.9x ->44.5x/43.9x. This is inconsistent with treating read
cutting as broad normalization of early-layer gradients.

There are specific genuine reductions on uncut components: L0 memory-MLP
wi_1 falls17.1x ->4.2x (V)/2.3x (joint), while its post-attention RMSNorm
scale falls23.7x ->4.6x/6.8x. Shared full_m_read_down also falls, especially
L0 15.7x ->1.1x and L5 10.2x ->2.2x under joint cut. These changes show
selective routing/propagation effects; they do not remove the remaining large
ordinary-MLP/W_O gradients. Near-normal values here describe this one batch,
not stable training or a general mechanism proof.



### 2026-11-25: paired sampled pre-W_O localO / AV amplitudes

Owner2448 asks whether localO raises H=AV+O and thereby explains W_O gradients.
Reanalyze EXISTING r17 sink-origin captures, not a new TPU measurement. Exact
source hashes match sink-origin.json. Four sequences9/40/73/104 x11positions
0/1/2/8/16/128/512/1211/2048/3072/4095 on batch35676, valid segment positions
only; all20heads x96channels. Compare CP35635 old parameters with CP35675 peak
parameters on this SAME batch. These are NOT first-spike35636 activations or
fullbatch statistics. The instrumented fullbatch losses differ from native AOT
by4.29e-6 old/.00040269 peak; this forward capture has the recorded1e-3 fidelity
gate, not the exact-forward guarantees of the newer backward-cut experiment.

Read_O is captured after read gates/scales; actual splash V exactly equals
read_V, consistent with folded common VO scales. AV is native Splash output,
not the FP32 attention reconstruction. Reconstruct H with actual BF16 addition;
its energy agrees with the FP64 sum/cross identity within.5%. Source and CPU
analysis: training_reports/xl_localov_stopgrad_20261124/amplitude-reanalysis/
{analyze.py,paired-amplitudes.json}.

|Layer|Normal O/AV|Peak O/AV|Normal O energy fraction|Peak O energy fraction|Peak H / normal H|
|---|---:|---:|---:|---:|---:|
|L0|1.84|1.62|77.3%|72.5%|0.984|
|L1|2.08|2.33|81.2%|84.4%|0.890|
|L2|1.47|1.60|68.4%|71.8%|0.907|
|L3|1.05|0.91|52.3%|45.6%|1.040|
|L4|1.18|1.16|58.1%|57.4%|1.067|
|L5|0.90|0.81|44.7%|39.4%|1.098|

O energy fraction means ||O||^2/(||O||^2+||AV||^2), EXCLUDING the cross term.
It is not a fraction of H energy or W_O gradient attribution. O/AV is an L2
norm ratio over the sampled valid positions/head/channels. Sampled O/AV cosine
is .056,-.058,-.052,-.219,-.243,-.229 under old parameters; at peak it is
.061,-.061,-.089,-.226,-.251,-.221. L3--L5 show partial cancellation.

LocalO is a substantial forward component already in the normal parameter
state: it exceeds AV in L0--L4 and accounts for68--81% of separate O+AV energy
in L0--L2. This confirms a meaningful difference from standard AV-only output.
However, the sampled summed H changes only -.110 to+.098 between old/peak
parameters, while the global gradient changes by orders of magnitude. The
available sampled forward captures do not support a uniform large-H peak
explosion. They do not rule out rare large-H positions, first-spike behavior,
or correlation with the incoming gradient G. A W_O gradient decomposition
requires AV^T G and O^T G and their cross alignment, plus paired normal/spike
H and G at the actual anomalous positions; raw forward energy fractions alone
cannot choose the H-versus-G explanation. Earlier preference for G based on
local MLP gains is evidence of a possible route, not a completed factor
attribution. First-spike saved tensor-onset captures retain AV only at the
target layer and do not contain first6 localO, so they cannot fill that gap.


### Capture-path ablation at normal35635 (2026-11-27)

Attempt-e repair2 completed a diagnostic of the instrumentation itself, not the
requested scientific AV/O decomposition. The historical d54 AOT trajectory,
B128/T4096, four workers and all sixteen input shards were checked. Native
unclipped W_O gradients were independently recovered from the actual Adam-Pax
before/after moments (second-moment recurrence relative error 2.36e-8).

| Observation in the gradient graph | Full gradient difference vs plain | First6 W_O difference vs native |
| --- | ---: | ---: |
| Identity custom VJP, no callback | 0 exactly | about 8.6e-8 |
| Retained AV/O/H, observed during backward; G unobserved | 0.69788% | 1.12904%–1.20661% |
| G only, no retained AV/O/H | 0 exactly | about 8.6e-8 |

All four workers agreed, their replicated gradient arrays were byte-identical,
and loss remained 2.0112438201904297 in every arm. Only the primal-observation
arm failed the unchanged 1% per-leaf gate. A forward-callback arm failed TPU
host-callback channel registration under both ordinary and explicit compiled
execution; the measured primal arm instead observes retained operands during
backward. Optimized HLO was saved, but its byte identity with the executed JIT
was not independently established.

This localizes the observed instrumentation mismatch to retaining/observing
primal boundaries in this configuration. It excludes bare custom VJP or G-only
observation as sufficient explanations here; retention, callback effects and
compiler fusion/precision remain competing mechanisms. It says nothing yet
about whether AV, localO, H magnitude or G causes the first spike.

Next: retain G alone in the validated gradient graph and capture primal
boundaries in a separate forward execution using the exact same state, batch
and RNG. That method must independently pass forward identity, row/input
coverage and global native W_O/branch reconstruction gates before interpretation.
Neither this normal-only diagnostic nor a successful run closes the unfinished
first-spike AV/O analysis. Evidence:
`training_reports/xl_wo_split_20261125/attempt-e/capture-ablation-review.json`.
