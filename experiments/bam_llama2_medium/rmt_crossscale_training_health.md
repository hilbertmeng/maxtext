# MediumProp versus XLProp dynamic RMT B training health

Current conclusion (2026-10-01): no causal root has been established. The strongest
localized forward difference is the terminal MLP write/feature concentration.
But its large raw static/dynamic W2-gradient contrast largely disappears after
checkpoint Adam-denominator scaling (median ratio .0513 Medium/.0622 XL).
Do not attribute the loss-gain decay to static-gradient domination, late global
clipping, dead dynamic routes, or output calibration: these explanations are not
supported by the paired probes. Sparse actual optimizer/update evidence is essential.

Question: why does the XLProp all-M-read-pre-norm dynamic RMT lose its advantage
while MediumProp retains it? Diagnose training health and distinguish an optimizer
problem, saturated/collapsed representations, and output calibration. Do not equate
lower loss than a failed parent with a healthy trajectory.

Training runtime shared562bfb9397c5e5f5314c19b3f109b2feba1a0596:
- Medium RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm,
  checkpoint5400 (40% of13500), D1200/head16x75/M48x75,C8,R256.
- XL RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm,
  checkpoint17500 (35% of50000), D1920/head20x96/M60x96,C10,R384.
These are near-progress, not exactly matched progress; the early checkpoints have
already been pruned. Historical TB supplies the temporal evidence.

Diagnostic worktree /data0/xd/rmt-crossscale-health, codex/rmt-crossscale-health,
created from the exact training runtime. Main keeps this report only.
Artifacts /data0/xd/bam_diagnostics/rmt-crossscale-health-20261001/.
TPUs owned by this diagnosis: EW4a v6e-1
xd-v6e-1-rmt-health-medium-261001 and xd-v6e-1-rmt-health-xl-261001.
Retained STANDARD/FLEX compilers and ongoing training TPUs are untouched.

Initial evidence: RMT-B / Mudd MHA-gain ratio stays near1.55 on Medium but
falls1.38->.98 on XL. XL BAM catches B because B's extra benefit disappears;
BAM's own ratio also declines. Existing TB shows no renewed late global clipping.
Raw-gradient medians grow.245->.469 on XL but decline.390->.257 on Medium.
XL unembedding static/dynamic RMS rise1.64/1.58 at5000 to7.70/7.14 at17500;
Medium near matched progress is1.73/1.29. XL unembedding gates approach1.
OutputHead explicitly omits vector output RMSNorm for RMT; final full-M norm
precedes both reads, so large readout amplitude can reach logits. It can also be
an innocuous gauge change compensated by logits weights; measure it, do not assume.

Probe runner MaxText/rmt_checkpoint_health.py: read-only parameter restore,
no optimizer updates/saves; same32 fixed-seed, shuffled TruePile4096 sequences
with full hashes and per-sequence outputs. Four sequences additionally produce
per-parameter/per-layer gradients and activation/cotangent scalar taps.
Masked-vocabulary-centered logit RMS, entropy/confidence, temperature-sweep CE
and its scale derivative directly test output calibration, avoiding softmax-invariant
logit offsets. Fixed-position matrix participation rank/top-energy power estimate
and attention entropy/concentration distinguish representation collapse from simple
amplitude growth. Parameter norms and gradient shares distinguish scale growth,
vanishing or dominating parameter groups. Probe-off/on CE and gradient identity
is a required focused CPU gate; actual checkpoint baseline/probe CE is checked too.

Missing training metrics already exposed: summed unembedding RMS; vocabulary-
centered logits RMS/entropy/calibration; grouped gradient shares/update-to-weight
ratios; raw carry M scale and dM/M; matrix rank/coherence; attention entropy and
NoPE/RoPE logit contribution balance. Determine diagnostic value before selecting
a small permanent metric set. No new formal training recipe is prescribed yet.


Probe CPU gate passed25.78s: unchanged CE and all parameter gradients on a scanned
reduced model, with finite calibration, attention and matrix diagnostics.
Initial TPU bf16 probe-off/on CE differed by+.000707 (same params/data, mathematical
identity, altered compiler fusion); record every gate and reject>.002. Paired
interventions must use one instrumented computation and show effects above the
measured numerical drift. No causal result is inferred from instrumentation alone.
Cohort uses the tail-four TruePile shards (full4097 records, outside both seen
prefixes), not the memorized first training batches or padded legacy validation.
Only the source params are restored. Diagnostic-only parameter restore is passed
through RMT_HEALTH_CHECKPOINT after parsing, keeping checkpoint manager disabled.
Latest probe source ff5f42f644e2b2fdd15d9f697098d9fb267820f8.
GCS artifacts verified prefix:
gs://newproject-1-llm_projects_europe-west4/diagnostics/rmt-crossscale-health-20261001/.


Paired baseline capture completed on the same owned EW4a v6e-1; all32 full
batch-tensor hashes match. Medium checkpoint5400 and XL17500 remain near-progress,
not exactly progress-matched. Mean per-sequence figures:

| metric | Medium B | XL B |
| --- | ---: | ---: |
| summed unembedding RMS | 2.994 | 12.681 |
| vocabulary-centered logits RMS | 2.621 | 4.236 |
| vocabulary common offset | -5.413 | -85.886 |
| CE at logits factors .9 / 1 / 1.1 | 2.4003 / 2.3699 / 2.3878 | 2.1343 / 2.1099 / 2.1282 |
| final matrix participation rank | 9.31 | 6.61 |
| final matrix dominant-energy estimate | .255 | .411 |
| embedding-state participation rank | 3.57 | 1.58 |
| raw pre-final-norm carry RMS | 12.06 | 50.02 |
| LM-head fraction of gradient squared norm, four sequences | .336–.399 | .562–.788 |

Temperature calibration rejects a simple overconfidence explanation. The logit
common offset is a sharper numerical suspect: OutputHead casts both operands to
bf16 and returns bf16 projected logits, so a softmax-invariant offset can consume
precision before the CE/softmax. Probe eeda62a2157197334089aa91b14ae421585b1717
keeps returned tensors unchanged, compares fp32 dot output of the same bf16
operands, direct fp32-to-bf16 rounding, and vocabulary-centering before bf16
rounding. It also measures the actual custom-CE backward's probability-mass error.
CPU CE/all-gradient identity passed25.50s with these callbacks. Effects must be
paired within this probe; compiler-fusion drift between probe builds is not loss
improvement.

Health gaps to fix in subsequent training, in order of diagnostic priority:
1. Vocabulary-common logit offset alongside centered logits RMS; actual dtype at
   output/CE, quantization error and CE-gradient mass conservation. Attention's
   float32_logits option does not control OutputHead.
2. Raw carry RMS and combined attention/MLP dM-to-M ratio plus cosine; existing
   head/tail M-RMS metrics use normalized read state and cannot expose raw growth.
3. Parameter-group gradient shares before/after clipping and actual Adam update
   relative to effective weights; do not call raw gradient/weight an Adam update.
4. A few fixed-position/layer matrix ranks/top-energy estimates; summed readout
   RMS and static/dynamic covariance; existing branch RMS alone misses the sum.
5. NoPE/RoPE logit contribution ratio and attention entropy at selected query
   positions, to distinguish underuse of context from route amplitude changes.
Full rank/gradient probes should stay offline or at sparse intervals; permanently
record the inexpensive scale/update signals first. Gate saturation already exists
and should be inspected by depth; it is not a newly missing metric.

Medium's initial diagnostic TPU was preempted twice; backup UC1a/UE5a queues were
released after Medium forward succeeded on the already-installed XL diagnostic
TPU. All three discarded node/queue identities were verified absent. User-owned
retained compilers were untouched. The shared diagnostic TPU remains owned until
paired numeric artifacts have been uploaded and verified locally.


Numerical follow-up eeda62a: paired fp32 dot-output CE changes XL by-.000894
(SE.000150 across32 sequences), Medium by-.000035. No simple numerical-error
explanation for the full lost gain is established. Relative logits-gradient error
is ~2.22% XL versus.63% Medium. Actual CE-gradient mass RMS is.00167 XL versus
.00179 Medium, so mass-conservation error is not uniquely amplified on XL.
The cast-rounding control was compiler-elided (reported zero round-trip error);
do not interpret it as a faithful bf16 rounding measurement. Explicit bit-level
rounding is needed if this minor branch is revisited.

A sharper layer-local anomaly: XL last MLP raw content RMS78.02 versus.318 in
its preceding layer and.474 in Medium's final layer. Static MLP-write-key RMS
is similar(.0537 XL/.0552 Medium); key growth does not explain it. XL final
static write RMS31.42 versus incoming carry24.95 and dynamic write7.16.
Medium final static write.112 versus carry11.20 and dynamic3.06. Total final
MLP dM/M is1.436 XL versus.279 Medium. Last-layer static write switches from
small to dominant, unlike every preceding XL layer.

Historical TB did contain a warning: final tail dynamic/static MLP-write ratio
XL54.79@5000 ->.706@10000 ->.204@15000 ->.197@17500, while Medium remains
27.62 near-progress4720 and25.42@5400. It was missed in broad depth-band medians.
The missing pieces are raw content/static-write amplitude and dM/M, not the
existing branch-ratio signal. Future reports must show the terminal layer apart
from medians when checking output routes.

Paired causal runner d43026e6df46f929b98e97cd69b71ce58760589c compares final static
MLP-write factors0/.5/.9/1/1.1, final dynamic MLP-write-off, embedding address
bias-zero, and dynamic-unembedding-off. Restore once, preserve source params,
use the same compiled forward and32 hashes for every intervention. Tests of
checkpoint necessity do not establish that retraining with a removed path helps.
CPU forward/gradient identity passed23.97s after adding MLP-input taps.


Paired parameter-necessity results d43026e, mean CE delta vs unchanged checkpoint:

| intervention | Medium B | XL B |
| --- | ---: | ---: |
| last static MLP write off | +.000107 | +.006754 |
| last static MLP write x.5 | -.000024 | +.001998 |
| last static MLP write x.9 | -.000076 | +.000086 |
| last static MLP write x1.1 | -.000027 | -.000461 |
| last dynamic MLP write off | +.063571 | +.051680 |
| embedding address pre-RMS bias off | +.962837 | +.961556 |
| dynamic unembedding off | +.683123 | +.306534 |

The small x.9/x1.1 effects are not decisive. The large XL terminal static-write
amplitude is not shown to harm checkpoint loss; deleting it worsens loss. Bias
deletion harms both scales similarly, so it does not localize an XL-specific bias
failure. Dynamic unembedding is still needed, but its marginal checkpoint value
is much lower on XL despite the much larger RMS and nearly-open gate. Necessity
on a jointly adapted checkpoint is not the gain from training a component.
XL final MLP input RMS.937 versus.443 in its preceding layer; a ~2x input-RMS
change alone cannot explain ~245x output-RMS jump (.318->78.02). Shared MLP
weights' RMS is similar by layer: directional alignment/nonlinearity must be
measured, rather than inferring exploding weights from output size.

Next scalar-only geometry probe9333aff3 records MLP dense branches, gated-product
and output RMS, token-common energy fraction and token-variable RMS, plus the
unembedding proxy/key variation. It also uses explicit bit-level bf16 rounding
in the numerical controls, because TPU eliminated cast-roundtrip rounding in
eeda62a. CPU CE/gradient identity remains the gate; original training tensors
are unchanged.


9333aff geometry refines/rejects two hypotheses:
- XL terminal dense branches RMS1.15/1.20, maxima93/89.6, but SwiGLU-product
  RMS30.78/max8333; preceding-layer product.129/max27.9. Medium final product
  .275/max22.7. This is correlated large activations in a few features, not an
  ordinary input-RMS or weight-RMS change.
- XL final output's token-common energy fraction is.0825, not near1; the giant
  output is mostly token-variable. Its unembedding normalized-key common fraction
  .6865 is LOWER than Medium's.8943. Thus a simple token-constant output or dynamic
  read-key freezing explanation is rejected. Dynamic marginal value and variation
  are distinct; high variation does not establish better predictions.
- Explicit IEEE bf16 rounding now reproduces original-output CE; centering before
  rounding improves only the same small numerical effect. No dominant numerical
  root cause has been demonstrated.

The next probe locates top-neuron energy and token RMS in the final two MLPs,
retains per-token CE/mask and matched cohort IDs, and tests whether loss degradation
is concentrated on the tokens with massive activations. This avoids treating an
outlier RMS as sufficient proof of unhealthy training. These arrays are small,
local diagnostic artifacts, never training checkpoint writes.


9333aff additionally rejects a frozen-key explanation: normalized unembedding
read keys are more token-variable on XL (~31% variable energy) than Medium
(~11%), even though dynamic-read marginal CE value is lower. Terminal XL SwiGLU
branches have RMS1.15/1.20 and maxima93/89.6; their product RMS30.78/max8333
shows strong concentration and correlation in a few features. Final output RMS78
has only8.25% token-common energy. The primary next discriminator is per-neuron
and per-token energy concentration, with paired CE at exactly those tokens.
Do not propose clipping or larger epsilon from this observation; establish the
source and functional effect of large activations first.


fbdba4a token/feature localization: exactly the same four final hidden units
5363/3073/4107/5817 dominate all32 XL sequences; together ~96.2% of SwiGLU
product energy. Output-token RMS>30 occurs on ~9.67% of cohort tokens. Their
mean CE is very LOW (~.083, per-sequence averaging), versus~2.275 for the other
tokens. Examples are newlines/indentation, predictable code and LaTeX transitions;
IDs decoded with the official Pythia tokenizer downloaded from
https://huggingface.co/EleutherAI/pythia-70m/resolve/main/tokenizer.json.
Thus large activations are not directly synonymous with bad predictions. Test
component necessity at these tokens and compare with ordinary transformer/BAM
controls before assigning a cause.

AllLocal BAM health control uses its own exact training source6446a1f170bdf5a63b232d1a78a92afea526430a,
worktree /data0/xd/bam-crossscale-health, codex/bam-crossscale-health,
diagnostic76cfe21f2bf055111d887df9aa2a922e6c3390c8, checkpoint20750.
It shares the32 cohorts; its progress differs from RMT17500, so loss difference
is not a same-progress causal estimate. Only the diagnostic MLP/OutputHead patch
is transplanted, not RMT implementation or normalization. CPU module forward and
all-gradient identity passed5.59s. Actual checkpoint probe gate is retained.


BAM20750 control: final MLP product RMS2.87/top-four energy61.2%, output RMS11.41;
RMT17500 product30.78/top-four96.2%, output78.04. BAM has concentrated features too,
but much weaker than RMT. The common-cohort CE comparison is progress-confounded:
RMT large-activation tokens CE.0730 vs BAM.0422; ordinary tokens2.3281 vs2.3007.
Both subsets are worse on the earlier checkpoint; this is not evidence that RMT's
loss problem is selectively concentrated on its outliers. A targeted zeroing of
the four terminal MLP output rows is the next checkpoint-necessity test, retaining
matched per-token CE and activation masks. Do not extrapolate this into a claimed
retraining benefit without training evidence.


Selected-unit necessity probe53c02f82978007f0dd1903f7c324bd87207eb882:
zeroing terminal W2 rows5363/3073/4107/5817 worsens paired mean CE by+.004959
(SE.000884,32 sequences, one compiled forward). On the original large-output-token
mask, token-weighted CE rises+.037882; remaining tokens+.000533. Thus these units
are useful for the easy-token subset, not demonstrated harmful outliers. Their
W1/Wg L2 norms are ~.91–1.03 versus preceding-layer ~.84–.92; W2 norms are
.69–.72 versus~.83–.92. Large activation is learned directional alignment and
nonlinearity, not gross parameter-norm explosion. Probe-off/on CE drift-.000999
is recorded; all interventions are paired within this compilation. Full231 artifacts
and HEALTH_COMPLETE verified at xl-neuron-ablation/, mirrored GCS xl/53c02f8/.

Current causal status: no dominant root cause established. Global clipping,
output overconfidence, output bf16 quantization alone, token-constant final output,
frozen dynamic keys, and harmful terminal static-write/feature amplitude are not
supported as explanations for the full loss-gain decay. This does not exclude a
long-run optimization effect from numerics or feature concentration. Checkpoint
necessity tests cannot establish retraining benefit or locate when a route took
over. The clearest temporal evidence remains the terminal write-branch switch.

The next most discriminating training control is XL layer SharedWriteNorm alone,
keeping B's original embedding (static raw embedding plus independent W_content
dynamic content), original learned matrix-normalization options and unembedding.
The previous failed XL combination also changed embedding; it does not isolate
layer SharedWriteNorm. Paired normalization of the summed dynamic+static MLP
input is another concrete control, distinct from the historical static-MLP-only
prenorm experiment. Neither formal training control is launched by this diagnosis.

Permanent health additions should prioritize cheap, causally interpretable signals:
- First/middle/last layers separately: raw carry RMS, static and dynamic content
  RMS, total dM/M and write–carry cosine. Never hide the last layer in depth bands.
- Terminal MLP branch/product/output tail energy, top4-unit fraction, large-token
  fraction and CE on the same token mask; amplitude alone is not health.
- Parameter-group gradient-squared shares before/after clipping; actual Adam
  update/weight ratios at sparse steps, separated from raw-gradient ratios.
- Output common logit offset plus centered scale, output dtype and calibration;
  cheap summed static+dynamic read RMS/cosine. Exact rank/Gram/SVD and quantization
  controls remain sparse offline diagnostics, not expensive every-step statistics.
Existing gate distributions and head/tail write ratios must be read by actual
layer and with correct Medium/XL metric names. Their previous omission was an
analysis/reporting failure, not a lack of instrumented training health.


Targeted backward inspectiona68512cb (same32 cohort, first4 gradients):
terminal four units account for31–81% of W2 gradient energy, but only~.07–1.0%
of W1/Wg energy. Their necessity is demonstrated; gradient concentration alone
is not evidence of harmful competition, especially with coordinatewise Adam.
LM-head vocabulary-common gradient energy is only~1.4e-8–2.3e-7 of LM-head
gradient energy in these four batches. The common logit offset has no demonstrated
large gradient-energy sink. These are batch1 diagnostic gradients, not training
batch128 clipping statistics. All114 artifacts and HEALTH_COMPLETE verified.

A correction to the initial dynamic-unembedding interpretation: turning it off
changes calibration/amplitude. Existing paired temperature sweeps give best-grid
CE gaps~+.281 Medium and~+.269 XL, versus uncalibrated~+.682/~+.314. Thus lower
raw deletion loss on XL is insufficient evidence of lower information value.
A denser paired temperature scan is running before drawing a functional-collapse
conclusion. Dynamic/static cosine and gate openness also cannot settle this.


All-layer necessity controls34bee18ef983899fc730476cb53af5f95426410e,32 matched
cohorts, same compiled forward per scale. Mean CE increases; calibrated columns
use one global temperature chosen on this diagnostic cohort, not retraining:

| dynamic route disabled | Medium raw / calibrated | XL raw / calibrated |
| --- | ---: | ---: |
| QK, all layers | .671948 / .669965 | .594049 / .594004 |
| V, all layers | .219793 / .218840 | .343646 / .338792 |
| MLP read, all layers | .466075 / .453937 | .826793 / .822827 |
| MLP write, all layers | 5.045749 / 5.011151 | 7.554103 / 7.552063 |
| unembedding | .682995 / .274811 | .307007 / .263928 |

The calibrated unembedding difference XL-Medium is-.010883 (paired SE.016316).
The information-necessity difference is not established; most raw difference
was calibration. XL dynamic V/MLP routes remain highly necessary. This rejects
an across-the-board dead-dynamic-route account, not a training-benefit claim.
Forward whole-route removals move states far off the learned distribution;
large loss changes are necessity diagnostics, never additive component benefits.
All cohort hashes and all192 variant health files per scale are verified by
analyze_routes.py; full results route-necessity-calibration-summary.json.

Attention is not globally saturated on XL: at last-layer querychunk3840,
mean entropy4.343/max-probability.295 versus Medium3.386/.368; first-layer
6.680/.075 versus4.862/.250. Greater diffuseness is an observation, not proof of
bad attention. It can reflect token/task/model changes and requires a targeted
control. AllLocal BAM terminal Q/K gates are lower on XL than Medium, but stable
from17500 to21500 (Q~.025,K~.0125); no abrupt late closing explains the trend.

The sharper conditioning probe dd0b34258fae8b6e8029fce69cdbccf8fc252aed adds
identity taps on static write contents, paired with existing dynamic-content
cotangents. CPU unchanged CE/all-parameter gradients passed32.738s. Static writes
use raw y; dynamic writes RMSNorm(y), whose content Jacobian scales~1/RMS(y)
and projects away radial direction. Large output can redirect content gradients
even if features are useful. Measure actual branch derivatives and token strata,
not merely infer it from forward RMS. Four first-cohort gradients preserve
batch1 context; never compare their norms with training batch128 clipping.


## Actual terminal W2 write-path gradients

71a2dc79 (Medium) and 2c742530 (XL) use gradient-only identity masks on
terminal static/dynamic writes. Identical forward losses for both/static/dynamic
modes; the W2 gradient sum reconstructs the unmasked gradient with relative error
.0014-.0021 (bf16). No optimizer update. Four matched unseen sequences, batch1;
these are not training-batch128 aggregate gradients or Adam updates.

| sequence | Medium static/dynamic W2 gradient L2 | XL |
| --- | ---: | ---: |
| 0 | .064194 | 1.470224 |
| 1 | .041151 | .537343 |
| 2 | .056853 | .777632 |
| 3 | .048046 | .387540 |

Median ratio .05245 versus .65749 (12.54x). Static/dynamic gradient cosine is
.54-.58 Medium versus .24-.33 XL. The content-tensor gradient ratios themselves
are similar (.041-.061 versus .049-.064), so RMSNorm's local derivative scale
alone does not explain the parameter-gradient difference.

For y=H W2, dL/dW2=H^T dL/dy. SwiGLU features H weight the token/content gradients;
large feature concentration can change the W2 path balance even when global
content-gradient norms match. The dynamic path's per-head RMSNorm also removes
the radial gradient component, unlike the static raw-y write. Actual measured
W2 gradients, not an inference from forward amplitude, establish this difference.
The selected-four-unit split is a follow-up to locate this weighting. W2 ratios
alone neither establish harmful gradients nor explain the full training gap:
Adam is coordinatewise, the four units improve CE, and the earlier temporal
checkpoints/actual optimizer updates are unavailable in this probe.

## Health metric gaps and priorities

1. **Sparse path/group parameter gradients and actual Adam updates.** Record W1/Wg/W2,
   embedding address/content, QK keys and unembedding update/weight ratios;
   include the fraction dominated by Adam epsilon. At diagnostic checkpoints split
   terminal static/dynamic W2 gradients and their cosine. Global grad norm and
   content RMS/gradient ratios would miss the newly observed difference.
2. **Raw activation scale and concentration at explicit boundary layers.** Record
   raw carry M and summed MLP input, SwiGLU product and output RMS/tails, write dM/M;
   terminal top-unit energy and CE on the same high-activation token mask. Keep
   first/last layers separate from middle-layer summaries. High amplitude on easy
   tokens is not itself bad health.
3. **Centered output/calibration.** Separate vocabulary-common logit offset from
   centered scale; include output dtype and sparse temperature CE. Raw logits RMS
   and raw route-deletion loss alone misdiagnosed unembedding information value.
4. **Retention for temporal diagnosis.** Keep a small set of checkpoints around
   gain-ratio turning points, with optimizer state. Terminal parameters plus TB
   cannot identify when feature/gradient balance changed or establish its cause.

Already available but previously missed: per-layer gate distributions and terminal
head/tail dynamic/static write ratios. XL's final-layer tail ratio fell from54.8
at5000 to.706 at10000 and.197 at17500; grouping depth bands hid the last-layer switch.
Correct the analysis/report before adding duplicate instrumentation. Exact ranks,
Gram/SVD, backward route splits and Adam diagnostics should be sparse/offline;
do not add all of them to every training step or contaminate speed comparisons.

Selected-unit follow-up b1717ab5295bcb9608ae62145b1d47990ed85840, same four XL
sequences/forward and branch gradients, reports only host/device reductions; no
training computation changes. Units5363/3073/4107/5817 own94.5-96.0% of the
static-path W2 gradient energy, versus6.8-36.3% of the dynamic-path energy.
On all other W2 rows, static/dynamic gradient ratios are .3683/.1254/.1774/.1006;
on the four rows, 2.3902/2.0088/1.7818/.8928. Thus most static-gradient energy
is localized to the same four concentrated features. The remaining rows still
have higher ratios than Medium's full W2 (.041-.064); the effect is not solely
those four units. Parameter-group/coordinate Adam updates are the missing link
between this gradient redistribution and a claim of impaired optimization.

Practical next discriminator: XL B + **layer SharedWriteNorm only**, preserving
B's original embedding, matrix/vector normalization, unembedding and MLP widths.
The previous failed XL combination changed embedding too, so it cannot reject
this isolated remedy. Watch both the loss-gain trajectory and the terminal
write/parameter-update balance; reducing an amplitude metric without improving
loss would falsify the proposed remedy. This diagnosis does not silently start
that formal experiment or modify ongoing training runtimes.

Optimizer follow-up ffc9254510bff146ad35d01639612cb98c3ff8e0 reads only the
terminal W2 first/second moments and count from the existing read-only OCDBT/Zarr3
checkpoint. Do not restore an entire optimizer or write any checkpoint. The actual
runtime uses adam_pax: moment slots are already bias-corrected. Inspect epsilon
attenuation and gradients divided by the stored denominator, clearly separated
from next-step Adam updates (which would also change moment slots).
The previous W2 update can be reconstructed from checkpoint moments, schedule
at count-1 and decoupled WD, neglecting fp32 rounding; a CPU test against the
actual adam_pax update passes. This is a sparse diagnostic, not an ongoing
training instrumentation change. A large raw path-gradient ratio may be canceled
by per-coordinate Adam scaling; check this before calling it optimization failure.

## Stored optimizer correction: raw gradient difference is mostly compensated

ffc92545, four same-cohort W2 gradients per scale and their checkpoint Adam slots:

| metric | Medium5400 | XL17500 |
| --- | ---: | ---: |
| checkpoint Adam count | 5401 | 17501 |
| raw static/dynamic gradient median | .05245 | .65749 |
| stored-denominator-scaled gradient median | .05133 | .06221 |
| scaled ratios, sequences0..3 | .06202/.04008/.05564/.04701 | .09654/.05342/.06749/.05693 |
| sqrt(variance) below epsilon fraction | 0 | 0 |
| reconstructed previous W2 update/weight L2 | .002642 | .001667 |

The raw cross-scale median-ratio contrast12.54x falls to1.21x under the stored
Adam denominator. The raw static-gradient concentration is real, but is not
established static-update domination. Neither content-gradient scale nor raw
parameter-gradient scale alone diagnoses learning health. Scaled current gradients
are not literal next-step updates: the next moment slots would also change, and
these probes use batch1. The reconstructed previous checkpoint update includes
actual moments, schedule and WD, neglecting fp32 rounding; its lower XL ratio is
not itself evidence of impairment (different LR/parameter scale). No epsilon
suppression of terminal W2 is observed. This does not clear other parameter groups.

This correction downgrades the gradient-competition explanation. Useful four-unit
features, beneficial static-write necessity, and near-compensated W2 gradients
mean we have not found a demonstrably harmful terminal component. The existing
normalization intervention is still a discriminating training control because
Medium showed a benefit and XL's previous combination was confounded; it is not
a fix for a root cause proven here. Baseline catch-up/architecture allocation also
remain possible, but cannot be asserted from the present endpoint snapshots.
To resolve causality rather than extend correlation probes indefinitely, retain
matched model/baseline checkpoints around the turning point and their optimizer
states, then test the isolated layer-write normalization control.

Bounded optimizer-group inspection6fefc9137a8502e0538e49fa49585e5fdbdff207:
52 slot groups per scale, first/middle/last layers;115 slices each, all finite.
Five oversized groups skipped (three MLP matrices, logits kernel, token embedding);
terminal W2 is covered independently above. All other inspected groups use
already-corrected adam_pax moments. Medium has0 epsilon-dominated coordinates in
these slices; XL has1.62% in terminal VO gate_kernel and.0039% in first K-mix.
First-layer QK average epsilon attenuation factors are .903-.922 Medium versus
.832-.845 XL; this is a real moderate difference, not demonstrated broad update
collapse. Dynamic embedding address-bias attenuation is .999993/.999981: the
historical initialization failure is not a current bias-update stall. Full scalar
artifacts optimizer-groups/{medium,xl}.json. No training optimizer/model changed.
Future checks should log continuous epsilon attenuation as well as the fraction
with sqrt(v)<epsilon; a binary threshold alone misses modest attenuation.
