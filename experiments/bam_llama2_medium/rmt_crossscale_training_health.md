# MediumProp versus XLProp dynamic RMT B training health

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
