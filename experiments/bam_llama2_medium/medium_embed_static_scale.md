# Medium normalized static embedding write amplitude control

Autonomous Medium experiment1/2, following the2800 review of `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm`.
RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale006`; baselines `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm` and `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm`.
Worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`.
UE5a `xd-v5p-16-2909306-maxtext`; borrow idle FLEX_START EW4a v6e-1
`llm-jax-v6e-1-1`, never adopt lifecycle ownership.

Retain shared normalized embedding contents, full-M learned pre-norm and all
other parent settings. Multiply only static embedding address/readout by a fixed
.006 outside the learned seed-key parameter: do NOT shrink its initializer
(Adam could quickly undo that scale), do NOT scale dynamic contents. No new
parameters, MLP4100,431888672 parameters.18 identical L layers, layer scan,
pure JAX; TruePile4096 local UE5a replica,13500 schedule/checkpoint200.

Rationale: at step0, baseline E staticM RMS .005942 vs normalized-static .976303
(ratio .006086). DynamicM RMS .023469 was exactly equal; dynamic/static ratio
therefore shifted3.949 to.02404. At2800 the failed arm cosine is-.5899 versus
E+.09845. Fixed.006 restores initial balance to within~1.4% while retaining
normalization and its removal of per-head embedding magnitude from content.
Recovery near E would support initialization/balance as the large-loss mechanism;
a persistent substantial deficit would weaken that explanation. Neither outcome
alone establishes whether normalization is universally good or bad.

Bet at2800: RUN-E +.003, RUN-failed normalized-static approximately-.054;
terminal RUN-E +.003, speed.371 (flat). Review2800/5000; this is a causal control,
not solely a competition for the lowest loss. One autonomous Medium slot remains.

CPU scope: full target budget; forward/gradient and actual embedding health prove
static amplitude scales .006, dynamic amplitude and gates unchanged; legacy
normalized embedding equation/scope regression. CPU/AOT/prequeue parallel, all
gates required. Runtime7cba5c220f8541d99391a1b1f4622fadee4e6040.

Startup verified2026-09-30: focused CPU checks and retained FLEX AOT passed;
FIRST_STEP2 and step14 at.371step/s, flat versus E.371 with matched health.
Actual dataset gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096.
Finite falling losses through171; compiler retained. Training queue READY observed
13:09:14UTC, controller13:09:23; worker source ready13:11:23.
