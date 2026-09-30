# Medium learnable zero-initialized static embedding address

RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormSeedKeyZeroInit`.
User-directed. Worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`.
UE5a trainer `xd-v5p-16-2909308-maxtext`. Borrow retained EW4a FLEX_START
`llm-jax-v6e-1-1` for AOT without lifecycle ownership; never reclaim it.

Derive from SharedEmbedWriteNorm, initialize only static seed_key16x48 to zero,
static scale1. Both embedding content paths use the same per-head RMSNorm;
dynamic address/gates, layer writes and learned full-M pre-norm remain unchanged.
Static seed receives a nonzero loss gradient from the start and can grow;
Scale0 permanently suppresses that route and its gradient. No content projection
restored. Same parameter/RNG slots; MLP4100,431888672 parameters,18 identical L,
layer scan, pure JAX, TruePile4096 local replica.13500 steps/checkpoint200.

Direct baselines: Scale0, Scale006, LearnedScaleSharedEmbedNorm. Bet terminal
RUN-Scale0 -.001, speed.371step/s flat; review2800/5000. Track loss and embedding
static/dynamic RMS, cosine and gate. Initial ratio/cosine has a zero static
denominator and its epsilon-capped value is not a pathology.

Focused CPU gates: full effective scope/budget; all common initialized parameters
exactly match Scale0 excluding seed_key; initial forward/dynamic health match;
finite scanned gradients, nonzero seed gradient, one update opens static route.
Rerun Scale0 focused tests for the default normal-initialization regression.
CPU/AOT/trainer prequeue parallel; runtime9dfa158a266a105fb3154579dfa5e196aa5cf960.

Startup verified2026-09-30: four focused CPU gates pass35.1s; retained FLEX AOT
verified, FIRST_STEP2 and step19 reached. Actual worker confirms seed-key zero
initTrue, static scale1.0, Loaded compiled function, and local dataset
 gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096.
Loss finite and falling through44; .373step/s flat versus Scale0.372 and
Scale006/SharedEmbedNorm.371 with matched health. Queue READY observed15:28:07UTC,
controller15:28:15, train process launched15:30:38. Retained compiler preserved.
