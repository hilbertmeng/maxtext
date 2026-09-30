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
CPU/AOT/trainer prequeue parallel; runtime and startup pending.
