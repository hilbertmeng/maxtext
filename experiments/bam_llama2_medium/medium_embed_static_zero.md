# Medium zero-static embedding write

RUN `RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNormScale0`.
User-directed. Worktree `/data0/xd/rmt-xlprop-noo`, branch `codex/rmt-xlprop-noo`.
Trainer `xd-v5p-16-2909307-maxtext`, UE5a only; retained EW4a FLEX_START
`llm-jax-v6e-1-1` borrowed for AOT, never reclaimed.

Parent suffix `SharedEmbedWriteNormScale006`; set only
`rmt_embedding_static_write_scale=0`. Static contribution and its data gradient
are exactly zero throughout training. Retain seed_key parameter/RNG slot for
controlled initialization and identical optimizer tree; no independent content
projection restored. MLP4100 and431888672 parameters unchanged.18 L layers,
layer scan, pure JAX, TruePile4096 zone-local data, total13500/checkpoint200.
This is embedding-only: attention/MLP static writes remain unchanged.

Direct baselines: Scale006 and LearnedScaleSharedEmbedNorm. Historical
attention/MLP DynamicOnlyWrite lost+.013017 through2800, but retained static
embedding and removed a raw-content layer-write component, so is not this test.

Bet versus Scale006: terminal0 with plausible +/- .002, no firm winning bias;
speed.371step/s, flat. Review2800/5000. Inspect finite loss, embedding gate,
dynamic write RMS and raw gradients. Dynamic/static ratio has a zero denominator
and is no longer a useful metric; its finite epsilon-capped value is not a fault.

CPU targeted checks: full effective-config/budget equality; zero static write and
seed-key data gradient; unchanged dynamic amplitude/gates on common parameters;
finite scanned forward/backward with active embedding/address gradients.
Also rerun the existing nonzero scale test to protect its parent path. CPU/AOT
and trainer prequeue run concurrently with all gates required before training.
Runtime and startup evidence pending.
