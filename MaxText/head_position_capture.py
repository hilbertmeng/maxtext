"""Exact per-position head activation, preserving the native attention output.

Active iff the argmax key is neither position 0/1 nor a BOS, period or
newline token (GPT-NeoX/Pythia IDs 0, 15 and 187). Every query is retained.
The QK product is chunked along query length; no full T*T map is transferred.
"""
import numpy as np
import jax
import jax.numpy as jnp
import head_attention_capture as hooks

ACTIVE = False
TOKENS = None
CATEGORIES = None  # Kept for the shared runner API; not used for this definition.
METRICS = ('position_top_key_index', 'position_entropy', 'position_allowed_entropy')
EXCLUDED_TOKEN_IDS = (0, 15, 187)


def positions(length):
    return np.arange(length, dtype=np.int32)


def statistics(query, key, tokens, segments=None, window=None, soft_cap=0., chunk_size=64):
    """Return argmax key indices and two FP32 entropy arrays, all [B,H,T].

    Entropy is computed on the original attention map. Allowed entropy is an
    auxiliary conditional entropy after removing the exact excluded key set.
    Ties select the smallest key index, matching jnp.argmax.
    """
    length = query.shape[1]
    if key.shape[1] > 65536:
        raise ValueError('Stored uint16 key indices require source length <=65536')
    chunk_size = min(chunk_size, length)
    chunks = (length + chunk_size - 1) // chunk_size
    query = jnp.pad(query, ((0, 0), (0, chunks * chunk_size - length), (0, 0), (0, 0)))
    if query.shape[2] != key.shape[2]:
        assert query.shape[2] % key.shape[2] == 0
        key = jnp.repeat(key, query.shape[2] // key.shape[2], axis=2)
    src = jnp.arange(key.shape[1])
    excluded = (src[None] < 2) | (tokens == 0) | (tokens == 15) | (tokens == 187)
    if segments is not None:
        padded_segments = jnp.pad(segments, ((0, 0), (0, query.shape[1] - length)))

    def entropy(logits, valid):
        logp = jax.nn.log_softmax(jnp.where(valid, logits, -1e30), axis=-1)
        p = jnp.where(valid, jnp.exp(logp), 0.)
        return -jnp.sum(p * logp, axis=-1)

    def block(_, block_index):
        offset = block_index * chunk_size
        q = jax.lax.dynamic_slice_in_dim(query, offset, chunk_size, axis=1)
        qpos = offset + jnp.arange(chunk_size)
        logits = jnp.einsum('bqhd,bshd->bhqs', q.astype(jnp.float32),
                            key.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
        if soft_cap:
            logits = jnp.tanh(logits / soft_cap) * soft_cap
        valid = ((src[None, :] <= qpos[:, None]) & (qpos[:, None] < length))[None]
        if window is not None:
            valid = valid & (src[None, :] > qpos[:, None] - window)[None]
        if segments is not None:
            qs = jax.lax.dynamic_slice_in_dim(padded_segments, offset, chunk_size, axis=1)
            valid = valid & (qs[:, :, None] == segments[:, None, :]) & (qs[:, :, None] != 0)
        valid = valid[:, None]
        top = jnp.argmax(jnp.where(valid, logits, -1e30), axis=-1).astype(jnp.uint16)
        h = entropy(logits, valid)
        ah = entropy(logits, valid & ~excluded[:, None, None, :])
        return None, (top, h, ah)

    _, result = jax.lax.scan(block, None, jnp.arange(chunks))
    return tuple(jnp.transpose(x, (1, 2, 0, 3)).reshape(x.shape[1], x.shape[2], -1)[..., :length]
                 for x in result)


def record(module, query, key, segments, window=None, soft_cap=0.):
    if not ACTIVE or module.is_initializing():
        return
    if query.shape[1] != key.shape[1]:
        raise ValueError('Position observer requires full causal Q/K inputs')
    result = statistics(query, key, TOKENS, segments, window, soft_cap)
    for name, value in zip(METRICS, result):
        module.sow('intermediates', 'stream_' + name, value)


def install():
    # Shared hooks observe the exact Q/K passed to the historical native kernel.
    # Each probe process installs only this observer; model source is unchanged.
    hooks.install()
    hooks.record = record
