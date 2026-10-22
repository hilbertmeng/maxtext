"""Read-only sampled-query attention observations, preserving the native AV path.

Queries arrive pre-scaled and after all rotary/local-BAM additions. Category IDs
are exclusive: 0 content, 1 punctuation-only, 2 whitespace-only, 3 special,
4 punctuation/symbol-only (with at least one Unicode symbol).
First position is reported separately and excluded from content regardless of ID.
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp

ACTIVE = False
CATEGORIES = None
_INSTALLED = False
_BAM_STACK = []
METRICS = ('first', 'first4', 'punctuation', 'whitespace', 'special', 'content',
           'sink_union', 'entropy', 'entropy_normalized', 'content_entropy',
           'content_entropy_normalized', 'top1_content', 'top1_first',
           'top1_punctuation', 'local4', 'local32', 'content_far32',
           'content_uniform_share', 'peak', 'mean_lag', 'symbols', 'top_key_index')


def positions(length):
    # Prefix queries cannot fairly distinguish attention sinks from valid context.
    return np.unique(np.linspace(min(64, length - 1), length - 1, min(64, length), dtype=np.int32))


def statistics(query, key, categories, segments=None, window=None, soft_cap=0.):
    """Return [B,H,Q,metric], maps [B,H,8,S]; all stats accumulate FP32.

    The maps reconstruct softmax from the actual kernel inputs; they do not
    replace the native attention operator or its value output.
    """
    length = query.shape[1]
    qpos = jnp.asarray(positions(length))
    if query.shape[2] != key.shape[2]:
        assert query.shape[2] % key.shape[2] == 0
        key = jnp.repeat(key, query.shape[2] // key.shape[2], axis=2)
    logits = jnp.einsum('bqhd,bshd->bhqs', query[:, qpos].astype(jnp.float32),
                        key.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
    if soft_cap:
        logits = jnp.tanh(logits / soft_cap) * soft_cap
    src = jnp.arange(key.shape[1])
    valid = (src[None, :] <= qpos[:, None])[None]
    if window is not None:
        valid = valid & (src[None, :] > qpos[:, None] - window)[None]
    if segments is not None:
        valid = valid & (segments[:, qpos, None] == segments[:, None, :]) & (segments[:, qpos, None] != 0)
    valid = valid[:, None]
    alpha = jax.nn.softmax(jnp.where(valid, logits, -1e30), -1)
    first = src == 0
    cats = categories[:, None, None, :]
    content = (cats == 0) & ~first
    punct, space, special, symbols = cats == 1, cats == 2, cats == 3, cats == 4
    sink = first | punct | space | special | symbols
    def mass(mask):
        return jnp.sum(alpha * mask, -1)
    def entropy(a):
        return -jnp.sum(jnp.where(a > 0, a * jnp.log(jnp.maximum(a, 1e-30)), 0), -1)
    cmass = mass(content)
    cp = jnp.where(content, alpha, 0) / jnp.maximum(cmass[..., None], 1e-30)
    h, ch = entropy(alpha), entropy(cp)
    counts = jnp.sum(valid, -1)
    ccounts = jnp.sum(content & valid, -1)
    peak_idx = jnp.argmax(alpha, -1)
    topcat = jnp.take_along_axis(jnp.broadcast_to(cats, alpha.shape), peak_idx[..., None], -1)[..., 0]
    lag = qpos[:, None] - src[None, :]
    vals = [mass(first), mass(src < 4), mass(punct), mass(space), mass(special), cmass,
            mass(sink), h, h / jnp.maximum(jnp.log(jnp.maximum(counts, 2)), 1e-30),
            ch, ch / jnp.maximum(jnp.log(jnp.maximum(ccounts, 2)), 1e-30),
            ((topcat == 0) & (peak_idx != 0)).astype(jnp.float32),
            (peak_idx == 0).astype(jnp.float32), (topcat == 1).astype(jnp.float32),
            mass(lag < 4), mass(lag < 32), mass(content & (lag >= 32)),
            jnp.broadcast_to(ccounts / jnp.maximum(counts, 1), cmass.shape),
            jnp.max(alpha, -1), jnp.sum(alpha * lag, -1), mass(symbols), peak_idx.astype(jnp.float32)]
    vals = [jnp.broadcast_to(v, cmass.shape) for v in vals]
    stats = jnp.stack(vals, -1)
    # Preserve all heads for eight spread queries, enabling a map audit.
    map_idx = np.unique(np.linspace(0, len(positions(length)) - 1, 8, dtype=int))
    return stats, alpha[:, :, map_idx].astype(jnp.bfloat16)


def record(module, query, key, segments, window=None, soft_cap=0.):
    if not ACTIVE or module.is_initializing():
        return
    stats, maps = statistics(query, key, CATEGORIES, segments, window, soft_cap)
    # Batch remains leading; scan adds a leading layer dimension.
    module.sow('intermediates', 'stream_head_stats', stats)
    module.sow('intermediates', 'stream_head_maps', maps)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from layers import attentions
    original_bam = attentions.BamAttention.__call__
    @functools.wraps(original_bam)
    def bam(self, *args, **kwargs):
        _BAM_STACK.append(self)
        try:
            return original_bam(self, *args, **kwargs)
        finally:
            _BAM_STACK.pop()
    attentions.BamAttention.__call__ = bam
    if hasattr(attentions.BamAttention, '_attention_block'):
        original_block = attentions.BamAttention._attention_block
        @functools.wraps(original_block)
        def block(self, query, key, value, segments, *, q0, s0, window_size, **kwargs):
            if ACTIVE and (q0 != 0 or s0 != 0 or query.shape[1] != key.shape[1]):
                raise ValueError('Head observation requires a full causal block or Splash')
            record(self, query, key, segments, window_size, self.config.attn_logits_soft_cap)
            return original_block(self, query, key, value, segments, q0=q0, s0=s0,
                                  window_size=window_size, **kwargs)
        attentions.BamAttention._attention_block = block
    if hasattr(attentions, '_bam_splash_attention'):
        original_splash = attentions._bam_splash_attention
        @functools.wraps(original_splash)
        def splash(query, key, value, segments, cfg, mesh, window=None, **kwargs):
            record(_BAM_STACK[-1], query, key, segments, window, cfg.attn_logits_soft_cap)
            return original_splash(query, key, value, segments, cfg, mesh, window=window, **kwargs)
        attentions._bam_splash_attention = splash
    original_mha = attentions.AttentionOp.__call__
    @functools.wraps(original_mha)
    def mha(self, query, key, value, segments, *args, **kwargs):
        # All probed Mudd layers use the global flash path. Fail on other masks
        # instead of silently reconstructing a different attention pattern.
        window = self.sliding_window_size
        if str(self.attention_type).lower().endswith('global'):
            window = None
        record(self, query, key, segments, window, self.attn_logits_soft_cap)
        return original_mha(self, query, key, value, segments, *args, **kwargs)
    attentions.AttentionOp.__call__ = mha
