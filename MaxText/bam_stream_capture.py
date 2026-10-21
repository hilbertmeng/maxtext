"""Read-only forward observers; parameter trees and numerical results are unchanged."""
import functools
import numpy as np
from flax import linen as nn
import jax
import jax.numpy as jnp

ACTIVE = False
_INSTALLED = False


def positions(length):
    common = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2047]
    extra = np.linspace(0, length - 1, 9, dtype=int).tolist()
    return np.array(sorted(set(i for i in common + extra if i < length)), np.int32)


def enabled(module):
    return ACTIVE and not module.is_initializing()


def put(module, name, value, *, sample=True, matrix=False):
    if not enabled(module) or value is None:
        return
    value = jnp.asarray(nn.meta.unbox(value))
    if matrix and getattr(module.config, 'bam_pallas_core', False):
        body = getattr(module.config, 'bam_pallas_body', None)
        if body == 'v8t':
            value = jnp.swapaxes(value, -1, -2)
        elif body in ('v7', 'v7u'):
            value = jnp.transpose(value, (0, 3, 1, 2))
        else:
            value = jnp.transpose(value, (0, 3, 2, 1))
    if sample:
        value = value[:, positions(value.shape[1])]
    module.sow('intermediates', 'stream_' + name, value)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from layers import fusion, attentions, linears, models

    original = fusion.SubDecoderLayer.__call__
    @functools.wraps(original)
    def block(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if enabled(self):
            put(self, 'x_in', args[0]);put(self, 'x_out', result[0])
            put(self, 'M_in', kwargs.get('M_in'), matrix=True)
            put(self, 'M_out', result[1], matrix=True)
            if self.has_variable('params', 'mlp_write_gate_bias'):
                put(self, 'mlp_write_bias', self.get_variable('params', 'mlp_write_gate_bias'), sample=False)
        return result
    fusion.SubDecoderLayer.__call__ = block

    original_attn = attentions.BamAttention.__call__
    @functools.wraps(original_attn)
    def attention(self, *args, **kwargs):
        result = original_attn(self, *args, **kwargs)
        if enabled(self):
            put(self, 'attention_vector', result[0])
            for arm in 'qkvo':
                key = getattr(self, f'static_{arm}_key', None)
                if key is not None:
                    put(self, 'static_' + arm + '_key', key, sample=False)
            put(self, 'M_read_input', kwargs.get('M_in'), matrix=True)
            matrix = kwargs.get('M_in')
            if matrix is not None:
                put(self, 'M_previous_token', jnp.concatenate((matrix[:, :1], matrix[:, :-1]), axis=1), matrix=True)
            put(self, 'read_key_epsilon', self._read_key_epsilon, sample=False)
            put(self, 'read_key_scale', self._read_key_scale, sample=False)
            if hasattr(self, 'm_shift_bias'):
                put(self, 'm_shift_bias', self.m_shift_bias, sample=False)
            put(self, 'M_after_attention_call', result[1], matrix=True)
            if getattr(self.config, 'bam_pallas_core', False):
                data, logits, address = result[2]
                put(self, 'attention_write_data_raw', data)
                put(self, 'attention_write_logits', logits)
                put(self, 'attention_write_address_raw', address)
        return result
    attentions.BamAttention.__call__ = attention

    original_mlp = linears.MlpBlock.__call__
    @functools.wraps(original_mlp)
    def mlp(self, *args, **kwargs):
        result = original_mlp(self, *args, **kwargs)
        put(self, 'mlp_vector', result)
        return result
    linears.MlpBlock.__call__ = mlp

    original_dense = linears.DenseGeneral.__call__
    @functools.wraps(original_dense)
    def dense(self, *args, **kwargs):
        result = original_dense(self, *args, **kwargs)
        if self.name in ('mlp_address_up', 'mlp_write_gate', 'P_loc_up', 'M_shift_gate', 'W_emb_u', 'W_emb_v_up', 'W_emb_g'):
            put(self, 'dense_' + self.name, result)
        return result
    linears.DenseGeneral.__call__ = dense

    original_emb = models.EmbeddingBamWrite.__call__
    @functools.wraps(original_emb)
    def embedding(self, *args, **kwargs):
        result = original_emb(self, *args, **kwargs)
        put(self, 'embedding_vector', args[0]);put(self, 'embedding_M', result, matrix=True)
        if enabled(self):
            put(self, 'embedding_gate_bias', self.get_variable('params', 'emb_gw_b0'), sample=False)
        return result
    models.EmbeddingBamWrite.__call__ = embedding

    original_factors = attentions.BamAttention._deferred_write_factors
    @functools.wraps(original_factors)
    def factors(self, *args, **kwargs):
        result = original_factors(self, *args, **kwargs)
        for name, value in zip(('attention_write_content', 'attention_write_address', 'attention_write_gate'), result):
            put(self, name, value)
        return result
    attentions.BamAttention._deferred_write_factors = factors

    original_write = attentions.BamAttention._write
    @functools.wraps(original_write)
    def write(self, *args, **kwargs):
        result = original_write(self, *args, **kwargs)
        if enabled(self):
            put(self, 'attention_write_delta_M', result[0] - self.config.bam_lambda_decay * args[2])
            put(self, 'attention_write_gate', result[1])
        return result
    attentions.BamAttention._write = write

    original_merge = attentions.BamAttention.merge_mlp_write
    @functools.wraps(original_merge)
    def merge(self, mlp_head, mlp_gate, factors, M_in, independent_address=None):
        result = original_merge(self, mlp_head, mlp_gate, factors, M_in,
                                independent_address=independent_address)
        if enabled(self):
            content = self.write_data_norm(mlp_head) if self._write_data_rms else mlp_head
            scale = 1 / jnp.sqrt(self.num_query_heads) if self.config.bam_sqrt_n_scale else 1
            content = scale * mlp_gate[..., None] * content
            address = factors[1] if independent_address is None else self.write_address_norm(independent_address)
            put(self, 'mlp_write_content', content);put(self, 'mlp_write_address', address)
            put(self, 'mlp_write_gate', mlp_gate)
        return result
    attentions.BamAttention.merge_mlp_write = merge

    original_key = attentions.BamAttention._project_column_read_key
    @functools.wraps(original_key)
    def key(self, name, *args, **kwargs):
        result = original_key(self, name, *args, **kwargs)
        put(self, 'read_key_' + name, result)
        return result
    attentions.BamAttention._project_column_read_key = key

    original_gate = attentions.BamAttention._project_read_gate_logits
    @functools.wraps(original_gate)
    def gate(self, name, *args, **kwargs):
        result = original_gate(self, name, *args, **kwargs)
        put(self, 'read_logits_' + name, result)
        if enabled(self):
            put(self, 'read_gate_' + name, self._read_gate_activation(result.astype(jnp.float32)))
        return result
    attentions.BamAttention._project_read_gate_logits = gate

    original_static = attentions.BamAttention._static_gate
    @functools.wraps(original_static)
    def static(self, name, *args, **kwargs):
        result = original_static(self, name, *args, **kwargs)
        put(self, 'static_gate_' + name, result)
        return result
    attentions.BamAttention._static_gate = static

    original_fused = attentions.BamAttention._fused_input_projections
    @functools.wraps(original_fused)
    def fused(self, x, names):
        result = original_fused(self, x, names)
        if enabled(self):
            for name, value in result.items():
                if name in ('W_lq_c8_up', 'W_lk_c8_up', 'W_R_up'):
                    arm = {'W_lq_c8_up': 'q', 'W_lk_c8_up': 'k', 'W_R_up': 'vo'}[name]
                    value = value + jnp.asarray(getattr(self, 'general_' + arm + '_pre_bias'), value.dtype)
                    put(self, 'read_key_' + {'q':'W_lq_c8','k':'W_lk_c8','vo':'W_R'}[arm], value)
                elif name in ('W_lq_gate', 'W_lk_gate', 'W_R_gate', 'W_lv_gate'):
                    value = value + jnp.asarray(getattr(self, name + '_b0'), value.dtype)
                    put(self, 'read_logits_' + name, value)
                    put(self, 'read_gate_' + name, self._read_gate_activation(value.astype(jnp.float32)))
                elif name.startswith('general_') and name.endswith('_static_gate'):
                    value = value + jnp.asarray(getattr(self, name + '_bias'), value.dtype)
                    put(self, 'static_gate_' + name, 1.1 * jax.nn.sigmoid(value))
                elif name == 'M_shift_gate':
                    put(self, 'm_shift_gate', jax.nn.sigmoid(value.astype(jnp.float32) + self.m_shift_bias.astype(jnp.float32)))
        return result
    attentions.BamAttention._fused_input_projections = fused

