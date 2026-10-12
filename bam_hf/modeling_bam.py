"""PyTorch BAM full-M generalized-read decoder (inference port of the MaxText implementation).

Port of the executed path of MaxText `BamAttention` / `SubDecoderLayer` / `EmbeddingBamWrite`
(runtime b83223f) for `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile`;
the fused Pallas v8t core computes the same equations. Notation per token: M [K, V] (K = data/column
output, V = address), N heads.

  read key   r = norm(up(gelu(down(x))) + pre_bias)                     (one shared `down`)
  Q/K read   col = M r * 0.1 sigmoid(gate), truncated to 72 columns, + static read M[:72] s * g_static
  VO read    y = M norm(r_vo);  V = 0.1 sigmoid(lv) y + M s_v g_v ;  O_loc = 0.1 sigmoid(lo) y + M s_o g_o
  attention  Q = [col_q | RoPE(Wq x)], K = [col_k | RoPE(Wk x)], V as above;  o = softmax(QK^T/sqrt d) V + O_loc
  write      M += sum_n sigmoid(gw) norm(o[:K]) (x) norm(P_loc(x))   (+ the MLP write on its layers)

M is per token (it never mixes positions), so incremental decoding caches only attention K/V.

`config.bam_enabled=False` selects the parameter-matched MHA control (`Llama2XLPropTruePileMHA`,
runtime e30c1b8): the same decoder with standard Q/K/V projections, RoPE over the whole head and
no M stream, as in MaxText where both models share the fusion decoder layer.
"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
  from .configuration_bam import BamConfig
except ImportError:      # loaded as a top-level module
  from configuration_bam import BamConfig


def rms_norm(x, eps):
  """Parameter-free RMS norm over the last axis with FP32 statistics, returned in x.dtype."""
  f = x.float()
  return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def gelu(x):
  return F.gelu(x, approximate='tanh')     # flax nn.gelu default


class BamRMSNorm(nn.Module):
  """MaxText RMSNorm: zero-initialized scale applied as (1 + scale)."""

  def __init__(self, dim, eps):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(dim))
    self.eps = eps

  def forward(self, x):
    return rms_norm(x, self.eps) * (self.weight.to(x.dtype) + 1)


class HeadLinear(nn.Linear):
  """Linear map to [..., *out_shape] (MaxText DenseGeneral with multi-axis features)."""

  def __init__(self, in_features, out_shape, bias=False):
    self.out_shape = tuple(out_shape)
    super().__init__(in_features, math.prod(self.out_shape), bias=bias)

  def forward(self, x):
    return super().forward(x).unflatten(-1, self.out_shape)


class BamRotary(nn.Module):
  """MaxText RotaryEmbedding (half-split, not interleaved) over the standard Q/K tail."""

  def __init__(self, dim, min_timescale, max_timescale):
    super().__init__()
    fraction = 2 * torch.arange(dim // 2, dtype=torch.float32) / dim
    self.register_buffer('timescale', min_timescale * (max_timescale / min_timescale) ** fraction,
                         persistent=False)

  def forward(self, x, positions):
    """x [B, T, N, D]; positions [B, T]."""
    angle = positions.float()[:, :, None, None] / self.timescale
    sin, cos = torch.sin(angle).to(x.dtype), torch.cos(angle).to(x.dtype)
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class GatedLinear(nn.Module):
  """Gate logits = W x + b (the bias kept separately as in MaxText)."""

  def __init__(self, in_features, out_shape):
    super().__init__()
    self.proj = HeadLinear(in_features, out_shape)
    self.bias = nn.Parameter(torch.zeros(out_shape))

  def forward(self, x):
    return self.proj(x) + self.bias.to(x.dtype)


class BamEmbeddingWrite(nn.Module):
  """Initial M from one gated outer-product write of the token embedding."""

  def __init__(self, config: BamConfig):
    super().__init__()
    n, h = config.num_attention_heads, config.hidden_size
    self.data = HeadLinear(h, (n, config.bam_k))
    self.address_down = nn.Linear(h, config.embedding_write_bottleneck_dim, bias=False)
    self.address_up = HeadLinear(config.embedding_write_bottleneck_dim, (n, config.bam_v), bias=True)
    self.gate = GatedLinear(h, (n,))
    self.eps = config.rms_norm_eps

  def forward(self, x):
    gate = torch.sigmoid(self.gate(x))
    data = rms_norm(self.data(x), self.eps)
    address = rms_norm(self.address_up(gelu(self.address_down(x))), self.eps)
    return torch.einsum('btnk,btnv->btkv', gate[..., None] * data, address)


def causal_attention(q, k, v, layer_idx, attention_mask=None, past_key_value: Optional[Cache] = None):
  """q/k/v [B,T,N,D] with q pre-scaled; returns [B,T,N,D]. Appends k/v to the cache."""
  q, k, v = (z.transpose(1, 2) for z in (q, k, v))           # [B,N,T,D]
  if past_key_value is not None:
    k, v = past_key_value.update(k, v, layer_idx)
  if attention_mask is None:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] > 1, scale=1.0)
  else:
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=1.0)
  return y.transpose(1, 2)


class MhaAttention(nn.Module):
  """MaxText `Attention` as executed by the MHA control: Q/K/V projections, full-head RoPE."""

  def __init__(self, config: BamConfig, layer_idx: int):
    super().__init__()
    self.layer_idx = layer_idx
    h, n, d = config.hidden_size, config.num_attention_heads, config.head_dim
    self.head_dim = d
    self.q_proj = HeadLinear(h, (n, d))
    self.k_proj = HeadLinear(h, (n, d))
    self.v_proj = HeadLinear(h, (n, d))
    self.rotary = BamRotary(d, config.rope_min_timescale, config.rope_max_timescale)
    self.o_proj = nn.Linear(n * d, h, bias=False)

  def forward(self, x, M, positions, attention_mask=None, past_key_value=None, defer_write=False):
    q = self.rotary(self.q_proj(x), positions)
    k = self.rotary(self.k_proj(x), positions)
    q = q / torch.tensor(math.sqrt(self.head_dim), dtype=q.dtype)
    y = causal_attention(q, k, self.v_proj(x), self.layer_idx, attention_mask, past_key_value)
    return self.o_proj(y.flatten(-2)), M


class BamAttention(nn.Module):

  def __init__(self, config: BamConfig, layer_idx: int):
    super().__init__()
    self.layer_idx = layer_idx
    h, n, d = config.hidden_size, config.num_attention_heads, config.head_dim
    v, r = config.bam_v, config.read_bottleneck_dim
    self.num_heads, self.head_dim, self.bam_k = n, d, config.bam_k
    self.qk_cols = config.qk_col_dim
    self.read_eps, self.read_scale = config.read_key_eps, config.read_key_scale
    self.eps = config.rms_norm_eps
    self.static_gate_init = config.static_read_gate_init
    # Standard (rotated) Q/K tail.
    self.q_proj = HeadLinear(h, (n, config.standard_qk_dim))
    self.k_proj = HeadLinear(h, (n, config.standard_qk_dim))
    self.rotary = BamRotary(config.standard_qk_dim, config.rope_min_timescale, config.rope_max_timescale)
    # Read keys: shared GELU bottleneck, per-arm up projection + pre-bias.
    self.read_down = nn.Linear(h, r, bias=False)
    self.q_read_up = HeadLinear(r, (n, v))
    self.k_read_up = HeadLinear(r, (n, v))
    self.vo_read_up = HeadLinear(r, (n, v))
    self.q_read_pre_bias = nn.Parameter(torch.zeros(n, v))
    self.k_read_pre_bias = nn.Parameter(torch.zeros(n, v))
    self.vo_read_pre_bias = nn.Parameter(torch.zeros(n, v))
    # Read gates (dynamic) and static-read gates.
    self.q_read_gate = GatedLinear(h, (n,))
    self.k_read_gate = GatedLinear(h, (n,))
    self.v_read_gate = GatedLinear(h, (n,))
    self.o_read_gate = GatedLinear(h, (n,))
    self.q_static_gate = GatedLinear(h, (n,))
    self.k_static_gate = GatedLinear(h, (n,))
    self.v_static_gate = GatedLinear(h, (n,))
    self.o_static_gate = GatedLinear(h, (n,))
    # Static read keys [V, N].
    self.q_static_key = nn.Parameter(torch.zeros(v, n))
    self.k_static_key = nn.Parameter(torch.zeros(v, n))
    self.v_static_key = nn.Parameter(torch.zeros(v, n))
    self.o_static_key = nn.Parameter(torch.zeros(v, n))
    # Attention write.
    self.write_address_down = nn.Linear(h, config.write_bottleneck_dim, bias=False)
    self.write_address_up = HeadLinear(config.write_bottleneck_dim, (n, v), bias=True)
    self.write_gate = GatedLinear(h, (n,))
    self.o_proj = nn.Linear(n * d, h, bias=False)

  def _static_read(self, M, key, gate):
    """M [B,T,K',V] . s [V,N] -> [B,T,N,K'], times 1.1 sigmoid(gate) / init."""
    read = torch.einsum('btkv,vn->btnk', M, key.to(M.dtype))
    return read * (1.1 * torch.sigmoid(gate) / self.static_gate_init)[..., None]

  def _qk_read(self, M, x, hidden, up, pre_bias, gate, static_key, static_gate):
    key = up(hidden) + pre_bias.to(hidden.dtype)
    r = self.read_scale * torch.sigmoid(gate(x))[..., None] * rms_norm(key, self.read_eps)
    col = torch.einsum('btkv,btnv->btnk', M, r)[..., :self.qk_cols]
    return col + self._static_read(M[:, :, :self.qk_cols], static_key, static_gate(x))

  def forward(self, x, M, positions, attention_mask=None, past_key_value: Optional[Cache] = None,
              defer_write=False):
    """x [B,T,H] (pre-normed), M [B,T,K,V]. Returns (out, M_out or write factors, cache)."""
    hidden = gelu(self.read_down(x))
    q = torch.cat((self._qk_read(M, x, hidden, self.q_read_up, self.q_read_pre_bias, self.q_read_gate,
                                 self.q_static_key, self.q_static_gate),
                   self.rotary(self.q_proj(x), positions)), dim=-1)
    k = torch.cat((self._qk_read(M, x, hidden, self.k_read_up, self.k_read_pre_bias, self.k_read_gate,
                                 self.k_static_key, self.k_static_gate),
                   self.rotary(self.k_proj(x), positions)), dim=-1)
    # One ungated VO column read, then destination-specific gates; plus static V/O reads.
    vo = torch.einsum('btkv,btnv->btnk', M,
                      rms_norm(self.vo_read_up(hidden) + self.vo_read_pre_bias.to(hidden.dtype), self.read_eps))
    v = vo * (self.read_scale * torch.sigmoid(self.v_read_gate(x)))[..., None]
    v = v + self._static_read(M, self.v_static_key, self.v_static_gate(x))
    local_o = vo * (self.read_scale * torch.sigmoid(self.o_read_gate(x)))[..., None]
    local_o = local_o + self._static_read(M, self.o_static_key, self.o_static_gate(x))
    q = q / torch.tensor(math.sqrt(self.head_dim), dtype=q.dtype)

    o_head = causal_attention(q, k, v, self.layer_idx, attention_mask, past_key_value) + local_o

    content = rms_norm(o_head[..., :self.bam_k], self.eps)
    address = rms_norm(self.write_address_up(gelu(self.write_address_down(x))), self.eps)
    gated = torch.sigmoid(self.write_gate(x))[..., None] * content
    out = self.o_proj(o_head.flatten(-2))
    if defer_write:
      return out, (gated, address)
    return out, M + torch.einsum('btnk,btnv->btkv', gated, address)


class BamMLP(nn.Module):

  def __init__(self, hidden_size, intermediate_size):
    super().__init__()
    self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)   # MaxText wi_0
    self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)     # wi_1
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)   # wo

  def forward(self, x):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BamMLPWrite(nn.Module):
  """Second write of the layer's MLP output into M with its own gate and address."""

  def __init__(self, config: BamConfig):
    super().__init__()
    n = config.num_attention_heads
    self.gate = GatedLinear(config.hidden_size, (n,))
    self.address_down = nn.Linear(config.hidden_size, config.mlp_write_address_dim, bias=False)
    self.address_up = HeadLinear(config.mlp_write_address_dim, (n, config.bam_v), bias=True)
    self.num_heads, self.bam_k, self.eps = n, config.bam_k, config.rms_norm_eps

  def forward(self, hidden, mlp_out, M, attention_factors):
    gated, address = attention_factors
    content = rms_norm(mlp_out.unflatten(-1, (self.num_heads, self.bam_k)), self.eps)
    mlp_content = torch.sigmoid(self.gate(hidden))[..., None] * content
    mlp_address = rms_norm(self.address_up(gelu(self.address_down(hidden))), self.eps)
    return M + (torch.einsum('btnk,btnv->btkv', gated, address)
                + torch.einsum('btnk,btnv->btkv', mlp_content, mlp_address))


class BamDecoderLayer(nn.Module):

  def __init__(self, config: BamConfig, layer_idx: int):
    super().__init__()
    self.input_layernorm = BamRMSNorm(config.hidden_size, config.rms_norm_eps)
    self.self_attn = (BamAttention if config.bam_enabled else MhaAttention)(config, layer_idx)
    self.post_attention_layernorm = BamRMSNorm(config.hidden_size, config.rms_norm_eps)
    self.mlp = BamMLP(config.hidden_size, config.intermediate_sizes[layer_idx])
    self.mlp_write = BamMLPWrite(config) if layer_idx in config.mlp_write_layers else None

  def forward(self, x, M, positions, attention_mask=None, past_key_value=None):
    attn, M_or_factors = self.self_attn(self.input_layernorm(x), M, positions, attention_mask, past_key_value,
                                        defer_write=self.mlp_write is not None)
    x = x + attn
    hidden = self.post_attention_layernorm(x)
    mlp_out = self.mlp(hidden)
    M = (M_or_factors if self.mlp_write is None
         else self.mlp_write(hidden, mlp_out, M, M_or_factors))
    return mlp_out + x, M


class BamPreTrainedModel(PreTrainedModel):
  config_class = BamConfig
  base_model_prefix = 'model'
  _no_split_modules = ['BamDecoderLayer']
  _supports_cache_class = True
  supports_gradient_checkpointing = False

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(0.0, self.config.initializer_range)
      if getattr(module, 'bias', None) is not None:
        module.bias.data.zero_()


class BamModel(BamPreTrainedModel):

  def __init__(self, config: BamConfig):
    super().__init__(config)
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.embedding_write = BamEmbeddingWrite(config) if config.bam_enabled else None
    self.layers = nn.ModuleList(BamDecoderLayer(config, i) for i in range(config.num_hidden_layers))
    self.norm = BamRMSNorm(config.hidden_size, config.rms_norm_eps)
    self.post_init()

  def get_input_embeddings(self):
    return self.embed_tokens

  def set_input_embeddings(self, value):
    self.embed_tokens = value

  def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
              inputs_embeds=None, use_cache=None, output_hidden_states=None, return_dict=None, **kwargs):
    use_cache = self.config.use_cache if use_cache is None else use_cache
    return_dict = self.config.use_return_dict if return_dict is None else return_dict
    x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    b, t = x.shape[:2]
    if use_cache and not isinstance(past_key_values, Cache):
      past_key_values = (DynamicCache() if past_key_values is None
                         else DynamicCache.from_legacy_cache(past_key_values))
    past = past_key_values.get_seq_length() if past_key_values is not None else 0
    if position_ids is None:
      if attention_mask is not None:      # left padding: positions count real tokens only
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -t:]
      else:
        position_ids = torch.arange(past, past + t, device=x.device)[None].expand(b, t)
    mask = None
    if attention_mask is not None or (past > 0 and t > 1):
      s = past + t
      keep = torch.ones(b, s, dtype=torch.bool, device=x.device) if attention_mask is None \
          else attention_mask[:, -s:].bool()
      causal = torch.arange(s, device=x.device)[None, :] <= torch.arange(past, s, device=x.device)[:, None]
      allowed = causal[None] & keep[:, None, :]
      allowed |= ~allowed.any(-1, keepdim=True)               # fully padded query rows: avoid NaN
      mask = allowed[:, None]                                  # [B,1,T,S] bool
    M = self.embedding_write(x) if self.embedding_write is not None else None
    hidden_states = (x,) if output_hidden_states else None
    for layer in self.layers:
      x, M = layer(x, M, position_ids, mask, past_key_values if use_cache else None)
      if output_hidden_states:
        hidden_states += (x,)
    x = self.norm(x)
    if not return_dict:
      return (x, past_key_values if use_cache else None, hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values if use_cache else None,
                                   hidden_states=hidden_states)


class BamForCausalLM(BamPreTrainedModel):
  _tied_weights_keys = []

  def __init__(self, config: BamConfig):
    super().__init__(config)
    self.model = BamModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.post_init()

  def get_input_embeddings(self):
    return self.model.embed_tokens

  def set_input_embeddings(self, value):
    self.model.embed_tokens = value

  def get_output_embeddings(self):
    return self.lm_head

  def set_output_embeddings(self, new_embeddings):
    self.lm_head = new_embeddings

  def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
              inputs_embeds=None, labels=None, use_cache=None, output_hidden_states=None, return_dict=None,
              **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
    return_dict = self.config.use_return_dict if return_dict is None else return_dict
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                         output_hidden_states=output_hidden_states, return_dict=True)
    logits = self.lm_head(outputs.last_hidden_state)
    loss = None
    if labels is not None:
      loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
    if not return_dict:
      out = (logits, outputs.past_key_values, outputs.hidden_states)
      return (loss,) + out if loss is not None else out
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                  hidden_states=outputs.hidden_states)

  def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                    inputs_embeds=None, **kwargs):
    past = 0
    if past_key_values is not None:
      past = (past_key_values.get_seq_length() if isinstance(past_key_values, Cache)
              else past_key_values[0][0].shape[2])
      if past > 0:
        input_ids = input_ids[:, past:]
    position_ids = None
    if attention_mask is not None:
      position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -input_ids.shape[1]:]
    if inputs_embeds is not None and past == 0:
      model_inputs = {'inputs_embeds': inputs_embeds}
    else:
      model_inputs = {'input_ids': input_ids}
    model_inputs.update(past_key_values=past_key_values, attention_mask=attention_mask,
                        position_ids=position_ids, use_cache=kwargs.get('use_cache', True))
    return model_inputs

  def _reorder_cache(self, past_key_values, beam_idx):
    if isinstance(past_key_values, Cache):
      past_key_values.reorder_cache(beam_idx)
      return past_key_values
    return tuple(tuple(s.index_select(0, beam_idx.to(s.device)) for s in layer) for layer in past_key_values)
