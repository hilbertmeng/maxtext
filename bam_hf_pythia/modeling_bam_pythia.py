"""PyTorch port of the Pythia-backbone unit-H BAM decoder (reference path).

Port of the executed pure-JAX path of MaxText `BamAttention` / `SubDecoderLayer` /
`EmbeddingBamWrite` (runtime 5a48faa, Pallas/Splash off) for
`BamPythia2p8BHUnit300BFastKernelMPreNormQKScale`; the FastKernel Pallas path computes the same
equations with the same parameters. Notation per token: M [K, V] (K = data, V = address), N heads,
rms(.) = parameter-free RMS norm over the last axis.

  embedding  M0 = sum_n sigmoid(g_n) (U_n e / tau) (x) rms(up(gelu(down e)))
  M-shift    M~ = (1 - s) M + s M_prev   per V column, s = sigmoid(W_s x)   (read view only)
  read view  Mh = M~ / rms over all K*V entries
  read key   r_a = up_a(gelu(down x)) + pre_bias_a                        (one shared `down`)
  Q/K        q = [c * (Mh 0.1 sigmoid(g_q) rms(r_q) + Mh s_q 1.1 sigmoid(h_q) / 0.99) | W_q x]
             QKNorm over the 100 coordinates, RoPE on the last 20; logits q.k / sqrt(100)
  V / O_loc  v = c_vo (0.1 sigmoid(g_v) Mh rms(r_v) + Mh s_v gate_v);  O_loc likewise from r_o
  attention  strictly causal (a token never attends to itself; the first token's context is 0)
             o = softmax(q k^T) v + O_loc,  out = W_O o + b
  writes     M += sigmoid(gw) rms(o) (x) rms(P(x)) + sigmoid(gm) mem_mlp(h) (x) rms(A(h))
             (all layers but the last; the read view never feeds the carried M)

Decoder: Pythia serial residual with LayerNorms, x += attn(LN1 x); x += mlp(LN2 x).
"""
import importlib
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
  from .configuration_bam_pythia import BamPythiaConfig
except ImportError:      # repo scripts import this file as a top-level module
  BamPythiaConfig = importlib.import_module('configuration_bam_pythia').BamPythiaConfig


def rms_norm(x, eps):
  """Parameter-free RMS norm over the last axis with FP32 statistics, returned in x.dtype."""
  f = x.float()
  return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def gelu(x):
  return F.gelu(x, approximate='tanh')     # flax nn.gelu default (bottlenecks)


def heads(x, n):
  return x.unflatten(-1, (n, -1))


class LayerNorm(nn.LayerNorm):
  """NeoX LayerNorm with FP32 statistics, returned in the input dtype."""

  def forward(self, x):
    return F.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(),
                        self.eps).to(x.dtype)


class ScaledRMSNorm(nn.Module):
  """RMS norm with a direct multiplier (MaxText stores multiplier - 1)."""

  def __init__(self, dim, eps):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x):
    return rms_norm(x, self.eps) * self.weight.to(x.dtype)


class Rotary(nn.Module):
  """MaxText RotaryEmbedding (half split) over `dim` coordinates."""

  def __init__(self, dim, min_timescale, max_timescale):
    super().__init__()      # no buffer: from_pretrained builds modules on the meta device
    self.dim, self.min_timescale, self.max_timescale = dim, min_timescale, max_timescale

  def table(self, positions, dtype):
    """(cos, sin) [B, T, 1, D/2] for integer positions [B, T]."""
    fraction = 2 * torch.arange(self.dim // 2, dtype=torch.float32, device=positions.device) / self.dim
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
    angle = positions[..., None, None].float() / timescale
    return angle.cos().to(dtype), angle.sin().to(dtype)

  @staticmethod
  def apply(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def strict_causal_attention(q, k, v, allowed):
  """q/k [B,T,N,Dq] (q pre-scaled), v [B,S,N,Dv], allowed [B or 1,T,S] bool.

  Rows without any allowed key return zero (MaxText's negative-diagonal attention)."""
  empty = ~allowed.any(-1)                                   # [B,T]
  mask = (allowed | empty[..., None])[:, None]               # avoid NaN rows
  y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                     attn_mask=mask, scale=1.0).transpose(1, 2)
  return y.masked_fill(empty[..., None, None], 0)


class BamEmbeddingWrite(nn.Module):
  """MaxText EmbeddingBamWrite: one learned gated write from the token embedding."""

  def __init__(self, config: BamPythiaConfig):
    super().__init__()
    d, n, k, v = config.hidden_size, config.num_attention_heads, config.bam_k, config.bam_v
    self.n, self.eps, self.tau = n, config.write_eps, config.embedding_write_content_divisor
    self.data = nn.Linear(d, n * k, bias=False)
    self.address_down = nn.Linear(d, config.embedding_write_bottleneck_dim, bias=False)
    self.address_up = nn.Linear(config.embedding_write_bottleneck_dim, n * v, bias=True)
    self.gate = nn.Linear(d, n, bias=True)

  def forward(self, e):
    data = heads((self.data(e).float() / self.tau).to(e.dtype), self.n)
    address = rms_norm(heads(self.address_up(gelu(self.address_down(e))), self.n), self.eps)
    gate = torch.sigmoid(self.gate(e))
    return torch.einsum('btnk,btnv->btkv', gate[..., None] * data, address)


class BamAttention(nn.Module):
  """MaxText BamAttention, mode local_qk+local_v+local_o with generalized full-M column reads."""

  def __init__(self, config: BamPythiaConfig, layer_idx: int):
    super().__init__()
    d, n = config.hidden_size, config.num_attention_heads
    k, v, s, r = config.bam_k, config.bam_v, config.standard_qk_dim, config.read_bottleneck_dim
    self.layer_idx, self.n, self.k, self.v, self.s = layer_idx, n, k, v, s
    self.eps, self.key_eps, self.key_scale = config.layer_norm_eps, config.read_key_eps, config.read_key_scale
    self.write_eps, self.static_init = config.write_eps, config.static_read_gate_init
    self.m_shift = nn.Linear(d, v, bias=True)
    self.query = nn.Linear(d, n * s, bias=True)
    self.key = nn.Linear(d, n * s, bias=True)
    self.read_down = nn.Linear(d, r, bias=False)
    for arm in 'qkvo':                       # key up-projections carry the generalized pre-biases
      setattr(self, f'{arm}_key_up', nn.Linear(r, n * v, bias=True))
      setattr(self, f'{arm}_gate', nn.Linear(d, n, bias=True))            # dynamic read gates
      setattr(self, f'{arm}_static_gate', nn.Linear(d, n, bias=True))     # static read gates
      setattr(self, f'{arm}_static_key', nn.Parameter(torch.zeros(v, n)))
    self.qk_scale = nn.Parameter(torch.ones(1))
    self.vo_scale = nn.Parameter(torch.ones(1))
    self.q_norm = ScaledRMSNorm(k + s, config.layer_norm_eps)
    self.k_norm = ScaledRMSNorm(k + s, config.layer_norm_eps)
    self.rotary = Rotary(s, config.rope_min_timescale, config.rope_max_timescale)
    self.out = nn.Linear(n * k, d, bias=True)
    self.has_write = layer_idx in config.attention_write_layers
    if self.has_write:
      self.write_down = nn.Linear(d, config.write_bottleneck_dim, bias=False)
      self.write_up = nn.Linear(config.write_bottleneck_dim, n * v, bias=True)
      self.write_gate = nn.Linear(d, n, bias=True)

  def read_view(self, x, M, M_prev, shift_valid):
    """Previous-token column blend (when valid), then the whole-matrix RMS normalization."""
    s = torch.sigmoid(self.m_shift(x).float()).to(M.dtype)[..., None, :]       # [B,T,1,V]
    mixed = (1 - s) * M + s * M_prev
    M = torch.where(shift_valid[..., None, None], mixed, M)
    f = M.float()
    return (f * torch.rsqrt(f.pow(2).mean((-2, -1), keepdim=True) + self.eps)).to(M.dtype)

  def _static(self, Mh, arm, x):
    read = torch.einsum('btkv,vn->btnk', Mh, getattr(self, f'{arm}_static_key').to(Mh.dtype))
    gate = 1.1 * torch.sigmoid(getattr(self, f'{arm}_static_gate')(x))
    return read * (gate / self.static_init)[..., None]

  def _gate(self, arm, x):
    return (self.key_scale * torch.sigmoid(getattr(self, f'{arm}_gate')(x)))[..., None]

  def _dynamic(self, Mh, arm, hidden, key_gate=None):
    """Q/K gate the normalized key before the read; V/O gate the read output (as in MaxText)."""
    r = rms_norm(heads(getattr(self, f'{arm}_key_up')(hidden), self.n), self.key_eps)
    if key_gate is not None:
      r = key_gate * r
    return torch.einsum('btkv,btnv->btnk', Mh, r)

  def forward(self, x, M, M_prev, shift_valid, rope, allowed, past_key_value: Optional[Cache] = None):
    """x [B,T,D] (normalized), M [B,T,K,V] raw layer input. Returns (out, write factors or None)."""
    Mh = self.read_view(x, M, M_prev, shift_valid)
    hidden = gelu(self.read_down(x))
    q_std, k_std = heads(self.query(x), self.n), heads(self.key(x), self.n)
    local = []
    for arm in 'qk':
      col = self._dynamic(Mh, arm, hidden, self._gate(arm, x)) + self._static(Mh, arm, x)
      local.append((col.float() * self.qk_scale).to(col.dtype))
    q = self.q_norm(torch.cat((local[0], q_std), -1))
    kk = self.k_norm(torch.cat((local[1], k_std), -1))
    cos, sin = rope
    q = torch.cat((q[..., :self.k], Rotary.apply(q[..., self.k:], cos, sin)), -1)
    kk = torch.cat((kk[..., :self.k], Rotary.apply(kk[..., self.k:], cos, sin)), -1)
    value = self._dynamic(Mh, 'v', hidden) * self._gate('v', x) + self._static(Mh, 'v', x)
    local_o = self._dynamic(Mh, 'o', hidden) * self._gate('o', x) + self._static(Mh, 'o', x)
    value = (value.float() * self.vo_scale).to(value.dtype)
    local_o = (local_o.float() * self.vo_scale).to(local_o.dtype)
    q = q / torch.tensor(math.sqrt(self.k + self.s), dtype=q.dtype)
    if past_key_value is not None:
      kk, value_all = (z.transpose(1, 2) for z in past_key_value.update(
          kk.transpose(1, 2), value.transpose(1, 2), self.layer_idx))
    else:
      value_all = value
    o_head = strict_causal_attention(q, kk, value_all, allowed) + local_o
    out = self.out(o_head.flatten(-2))
    if not self.has_write:
      return out, None
    gate = torch.sigmoid(self.write_gate(x))
    content = gate[..., None] * rms_norm(o_head, self.write_eps)
    address = rms_norm(heads(self.write_up(gelu(self.write_down(x))), self.n), self.write_eps)
    return out, (content, address)


class BamMLP(nn.Module):

  def __init__(self, hidden_size, intermediate_size, bias=True):
    super().__init__()
    self.fc_in = nn.Linear(hidden_size, intermediate_size, bias=bias)
    self.fc_out = nn.Linear(intermediate_size, hidden_size, bias=bias)

  def forward(self, x):
    return self.fc_out(F.gelu(self.fc_in(x)))         # gelu_exact


class BamDecoderLayer(nn.Module):

  def __init__(self, config: BamPythiaConfig, layer_idx: int):
    super().__init__()
    d, n = config.hidden_size, config.num_attention_heads
    self.n, self.write_eps = n, config.write_eps
    self.input_layernorm = LayerNorm(d, eps=config.layer_norm_eps)
    self.post_attention_layernorm = LayerNorm(d, eps=config.layer_norm_eps)
    self.self_attn = BamAttention(config, layer_idx)
    self.mlp = BamMLP(d, config.intermediate_size_of(layer_idx))
    self.has_mlp_write = layer_idx in config.mlp_write_layers
    if self.has_mlp_write:
      assert self.self_attn.has_write, 'MLP-only writes are not part of this model'
      self.memory_mlp = BamMLP(d, config.memory_mlp_dim, bias=False)
      self.mlp_write_gate = nn.Linear(d, n, bias=True)
      self.mlp_address_down = nn.Linear(d, config.mlp_write_address_dim, bias=False)
      self.mlp_address_up = nn.Linear(config.mlp_write_address_dim, n * config.bam_v, bias=True)
      self.tau = config.mlp_write_content_divisor

  def forward(self, x, M, M_prev, shift_valid, rope, allowed, past_key_value=None):
    attn, factors = self.self_attn(self.input_layernorm(x), M, M_prev, shift_valid, rope, allowed,
                                   past_key_value)
    x = x + attn
    h = self.post_attention_layernorm(x)
    if factors is not None:
      content, address = factors
      dM = torch.einsum('btnk,btnv->btkv', content, address)
      if self.has_mlp_write:
        memory = heads((self.memory_mlp(h).float() / self.tau).to(h.dtype), self.n)
        mlp_content = torch.sigmoid(self.mlp_write_gate(h))[..., None] * memory
        mlp_address = rms_norm(heads(self.mlp_address_up(gelu(self.mlp_address_down(h))), self.n),
                               self.write_eps)
        dM = dM + torch.einsum('btnk,btnv->btkv', mlp_content, mlp_address)
      M = M + dM
    return x + self.mlp(h), M


class BamPythiaPreTrainedModel(PreTrainedModel):
  config_class = BamPythiaConfig
  base_model_prefix = 'model'
  _no_split_modules = ['BamDecoderLayer']
  _supports_cache_class = True
  supports_gradient_checkpointing = False

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(0.0, self.config.initializer_range)
      if getattr(module, 'bias', None) is not None:
        module.bias.data.zero_()


class BamPythiaModel(BamPythiaPreTrainedModel):
  """Decoder stack. The cache holds attention K/V plus, per layer, the last token's raw input M."""

  def __init__(self, config: BamPythiaConfig):
    super().__init__(config)
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.embedding_write = BamEmbeddingWrite(config)
    self.layers = nn.ModuleList(BamDecoderLayer(config, i) for i in range(config.num_hidden_layers))
    self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.post_init()

  def get_input_embeddings(self):
    return self.embed_tokens

  def set_input_embeddings(self, value):
    self.embed_tokens = value

  def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
              inputs_embeds=None, use_cache=None, output_hidden_states=None, return_dict=None, **kwargs):
    use_cache = self.config.use_cache if use_cache is None else use_cache
    x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    b, t = x.shape[:2]
    if use_cache and past_key_values is None:
      past_key_values = DynamicCache()
    cache = past_key_values if use_cache else None
    past = cache.get_seq_length() if cache is not None else 0
    if position_ids is None:
      if attention_mask is not None:      # left padding: positions count real tokens only
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -t:]
      else:
        position_ids = torch.arange(past, past + t, device=x.device)[None].expand(b, t)
    s = past + t
    allowed = (torch.arange(s, device=x.device)[None, :]
               < torch.arange(past, s, device=x.device)[:, None])[None]          # strict causal [1,T,S]
    if attention_mask is not None:
      allowed = allowed & attention_mask[:, None, -s:].bool()
    # M-shift validity: the previous token exists in this sequence and is at position - 1.
    prev_m = getattr(cache, 'bam_m', None) if past else None
    prev_pos = (cache.bam_pos[:, None] if prev_m is not None else position_ids[:, :1])
    prev_pos = torch.cat((prev_pos, position_ids[:, :-1]), 1)
    shift_valid = position_ids == prev_pos + 1
    if prev_m is None:
      shift_valid[:, 0] = False
    rope = self.layers[0].self_attn.rotary.table(position_ids, x.dtype)
    M = self.embedding_write(x)
    hidden_states = (x,) if output_hidden_states else None
    last_m = []
    for i, layer in enumerate(self.layers):
      first = prev_m[i][:, None] if prev_m is not None else M[:, :1]
      M_prev = torch.cat((first, M[:, :-1]), 1)
      last_m.append(M[:, -1])
      x, M = layer(x, M, M_prev, shift_valid, rope, allowed, cache)
      if output_hidden_states:
        hidden_states += (x,)
    if cache is not None:
      cache.bam_m, cache.bam_pos = last_m, position_ids[:, -1]
    x = self.norm(x)
    return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=cache, hidden_states=hidden_states)


class BamPythiaForCausalLM(BamPythiaPreTrainedModel, GenerationMixin):
  _tied_weights_keys = []

  def __init__(self, config: BamPythiaConfig):
    super().__init__(config)
    self.model = BamPythiaModel(config)
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

  def logits(self, hidden):
    """FP32 logits from model-dtype operands (MaxText bf16_dot_fp32_output)."""
    w = self.lm_head.weight
    if hidden.dtype == torch.float32 or not hidden.is_cuda:
      return F.linear(hidden.float(), w.float())
    flat = hidden.reshape(-1, hidden.shape[-1])
    return torch.mm(flat, w.t(), out_dtype=torch.float32).view(*hidden.shape[:-1], -1)

  def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
              inputs_embeds=None, labels=None, use_cache=None, output_hidden_states=None, return_dict=None,
              logits_to_keep=0, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                         output_hidden_states=output_hidden_states)
    hidden = outputs.last_hidden_state
    logits = self.logits(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)
    loss = None
    if labels is not None:
      loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100)
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                  hidden_states=outputs.hidden_states)

  def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                    inputs_embeds=None, cache_position=None, **kwargs):
    past = past_key_values.get_seq_length() if past_key_values is not None else 0
    if past > 0:
      input_ids = input_ids[:, past:]
    position_ids = None
    if attention_mask is not None:
      position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -input_ids.shape[1]:]
    model_inputs = ({'inputs_embeds': inputs_embeds} if inputs_embeds is not None and past == 0
                    else {'input_ids': input_ids})
    model_inputs.update(past_key_values=past_key_values, attention_mask=attention_mask,
                        position_ids=position_ids, use_cache=kwargs.get('use_cache', True))
    return model_inputs

  def _reorder_cache(self, past_key_values, beam_idx):
    past_key_values.reorder_cache(beam_idx)
    if getattr(past_key_values, 'bam_m', None) is not None:
      past_key_values.bam_m = [m.index_select(0, beam_idx.to(m.device)) for m in past_key_values.bam_m]
      past_key_values.bam_pos = past_key_values.bam_pos.index_select(0, beam_idx.to(past_key_values.bam_pos.device))
    return past_key_values
