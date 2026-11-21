"""Self-contained, single-device JAX reference & observer forward passes for §5.3 Mechanistic Analysis.

Covers all three 300B-token 2.8B models on matched Pile validation sequences & ICL prompts:
  1. LoomPythia-2.8B  (`BamPythia2p8BHUnit300BFastKernelMPreNormQKScale`, step 143000)
  2. mHC-Pythia-2.8B  (`Pythia2p8BSerialQKNormMHC300B`, step 143000)
  3. Pythia-2.8B      (`EleutherAI/pythia-2.8b`, step 143000)

Every model implementation is verified against its canonical MaxText / HuggingFace reference
before TPU execution.
"""
from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Shared primitives (matching MaxText / NeoX FP32 statistics, BF16 activations)
# ---------------------------------------------------------------------------

def rms_norm(x: jax.Array, eps: float) -> jax.Array:
  f = x.astype(jnp.float32)
  r = jax.lax.rsqrt(jnp.mean(jnp.square(f), axis=-1, keepdims=True) + eps)
  return (f * r).astype(x.dtype)


def scaled_rms_norm(x: jax.Array, scale: jax.Array, eps: float) -> jax.Array:
  """MaxText QKNorm stores (multiplier - 1) in `scale`; caller passes `1 + scale` or raw weight."""
  return rms_norm(x, eps) * scale.astype(x.dtype)


def layer_norm(x: jax.Array, scale: jax.Array, bias: jax.Array, eps: float = 1e-5) -> jax.Array:
  f = x.astype(jnp.float32)
  mean = jnp.mean(f, axis=-1, keepdims=True)
  var = jnp.mean(jnp.square(f - mean), axis=-1, keepdims=True)
  normed = (f - mean) * jax.lax.rsqrt(var + eps)
  out = normed * scale.astype(jnp.float32) + bias.astype(jnp.float32)
  return out.astype(x.dtype)


def gelu_tanh(x: jax.Array) -> jax.Array:
  return jax.nn.gelu(x, approximate=True)


def gelu_exact(x: jax.Array) -> jax.Array:
  return jax.nn.gelu(x, approximate=False)


def rope_table(positions: jax.Array, rotary_dim: int, dtype: jnp.dtype,
               min_timescale: float = 1.0, max_timescale: float = 10000.0) -> tuple[jax.Array, jax.Array]:
  """positions: [B, T] -> (cos, sin) each [B, T, 1, rotary_dim // 2]."""
  half = rotary_dim // 2
  fraction = 2.0 * jnp.arange(half, dtype=jnp.float32) / float(rotary_dim)
  timescale = min_timescale * (max_timescale / min_timescale) ** fraction
  angle = positions[..., None, None].astype(jnp.float32) / timescale
  return jnp.cos(angle).astype(dtype), jnp.sin(angle).astype(dtype)


def apply_rope_half(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
  half = x.shape[-1] // 2
  a, b = x[..., :half], x[..., half:]
  return jnp.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1)


def bf16_logits_and_xent(
    hidden: jax.Array,
    logits_dense_dv: jax.Array,
    targets: jax.Array,
    logit_pos: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """hidden: [B, T, D], logits_dense_dv: [D, V] -> (xent [B, T], logits_sel [B, V])."""
  logits = jnp.matmul(
      hidden.astype(jnp.bfloat16),
      logits_dense_dv.astype(jnp.bfloat16),
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  xent = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
  sel = logits[:, -1, :] if logit_pos is None else logits[jnp.arange(logits.shape[0]), logit_pos]
  return xent, sel


# ---------------------------------------------------------------------------
# Shared Attention Head Activation & Sink Statistics (on sampled query positions)
# ---------------------------------------------------------------------------

HEAD_STAT_NAMES = (
    "active_paper",          # argmax not in {0, 1} and token not in paper_sink_ids (<bos>/., \n)
    "active_paper_strict",   # argmax not in {0, 1} and token not in {0, 15, 187}
    "active_lexical",        # argmax is lexical content (cat == 0 and pos >= 2)
    "active_paper_noself",   # active_paper evaluated with self-diagonal masked out (fair vs Loom)
    "active_lexical_noself", # active_lexical evaluated with self-diagonal masked out
    "self_argmax",           # argmax == query_pos (0 for Loom since strictly causal)
    "first2_argmax",         # argmax in {0, 1}
    "mass_first2",           # sum_{s < 2} alpha_s
    "mass_paper_sink",       # sum_{s in {0,1} or paper_sink_ids} alpha_s
    "mass_lexical_sink",     # sum_{s == 0 or cat != 0} alpha_s
    "mass_content",          # sum_{s >= 2 and cat == 0} alpha_s
    "mass_self",             # alpha_{s = q}
    "entropy_norm",          # H(alpha) / log(valid_count)
    "peak_prob",             # max_s alpha_s
)


def compute_head_stats(
    query_prescaled: jax.Array,
    key: jax.Array,
    qpos: jax.Array,
    categories: jax.Array,
    paper_sink_mask: jax.Array,
    strict_sink_mask: jax.Array,
    strictly_causal: bool,
) -> jax.Array:
  """Compute [B, H, Q, len(HEAD_STAT_NAMES)] FP32 statistics at sampled query indices `qpos`.

  Args:
    query_prescaled: [B, T, H, D_qk], already divided by sqrt(D_qk).
    key: [B, T, H, D_qk].
    qpos: [Q] int32 query positions (all >= 2).
    categories: [B, T] int32 lexical categories (0=content, 1=punct, 2=space, 3=special, 4=symbol).
    paper_sink_mask: [B, T] bool (True if token is <eos>=0, '.', ' .', '\\n', '\\n\\n').
    strict_sink_mask: [B, T] bool (True if token is in {0, 15, 187}).
    strictly_causal: True for LoomPythia (s < q), False for Pythia / mHC-Pythia (s <= q).
  """
  t_len = key.shape[1]
  q_sel = query_prescaled[:, qpos].astype(jnp.float32)  # [B, Q, H, D]
  k_all = key.astype(jnp.float32)                      # [B, T, H, D]
  logits = jnp.einsum(
      "bqhd,bshd->bhqs", q_sel, k_all, precision=jax.lax.Precision.HIGHEST
  )
  src = jnp.arange(t_len, dtype=jnp.int32)
  if strictly_causal:
    valid = (src[None, :] < qpos[:, None])[None, None, :, :]
  else:
    valid = (src[None, :] <= qpos[:, None])[None, None, :, :]

  alpha = jax.nn.softmax(jnp.where(valid, logits, -1.0e30), axis=-1)  # [B, H, Q, T]

  # Also compute self-excluded softmax (s < q) so Pythia/mHC can be compared fairly without diagonal
  valid_noself = (src[None, :] < qpos[:, None])[None, None, :, :]
  alpha_noself = jax.nn.softmax(jnp.where(valid_noself, logits, -1.0e30), axis=-1)

  first2 = (src < 2)[None, None, None, :]                  # [1, 1, 1, T]
  first1 = (src == 0)[None, None, None, :]
  is_self = (src[None, :] == qpos[:, None])[None, None, :, :]

  p_sink = first2 | paper_sink_mask[:, None, None, :]      # [B, 1, 1, T]
  s_sink = first2 | strict_sink_mask[:, None, None, :]     # [B, 1, 1, T]
  lex_content = (~first2) & (categories[:, None, None, :] == 0)
  lex_sink = first1 | (categories[:, None, None, :] != 0)

  peak_idx = jnp.argmax(alpha, axis=-1)                    # [B, H, Q]
  peak_idx_ns = jnp.argmax(alpha_noself, axis=-1)          # [B, H, Q]

  def gather_mask(mask_b11t: jax.Array, idx_bhq: jax.Array) -> jax.Array:
    bmask = jnp.broadcast_to(mask_b11t, alpha.shape)
    return jnp.take_along_axis(bmask, idx_bhq[..., None], axis=-1)[..., 0]

  active_paper = (~gather_mask(p_sink, peak_idx)).astype(jnp.float32)
  active_strict = (~gather_mask(s_sink, peak_idx)).astype(jnp.float32)
  active_lex = gather_mask(lex_content, peak_idx).astype(jnp.float32)
  active_paper_ns = (~gather_mask(p_sink, peak_idx_ns)).astype(jnp.float32)
  active_lex_ns = gather_mask(lex_content, peak_idx_ns).astype(jnp.float32)

  self_argmax = (peak_idx == qpos[None, None, :]).astype(jnp.float32)
  first2_argmax = (peak_idx < 2).astype(jnp.float32)

  mass_first2 = jnp.sum(alpha * first2, axis=-1)
  mass_paper_sink = jnp.sum(alpha * p_sink, axis=-1)
  mass_lex_sink = jnp.sum(alpha * lex_sink, axis=-1)
  mass_content = jnp.sum(alpha * lex_content, axis=-1)
  mass_self = jnp.sum(alpha * is_self, axis=-1)

  counts = jnp.sum(valid.astype(jnp.float32), axis=-1)
  ent = -jnp.sum(jnp.where(alpha > 0, alpha * jnp.log(jnp.maximum(alpha, 1.0e-30)), 0.0), axis=-1)
  ent_norm = ent / jnp.maximum(jnp.log(jnp.maximum(counts, 2.0)), 1.0e-12)
  peak_prob = jnp.max(alpha, axis=-1)

  out = [
      active_paper,
      active_strict,
      active_lex,
      active_paper_ns,
      active_lex_ns,
      self_argmax,
      first2_argmax,
      mass_first2,
      mass_paper_sink,
      mass_lex_sink,
      mass_content,
      mass_self,
      ent_norm,
      peak_prob,
  ]
  out = [jnp.broadcast_to(v, active_paper.shape) for v in out]
  return jnp.stack(out, axis=-1)


def token_cos_mean(a: jax.Array, b: jax.Array, eps: float = 1.0e-12) -> jax.Array:
  """Mean per-token cosine similarity over [B, T, ...], flattening trailing dims."""
  af = a.astype(jnp.float32).reshape(a.shape[0], a.shape[1], -1)
  bf = b.astype(jnp.float32).reshape(b.shape[0], b.shape[1], -1)
  dot = jnp.sum(af * bf, axis=-1)
  na = jnp.linalg.norm(af, axis=-1)
  nb = jnp.linalg.norm(bf, axis=-1)
  return jnp.mean(dot / jnp.maximum(na * nb, eps))


# ---------------------------------------------------------------------------
# 1. LoomPythia-2.8B (`BamPythia2p8BHUnit300BFastKernelMPreNormQKScale`)
# ---------------------------------------------------------------------------

def unscan_loom_params(params_tree: dict[str, Any]) -> dict[str, Any]:
  """Convert MaxText Orbax `params['params']` tree into explicit per-layer dict for clean JIT."""
  p = params_tree["params"] if "params" in params_tree else params_tree
  dec = p["decoder"]
  scanned = dec["layers"]["block"]
  final_blk = dec["final_local_layer"]["block"]
  n_scan = scanned["mlp"]["wi"]["kernel"].shape[1]
  layers = []
  for i in range(n_scan):
    layers.append(jax.tree.map(lambda x, idx=i: x[:, idx] if x.ndim >= 2 and x.shape[1] == n_scan else jnp.take(x, idx, axis=1), scanned))
  layers.append(final_blk)
  return {
      "embedding": p["token_embedder"]["embedding"],
      "emb_write": dec["embedding_bam_write"],
      "layers": layers,
      "final_norm": dec["lm_head"]["decoder_norm"],
      "logits_dense": dec["lm_head"]["logits_dense"],
  }


def _slice_scanned_layer(scanned_tree: dict[str, Any], idx: int) -> dict[str, Any]:
  """Slice param_scan_axis=1 across every leaf in `decoder/layers/block`."""
  return jax.tree.map(lambda x: jnp.take(x, idx, axis=1), scanned_tree)


def prepare_loom_params(params_tree: dict[str, Any], dtype: jnp.dtype = jnp.bfloat16) -> dict[str, Any]:
  """Extract and cast weights to execution dtype while preserving FP32 norms/scales where MaxText does."""
  p = params_tree["params"] if "params" in params_tree else params_tree
  dec = p["decoder"]
  scanned = dec["layers"]["block"]
  n_scan = scanned["mlp"]["wi"]["kernel"].shape[1]
  layers = [_slice_scanned_layer(scanned, i) for i in range(n_scan)]
  layers.append(dec["final_local_layer"]["block"])
  return {
      "embedding": p["token_embedder"]["embedding"].astype(dtype),
      "emb_write": dec["embedding_bam_write"],
      "layers": layers,
      "final_norm": dec["lm_head"]["decoder_norm"],
      "logits_dense": dec["lm_head"]["logits_dense"].astype(dtype),
  }


def loom_forward_and_observe(
    u_params: dict[str, Any],
    inputs: jax.Array,
    targets: jax.Array,
    sample_pos: jax.Array,
    head_qpos: jax.Array,
    categories: jax.Array,
    paper_sink_mask: jax.Array,
    strict_sink_mask: jax.Array,
    *,
    emb_tau: float = 0.0020825863,
    mlp_tau: float = 1.0,
    ln_eps: float = 1.0e-5,
    read_key_eps: float = 1.0e-4,
    read_key_scale: float = 0.1,
    write_eps: float = 1.0e-6,
    static_gate_init: float = 0.99,
    ablation: str = "none",
    logit_pos: jax.Array | None = None,
) -> dict[str, jax.Array]:
  """Exact forward pass of BamPythia2p8BHUnit300BFastKernelMPreNormQKScale + compact sliced probes.

  Supported `ablation` modes (for whole-network causal necessity checks):
    - "none": exact unmodified forward pass
    - "no_dynamic_read": zero dynamic Q/K/V/O read branches (keep static reads)
    - "no_static_read": zero static Q/K/V/O read branches (keep dynamic reads)
    - "no_m_shift": set MShift gate z = 0 (disable cross-token shift)
    - "no_local_o": zero local_o skip readout
    - "no_emb_write": zero embedding M write
    - "no_attn_write": zero attention M writes
    - "no_mlp_write": zero memory-FFN M writes
  """
  dtype = jnp.bfloat16
  b_sz, t_len = inputs.shape
  positions = jnp.broadcast_to(jnp.arange(t_len, dtype=jnp.int32), (b_sz, t_len))
  prev_pos = jnp.maximum(sample_pos - 1, 0)

  x = u_params["embedding"][inputs].astype(dtype)  # [B, T, D]
  num_layers = len(u_params["layers"])

  # 1. Embedding write -> initial M [B, T, K, V]
  ew = u_params["emb_write"]
  emb_u = jnp.einsum("btd,dnk->btnk", x, ew["W_emb_u"]["kernel"].astype(dtype))
  emb_data = (emb_u.astype(jnp.float32) / emb_tau).astype(dtype)
  emb_h = gelu_tanh(jnp.einsum("btd,dr->btr", x, ew["W_emb_v_down"]["kernel"].astype(dtype)))
  emb_addr_raw = (
      jnp.einsum("btr,rnv->btnv", emb_h, ew["W_emb_v_up"]["kernel"].astype(dtype))
      + ew["W_emb_v_up"]["bias"].astype(dtype)
  )
  emb_addr = rms_norm(emb_addr_raw, write_eps)
  emb_gate = jax.nn.sigmoid(
      jnp.einsum("btd,dn->btn", x, ew["W_emb_g"]["kernel"].astype(dtype))
      + ew["emb_gw_b0"].astype(dtype)
  )
  if ablation == "no_emb_write":
    emb_gate = jnp.zeros_like(emb_gate)
  emb_content = (emb_gate[..., None] * emb_data).astype(dtype)  # [B, T, N, K]
  M = jnp.sum(emb_content[..., :, None] * emb_addr[..., None, :], axis=-3)  # [B, T, K, V]

  n_heads, k_dim = emb_content.shape[2], emb_content.shape[3]
  s_dim = u_params["layers"][0]["self_attention"]["query"]["bias"].shape[-1]
  cos, sin = rope_table(positions, s_dim, dtype)

  # Strict causal mask for LoomPythia (negative_diagonal = True: s < t)
  src_idx = jnp.arange(t_len, dtype=jnp.int32)[None, :]
  tgt_idx = jnp.arange(t_len, dtype=jnp.int32)[:, None]
  strict_allowed = src_idx < tgt_idx  # [T, T]; row 0 is all False!
  row_empty = ~jnp.any(strict_allowed, axis=-1, keepdims=True)  # [T, 1]
  safe_mask = strict_allowed | row_empty  # avoid NaN at t=0

  # Sliced writer lists at `sample_pos` and `prev_pos` (B=1 or small B)
  writers_o_cur = [emb_content[:, sample_pos].astype(jnp.float32)]
  writers_k_cur = [emb_addr[:, sample_pos].astype(jnp.float32)]
  writers_o_prev = [emb_content[:, prev_pos].astype(jnp.float32)]
  writers_k_prev = [emb_addr[:, prev_pos].astype(jnp.float32)]

  # Per-layer reader & representation collectors
  layer_z = []
  layer_m_rms = []
  layer_q_dyn = []
  layer_q_stat = []
  layer_u_read = []
  layer_head_stats = []

  prev_res_in = None
  prev_ln_in = None
  prev_q_local = None
  prev_k_local = None
  prev_v_local = None
  prev_o_local = None
  adj_cos_res = []
  adj_cos_ln = []
  adj_cos_q = []
  adj_cos_k = []
  adj_cos_v = []
  adj_cos_o = []

  for l_idx, lyr in enumerate(u_params["layers"]):
    sa = lyr["self_attention"]
    res_in = x
    lnx = layer_norm(
        x,
        lyr["pre_self_attention_layer_norm"]["scale"],
        lyr["pre_self_attention_layer_norm"]["bias"],
        ln_eps,
    )

    # Read-only MShift + FP32 whole-matrix RMS pre-norm
    m_shift_logits = (
        jnp.einsum("btd,dv->btv", lnx, sa["M_shift_gate"]["kernel"].astype(dtype)).astype(jnp.float32)
        + sa["M_shift_gate_b0"].astype(jnp.float32)
    )
    z_gate = jax.nn.sigmoid(m_shift_logits)  # [B, T, V] in FP32
    if ablation == "no_m_shift":
      z_gate = jnp.zeros_like(z_gate)
    z_bf16 = z_gate.astype(dtype)
    M_prev = jnp.concatenate((M[:, :1], M[:, :-1]), axis=1)
    shift_valid = (positions > 0)[..., None, None]
    M_mixed = ((1.0 - z_bf16)[..., None, :] * M + z_bf16[..., None, :] * M_prev).astype(dtype)
    M_read = jnp.where(shift_valid, M_mixed, M)

    M_read_f32 = M_read.astype(jnp.float32)
    inv_m_rms = jax.lax.rsqrt(jnp.mean(jnp.square(M_read_f32), axis=(-2, -1), keepdims=True) + ln_eps)
    Mh = (M_read_f32 * inv_m_rms).astype(dtype)  # [B, T, K, V]

    # Shared read bottleneck
    shared_h = gelu_tanh(jnp.einsum("btd,dr->btr", lnx, sa["full_m_read_down"]["kernel"].astype(dtype)))

    qk_scale = sa["common_qk_read_scale"][0].astype(jnp.float32)
    vo_scale = sa["common_vo_read_scale"][0].astype(jnp.float32)
    if ablation == "no_dynamic_read":
      dyn_mult = 0.0
    else:
      dyn_mult = 1.0
    if ablation == "no_static_read":
      stat_mult = 0.0
    else:
      stat_mult = 1.0

    # Dynamic & static Q / K / V / O reads (preserving exact MaxText BF16 ordering)
    def dyn_key_normed(up_name: str, bias_name: str, squeeze_f: bool = False) -> jax.Array:
      k_up = sa[up_name]["kernel"].astype(dtype)
      if squeeze_f:
        raw = jnp.einsum("btr,rnfv->btnfv", shared_h, k_up) + sa[bias_name].astype(dtype)
        raw = jnp.squeeze(raw, axis=-2)
      else:
        raw = jnp.einsum("btr,rnv->btnv", shared_h, k_up) + sa[bias_name].astype(dtype)
      return rms_norm(raw, read_key_eps)

    def read_gate_sig(g_name: str, squeeze_f: bool = False) -> jax.Array:
      k_g = sa[g_name]["kernel"].astype(dtype)
      b_g = sa[g_name + "_b0"].astype(dtype)
      if squeeze_f:
        lg = jnp.squeeze(jnp.einsum("btd,dnfc->btnfc", lnx, k_g) + b_g, axis=-2)[..., 0]
      else:
        lg = (jnp.einsum("btd,dnc->btnc", lnx, k_g) + b_g)[..., 0]
      return jax.nn.sigmoid(lg)  # [B, T, N] in bf16

    def static_branch(arm: str) -> tuple[jax.Array, jax.Array]:
      sk = sa[f"static_{arm}_key"].astype(dtype)  # [V, N]
      sg_logits = (
          jnp.einsum("btd,dn->btn", lnx, sa[f"general_{arm}_static_gate"]["kernel"].astype(dtype))
          + sa[f"general_{arm}_static_gate_bias"].astype(dtype)
      )
      sg = (1.1 * jax.nn.sigmoid(sg_logits.astype(jnp.float32))).astype(dtype)  # [B, T, N]
      mult = (sg / static_gate_init) * stat_mult  # [B, T, N]
      u_st = jnp.einsum("btkv,vn->btnk", Mh, sk) * mult[..., None]
      q_st_eff = sk.T[None, None, :, :].astype(jnp.float32) * mult[..., None].astype(jnp.float32)  # [B, T, N, V]
      return u_st, q_st_eff

    # Q and K: gate scales normalized key before contraction, then common_qk_scale
    rk_q = dyn_key_normed("W_lq_c8_up", "general_q_pre_bias")
    g_q = (read_key_scale * read_gate_sig("W_lq_gate") * dyn_mult).astype(dtype)
    u_q_dyn = jnp.einsum("btkv,btnv->btnk", Mh, g_q[..., None] * rk_q)
    u_q_stat, q_q_stat = static_branch("q")
    q_local = ((u_q_dyn + u_q_stat).astype(jnp.float32) * qk_scale).astype(dtype)
    q_q_dyn_eff = (g_q[..., None].astype(jnp.float32) * rk_q.astype(jnp.float32)) * qk_scale
    q_q_stat_eff = q_q_stat * qk_scale

    rk_k = dyn_key_normed("W_lk_c8_up", "general_k_pre_bias")
    g_k = (read_key_scale * read_gate_sig("W_lk_gate") * dyn_mult).astype(dtype)
    u_k_dyn = jnp.einsum("btkv,btnv->btnk", Mh, g_k[..., None] * rk_k)
    u_k_stat, q_k_stat = static_branch("k")
    k_local = ((u_k_dyn + u_k_stat).astype(jnp.float32) * qk_scale).astype(dtype)
    q_k_dyn_eff = (g_k[..., None].astype(jnp.float32) * rk_k.astype(jnp.float32)) * qk_scale
    q_k_stat_eff = q_k_stat * qk_scale

    # V and O: contraction first, then gate, then common_vo_scale
    rk_v = dyn_key_normed("W_lv_c8_up", "general_v_pre_bias")
    g_v = (read_key_scale * read_gate_sig("W_lv_gate") * dyn_mult).astype(dtype)
    u_v_dyn = jnp.einsum("btkv,btnv->btnk", Mh, rk_v) * g_v[..., None]
    u_v_stat, q_v_stat = static_branch("v")
    v_local = ((u_v_dyn + u_v_stat).astype(jnp.float32) * vo_scale).astype(dtype)
    q_v_dyn_eff = (g_v[..., None].astype(jnp.float32) * rk_v.astype(jnp.float32)) * vo_scale
    q_v_stat_eff = q_v_stat * vo_scale

    rk_o = dyn_key_normed("W_R_up", "general_vo_pre_bias", squeeze_f=True)
    g_o = (read_key_scale * read_gate_sig("W_R_gate", squeeze_f=True) * dyn_mult).astype(dtype)
    u_o_dyn = jnp.einsum("btkv,btnv->btnk", Mh, rk_o) * g_o[..., None]
    u_o_stat, q_o_stat = static_branch("o")
    o_local = ((u_o_dyn + u_o_stat).astype(jnp.float32) * vo_scale).astype(dtype)
    if ablation == "no_local_o":
      o_local = jnp.zeros_like(o_local)
      q_o_dyn_eff = jnp.zeros_like(q_v_dyn_eff)
      q_o_stat_eff = jnp.zeros_like(q_v_stat_eff)
    else:
      q_o_dyn_eff = (g_o[..., None].astype(jnp.float32) * rk_o.astype(jnp.float32)) * vo_scale
      q_o_stat_eff = q_o_stat * vo_scale

    # Adjacent-layer stream cosines
    if l_idx > 0:
      adj_cos_res.append(token_cos_mean(res_in, prev_res_in))
      adj_cos_ln.append(token_cos_mean(lnx, prev_ln_in))
      adj_cos_q.append(token_cos_mean(q_local, prev_q_local))
      adj_cos_k.append(token_cos_mean(k_local, prev_k_local))
      adj_cos_v.append(token_cos_mean(v_local, prev_v_local))
      adj_cos_o.append(token_cos_mean(o_local, prev_o_local))
    prev_res_in, prev_ln_in = res_in, lnx
    prev_q_local, prev_k_local, prev_v_local, prev_o_local = q_local, k_local, v_local, o_local

    # Record sliced reader vectors at `sample_pos`: [B, P, 4, N, V/K]
    layer_z.append(z_gate[:, sample_pos])                                # [B, P, V]
    layer_m_rms.append(1.0 / inv_m_rms[:, sample_pos, 0, 0])             # [B, P]
    layer_q_dyn.append(jnp.stack(
        (q_q_dyn_eff[:, sample_pos], q_k_dyn_eff[:, sample_pos],
         q_v_dyn_eff[:, sample_pos], q_o_dyn_eff[:, sample_pos]), axis=2))  # [B, P, 4, N, V]
    layer_q_stat.append(jnp.stack(
        (q_q_stat_eff[:, sample_pos], q_k_stat_eff[:, sample_pos],
         q_v_stat_eff[:, sample_pos], q_o_stat_eff[:, sample_pos]), axis=2)) # [B, P, 4, N, V]
    layer_u_read.append(jnp.stack(
        (q_local[:, sample_pos], k_local[:, sample_pos],
         v_local[:, sample_pos], o_local[:, sample_pos]), axis=2).astype(jnp.float32))  # [B, P, 4, N, K]

    # Standard RoPE Q/K tail + full-100 QKNorm + RoPE on tail 20
    q_std = (
        jnp.einsum("btd,dns->btns", lnx, sa["query"]["kernel"].astype(dtype))
        + sa["query"]["bias"].astype(dtype)
    )
    k_std = (
        jnp.einsum("btd,dns->btns", lnx, sa["key"]["kernel"].astype(dtype))
        + sa["key"]["bias"].astype(dtype)
    )
    q_full = scaled_rms_norm(
        jnp.concatenate((q_local, q_std), axis=-1),
        1.0 + sa["qk_norm"]["q_norm"]["scale"].astype(jnp.float32),
        ln_eps,
    )
    k_full = scaled_rms_norm(
        jnp.concatenate((k_local, k_std), axis=-1),
        1.0 + sa["qk_norm"]["k_norm"]["scale"].astype(jnp.float32),
        ln_eps,
    )
    q_full = jnp.concatenate((q_full[..., :k_dim], apply_rope_half(q_full[..., k_dim:], cos, sin)), axis=-1)
    k_full = jnp.concatenate((k_full[..., :k_dim], apply_rope_half(k_full[..., k_dim:], cos, sin)), axis=-1)
    q_prescaled = q_full / jnp.asarray(math.sqrt(k_dim + s_dim), dtype=dtype)

    # Head activation & sink statistics
    layer_head_stats.append(compute_head_stats(
        q_prescaled, k_full, head_qpos, categories, paper_sink_mask, strict_sink_mask,
        strictly_causal=True,
    ))

    # Strictly causal attention (BF16 operands, FP32 softmax)
    attn_logits = jnp.einsum("bthd,bshd->bhts", q_prescaled, k_full).astype(jnp.float32)
    attn_weights = jax.nn.softmax(jnp.where(safe_mask[None, None, :, :], attn_logits, -1.0e30), axis=-1)
    attn_weights = jnp.where(row_empty[None, None, :, :], 0.0, attn_weights).astype(dtype)
    y_attn = jnp.einsum("bhts,bshk->bthk", attn_weights, v_local)
    o_head = y_attn + o_local
    attn_out = (
        jnp.einsum("btnk,nkd->btd", o_head, sa["out"]["kernel"].astype(dtype))
        + sa["out"]["bias"].astype(dtype)
    )

    x = x + attn_out
    h_post = layer_norm(
        x,
        lyr["post_self_attention_layer_norm"]["scale"],
        lyr["post_self_attention_layer_norm"]["bias"],
        ln_eps,
    )

    # Writes on all layers except final_local_layer
    if l_idx < num_layers - 1:
      gw = jax.nn.sigmoid(
          jnp.einsum("btd,dn->btn", lnx, sa["W_gw"]["kernel"].astype(dtype))
          + sa["gw_b0"].astype(dtype)
      )
      if ablation == "no_attn_write":
        gw = jnp.zeros_like(gw)
      attn_w_content = (gw[..., None] * rms_norm(o_head, write_eps)).astype(dtype)
      p_hid = gelu_tanh(jnp.einsum("btd,dr->btr", lnx, sa["P_loc_down"]["kernel"].astype(dtype)))
      attn_w_addr = rms_norm(
          jnp.einsum("btr,rnv->btnv", p_hid, sa["P_loc_up"]["kernel"].astype(dtype))
          + sa["P_loc_up"]["bias"].astype(dtype),
          write_eps,
      )
      dM_attn = jnp.sum(attn_w_content[..., :, None] * attn_w_addr[..., None, :], axis=-3)

      # Memory FFN write
      mem_hid = gelu_exact(jnp.einsum("btd,df->btf", h_post, lyr["memory_mlp"]["wi"]["kernel"].astype(dtype)))
      mem_out = jnp.einsum("btf,fd->btd", mem_hid, lyr["memory_mlp"]["wo"]["kernel"].astype(dtype))
      mem_heads = (mem_out.reshape(b_sz, t_len, n_heads, k_dim).astype(jnp.float32) / mlp_tau).astype(dtype)
      gm = jax.nn.sigmoid(
          jnp.einsum("btd,dn->btn", h_post, lyr["mlp_write_gate"]["kernel"].astype(dtype))
          + lyr["mlp_write_gate_bias"].astype(dtype)
      )
      if ablation == "no_mlp_write":
        gm = jnp.zeros_like(gm)
      mlp_w_content = (gm[..., None] * mem_heads).astype(dtype)
      ma_hid = gelu_tanh(jnp.einsum("btd,dr->btr", h_post, lyr["mlp_address_down"]["kernel"].astype(dtype)))
      mlp_w_addr = rms_norm(
          jnp.einsum("btr,rnv->btnv", ma_hid, lyr["mlp_address_up"]["kernel"].astype(dtype))
          + lyr["mlp_address_up"]["bias"].astype(dtype),
          write_eps,
      )
      dM_mlp = jnp.sum(mlp_w_content[..., :, None] * mlp_w_addr[..., None, :], axis=-3)
      M = M + (dM_attn + dM_mlp)

      writers_o_cur.append(attn_w_content[:, sample_pos].astype(jnp.float32))
      writers_k_cur.append(attn_w_addr[:, sample_pos].astype(jnp.float32))
      writers_o_prev.append(attn_w_content[:, prev_pos].astype(jnp.float32))
      writers_k_prev.append(attn_w_addr[:, prev_pos].astype(jnp.float32))

      writers_o_cur.append(mlp_w_content[:, sample_pos].astype(jnp.float32))
      writers_k_cur.append(mlp_w_addr[:, sample_pos].astype(jnp.float32))
      writers_o_prev.append(mlp_w_content[:, prev_pos].astype(jnp.float32))
      writers_k_prev.append(mlp_w_addr[:, prev_pos].astype(jnp.float32))

    # Ordinary residual FFN
    mlp_hid = gelu_exact(
        jnp.einsum("btd,df->btf", h_post, lyr["mlp"]["wi"]["kernel"].astype(dtype))
        + lyr["mlp"]["wi"]["bias"].astype(dtype)
    )
    mlp_out = (
        jnp.einsum("btf,fd->btd", mlp_hid, lyr["mlp"]["wo"]["kernel"].astype(dtype))
        + lyr["mlp"]["wo"]["bias"].astype(dtype)
    )
    x = x + mlp_out

  final_h = layer_norm(
      x,
      u_params["final_norm"]["scale"],
      u_params["final_norm"]["bias"],
      ln_eps,
  )
  xent, last_logits = bf16_logits_and_xent(final_h, u_params["logits_dense"], targets, logit_pos=logit_pos)

  return {
      "xent": xent,                                              # [B, T]
      "last_logits": last_logits,                                # [B, V_vocab]
      "writers_o_cur": jnp.concatenate(writers_o_cur, axis=2),   # [B, P, W_all, K]
      "writers_k_cur": jnp.concatenate(writers_k_cur, axis=2),   # [B, P, W_all, V]
      "writers_o_prev": jnp.concatenate(writers_o_prev, axis=2), # [B, P, W_all, K]
      "writers_k_prev": jnp.concatenate(writers_k_prev, axis=2), # [B, P, W_all, V]
      "layer_z": jnp.stack(layer_z, axis=0),                     # [L, B, P, V]
      "layer_m_rms": jnp.stack(layer_m_rms, axis=0),             # [L, B, P]
      "layer_q_dyn": jnp.stack(layer_q_dyn, axis=0),             # [L, B, P, 4, N, V]
      "layer_q_stat": jnp.stack(layer_q_stat, axis=0),           # [L, B, P, 4, N, V]
      "layer_u_read": jnp.stack(layer_u_read, axis=0),           # [L, B, P, 4, N, K]
      "head_stats": jnp.stack(layer_head_stats, axis=0),         # [L, B, N, Q, M]
      "adj_cos": jnp.stack([
          jnp.stack(adj_cos_res),
          jnp.stack(adj_cos_ln),
          jnp.stack(adj_cos_q),
          jnp.stack(adj_cos_k),
          jnp.stack(adj_cos_v),
          jnp.stack(adj_cos_o),
      ], axis=0),                                                # [6, L-1]
  }


# ---------------------------------------------------------------------------
# 2. mHC-Pythia-2.8B (`Pythia2p8BSerialQKNormMHC300B`)
# ---------------------------------------------------------------------------

def sinkhorn_fp32(logits: jax.Array, iterations: int = 20, eps: float = 1.0e-6) -> jax.Array:
  m = jax.nn.softmax(logits.astype(jnp.float32), axis=-1) + eps
  m = m / (jnp.sum(m, axis=-2, keepdims=True) + eps)
  for _ in range(iterations - 1):
    m = m / (jnp.sum(m, axis=-1, keepdims=True) + eps)
    m = m / (jnp.sum(m, axis=-2, keepdims=True) + eps)
  return m


def mhc_connector(state: jax.Array, conn: dict[str, jax.Array], n_streams: int = 4,
                  norm_eps: float = 1.0e-6, pre_eps: float = 1.0e-6,
                  sinkhorn_iters: int = 20, sinkhorn_eps: float = 1.0e-6) -> tuple[jax.Array, jax.Array, jax.Array]:
  b_sz, t_len, _, d_dim = state.shape
  x = state.astype(jnp.float32).reshape(b_sz, t_len, n_streams * d_dim)
  r = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + norm_eps)
  kernel = conn["norm_scale"].astype(jnp.float32)[:, None] * conn["router_kernel"].astype(jnp.float32)
  a = conn["router_scale"].astype(jnp.float32)
  alphas = jnp.concatenate((
      jnp.broadcast_to(a[0], (n_streams,)),
      jnp.broadcast_to(a[1], (n_streams,)),
      jnp.broadcast_to(a[2], (n_streams * n_streams,)),
  ))
  z = r * jnp.einsum("btx,xz->btz", x, kernel, precision=jax.lax.Precision.HIGHEST) * alphas + conn["router_bias"].astype(jnp.float32)
  pre = jax.nn.sigmoid(z[..., :n_streams]) + pre_eps
  post = 2.0 * jax.nn.sigmoid(z[..., n_streams:2 * n_streams])
  mix = sinkhorn_fp32(z[..., 2 * n_streams:].reshape(b_sz, t_len, n_streams, n_streams), sinkhorn_iters, sinkhorn_eps)
  read = jnp.einsum("btn,btnd->btd", pre, state.astype(jnp.float32)).astype(state.dtype)
  return read, post, mix


def mhc_update(state: jax.Array, branch: jax.Array, post: jax.Array, mix: jax.Array) -> jax.Array:
  residual = jnp.einsum("btij,btid->btjd", mix, state.astype(jnp.float32))
  return (residual + post[..., None] * branch.astype(jnp.float32)[..., None, :]).astype(state.dtype)


def prepare_mhc_params(params_tree: dict[str, Any], dtype: jnp.dtype = jnp.bfloat16) -> dict[str, Any]:
  p = params_tree["params"] if "params" in params_tree else params_tree
  dec = p["decoder"]
  rs = dec["residual_stack"]
  num_layers = len([k for k in rs if k.startswith("layers_")])
  layers = [rs[f"layers_{i}"] for i in range(num_layers)]
  return {
      "embedding": p["token_embedder"]["embedding"].astype(dtype),
      "layers": layers,
      "output_connection": rs["output_connection"],
      "final_norm": dec["lm_head"]["decoder_norm"],
      "logits_dense": dec["lm_head"]["logits_dense"].astype(dtype),
  }


def mhc_forward_and_observe(
    m_params: dict[str, Any],
    inputs: jax.Array,
    targets: jax.Array,
    head_qpos: jax.Array,
    categories: jax.Array,
    paper_sink_mask: jax.Array,
    strict_sink_mask: jax.Array,
    *,
    dtype: jnp.dtype = jnp.bfloat16,
    rotary_dim: int = 20,
    n_streams: int = 4,
    ln_eps: float = 1.0e-5,
) -> dict[str, jax.Array]:
  b_sz, t_len = inputs.shape
  positions = jnp.broadcast_to(jnp.arange(t_len, dtype=jnp.int32), (b_sz, t_len))
  cos, sin = rope_table(positions, rotary_dim, dtype)

  x = m_params["embedding"][inputs].astype(dtype)
  state = jnp.broadcast_to(x[:, :, None, :], (b_sz, t_len, n_streams, x.shape[-1]))

  src_idx = jnp.arange(t_len, dtype=jnp.int32)[None, :]
  tgt_idx = jnp.arange(t_len, dtype=jnp.int32)[:, None]
  causal = (src_idx <= tgt_idx)[None, None, :, :]

  head_stats = []
  adj_cos_read = []
  adj_cos_ln = []
  adj_cos_state = []
  prev_read = None
  prev_ln = None
  prev_state = None

  for l_idx, lyr in enumerate(m_params["layers"]):
    sa = lyr["self_attention"]
    state_in = state
    read_attn, post_a, mix_a = mhc_connector(state, lyr["attn_connection"], n_streams=n_streams)
    lnx = layer_norm(
        read_attn,
        lyr["pre_self_attention_layer_norm"]["scale"],
        lyr["pre_self_attention_layer_norm"]["bias"],
        ln_eps,
    )
    if l_idx > 0:
      adj_cos_state.append(token_cos_mean(state_in, prev_state))
      adj_cos_read.append(token_cos_mean(read_attn, prev_read))
      adj_cos_ln.append(token_cos_mean(lnx, prev_ln))
    prev_state, prev_read, prev_ln = state_in, read_attn, lnx

    q = jnp.einsum("btd,dnh->btnh", lnx, sa["query"]["kernel"].astype(dtype)) + sa["query"]["bias"].astype(dtype)
    k = jnp.einsum("btd,dnh->btnh", lnx, sa["key"]["kernel"].astype(dtype)) + sa["key"]["bias"].astype(dtype)
    v = jnp.einsum("btd,dnh->btnh", lnx, sa["value"]["kernel"].astype(dtype)) + sa["value"]["bias"].astype(dtype)
    h_dim = q.shape[-1]

    q = scaled_rms_norm(q, 1.0 + sa["qk_norm"]["q_norm"]["scale"].astype(jnp.float32), ln_eps)
    k = scaled_rms_norm(k, 1.0 + sa["qk_norm"]["k_norm"]["scale"].astype(jnp.float32), ln_eps)
    q = jnp.concatenate((apply_rope_half(q[..., :rotary_dim], cos, sin), q[..., rotary_dim:]), axis=-1)
    k = jnp.concatenate((apply_rope_half(k[..., :rotary_dim], cos, sin), k[..., rotary_dim:]), axis=-1)
    q_prescaled = q / jnp.asarray(math.sqrt(h_dim), dtype=dtype)

    head_stats.append(compute_head_stats(
        q_prescaled, k, head_qpos, categories, paper_sink_mask, strict_sink_mask,
        strictly_causal=False,
    ))

    attn_logits = jnp.einsum("bthd,bshd->bhts", q_prescaled, k).astype(jnp.float32)
    attn_weights = jax.nn.softmax(jnp.where(causal, attn_logits, -1.0e30), axis=-1).astype(dtype)
    y_attn = jnp.einsum("bhts,bshd->bthd", attn_weights, v)
    attn_out = jnp.einsum("bthd,hdo->bto", y_attn, sa["out"]["kernel"].astype(dtype)) + sa["out"]["bias"].astype(dtype)
    state = mhc_update(state, attn_out, post_a, mix_a)

    read_mlp, post_m, mix_m = mhc_connector(state, lyr["mlp_connection"], n_streams=n_streams)
    h_mlp = layer_norm(
        read_mlp,
        lyr["post_self_attention_layer_norm"]["scale"],
        lyr["post_self_attention_layer_norm"]["bias"],
        ln_eps,
    )
    mlp_i = lyr["mlp"]["wi"]["kernel"].shape[-1]
    mlp_hid = gelu_exact(
        jnp.einsum("btd,df->btf", h_mlp, lyr["mlp"]["wi"]["kernel"][:, :mlp_i].astype(dtype))
        + lyr["mlp"]["wi"]["bias"][:mlp_i].astype(dtype)
    )
    mlp_out = (
        jnp.einsum("btf,fd->btd", mlp_hid, lyr["mlp"]["wo"]["kernel"][:mlp_i, :].astype(dtype))
        + lyr["mlp"]["wo"]["bias"].astype(dtype)
    )
    state = mhc_update(state, mlp_out, post_m, mix_m)

  # Output stream reduction
  oc = m_params["output_connection"]
  flat_s = state.astype(jnp.float32).reshape(b_sz, t_len, n_streams * x.shape[-1])
  r_out = jax.lax.rsqrt(jnp.mean(jnp.square(flat_s), axis=-1, keepdims=True) + 1.0e-6)
  h_k = oc["norm_scale"].astype(jnp.float32)[:, None] * oc["head_kernel"].astype(jnp.float32)
  h_logits = r_out * jnp.einsum("btx,xn->btn", flat_s, h_k, precision=jax.lax.Precision.HIGHEST)
  h_gates = jax.nn.sigmoid(h_logits * oc["head_scale"].astype(jnp.float32) + oc["head_bias"].astype(jnp.float32)) + 1.0e-6
  reduced = jnp.einsum("btn,btnd->btd", h_gates, state.astype(jnp.float32)).astype(dtype)

  final_h = layer_norm(
      reduced,
      m_params["final_norm"]["scale"],
      m_params["final_norm"]["bias"],
      ln_eps,
  )
  xent, last_logits = bf16_logits_and_xent(final_h, m_params["logits_dense"], targets)

  return {
      "xent": xent,
      "last_logits": last_logits,
      "head_stats": jnp.stack(head_stats, axis=0),
      "adj_cos": jnp.stack([
          jnp.stack(adj_cos_state),
          jnp.stack(adj_cos_read),
          jnp.stack(adj_cos_ln),
      ], axis=0),
  }


# ---------------------------------------------------------------------------
# 3. Official Pythia-2.8B (`EleutherAI/pythia-2.8b`, GPT-NeoX parallel residual)
# ---------------------------------------------------------------------------

def prepare_pythia_hf_params(hf_tensors: dict[str, np.ndarray], dtype: jnp.dtype = jnp.bfloat16,
                             num_layers: int = 32, num_heads: int = 32, head_dim: int = 80) -> dict[str, Any]:
  """Convert HuggingFace `EleutherAI/pythia-2.8b` safetensors dict into JAX arrays."""
  d_model = num_heads * head_dim
  layers = []
  for i in range(num_layers):
    pfx = f"gpt_neox.layers.{i}."
    # NeoX query_key_value.weight is [num_heads * 3 * head_dim, d_model] -> transpose to [d_model, H, 3, D]
    qkv_w = hf_tensors[pfx + "attention.query_key_value.weight"].T.reshape(d_model, num_heads, 3, head_dim)
    qkv_b = hf_tensors[pfx + "attention.query_key_value.bias"].reshape(num_heads, 3, head_dim)
    layers.append({
        "in_ln_scale": jnp.asarray(hf_tensors[pfx + "input_layernorm.weight"], jnp.float32),
        "in_ln_bias": jnp.asarray(hf_tensors[pfx + "input_layernorm.bias"], jnp.float32),
        "post_ln_scale": jnp.asarray(hf_tensors[pfx + "post_attention_layernorm.weight"], jnp.float32),
        "post_ln_bias": jnp.asarray(hf_tensors[pfx + "post_attention_layernorm.bias"], jnp.float32),
        "q_w": jnp.asarray(qkv_w[:, :, 0, :], dtype),
        "k_w": jnp.asarray(qkv_w[:, :, 1, :], dtype),
        "v_w": jnp.asarray(qkv_w[:, :, 2, :], dtype),
        "q_b": jnp.asarray(qkv_b[:, 0, :], dtype),
        "k_b": jnp.asarray(qkv_b[:, 1, :], dtype),
        "v_b": jnp.asarray(qkv_b[:, 2, :], dtype),
        "out_w": jnp.asarray(hf_tensors[pfx + "attention.dense.weight"].T.reshape(num_heads, head_dim, d_model), dtype),
        "out_b": jnp.asarray(hf_tensors[pfx + "attention.dense.bias"], dtype),
        "wi_w": jnp.asarray(hf_tensors[pfx + "mlp.dense_h_to_4h.weight"].T, dtype),
        "wi_b": jnp.asarray(hf_tensors[pfx + "mlp.dense_h_to_4h.bias"], dtype),
        "wo_w": jnp.asarray(hf_tensors[pfx + "mlp.dense_4h_to_h.weight"].T, dtype),
        "wo_b": jnp.asarray(hf_tensors[pfx + "mlp.dense_4h_to_h.bias"], dtype),
    })
  return {
      "embedding": jnp.asarray(hf_tensors["gpt_neox.embed_in.weight"], dtype),
      "layers": layers,
      "final_ln_scale": jnp.asarray(hf_tensors["gpt_neox.final_layer_norm.weight"], jnp.float32),
      "final_ln_bias": jnp.asarray(hf_tensors["gpt_neox.final_layer_norm.bias"], jnp.float32),
      "logits_dense": jnp.asarray(hf_tensors["embed_out.weight"].T, dtype),
  }


def pythia_forward_and_observe(
    p_params: dict[str, Any],
    inputs: jax.Array,
    targets: jax.Array,
    head_qpos: jax.Array,
    categories: jax.Array,
    paper_sink_mask: jax.Array,
    strict_sink_mask: jax.Array,
    *,
    dtype: jnp.dtype = jnp.bfloat16,
    rotary_dim: int = 20,
    ln_eps: float = 1.0e-5,
) -> dict[str, jax.Array]:
  b_sz, t_len = inputs.shape
  positions = jnp.broadcast_to(jnp.arange(t_len, dtype=jnp.int32), (b_sz, t_len))
  cos, sin = rope_table(positions, rotary_dim, dtype)

  x = p_params["embedding"][inputs].astype(dtype)
  src_idx = jnp.arange(t_len, dtype=jnp.int32)[None, :]
  tgt_idx = jnp.arange(t_len, dtype=jnp.int32)[:, None]
  causal = (src_idx <= tgt_idx)[None, None, :, :]

  head_stats = []
  adj_cos_res = []
  adj_cos_ln = []
  prev_res = None
  prev_ln = None

  for l_idx, lyr in enumerate(p_params["layers"]):
    res_in = x
    lnx_attn = layer_norm(x, lyr["in_ln_scale"], lyr["in_ln_bias"], ln_eps)
    lnx_mlp = layer_norm(x, lyr["post_ln_scale"], lyr["post_ln_bias"], ln_eps)
    if l_idx > 0:
      adj_cos_res.append(token_cos_mean(res_in, prev_res))
      adj_cos_ln.append(token_cos_mean(lnx_attn, prev_ln))
    prev_res, prev_ln = res_in, lnx_attn

    q = jnp.einsum("btd,dnh->btnh", lnx_attn, lyr["q_w"]) + lyr["q_b"]
    k = jnp.einsum("btd,dnh->btnh", lnx_attn, lyr["k_w"]) + lyr["k_b"]
    v = jnp.einsum("btd,dnh->btnh", lnx_attn, lyr["v_w"]) + lyr["v_b"]
    h_dim = q.shape[-1]

    q = jnp.concatenate((apply_rope_half(q[..., :rotary_dim], cos, sin), q[..., rotary_dim:]), axis=-1)
    k = jnp.concatenate((apply_rope_half(k[..., :rotary_dim], cos, sin), k[..., rotary_dim:]), axis=-1)
    q_prescaled = q / jnp.asarray(math.sqrt(h_dim), dtype=dtype)

    head_stats.append(compute_head_stats(
        q_prescaled, k, head_qpos, categories, paper_sink_mask, strict_sink_mask,
        strictly_causal=False,
    ))

    attn_logits = jnp.einsum("bthd,bshd->bhts", q_prescaled, k).astype(jnp.float32)
    attn_weights = jax.nn.softmax(jnp.where(causal, attn_logits, -1.0e30), axis=-1).astype(dtype)
    y_attn = jnp.einsum("bhts,bshd->bthd", attn_weights, v)
    attn_out = jnp.einsum("bthd,hdo->bto", y_attn, lyr["out_w"]) + lyr["out_b"]

    mlp_hid = gelu_exact(jnp.einsum("btd,df->btf", lnx_mlp, lyr["wi_w"]) + lyr["wi_b"])
    mlp_out = jnp.einsum("btf,fd->btd", mlp_hid, lyr["wo_w"]) + lyr["wo_b"]

    # Official EleutherAI/pythia-2.8b uses parallel residual: x = x + attn_out + mlp_out
    x = x + attn_out + mlp_out

  final_h = layer_norm(x, p_params["final_ln_scale"], p_params["final_ln_bias"], ln_eps)
  xent, last_logits = bf16_logits_and_xent(final_h, p_params["logits_dense"], targets)

  return {
      "xent": xent,
      "last_logits": last_logits,
      "head_stats": jnp.stack(head_stats, axis=0),
      "adj_cos": jnp.stack([
          jnp.stack(adj_cos_res),
          jnp.stack(adj_cos_ln),
      ], axis=0),
  }
