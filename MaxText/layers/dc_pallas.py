"""Two-Pass Global Flash-DCMHA Pallas kernel for query-wise Dynamic Composed Attention (DC2).

Algebraic structure (query-wise DC2: key_wise=False, static_proj=False):
  For query position q and key position k (0 <= k <= q), head h in [0, H):
    S[m, k, q]       = <K[k, m, :], Q[q, m, :]>
    S_tilde[h, k, q] = (1 + pre_dd[q, h]) * S[h, k, q]
                       + sum_{i=0}^{1} pre_w2[q, i, h] * sum_{m=0}^{H-1} pre_w1[q, i, m] * S[m, k, q]
    P[m, k, q]       = softmax_{k <= q}(S_tilde[m, k, q])
    P_tilde[h, k, q] = (1 + post_dd[q, h]) * P[h, k, q]
                       + sum_{i=0}^{1} post_w2[q, i, h] * sum_{m=0}^{H-1} post_w1[q, i, m] * P[m, k, q]
    Out[q, h, :]     = sum_{k <= q} P_tilde[h, k, q] * V[k, h, :]

Key hardware optimizations on TPU (v5p / v6e):
  1. Two-pass streaming eliminates all O(T^2) HBM materialization:
     - Forward Pass 1 streams K only (no V DMA) to compute exact per-(head, query) logsumexp (LSE).
     - Forward Pass 2 streams (K, V), reconstructs exact P = exp(S_tilde - LSE) in VMEM, applies
       post-composition, and accumulates Out in VMEM.
  2. In-VMEM channel-space MXU factorization:
     - Pre-composition cross-head reductions sum_m pre_w1[q, i, m] * <K_m, Q_m> are reformulated as
       single (H*D)-wide MXU matmuls K_flat^T @ (pre_w1 * Q)_flat, moving rank-2 cross-head mixing
       off the slow VPU onto the BF16 MXU with zero extra HBM tensors.
     - Post-composition projections sum_k g_i[k, q] * V[k, h, :] similarly execute as (H*D)-wide MXU
       matmuls V_flat @ g_i, keeping VMEM I/O strictly at (Q, K, V) -> Out.
  3. Sub-blocked KV tiles (block_kv / block_kv_compute via lax.fori_loop) amortize Q/dO loads and
     fused-backward partial reductions across large KV blocks while bounding live VMEM stack usage.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

from flax import linen as nn
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def pack_dc2_weights(qw1: jax.Array, qw2: jax.Array, qdd: jax.Array) -> jax.Array:
  """Packs (qw1 [B,T,1,2,H], qw2 [B,T,1,2,H], qdd [B,T,1,H]) into [B, 5*H, T] FP32."""
  b, t, _, _, h = qw1.shape
  w1 = qw1[:, :, 0, :, :].astype(F32).transpose(0, 2, 3, 1)            # [B, 2, H, T]
  w2 = qw2[:, :, 0, :, :].astype(F32).transpose(0, 2, 3, 1)            # [B, 2, H, T]
  d1 = (1.0 + qdd[:, :, 0, :].astype(F32)).transpose(0, 2, 1)[:, None] # [B, 1, H, T]
  return jnp.concatenate([w1, w2, d1], axis=1).reshape(b, 5 * h, t)


def _reference_dc2_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    wpre: jax.Array,
    wpost: jax.Array,
    seg_ids: Optional[jax.Array] = None,
    dot_prec: Any = None,
) -> jax.Array:
  """Exact pure-JAX fallback (used on CPU or when seq_len < block size)."""
  b, t, h, _ = q.shape
  s = jnp.einsum("bthd,bshd->bhts", q.astype(F32), k.astype(F32), precision=dot_prec, preferred_element_type=F32)

  def _apply_packed(w_packed: jax.Array, x: jax.Array) -> jax.Array:
    w1_0, w1_1 = w_packed[:, 0:h, :], w_packed[:, h : 2 * h, :]
    w2_0, w2_1 = w_packed[:, 2 * h : 3 * h, :], w_packed[:, 3 * h : 4 * h, :]
    d1 = w_packed[:, 4 * h : 5 * h, :]
    h0 = jnp.einsum("bmts,bmt->bts", x, w1_0, precision=dot_prec, preferred_element_type=F32)
    h1 = jnp.einsum("bmts,bmt->bts", x, w1_1, precision=dot_prec, preferred_element_type=F32)
    return x * d1[:, :, :, None] + h0[:, None, :, :] * w2_0[:, :, :, None] + h1[:, None, :, :] * w2_1[:, :, :, None]

  s_tilde = _apply_packed(wpre, s)
  valid = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))[None, None, :, :]
  if seg_ids is not None:
    seg_match = (seg_ids[:, :, None] == seg_ids[:, None, :])[:, None, :, :]
    valid = valid & seg_match
  s_masked = jnp.where(valid, s_tilde, DEFAULT_MASK_VALUE)
  p = jax.nn.softmax(s_masked, axis=-1)
  p_tilde = jnp.where(valid, _apply_packed(wpost, p), 0.0).astype(v.dtype)
  return jnp.einsum("bhts,bshd->bthd", p_tilde, v, precision=dot_prec, preferred_element_type=F32).astype(v.dtype)


@functools.lru_cache(maxsize=32)
def _q_major_schedule(seq_len: int, b_q: int, b_kv: int):
  nq, nkv = seq_len // b_q, seq_len // b_kv
  kv_idx = np.zeros((nq, nkv), dtype=np.int32)
  run_mask = np.zeros((nq, nkv), dtype=np.int32)
  kv_counts = np.zeros((nq,), dtype=np.int32)
  for i in range(nq):
    q_start, q_end = i * b_q, (i + 1) * b_q - 1
    last_valid = min(nkv - 1, q_end // b_kv)
    kv_counts[i] = last_valid + 1
    for j in range(nkv):
      k_start, k_end = j * b_kv, (j + 1) * b_kv - 1
      if k_start <= q_end:
        kv_idx[i, j] = j
        run_mask[i, j] = 2 if k_end <= q_start else 1
      else:
        kv_idx[i, j] = last_valid
        run_mask[i, j] = 0
  return kv_idx, run_mask, kv_counts


@functools.lru_cache(maxsize=32)
def _kv_major_schedule(seq_len: int, b_q: int, b_kv: int):
  nq, nkv = seq_len // b_q, seq_len // b_kv
  q_idx = np.zeros((nkv, nq), dtype=np.int32)
  run_mask = np.zeros((nkv, nq), dtype=np.int32)
  for j in range(nkv):
    k_start, k_end = j * b_kv, (j + 1) * b_kv - 1
    first_valid = k_start // b_q
    for i in range(nq):
      q_start, q_end = i * b_q, (i + 1) * b_q - 1
      if q_end >= k_start:
        q_idx[j, i] = i
        run_mask[j, i] = 2 if k_end <= q_start else 1
      else:
        q_idx[j, i] = first_valid
        run_mask[j, i] = 0
  return q_idx, run_mask


def _sub_valid(
    mode: jax.Array,
    q_offset: jax.Array,
    k_offset: jax.Array,
    bq: int,
    ckv: int,
    q_seg: Optional[jax.Array] = None,
    k_seg: Optional[jax.Array] = None,
) -> jax.Array:
  should_not_mask = mode != 1
  q_pos = q_offset + lax.broadcasted_iota(jnp.int32, (ckv, bq), 1)
  k_pos = k_offset + lax.broadcasted_iota(jnp.int32, (ckv, bq), 0)
  valid = (q_pos >= k_pos) | should_not_mask
  if q_seg is not None and k_seg is not None:
    # q_seg: [1, bq], k_seg: [ckv, 128] -> repeat k_seg[:ckv, :1] or compare via broadcast
    valid = valid & (q_seg == k_seg)
  return valid


@functools.lru_cache(maxsize=16)
def _make_tpu_dc2_primitive(
    num_heads: int,
    head_dim: int,
    bq: int,
    bkv: int,
    ckv: int,
    bwd_bq: int,
    bwd_bkv: int,
    bwd_ckv: int,
    vmem_mib: int,
    use_segment_ids: bool,
    dot_prec: Any = None,
):
  """Builds a cached custom_vjp Two-Pass Global Flash-DCMHA Pallas primitive."""
  if num_heads % 8 != 0 or head_dim % 8 != 0:
    raise ValueError(f"dc_pallas requires num_heads ({num_heads}) and head_dim ({head_dim}) divisible by 8.")
  if bq % 128 != 0 or ckv % 128 != 0 or bwd_bq % 128 != 0 or bwd_ckv % 128 != 0:
    raise ValueError("dc_pallas block sizes must be multiples of 128.")
  if bkv % ckv != 0 or bwd_bkv % bwd_ckv != 0:
    raise ValueError("block_kv must be a multiple of block_kv_compute.")

  H = num_heads
  D = head_dim
  HD = H * D
  W_ROWS = 5 * H
  num_fwd_sub = bkv // ckv
  num_bwd_sub = bwd_bkv // bwd_ckv
  cparams = pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "arbitrary"),
      vmem_limit_bytes=int(vmem_mib) * 1024 * 1024,
  )

  def _dot(lhs, rhs, dims):
    return lax.dot_general(lhs, rhs, dims, precision=dot_prec, preferred_element_type=F32)

  def _prep_q(q, wpre):
    qf = q.astype(F32)
    bq_ = q.shape[-1]
    qhat = (qf * wpre[4 * H : 5 * H][:, None, :]).astype(q.dtype)
    qw1_0 = (qf * wpre[0:H][:, None, :]).reshape(HD, bq_).astype(q.dtype)
    qw1_1 = (qf * wpre[H : 2 * H][:, None, :]).reshape(HD, bq_).astype(q.dtype)
    return qhat, qw1_0, qw1_1

  def _prep_do(do, wpost):
    dof = do.astype(F32)
    bq_ = do.shape[-1]
    dobar = (dof * wpost[4 * H : 5 * H][:, None, :]).astype(do.dtype)
    dg0 = (dof * wpost[2 * H : 3 * H][:, None, :]).reshape(HD, bq_).astype(do.dtype)
    dg1 = (dof * wpost[3 * H : 4 * H][:, None, :]).reshape(HD, bq_).astype(do.dtype)
    return dobar, dg0, dg1

  def _scores_from_prepped(k_sub, qhat, qw1_0, qw1_1, wpre):
    k_flat = k_sub.reshape(HD, k_sub.shape[-1])
    s_hat = _dot(k_sub, qhat, (((1,), (1,)), ((0,), (0,))))
    h0 = _dot(k_flat, qw1_0, (((0,), (0,)), ((), ())))
    h1 = _dot(k_flat, qw1_1, (((0,), (0,)), ((), ())))
    s_tilde = (
        s_hat
        + wpre[2 * H : 3 * H][:, None, :] * h0[None, :, :]
        + wpre[3 * H : 4 * H][:, None, :] * h1[None, :, :]
    )
    return s_tilde, h0, h1, k_flat

  def _load_seg_pair(q_seg_ref, k_seg_ref, sub, step_ckv, target_bq):
    if not use_segment_ids:
      return None, None
    q_s = q_seg_ref[:1, :]  # [1, target_bq]
    k_slice = k_seg_ref[pl.ds(sub * step_ckv, step_ckv), :]  # [step_ckv, 128]
    k_s = pltpu.repeat(k_slice, target_bq // 128, axis=1)    # [step_ckv, target_bq]
    return q_s, k_s

  def _fwd_pass1_kernel(
      kv_idx_ref, run_mask_ref, kv_cnt_ref,
      q_ref, k_ref, wpre_ref, q_seg_ref, k_seg_ref,
      lse_ref, m_scratch_ref, l_scratch_ref,
  ):
    q_idx, step_idx = pl.program_id(1), pl.program_id(2)
    nkv = run_mask_ref.shape[1]
    need_pass1 = kv_cnt_ref[q_idx] > 1

    @pl.when(step_idx == 0)
    def _init():
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, DEFAULT_MASK_VALUE)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

    mode = run_mask_ref[q_idx, step_idx]
    kv_idx = kv_idx_ref[q_idx, step_idx]

    @pl.when((mode > 0) & need_pass1)
    def _body():
      wpre = wpre_ref[...]
      qhat, qw1_0, qw1_1 = _prep_q(q_ref[...], wpre)

      def sub_step(sub, _):
        k_sub = k_ref[:, :, pl.ds(sub * ckv, ckv)]
        s_tilde, _, _, _ = _scores_from_prepped(k_sub, qhat, qw1_0, qw1_1, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, ckv, bq)
        valid = _sub_valid(mode, q_idx * bq, kv_idx * bkv + sub * ckv, bq, ckv, q_s, k_s)
        s_tilde = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        m_prev = m_scratch_ref[...]
        m_next = jnp.maximum(m_prev, jnp.max(s_tilde, axis=1))
        l_scratch_ref[...] = jnp.exp(m_prev - m_next) * l_scratch_ref[...] + jnp.sum(
            jnp.exp(s_tilde - m_next[:, None, :]), axis=1
        )
        m_scratch_ref[...] = m_next

      if num_fwd_sub == 1:
        sub_step(0, None)
      else:
        lax.fori_loop(0, num_fwd_sub, sub_step, None)

    @pl.when((step_idx == nkv - 1) & need_pass1)
    def _done():
      lse_ref[...] = m_scratch_ref[...] + jnp.log(l_scratch_ref[...])

  def _fwd_pass2_kernel(
      kv_idx_ref, run_mask_ref, kv_cnt_ref,
      q_ref, k_ref, v_ref, wpre_ref, wpost_ref, lse_in_ref, q_seg_ref, k_seg_ref,
      o_ref, lse_out_ref, o_scratch_ref,
  ):
    q_idx, step_idx = pl.program_id(1), pl.program_id(2)
    nkv = run_mask_ref.shape[1]
    single_blk = kv_cnt_ref[q_idx] == 1

    @pl.when(step_idx == 0)
    def _init():
      o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

    mode = run_mask_ref[q_idx, step_idx]
    kv_idx = kv_idx_ref[q_idx, step_idx]

    @pl.when(mode > 0)
    def _body():
      wpre, wpost = wpre_ref[...], wpost_ref[...]
      qhat, qw1_0, qw1_1 = _prep_q(q_ref[...], wpre)

      def sub_step(sub, _):
        sl = pl.ds(sub * ckv, ckv)
        k_sub, v_sub = k_ref[:, :, sl], v_ref[:, :, sl]
        v_flat = v_sub.reshape(HD, ckv)
        s_tilde, _, _, _ = _scores_from_prepped(k_sub, qhat, qw1_0, qw1_1, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, ckv, bq)
        valid = _sub_valid(mode, q_idx * bq, kv_idx * bkv + sub * ckv, bq, ckv, q_s, k_s)
        s_masked = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        if num_fwd_sub == 1:
          m_loc = jnp.max(s_masked, axis=1)
          lse_loc = m_loc + jnp.log(jnp.sum(jnp.exp(s_masked - m_loc[:, None, :]), axis=1))
          lse = jnp.where(single_blk, lse_loc, lse_in_ref[...])
        else:
          lse = lse_in_ref[...]
        lse_out_ref[...] = lse
        p = jnp.where(valid[None, :, :], jnp.exp(s_masked - lse[:, None, :]), 0.0)
        obar_sub = _dot(v_sub, p.astype(v_sub.dtype), (((2,), (1,)), ((0,), (0,))))
        g0 = p[0] * wpost[0:1, :]
        g1 = p[0] * wpost[H : H + 1, :]
        for m in range(1, H):
          pm = p[m]
          g0 = g0 + pm * wpost[m : m + 1, :]
          g1 = g1 + pm * wpost[H + m : H + m + 1, :]
        g0_sub = _dot(v_flat, g0.astype(v_sub.dtype), (((1,), (0,)), ((), ()))).reshape(H, D, bq)
        g1_sub = _dot(v_flat, g1.astype(v_sub.dtype), (((1,), (0,)), ((), ()))).reshape(H, D, bq)
        o_scratch_ref[...] = (
            o_scratch_ref[...]
            + obar_sub * wpost[4 * H : 5 * H][:, None, :]
            + g0_sub * wpost[2 * H : 3 * H][:, None, :]
            + g1_sub * wpost[3 * H : 4 * H][:, None, :]
        )

      if num_fwd_sub == 1:
        sub_step(0, None)
      else:
        lax.fori_loop(0, num_fwd_sub, sub_step, None)

    @pl.when(step_idx == nkv - 1)
    def _done():
      o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)

  def _prepare_seg_arrays(seg_ids, b_dim, t_dim):
    if not use_segment_ids:
      return None, None
    q_seg = lax.broadcast_in_dim(seg_ids, (b_dim, 8, t_dim), (0, 2))
    k_seg = lax.broadcast_in_dim(seg_ids, (b_dim, t_dim, 128), (0, 1))
    return q_seg, k_seg

  def _fwd_impl(qt, kt, vt, wpre, wpost, seg_ids):
    B, _, _, T = qt.shape
    nq, nkv = T // bq, T // bkv
    kv_idx_np, run_mask_np, kv_cnt_np = _q_major_schedule(T, bq, bkv)
    kv_idx_arr = jnp.asarray(kv_idx_np)
    run_mask_arr = jnp.asarray(run_mask_np)
    kv_cnt_arr = jnp.full((nq,), 2, dtype=jnp.int32) if num_fwd_sub > 1 else jnp.asarray(kv_cnt_np)

    q_seg, k_seg = _prepare_seg_arrays(seg_ids, B, T)
    q_spec = pl.BlockSpec((None, H, D, bq), lambda b, i, j, *_: (b, 0, 0, i))
    k_spec = pl.BlockSpec((None, H, D, bkv), lambda b, i, j, kv_ref, *_: (b, 0, 0, kv_ref[i, j]))
    w_spec = pl.BlockSpec((None, W_ROWS, bq), lambda b, i, j, *_: (b, 0, i))
    lse_spec = pl.BlockSpec((None, H, bq), lambda b, i, j, *_: (b, 0, i))
    q_seg_spec = pl.BlockSpec((None, 8, bq), lambda b, i, j, *_: (b, 0, i)) if use_segment_ids else None
    k_seg_spec = (
        pl.BlockSpec((None, bkv, 128), lambda b, i, j, kv_ref, *_: (b, kv_ref[i, j], 0))
        if use_segment_ids
        else None
    )

    lse_tmp = pl.pallas_call(
        _fwd_pass1_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[q_spec, k_spec, w_spec, q_seg_spec, k_seg_spec],
            out_specs=lse_spec,
            scratch_shapes=[pltpu.VMEM((H, bq), F32), pltpu.VMEM((H, bq), F32)],
            grid=(B, nq, nkv),
        ),
        out_shape=jax.ShapeDtypeStruct((B, H, T), F32),
        compiler_params=cparams,
        name="dc2_flash_fwd_pass1",
    )(kv_idx_arr, run_mask_arr, kv_cnt_arr, qt, kt, wpre, q_seg, k_seg)

    out_t, lse = pl.pallas_call(
        _fwd_pass2_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[q_spec, k_spec, k_spec, w_spec, w_spec, lse_spec, q_seg_spec, k_seg_spec],
            out_specs=[q_spec, lse_spec],
            scratch_shapes=[pltpu.VMEM((H, D, bq), F32)],
            grid=(B, nq, nkv),
        ),
        out_shape=[jax.ShapeDtypeStruct((B, H, D, T), qt.dtype), jax.ShapeDtypeStruct((B, H, T), F32)],
        compiler_params=cparams,
        name="dc2_flash_fwd_pass2",
    )(kv_idx_arr, run_mask_arr, kv_cnt_arr, qt, kt, vt, wpre, wpost, lse_tmp, q_seg, k_seg)
    return out_t, lse

  def _bwd_pass1_kernel(
      kv_idx_ref, run_mask_ref,
      q_ref, k_ref, v_ref, do_ref, wpre_ref, wpost_ref, lse_ref, q_seg_ref, k_seg_ref,
      delta_ref, dwpost_ref, delta_scratch_ref, dwpost_scratch_ref,
  ):
    q_idx, step_idx = pl.program_id(1), pl.program_id(2)
    nkv = run_mask_ref.shape[1]

    @pl.when(step_idx == 0)
    def _init():
      delta_scratch_ref[...] = jnp.zeros_like(delta_scratch_ref)
      dwpost_scratch_ref[...] = jnp.zeros_like(dwpost_scratch_ref)

    mode = run_mask_ref[q_idx, step_idx]
    kv_idx = kv_idx_ref[q_idx, step_idx]

    @pl.when(mode > 0)
    def _body():
      wpre, wpost, lse = wpre_ref[...], wpost_ref[...], lse_ref[...]
      do = do_ref[...]
      qhat, qw1_0, qw1_1 = _prep_q(q_ref[...], wpre)

      def sub_step(sub, _):
        sl = pl.ds(sub * bwd_ckv, bwd_ckv)
        k_sub, v_sub = k_ref[:, :, sl], v_ref[:, :, sl]
        s_tilde, _, _, _ = _scores_from_prepped(k_sub, qhat, qw1_0, qw1_1, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, bwd_ckv, bwd_bq)
        valid = _sub_valid(mode, q_idx * bwd_bq, kv_idx * bwd_bkv + sub * bwd_ckv, bwd_bq, bwd_ckv, q_s, k_s)
        s_masked = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        p = jnp.where(valid[None, :, :], jnp.exp(s_masked - lse[:, None, :]), 0.0)
        dp_tilde = jnp.where(valid[None, :, :], _dot(v_sub, do, (((1,), (1,)), ((0,), (0,)))), 0.0)

        g0 = p[0] * wpost[0:1, :]
        g1 = p[0] * wpost[H : H + 1, :]
        dg0 = dp_tilde[0] * wpost[2 * H : 2 * H + 1, :]
        dg1 = dp_tilde[0] * wpost[3 * H : 3 * H + 1, :]
        for m in range(1, H):
          pm, dpm = p[m], dp_tilde[m]
          g0 = g0 + pm * wpost[m : m + 1, :]
          g1 = g1 + pm * wpost[H + m : H + m + 1, :]
          dg0 = dg0 + dpm * wpost[2 * H + m : 2 * H + m + 1, :]
          dg1 = dg1 + dpm * wpost[3 * H + m : 3 * H + m + 1, :]

        dw1_0 = jnp.stack([jnp.sum(dg0 * p[m], axis=0) for m in range(H)], axis=0)
        dw1_1 = jnp.stack([jnp.sum(dg1 * p[m], axis=0) for m in range(H)], axis=0)
        dw2_0 = jnp.stack([jnp.sum(dp_tilde[m] * g0, axis=0) for m in range(H)], axis=0)
        dw2_1 = jnp.stack([jnp.sum(dp_tilde[m] * g1, axis=0) for m in range(H)], axis=0)
        ddd = jnp.sum(dp_tilde * p, axis=1)
        dwpost_scratch_ref[...] = dwpost_scratch_ref[...] + jnp.concatenate([dw1_0, dw1_1, dw2_0, dw2_1, ddd], axis=0)

        dp = (
            dp_tilde * wpost[4 * H : 5 * H][:, None, :]
            + wpost[0:H][:, None, :] * dg0[None, :, :]
            + wpost[H : 2 * H][:, None, :] * dg1[None, :, :]
        )
        delta_scratch_ref[...] = delta_scratch_ref[...] + jnp.sum(p * dp, axis=1)

      if num_bwd_sub == 1:
        sub_step(0, None)
      else:
        lax.fori_loop(0, num_bwd_sub, sub_step, None)

    @pl.when(step_idx == nkv - 1)
    def _done():
      delta_ref[...] = delta_scratch_ref[...]
      dwpost_ref[...] = dwpost_scratch_ref[...]

  def _bwd_fused_kv_kernel(
      q_idx_ref, run_mask_ref,
      q_ref, k_ref, v_ref, do_ref, wpre_ref, wpost_ref, lse_ref, delta_ref, q_seg_ref, k_seg_ref,
      dq_part_ref, dwpre_part_ref, dk_ref, dv_ref,
      dk_scratch_ref, dv_scratch_ref, dq_s_ref,
  ):
    kv_idx, step_idx = pl.program_id(1), pl.program_id(2)
    nq = run_mask_ref.shape[1]

    @pl.when(step_idx == 0)
    def _init():
      dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
      dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

    mode = run_mask_ref[kv_idx, step_idx]
    q_idx = q_idx_ref[kv_idx, step_idx]

    @pl.when(mode == 0)
    def _zero():
      dq_part_ref[...] = jnp.zeros_like(dq_part_ref)
      dwpre_part_ref[...] = jnp.zeros_like(dwpre_part_ref)

    @pl.when(mode > 0)
    def _body():
      q, do = q_ref[...], do_ref[...]
      qf = q.astype(F32)
      wpre, wpost, lse, delta = wpre_ref[...], wpost_ref[...], lse_ref[...], delta_ref[...]
      qhat, qw1_0, qw1_1 = _prep_q(q, wpre)
      dobar, dg0_vec, dg1_vec = _prep_do(do, wpost)

      dq_s_ref[...] = jnp.zeros_like(dq_s_ref)
      dwpre_part_ref[...] = jnp.zeros_like(dwpre_part_ref)

      def sub_step(sub, _):
        sl = pl.ds(sub * bwd_ckv, bwd_ckv)
        k_sub, v_sub = k_ref[:, :, sl], v_ref[:, :, sl]
        v_flat = v_sub.reshape(HD, bwd_ckv)
        s_tilde, h0, h1, k_flat = _scores_from_prepped(k_sub, qhat, qw1_0, qw1_1, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, bwd_ckv, bwd_bq)
        valid = _sub_valid(mode, q_idx * bwd_bq, kv_idx * bwd_bkv + sub * bwd_ckv, bwd_bq, bwd_ckv, q_s, k_s)
        s_masked = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        p = jnp.where(valid[None, :, :], jnp.exp(s_masked - lse[:, None, :]), 0.0)

        g0 = p[0] * wpost[0:1, :]
        g1 = p[0] * wpost[H : H + 1, :]
        for m in range(1, H):
          pm = p[m]
          g0 = g0 + pm * wpost[m : m + 1, :]
          g1 = g1 + pm * wpost[H + m : H + m + 1, :]
        dv_bar = _dot(dobar, p.astype(dobar.dtype), (((2,), (2,)), ((0,), (0,))))
        dv_g = (
            _dot(dg0_vec, g0.astype(dg0_vec.dtype), (((1,), (1,)), ((), ())))
            + _dot(dg1_vec, g1.astype(dg1_vec.dtype), (((1,), (1,)), ((), ())))
        ).reshape(H, D, bwd_ckv)
        dv_scratch_ref[:, :, sl] = dv_scratch_ref[:, :, sl] + dv_bar + dv_g

        dp_bar = _dot(v_sub, dobar, (((1,), (1,)), ((0,), (0,))))
        dg0 = jnp.where(valid, _dot(v_flat, dg0_vec, (((0,), (0,)), ((), ()))), 0.0)
        dg1 = jnp.where(valid, _dot(v_flat, dg1_vec, (((0,), (0,)), ((), ()))), 0.0)
        dp = (
            jnp.where(valid[None, :, :], dp_bar, 0.0)
            + wpost[0:H][:, None, :] * dg0[None, :, :]
            + wpost[H : 2 * H][:, None, :] * dg1[None, :, :]
        )
        ds_tilde = p * (dp - delta[:, None, :])

        dw2_0 = jnp.stack([jnp.sum(ds_tilde[m] * h0, axis=0) for m in range(H)], axis=0)
        dw2_1 = jnp.stack([jnp.sum(ds_tilde[m] * h1, axis=0) for m in range(H)], axis=0)

        dh0 = ds_tilde[0] * wpre[2 * H : 2 * H + 1, :]
        dh1 = ds_tilde[0] * wpre[3 * H : 3 * H + 1, :]
        for m in range(1, H):
          dsm = ds_tilde[m]
          dh0 = dh0 + dsm * wpre[2 * H + m : 2 * H + m + 1, :]
          dh1 = dh1 + dsm * wpre[3 * H + m : 3 * H + m + 1, :]

        dk_bar = _dot(qhat, ds_tilde.astype(qhat.dtype), (((2,), (2,)), ((0,), (0,))))
        dk_h = (
            _dot(qw1_0, dh0.astype(qw1_0.dtype), (((1,), (1,)), ((), ())))
            + _dot(qw1_1, dh1.astype(qw1_1.dtype), (((1,), (1,)), ((), ())))
        ).reshape(H, D, bwd_ckv)
        dk_scratch_ref[:, :, sl] = dk_scratch_ref[:, :, sl] + dk_bar + dk_h

        dqhat_sub = _dot(k_sub, ds_tilde.astype(k_sub.dtype), (((2,), (1,)), ((0,), (0,))))
        dqw1_0_sub = _dot(k_flat, dh0.astype(k_sub.dtype), (((1,), (0,)), ((), ()))).reshape(H, D, bwd_bq)
        dqw1_1_sub = _dot(k_flat, dh1.astype(k_sub.dtype), (((1,), (0,)), ((), ()))).reshape(H, D, bwd_bq)

        dw1_0 = jnp.sum(dqw1_0_sub * qf, axis=1)
        dw1_1 = jnp.sum(dqw1_1_sub * qf, axis=1)
        ddd = jnp.sum(dqhat_sub * qf, axis=1)
        dq_sub = (
            dqhat_sub * wpre[4 * H : 5 * H][:, None, :]
            + dqw1_0_sub * wpre[0:H][:, None, :]
            + dqw1_1_sub * wpre[H : 2 * H][:, None, :]
        )
        dq_s_ref[...] = dq_s_ref[...] + dq_sub
        dwpre_part_ref[...] = dwpre_part_ref[...] + jnp.concatenate([dw1_0, dw1_1, dw2_0, dw2_1, ddd], axis=0)

      if num_bwd_sub == 1:
        sub_step(0, None)
      else:
        lax.fori_loop(0, num_bwd_sub, sub_step, None)
      dq_part_ref[...] = dq_s_ref[...].astype(dq_part_ref.dtype)

    @pl.when(step_idx == nq - 1)
    def _done():
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

  @jax.custom_vjp
  def _core_op(qt, kt, vt, wpre, wpost, seg_ids):
    out_t, _ = _fwd_impl(qt, kt, vt, wpre, wpost, seg_ids)
    return out_t

  def _core_fwd_vjp(qt, kt, vt, wpre, wpost, seg_ids):
    out_t, lse = _fwd_impl(qt, kt, vt, wpre, wpost, seg_ids)
    return out_t, (qt, kt, vt, wpre, wpost, lse, seg_ids)

  def _core_bwd_vjp(res, dot):
    qt, kt, vt, wpre, wpost, lse, seg_ids = res
    B, _, _, T = qt.shape
    nq, nkv = T // bwd_bq, T // bwd_bkv
    kv_idx_np, q_run_mask_np, _ = _q_major_schedule(T, bwd_bq, bwd_bkv)
    q_idx_np, kv_run_mask_np = _kv_major_schedule(T, bwd_bq, bwd_bkv)
    kv_idx_arr, q_run_mask = jnp.asarray(kv_idx_np), jnp.asarray(q_run_mask_np)
    q_idx_arr, kv_run_mask = jnp.asarray(q_idx_np), jnp.asarray(kv_run_mask_np)
    q_seg, k_seg = _prepare_seg_arrays(seg_ids, B, T)

    q_spec = pl.BlockSpec((None, H, D, bwd_bq), lambda b, i, j, *_: (b, 0, 0, i))
    k_spec = pl.BlockSpec((None, H, D, bwd_bkv), lambda b, i, j, kv_ref, _: (b, 0, 0, kv_ref[i, j]))
    w_spec = pl.BlockSpec((None, W_ROWS, bwd_bq), lambda b, i, j, *_: (b, 0, i))
    lse_spec = pl.BlockSpec((None, H, bwd_bq), lambda b, i, j, *_: (b, 0, i))
    q_seg_spec = pl.BlockSpec((None, 8, bwd_bq), lambda b, i, j, *_: (b, 0, i)) if use_segment_ids else None
    k_seg_spec = (
        pl.BlockSpec((None, bwd_bkv, 128), lambda b, i, j, kv_ref, _: (b, kv_ref[i, j], 0))
        if use_segment_ids
        else None
    )

    delta, dwpost = pl.pallas_call(
        _bwd_pass1_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[q_spec, k_spec, k_spec, q_spec, w_spec, w_spec, lse_spec, q_seg_spec, k_seg_spec],
            out_specs=[lse_spec, w_spec],
            scratch_shapes=[pltpu.VMEM((H, bwd_bq), F32), pltpu.VMEM((W_ROWS, bwd_bq), F32)],
            grid=(B, nq, nkv),
        ),
        out_shape=[jax.ShapeDtypeStruct((B, H, T), F32), jax.ShapeDtypeStruct((B, W_ROWS, T), F32)],
        compiler_params=cparams,
        name="dc2_flash_bwd_delta",
    )(kv_idx_arr, q_run_mask, qt, kt, vt, dot, wpre, wpost, lse, q_seg, k_seg)

    q_spec_kv = pl.BlockSpec((None, H, D, bwd_bq), lambda b, j, i, q_ref, _: (b, 0, 0, q_ref[j, i]))
    k_spec_kv = pl.BlockSpec((None, H, D, bwd_bkv), lambda b, j, i, *_: (b, 0, 0, j))
    w_spec_kv = pl.BlockSpec((None, W_ROWS, bwd_bq), lambda b, j, i, q_ref, _: (b, 0, q_ref[j, i]))
    lse_spec_kv = pl.BlockSpec((None, H, bwd_bq), lambda b, j, i, q_ref, _: (b, 0, q_ref[j, i]))
    q_seg_kv = pl.BlockSpec((None, 8, bwd_bq), lambda b, j, i, q_ref, _: (b, 0, q_ref[j, i])) if use_segment_ids else None
    k_seg_kv = pl.BlockSpec((None, bwd_bkv, 128), lambda b, j, i, *_: (b, j, 0)) if use_segment_ids else None

    dq_part, dwpre_part, dkt, dvt = pl.pallas_call(
        _bwd_fused_kv_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                q_spec_kv, k_spec_kv, k_spec_kv, q_spec_kv,
                w_spec_kv, w_spec_kv, lse_spec_kv, lse_spec_kv,
                q_seg_kv, k_seg_kv,
            ],
            out_specs=[
                pl.BlockSpec((None, None, H, D, bwd_bq), lambda b, j, i, *_: (j, b, 0, 0, i)),
                pl.BlockSpec((None, None, W_ROWS, bwd_bq), lambda b, j, i, *_: (j, b, 0, i)),
                k_spec_kv,
                k_spec_kv,
            ],
            scratch_shapes=[
                pltpu.VMEM((H, D, bwd_bkv), F32),
                pltpu.VMEM((H, D, bwd_bkv), F32),
                pltpu.VMEM((H, D, bwd_bq), F32),
            ],
            grid=(B, nkv, nq),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((nkv, B, H, D, T), qt.dtype),
            jax.ShapeDtypeStruct((nkv, B, W_ROWS, T), F32),
            jax.ShapeDtypeStruct((B, H, D, T), kt.dtype),
            jax.ShapeDtypeStruct((B, H, D, T), vt.dtype),
        ],
        compiler_params=cparams,
        name="dc2_flash_bwd_fused_kv",
    )(q_idx_arr, kv_run_mask, qt, kt, vt, dot, wpre, wpost, lse, delta, q_seg, k_seg)
    dqt = jnp.sum(dq_part.astype(F32), axis=0).astype(qt.dtype)
    dwpre = jnp.sum(dwpre_part, axis=0)
    return dqt, dkt, dvt, dwpre, dwpost, None

  _core_op.defvjp(_core_fwd_vjp, _core_bwd_vjp)
  return _core_op


def dc2_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    pre_q_dw_args: tuple[jax.Array, jax.Array, jax.Array],
    post_q_dw_args: tuple[jax.Array, jax.Array, jax.Array],
    decoder_segment_ids: Optional[jax.Array] = None,
    config: Any = None,
    mesh: Optional[jax.sharding.Mesh] = None,
    dot_prec: Any = None,
) -> jax.Array:
  """Entry point for Two-Pass Global Flash-DCMHA on query/key/value [B, T, H, D]."""
  _, seq_len, num_heads, head_dim = query.shape
  if dot_prec is None and getattr(config, "matmul_precision", "") == "highest":
    dot_prec = lax.Precision.HIGHEST
  wpre = pack_dc2_weights(*pre_q_dw_args)
  wpost = pack_dc2_weights(*post_q_dw_args)

  bq = min(int(getattr(config, "dc_block_q", 512)), seq_len)
  bkv = min(int(getattr(config, "dc_block_kv", 512)), seq_len)
  ckv = min(int(getattr(config, "dc_block_kv_compute", 256)), bkv)
  bwd_bq = min(int(getattr(config, "dc_bwd_block_q", 512)), seq_len)
  bwd_bkv = min(int(getattr(config, "dc_bwd_block_kv", 512)), seq_len)
  bwd_ckv = min(int(getattr(config, "dc_bwd_block_kv_compute", 256)), bwd_bkv)
  vmem_mib = int(getattr(config, "dc_vmem_mib", 96))

  can_use_tpu_pallas = (
      jax.default_backend() == "tpu"
      and seq_len >= 128
      and seq_len % bq == 0
      and seq_len % bkv == 0
      and seq_len % bwd_bq == 0
      and seq_len % bwd_bkv == 0
      and num_heads % 8 == 0
      and head_dim % 8 == 0
  )
  if not can_use_tpu_pallas:
    return _reference_dc2_attention(query, key, value, wpre, wpost, seg_ids=decoder_segment_ids, dot_prec=dot_prec)

  core_fn = _make_tpu_dc2_primitive(
      num_heads=num_heads,
      head_dim=head_dim,
      bq=bq,
      bkv=bkv,
      ckv=ckv,
      bwd_bq=bwd_bq,
      bwd_bkv=bwd_bkv,
      bwd_ckv=bwd_ckv,
      vmem_mib=vmem_mib,
      use_segment_ids=decoder_segment_ids is not None,
      dot_prec=dot_prec,
  )

  def _local_run(q_loc, k_loc, v_loc, wpre_loc, wpost_loc, seg_loc):
    out_t = core_fn(
        q_loc.transpose(0, 2, 3, 1),
        k_loc.transpose(0, 2, 3, 1),
        v_loc.transpose(0, 2, 3, 1),
        wpre_loc,
        wpost_loc,
        seg_loc,
    )
    return out_t.transpose(0, 3, 1, 2)

  if mesh is None:
    return _local_run(query, key, value, wpre, wpost, decoder_segment_ids)

  qkv_spec = nn.logical_to_mesh_axes(("activation_kv_batch", None, None, None))
  w_spec = nn.logical_to_mesh_axes(("activation_batch", None, None))
  seg_spec = nn.logical_to_mesh_axes(("activation_batch", None)) if decoder_segment_ids is not None else None

  return shard_map(
      _local_run,
      mesh=mesh,
      in_specs=(qkv_spec, qkv_spec, qkv_spec, w_spec, w_spec, seg_spec),
      out_specs=qkv_spec,
      check_rep=False,
  )(query, key, value, wpre, wpost, decoder_segment_ids)
