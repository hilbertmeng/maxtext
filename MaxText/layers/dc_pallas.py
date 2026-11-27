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

Key hardware & algebraic optimizations on TPU (v5p / v6e):
  1. Two-pass streaming eliminates all O(T^2) HBM materialization:
     - Primal forward fuses Pass 1 (K-only online-softmax LSE, V DMA pinned/skipped) and Pass 2
       (exact P reconstruction + post-compose + PV) into a single Pallas kernel with stationary Q
       preparation in VMEM.
  2. In-VMEM channel-space MXU factorization:
     - Rank-2 cross-head reductions sum_m pre_w1[q, i, m] * <K_m, Q_m> execute on the (H*D)-wide
       BF16 MXU as K_flat^T @ (pre_w1 * Q)_flat; post-compose rank-2 value sums execute on the MXU
       as V_flat @ g_i.
  3. Exact O(T) backward Delta decomposition via saved pre-post-compose output O_bar:
     - Saving O_bar = sum_k P * V alongside rank-2 summaries G_01 allows outer O(T) differentiation
       of post_w2 and post_dd, and collapses the softmax backward row-dot
         Delta[m, q] = sum_k P[m, k, q] * dP[m, k, q]
                     = <O_bar[m, :, q], dO_bar[m, :, q]>
                       + sum_{i=0}^{1} post_w1[i, m, q] * dw1_post[i, m, q],
       eliminating the 16-head V^T @ dO matmul and 8 of 10 VPU reductions from Backward Pass 1.
  4. Optional residual checkpointing (`bam_splash_residual_name`) preserves (O_bar, G_01, LSE) across
     layer rematerialization, matching SplashAttention's `residual_checkpoint_name` contract.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

from flax import linen as nn
import jax
from jax import ad_checkpoint
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
  _, t, h, _ = q.shape
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
def _fused_fwd_schedule(seq_len: int, b_q: int, b_kv: int, num_sub: int):
  nq, nkv = seq_len // b_q, seq_len // b_kv
  width = 2 * nkv
  k_idx = np.zeros((nq, width), dtype=np.int32)
  v_idx = np.zeros((nq, width), dtype=np.int32)
  mode = np.zeros((nq, width), dtype=np.int32)
  for i in range(nq):
    q_start, q_end = i * b_q, (i + 1) * b_q - 1
    cnt = min(nkv, q_end // b_kv + 1)
    if cnt == 1 and num_sub == 1:
      k_idx[i, :] = 0
      v_idx[i, :] = 0
      mode[i, 0] = 5
    else:
      for j in range(cnt):
        k_end = (j + 1) * b_kv - 1
        k_idx[i, j] = j
        v_idx[i, j] = 0
        mode[i, j] = 2 if k_end <= q_start else 1
        k_idx[i, cnt + j] = j
        v_idx[i, cnt + j] = j
        mode[i, cnt + j] = 4 if k_end <= q_start else 3
      for s in range(2 * cnt, width):
        k_idx[i, s] = cnt - 1
        v_idx[i, s] = cnt - 1
        mode[i, s] = 0
  return k_idx, v_idx, mode


@functools.lru_cache(maxsize=32)
def _q_major_schedule(seq_len: int, b_q: int, b_kv: int):
  nq, nkv = seq_len // b_q, seq_len // b_kv
  kv_idx = np.zeros((nq, nkv), dtype=np.int32)
  run_mask = np.zeros((nq, nkv), dtype=np.int32)
  for i in range(nq):
    q_start, q_end = i * b_q, (i + 1) * b_q - 1
    last_valid = min(nkv - 1, q_end // b_kv)
    for j in range(nkv):
      k_end = (j + 1) * b_kv - 1
      if j * b_kv <= q_end:
        kv_idx[i, j] = j
        run_mask[i, j] = 2 if k_end <= q_start else 1
      else:
        kv_idx[i, j] = last_valid
        run_mask[i, j] = 0
  return kv_idx, run_mask


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
    valid = valid & (q_seg == k_seg)
  return valid


@functools.lru_cache(maxsize=16)
def _make_tpu_dc2_kernels(
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
  """Builds local single-shard Pallas callables for primal fwd, VJP fwd, and VJP bwd."""
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
  W1_ROWS = 2 * H
  num_fwd_sub = bkv // ckv
  num_bwd_sub = bwd_bkv // bwd_ckv
  cparams = pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "arbitrary"),
      vmem_limit_bytes=int(vmem_mib) * 1024 * 1024,
  )

  def _dot(lhs, rhs, dims):
    return lax.dot_general(lhs, rhs, dims, precision=dot_prec, preferred_element_type=F32)

  def _scores(k_sub, qhat, qw1_0, qw1_1, wpre):
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
    q_s = q_seg_ref[:1, :]
    k_slice = k_seg_ref[pl.ds(sub * step_ckv, step_ckv), :]
    k_s = pltpu.repeat(k_slice, target_bq // 128, axis=1)
    return q_s, k_s

  def _prepare_seg_arrays(seg_ids, b_dim, t_dim):
    if not use_segment_ids:
      return None, None
    q_seg = lax.broadcast_in_dim(seg_ids, (b_dim, 8, t_dim), (0, 2))
    k_seg = lax.broadcast_in_dim(seg_ids, (b_dim, t_dim, 128), (0, 1))
    return q_seg, k_seg

  # 1. Primal Fused Two-Pass Forward (outputs O_bar and G_01 in ONE kernel launch!)
  def _fwd_fused_obar_kernel(
      k_idx_ref, v_idx_ref, mode_ref,
      q_ref, k_ref, v_ref, wpre_ref, w1post_ref, q_seg_ref, k_seg_ref,
      obar_ref, g01_ref, lse_ref,
      m_s_ref, l_s_ref, obar_s_ref, g0_s_ref, g1_s_ref, qhat_s_ref, qw0_s_ref, qw1_s_ref,
  ):
    q_idx, step_idx = pl.program_id(1), pl.program_id(2)
    width = mode_ref.shape[1]

    @pl.when(step_idx == 0)
    def _init():
      m_s_ref[...] = jnp.full_like(m_s_ref, DEFAULT_MASK_VALUE)
      l_s_ref[...] = jnp.zeros_like(l_s_ref)
      obar_s_ref[...] = jnp.zeros_like(obar_s_ref)
      g0_s_ref[...] = jnp.zeros_like(g0_s_ref)
      g1_s_ref[...] = jnp.zeros_like(g1_s_ref)
      qf = q_ref[...].astype(F32)
      wpre = wpre_ref[...]
      qhat_s_ref[...] = (qf * wpre[4 * H : 5 * H][:, None, :]).astype(q_ref.dtype)
      qw0_s_ref[...] = (qf * wpre[0:H][:, None, :]).reshape(HD, bq).astype(q_ref.dtype)
      qw1_s_ref[...] = (qf * wpre[H : 2 * H][:, None, :]).reshape(HD, bq).astype(q_ref.dtype)

    md = mode_ref[q_idx, step_idx]
    kv_idx = k_idx_ref[q_idx, step_idx]

    @pl.when((md == 1) | (md == 2))
    def _pass1():
      wpre = wpre_ref[...]
      qhat, qw0, qw1 = qhat_s_ref[...], qw0_s_ref[...], qw1_s_ref[...]

      def sub_p1(sub, _):
        k_sub = k_ref[:, :, pl.ds(sub * ckv, ckv)]
        s_tilde, _, _, _ = _scores(k_sub, qhat, qw0, qw1, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, ckv, bq)
        valid = _sub_valid(md, q_idx * bq, kv_idx * bkv + sub * ckv, bq, ckv, q_s, k_s)
        s_tilde = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        m_prev = m_s_ref[...]
        m_next = jnp.maximum(m_prev, jnp.max(s_tilde, axis=1))
        l_s_ref[...] = jnp.exp(m_prev - m_next) * l_s_ref[...] + jnp.sum(
            jnp.exp(s_tilde - m_next[:, None, :]), axis=1
        )
        m_s_ref[...] = m_next

      if num_fwd_sub == 1:
        sub_p1(0, None)
      else:
        lax.fori_loop(0, num_fwd_sub, sub_p1, None)

    @pl.when(md >= 3)
    def _pass2():
      wpre, w1post = wpre_ref[...], w1post_ref[...]
      qhat, qw0, qw1 = qhat_s_ref[...], qw0_s_ref[...], qw1_s_ref[...]
      mask_mode = jnp.where((md == 3) | (md == 5), 1, 2)

      def sub_p2(sub, _):
        sl = pl.ds(sub * ckv, ckv)
        k_sub, v_sub = k_ref[:, :, sl], v_ref[:, :, sl]
        v_flat = v_sub.reshape(HD, ckv)
        s_tilde, _, _, _ = _scores(k_sub, qhat, qw0, qw1, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, ckv, bq)
        valid = _sub_valid(mask_mode, q_idx * bq, kv_idx * bkv + sub * ckv, bq, ckv, q_s, k_s)
        s_masked = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        if num_fwd_sub == 1:
          m_loc = jnp.max(s_masked, axis=1)
          l_loc = jnp.sum(jnp.exp(s_masked - m_loc[:, None, :]), axis=1)
          m_use = jnp.where(md == 5, m_loc, m_s_ref[...])
          l_use = jnp.where(md == 5, l_loc, l_s_ref[...])
          m_s_ref[...], l_s_ref[...] = m_use, l_use
          lse = m_use + jnp.log(l_use)
        else:
          lse = m_s_ref[...] + jnp.log(l_s_ref[...])
        p = jnp.where(valid[None, :, :], jnp.exp(s_masked - lse[:, None, :]), 0.0)
        obar_s_ref[...] = obar_s_ref[...] + _dot(v_sub, p.astype(v_sub.dtype), (((2,), (1,)), ((0,), (0,))))
        g0 = jnp.sum(p * w1post[0:H][:, None, :], axis=0)
        g1 = jnp.sum(p * w1post[H : 2 * H][:, None, :], axis=0)
        g0_s_ref[...] = g0_s_ref[...] + _dot(v_flat, g0.astype(v_sub.dtype), (((1,), (0,)), ((), ())))
        g1_s_ref[...] = g1_s_ref[...] + _dot(v_flat, g1.astype(v_sub.dtype), (((1,), (0,)), ((), ())))

      if num_fwd_sub == 1:
        sub_p2(0, None)
      else:
        lax.fori_loop(0, num_fwd_sub, sub_p2, None)

    @pl.when(step_idx == width - 1)
    def _done():
      obar_ref[...] = obar_s_ref[...].astype(obar_ref.dtype)
      g01_ref[...] = jnp.stack([g0_s_ref[...], g1_s_ref[...]], axis=0).astype(g01_ref.dtype)
      lse_ref[...] = m_s_ref[...] + jnp.log(l_s_ref[...])

  def fwd_local(qt, kt, vt, wpre, w1post, seg_ids):
    B, _, _, T = qt.shape
    nq, nkv = T // bq, T // bkv
    k_idx_np, v_idx_np, mode_np = _fused_fwd_schedule(T, bq, bkv, num_fwd_sub)
    k_idx_arr, v_idx_arr, mode_arr = jnp.asarray(k_idx_np), jnp.asarray(v_idx_np), jnp.asarray(mode_np)
    q_seg, k_seg = _prepare_seg_arrays(seg_ids, B, T)

    q_spec = pl.BlockSpec((None, H, D, bq), lambda b, i, s, *_: (b, 0, 0, i))
    k_spec = pl.BlockSpec((None, H, D, bkv), lambda b, i, s, kr, vr, mr: (b, 0, 0, kr[i, s]))
    v_spec = pl.BlockSpec((None, H, D, bkv), lambda b, i, s, kr, vr, mr: (b, 0, 0, vr[i, s]))
    w_spec = pl.BlockSpec((None, W_ROWS, bq), lambda b, i, s, *_: (b, 0, i))
    w1_spec = pl.BlockSpec((None, W1_ROWS, bq), lambda b, i, s, *_: (b, 0, i))
    g_spec = pl.BlockSpec((None, 2, HD, bq), lambda b, i, s, *_: (b, 0, 0, i))
    lse_spec = pl.BlockSpec((None, H, bq), lambda b, i, s, *_: (b, 0, i))
    q_seg_spec = pl.BlockSpec((None, 8, bq), lambda b, i, s, *_: (b, 0, i)) if use_segment_ids else None
    k_seg_spec = (
        pl.BlockSpec((None, bkv, 128), lambda b, i, s, kr, vr, mr: (b, kr[i, s], 0))
        if use_segment_ids
        else None
    )

    return pl.pallas_call(
        _fwd_fused_obar_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[q_spec, k_spec, v_spec, w_spec, w1_spec, q_seg_spec, k_seg_spec],
            out_specs=[q_spec, g_spec, lse_spec],
            scratch_shapes=[
                pltpu.VMEM((H, bq), F32),
                pltpu.VMEM((H, bq), F32),
                pltpu.VMEM((H, D, bq), F32),
                pltpu.VMEM((HD, bq), F32),
                pltpu.VMEM((HD, bq), F32),
                pltpu.VMEM((H, D, bq), qt.dtype),
                pltpu.VMEM((HD, bq), qt.dtype),
                pltpu.VMEM((HD, bq), qt.dtype),
            ],
            grid=(B, nq, 2 * nkv),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, H, D, T), qt.dtype),
            jax.ShapeDtypeStruct((B, 2, HD, T), qt.dtype),
            jax.ShapeDtypeStruct((B, H, T), F32),
        ],
        compiler_params=cparams,
        name="dc2_flash_fwd_fused",
    )(k_idx_arr, v_idx_arr, mode_arr, qt, kt, vt, wpre, w1post, q_seg, k_seg)

  # 2. Ultra-Lean Backward Pass 1: computes ONLY dw1post (zero 16-head V^T @ dO matmul!)
  def _bwd_lean_dw1_kernel(
      kv_idx_ref, run_mask_ref,
      q_ref, k_ref, v_ref, dg01_ref, wpre_ref, lse_ref, q_seg_ref, k_seg_ref,
      dw1_ref, dw1_s_ref, qhat_s_ref, qw0_s_ref, qw1_s_ref,
  ):
    q_idx, step_idx = pl.program_id(1), pl.program_id(2)
    nkv = run_mask_ref.shape[1]

    @pl.when(step_idx == 0)
    def _init():
      dw1_s_ref[...] = jnp.zeros_like(dw1_s_ref)
      qf = q_ref[...].astype(F32)
      wpre = wpre_ref[...]
      qhat_s_ref[...] = (qf * wpre[4 * H : 5 * H][:, None, :]).astype(q_ref.dtype)
      qw0_s_ref[...] = (qf * wpre[0:H][:, None, :]).reshape(HD, bwd_bq).astype(q_ref.dtype)
      qw1_s_ref[...] = (qf * wpre[H : 2 * H][:, None, :]).reshape(HD, bwd_bq).astype(q_ref.dtype)

    mode = run_mask_ref[q_idx, step_idx]
    kv_idx = kv_idx_ref[q_idx, step_idx]

    @pl.when(mode > 0)
    def _body():
      wpre, lse, dg01 = wpre_ref[...], lse_ref[...], dg01_ref[...]
      qhat, qw0, qw1 = qhat_s_ref[...], qw0_s_ref[...], qw1_s_ref[...]

      def sub_step(sub, _):
        sl = pl.ds(sub * bwd_ckv, bwd_ckv)
        k_sub, v_sub = k_ref[:, :, sl], v_ref[:, :, sl]
        v_flat = v_sub.reshape(HD, bwd_ckv)
        s_tilde, _, _, _ = _scores(k_sub, qhat, qw0, qw1, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, bwd_ckv, bwd_bq)
        valid = _sub_valid(mode, q_idx * bwd_bq, kv_idx * bwd_bkv + sub * bwd_ckv, bwd_bq, bwd_ckv, q_s, k_s)
        s_masked = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        p = jnp.where(valid[None, :, :], jnp.exp(s_masked - lse[:, None, :]), 0.0)
        dg0 = jnp.where(valid, _dot(v_flat, dg01[0], (((0,), (0,)), ((), ()))), 0.0)
        dg1 = jnp.where(valid, _dot(v_flat, dg01[1], (((0,), (0,)), ((), ()))), 0.0)
        dw1_0 = jnp.sum(dg0[None, :, :] * p, axis=1)
        dw1_1 = jnp.sum(dg1[None, :, :] * p, axis=1)
        dw1_s_ref[...] = dw1_s_ref[...] + jnp.concatenate([dw1_0, dw1_1], axis=0)

      if num_bwd_sub == 1:
        sub_step(0, None)
      else:
        lax.fori_loop(0, num_bwd_sub, sub_step, None)

    @pl.when(step_idx == nkv - 1)
    def _done():
      dw1_ref[...] = dw1_s_ref[...]

  # 3. Fused KV-Major Backward Pass 2: computes dQ, dK, dV, dwpre
  def _bwd_fused_kv_kernel(
      q_idx_ref, run_mask_ref,
      q_ref, k_ref, v_ref, dobar_ref, dg01_ref, wpre_ref, w1post_ref, lse_ref, delta_ref, q_seg_ref, k_seg_ref,
      dq_part_ref, dwpre_part_ref, dk_ref, dv_ref,
      dk_s_ref, dv_s_ref, dq_s_ref,
  ):
    kv_idx, step_idx = pl.program_id(1), pl.program_id(2)
    nq = run_mask_ref.shape[1]

    @pl.when(step_idx == 0)
    def _init():
      dk_s_ref[...] = jnp.zeros_like(dk_s_ref)
      dv_s_ref[...] = jnp.zeros_like(dv_s_ref)

    mode = run_mask_ref[kv_idx, step_idx]
    q_idx = q_idx_ref[kv_idx, step_idx]

    @pl.when(mode == 0)
    def _zero():
      dq_part_ref[...] = jnp.zeros_like(dq_part_ref)
      dwpre_part_ref[...] = jnp.zeros_like(dwpre_part_ref)

    @pl.when(mode > 0)
    def _body():
      q, dobar, dg01 = q_ref[...], dobar_ref[...], dg01_ref[...]
      qf = q.astype(F32)
      wpre, w1post, lse, delta = wpre_ref[...], w1post_ref[...], lse_ref[...], delta_ref[...]
      qhat = (qf * wpre[4 * H : 5 * H][:, None, :]).astype(q.dtype)
      qw0 = (qf * wpre[0:H][:, None, :]).reshape(HD, bwd_bq).astype(q.dtype)
      qw1 = (qf * wpre[H : 2 * H][:, None, :]).reshape(HD, bwd_bq).astype(q.dtype)

      dq_s_ref[...] = jnp.zeros_like(dq_s_ref)
      dwpre_part_ref[...] = jnp.zeros_like(dwpre_part_ref)

      def sub_step(sub, _):
        sl = pl.ds(sub * bwd_ckv, bwd_ckv)
        k_sub, v_sub = k_ref[:, :, sl], v_ref[:, :, sl]
        v_flat = v_sub.reshape(HD, bwd_ckv)
        s_tilde, h0, h1, k_flat = _scores(k_sub, qhat, qw0, qw1, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, bwd_ckv, bwd_bq)
        valid = _sub_valid(mode, q_idx * bwd_bq, kv_idx * bwd_bkv + sub * bwd_ckv, bwd_bq, bwd_ckv, q_s, k_s)
        s_masked = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        p = jnp.where(valid[None, :, :], jnp.exp(s_masked - lse[:, None, :]), 0.0)

        g0 = jnp.sum(p * w1post[0:H][:, None, :], axis=0)
        g1 = jnp.sum(p * w1post[H : 2 * H][:, None, :], axis=0)
        dv_bar = _dot(dobar, p.astype(dobar.dtype), (((2,), (2,)), ((0,), (0,))))
        dv_g = (
            _dot(dg01[0], g0.astype(dg01.dtype), (((1,), (1,)), ((), ())))
            + _dot(dg01[1], g1.astype(dg01.dtype), (((1,), (1,)), ((), ())))
        ).reshape(H, D, bwd_ckv)
        dv_s_ref[:, :, sl] = dv_s_ref[:, :, sl] + dv_bar + dv_g

        dp_bar = _dot(v_sub, dobar, (((1,), (1,)), ((0,), (0,))))
        dg0 = jnp.where(valid, _dot(v_flat, dg01[0], (((0,), (0,)), ((), ()))), 0.0)
        dg1 = jnp.where(valid, _dot(v_flat, dg01[1], (((0,), (0,)), ((), ()))), 0.0)
        dp = (
            jnp.where(valid[None, :, :], dp_bar, 0.0)
            + w1post[0:H][:, None, :] * dg0[None, :, :]
            + w1post[H : 2 * H][:, None, :] * dg1[None, :, :]
        )
        ds_tilde = p * (dp - delta[:, None, :])

        dw2_0 = jnp.sum(ds_tilde * h0[None, :, :], axis=1)
        dw2_1 = jnp.sum(ds_tilde * h1[None, :, :], axis=1)
        dh0 = jnp.sum(ds_tilde * wpre[2 * H : 3 * H][:, None, :], axis=0)
        dh1 = jnp.sum(ds_tilde * wpre[3 * H : 4 * H][:, None, :], axis=0)

        dk_bar = _dot(qhat, ds_tilde.astype(qhat.dtype), (((2,), (2,)), ((0,), (0,))))
        dk_h = (
            _dot(qw0, dh0.astype(qw0.dtype), (((1,), (1,)), ((), ())))
            + _dot(qw1, dh1.astype(qw1.dtype), (((1,), (1,)), ((), ())))
        ).reshape(H, D, bwd_ckv)
        dk_s_ref[:, :, sl] = dk_s_ref[:, :, sl] + dk_bar + dk_h

        dqhat_sub = _dot(k_sub, ds_tilde.astype(k_sub.dtype), (((2,), (1,)), ((0,), (0,))))
        dqw0_sub = _dot(k_flat, dh0.astype(k_sub.dtype), (((1,), (0,)), ((), ()))).reshape(H, D, bwd_bq)
        dqw1_sub = _dot(k_flat, dh1.astype(k_sub.dtype), (((1,), (0,)), ((), ()))).reshape(H, D, bwd_bq)

        dw1_0 = jnp.sum(dqw0_sub * qf, axis=1)
        dw1_1 = jnp.sum(dqw1_sub * qf, axis=1)
        ddd = jnp.sum(dqhat_sub * qf, axis=1)
        dq_sub = (
            dqhat_sub * wpre[4 * H : 5 * H][:, None, :]
            + dqw0_sub * wpre[0:H][:, None, :]
            + dqw1_sub * wpre[H : 2 * H][:, None, :]
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
      dk_ref[...] = dk_s_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_s_ref[...].astype(dv_ref.dtype)

  def bwd_local(qt, kt, vt, wpre, w1post, obar_t, lse, seg_ids, dobar_t, dg01_t):
    B, _, _, T = qt.shape
    nq, nkv = T // bwd_bq, T // bwd_bkv
    kv_idx_np, q_run_mask_np = _q_major_schedule(T, bwd_bq, bwd_bkv)
    q_idx_np, kv_run_mask_np = _kv_major_schedule(T, bwd_bq, bwd_bkv)
    kv_idx_arr, q_run_mask = jnp.asarray(kv_idx_np), jnp.asarray(q_run_mask_np)
    q_idx_arr, kv_run_mask = jnp.asarray(q_idx_np), jnp.asarray(kv_run_mask_np)
    q_seg, k_seg = _prepare_seg_arrays(seg_ids, B, T)

    q_spec = pl.BlockSpec((None, H, D, bwd_bq), lambda b, i, j, *_: (b, 0, 0, i))
    k_spec = pl.BlockSpec((None, H, D, bwd_bkv), lambda b, i, j, kv_ref, _: (b, 0, 0, kv_ref[i, j]))
    w_spec = pl.BlockSpec((None, W_ROWS, bwd_bq), lambda b, i, j, *_: (b, 0, i))
    w1_spec = pl.BlockSpec((None, W1_ROWS, bwd_bq), lambda b, i, j, *_: (b, 0, i))
    g_spec = pl.BlockSpec((None, 2, HD, bwd_bq), lambda b, i, j, *_: (b, 0, 0, i))
    lse_spec = pl.BlockSpec((None, H, bwd_bq), lambda b, i, j, *_: (b, 0, i))
    q_seg_spec = pl.BlockSpec((None, 8, bwd_bq), lambda b, i, j, *_: (b, 0, i)) if use_segment_ids else None
    k_seg_spec = (
        pl.BlockSpec((None, bwd_bkv, 128), lambda b, i, j, kv_ref, _: (b, kv_ref[i, j], 0))
        if use_segment_ids
        else None
    )

    dw1post = pl.pallas_call(
        _bwd_lean_dw1_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[q_spec, k_spec, k_spec, g_spec, w_spec, lse_spec, q_seg_spec, k_seg_spec],
            out_specs=w1_spec,
            scratch_shapes=[
                pltpu.VMEM((W1_ROWS, bwd_bq), F32),
                pltpu.VMEM((H, D, bwd_bq), qt.dtype),
                pltpu.VMEM((HD, bwd_bq), qt.dtype),
                pltpu.VMEM((HD, bwd_bq), qt.dtype),
            ],
            grid=(B, nq, nkv),
        ),
        out_shape=jax.ShapeDtypeStruct((B, W1_ROWS, T), F32),
        compiler_params=cparams,
        name="dc2_flash_bwd_dw1",
    )(kv_idx_arr, q_run_mask, qt, kt, vt, dg01_t, wpre, lse, q_seg, k_seg)

    delta = (
        jnp.sum(obar_t.astype(F32) * dobar_t.astype(F32), axis=2)
        + w1post[:, 0:H, :] * dw1post[:, 0:H, :]
        + w1post[:, H : 2 * H, :] * dw1post[:, H : 2 * H, :]
    )

    q_spec_kv = pl.BlockSpec((None, H, D, bwd_bq), lambda b, j, i, qr, _: (b, 0, 0, qr[j, i]))
    g_spec_kv = pl.BlockSpec((None, 2, HD, bwd_bq), lambda b, j, i, qr, _: (b, 0, 0, qr[j, i]))
    k_spec_kv = pl.BlockSpec((None, H, D, bwd_bkv), lambda b, j, i, *_: (b, 0, 0, j))
    w_spec_kv = pl.BlockSpec((None, W_ROWS, bwd_bq), lambda b, j, i, qr, _: (b, 0, qr[j, i]))
    w1_spec_kv = pl.BlockSpec((None, W1_ROWS, bwd_bq), lambda b, j, i, qr, _: (b, 0, qr[j, i]))
    lse_spec_kv = pl.BlockSpec((None, H, bwd_bq), lambda b, j, i, qr, _: (b, 0, qr[j, i]))
    q_seg_kv = pl.BlockSpec((None, 8, bwd_bq), lambda b, j, i, qr, _: (b, 0, qr[j, i])) if use_segment_ids else None
    k_seg_kv = pl.BlockSpec((None, bwd_bkv, 128), lambda b, j, i, *_: (b, j, 0)) if use_segment_ids else None

    dq_part, dwpre_part, dkt, dvt = pl.pallas_call(
        _bwd_fused_kv_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                q_spec_kv, k_spec_kv, k_spec_kv, q_spec_kv, g_spec_kv,
                w_spec_kv, w1_spec_kv, lse_spec_kv, lse_spec_kv,
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
    )(q_idx_arr, kv_run_mask, qt, kt, vt, dobar_t, dg01_t, wpre, w1post, lse, delta, q_seg, k_seg)
    dqt = jnp.sum(dq_part.astype(F32), axis=0).astype(qt.dtype)
    dwpre = jnp.sum(dwpre_part, axis=0)
    return dqt, dkt, dvt, dwpre, dw1post

  return fwd_local, bwd_local


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
  bsz, seq_len, num_heads, head_dim = query.shape
  if dot_prec is None and getattr(config, "matmul_precision", "") == "highest":
    dot_prec = lax.Precision.HIGHEST

  wpre = pack_dc2_weights(*pre_q_dw_args)
  post_qw1, post_qw2, post_qdd = post_q_dw_args

  bq = min(int(getattr(config, "dc_block_q", 512)), seq_len)
  bkv = min(int(getattr(config, "dc_block_kv", 512)), seq_len)
  ckv = min(int(getattr(config, "dc_block_kv_compute", 256)), bkv)
  bwd_bq = min(int(getattr(config, "dc_bwd_block_q", 512)), seq_len)
  bwd_bkv = min(int(getattr(config, "dc_bwd_block_kv", 512)), seq_len)
  bwd_ckv = min(int(getattr(config, "dc_bwd_block_kv_compute", 256)), bwd_bkv)
  vmem_mib = int(getattr(config, "dc_vmem_mib", 96))
  residual_name = getattr(config, "bam_splash_residual_name", None) or None

  is_tpu_target = (
      jax.default_backend() == "tpu"
      or (mesh is not None and getattr(mesh.devices.flat[0], "platform", "") == "tpu")
      or bool(getattr(config, "compile_topology", ""))
  )
  can_use_tpu_pallas = (
      is_tpu_target
      and seq_len >= 128
      and seq_len % bq == 0
      and seq_len % bkv == 0
      and seq_len % bwd_bq == 0
      and seq_len % bwd_bkv == 0
      and num_heads % 8 == 0
      and head_dim % 8 == 0
  )
  if not can_use_tpu_pallas:
    wpost = pack_dc2_weights(*post_q_dw_args)
    return _reference_dc2_attention(query, key, value, wpre, wpost, seg_ids=decoder_segment_ids, dot_prec=dot_prec)

  fwd_local, bwd_local = _make_tpu_dc2_kernels(
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

  w1post = post_qw1[:, :, 0, :, :].astype(F32).transpose(0, 2, 3, 1).reshape(bsz, 2 * num_heads, seq_len)
  qt_in = query.transpose(0, 2, 3, 1)
  kt_in = key.transpose(0, 2, 3, 1)
  vt_in = value.transpose(0, 2, 3, 1)

  if mesh is None:
    def _run_fwd_sharded(qt, kt, vt, wp, w1p, seg):
      return fwd_local(qt, kt, vt, wp, w1p, seg)
    def _run_bwd_sharded(qt, kt, vt, wp, w1p, obar_t, lse, seg, dobar_t, dg01_t):
      return bwd_local(qt, kt, vt, wp, w1p, obar_t, lse, seg, dobar_t, dg01_t)
  else:
    q_t_spec = nn.logical_to_mesh_axes(("activation_kv_batch", None, None, None))
    g_t_spec = nn.logical_to_mesh_axes(("activation_kv_batch", None, None, None))
    w_spec = nn.logical_to_mesh_axes(("activation_batch", None, None))
    lse_spec = nn.logical_to_mesh_axes(("activation_batch", None, None))
    seg_spec = nn.logical_to_mesh_axes(("activation_batch", None)) if decoder_segment_ids is not None else None

    _run_fwd_sharded = shard_map(
        fwd_local,
        mesh=mesh,
        in_specs=(q_t_spec, q_t_spec, q_t_spec, w_spec, w_spec, seg_spec),
        out_specs=(q_t_spec, g_t_spec, lse_spec),
        check_rep=False,
    )
    _run_bwd_sharded = shard_map(
        bwd_local,
        mesh=mesh,
        in_specs=(q_t_spec, q_t_spec, q_t_spec, w_spec, w_spec, q_t_spec, lse_spec, seg_spec, q_t_spec, g_t_spec),
        out_specs=(q_t_spec, q_t_spec, q_t_spec, w_spec, w_spec),
        check_rep=False,
    )

  @jax.custom_vjp
  def _sharded_core(qt, kt, vt, wp, w1p, seg):
    obar_t, g01_t, _ = _run_fwd_sharded(qt, kt, vt, wp, w1p, seg)
    return obar_t, g01_t

  def _sharded_fwd_vjp(qt, kt, vt, wp, w1p, seg):
    obar_t, g01_t, lse = _run_fwd_sharded(qt, kt, vt, wp, w1p, seg)
    if residual_name is not None:
      obar_t = ad_checkpoint.checkpoint_name(obar_t, name=residual_name)
      g01_t = ad_checkpoint.checkpoint_name(g01_t, name=residual_name)
      lse = ad_checkpoint.checkpoint_name(lse, name=residual_name)
    return (obar_t, g01_t), (qt, kt, vt, wp, w1p, obar_t, lse, seg)

  def _sharded_bwd_vjp(res, cts):
    qt, kt, vt, wp, w1p, obar_t, lse, seg = res
    dobar_t, dg01_t = cts
    dqt, dkt, dvt, dwp, dw1p = _run_bwd_sharded(qt, kt, vt, wp, w1p, obar_t, lse, seg, dobar_t, dg01_t)
    return dqt, dkt, dvt, dwp, dw1p, None

  _sharded_core.defvjp(_sharded_fwd_vjp, _sharded_bwd_vjp)

  obar_t, g01_t = _sharded_core(qt_in, kt_in, vt_in, wpre, w1post, decoder_segment_ids)
  obar = obar_t.transpose(0, 3, 1, 2).astype(F32)
  g01 = g01_t.transpose(0, 3, 1, 2).reshape(bsz, seq_len, 2, num_heads, head_dim).astype(F32)
  out = (
      obar * (1.0 + post_qdd[:, :, 0, :].astype(F32))[:, :, :, None]
      + g01[:, :, 0] * post_qw2[:, :, 0, 0, :, None].astype(F32)
      + g01[:, :, 1] * post_qw2[:, :, 0, 1, :, None].astype(F32)
  )
  return out.astype(value.dtype)
