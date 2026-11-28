"""Unified Two-Pass Global Flash-DCMHA Pallas kernel for (grouped) query-wise DC2.

Supports arbitrary head grouping `G = dc_num_groups` (`M = H // G` heads per group) and arbitrary
within-group dynamic composition rank `R = dc_dynamic_rank` (including the ungrouped `G=1, M=16, R=2`
and grouped `G=4, M=4, R=1` configurations):

  For group g in [0, G), query q, key k (0 <= k <= q), within-group head h in [0, M):
    S[g, m, k, q]       = <K[k, g, m, :], Q[q, g, m, :]>
    S_tilde[g, h, k, q] = (1 + pre_dd[q, g, h]) * S[g, h, k, q]
                          + sum_{r=0}^{R-1} pre_w2[q, g, r, h] * sum_{m=0}^{M-1} pre_w1[q, g, r, m] * S[g, m, k, q]
    P[g, m, k, q]       = softmax_{k <= q}(S_tilde[g, m, k, q])
    P_tilde[g, h, k, q] = (1 + post_dd[q, g, h]) * P[g, h, k, q]
                          + sum_{r=0}^{R-1} post_w2[q, g, r, h] * sum_{m=0}^{M-1} post_w1[q, g, r, m] * P[g, m, k, q]
    Out[q, g, h, :]     = sum_{k <= q} P_tilde[g, h, k, q] * V[k, g, h, :]

Hardware & algebraic optimizations on TPU (v5p / v6e):
  1. Group-parallel 4D Pallas grid `(B, G, N_q, ...)`: when `G=4`, each Pallas instance holds only
     `M=4` heads in VMEM (4x smaller score tiles), unlocking 1024x1024 (`compute=512`) blocks matching
     `SplashTunedBlocks1024` and halving backward partial-dQ HBM traffic (`N_kv: 8 -> 4`).
  2. Within-group `(M*D)`-wide MXU channel factorization + single-kernel fused 2-pass forward.
  3. Exact O(T) backward Delta identity via saved `O_bar`:
       Delta[g, m, q] = <O_bar[g, m, :, q], dO_bar[g, m, :, q]>
                        + sum_{r=0}^{R-1} post_w1[g, r, m, q] * dw1_post[g, r, m, q],
     eliminating the M-head `V^T @ dO` matmul and 80% of VPU reductions from Backward Pass 1.
  4. Optional residual checkpointing (`bam_splash_residual_name`) preserves `(O_bar, G_r, LSE)` across
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


def _pad8(n: int) -> int:
  return ((n + 7) // 8) * 8


def pack_dc2_pre_weights(qw1: jax.Array, qw2: jax.Array, qdd: jax.Array) -> jax.Array:
  """Packs (qw1 [B,T,G,R,M], qw2 [B,T,G,R,M], 1+qdd [B,T,G,M]) into [B, G, W_PAD, T] FP32."""
  b, t, g, r, m = qw1.shape
  w1 = qw1.astype(F32).transpose(0, 2, 3, 4, 1).reshape(b, g, r * m, t)
  w2 = qw2.astype(F32).transpose(0, 2, 3, 4, 1).reshape(b, g, r * m, t)
  d1 = (1.0 + qdd.astype(F32)).transpose(0, 2, 3, 1)  # [B, G, M, T]
  cat = jnp.concatenate([w1, w2, d1], axis=2)        # [B, G, (2R+1)*M, T]
  pad = _pad8(cat.shape[2]) - cat.shape[2]
  return jnp.pad(cat, ((0, 0), (0, 0), (0, pad), (0, 0))) if pad else cat


def pack_dc2_w1post(post_qw1: jax.Array) -> jax.Array:
  """Packs post_qw1 [B,T,G,R,M] into [B, G, W1_PAD, T] FP32."""
  b, t, g, r, m = post_qw1.shape
  w1 = post_qw1.astype(F32).transpose(0, 2, 3, 4, 1).reshape(b, g, r * m, t)
  pad = _pad8(w1.shape[2]) - w1.shape[2]
  return jnp.pad(w1, ((0, 0), (0, 0), (0, pad), (0, 0))) if pad else w1


def _reference_dc2_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    pre_args: tuple[jax.Array, jax.Array, jax.Array],
    post_args: tuple[jax.Array, jax.Array, jax.Array],
    seg_ids: Optional[jax.Array] = None,
    dot_prec: Any = None,
) -> jax.Array:
  """Exact pure-JAX reference for arbitrary (G, R) (used on CPU or when seq_len < block size)."""
  b, t, h, d = q.shape
  g = pre_args[0].shape[2]
  m = h // g
  qg = q.reshape(b, t, g, m, d).astype(F32)
  kg = k.reshape(b, t, g, m, d).astype(F32)
  vg = v.reshape(b, t, g, m, d)
  s = jnp.einsum("btgmd,bsgmd->bgmts", qg, kg, precision=dot_prec, preferred_element_type=F32)

  def _apply_chp(x: jax.Array, args: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
    w1, w2, dd = [a.astype(F32) for a in args]
    hid = jnp.einsum("bgmts,btgim->bgits", x, w1, precision=dot_prec, preferred_element_type=F32)
    out = jnp.einsum("bgits,btgim->bgmts", hid, w2, precision=dot_prec, preferred_element_type=F32)
    return x * (1.0 + dd.transpose(0, 2, 3, 1)[:, :, :, :, None]) + out

  s_tilde = _apply_chp(s, pre_args)
  valid = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))[None, None, None, :, :]
  if seg_ids is not None:
    seg_match = (seg_ids[:, :, None] == seg_ids[:, None, :])[:, None, None, :, :]
    valid = valid & seg_match
  p = jax.nn.softmax(jnp.where(valid, s_tilde, DEFAULT_MASK_VALUE), axis=-1)
  p_tilde = jnp.where(valid, _apply_chp(p, post_args), 0.0).astype(v.dtype)
  out = jnp.einsum("bgmts,bsgmd->btgmd", p_tilde, vg, precision=dot_prec, preferred_element_type=F32)
  return out.reshape(b, t, h, d).astype(v.dtype)


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
    num_groups: int,
    group_heads: int,
    rank: int,
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
  """Builds unified group-parallel Pallas callables for arbitrary (G, M, R, D)."""
  G, M, R, D = num_groups, group_heads, rank, head_dim
  MD = M * D
  if D % 8 != 0 or MD % 128 != 0:
    raise ValueError(f"dc_pallas requires D ({D}) % 8 == 0 and M*D ({MD}) % 128 == 0.")
  if bq % 128 != 0 or ckv % 128 != 0 or bwd_bq % 128 != 0 or bwd_ckv % 128 != 0:
    raise ValueError("dc_pallas block sizes must be multiples of 128.")
  if bkv % ckv != 0 or bwd_bkv % bwd_ckv != 0:
    raise ValueError("block_kv must be a multiple of block_kv_compute.")

  M_PAD = _pad8(M)
  W_ACT = (2 * R + 1) * M
  W_PAD = _pad8(W_ACT)
  W1_ACT = R * M
  W1_PAD = _pad8(W1_ACT)
  num_fwd_sub = bkv // ckv
  num_bwd_sub = bwd_bkv // bwd_ckv
  cparams = pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
      vmem_limit_bytes=int(vmem_mib) * 1024 * 1024,
  )

  def _dot(lhs, rhs, dims):
    return lax.dot_general(lhs, rhs, dims, precision=dot_prec, preferred_element_type=F32)

  def _pad_m(x):
    return jnp.pad(x, ((0, M_PAD - M), (0, 0))) if M_PAD > M else x

  def _scores(k_sub, qhat, qw_all, wpre):
    """k_sub:[M,D,ckv], qhat:[M,D,bq], qw_all:[R,MD,bq], wpre:[W_PAD,bq] -> s_tilde, hs_list, k_flat."""
    k_flat = k_sub.reshape(MD, k_sub.shape[-1])
    s_tilde = _dot(k_sub, qhat, (((1,), (1,)), ((0,), (0,))))
    hs = []
    for r in range(R):
      hr = _dot(k_flat, qw_all[r], (((0,), (0,)), ((), ())))
      hs.append(hr)
      w2_r = wpre[(R + r) * M : (R + r + 1) * M]
      s_tilde = s_tilde + w2_r[:, None, :] * hr[None, :, :]
    return s_tilde, hs, k_flat

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

  # 1. Unified Single-Kernel Fused Two-Pass Forward (grid: B, G, nq, 2*nkv)
  def _fwd_fused_obar_kernel(
      k_idx_ref, v_idx_ref, mode_ref,
      q_ref, k_ref, v_ref, wpre_ref, w1post_ref, q_seg_ref, k_seg_ref,
      obar_ref, gr_ref, lse_ref,
      m_s_ref, l_s_ref, obar_s_ref, gr_s_ref, qhat_s_ref, qw_s_ref,
  ):
    q_idx, step_idx = pl.program_id(2), pl.program_id(3)
    width = mode_ref.shape[1]

    @pl.when(step_idx == 0)
    def _init():
      m_s_ref[...] = jnp.full_like(m_s_ref, DEFAULT_MASK_VALUE)
      l_s_ref[...] = jnp.zeros_like(l_s_ref)
      obar_s_ref[...] = jnp.zeros_like(obar_s_ref)
      gr_s_ref[...] = jnp.zeros_like(gr_s_ref)
      qf = q_ref[...].astype(F32)
      wpre = wpre_ref[...]
      qhat_s_ref[...] = (qf * wpre[2 * R * M : (2 * R + 1) * M][:, None, :]).astype(q_ref.dtype)
      qw_list = [
          (qf * wpre[r * M : (r + 1) * M][:, None, :]).reshape(MD, bq).astype(q_ref.dtype)
          for r in range(R)
      ]
      qw_s_ref[...] = jnp.stack(qw_list, axis=0)

    md = mode_ref[q_idx, step_idx]
    kv_idx = k_idx_ref[q_idx, step_idx]

    @pl.when((md == 1) | (md == 2))
    def _pass1():
      wpre = wpre_ref[...]
      qhat, qw_all = qhat_s_ref[...], qw_s_ref[...]

      def sub_p1(sub, _):
        k_sub = k_ref[:, :, pl.ds(sub * ckv, ckv)]
        s_tilde, _, _ = _scores(k_sub, qhat, qw_all, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, ckv, bq)
        valid = _sub_valid(md, q_idx * bq, kv_idx * bkv + sub * ckv, bq, ckv, q_s, k_s)
        s_tilde = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        m_prev = m_s_ref[...][:M, :]
        m_next = jnp.maximum(m_prev, jnp.max(s_tilde, axis=1))
        l_next = jnp.exp(m_prev - m_next) * l_s_ref[...][:M, :] + jnp.sum(
            jnp.exp(s_tilde - m_next[:, None, :]), axis=1
        )
        m_s_ref[...], l_s_ref[...] = _pad_m(m_next), _pad_m(l_next)

      if num_fwd_sub == 1:
        sub_p1(0, None)
      else:
        lax.fori_loop(0, num_fwd_sub, sub_p1, None)

    @pl.when(md >= 3)
    def _pass2():
      wpre, w1post = wpre_ref[...], w1post_ref[...]
      qhat, qw_all = qhat_s_ref[...], qw_s_ref[...]
      mask_mode = jnp.where((md == 3) | (md == 5), 1, 2)

      def sub_p2(sub, _):
        sl = pl.ds(sub * ckv, ckv)
        k_sub, v_sub = k_ref[:, :, sl], v_ref[:, :, sl]
        v_flat = v_sub.reshape(MD, ckv)
        s_tilde, _, _ = _scores(k_sub, qhat, qw_all, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, ckv, bq)
        valid = _sub_valid(mask_mode, q_idx * bq, kv_idx * bkv + sub * ckv, bq, ckv, q_s, k_s)
        s_masked = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        if num_fwd_sub == 1:
          m_loc = jnp.max(s_masked, axis=1)
          l_loc = jnp.sum(jnp.exp(s_masked - m_loc[:, None, :]), axis=1)
          m_use = jnp.where(md == 5, m_loc, m_s_ref[...][:M, :])
          l_use = jnp.where(md == 5, l_loc, l_s_ref[...][:M, :])
          m_s_ref[...], l_s_ref[...] = _pad_m(m_use), _pad_m(l_use)
          lse = m_use + jnp.log(l_use)
        else:
          lse = m_s_ref[...][:M, :] + jnp.log(l_s_ref[...][:M, :])
        p = jnp.where(valid[None, :, :], jnp.exp(s_masked - lse[:, None, :]), 0.0)
        obar_s_ref[...] = obar_s_ref[...] + _dot(v_sub, p.astype(v_sub.dtype), (((2,), (1,)), ((0,), (0,))))
        gr_outs = []
        for r in range(R):
          gr = jnp.sum(p * w1post[r * M : (r + 1) * M][:, None, :], axis=0)
          gr_outs.append(_dot(v_flat, gr.astype(v_sub.dtype), (((1,), (0,)), ((), ()))))
        gr_s_ref[...] = gr_s_ref[...] + jnp.stack(gr_outs, axis=0)

      if num_fwd_sub == 1:
        sub_p2(0, None)
      else:
        lax.fori_loop(0, num_fwd_sub, sub_p2, None)

    @pl.when(step_idx == width - 1)
    def _done():
      obar_ref[...] = obar_s_ref[...].astype(obar_ref.dtype)
      gr_ref[...] = gr_s_ref[...].astype(gr_ref.dtype)
      lse_ref[...] = _pad_m(m_s_ref[...][:M, :] + jnp.log(l_s_ref[...][:M, :]))

  def fwd_local(qg, kg, vg, wpre, w1post, seg_ids):
    B, _, _, _, T = qg.shape
    nq, nkv = T // bq, T // bkv
    k_idx_np, v_idx_np, mode_np = _fused_fwd_schedule(T, bq, bkv, num_fwd_sub)
    k_idx_arr, v_idx_arr, mode_arr = jnp.asarray(k_idx_np), jnp.asarray(v_idx_np), jnp.asarray(mode_np)
    q_seg, k_seg = _prepare_seg_arrays(seg_ids, B, T)

    q_spec = pl.BlockSpec((None, None, M, D, bq), lambda b, g, i, s, *_: (b, g, 0, 0, i))
    k_spec = pl.BlockSpec((None, None, M, D, bkv), lambda b, g, i, s, kr, vr, mr: (b, g, 0, 0, kr[i, s]))
    v_spec = pl.BlockSpec((None, None, M, D, bkv), lambda b, g, i, s, kr, vr, mr: (b, g, 0, 0, vr[i, s]))
    w_spec = pl.BlockSpec((None, None, W_PAD, bq), lambda b, g, i, s, *_: (b, g, 0, i))
    w1_spec = pl.BlockSpec((None, None, W1_PAD, bq), lambda b, g, i, s, *_: (b, g, 0, i))
    gr_spec = pl.BlockSpec((None, None, R, MD, bq), lambda b, g, i, s, *_: (b, g, 0, 0, i))
    lse_spec = pl.BlockSpec((None, None, M_PAD, bq), lambda b, g, i, s, *_: (b, g, 0, i))
    q_seg_spec = pl.BlockSpec((None, 8, bq), lambda b, g, i, s, *_: (b, 0, i)) if use_segment_ids else None
    k_seg_spec = (
        pl.BlockSpec((None, bkv, 128), lambda b, g, i, s, kr, vr, mr: (b, kr[i, s], 0))
        if use_segment_ids
        else None
    )

    return pl.pallas_call(
        _fwd_fused_obar_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[q_spec, k_spec, v_spec, w_spec, w1_spec, q_seg_spec, k_seg_spec],
            out_specs=[q_spec, gr_spec, lse_spec],
            scratch_shapes=[
                pltpu.VMEM((M_PAD, bq), F32),
                pltpu.VMEM((M_PAD, bq), F32),
                pltpu.VMEM((M, D, bq), F32),
                pltpu.VMEM((R, MD, bq), F32),
                pltpu.VMEM((M, D, bq), qg.dtype),
                pltpu.VMEM((R, MD, bq), qg.dtype),
            ],
            grid=(B, G, nq, 2 * nkv),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, G, M, D, T), qg.dtype),
            jax.ShapeDtypeStruct((B, G, R, MD, T), qg.dtype),
            jax.ShapeDtypeStruct((B, G, M_PAD, T), F32),
        ],
        compiler_params=cparams,
        name="dc2_flash_fwd_fused",
    )(k_idx_arr, v_idx_arr, mode_arr, qg, kg, vg, wpre, w1post, q_seg, k_seg)

  # 2. Unified Ultra-Lean Backward Pass 1: computes ONLY dw1post (grid: B, G, nq, nkv)
  def _bwd_lean_dw1_kernel(
      kv_idx_ref, run_mask_ref,
      q_ref, k_ref, v_ref, dgr_ref, wpre_ref, lse_ref, q_seg_ref, k_seg_ref,
      dw1_ref, dw1_s_ref, qhat_s_ref, qw_s_ref,
  ):
    q_idx, step_idx = pl.program_id(2), pl.program_id(3)
    nkv = run_mask_ref.shape[1]

    @pl.when(step_idx == 0)
    def _init():
      dw1_s_ref[...] = jnp.zeros_like(dw1_s_ref)
      qf = q_ref[...].astype(F32)
      wpre = wpre_ref[...]
      qhat_s_ref[...] = (qf * wpre[2 * R * M : (2 * R + 1) * M][:, None, :]).astype(q_ref.dtype)
      qw_list = [
          (qf * wpre[r * M : (r + 1) * M][:, None, :]).reshape(MD, bwd_bq).astype(q_ref.dtype)
          for r in range(R)
      ]
      qw_s_ref[...] = jnp.stack(qw_list, axis=0)

    mode = run_mask_ref[q_idx, step_idx]
    kv_idx = kv_idx_ref[q_idx, step_idx]

    @pl.when(mode > 0)
    def _body():
      wpre, lse, dgr = wpre_ref[...], lse_ref[...][:M, :], dgr_ref[...]
      qhat, qw_all = qhat_s_ref[...], qw_s_ref[...]

      def sub_step(sub, _):
        sl = pl.ds(sub * bwd_ckv, bwd_ckv)
        k_sub, v_sub = k_ref[:, :, sl], v_ref[:, :, sl]
        v_flat = v_sub.reshape(MD, bwd_ckv)
        s_tilde, _, _ = _scores(k_sub, qhat, qw_all, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, bwd_ckv, bwd_bq)
        valid = _sub_valid(mode, q_idx * bwd_bq, kv_idx * bwd_bkv + sub * bwd_ckv, bwd_bq, bwd_ckv, q_s, k_s)
        s_masked = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        p = jnp.where(valid[None, :, :], jnp.exp(s_masked - lse[:, None, :]), 0.0)
        dw1_parts = []
        for r in range(R):
          dgr_k = jnp.where(valid, _dot(v_flat, dgr[r], (((0,), (0,)), ((), ()))), 0.0)
          dw1_parts.append(jnp.sum(dgr_k[None, :, :] * p, axis=1))
        dw1_cat = jnp.concatenate(dw1_parts, axis=0)
        if W1_PAD > W1_ACT:
          dw1_cat = jnp.pad(dw1_cat, ((0, W1_PAD - W1_ACT), (0, 0)))
        dw1_s_ref[...] = dw1_s_ref[...] + dw1_cat

      if num_bwd_sub == 1:
        sub_step(0, None)
      else:
        lax.fori_loop(0, num_bwd_sub, sub_step, None)

    @pl.when(step_idx == nkv - 1)
    def _done():
      dw1_ref[...] = dw1_s_ref[...]

  # 3. Unified Fused KV-Major Backward Pass 2 (grid: B, G, nkv, nq)
  def _bwd_fused_kv_kernel(
      q_idx_ref, run_mask_ref,
      q_ref, k_ref, v_ref, dobar_ref, dgr_ref, wpre_ref, w1post_ref, lse_ref, delta_ref, q_seg_ref, k_seg_ref,
      dq_part_ref, dwpre_part_ref, dk_ref, dv_ref,
      dk_s_ref, dv_s_ref, dq_s_ref,
  ):
    kv_idx, step_idx = pl.program_id(2), pl.program_id(3)
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
      q, dobar, dgr = q_ref[...], dobar_ref[...], dgr_ref[...]
      qf = q.astype(F32)
      wpre, w1post = wpre_ref[...], w1post_ref[...]
      lse, delta = lse_ref[...][:M, :], delta_ref[...][:M, :]
      qhat = (qf * wpre[2 * R * M : (2 * R + 1) * M][:, None, :]).astype(q.dtype)
      qw_all = [
          (qf * wpre[r * M : (r + 1) * M][:, None, :]).reshape(MD, bwd_bq).astype(q.dtype)
          for r in range(R)
      ]

      dq_s_ref[...] = jnp.zeros_like(dq_s_ref)
      dwpre_part_ref[...] = jnp.zeros_like(dwpre_part_ref)

      def sub_step(sub, _):
        sl = pl.ds(sub * bwd_ckv, bwd_ckv)
        k_sub, v_sub = k_ref[:, :, sl], v_ref[:, :, sl]
        v_flat = v_sub.reshape(MD, bwd_ckv)
        s_tilde, hs, k_flat = _scores(k_sub, qhat, qw_all, wpre)
        q_s, k_s = _load_seg_pair(q_seg_ref, k_seg_ref, sub, bwd_ckv, bwd_bq)
        valid = _sub_valid(mode, q_idx * bwd_bq, kv_idx * bwd_bkv + sub * bwd_ckv, bwd_bq, bwd_ckv, q_s, k_s)
        s_masked = jnp.where(valid[None, :, :], s_tilde, DEFAULT_MASK_VALUE)
        p = jnp.where(valid[None, :, :], jnp.exp(s_masked - lse[:, None, :]), 0.0)

        dv_step = _dot(dobar, p.astype(dobar.dtype), (((2,), (2,)), ((0,), (0,))))
        for r in range(R):
          gr = jnp.sum(p * w1post[r * M : (r + 1) * M][:, None, :], axis=0)
          dv_step = dv_step + _dot(dgr[r], gr.astype(dgr.dtype), (((1,), (1,)), ((), ()))).reshape(M, D, bwd_ckv)
        dv_s_ref[:, :, sl] = dv_s_ref[:, :, sl] + dv_step

        dp = jnp.where(valid[None, :, :], _dot(v_sub, dobar, (((1,), (1,)), ((0,), (0,)))), 0.0)
        for r in range(R):
          dgr_k = jnp.where(valid, _dot(v_flat, dgr[r], (((0,), (0,)), ((), ()))), 0.0)
          dp = dp + w1post[r * M : (r + 1) * M][:, None, :] * dgr_k[None, :, :]
        ds_tilde = p * (dp - delta[:, None, :])

        dk_step = _dot(qhat, ds_tilde.astype(qhat.dtype), (((2,), (2,)), ((0,), (0,))))
        dqhat_sub = _dot(k_sub, ds_tilde.astype(k_sub.dtype), (((2,), (1,)), ((0,), (0,))))
        dq_sub = dqhat_sub * wpre[2 * R * M : (2 * R + 1) * M][:, None, :]
        dw1_list, dw2_list = [], []
        for r in range(R):
          dw2_list.append(jnp.sum(ds_tilde * hs[r][None, :, :], axis=1))
          dhr = jnp.sum(ds_tilde * wpre[(R + r) * M : (R + r + 1) * M][:, None, :], axis=0)
          dk_step = dk_step + _dot(qw_all[r], dhr.astype(qhat.dtype), (((1,), (1,)), ((), ()))).reshape(M, D, bwd_ckv)
          dqwr = _dot(k_flat, dhr.astype(k_sub.dtype), (((1,), (0,)), ((), ()))).reshape(M, D, bwd_bq)
          dw1_list.append(jnp.sum(dqwr * qf, axis=1))
          dq_sub = dq_sub + dqwr * wpre[r * M : (r + 1) * M][:, None, :]

        dk_s_ref[:, :, sl] = dk_s_ref[:, :, sl] + dk_step
        ddd = jnp.sum(dqhat_sub * qf, axis=1)
        dq_s_ref[...] = dq_s_ref[...] + dq_sub
        dwp = jnp.concatenate(dw1_list + dw2_list + [ddd], axis=0)
        if W_PAD > W_ACT:
          dwp = jnp.pad(dwp, ((0, W_PAD - W_ACT), (0, 0)))
        dwpre_part_ref[...] = dwpre_part_ref[...] + dwp

      if num_bwd_sub == 1:
        sub_step(0, None)
      else:
        lax.fori_loop(0, num_bwd_sub, sub_step, None)
      dq_part_ref[...] = dq_s_ref[...].astype(dq_part_ref.dtype)

    @pl.when(step_idx == nq - 1)
    def _done():
      dk_ref[...] = dk_s_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_s_ref[...].astype(dv_ref.dtype)

  def bwd_local(qg, kg, vg, wpre, w1post, obar_g, lse, seg_ids, dobar_g, dgr_g):
    B, _, _, _, T = qg.shape
    nq, nkv = T // bwd_bq, T // bwd_bkv
    kv_idx_np, q_run_mask_np = _q_major_schedule(T, bwd_bq, bwd_bkv)
    q_idx_np, kv_run_mask_np = _kv_major_schedule(T, bwd_bq, bwd_bkv)
    kv_idx_arr, q_run_mask = jnp.asarray(kv_idx_np), jnp.asarray(q_run_mask_np)
    q_idx_arr, kv_run_mask = jnp.asarray(q_idx_np), jnp.asarray(kv_run_mask_np)
    q_seg, k_seg = _prepare_seg_arrays(seg_ids, B, T)

    q_spec = pl.BlockSpec((None, None, M, D, bwd_bq), lambda b, g, i, j, *_: (b, g, 0, 0, i))
    k_spec = pl.BlockSpec((None, None, M, D, bwd_bkv), lambda b, g, i, j, kv_ref, _: (b, g, 0, 0, kv_ref[i, j]))
    w_spec = pl.BlockSpec((None, None, W_PAD, bwd_bq), lambda b, g, i, j, *_: (b, g, 0, i))
    w1_spec = pl.BlockSpec((None, None, W1_PAD, bwd_bq), lambda b, g, i, j, *_: (b, g, 0, i))
    gr_spec = pl.BlockSpec((None, None, R, MD, bwd_bq), lambda b, g, i, j, *_: (b, g, 0, 0, i))
    lse_spec = pl.BlockSpec((None, None, M_PAD, bwd_bq), lambda b, g, i, j, *_: (b, g, 0, i))
    q_seg_spec = pl.BlockSpec((None, 8, bwd_bq), lambda b, g, i, j, *_: (b, 0, i)) if use_segment_ids else None
    k_seg_spec = (
        pl.BlockSpec((None, bwd_bkv, 128), lambda b, g, i, j, kv_ref, _: (b, kv_ref[i, j], 0))
        if use_segment_ids
        else None
    )

    dw1post = pl.pallas_call(
        _bwd_lean_dw1_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[q_spec, k_spec, k_spec, gr_spec, w_spec, lse_spec, q_seg_spec, k_seg_spec],
            out_specs=w1_spec,
            scratch_shapes=[
                pltpu.VMEM((W1_PAD, bwd_bq), F32),
                pltpu.VMEM((M, D, bwd_bq), qg.dtype),
                pltpu.VMEM((R, MD, bwd_bq), qg.dtype),
            ],
            grid=(B, G, nq, nkv),
        ),
        out_shape=jax.ShapeDtypeStruct((B, G, W1_PAD, T), F32),
        compiler_params=cparams,
        name="dc2_flash_bwd_dw1",
    )(kv_idx_arr, q_run_mask, qg, kg, vg, dgr_g, wpre, lse, q_seg, k_seg)

    delta_act = jnp.sum(obar_g.astype(F32) * dobar_g.astype(F32), axis=3)
    for r in range(R):
      delta_act = delta_act + w1post[:, :, r * M : (r + 1) * M, :] * dw1post[:, :, r * M : (r + 1) * M, :]
    delta = jnp.pad(delta_act, ((0, 0), (0, 0), (0, M_PAD - M), (0, 0))) if M_PAD > M else delta_act

    q_spec_kv = pl.BlockSpec((None, None, M, D, bwd_bq), lambda b, g, j, i, qr, _: (b, g, 0, 0, qr[j, i]))
    gr_spec_kv = pl.BlockSpec((None, None, R, MD, bwd_bq), lambda b, g, j, i, qr, _: (b, g, 0, 0, qr[j, i]))
    k_spec_kv = pl.BlockSpec((None, None, M, D, bwd_bkv), lambda b, g, j, i, *_: (b, g, 0, 0, j))
    w_spec_kv = pl.BlockSpec((None, None, W_PAD, bwd_bq), lambda b, g, j, i, qr, _: (b, g, 0, qr[j, i]))
    w1_spec_kv = pl.BlockSpec((None, None, W1_PAD, bwd_bq), lambda b, g, j, i, qr, _: (b, g, 0, qr[j, i]))
    lse_spec_kv = pl.BlockSpec((None, None, M_PAD, bwd_bq), lambda b, g, j, i, qr, _: (b, g, 0, qr[j, i]))
    q_seg_kv = pl.BlockSpec((None, 8, bwd_bq), lambda b, g, j, i, qr, _: (b, 0, qr[j, i])) if use_segment_ids else None
    k_seg_kv = pl.BlockSpec((None, bwd_bkv, 128), lambda b, g, j, i, *_: (b, j, 0)) if use_segment_ids else None

    dq_part, dwpre_part, dkg, dvg = pl.pallas_call(
        _bwd_fused_kv_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                q_spec_kv, k_spec_kv, k_spec_kv, q_spec_kv, gr_spec_kv,
                w_spec_kv, w1_spec_kv, lse_spec_kv, lse_spec_kv,
                q_seg_kv, k_seg_kv,
            ],
            out_specs=[
                pl.BlockSpec((None, None, None, M, D, bwd_bq), lambda b, g, j, i, *_: (j, b, g, 0, 0, i)),
                pl.BlockSpec((None, None, None, W_PAD, bwd_bq), lambda b, g, j, i, *_: (j, b, g, 0, i)),
                k_spec_kv,
                k_spec_kv,
            ],
            scratch_shapes=[
                pltpu.VMEM((M, D, bwd_bkv), F32),
                pltpu.VMEM((M, D, bwd_bkv), F32),
                pltpu.VMEM((M, D, bwd_bq), F32),
            ],
            grid=(B, G, nkv, nq),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((nkv, B, G, M, D, T), qg.dtype),
            jax.ShapeDtypeStruct((nkv, B, G, W_PAD, T), F32),
            jax.ShapeDtypeStruct((B, G, M, D, T), kg.dtype),
            jax.ShapeDtypeStruct((B, G, M, D, T), vg.dtype),
        ],
        compiler_params=cparams,
        name="dc2_flash_bwd_fused_kv",
    )(q_idx_arr, kv_run_mask, qg, kg, vg, dobar_g, dgr_g, wpre, w1post, lse, delta, q_seg, k_seg)
    dqg = jnp.sum(dq_part.astype(F32), axis=0).astype(qg.dtype)
    dwpre = jnp.sum(dwpre_part, axis=0)
    return dqg, dkg, dvg, dwpre, dw1post

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
  """Entry point for Unified Two-Pass Global Flash-DCMHA on query/key/value [B, T, H, D]."""
  bsz, seq_len, num_heads, head_dim = query.shape
  if dot_prec is None and getattr(config, "matmul_precision", "") == "highest":
    dot_prec = lax.Precision.HIGHEST

  num_groups = int(pre_q_dw_args[0].shape[2])
  rank = int(pre_q_dw_args[0].shape[3])
  group_heads = num_heads // num_groups

  if num_groups > 1:
    bq = min(int(getattr(config, "dc_grouped_block_q", 1024)), seq_len)
    bkv = min(int(getattr(config, "dc_grouped_block_kv", 1024)), seq_len)
    ckv = min(int(getattr(config, "dc_grouped_block_kv_compute", 512)), bkv)
    bwd_bq = min(int(getattr(config, "dc_grouped_bwd_block_q", 1024)), seq_len)
    bwd_bkv = min(int(getattr(config, "dc_grouped_bwd_block_kv", 1024)), seq_len)
    bwd_ckv = min(int(getattr(config, "dc_grouped_bwd_block_kv_compute", 512)), bwd_bkv)
  else:
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
      and head_dim % 8 == 0
      and (group_heads * head_dim) % 128 == 0
  )
  if not can_use_tpu_pallas:
    return _reference_dc2_attention(query, key, value, pre_q_dw_args, post_q_dw_args, seg_ids=decoder_segment_ids, dot_prec=dot_prec)

  fwd_local, bwd_local = _make_tpu_dc2_kernels(
      num_groups=num_groups,
      group_heads=group_heads,
      rank=rank,
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

  wpre = pack_dc2_pre_weights(*pre_q_dw_args)
  post_qw1, post_qw2, post_qdd = post_q_dw_args
  w1post = pack_dc2_w1post(post_qw1)

  qg_in = query.reshape(bsz, seq_len, num_groups, group_heads, head_dim).transpose(0, 2, 3, 4, 1)
  kg_in = key.reshape(bsz, seq_len, num_groups, group_heads, head_dim).transpose(0, 2, 3, 4, 1)
  vg_in = value.reshape(bsz, seq_len, num_groups, group_heads, head_dim).transpose(0, 2, 3, 4, 1)

  if mesh is None:
    def _run_fwd_sharded(qg, kg, vg, wp, w1p, seg):
      return fwd_local(qg, kg, vg, wp, w1p, seg)
    def _run_bwd_sharded(qg, kg, vg, wp, w1p, obar_g, lse, seg, dobar_g, dgr_g):
      return bwd_local(qg, kg, vg, wp, w1p, obar_g, lse, seg, dobar_g, dgr_g)
  else:
    q_g_spec = nn.logical_to_mesh_axes(("activation_kv_batch", None, None, None, None))
    gr_spec = nn.logical_to_mesh_axes(("activation_kv_batch", None, None, None, None))
    w_spec = nn.logical_to_mesh_axes(("activation_batch", None, None, None))
    lse_spec = nn.logical_to_mesh_axes(("activation_batch", None, None, None))
    seg_spec = nn.logical_to_mesh_axes(("activation_batch", None)) if decoder_segment_ids is not None else None

    _run_fwd_sharded = shard_map(
        fwd_local,
        mesh=mesh,
        in_specs=(q_g_spec, q_g_spec, q_g_spec, w_spec, w_spec, seg_spec),
        out_specs=(q_g_spec, gr_spec, lse_spec),
        check_rep=False,
    )
    _run_bwd_sharded = shard_map(
        bwd_local,
        mesh=mesh,
        in_specs=(q_g_spec, q_g_spec, q_g_spec, w_spec, w_spec, q_g_spec, lse_spec, seg_spec, q_g_spec, gr_spec),
        out_specs=(q_g_spec, q_g_spec, q_g_spec, w_spec, w_spec),
        check_rep=False,
    )

  @jax.custom_vjp
  def _sharded_core(qg, kg, vg, wp, w1p, seg):
    obar_g, gr_g, _ = _run_fwd_sharded(qg, kg, vg, wp, w1p, seg)
    return obar_g, gr_g

  def _sharded_fwd_vjp(qg, kg, vg, wp, w1p, seg):
    obar_g, gr_g, lse = _run_fwd_sharded(qg, kg, vg, wp, w1p, seg)
    if residual_name is not None:
      obar_g = ad_checkpoint.checkpoint_name(obar_g, name=residual_name)
      gr_g = ad_checkpoint.checkpoint_name(gr_g, name=residual_name)
      lse = ad_checkpoint.checkpoint_name(lse, name=residual_name)
    return (obar_g, gr_g), (qg, kg, vg, wp, w1p, obar_g, lse, seg)

  def _sharded_bwd_vjp(res, cts):
    qg, kg, vg, wp, w1p, obar_g, lse, seg = res
    dobar_g, dgr_g = cts
    dqg, dkg, dvg, dwp, dw1p = _run_bwd_sharded(qg, kg, vg, wp, w1p, obar_g, lse, seg, dobar_g, dgr_g)
    return dqg, dkg, dvg, dwp, dw1p, None

  _sharded_core.defvjp(_sharded_fwd_vjp, _sharded_bwd_vjp)

  obar_g, gr_g = _sharded_core(qg_in, kg_in, vg_in, wpre, w1post, decoder_segment_ids)
  obar = obar_g.transpose(0, 4, 1, 2, 3).astype(F32)  # [B, T, G, M, D]
  gr = gr_g.transpose(0, 4, 1, 2, 3).reshape(bsz, seq_len, num_groups, rank, group_heads, head_dim).astype(F32)
  out = obar * (1.0 + post_qdd.astype(F32))[..., None] + jnp.sum(
      gr * post_qw2.astype(F32)[..., None], axis=3
  )
  return out.reshape(bsz, seq_len, num_heads, head_dim).astype(value.dtype)
