"""v8 fused BAM core: token-major M [B,T,V,K]; every M contraction is a per-token MXU matmul.

v7 runs the dynamic reads/writes on the VPU (each token has its own M_t, so no MXU operand is
shared across tokens): ~160 VALU cycles per token for a full-M read on v5p. Here M_t [V,K] is the
stationary MXU operand of a per-token matmul (batched dot over the token axis), ~13 cycles/token
for 64 rows. All reads of one layer are linear in M_t, so they collapse into ONE matmul per token:

  read:   Y_t [R,K] = X_t [R,V] @ M_t [V,K]       X stacks every read key, e.g. for the full-M
          generalized read R = 4N rows  q: gq*norm(rq) P^T + sg_q S_q^T, k: likewise, v/o: the
          gated VO key plus the gated static key.  Keys/gates/norms are built by XLA (autodiff).
  write:  M'_t   = M_t + A_t^T [V,N] @ C_t [N,K]  A = norm(address), C = sigmoid(logit) norm(content)

Reverses are matmuls too: dX = dY M^T, dM = X^T dY (+ the passthrough cotangent, fused);
dA = C G^T, dC = A G.  Operands are the model dtype (BF16 in training), accumulation FP32.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl

from layers.bam_pallas import F32, _freeze, _map_batch, _params


def _bdot(x, y, cx, cy):
  """Token-batched matmul: contract x axis cx with y axis cy, batch over axis 0."""
  return jax.lax.dot_general(x, y, (((cx,), (cy,)), ((0,), (0,))), preferred_element_type=F32)


def _token_spec(shape, tile):
  return pl.BlockSpec((None, tile) + tuple(shape), lambda b, i: (b, i) + (0,) * len(shape))


def _chunks(tile, chunk):
  return [pl.ds(c0, min(chunk, tile - c0)) for c0 in range(0, tile, chunk)]


# ---------------------------------------------------------------------------
# Read: Y = X @ M per token, with the M passthrough (its cotangent added in the reverse).

def _read_forward_call(x, m, opts):
  b, t, r, v = x.shape
  kdim = m.shape[-1]
  tile = min(opts['tile'], t)

  def kernel(x_ref, m_ref, y_ref):
    for s in _chunks(tile, opts['chunk']):
      y_ref[s] = _bdot(x_ref[s], m_ref[s], 2, 1).astype(y_ref.dtype)

  return pl.pallas_call(
      kernel, grid=(b, t // tile), in_specs=[_token_spec((r, v), tile), _token_spec((v, kdim), tile)],
      out_specs=_token_spec((r, kdim), tile), out_shape=jax.ShapeDtypeStruct((b, t, r, kdim), x.dtype),
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read')(x, m)


def _read_backward_call(x, m, dy, gm, opts):
  b, t, r, v = x.shape
  kdim = m.shape[-1]
  tile = min(opts['reverse_tile'], t)

  def kernel(x_ref, m_ref, dy_ref, gm_ref, dx_ref, dm_ref):
    for s in _chunks(tile, opts['chunk']):
      dy_s = dy_ref[s]
      dx_ref[s] = _bdot(dy_s, m_ref[s], 2, 2).astype(dx_ref.dtype)             # [R,K] x [V,K]^T
      dm_ref[s] = (_bdot(x_ref[s], dy_s, 1, 1) + gm_ref[s].astype(F32)).astype(dm_ref.dtype)

  mspec = _token_spec((v, kdim), tile)
  return pl.pallas_call(
      kernel, grid=(b, t // tile),
      in_specs=[_token_spec((r, v), tile), mspec, _token_spec((r, kdim), tile), mspec],
      out_specs=[_token_spec((r, v), tile), mspec],
      out_shape=(jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct(m.shape, m.dtype)),
      input_output_aliases={3: 1}, interpret=opts['interpret'],
      compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read_backward')(x, m, dy, gm)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _read(x, m, opts):
  return _read_forward_call(x, m, dict(opts)), m


def _read_fwd(x, m, opts):
  return _read(x, m, opts), (x, m)


def _read_bwd(opts, res, cts):
  x, m = res
  dy, gm = cts
  return _read_backward_call(x, m, dy, gm, dict(opts))


_read.defvjp(_read_fwd, _read_bwd)


def read(m, x, *, tile=128, reverse_tile=128, chunk=8, vmem_mib=None, interpret=False):
  """m [B,T,V,K], x [B,T,R,V] -> (y [B,T,R,K], m passthrough). y is named 'bam_read_out'."""
  opts = _freeze(dict(tile=tile, reverse_tile=reverse_tile, chunk=chunk, vmem_mib=vmem_mib,
                      interpret=interpret))
  y, m = _map_batch(lambda x, m: _read(x, m, opts), (x, m), (True, True), 2)
  return checkpoint_name(y, 'bam_read_out'), m


# ---------------------------------------------------------------------------
# Write: M' = M + A^T C per token (in place).

def _write_forward_call(m, a, c, opts):
  b, t, v, kdim = m.shape
  n = a.shape[2]
  tile = min(opts['tile'], t)

  def kernel(m_ref, a_ref, c_ref, o_ref):
    for s in _chunks(tile, opts['chunk']):
      o_ref[s] = (m_ref[s].astype(F32) + _bdot(a_ref[s], c_ref[s], 1, 1)).astype(o_ref.dtype)

  mspec = _token_spec((v, kdim), tile)
  return pl.pallas_call(
      kernel, grid=(b, t // tile), in_specs=[mspec, _token_spec((n, v), tile), _token_spec((n, kdim), tile)],
      out_specs=mspec, out_shape=jax.ShapeDtypeStruct(m.shape, m.dtype), input_output_aliases={0: 0},
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_write')(m, a, c)


def _write_backward_call(g, a, c, opts):
  b, t, v, kdim = g.shape
  n = a.shape[2]
  tile = min(opts['reverse_tile'], t)

  def kernel(g_ref, a_ref, c_ref, da_ref, dc_ref):
    for s in _chunks(tile, opts['chunk']):
      g_s = g_ref[s]
      da_ref[s] = _bdot(c_ref[s], g_s, 2, 2).astype(da_ref.dtype)     # [N,K] x [V,K]^T
      dc_ref[s] = _bdot(a_ref[s], g_s, 2, 1).astype(dc_ref.dtype)     # [N,V] x [V,K]

  aspec, cspec = _token_spec((n, v), tile), _token_spec((n, kdim), tile)
  return pl.pallas_call(
      kernel, grid=(b, t // tile), in_specs=[_token_spec((v, kdim), tile), aspec, cspec],
      out_specs=[aspec, cspec],
      out_shape=(jax.ShapeDtypeStruct(a.shape, a.dtype), jax.ShapeDtypeStruct(c.shape, c.dtype)),
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_write_backward')(g, a, c)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _write(m, a, c, opts):
  return _write_forward_call(m, a, c, dict(opts))


def _write_fwd(m, a, c, opts):
  return _write_forward_call(m, a, c, dict(opts)), (a, c)


def _write_bwd(opts, res, g):
  a, c = res
  return (g,) + tuple(_write_backward_call(g, a, c, dict(opts)))


_write.defvjp(_write_fwd, _write_bwd)


def write(m, a, c, *, tile=128, reverse_tile=128, chunk=8, vmem_mib=None, interpret=False):
  """m [B,T,V,K] + sum_n a[n]^T c[n]; a [B,T,N,V], c [B,T,N,K] (normalized, gated factors)."""
  opts = _freeze(dict(tile=tile, reverse_tile=reverse_tile, chunk=chunk, vmem_mib=vmem_mib,
                      interpret=interpret))
  return _map_batch(lambda m, a, c: _write(m, a, c, opts), (m, a, c), (True, True, True), 1)


# ---------------------------------------------------------------------------
# XLA-side factors (autodiffed): the same math as layers/bam_pallas.py read_tile / write_tile.

def _rms(x, eps):
  f = x.astype(F32)
  return f * jax.lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + eps)


def read_keys(q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits, static_weight, *, heads,
              read_epsilon, key_scale, static_gates=None, dtype):
  """Stacked read keys X [B,T,4N,V] (rows q, k, v, o per head).

  q_key/k_key/vo_key [B,T,N,C]; logits [B,T,N]; static_weight [4N+C, V] = [S_q; S_k; S_v; S_o; P^T]
  (P the [V,C] compression, identity for full-M reads); static_gates: optional (q, k, v, o) [B,T,N]."""
  n = heads
  sw = static_weight.astype(F32)
  s_rows = [sw[i * n:(i + 1) * n] for i in range(4)]                    # [N,V]
  p_t = sw[4 * n:]                                                       # [C,V]
  gate = lambda l: key_scale * jax.nn.sigmoid(l.astype(F32))[..., None]
  nr = _rms(vo_key, read_epsilon)
  dyn = [gate(q_logits) * _rms(q_key, read_epsilon), gate(k_logits) * _rms(k_key, read_epsilon),
         gate(v_logits) * nr, gate(o_logits) * nr]                       # [B,T,N,C]
  rows = []
  for i in range(4):
    st = s_rows[i] if static_gates is None else static_gates[i].astype(F32)[..., None] * s_rows[i]
    rows.append(jnp.einsum('btnc,cv->btnv', dyn[i], p_t) + st)
  return jnp.concatenate(rows, axis=2).astype(dtype)


def split_reads(y, query_std, key_std, *, heads, qk_cols, q_scale=1.0):
  """Y [B,T,4N,K] -> (query, key, value, local_o) [B,T,N,K]; q/k columns >= qk_cols are standard."""
  n = heads
  q, k, v, o = (y[:, :, i * n:(i + 1) * n] for i in range(4))
  q = jnp.concatenate((q[..., :qk_cols], query_std.astype(y.dtype)), axis=-1)
  k = jnp.concatenate((k[..., :qk_cols], key_std.astype(y.dtype)), axis=-1)
  if q_scale != 1.0:
    q = (q * q_scale).astype(y.dtype)
  return q, k, v, o


def write_factors(groups, *, epsilon, dtype):
  """groups of (content [B,T,N,K], logits [B,T,N], address [B,T,N,V]) -> stacked A, C."""
  a = [_rms(address, epsilon) for _, _, address in groups]
  c = [jax.nn.sigmoid(logits.astype(F32))[..., None] * _rms(content, epsilon)
       for content, logits, _ in groups]
  return jnp.concatenate(a, axis=2).astype(dtype), jnp.concatenate(c, axis=2).astype(dtype)
