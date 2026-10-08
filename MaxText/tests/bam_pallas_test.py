"""CPU interpret-mode checks of the BAM Pallas core against the original token-major math."""
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'MaxText')
from layers import bam_pallas as bp  # pylint: disable=wrong-import-position
from layers import normalizations  # pylint: disable=wrong-import-position

B, T, V, K, N, C, R, QKC = 2, 256, 40, 96, 4, 10, 24, 72
READ_EPS, WRITE_EPS, SCALE = 1e-4, 1e-6, 0.2


def rms(x, eps, dtype):
  return normalizations.rms_norm(x, dtype=dtype, epsilon=eps, statistics_dtype=jnp.float32)


def original_read(m_tm, sq, sk, sv, so, p, rq, lq, rk, lk, rr, lo, lv, qs, ks):
  """Token-major transcription of BamAttention's DirectC10 AllLocal read path."""
  dt = m_tm.dtype
  mc = jnp.einsum('btkv,vc->btkc', m_tm, p.astype(dt))

  def direct(r, l, s):
    key = SCALE * jax.nn.sigmoid(l)[..., None].astype(dt) * rms(r, READ_EPS, dt)
    col = jnp.einsum('btkv,btnv->btnk', mc, key)[..., :QKC]
    static = jnp.einsum('btkv,vn->btnk', m_tm[..., :QKC, :], s.astype(dt))
    return col + static

  q = jnp.concatenate((direct(rq, lq, sq), qs), -1)
  k = jnp.concatenate((direct(rk, lk, sk), ks), -1)
  y = jnp.einsum('btkv,btnv->btnk', mc, rms(rr, READ_EPS, dt))
  v = y * (SCALE * jax.nn.sigmoid(lv))[..., None].astype(dt) + jnp.einsum('btkv,vn->btnk', m_tm, sv.astype(dt))
  o = y * (SCALE * jax.nn.sigmoid(lo))[..., None].astype(dt) + jnp.einsum('btkv,vn->btnk', m_tm, so.astype(dt))
  return q, k, v, o


def original_write(m_tm, *groups):
  dt = m_tm.dtype
  dm = 0
  for content, logits, address in (groups[i:i + 3] for i in range(0, len(groups), 3)):
    c = jax.nn.sigmoid(logits)[..., None].astype(dt) * rms(content, WRITE_EPS, dt)
    a = rms(address, WRITE_EPS, dt)
    dm = dm + jnp.einsum('btnk,btnv->btkv', c, a)
  return m_tm + dm


def to_minor_m(m_tm):
  return jnp.transpose(m_tm, (0, 3, 2, 1))   # [B,T,K,V] -> [B,V,K,T]


def inputs(dtype, seed=0):
  ks = iter(jax.random.split(jax.random.PRNGKey(seed), 32))
  r = lambda shape, s=1.0: (s * jax.random.normal(next(ks), shape)).astype(dtype)
  m_tm = r((B, T, K, V))
  params = (r((V, N), .3), r((V, N), .3), r((V, N), .3), r((V, N), .3), r((V, C), .3))
  tokens = (r((B, T, N, C)), r((B, T, N)), r((B, T, N, C)), r((B, T, N)), r((B, T, N, C)),
            r((B, T, N)), r((B, T, N)), r((B, T, N, R)), r((B, T, N, R)))
  return m_tm, params, tokens


def kernel_read(m_tm, params, tokens, body='blocked'):
  sw = bp.static_weight(*params)
  return bp.read(to_minor_m(m_tm), sw, *tokens, qk_cols=QKC, read_epsilon=READ_EPS,
                 key_scale=SCALE, interpret=True, body=body)


def write_groups(dtype, n_groups, seed=1):
  ks = iter(jax.random.split(jax.random.PRNGKey(seed), 16))
  out = []
  for _ in range(n_groups):
    out.append((jax.random.normal(next(ks), (B, T, N, K)).astype(dtype),
                jax.random.normal(next(ks), (B, T, N)).astype(dtype),
                jax.random.normal(next(ks), (B, T, N, V)).astype(dtype)))
  return out


class BamPallasTest(unittest.TestCase):

  def assert_close(self, a, b, tol, name):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    err = np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)
    self.assertLess(err, tol, f'{name}: relative max error {err:.3e}')

  def test_read_forward(self):
    for dtype, tol in ((jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)):
      m_tm, params, tokens = inputs(dtype)
      want = original_read(m_tm, *params, *tokens)
      for body in ('tile', 'blocked2', 'kmajor', 'v3'):
        got = kernel_read(m_tm, params, tokens, body)
        for name, a, b in zip('qkvo', got, want):
          self.assert_close(a, b, tol, f'{body} {dtype.__name__} {name}')

  def test_read_gradients(self):
    m_tm, params, tokens = inputs(jnp.float32, seed=3)
    cts = [jax.random.normal(jax.random.PRNGKey(10 + i), (B, T, N, K)) for i in range(4)]

    def loss(fn, m_tm, params, tokens):
      return sum(jnp.sum(o * c) for o, c in zip(fn(m_tm, params, tokens), cts))

    want = jax.grad(lambda *a: loss(lambda m, p, t: original_read(m, *p, *t), *a), argnums=(0, 1, 2))(m_tm, params, tokens)
    for body in ('tile', 'blocked2', 'kmajor', 'v3'):
      got = jax.grad(lambda *a: loss(lambda m, p, t: kernel_read(m, p, t, body), *a), argnums=(0, 1, 2))(m_tm, params, tokens)
      self._compare_grads(got, want, body)

  def _compare_grads(self, got, want, body):
    for name, a, b in zip(['m'] + [f'param{i}' for i in range(5)] + [f'token{i}' for i in range(9)],
                          jax.tree.leaves(got), jax.tree.leaves(want)):
      self.assert_close(a, b, 1e-4, f'{body} grad {name}')

  def test_write_forward_and_gradients(self):
    for n_groups in (1, 2):
      for dtype, tol in ((jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)):
        m_tm, _, _ = inputs(dtype)
        groups = write_groups(dtype, n_groups)
        flat = [x for g in groups for x in g]
        want = original_write(m_tm, *flat)
        for body in ('tile', 'blocked2', 'kmajor', 'v3'):
          got = jnp.transpose(bp.write(to_minor_m(m_tm), groups, epsilon=WRITE_EPS, interpret=True, body=body), (0, 3, 2, 1))
          self.assert_close(got, want, tol, f'{body} write {n_groups} {dtype.__name__}')
      m_tm, _, _ = inputs(jnp.float32, seed=5)
      groups = write_groups(jnp.float32, n_groups, seed=6)
      flat = [x for g in groups for x in g]
      ct = jax.random.normal(jax.random.PRNGKey(7), (B, T, K, V))
      want = jax.grad(lambda m, *f: jnp.sum(original_write(m, *f) * ct), argnums=tuple(range(1 + len(flat))))(m_tm, *flat)

      for body in ('tile', 'blocked2', 'kmajor', 'v3'):
        def kernel_loss(m, *f):
          out = bp.write(to_minor_m(m), [f[i:i + 3] for i in range(0, len(f), 3)], epsilon=WRITE_EPS,
                         interpret=True, body=body)
          return jnp.sum(jnp.transpose(out, (0, 3, 2, 1)) * ct)

        got = jax.grad(kernel_loss, argnums=tuple(range(1 + len(flat))))(m_tm, *flat)
        for i, (a, b) in enumerate(zip(got, want)):
          self.assert_close(a, b, 1e-4, f'{body} write grad {n_groups} arg{i}')


if __name__ == '__main__':
  unittest.main()
