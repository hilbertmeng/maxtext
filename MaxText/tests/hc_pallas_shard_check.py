# Multi-device CPU check of every hc_pallas shard_map wrapper (interpret mode), vs the XLA references.
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
import sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import jax, jax.numpy as jnp, numpy as np
import flax.linen.partitioning as nnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
jax.config.update('jax_default_matmul_precision', 'highest')
import exp
from layers import residual_connections as rc, hc_pallas as hp
mesh = Mesh(np.array(jax.devices()).reshape(1, 4), ('data', 'fsdp'))
rules = (('activation_batch', ('data', 'fsdp')), ('activation_norm_length', None), ('activation_embed', None), ('embed', None))
n, B, T, D = 4, 8, 16, 128
r = np.random.default_rng(0)
with mesh, nnp.axis_rules(rules):
  L = jnp.asarray(r.normal(size=(n, n, B, T)) * 2, jnp.float32)
  W = jnp.asarray(r.normal(size=L.shape), jnp.float32)
  f = lambda fn: jax.jit(jax.value_and_grad(lambda l: jnp.sum(fn(l) * W)))
  a, b = f(lambda l: rc.sinkhorn_major(l, 20, 1e-6))(L), f(lambda l: hp.sinkhorn_pallas(l, 20, 1e-6, tile=8, interpret=True))(L)
  sk = float(jnp.linalg.norm(a[1] - b[1]) / jnp.linalg.norm(a[1]))
  print('sinkhorn', float(a[0]), float(b[0]), sk)
  assert abs(float(a[0]) - float(b[0])) < 1e-4 * abs(float(a[0])) and sk < 1e-5, sk
  fields = {k: v for k, v in vars(exp.MHCConfig).items() if not k.startswith('_')}
  for mode, Mod in (('mhc', rc.ManifoldHyperConnection), ('dhc', rc.DynamicHyperConnection)):
    state = jnp.asarray(r.normal(size=(n, B, T, D)), jnp.float32)
    res = {}
    for pallas in (False, True):
      cfg = types.SimpleNamespace(**dict(fields, weight_dtype=jnp.float32, dhc_norm_epsilon=1e-5, mhc_stream_major_layout=True,
          mhc_pallas_connector=pallas, mhc_pallas_sinkhorn=pallas and mode == 'mhc', mhc_pallas_interpret=True))
      m = Mod(cfg, layer_index=3)
      p = m.init(jax.random.key(0), state)
      p = jax.tree.map(lambda v: v + .1 * jnp.asarray(r.normal(size=v.shape), v.dtype), p) if not pallas else res[False][2]
      def loss(p, s):
        out = m.apply(p, s); upd = (hp.mhc_update_pallas if pallas else rc.mhc_update_major)
        s2 = upd(out[3] if len(out) == 4 else s, out[0], out[1], out[2], **({'interpret': True, 'tile': 8} if pallas else {}))
        return jnp.sum(jnp.sin(s2)) + jnp.sum(jnp.cos(out[0]))
      v, g = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(p, state)
      res[pallas] = (v, g, p)
    (v0, g0, _), (v1, g1, _) = res[False], res[True]
    rel = lambda x, y: float(jnp.linalg.norm(x - y) / jnp.linalg.norm(y))
    worst = max(rel(x, y) for x, y in zip(jax.tree.leaves(g1[0]), jax.tree.leaves(g0[0])))
    print(mode, float(v0), float(v1), 'dstate', rel(g1[1], g0[1]), 'dparams max', worst)
    assert abs(float(v0) - float(v1)) < 1e-5 * abs(float(v0)) and rel(g1[1], g0[1]) < 1e-5 and worst < 1e-4, mode
print('HC_PALLAS_SHARD_OK')
