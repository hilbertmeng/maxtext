"""Compute the 2.8B embedding divisor; verify it with an embedding-only CPU probe.

No checkpoint, dataset, full Transformer or TPU is needed. Inputs have the
target's Gaussian .006 embedding distribution; this is an initialization check,
not a trained-model or Pile loss evaluation.
"""
import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from scipy.integrate import quad
from scipy.special import gammaln

from bam_prop_test_utils import MLPWriteTest
from layers import initializers, linears
from layers.models import EmbeddingBamWrite

BASE = ('BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1'
        'AllR640SeparateVOKeysLogitDiagNegativeQK80')


def gaussian_divisor(width, head_dim, sigma_embedding=.006, sigma_projection=.006, epsilon=1e-6):
  variance = width * sigma_embedding**2 * sigma_projection**2
  shape = head_dim / 2
  # q/variance ~ chi2(K)/K. Match E[q] / E[q/(q+epsilon)].
  def integrand(x):
    if x == 0:
      return 0.
    density = math.exp((shape - 1) * math.log(x) - x - gammaln(shape))
    q = variance * x / shape
    return density * q / (q + epsilon)
  normalized_energy = quad(integrand, 0, max(200, shape * 5), epsabs=1e-12)[0]
  return math.sqrt(variance / normalized_energy)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  helper = MLPWriteTest()
  helper.setUp()
  try:
    cfg = helper.config(BASE)
    eps = initializers.get_bam_write_epsilon(cfg)
    tau = gaussian_divisor(cfg.emb_dim, cfg.bam_k, epsilon=eps)
    assert cfg.bam_init_method == cfg.token_embedding_init_method == '006normal'
    cfg.get_keys().update(bam_pallas_core=False)
    module = EmbeddingBamWrite(cfg, cfg.emb_bam_num_head, cfg.dtype,
        cfg.weight_dtype, None, initializers.get_bam_init(cfg))
    captured = {}

    def intercept(next_fun, call_args, kwargs, context):
      result = next_fun(*call_args, **kwargs)
      if isinstance(context.module, linears.DenseGeneral):
        captured[context.module.name] = result
      return result

    def inputs(key):
      return (jax.random.normal(key, (1, 128, cfg.emb_dim)) * .006).astype(cfg.dtype)

    params = module.init(jax.random.key(0), inputs(jax.random.key(1)))

    @jax.jit
    def energies(x):
      with nn.intercept_methods(intercept):
        module.apply(params, x)
      content = captured['W_emb_u'].astype(jnp.float32)
      address = captured['W_emb_v_up'].astype(jnp.float32)
      address = (address * jax.lax.rsqrt(jnp.mean(address**2, -1, keepdims=True) + eps)).astype(cfg.dtype)
      bias = nn.unbox(params['params']['emb_gw_b0']).astype(cfg.dtype)
      gate = jax.nn.sigmoid(captured['W_emb_g'] + bias)
      if cfg.bam_sqrt_n_scale:
        gate /= jnp.sqrt(jnp.asarray(cfg.emb_bam_num_head, gate.dtype))
      normed = (content * jax.lax.rsqrt(jnp.mean(content**2, -1, keepdims=True) + eps)).astype(cfg.dtype)
      fixed = (content / tau).astype(cfg.dtype)
      raw = content.astype(cfg.dtype)
      def energy(c):
        delta = jnp.einsum('btnk,btnv->btkv',
            (gate[..., None] * c).astype(cfg.dtype), address,
            preferred_element_type=jnp.float32)
        return jnp.mean(delta**2)
      return jnp.stack([energy(raw), energy(normed), energy(fixed)])

    batches = np.asarray([energies(inputs(jax.random.key(100 + i))) for i in range(8)])
    records = {}
    for name, subset in [('calibration', batches[:4]), ('heldout', batches[4:])]:
      raw, normed, fixed = subset.mean(axis=0)
      records[name] = dict(tokens=512, empirical_divisor=float(np.sqrt(raw / normed)),
          fixed_to_normalized_write_energy=float(fixed / normed))
      assert abs(float(fixed / normed) - 1.) < .03, records[name]
    result = dict(exp=BASE, input_distribution='independent Gaussian embedding std .006',
        model_initialization_seed=0, dimension=cfg.emb_dim, heads=cfg.emb_bam_num_head,
        k=cfg.bam_k, v=cfg.bam_v, epsilon=eps, theoretical_divisor=tau,
        config_divisor=.0020825863, dtype=str(cfg.dtype), **records)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
  finally:
    helper.tearDown()


if __name__ == '__main__':
  main()
