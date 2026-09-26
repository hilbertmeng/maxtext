"""Paired write/backprop microbenchmark; does not launch or modify training."""
import argparse
import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
from layers import normalizations, rmt


def main():
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--batch', type=int, default=8)
  parser.add_argument('--tokens', type=int, default=4096)
  parser.add_argument('--samples', type=int, default=20)
  parser.add_argument('--warmup', type=int, default=5)
  parser.add_argument('--no-profile', action='store_true')
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  prefix = (args.batch, args.tokens)
  rng = jax.random.split(jax.random.key(41), 5)
  inputs = (jax.random.normal(rng[0], prefix+(16, 48), dtype=jnp.bfloat16),
            jnp.full(prefix+(16,), -2.1972246, jnp.bfloat16),
            jax.random.normal(rng[1], prefix+(16, 75), dtype=jnp.bfloat16),
            .25*jax.random.normal(rng[2], (16, 48), dtype=jnp.float32),
            jax.random.normal(rng[3], prefix+(48, 75), dtype=jnp.bfloat16))
  jax.block_until_ready(inputs)
  results = {'device': str(jax.devices()[0]), 'jax_version': jax.__version__,
             'shapes': [list(x.shape) for x in inputs], 'arms': []}
  for single in (False, True):
    for health in (False, True):
      label = f'{"single" if single else "double"}_health{int(health)}'

      def objective(raw_address, logits, data, static_address, matrix):
        address = normalizations.rms_norm(raw_address, dtype=jnp.bfloat16, epsilon=1e-6)
        gated_address = jax.nn.sigmoid(logits)[..., None]*address
        if single:
          c = jax.lax.rsqrt(jnp.mean(jnp.square(data.astype(jnp.float32)),
                                   axis=-1, keepdims=True)+1e-6)
          dyn_address = (gated_address.astype(jnp.float32)*c).astype(jnp.bfloat16)
          coefficient = static_address.astype(jnp.bfloat16)+dyn_address
          write = jnp.einsum('btnk,btnv->btkv', coefficient, data)
          metrics = (rmt._factorized_write_health(dyn_address,
                          static_address.astype(jnp.bfloat16), data) if health else ())
          updated = matrix+write
        else:
          norm_data = normalizations.rms_norm(data, dtype=jnp.bfloat16, epsilon=1e-6)
          dynamic = jnp.einsum('btnk,btnv->btkv', gated_address, norm_data)
          static = jnp.einsum('btnv,nk->btkv', data, static_address.astype(jnp.bfloat16))
          metrics = (tuple(v for part in (slice(None,16),slice(16,None))
                     for v in rmt._write_health(dynamic[...,part,:],static[...,part,:]))
                     if health else ())
          updated = matrix+static+dynamic
        loss = jnp.mean(jnp.square(jnp.tanh(updated).astype(jnp.float32)))
        return loss, metrics

      step = jax.jit(jax.value_and_grad(objective, argnums=(0,1,2,3,4), has_aux=True))
      start = time.perf_counter()
      executable = step.lower(*inputs).compile()
      compile_seconds = time.perf_counter()-start
      for _ in range(args.warmup):
        output = jax.block_until_ready(executable(*inputs))
      seconds = []
      for _ in range(args.samples):
        start = time.perf_counter()
        output = jax.block_until_ready(executable(*inputs))
        seconds.append(time.perf_counter()-start)
      if not args.no_profile:
        with jax.profiler.trace(str(args.output/label), create_perfetto_link=False):
          for _ in range(5):
            output = jax.block_until_ready(executable(*inputs))
      row = {'arm': label, 'median_ms': 1000*statistics.median(seconds),
             'min_ms': 1000*min(seconds), 'max_ms': 1000*max(seconds),
             'compile_seconds': compile_seconds, 'samples_seconds': seconds,
             'loss': float(output[0][0])}
      results['arms'].append(row)
      print(json.dumps(row), flush=True)
      (args.output/'summary.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
  main()
