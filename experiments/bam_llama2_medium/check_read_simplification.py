#!/usr/bin/env python3
"""Compare nonzero read outputs and M/key/mix gradients to a sealed Git control."""
import argparse
import ast
import json
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
from layers import attentions as att


def load_reference(commit):
  source = subprocess.check_output(
      ['git', 'show', f'{commit}:MaxText/layers/attentions.py'], text=True)
  tree = ast.parse(source)
  node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'factorized_head_bam_read')
  namespace = dict(vars(att))
  exec(compile(ast.Module(body=[node], type_ignores=[]), '<reference>', 'exec'), namespace)
  return namespace['factorized_head_bam_read']


def joined(result):
  return jnp.concatenate(result, axis=-1) if isinstance(result, tuple) else result


def main():
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument('--reference', default='5066e56')
  args = parser.parse_args()
  reference = load_reference(args.reference)
  records = []
  for dtype in (jnp.float32, jnp.bfloat16):
    for rank in (1, 2, 4):
      for routing in (('legacy',) if rank == 1 else ('legacy', 'shared_rank_gate', 'head_rank_gate')):
        for side in ('both', 'row', 'col'):
          keys = jax.random.split(jax.random.key(42), 5)
          matrix = jax.random.normal(keys[0], (1, 2, 4, 6)).astype(dtype)
          key_shape = (1, 2, 10) if rank == 1 else (1, 2, rank, 10)
          mix_shape = (1, 2, 3, 2) if rank == 1 else (1, 2, 3, 2, rank)
          key = jax.random.normal(keys[1], key_shape).astype(dtype)
          mix = jax.random.normal(keys[2], mix_shape).astype(dtype)
          gate_shape = (1, 2, rank, 2) if routing == 'shared_rank_gate' else (1, 2, 2)
          gate = jax.random.normal(keys[3], gate_shape).astype(dtype) - 2
          projection = jax.random.normal(keys[4], (6, 2)).astype(dtype)
          options = dict(rank=rank, rank_routing=routing, read_side=side,
                         rms_epsilon=1e-4, key_mode='rms_gate', key_scale=2.,
                         key_gate_logits=gate, v_projection=projection,
                         head_rank_gate_bias=jnp.array([-2., -3.], dtype),
                         side_amplitude=jnp.array([.7, 1.2], dtype))
          for implementation in ('dot_btn', 'mul_reduce_btn'):
            options['implementation'] = implementation
            def forward(fn, m, k, h):
              return joined(fn(m, None, lambda _: k, lambda _: h, **options))
            expected = forward(reference, matrix, key, mix)
            actual = forward(att.factorized_head_bam_read, matrix, key, mix)
            # A random cotangent avoids the near-constant squared norm of RMS
            # outputs, whose near-zero gradient makes relative errors unstable.
            cotangent = jax.random.normal(jax.random.key(123), expected.shape)
            def objective(fn, m, k, h):
              return jnp.sum(forward(fn, m, k, h).astype(jnp.float32) * cotangent)
            old_grads = jax.grad(lambda m, k, h: objective(reference, m, k, h), (0, 1, 2))(matrix, key, mix)
            new_grads = jax.grad(lambda m, k, h: objective(att.factorized_head_bam_read, m, k, h), (0, 1, 2))(matrix, key, mix)
            errors = []
            for old, new in zip((expected, *old_grads), (actual, *new_grads)):
              old, new = np.asarray(old, dtype=np.float32), np.asarray(new, dtype=np.float32)
              relative = float(np.linalg.norm(new-old) / max(np.linalg.norm(old), 1e-12))
              errors.append(relative)
              assert np.isfinite(new).all()
              assert relative < (0.04 if dtype == jnp.bfloat16 else 2e-5), (rank, routing, side, implementation, errors)
            records.append(dict(dtype=str(dtype), rank=rank, routing=routing,
                                side=side, implementation=implementation,
                                relative_l2_output_M_key_mix=errors))
  print(json.dumps({'reference': args.reference, 'cases': records,
                    'max_relative_l2': max(max(r['relative_l2_output_M_key_mix']) for r in records)}, indent=2))


if __name__ == '__main__':
  main()
