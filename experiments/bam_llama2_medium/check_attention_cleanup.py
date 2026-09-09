#!/usr/bin/env python3
"""Compare retained BAM parameter trees, outputs and VJPs with a sealed runtime."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import max_utils
import pyconfig
from layers import attentions as att


def reference_class(commit):
  source = subprocess.check_output(
      ['git', 'show', f'{commit}:MaxText/layers/attentions.py'], text=True)
  nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
           or isinstance(n, ast.ClassDef) and n.name in ('GroupedRMSNorm', 'BamAttention')]
  namespace = dict(vars(att))
  exec(compile(ast.Module(body=nodes, type_ignores=[]), '<reference>', 'exec'), namespace)
  return namespace['BamAttention']


def compare(old, new):
  assert jax.tree.structure(old) == jax.tree.structure(new)
  error = 0.
  for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    assert np.isfinite(b).all()
    np.testing.assert_array_equal(a, b)
    error = max(error, float(np.max(np.abs(a-b))))
  return error


def main():
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument('--reference', default='34fa43b')
  parser.add_argument('--start-case', type=int, default=0,
                      help='Resume at a zero-based case index.')
  args = parser.parse_args()
  old_class = reference_class(args.reference)
  medium = 'BamLlama2MediumV2C256'
  xl = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2'
  cases = [
      (medium+'ScanAotCleanControl', 'local_qk+full', {}),
      (medium+'LocalFetchC8SharedReadNonScan', 'local_qk+local_o', {}),
      (medium+'LocalFetchC8SharedReadNonScan', 'local_qk+full', {}),
      (medium+'SeededPaired40', 'local_qk+full', {}),
      (medium+'Paired40LocalQKRank2HeadRankGate', 'local_qk+full', {}),
      (xl, 'local_qk+full', {}),
      (xl, 'local_qk+full', dict(bam_local_qk_pre_rms_bias=False,
          bam_fetch_diagonal_one=False, bam_write_data_rms=False, bam_m_read_norm='rms')),
      (xl+'Gate050InterpolatedReadRowOnly', 'local_qk+full', {}),
      (medium+'ScanAotCleanGate050FixedAmplitude', 'local_qk+full', {}),
      (medium+'ScanAotCleanControl', 'local_qk+full',
       dict(bam_pack_factorized_local_qk=False)),
  ]
  for name, mode, overrides in cases[args.start_case:]:
    head = 128 if name.startswith('BamLlama2XL') else 64
    with tempfile.TemporaryDirectory() as output:
      (Path(output)/'test').mkdir()
      cfg = pyconfig.initialize(
          [None, str(Path(__file__).resolve().parents[2]/'MaxText/configs/base.yml')],
          exp_class=name, run_name='test', enable_checkpointing=False,
          base_output_directory=output+'/', jax_cache_dir='', log_config=False,
          dataset_type='synthetic', base_emb_dim=2*head, base_num_query_heads=2,
          base_num_kv_heads=2, base_num_decoder_layers=4, base_mlp_dim=256,
          head_dim=head, max_target_length=4, max_prefill_predict_length=4,
          query_chunk_size=2, per_device_batch_size=1.)
      cfg.get_keys().update(bam_write_v_bottleneck_dim=32, **overrides)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      kwargs = dict(config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=head,
          max_target_length=4, max_prefill_predict_length=4, mesh=mesh,
          attention_kernel='dot_product_chunk', dtype=cfg.dtype,
          layer_mode=mode, attention_type=cfg.attention_type,
          bam_k=cfg.bam_k, bam_v=cfg.bam_v)
      old, new = old_class(**kwargs), att.BamAttention(**kwargs)
      x = jax.random.normal(jax.random.key(1), (1,4,2*head), dtype=cfg.dtype)
      m = jax.random.normal(jax.random.key(2), (1,4,cfg.bam_k,cfg.bam_v), dtype=cfg.dtype)
      def forward(module, params, x, m):
        return module.apply({'params':params}, x, x, jnp.arange(4)[None],
            jnp.ones((1,4), jnp.int32), M_in=m, deterministic=True, layer_index=2)
      def initialize(module):
        return module.init({'params':jax.random.key(3), 'aqt':jax.random.key(4)},
            x,x,jnp.arange(4)[None],jnp.ones((1,4),jnp.int32),
            M_in=m, deterministic=True, layer_index=2)['params']
      params = initialize(old)
      compare(params, initialize(new))
      # Activate zero-initialized read projections so output/gradient checks
      # exercise the read paths rather than their zero-initialization bypass.
      params = jax.tree.map(lambda a: a + jax.random.normal(
          jax.random.key(5), a.shape, dtype=a.dtype)*jnp.asarray(.01,a.dtype), params)
      expected = forward(old, params, x, m)
      actual = forward(new, params, x, m)
      compare(expected, actual)
      cotangents = tuple(jax.random.normal(jax.random.key(i+6), y.shape)
                        for i,y in enumerate(expected))
      def objective(module,p,h,state):
        return sum(jnp.sum(y.astype(jnp.float32)*c) for y,c in
                   zip(forward(module,p,h,state),cotangents))
      gradients = [jax.grad(lambda p,h,state: objective(module,p,h,state), (0,1,2))(
          params,x,m) for module in (old,new)]
      compare(*gradients)
      print(json.dumps(dict(experiment=name, mode=mode, overrides=overrides,
          params_output_Mout_param_x_M_grad_max_abs_error=0.)), flush=True)
    jax.clear_caches()


if __name__ == '__main__':
  main()
