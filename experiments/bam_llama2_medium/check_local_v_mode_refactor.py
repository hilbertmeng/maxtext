#!/usr/bin/env python3
"""Compare LocalV refactoring to a saved working-tree snapshot, including scan."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import exp
import max_utils
import pyconfig
from layers import attentions, fusion


def load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def equal(old, new):
  assert jax.tree.structure(old) == jax.tree.structure(new)
  for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    assert np.isfinite(b).all()
    np.testing.assert_array_equal(a, b)


def perturb(params):
  return jax.tree.map(lambda a: a + jnp.asarray(.01, a.dtype) *
      jax.random.normal(jax.random.key(51), a.shape, a.dtype), params)


def config(name, root, head=64, layers=4):
  cfg = pyconfig.initialize(
      [None, str(Path(__file__).resolve().parents[2]/'MaxText/configs/base.yml')],
      exp_class=name, run_name='test', enable_checkpointing=False,
      base_output_directory=str(root)+'/', jax_cache_dir='', log_config=False,
      dataset_type='synthetic', base_emb_dim=2*head, base_num_query_heads=2,
      base_num_kv_heads=2, base_num_decoder_layers=layers, base_mlp_dim=256,
      head_dim=head, max_target_length=4, max_prefill_predict_length=4,
      query_chunk_size=2, per_device_batch_size=1.)
  cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
  return cfg


def old_config(cfg, old_class):
  keys = copy.deepcopy(cfg.get_keys())
  for field, default in (('bam_layer_modes', []), ('bam_local_v_rank', None),
                         ('bam_local_o_v_mode', 'none')):
    keys[field] = copy.deepcopy(getattr(old_class, field, default))
  return pyconfig.HyperParameters(SimpleNamespace(keys=keys))


def select(value, i):
  return value[i] if isinstance(value, list) else value


def compare_modules(cfg, before_cfg, old_att, index, head):
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  common = dict(num_query_heads=2, num_kv_heads=2, head_dim=head,
      max_target_length=4, max_prefill_predict_length=4, mesh=mesh,
      attention_kernel='dot_product_chunk', dtype=cfg.dtype,
      attention_type=cfg.attention_type, bam_k=cfg.bam_k, bam_v=cfg.bam_v)
  old = old_att.BamAttention(config=before_cfg,
      layer_mode=select(before_cfg.bam_layer_modes, index),
      local_v_mode=select(before_cfg.bam_local_o_v_mode, index), **common)
  new = attentions.BamAttention(config=cfg,
      layer_mode=select(cfg.bam_layer_modes, index), layer_inx=index, **common)
  x = jax.random.normal(jax.random.key(1), (1, 4, 2*head), cfg.dtype)
  m = jax.random.normal(jax.random.key(2), (1, 4, cfg.bam_k, cfg.bam_v), cfg.dtype)
  def call(module, params, h, state):
    return module.apply({'params': params}, h, h, jnp.arange(4)[None],
        jnp.ones((1, 4), jnp.int32), M_in=state, deterministic=True)
  def initialize(module):
    return module.init({'params': jax.random.key(3)}, x, x,
        jnp.arange(4)[None], jnp.ones((1, 4), jnp.int32), M_in=m)['params']
  params = initialize(old)
  equal(params, initialize(new))
  params = perturb(params)
  expected = call(old, params, x, m)
  equal(expected, call(new, params, x, m))
  cotangent = tuple(jax.random.normal(jax.random.key(i+7), y.shape) for i, y in enumerate(expected))
  def objective(module, p, h, state):
    return sum(jnp.sum(y.astype(jnp.float32)*c)
               for y,c in zip(call(module, p, h, state), cotangent))
  gradients = [jax.grad(lambda p,h,state: objective(module,p,h,state), (0,1,2))(
      params,x,m) for module in (old,new)]
  equal(*gradients)


def compare_scan(cfg, before_cfg, old_fusion):
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  modules = [nn.scan(cls, variable_axes={'params': cfg.param_scan_axis},
      split_rngs={'params': True, 'dropout': False},
      in_axes=(nn.broadcast,)*10+(0,), length=2,
      metadata_params={nn.PARTITION_NAME: 'layers'})(
          c, mesh, 4, all_global_attention=True)
      for cls,c in ((old_fusion.BamLayerPair,before_cfg),(fusion.BamLayerPair,cfg))]
  x = jax.random.normal(jax.random.key(21), (1,4,128), cfg.dtype)
  m = jax.random.normal(jax.random.key(22), (1,4,32,32), cfg.dtype)
  args = (jnp.ones((1,4),jnp.int32), jnp.arange(4)[None],
      jnp.ones((1,4),jnp.int32), None, True, 'train', None, None, None, None, jnp.arange(2))
  params = modules[0].init({'params':jax.random.key(23)}, (x,m), *args)['params']
  equal(params, modules[1].init({'params':jax.random.key(23)}, (x,m), *args)['params'])
  params = perturb(params)
  def call(module, p, h, state):
    carry,_ = module.apply({'params':p}, (h,state), *args)
    return carry
  expected = call(modules[0],params,x,m)
  equal(expected,call(modules[1],params,x,m))
  def objective(module,p,h,state):
    y,out_m=call(module,p,h,state)
    return jnp.sum(y.astype(jnp.float32)*.13)+jnp.sum(out_m.astype(jnp.float32)*.17)
  gradients=[jax.grad(lambda p,h,state: objective(module,p,h,state),(0,1,2))(
      params,x,m) for module in modules]
  equal(*gradients)


def main():
  parser=argparse.ArgumentParser(__doc__)
  parser.add_argument('--before',type=Path,required=True,help='Snapshot root containing MaxText/.')
  parser.add_argument('--case',help='Run one named case.')
  args=parser.parse_args()
  old_exp=load('_local_v_old_exp',args.before/'MaxText/exp.py')
  old_cfg=load('_local_v_old_config',args.before/'MaxText/bam_config.py')
  old_att=load('_local_v_old_attentions',args.before/'MaxText/layers/attentions.py')
  old_att.validate_bam_config=old_cfg.validate_bam_config
  old_fusion=load('_local_v_old_fusion',args.before/'MaxText/layers/fusion.py')
  old_fusion.attentions=old_att
  prefix='BamLlama2MediumV2C256'
  mixed=prefix+'LocalFetchC8SharedIndependentSharedLLLFScan'
  cases=[('full_control',prefix+'ScanAotCleanControl',0,64),
         ('o_only',prefix+'LocalFetchC8NonScan',0,64),
         ('v_rank2',prefix+'LocalFetchC8LocalVNonScan',0,64),
         ('v_shared',prefix+'LocalFetchC8SharedReadNonScan',0,64),
         ('v_shared_full_m',prefix+'LocalFetchFullSharedReadScan',0,64),
         *[(f'mixed_layer_{i}',mixed,i,64) for i in range(4)],
         ('v_rank4','BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',0,64),
         ('xl_rank4','BamXLIndependentLLFLocalVRank4CFp32AlignedRow',0,128),
         ('mixed_scan',mixed,0,64)]
  assert args.case is None or args.case in {c[0] for c in cases}
  for label,name,index,head in cases:
    if args.case and label!=args.case:continue
    with tempfile.TemporaryDirectory() as output:
      (Path(output)/'test').mkdir()
      cfg=config(name,output,head,8 if label=='mixed_scan' else 4)
      before_cfg=old_config(cfg,getattr(old_exp,name))
      if label=='mixed_scan':compare_scan(cfg,before_cfg,old_fusion)
      else:compare_modules(cfg,before_cfg,old_att,index,head)
      print(json.dumps(dict(case=label,experiment=name,layer_index=index,
          params_output_Mout_param_x_M_grad_max_abs_error=0.)),flush=True)
    jax.clear_caches()


if __name__=='__main__':
  main()
