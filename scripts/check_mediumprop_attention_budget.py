"""CPU gate: exact budgets, unequal-head writes, gradients, and Splash AOT selection."""
import argparse
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
import train_compile
from layers.attentions import BamAttention, _bam_splash_enabled
from layers.models import Transformer

PREFIX='BamMediumPropD1152'
SUFFIX='AllLocalMLPWriteIndependentEveryThirdTruePile'
ARMS=[('H24K96V32C8',432112464),('H24K96V48C12',432114000),
      ('H32K72V48C12',432125504),('H32K72V64C16',432119616)]

def config(name, tmp):
  Path(tmp,'audit').mkdir(exist_ok=True)
  return pyconfig.initialize([None,'MaxText/configs/base.yml'], exp_class=name,
      run_name='audit',enable_checkpointing=False,base_output_directory=tmp+'/',
      jax_cache_dir='',log_config=False,dataset_type='synthetic',
      max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,
      per_device_batch_size=1.)


def shapes():
  with tempfile.TemporaryDirectory() as tmp:
    for suffix,expected in ARMS:
      name=PREFIX+suffix+SUFFIX
      cfg=config(name,tmp)
      assert cfg.emb_dim == cfg.bam_mlp_write_num_heads*cfg.bam_k == 1152
      assert cfg.bam_standard_qk_dim*4 == cfg.head_dim
      assert cfg.bam_local_qk_col_output_dim+cfg.bam_standard_qk_dim == cfg.head_dim
      assert cfg.bam_splash_attention and cfg.bam_splash_seq_minor and not cfg.bam_pallas_core
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,cfg)
      flat=flatten_dict(nn.unbox(args[0].params))
      count=sum(math.prod(x.shape) for x in flat.values())
      assert count==expected,(name,count,expected)
      keys={'/'.join(k):list(v.shape) for k,v in flat.items()
            if 'mlp_address_up' in k or 'mlp_write_gate_bias' in k or 'W_emb_u' in k}
      print('EXACT_BUDGET_OK',json.dumps(dict(name=name,count=count,selected_shapes=keys)),flush=True)


def merge():
  keys=jax.random.split(jax.random.PRNGKey(7),8)
  shape=lambda key,s:jax.random.normal(key,s)
  state=shape(keys[0],(2,3,5,7))
  ac=shape(keys[1],(2,3,4,5)); aa=shape(keys[2],(2,3,4,7))
  gate=jax.nn.sigmoid(shape(keys[3],(2,3,2)))
  y=shape(keys[4],(2,3,2,5)); address=shape(keys[5],(2,3,2,7))
  norm=lambda x:x*jax.lax.rsqrt(jnp.mean(x*x,axis=-1,keepdims=True)+1e-6)
  factors=(ac,aa,jnp.ones((2,3,4)))
  for mode in ('dot','mul_reduce'):
    receiver=SimpleNamespace(num_query_heads=4,_write_data_rms=True,
        write_data_norm=norm,write_address_norm=norm,_concat_health=False,
        _write_outer_implementation=mode,
        config=SimpleNamespace(bam_sqrt_n_scale=True,bam_lambda_decay=.7))
    def actual(y):return BamAttention.merge_mlp_write(receiver,y,gate,factors,state,address)
    def expected(y):
      return .7*state+jnp.einsum('btnk,btnv->btkv',ac,aa)+jnp.einsum(
          'btnk,btnv->btkv',norm(y)*gate[...,None]/jnp.sqrt(2),norm(address))
    np.testing.assert_allclose(actual(y),expected(y),rtol=1e-5,atol=1e-5)
    a=jax.grad(lambda y:jnp.sum(actual(y)**2))(y)
    b=jax.grad(lambda y:jnp.sum(expected(y)**2))(y)
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-4)
  # CPU fixture must stay on C256, while offline TPU AOT must trace Splash.
  cfg=SimpleNamespace(bam_splash_attention=True,compile_topology='')
  assert not _bam_splash_enabled(cfg,256)
  cfg.compile_topology='v5p-16'
  assert _bam_splash_enabled(cfg,4096) and not _bam_splash_enabled(cfg,4)
  cfg.bam_splash_attention=False
  assert not _bam_splash_enabled(cfg,4096)
  print('UNEQUAL_HEAD_MERGE_GRADIENT_AND_AOT_SELECTION_OK',flush=True)


def tiny():
  with tempfile.TemporaryDirectory() as tmp:
    cfg=config(PREFIX+ARMS[0][0]+SUFFIX,tmp)
    cfg.get_keys().update(base_emb_dim=32,emb_dim=32,base_num_query_heads=8,
      base_num_kv_heads=8,num_query_heads=8,num_kv_heads=8,head_dim=8,bam_k=8,
      bam_v=8,bam_abs_v_compression_dim=2,bam_standard_qk_dim=2,
      bam_local_qk_col_output_dim=6,bam_partial_rope_nope_dim=6,
      bam_write_v_bottleneck_dim=16,emb_bam_num_head=8,emb_bam_v_bottleneck_dim=16,
      bam_mlp_write_num_heads=4,bam_mlp_write_address_rank=8,base_mlp_dim=32,
      mlp_dim=32,mlp_dim_by_block=[32,24,32],base_num_decoder_layers=3,
      num_decoder_layers=3,bam_layer_modes=['local_qk+local_v+local_o']*3,
      vocab_size=128,dtype='float32')
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    model=Transformer(cfg,mesh,quant=None)
    tokens=jnp.array([[1,4,8,2]],jnp.int32)
    pos=jnp.arange(4,dtype=jnp.int32)[None,:]; seg=jnp.ones_like(tokens)
    rng={'params':jax.random.PRNGKey(1),'dropout':jax.random.PRNGKey(2),'aqt':jax.random.PRNGKey(3)}
    variables=model.init(rng,tokens,pos,seg,tokens)
    def loss(params):
      result=model.apply({'params':params},tokens,pos,seg,tokens,
          enable_dropout=False,rngs=rng,mutable=['intermediates'])
      output=result[0]
      if isinstance(output,tuple): output=output[0]
      return jnp.mean(output.astype(jnp.float32)**2)
    value,grads=jax.value_and_grad(loss)(variables['params'])
    assert np.isfinite(np.asarray(value)).all()
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))
    flat=flatten_dict(nn.unbox(grads))
    address=[v for k,v in flat.items() if 'mlp_address_up' in k and k[-1]=='kernel']
    assert address and any(float(jnp.linalg.norm(v))>0 for v in address)
    print('TINY_MODEL_FORWARD_AND_GRADIENT_OK',float(value),flush=True)

if __name__=='__main__':
  p=argparse.ArgumentParser(); p.add_argument('--part',choices=['shapes','merge','tiny','all'],default='all')
  part=p.parse_args().part
  for name,fn in [('merge',merge),('shapes',shapes),('tiny',tiny)]:
    if part in (name,'all'):fn()
