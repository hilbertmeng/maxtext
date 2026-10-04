"""Prefix/full MLP writes: support, untouched attention, scan health, exact budget."""
import contextlib
import copy
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from layers import rmt
from tests.rmt_xlprop_test import XLPropTest

BASE='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero'
EXP=BASE+'MLPFront16FullEveryThird'

class PrefixWriteTest(unittest.TestCase):
  config=XLPropTest.config
  model_args=XLPropTest.model_args

  def test_budget_and_scanned_shapes(self):
    cfg=self.config(EXP)
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1)))
    count=sum(math.prod(v.shape) for v in jax.tree.leaves(params))
    self.assertEqual(count,431759072+12*(3*1200*37-132096))
    self.assertEqual(cfg.rmt_mlp_dim_by_block,[4137,4100,4137])
    for i,rows in enumerate([16,48,16]):
      layer=params['decoder']['layers'][f'layer_{i}']
      shape=list(layer['dynamic_mlp_write']['address_up'].shape);shape.pop(cfg.param_scan_axis)
      self.assertEqual(shape,[256,16*rows])
      shape=list(layer['mlp_write_key'].shape);shape.pop(cfg.param_scan_axis)
      self.assertEqual(shape,[16,rows])
    print('PARAMS',count,'SAVED_ADDRESS',12*132096)

  def test_write_support_and_full_parent_equivalence(self):
    cfg=self.config(BASE,base_num_decoder_layers=3,base_emb_dim=512,head_dim=32,base_mlp_dim=96,vocab_size=128)
    cfg.get_keys()['dtype']=jnp.float32
    M=jax.random.normal(jax.random.key(22),(1,4,48,32));ids=jnp.ones((1,4),jnp.int32);pos=jnp.arange(4)[None]
    parents=[]
    for rows in [None,48,16]:
      layer=rmt.RMTLayer(cfg,mlp_write_rows=rows)
      with contextlib.redirect_stdout(io.StringIO()):
        params=nn.unbox(layer.init(jax.random.key(3),M,ids,pos,True,0)['params'])
      out=layer.apply({'params':params},M,ids,pos,True,0)[0]
      if rows!=16:
        parents.append((params,out));continue
      q=copy.deepcopy(params);q['mlp_write_key']=jnp.zeros_like(q['mlp_write_key'])
      def no_dynamic_mlp(next_fun,args,kwargs,ctx):
        out=next_fun(*args,**kwargs)
        if ctx.module.name=='dynamic_mlp_write' and ctx.method_name=='__call__':
          return jnp.zeros_like(out[0]),out[1]
        return out
      with nn.intercept_methods(no_dynamic_mlp):
        before=layer.apply({'params':q},M,ids,pos,True,0)[0]
      np.testing.assert_array_equal(out[...,16:,:],before[...,16:,:])
      self.assertGreater(float(jnp.linalg.norm(out[...,:16,:]-before[...,:16,:])),0.)
      for k in ['attn_write_key','dynamic_attn_write','dynamic_qk_read','dynamic_vo_read']:
        if k in params:
          for a,b in zip(jax.tree.leaves(params[k]),jax.tree.leaves(parents[0][0][k])):np.testing.assert_array_equal(a,b)
    for a,b in zip(jax.tree.leaves(parents[0][0]),jax.tree.leaves(parents[1][0])):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(parents[0][1],parents[1][1])

  def test_block_forward_gradient_health(self):
    cfg=self.config(EXP,base_num_decoder_layers=6,base_emb_dim=512,head_dim=32,base_mlp_dim=96,vocab_size=128)
    cfg.get_keys().update(dtype=jnp.float32,rmt_mlp_dim_by_block=[112,96,112])
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(7),**args)['params'])
      def loss(p):
        y,aux=model.apply({'params':p},**args,mutable=['intermediates'])
        return jnp.mean(y[0]),aux
      (v,aux),grad=jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves((v,aux,grad))))
    from train import record_rmt_dynamic_health_metrics
    metrics={'scalar':{}};record_rmt_dynamic_health_metrics(metrics,aux,cfg)
    for l in range(6):
      self.assertIn(f'rmt/dynamic/layer_{l:03d}/mlp_write_gate_mean',metrics['scalar'])
      ratio=metrics['scalar'][f'rmt/dynamic/layer_{l:03d}/mlp_write_tail32_ratio']
      if l % 3 != 1:self.assertEqual(float(ratio),0.)
      else:self.assertGreater(float(ratio),0.)
    for i in range(3):
      g=grad['decoder']['layers'][f'layer_{i}']['dynamic_mlp_write']['address_up']
      self.assertGreater(float(jnp.linalg.norm(g)),0.)

if __name__=='__main__':unittest.main()
