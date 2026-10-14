"""Cross-layer sharing: real parameter ties, budgets, scan differentiation and causality."""
import json
import os
import unittest
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers.models import Transformer

ARMS = json.loads(Path('experiments/bam_llama2_medium/cross_layer_race_arms.json').read_text())
if os.environ.get('CROSS_LAYER_REPAIR_ONLY'):
  ARMS = [a for a in ARMS if a['mode']=='conservative']
if os.environ.get('CROSS_LAYER_NEW_CONTROLS_ONLY'):
  ARMS = [a for a in ARMS if 'paired_run' in a]
class CrossLayerSharingTest(MLPWriteTest, unittest.TestCase):
  def mesh(self,c):
    return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
  def test_formal_budgets(self):
    for arm in ARMS:
      c=self.config(arm['run']); mesh=self.mesh(c)
      # Match the formal eight-chip FSDP layout, without allocating full weights.
      shape=[1]*len(c.mesh_axes);shape[c.mesh_axes.index('fsdp')]=8
      mesh=jax.sharding.Mesh(np.array(jax.devices()[:8]).reshape(shape),c.mesh_axes)
      args,_,shardings,_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(nn.unbox(args[0].params))
      count=sum(int(np.prod(x.shape)) for x in flat.values())
      specs=flatten_dict(nn.unbox(shardings.params))
      per_chip=sum(int(np.prod(specs[p].shard_shape(v.shape))) for p,v in flat.items())
      self.assertLess(per_chip/(count/8)-1,c.sharding_tolerance,(arm['run'],per_chip,count))
      self.assertEqual(count,arm['params'],(arm['run'],count))
      bank=[v for p,v in flat.items() if p[-1]=='cross_layer_down']
      self.assertEqual(len(bank),1)
      self.assertEqual(bank[0].shape,(18//arm['group_size'],1200,arm['rank']))
      self.assertFalse(any('full_m_read_down' in p or 'P_loc_down' in p for p in flat))
      self.assertEqual(any('mlp_address_down' in p for p in flat),not arm['mlp_shared'])
      bridges=[v for p,v in flat.items() if 'cross_layer_bridge' in p]
      self.assertEqual(len(bridges),3 if arm['mode']=='conservative' else 0)
      for b in bridges:self.assertEqual(b.shape,(arm['rank'],6,arm['old_rank']))
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      self.assertEqual(c.checkpoint_period,200)
      print('CROSS_LAYER_BUDGET_OK',arm['run'],count,flush=True)
  def test_scan_gradients_and_causality(self):
    for arm in ARMS:
      with self.subTest(run=arm['run']):
        old=8 if arm['old_rank']==256 else 12; new=12 if arm['rank']==384 else 16
        c=self.config(arm['run'],bam_splash_attention=False,base_emb_dim=150,
          base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=6,
          base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
        rank=old if arm['mode']=='conservative' else new
        c.get_keys().update(emb_bam_num_head=2,emb_bam_v_bottleneck_dim=8,
          bam_cross_layer_rank=new,bam_local_full_m_read_bottleneck_dim=rank,
          bam_write_v_bottleneck_dim=rank,bam_mlp_write_address_rank=rank if arm['mlp_shared'] else old,
          mlp_dim_by_block=[32,24,32],bam_layer_modes=c.bam_layer_modes[:6],remat_policy='full')
        mesh=self.mesh(c);model=Transformer(c,mesh,quant=None)
        tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
          params=model.init({'params':jax.random.key(30),'dropout':jax.random.key(31),'aqt':jax.random.key(32)},tok,pos,tok,mask,mask,enable_dropout=False)['params']
          def output(p,t):return model.apply({'params':p},t,pos,t,mask,mask,enable_dropout=False,rngs={'aqt':jax.random.key(32)})[0]
          val,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(output(p,tok)**2)))(params)
          flat=flatten_dict(nn.unbox(grad))
          self.assertTrue(np.isfinite(float(val)))
          self.assertTrue(all(np.isfinite(np.asarray(g)).all() for g in flat.values()))
          bank=next(g for p,g in flat.items() if p[-1]=='cross_layer_down')
          self.assertTrue(np.all(np.sum(np.asarray(bank)**2,axis=(1,2))>0))
          for name in ['P_loc_up','mlp_address_up','M_shift_gate']+(['cross_layer_bridge'] if arm['mode']=='conservative' else []):
            self.assertGreater(sum(float(jnp.sum(g*g)) for p,g in flat.items() if name in p),0,name)
          np.testing.assert_allclose(output(params,tok)[:,:3],output(params,tok.at[:,3].set(5))[:,:3],rtol=1e-6,atol=1e-6)
        print('CROSS_LAYER_GRAD_OK',arm['run'],flush=True)
        jax.clear_caches()
if __name__=='__main__':unittest.main()
