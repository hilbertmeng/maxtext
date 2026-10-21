"""Q/K gate removal: unchanged initial operator, active remaining reads, exact budget."""
import math
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers.models import Transformer
PARENT='BamMediumPropMShiftReadG050FullQKNormTruePile'
CHILD='BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile'
class QKStaticGateTest(MLPWriteTest,unittest.TestCase):
 def mesh(self,c): return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
 def test_budget_and_formal_controls(self):
  totals=[];trees=[]
  for exp in (PARENT,CHILD):
   c=self.config(exp);args,*_=train_compile.get_shaped_inputs(self.mesh(c),c)
   flat=flatten_dict(nn.unbox(args[0].params));trees.append(flat);totals.append(sum(math.prod(v.shape) for v in flat.values()))
  removed=[p for p in trees[0] if p not in trees[1]]
  self.assertTrue(removed)
  self.assertTrue(all(any('general_'+a+'_static_gate' in part for a in ('q','k') for part in p) for p in removed),removed)
  self.assertEqual(sum(math.prod(trees[0][p].shape) for p in removed),691776)
  self.assertEqual(totals,[432114380,431422604]);self.assertEqual(totals[1]-totals[0],-691776)
  self.assertFalse(c.bam_qk_static_read_gate);self.assertTrue(c.bam_qk_norm_after_concat and c.qk_norm and c.scan_layers)
  self.assertEqual(c.mlp_dim_by_block,[3802,3676,3802]);self.assertEqual(c.bam_static_read_gate_init,.99)
  self.assertFalse(c.record_training_health_metrics or c.bam_record_concat_health or c.bam_record_m_shift_health)
  self.assertEqual(c.tensorboard_gcs_rotate_period,500);self.assertEqual(c.checkpoint_period,200)
  self.assertEqual((c.sa_block_q,c.sa_block_kv,c.sa_block_kv_compute,c.sa_block_q_dkv,c.sa_block_kv_dkv,c.sa_block_kv_dkv_compute,c.sa_use_fused_bwd_kernel),(1024,1024,512,1024,1024,1024,True))
  print('NO_QK_STATIC_GATE_BUDGET_OK',totals,'removed',691776,flush=True)
 def test_initial_equivalence_gradients_and_causality(self):
  c=self.config(PARENT,bam_splash_attention=False,base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
  c.get_keys().update(emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8,bam_layer_modes=c.bam_layer_modes[:3],mlp_dim_by_block=[32,24,32])
  mesh=self.mesh(c);tok=jnp.array([[1,2,3,4]]);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   model=Transformer(c,mesh,quant=None)
   p=nn.unbox(model.init(jax.random.key(24),tok,pos,tok,mask,mask,enable_dropout=False)['params'])
   f=lambda model,p,t:model.apply({'params':p},t,pos,t,mask,mask,enable_dropout=False)[0]
   old=f(model,p,tok);oldflat=flatten_dict(p)
   for path,v in oldflat.items():
    if any('general_'+a+'_static_gate_bias' in path for a in ('q','k')):np.testing.assert_allclose(v,math.log(9),rtol=1e-6)
    if any('general_'+a+'_static_gate' in path for a in ('q','k')) and path[-1]=='kernel':np.testing.assert_array_equal(v,0)
   c.get_keys().update(bam_qk_static_read_gate=False);newmodel=Transformer(c,mesh,quant=None)
   newp=nn.unbox(newmodel.init(jax.random.key(24),tok,pos,tok,mask,mask,enable_dropout=False)['params']);flat=flatten_dict(newp)
   for path,v in flat.items(): np.testing.assert_array_equal(v,oldflat[path])
   # O static-key starts at zero, so its gate gradient is initially zero by design.
   self.assertTrue(any('general_o_static_gate' in path for path in flat))
   new=f(newmodel,newp,tok);np.testing.assert_allclose(new,old,rtol=3e-5,atol=3e-6)
   loss,g=jax.jit(jax.value_and_grad(lambda params:jnp.mean(f(newmodel,params,tok)**2)))(newp)
   gf=flatten_dict(nn.unbox(g));self.assertTrue(np.isfinite(float(loss)))
   self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in gf.values()))
   for name in ('static_q_key','static_k_key','general_v_static_gate','W_lq_gate','W_lk_gate','q_norm','k_norm'):
    self.assertGreater(sum(float(jnp.sum(v*v)) for path,v in gf.items() if name in path),0,name)
   np.testing.assert_allclose(new[:,:3],f(newmodel,newp,tok.at[:,3].set(5))[:,:3],atol=1e-6,rtol=1e-6)
   print('NO_QK_STATIC_GATE_INIT_AND_GRAD_OK',flush=True)
if __name__=='__main__': unittest.main()
