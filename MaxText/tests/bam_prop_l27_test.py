"""Exact equal-budget L27 MHA/BAM and their real scanned forward/gradient gates."""
import functools,unittest,numpy as np,jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import exp,max_utils,train,train_compile
from layers.models import Transformer
from layers import quantizations
from bam_mlp_write_test import MLPWriteTest
MHA='BamMHAMediumPropL27C256TruePile'
BAM='BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
class L27Test(unittest.TestCase):
 setUp=MLPWriteTest.setUp
 tearDown=MLPWriteTest.tearDown
 config=MLPWriteTest.config
 def test_exact_full_budget_and_both_scan_layouts(self):
  for name,target in [(MHA,432142800),(BAM,432128192)]:
   self.assertTrue(hasattr(exp,name))
   c=self.config(name);self.assertEqual(c.model_name,name)
   self.assertEqual((c.num_decoder_layers,c.emb_dim,c.num_query_heads,c.head_dim),(27,1200,16,75))
   self.assertEqual(c.DATASET_VARIANT,'truepile4096');self.assertEqual(c.steps,13500)
   self.assertTrue(c.record_training_health_metrics);self.assertFalse(c.qk_norm)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(args[0].params)
   actual=sum(int(np.prod(v.shape)) for v in flat.values());self.assertEqual(actual,target)
   if name==MHA:
    self.assertTrue(c.bam_mha_control);self.assertFalse(c.bam_pair_scan)
    self.assertEqual(c.mlp_dim,1600);self.assertIsNone(c.mlp_dim_by_block)
    self.assertEqual(c.bam_partial_rope_nope_dim,1)
    self.assertFalse(any('mlp_address_down' in p for p in flat))
   else:
    self.assertFalse(c.bam_mha_control);self.assertTrue(c.bam_pair_scan)
    self.assertEqual(c.mlp_dim_by_block,[2311,2184,2311])
    self.assertEqual((c.bam_k,c.bam_v,c.bam_abs_v_compression_dim),(75,32,8))
    self.assertEqual((c.bam_local_qk_col_output_dim,c.bam_partial_rope_nope_dim,c.bam_standard_qk_dim),(57,57,18))
    self.assertFalse(c.bam_local_qk_add_before_rope);self.assertFalse(c.bam_local_qk_direct_c8)
    self.assertTrue(c.bam_local_qk_share_basis);self.assertEqual(c.bam_mlp_write_address_rank,256)
    down=[v for p,v in flat.items() if 'mlp_address_down' in p]
    self.assertEqual(len(down),1);self.assertEqual(sorted(down[0].shape),sorted((1200,9,256)))
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    scalars=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
   if name==BAM:
    for i in range(27):
     self.assertIn(f'bam/concat/qk_scores/layer_{i:03d}/bam_over_standard',scalars)
     self.assertEqual(f'bam/concat/mlp_write_gate/layer_{i:03d}/mean' in scalars,i%3==1)
   print('L27_EXACT_PARAMS_FULL_TRAINSTEP_SCAN_HEALTH_OK',name,actual,flush=True)
 def test_real_mha_bam_forward_and_consumed_gradients(self):
  for name in (MHA,BAM):
   c=self.config(name);c.get_keys().update(base_emb_dim=150,emb_dim=150,base_num_query_heads=2,num_query_heads=2,base_num_kv_heads=2,num_kv_heads=2,base_mlp_dim=64,mlp_dim=64,mlp_dim_by_block=None if name==MHA else [64]*3,vocab_size=128,bam_write_v_bottleneck_dim=32,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=32)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quantizations.configure_quantization(c));tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok);call=(tok,pos,tok,mask,mask)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
    def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0])
    value,grad=jax.jit(jax.value_and_grad(loss))(params)
   self.assertTrue(np.isfinite(float(value)));self.assertTrue(all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree.leaves(grad)))
   flat=flatten_dict(nn.unbox(grad))
   if name==BAM:
    for needle in ('mlp_address_down','mlp_write_gate'):
     gs=[v for p,v in flat.items() if needle in p];self.assertTrue(gs);self.assertGreater(sum(float(jnp.sum(v.astype(jnp.float32)**2)) for v in gs),0)
   print('L27_REAL27_FORWARD_FINITE_CONSUMED_GRAD_OK',name,float(value),flush=True)
if __name__=='__main__':unittest.main()
