"""Focused gates for independent Q/K/VO compressed matrix views."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train_compile
from layers.models import Transformer
from layers import quantizations
from bam_mlp_write_test import MLPWriteTest

PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectC8TruePile'
EXP=PARENT.replace('DirectC8TruePile','DirectC8SeparateQKVProjectionTruePile')
EXTRA={'local_q_c_projection','local_k_c_projection'}

class SeparateQKVTest(unittest.TestCase):
 setUp=MLPWriteTest.setUp
 tearDown=MLPWriteTest.tearDown
 config=MLPWriteTest.config
 def test_full_budget(self):
  c=self.config(EXP)
  self.assertEqual(c.DATASET_VARIANT,'truepile4096')
  self.assertEqual(c.mlp_dim_by_block,[3901,3774,3901])
  self.assertTrue(c.bam_local_qkv_separate_c_projection)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
  flat=flatten_dict(nn.unbox(args[0].params))
  count=sum(int(np.prod(v.shape)) for v in flat.values())
  self.assertEqual(count,432103040)
  for name in EXTRA:
   vals=[v for p,v in flat.items() if name in p]
   self.assertEqual(len(vals),3)
   self.assertTrue(all(sorted(v.shape)==sorted((32,8,6)) for v in vals))
  print('FULL_BUDGET_OK',count,flush=True)
 def test_init_equivalence_and_gradient_partition(self):
  models=[];parameters=[]
  for exp in (PARENT,EXP):
   c=self.config(exp,dtype='float32',weight_dtype='float32')
   c.get_keys().update(base_emb_dim=150,emb_dim=150,num_query_heads=2,num_kv_heads=2,
     base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,num_decoder_layers=3,
     base_mlp_dim=64,mlp_dim=64,mlp_dim_by_block=[64]*3,vocab_size=32,
     bam_layer_modes=['local_qk+local_v+local_o']*3,bam_write_v_bottleneck_dim=16,
     emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   model=Transformer(c,mesh,quantizations.configure_quantization(c))
   tokens=jnp.array([[1,2,3,4]],jnp.int32)
   call=(tokens,jnp.arange(4)[None],tokens,jnp.ones_like(tokens),jnp.ones_like(tokens))
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    p=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
   models.append(model);parameters.append(p)
  flat0,flat1=[flatten_dict(nn.unbox(p)) for p in parameters]
  self.assertEqual(set(flat0),{p for p in flat1 if not EXTRA.intersection(p)})
  for p,v in flat0.items():np.testing.assert_array_equal(v,flat1[p],err_msg='/'.join(p))
  for p,v in flat1.items():
   if EXTRA.intersection(p):np.testing.assert_array_equal(v,flat1[p[:-1]+('abs_v_cache_projection',)])
  outputs=[];grads=[]
  for model,p in zip(models,parameters):
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    def loss(params):
     logits=model.apply({'params':params},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]
     return jnp.mean(jax.nn.log_softmax(logits)[...,1]),logits
    (value,output),g=jax.jit(jax.value_and_grad(loss,has_aux=True))(p)
   self.assertTrue(np.isfinite(float(value)))
   self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(g)))
   outputs.append(output);grads.append(flatten_dict(nn.unbox(g)))
  np.testing.assert_array_equal(*outputs)
  for name in EXTRA:
   self.assertGreater(sum(float(jnp.sum(v*v)) for p,v in grads[1].items() if name in p),0.,name)
  for p,v in grads[0].items():
   if p[-1]=='abs_v_cache_projection':
    combined=grads[1][p]+sum(grads[1][p[:-1]+(name,)] for name in EXTRA)
    np.testing.assert_allclose(v,combined,rtol=3e-4,atol=3e-6)
   else:np.testing.assert_allclose(v,grads[1][p],rtol=3e-4,atol=3e-6,err_msg='/'.join(p))
  print('CLONED_INIT_OUTPUT_EQUAL_GRADIENT_PARTITION_OK',flush=True)

if __name__=='__main__':unittest.main()
