"""One-tenth static V seed on the regular dynamic VO treatment."""
import unittest
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,train_compile
from layers.models import Transformer
from layers import quantizations
import bam_mlp_write_test as helper
PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroVOReadNormalTruePile'
EXP=PARENT.replace('TruePile','StaticVSmallTruePile')
class LocalVSmallTest(unittest.TestCase):
  setUp=helper.MLPWriteTest.setUp
  tearDown=helper.MLPWriteTest.tearDown
  config=helper.MLPWriteTest.config
  def test_full_budget_and_single_leaf_scale(self):
    c=self.config(EXP)
    self.assertEqual(c.bam_local_v_static_init_scale,.1)
    self.assertFalse(c.bam_local_v_static_zero_init)
    self.assertFalse(c.bam_attn_write_content_pre_rms_bias)
    self.assertEqual(c.bam_read_key_scale,.2)
    self.assertEqual(c.bam_local_vo_read_key_scale,1.)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,*_=train_compile.get_shaped_inputs(mesh,c)
    self.assertEqual(sum(int(np.prod(v.shape)) for v in jax.tree.leaves(args[0].params)),432096128)
    t=jnp.array([[1,2,3,4]],jnp.int32);call=(t,jnp.arange(4)[None],t,jnp.ones_like(t),jnp.ones_like(t))
    trees=[]
    for name,scale in ((PARENT,None),(EXP,.1),(EXP,1.)):
      c=self.config(name,dtype='float32',weight_dtype='float32')
      c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,
          base_num_decoder_layers=3,num_decoder_layers=3,mlp_dim=64,mlp_dim_by_block=[64]*3,
          vocab_size=32,bam_layer_modes=['local_qk+local_v+local_o']*3,
          bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
      if scale is not None:c.get_keys()['bam_local_v_static_init_scale']=scale
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model=Transformer(c,mesh,quantizations.configure_quantization(c))
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
        if scale==.1:
          def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]**2)
          value,grad=jax.jit(jax.value_and_grad(loss))(p)
      trees.append(flatten_dict(nn.unbox(p)))
    for path,v in trees[1].items():
      if 'static_v_key' in path:
        np.testing.assert_array_equal(trees[0][path],0.)
        np.testing.assert_allclose(v,.1*trees[2][path],rtol=2e-6,atol=1e-8)
        self.assertGreater(float(jnp.linalg.norm(v)),0.)
      else:np.testing.assert_array_equal(v,trees[0][path])
    self.assertTrue(np.isfinite(float(value)))
    self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
    flat=flatten_dict(nn.unbox(grad))
    self.assertGreater(sum(float(jnp.sum(v*v)) for p,v in flat.items() if 'static_v_key' in p),0.)
    print('STATIC_V_SMALL_BUDGET_ONLY_STATIC_V_CHANGED_EXACT_01_SCALE_FINITE_GRAD_OK',flush=True)
if __name__=='__main__':unittest.main()
