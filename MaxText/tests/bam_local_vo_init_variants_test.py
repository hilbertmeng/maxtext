"""Initializer-only experiments: parameter isolation, geometry and scanned gradients."""
import functools, unittest
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train_compile
import bam_mlp_write_test as fixture
from layers.models import Transformer
from layers import quantizations
PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
EXPS=[PARENT.replace('TruePile',s+'TruePile') for s in ('VOReadKeySmallInit','StaticVOrthogonalInit')]
class VOInitTest(unittest.TestCase):
  setUp=fixture.MLPWriteTest.setUp
  tearDown=fixture.MLPWriteTest.tearDown
  config=fixture.MLPWriteTest.config
  def test_budget_and_defaults(self):
    for name in [PARENT,*EXPS]:
      c=self.config(name)
      self.assertEqual(c.bam_read_key_scale,.2)
      self.assertIsNone(c.bam_local_vo_read_key_scale)
      self.assertIsNone(c.bam_local_v_read_gate_init)
      self.assertEqual(c.bam_read_gate_init,.05)
      self.assertFalse(c.bam_local_v_static_zero_init)
      self.assertEqual(c.bam_fetched_read_kernel_init,'zero')
      self.assertEqual(c.mlp_dim_by_block,[3901,3774,3901])
      self.assertTrue(c.bam_record_concat_health)
      self.assertEqual(c.DATASET_VARIANT,'truepile4096')
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
      self.assertEqual(sum(int(np.prod(v.shape)) for v in jax.tree.leaves(args[0].params)),432096128)
    print('ALL_THREE_432096128_ORIGINAL_GATES_SCALES_HEALTH_OK',flush=True)
  def test_isolation_geometry_and_scanned_gradient(self):
    tokens=jnp.array([[1,2,3,4]],jnp.int32);call=(tokens,jnp.arange(4)[None],tokens,jnp.ones_like(tokens),jnp.ones_like(tokens))
    trees=[]
    for name in [PARENT,*EXPS]:
      c=self.config(name,dtype='float32',weight_dtype='float32')
      # Keep production D so the small initializer has the proposed .001 projected RMS.
      c.get_keys().update(base_num_decoder_layers=3,num_decoder_layers=3,mlp_dim=64,mlp_dim_by_block=[64]*3,
          vocab_size=32,bam_layer_modes=['local_qk+local_v+local_o']*3,
          bam_write_v_bottleneck_dim=16,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model=Transformer(c,mesh,quantizations.configure_quantization(c))
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
        if name!=PARENT:
          def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]**2)
          value,grad=jax.jit(jax.value_and_grad(loss))(p)
          self.assertTrue(np.isfinite(float(value)))
          self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      flat=flatten_dict(nn.unbox(p));trees.append(flat)
      if name==PARENT:continue
      self.assertEqual(trees[0].keys(),flat.keys())
      changed=[]
      for path,v in flat.items():
        if not np.array_equal(v,trees[0][path]):changed.append(path)
        if name==EXPS[0] and 'W_R' in path and 'kernel' in path:
          np.testing.assert_array_equal(trees[0][path],0)
          np.testing.assert_allclose(float(jnp.std(v)),.001/np.sqrt(1200),rtol=.025)
          # Scan axis 1: [D,L,H,F,C]. Test raw pre-RMS key magnitude directly.
          x=jax.random.normal(jax.random.key(101),(128,1200));x=x/jnp.sqrt(jnp.mean(x*x,axis=-1,keepdims=True))
          raw=jnp.einsum('td,dlhfc->tlhfc',x,v)
          np.testing.assert_allclose(float(jnp.sqrt(jnp.mean(raw*raw))),.001,rtol=.025)
        if name==EXPS[1] and 'static_v_key' in path:
          # [V,L,H] under scan axis1; each layer's 16 columns are orthonormal.
          for key in np.moveaxis(np.asarray(v),1,0):np.testing.assert_allclose(key.T@key,np.eye(16),atol=5e-6)
      expected='W_R' if name==EXPS[0] else 'static_v_key'
      self.assertTrue(changed)
      self.assertTrue(all(expected in p for p in changed),changed)
      print('ISOLATED_INIT_FINITE_GRAD',name,changed,flush=True)
if __name__=='__main__':unittest.main()
